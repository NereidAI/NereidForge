# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Production edge-package orchestration for model versions.

The backend owns only the state machine and storage locations. It writes an
immutable ``nereidforge.package-request/1`` document and consumes the generic
``nereidforge.package-result/1`` written by the custom packager container.
Dataset traversal, IR introspection, decoder semantics, CPU/GPU verification,
OVMS layout and descriptor authoring are not implemented here (ADR-017).

Records are keyed by PRECISION (``fp32``/``fp16``/``int8``), never by device.
IR retains the existing FP16/INT8 numeric-verification path; direct ONNX is
FP32-only and structurally verified. The tar lives at
``nereid-model-<precision>.tar.gz``; identical loose payload is retained under
``staging/<precision>/``.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import and_, select

from app.clients.s3 import SeaweedFSClient
from app.core.util import new_id
from app.models.orm import (
    AsyncTask,
    DatasetVersion,
    Model as ModelORM,
    ModelVersion as ModelVersionORM,
)
from app.services.model_format import artifact_capability_permitted, ir_companion_bin
from app.services.models import ModelService
from app.services.package_contract import (
    build_package_request,
    parse_package_result,
    precision_enum,
)

# API/metadata precision keys (lowercase); descriptor enums are uppercase.
FP32 = "fp32"
FP16 = "fp16"
INT8 = "int8"
PRECISIONS = {FP32, FP16, INT8}


def _now() -> str:
    """Return a UTC ISO-8601 timestamp for package record fields."""
    return datetime.now(timezone.utc).isoformat()


class ModelPackageService:
    """Create package tasks and consume package-result documents."""

    def __init__(
        self,
        s3: SeaweedFSClient,
        models: ModelService,
        dataset: Optional[Any] = None,
    ) -> None:
        self.s3 = s3
        self.models = models
        # Retained for API construction compatibility; dataset CONTENT is read
        # by the packager container, not by the core service.
        self.dataset = dataset
        self.db = models.db

    # -- resolution ---------------------------------------------------------

    def _model_orm(self, tenant_id: str, ref: str) -> Optional[ModelORM]:
        return self.db.scalar(
            select(ModelORM).where(
                and_(
                    ModelORM.tenant_id == tenant_id,
                    (ModelORM.id == ref) | (ModelORM.name == ref),
                )
            )
        )

    def _version_orm(
        self, tenant_id: str, model_id: str, version: str
    ) -> Optional[ModelVersionORM]:
        return self.db.scalar(
            select(ModelVersionORM).where(
                and_(
                    ModelVersionORM.model_id == model_id,
                    ModelVersionORM.tenant_id == tenant_id,
                    ModelVersionORM.version == version,
                )
            )
        )

    @staticmethod
    def _uris(tenant_id: str, model_id: str, version: str, precision: str, request_id: str):
        """Compute every S3 location one precision package touches.

        Layout (solution D12): tar + sidecar at the version package root,
        immutable request/result audit docs under requests|results/, and the
        loose identical payload under staging/<precision>/.
        """
        prefix = f"packages/{model_id}/{version}"
        if tenant_id and tenant_id != "default":
            prefix = f"{tenant_id}/{prefix}"
        base = f"s3://nereidforge/{prefix}"
        return {
            "package_uri": f"{base}/nereid-model-{precision}.tar.gz",
            "sidecar_uri": f"{base}/nereid-model-{precision}.sidecar.json",
            "request_uri": f"{base}/requests/{request_id}.json",
            "result_uri": f"{base}/results/{request_id}.result.json",
            "staging_uri": f"{base}/staging/{precision}/",
        }

    def _read_json_uri(self, uri: str) -> Optional[Any]:
        if not uri.startswith("s3://"):
            return None
        try:
            return json.loads(self.s3.read_object(self.s3.s3_uri_to_filer_path(uri)))
        except Exception:
            return None

    def _source_training_metadata(self, tenant_id: str, md: dict[str, Any]) -> dict[str, Any]:
        source_model = str(md.get("source_model_id") or "")
        source_version = str(md.get("source_version") or "")
        if not source_model or not source_version:
            return {}
        row = self._version_orm(tenant_id, source_model, source_version)
        return row.metadata_ if row else {}

    def _dataset_version_uri(self, tenant_id: str, md: dict[str, Any]) -> str:
        """Resolve only the dataset version POINTER; its content stays in the job."""
        ds_id = str(md.get("dataset_id") or "")
        ds_ver_raw = md.get("dataset_version") or 1
        if not ds_id:
            source_md = self._source_training_metadata(tenant_id, md)
            return self._dataset_version_uri(tenant_id, source_md) if source_md else ""
        try:
            ds_ver = int(ds_ver_raw)
        except (TypeError, ValueError):
            ds_ver = 1
        row = self.db.scalar(
            select(DatasetVersion).where(
                and_(
                    DatasetVersion.dataset_id == ds_id,
                    DatasetVersion.tenant_id == tenant_id,
                    DatasetVersion.version == ds_ver,
                )
            )
        )
        return str(row.uri or "").rstrip("/") if row else ""

    @staticmethod
    def _revision(md: dict[str, Any], version: ModelVersionORM) -> str:
        job_id = str(md.get("job_id") or "")
        tail = job_id.rsplit("-", 1)[-1] if job_id else ""
        return tail if tail.isdigit() else str(version.id)

    def _supporting_artifacts(
        self, tenant_id: str, xml_uri: str, job_id: str
    ) -> list[dict[str, str]]:
        artifacts: list[dict[str, str]] = []
        metrics_uri = ""
        if xml_uri.endswith(".optimized.xml"):
            metrics_uri = xml_uri[: -len(".optimized.xml")] + ".metrics.json"
        if metrics_uri and self.s3.object_size(metrics_uri) > 0:
            artifacts.append({"type": "optimization_metrics", "uri": metrics_uri})

        if job_id:
            report_key = f"evaluation-reports/{job_id}/report.json"
            if tenant_id and tenant_id != "default":
                report_key = f"{tenant_id}/{report_key}"
            report_uri = f"s3://nereidforge/{report_key}"
            if self.s3.object_size(report_uri) > 0:
                artifacts.append({"type": "gate_report", "uri": report_uri})
        return artifacts

    def resolve_request(
        self,
        tenant_id: str,
        model: ModelORM,
        version: ModelVersionORM,
        precision: str,
        request_id: str,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        """Resolve an uploaded ONNX or optimized IR source without sharing graph logic."""
        md = version.metadata_ or {}
        source_md = self._source_training_metadata(tenant_id, md)
        source_uri = str(
            md.get("optimized_uri") or md.get("weights_uri") or version.file_path or ""
        )
        precision_enum_value = precision_enum(precision)
        precision = precision_enum_value.lower()
        uris = self._uris(tenant_id, model.id, version.version, precision, request_id)
        family_hint = str(
            md.get("family")
            or source_md.get("family")
            or md.get("arch")
            or source_md.get("arch")
            or source_md.get("base_model")
            or "yolov8"
        )
        output = {
            "package_uri": uris["package_uri"],
            "sidecar_uri": uris["sidecar_uri"],
            "result_uri": uris["result_uri"],
            "staging_uri": uris["staging_uri"],
        }
        model_identity = {
            "id": model.id,
            "version": version.version,
            "revision": self._revision(md, version),
            "task": model.task_type or "detection",
        }

        if source_uri.lower().endswith(".onnx"):
            if self.s3.object_size(source_uri) <= 0:
                raise ValueError(f"ONNX source artifact is missing: {source_uri}")
            source = {"onnx_uri": source_uri, "family_hint": family_hint}
        else:
            xml_uri = source_uri
            if not xml_uri.endswith(".xml"):
                raise ValueError("edge packaging requires an ONNX or OpenVINO IR .xml artifact")
            bin_uri = str(md.get("ir_bin_uri") or ir_companion_bin(xml_uri) or "")
            if not bin_uri or self.s3.object_size(bin_uri) <= 0:
                raise ValueError(f"OpenVINO companion weights are missing for {xml_uri}")
            dataset_uri = self._dataset_version_uri(tenant_id, md)
            if not dataset_uri:
                raise ValueError("the source model version must reference a dataset version URI")
            source = {
                "ir_xml_uri": xml_uri,
                "ir_bin_uri": bin_uri,
                "dataset_version_uri": dataset_uri + "/",
                "family_hint": family_hint,
                "supporting_artifacts": self._supporting_artifacts(
                    tenant_id, xml_uri, str(md.get("job_id") or "")
                ),
            }

        request = build_package_request(
            request_id=request_id,
            tenant_id=tenant_id,
            model=model_identity,
            source=source,
            precision=precision_enum_value,
            output=output,
        )
        return request, uris

    # -- task/state ---------------------------------------------------------

    @staticmethod
    def list_package_records(metadata: dict[str, Any]) -> list[dict[str, Any]]:
        records = metadata.get("packages")
        return records if isinstance(records, list) else []

    @classmethod
    def available_packages(cls, metadata: dict[str, Any]) -> list[dict[str, Any]]:
        """Return ready precision packages for NH/NL passthrough and download."""
        return [
            r for r in cls.list_package_records(metadata)
            if isinstance(r, dict) and r.get("status") == "ready"
            and str(r.get("precision") or "") in PRECISIONS
        ]

    def enqueue(
        self,
        tenant_id: str,
        model_ref: str,
        version_ref: str,
        precision: str,
        actor: str,
    ) -> dict[str, Any]:
        precision = str(precision or "").strip().lower()
        if precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {sorted(PRECISIONS)}")
        model = self._model_orm(tenant_id, model_ref)
        if not model:
            raise LookupError(f"model {model_ref!r} not found")
        version = self._version_orm(tenant_id, model.id, version_ref)
        if not version:
            raise LookupError(f"model version {version_ref!r} not found")

        md = dict(version.metadata_ or {})
        if not artifact_capability_permitted(md, "package"):
            artifact_kind = str(md.get("artifact_format") or "this artifact")
            raise ValueError(f"{artifact_kind} does not support edge packaging in this release")
        records = [dict(x) for x in self.list_package_records(md)]
        active = next((r for r in records if r.get("precision") == precision), None)
        if active and active.get("status") in ("pending", "packaging", "running"):
            raise RuntimeError(f"{precision} package is already being built")

        request_id = new_id("task")
        request, uris = self.resolve_request(tenant_id, model, version, precision, request_id)
        self.s3.put_object(
            self.s3.s3_uri_to_filer_path(uris["request_uri"]),
            json.dumps(request, ensure_ascii=False, indent=2).encode() + b"\n",
        )

        record = {
            "precision": precision,
            "revision": request["model"]["revision"],
            "hardware_profile": request["hardware"]["profile"],
            "status": "pending",
            "task_id": request_id,
            "job_name": "",
            "request_uri": uris["request_uri"],
            "result_uri": uris["result_uri"],
            "package_uri": uris["package_uri"],
            "sidecar_uri": uris["sidecar_uri"],
            "staging_uri": uris["staging_uri"],
            "package_sha256": "",
            "package_bytes": 0,
            "detection_count": None,
            "devices": [],
            "quality_gate_passed": None,
            "error": "",
            "created_at": _now(),
            "updated_at": _now(),
        }
        records = [r for r in records if r.get("precision") != precision]
        records.append(record)
        md["packages"] = records
        version.metadata_ = md

        params = {
            "tenant": tenant_id,
            "model_id": model.id,
            "version": version.version,
            "precision": precision,
            "revision": request["model"]["revision"],
            "hardware_profile": request["hardware"]["profile"],
            "request_uri": uris["request_uri"],
            "result_uri": uris["result_uri"],
            "package_uri": uris["package_uri"],
            "sidecar_uri": uris["sidecar_uri"],
            "staging_uri": uris["staging_uri"],
        }
        task = AsyncTask(
            id=request_id,
            tenant_id=tenant_id,
            type="model_package",
            status="pending",
            progress=0,
            params=params,
            created_by=actor or "nereidforge-worker",
        )
        self.db.add(task)
        self.db.commit()
        return {"task_id": request_id, **record, "params": params}

    def set_job_name(self, task_id: str, job_name: str) -> None:
        task = self.db.get(AsyncTask, task_id)
        if not task:
            return
        params = dict(task.params or {})
        params["job_name"] = job_name
        task.params = params
        self._update_record(
            str(params.get("model_id") or ""),
            str(params.get("version") or ""),
            str(params.get("precision") or ""),
            {"job_name": job_name, "status": "running", "updated_at": _now()},
            tenant_id=task.tenant_id,
        )

    def mark_failed(self, task: AsyncTask, message: str, code: str = "INFRASTRUCTURE_ERROR") -> None:
        params = task.params or {}
        self._update_record(
            str(params.get("model_id") or ""),
            str(params.get("version") or ""),
            str(params.get("precision") or ""),
            {
                "status": "failed",
                "error": f"{code}: {message}"[:2000],
                "updated_at": _now(),
            },
            tenant_id=task.tenant_id,
        )

    @staticmethod
    def _device_summary(verification: dict[str, Any]) -> list[dict[str, str]]:
        out = []
        for d in verification.get("devices") or []:
            if isinstance(d, dict) and d.get("device"):
                out.append({
                    "device": str(d["device"]),
                    "status": str(d.get("status") or "unverified"),
                })
        return out

    def record_result(self, task: AsyncTask) -> dict[str, Any]:
        params = task.params or {}
        result_uri = str(params.get("result_uri") or "")
        result = self._read_json_uri(result_uri)
        if not isinstance(result, dict):
            raise RuntimeError(f"package result missing or unreadable: {result_uri}")
        parsed = parse_package_result(result)
        verification = parsed.get("verification") or {}
        devices = self._device_summary(verification)
        gate = verification.get("quality_gate") if isinstance(
            verification.get("quality_gate"), dict
        ) else {}
        if parsed["status"] == "failed":
            error = parsed["error"]
            record = {
                "status": "failed",
                "error": f"{error.get('code') or 'FAILED'}: {error.get('message') or ''}"[:2000],
                "detection_count": verification.get("detection_count"),
                "devices": devices,
                "updated_at": _now(),
            }
        else:
            package = parsed["package"]
            record = {
                "status": "ready",
                "error": "",
                "task_id": task.id,
                "revision": str(params.get("revision") or ""),
                "hardware_profile": str(params.get("hardware_profile") or ""),
                "job_name": str(params.get("job_name") or ""),
                "request_uri": str(params.get("request_uri") or ""),
                "result_uri": result_uri,
                "package_uri": str(package.get("uri") or ""),
                "sidecar_uri": parsed.get("sidecar_uri") or params.get("sidecar_uri") or "",
                "package_sha256": str(package.get("sha256") or ""),
                "package_bytes": int(package.get("bytes") or 0),
                "staging_uri": parsed.get("staging_uri") or params.get("staging_uri") or "",
                "detection_count": verification.get("detection_count"),
                "devices": devices,
                "quality_gate_passed": (
                    bool(gate.get("passed")) if gate else None
                ),
                "map50_baseline": gate.get("baseline"),
                "map50_candidate": gate.get("candidate"),
                "updated_at": _now(),
            }
        self._update_record(
            str(params.get("model_id") or ""),
            str(params.get("version") or ""),
            str(params.get("precision") or ""),
            record,
            tenant_id=task.tenant_id,
        )
        return record

    def _update_record(
        self,
        model_id: str,
        version: str,
        precision: str,
        patch: dict[str, Any],
        tenant_id: str = "default",
    ) -> None:
        if not model_id or not version or precision not in PRECISIONS:
            return
        row = self._version_orm(tenant_id, model_id, version)
        if row is None:
            return
        md = dict(row.metadata_ or {})
        records = [dict(x) for x in self.list_package_records(md)]
        for rec in records:
            if rec.get("precision") == precision:
                rec.update(patch)
                break
        else:
            records.append({"precision": precision, **patch})
        md["packages"] = records
        row.metadata_ = md
        self.db.commit()
