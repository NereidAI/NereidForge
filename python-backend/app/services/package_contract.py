# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Versioned request/result contract for the edge-model packager job.

NF core writes the immutable request and consumes the generic result. It must
not share COCO, OpenVINO, decoder, OVMS, or coordinate-space helpers with the
packager container (ADR-017).

IR packages remain precision-specific (FP16/INT8) and device-verified. Direct
ONNX packages are FP32-only and use structural validation without inventing a
dataset, golden output, or device-execution claim.
"""
from __future__ import annotations

from typing import Any

PACKAGE_REQUEST_SCHEMA = "nereidforge.package-request/1"
PACKAGE_RESULT_SCHEMA = "nereidforge.package-result/1"

# IR uses the evaluator-xpu CPU/GPU profile (pit 174/177); direct ONNX uses
# a CPU-only structural profile and never reserves or claims a GPU check.
EDGE_VERIFIED = "edge-verified"
ONNX_STRUCTURAL = "onnx-structural"
HARDWARE_PROFILES = {EDGE_VERIFIED, ONNX_STRUCTURAL}

FP16 = "FP16"
INT8 = "INT8"
FP32 = "FP32"
PACKAGE_PRECISIONS = {FP16, FP32, INT8}
IR_PACKAGE_PRECISIONS = {FP16, INT8}
_PRECISION_BY_LOWER = {"fp16": FP16, "int8": INT8, "fp32": FP32}

DEFAULT_POLICY = {
    "confidence_threshold": 0.25,
    "nms_iou_threshold": 0.45,
    "score_delta": 0.02,
    "box_iou": 0.9,
}


def precision_enum(value: Any) -> str:
    """Normalize an API precision to its uppercase descriptor enum."""
    key = str(value or "").strip().lower()
    if key not in _PRECISION_BY_LOWER:
        raise ValueError(f"precision must be one of {sorted(_PRECISION_BY_LOWER)}")
    return _PRECISION_BY_LOWER[key]


def _require(value: Any, field: str) -> Any:
    """Reject a missing or whitespace-only scalar contract field."""
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(f"{field} is required")
    return value


def _require_s3_uri(value: Any, field: str) -> str:
    """Validate a non-empty s3://bucket/key contract URI field."""
    uri = str(_require(value, field)).strip()
    if not uri.startswith("s3://") or len(uri.split("/", 3)) < 4:
        raise ValueError(f"{field} must be a non-empty s3://bucket/key URI")
    return uri


def _require_section(value: Any, field: str) -> dict[str, Any]:
    """Validate that a contract section is a JSON object."""
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return value


def build_package_request(
    *,
    request_id: str,
    tenant_id: str,
    model: dict[str, Any],
    source: dict[str, Any],
    precision: str,
    output: dict[str, Any],
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and render an immutable ``package-request/1`` document."""
    request_id = str(_require(request_id, "request_id"))
    tenant_id = str(_require(tenant_id, "tenant_id"))
    precision_enum_value = precision_enum(precision)
    model = _require_section(model, "model")
    source = _require_section(source, "source")
    output = _require_section(output, "output")
    onnx_source = source.get("onnx_uri") is not None
    if onnx_source and precision_enum_value != FP32:
        raise ValueError("direct ONNX packages require fp32 precision")
    if not onnx_source and precision_enum_value not in IR_PACKAGE_PRECISIONS:
        raise ValueError(f"IR edge packages support only {sorted(IR_PACKAGE_PRECISIONS)}")

    policy_defaults = dict(DEFAULT_POLICY)
    if onnx_source:
        # OpenCV's modern YuNet reference uses 0.3 IoU; IR/YOLO keeps 0.45.
        policy_defaults["nms_iou_threshold"] = 0.3
    merged_policy = {**policy_defaults, **(policy or {})}
    for key in DEFAULT_POLICY:
        try:
            merged_policy[key] = float(merged_policy[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"policy.{key} must be numeric") from exc
        if merged_policy[key] < 0:
            raise ValueError(f"policy.{key} must not be negative")

    artifacts = []
    for item in source.get("supporting_artifacts") or []:
        if not isinstance(item, dict):
            raise ValueError("source.supporting_artifacts entries must be objects")
        artifact_type = str(_require(item.get("type"), "supporting artifact type"))
        artifacts.append({
            "type": artifact_type,
            "uri": _require_s3_uri(item.get("uri"), f"supporting artifact {artifact_type}.uri"),
        })

    if onnx_source:
        if any(source.get(key) for key in ("ir_xml_uri", "ir_bin_uri", "dataset_version_uri")):
            raise ValueError("ONNX and IR package sources are mutually exclusive")
        request_source = {
            "onnx_uri": _require_s3_uri(source.get("onnx_uri"), "source.onnx_uri"),
            "supporting_artifacts": artifacts,
        }
        hardware = {
            "profile": ONNX_STRUCTURAL,
            "require_numeric_verification": False,
            "verify_devices": [],
        }
    else:
        request_source = {
            "ir_xml_uri": _require_s3_uri(source.get("ir_xml_uri"), "source.ir_xml_uri"),
            "ir_bin_uri": _require_s3_uri(source.get("ir_bin_uri"), "source.ir_bin_uri"),
            "dataset_version_uri": _require_s3_uri(
                source.get("dataset_version_uri"), "source.dataset_version_uri"
            ),
            "supporting_artifacts": artifacts,
        }
        # Existing IR packages retain mandatory CPU/GPU numerical verification.
        hardware = {
            "profile": EDGE_VERIFIED,
            "require_numeric_verification": True,
            "verify_devices": ["CPU", "GPU"],
        }

    request = {
        "schema": PACKAGE_REQUEST_SCHEMA,
        "request_id": request_id,
        "tenant_id": tenant_id,
        "model": {
            "id": str(_require(model.get("id"), "model.id")),
            "version": str(_require(model.get("version"), "model.version")),
            "revision": str(_require(model.get("revision"), "model.revision")),
            "task": str(_require(model.get("task"), "model.task")),
        },
        "precision": precision_enum_value,
        "source": request_source,
        "hardware": hardware,
        "output": {
            "package_uri": _require_s3_uri(output.get("package_uri"), "output.package_uri"),
            "sidecar_uri": _require_s3_uri(output.get("sidecar_uri"), "output.sidecar_uri"),
            "result_uri": _require_s3_uri(output.get("result_uri"), "output.result_uri"),
            "staging_uri": _require_s3_uri(output.get("staging_uri"), "output.staging_uri"),
        },
        "policy": merged_policy,
    }
    family_hint = str(source.get("family_hint") or "").strip()
    if onnx_source and not family_hint:
        raise ValueError("source.family_hint is required for direct ONNX packaging")
    if family_hint:
        request["source"]["family_hint"] = family_hint
    return request


def parse_package_result(result: Any) -> dict[str, Any]:
    """Normalize a ``package-result/1`` document for backend state updates."""
    if not isinstance(result, dict):
        raise ValueError("package result must be a JSON object")
    if result.get("schema") != PACKAGE_RESULT_SCHEMA:
        raise ValueError(f"expected {PACKAGE_RESULT_SCHEMA} result")
    request_id = str(_require(result.get("request_id"), "result.request_id"))
    status = str(_require(result.get("status"), "result.status")).lower()
    if status not in {"succeeded", "failed"}:
        raise ValueError("result.status must be succeeded or failed")

    precision = precision_enum(result.get("precision") or FP16)
    raw_error = result.get("error") if isinstance(result.get("error"), dict) else {}
    error = {
        "code": str(raw_error.get("code") or ""),
        "message": str(raw_error.get("message") or ""),
    }
    package: dict[str, Any] = {}
    sidecar_uri = ""
    staging_uri = ""

    if status == "succeeded":
        raw_package = result.get("package")
        if not isinstance(raw_package, dict):
            raise ValueError("succeeded result must include package")
        uri = _require_s3_uri(raw_package.get("uri"), "package.uri")
        try:
            size = int(raw_package.get("bytes") or 0)
        except (TypeError, ValueError) as exc:
            raise ValueError("package.bytes must be an integer") from exc
        sha = str(_require(raw_package.get("sha256"), "package.sha256"))
        if size <= 0 or len(sha) < 16:
            raise ValueError("succeeded result has an invalid package")
        sidecar_uri = _require_s3_uri(result.get("sidecar_uri"), "sidecar_uri")
        raw_staging = result.get("staging") if isinstance(result.get("staging"), dict) else {}
        staging_uri = str(raw_staging.get("uri") or "")
        if staging_uri and not staging_uri.startswith("s3://"):
            raise ValueError("staging.uri must be an s3:// URI")
        package = {"uri": uri, "bytes": size, "sha256": sha}
    elif not error["code"]:
        raise ValueError("failed result must include a machine-readable error code")

    verification = result.get("verification")
    if verification is not None and not isinstance(verification, dict):
        raise ValueError("result.verification must be an object when present")

    return {
        "request_id": request_id,
        "precision": precision,
        "status": status,
        "error": error,
        "package": package,
        "sidecar_uri": sidecar_uri,
        "staging_uri": staging_uri,
        "verification": verification or {},
    }
