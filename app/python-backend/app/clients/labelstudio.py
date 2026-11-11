# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Label Studio REST client.

Mirrors the Go LabelStudioClient (internal/service/labelstudio.go):
- project/task/storage/ml-backend management
- annotation and prediction persistence
- direct ML backend /predict calls for semi-auto annotation
- s3:// image URL resolution helpers
"""
from __future__ import annotations

import base64
from typing import Any, Optional

import httpx

from app.core.config import config


class LabelStudioClient:
    """Thin REST client for Label Studio + its ML backend predict endpoint."""

    def __init__(self) -> None:
        self.base_url = (config.labelstudio.host or "").rstrip("/")
        self.public_url = (config.labelstudio.public_host or self.base_url).rstrip("/")
        self.token = config.labelstudio.token
        if not self.token:
            # Env fallback: run-dev.sh injects NEREIDFORGE_LABELSTUDIO_TOKEN but
            # the yaml token stays empty; make sure it is honored even when the
            # config object was built before the env var was visible.
            import os
            self.token = os.environ.get("NEREIDFORGE_LABELSTUDIO_TOKEN", "")
        self.ml_backend_url = config.labelstudio.ml_backend_url
        self.ml_backend_cluster = config.labelstudio.ml_backend_cluster_url
        self.timeout = httpx.Timeout(config.labelstudio.timeout or 30)

    def enabled(self) -> bool:
        """Return True when Label Studio is configured."""
        return bool(self.base_url and self.token)

    def _headers(self) -> dict[str, str]:
        if not self.token:
            return {}
        return {"Authorization": f"Token {self.token}"}

    def _get(self, path: str, params: Optional[dict] = None) -> Any:
        resp = httpx.get(f"{self.base_url}{path}", headers=self._headers(), params=params, timeout=self.timeout)
        if resp.status_code >= 400:
            raise RuntimeError(f"labelstudio get {path} failed: {resp.status_code} {resp.text[:300]}")
        return resp.json()

    def _post(self, path: str, body: Any) -> Any:
        resp = httpx.post(f"{self.base_url}{path}", json=body, headers=self._headers(), timeout=self.timeout)
        if resp.status_code not in (200, 201):
            raise RuntimeError(f"labelstudio post {path} failed: {resp.status_code} {resp.text[:300]}")
        return resp.json()

    def _patch(self, path: str, body: Any) -> Any:
        resp = httpx.patch(f"{self.base_url}{path}", json=body, headers=self._headers(), timeout=self.timeout)
        if resp.status_code >= 400:
            raise RuntimeError(f"labelstudio patch {path} failed: {resp.status_code} {resp.text[:300]}")
        return resp.json()

    def _delete(self, path: str) -> None:
        resp = httpx.delete(f"{self.base_url}{path}", headers=self._headers(), timeout=self.timeout)
        if resp.status_code >= 300:
            raise RuntimeError(f"labelstudio delete {path} failed: {resp.status_code} {resp.text[:300]}")

    # -- projects -----------------------------------------------------------

    def create_project(self, title: str, description: str, label_config: str) -> dict[str, Any]:
        """Create an LS project."""
        return self._post("/api/projects/", {
            "title": title,
            "description": description,
            "label_config": label_config,
            "show_skip_button": True,
            "enable_empty_annotation": True,
        })

    def update_label_config(self, project_id: int, label_config: str) -> dict[str, Any]:
        """Replace an LS project's label_config (used to add labels mid-task).

        Existing annotations referencing unchanged label values are preserved;
        only the available label set is extended.
        """
        return self._patch(f"/api/projects/{project_id}", {"label_config": label_config})

    def get_project(self, project_id: int) -> dict[str, Any]:
        """Get an LS project by id."""
        return self._get(f"/api/projects/{project_id}/")

    def list_projects(self) -> list[dict[str, Any]]:
        """List all LS projects."""
        data = self._get("/api/projects/")
        return data if isinstance(data, list) else data.get("results", [])

    # -- storage / ml-backend -----------------------------------------------

    def create_s3_storage(self, project_id: int, bucket: str, endpoint: str, access_key: str, secret_key: str) -> None:
        """Attach an S3 source storage to an LS project (presign=False so the LS
        server proxies image bytes over its own host)."""
        self._post("/api/storages/s3/", {
            "project": project_id,
            "bucket": bucket,
            "s3_endpoint": endpoint,
            "region_name": "us-east-1",
            "aws_access_key_id": access_key,
            "aws_secret_access_key": secret_key,
            "prefix": "",
            "use_blob_urls": False,
            "presign": False,
        })

    def register_ml_backend(self, project_id: int, url: str, title: str) -> Optional[int]:
        """Register an ML backend (uses the cluster-internal URL the LS pod can reach)."""
        if not url:
            return None
        data = self._post("/api/ml/", {
            "project": project_id,
            "url": url,
            "title": title,
            "auto_update": True,
        })
        return data.get("id")

    # -- tasks --------------------------------------------------------------

    def import_tasks(self, project_id: int, tasks: list[dict]) -> None:
        """Import tasks into an LS project."""
        if not tasks:
            return
        self._post(f"/api/projects/{project_id}/import", tasks)

    def delete_all_tasks(self, project_id: int) -> None:
        """Delete all tasks in an LS project."""
        self._delete(f"/api/projects/{project_id}/tasks/")

    def list_tasks(
        self, project_id: int, page: int = 1, page_size: int = 50,
        include_results: bool = False,
    ) -> dict[str, Any]:
        """list_tasks returns one page, optionally embedding annotation/prediction results.

        Full results are opt-in because background consumers need only task metadata,
        while the annotation studio would otherwise require one detail request per task.
        """
        params: dict[str, Any] = {"project": project_id, "page": page, "page_size": page_size}
        if include_results:
            params["fields"] = "all"
        return self._get("/api/tasks/", params)

    def get_task(self, task_id: int) -> dict[str, Any]:
        """Get a single task with annotations/predictions."""
        return self._get(f"/api/tasks/{task_id}/")

    # -- annotations / predictions ------------------------------------------

    def save_annotation(self, task_id: int, result: list[Any]) -> None:
        """Save an annotation for a task."""
        self._post(f"/api/tasks/{task_id}/annotations/", {"result": result})

    def save_prediction(self, task_id: int, result: list[Any], score: float, model_version: str) -> None:
        """Save a model prediction for a task (task_id is the task's integer pk)."""
        self._post("/api/predictions/", {
            "task": task_id,
            "result": result,
            "score": score,
            "model_version": model_version,
        })

    def list_predictions(self, task_id: int) -> list[dict[str, Any]]:
        """List the predictions attached to a single task.

        Label Studio accepts a ``task`` filter on the predictions endpoint and
        returns either a bare list or a paginated ``results`` envelope.
        """
        data = self._get("/api/predictions/", {"task": task_id})
        if isinstance(data, list):
            return data
        results = data.get("results", [])
        return results if isinstance(results, list) else []

    def delete_prediction(self, prediction_id: int) -> None:
        """Delete a single prediction by its integer pk."""
        self._delete(f"/api/predictions/{prediction_id}/")

    def replace_prediction(
        self, task_id: int, result: list[Any], score: float, model_version: str
    ) -> None:
        """Refresh a task's auto prediction: drop prior predictions then save the new one.

        Pre-annotation is machine-generated state, so re-running it must
        overwrite stale boxes rather than stack a duplicate prediction every
        time the user clicks "pre-annotate". Human annotations are never
        touched here.
        """
        for old in self.list_predictions(task_id):
            pid = old.get("id")
            if pid is None:
                continue
            try:
                self.delete_prediction(int(pid))
            except Exception:
                # A stale prediction that fails to delete must not block saving
                # the fresh one; the new prediction is still written below.
                pass
        self.save_prediction(task_id, result, score, model_version)

    # -- ml backend predict -------------------------------------------------

    def export_annotations(self, project_id: int) -> list[dict[str, Any]]:
        """Export all tasks + annotations for a project (LS exportType=JSON)."""
        data = self._get(f"/api/projects/{project_id}/export", {"exportType": "JSON"})
        return data if isinstance(data, list) else []

    def run_ml_backend_predict(
        self, ml_url: str, tasks: list[dict], project_id: int, label_config: str
    ) -> list[dict[str, Any]]:
        """Call the ML backend /predict endpoint directly (no LS token needed)."""
        body = {
            "tasks": tasks,
            "project": str(project_id),
            "label_config": label_config,
            "params": {"context": {"project": str(project_id)}},
        }
        resp = httpx.post(f"{ml_url.rstrip('/')}/predict", json=body, timeout=self.timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"ml-backend predict failed: {resp.status_code} {resp.text[:300]}")
        data = resp.json()
        return data.get("results", []) or []

    # -- image URL helpers --------------------------------------------------

    def resolve_image_url(self, raw: Any, project_id: int) -> str:
        """Resolve a task image reference to a usable URL (http/s3 passthrough)."""
        if isinstance(raw, str):
            v = raw.strip()
            if v and (v.startswith("http://") or v.startswith("https://") or v.startswith("s3://")):
                return v
            if v.startswith("/data/upload/"):
                return self.public_url + v
            return ""
        return ""


def resolve_ls_image(raw: Any, ls_project_id: int) -> str:
    """Decode an LS task image reference into a unified s3:// path.

    LS rewrites s3://... into /tasks/{id}/resolve/?fileuri=<base64(s3://...)>;
    here we decode it back to s3:// for the backend (S3->HTTP proxy) to serve.
    """
    if not isinstance(raw, str) or not raw:
        return ""
    v = raw.strip()
    if v.startswith("s3://") or v.startswith("http://") or v.startswith("https://"):
        return v
    if "/resolve/" in v and "fileuri=" in v:
        idx = v.index("fileuri=") + len("fileuri=")
        try:
            decoded = base64.b64decode(v[idx:]).decode()
        except Exception:
            return ""
        return decoded
    return ""


def build_data_yaml(categories: list[str]) -> str:
    """Build a data.yaml (PP-YOLOE format) from a category list."""
    lines = [
        "names:",
    ]
    for i, c in enumerate(categories):
        lines.append(f"  {i}: {c}")
    lines.append("nc: " + str(len(categories)))
    return "\n".join(lines)


def inject_predict_attrs(label_config: str, score_threshold: float | None = None, model_path: str = "") -> str:
    """Inject ML-backend control-tag attrs into a label_config XML string.

    The NF ML backend reads these attributes off the control tag:
    - model_score_threshold: per-request confidence floor (overrides the
      MODEL_SCORE_THRESHOLD env default of 0.5, which is too high for
      small-object video)
    - model_path: weights file to run (local filename under MODEL_ROOT)
    """
    if score_threshold is None and not model_path:
        return label_config
    attrs = ""
    if score_threshold is not None:
        attrs += f' model_score_threshold="{min(1.0, max(0.01, float(score_threshold)))}"'
    if model_path:
        attrs += f' model_path="{model_path}"'
    # attach to the first control tag opening (RectangleLabels/PolygonLabels/...)
    import re

    def _inject(m: re.Match) -> str:
        tag = m.group(1)
        rest = m.group(2)
        return f"<{tag}{attrs}{rest}"

    out, n = re.subn(r"<(RectangleLabels|PolygonLabels|BrushLabels|KeyPointLabels|Choices)([^>]*)", _inject, label_config, count=1)
    return out if n else label_config


def build_label_config(schema: dict | None) -> str:
    """Generate an LS label_config XML from a project schema.

    Supports raw_config passthrough, labels+type, and a default COCO template.
    """
    if not schema:
        return _default_label_config()
    raw_cfg = schema.get("raw_config")
    if isinstance(raw_cfg, str) and raw_cfg.strip():
        return raw_cfg

    labels = schema.get("labels") or []
    rect_type = schema.get("type", "rectanglelabels")
    task_type = schema.get("task_type", "")

    # Text tasks need no labels.
    text_cfg = _text_task_config(task_type, labels)
    if text_cfg:
        return text_cfg

    if not labels:
        return _default_label_config()

    # OCR projects: one "text" rectangle per detected line, model_ocr routes
    # the control to the PaddleOCR backend instead of PP-YOLOE.
    if task_type == "ocr":
        return (
            "<View>\n"
            '  <Image name="image" value="$image"/>\n'
            '  <RectangleLabels name="label" toName="image" model_ocr="true">\n'
            '    <Label value="text"/>\n'
            "  </RectangleLabels>\n"
            "</View>"
        )

    tags = "".join(f'<Label value="{label}"/>' for label in labels)
    # Image classification: Choices control attached to the Image object.
    if rect_type == "choices" and task_type not in (
        "text-classification", "ner", "summarization",
    ):
        choices = "".join(f'<Choice value="{label}"/>' for label in labels)
        return (
            "<View>\n"
            '  <Image name="image" value="$image"/>\n'
            f'  <Choices name="label" toName="image">{choices}</Choices>\n'
            "</View>"
        )
    tag_map = {
        "polygonlabels": "PolygonLabels",
        "keypointlabels": "KeyPointLabels",
        "brushlabels": "BrushLabels",
        "choices": "Choices",
    }
    control = tag_map.get(rect_type, "RectangleLabels")
    return (
        "<View>\n"
        '  <Image name="image" value="$image"/>\n'
        f'  <{control} name="label" toName="image">{tags}</{control}>\n'
        "</View>"
    )


def _default_label_config() -> str:
    """Default COCO-style detection label config."""
    tags = "".join(f'<Label value="{label}"/>' for label in DEFAULT_DETECTION_LABELS)
    return (
        "<View>\n"
        '  <Image name="image" value="$image"/>\n'
        f'  <RectangleLabels name="label" toName="image">{tags}</RectangleLabels>\n'
        "</View>"
    )


# Canonical default detection label set (single source of truth). Used when a
# project is created without an explicit label set so the stored project schema,
# the provisioned LS/CVAT label config, and the NF annotation UI all agree.
DEFAULT_DETECTION_LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
]


def _text_task_config(task_type: str, labels: list[Any]) -> str:
    """Return a label config for text tasks, or empty if not applicable.

    task_type → control tag mapping:
    - text-classification: Choices over the label set
    - ner: per-token Labels (proper NER spans, not a TextArea dump)
    - summarization: TextArea (free-form summary)
    - text-generation / chat-rlhf: Hypertext + TextArea chat transcript
    """
    if task_type == "text-classification":
        choices = "".join(f'<Choice value="{label}"/>' for label in labels)
        return (
            "<View>\n"
            '  <Text name="text" value="$text"/>\n'
            f'  <Choices name="label" toName="text" showInLine="true">{choices}</Choices>\n'
            "</View>"
        )
    if task_type == "ner":
        tags = "".join(f'<Label value="{label}"/>' for label in labels)
        return (
            "<View>\n"
            '  <Text name="text" value="$text"/>\n'
            f'  <Labels name="label" toName="text">{tags}</Labels>\n'
            "</View>"
        )
    if task_type == "summarization":
        return (
            "<View>\n"
            '  <Text name="text" value="$text"/>\n'
            '  <TextArea name="label" toName="text" rows="4" '
            'placeholder="Write the summary..."/>\n'
            "</View>"
        )
    if task_type in ("text-generation", "chat-rlhf"):
        # chat transcript + generation/feedback areas; tasks carry the
        # conversation in $chat as preformatted hypertext
        return (
            "<View>\n"
            '  <Hypertext name="chat" value="$chat"/>\n'
            '  <Header value="Response"/>\n'
            '  <TextArea name="response" toName="chat" rows="6" '
            'placeholder="Model output / corrected response..."/>\n'
            '  <Header value="Rating"/>\n'
            '  <Rating name="rating" toName="chat" maxRating="5"/>\n'
            "</View>"
        )
    return ""


# Module-level singleton.
ls_client: LabelStudioClient = LabelStudioClient()
