# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""CVAT REST client.

Mirrors the LabelStudioClient pattern but targets CVAT 2.x API:
- project / task / annotation management
- cloud-storage-aware data import (SeaweedFS S3)
- annotation read/write for semi-auto annotation (import_mode=append)
- dataset export for training-data extraction

Key differences from LS:
- Auth: Token <PAT> (DRF token auth)
- Labels: JSON spec (not XML)
- Cloud storage: deployment-level, projects reference bucket+prefix
- Predictions: written directly as annotations (CVAT 2.x has no prediction layer)
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

import httpx

from app.core.config import config

logger = logging.getLogger(__name__)


class CvatClient:
    """Thin REST client for CVAT 2.x API."""

    def __init__(self) -> None:
        self.timeout = httpx.Timeout(config.cvat.timeout or 30)
        # Shared pooled client: the frame proxy fires many concurrent requests
        # during video playback. Creating a new connection per request (module
        # level httpx.get) paid TCP+ingress setup on every frame and measurably
        # throttled playback. Keep-alive reuse with a generous pool lets the
        # sync route's worker threads fetch frames in parallel. httpx.Client is
        # safe to share across threads.
        self._http = httpx.Client(
            timeout=self.timeout,
            limits=httpx.Limits(
                max_connections=64,
                max_keepalive_connections=32,
                keepalive_expiry=60.0,
            ),
        )

    def _base_url(self) -> str:
        return (config.cvat.host or os.environ.get("NEREIDFORGE_CVAT_HOST", "") or "").rstrip("/")

    def _public_url(self) -> str:
        return (config.cvat.public_host or os.environ.get("NEREIDFORGE_CVAT_PUBLIC_HOST", "") or self._base_url()).rstrip("/")

    def _token(self) -> str:
        return config.cvat.token or os.environ.get("NEREIDFORGE_CVAT_TOKEN", "")

    def enabled(self) -> bool:
        host = self._base_url()
        token = self._token()
        logger.debug("CvatClient.enabled check: host=%s token_set=%s", bool(host), bool(token))
        return bool(host and token)

    def _headers(self) -> dict[str, str]:
        token = self._token()
        if not token:
            return {}
        return {"Authorization": f"Token {token}"}

    def _get(self, path: str, params: Optional[dict] = None) -> Any:
        """GET JSON from CVAT API."""
        resp = self._http.get(
            f"{self._base_url()}{path}", headers=self._headers(),
            params=params,
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"cvat get {path} failed: {resp.status_code} {resp.text[:300]}")
        return resp.json()

    def _post(self, path: str, body: Any) -> Any:
        """POST JSON to CVAT API (accepts 200/201)."""
        resp = self._http.post(
            f"{self._base_url()}{path}", json=body, headers=self._headers(),
        )
        if resp.status_code not in (200, 201):
            raise RuntimeError(f"cvat post {path} failed: {resp.status_code} {resp.text[:300]}")
        return resp.json()

    def _patch(self, path: str, body: Any) -> Any:
        """PATCH JSON on CVAT API."""
        resp = self._http.patch(
            f"{self._base_url()}{path}", json=body, headers=self._headers(),
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"cvat patch {path} failed: {resp.status_code} {resp.text[:300]}")
        return resp.json()

    def _put(self, path: str, body: Any, params: Optional[dict] = None) -> Any:
        """PUT JSON on CVAT API (whole-resource replace semantics)."""
        resp = self._http.put(
            f"{self._base_url()}{path}", json=body, headers=self._headers(),
            params=params,
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"cvat put {path} failed: {resp.status_code} {resp.text[:300]}")
        if resp.status_code == 204 or not resp.content:
            return {}
        return resp.json()

    def _delete(self, path: str) -> None:
        """DELETE a resource on CVAT API."""
        resp = self._http.delete(
            f"{self._base_url()}{path}", headers=self._headers(),
        )
        if resp.status_code >= 300:
            raise RuntimeError(f"cvat delete {path} failed: {resp.status_code} {resp.text[:300]}")

    # -- projects -----------------------------------------------------------

    def create_project(
        self, name: str, description: str, labels: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Create a CVAT project with a JSON label spec.

        Each label dict should have: name, color (hex), and optionally
        attributes, type (e.g. "rectangle", "polygon", "points", "ellipse").
        """
        return self._post("/api/projects", {
            "name": name,
            "description": description,
            "labels": labels,
        })

    def get_project(self, project_id: int) -> dict[str, Any]:
        """Get a CVAT project by id."""
        return self._get(f"/api/projects/{project_id}")

    def list_labels(self, project_id: int) -> list[dict[str, Any]]:
        """List labels of a CVAT project as dicts (id, name, color).

        Some CVAT versions return label names as plain strings on the
        project endpoint; this endpoint always returns full dicts.
        """
        data = self._get("/api/labels", {"project_id": project_id, "page_size": 100})
        results = data.get("results") or []
        return [lb for lb in results if isinstance(lb, dict)]

    def add_labels(self, project_id: int, names: list[str]) -> list[dict[str, Any]]:
        """Append labels to an existing CVAT project.

        CVAT PATCH /api/projects semantics: labels carrying an id are updated
        (kept as-is here), labels without an id are created; labels omitted
        from the list would be deleted, so the full existing set is resent.
        Direct POST /api/labels is not allowed (405). Returns the label list.
        """
        existing = self.list_labels(project_id)
        have = {str(lb.get("name")) for lb in existing}
        payload = [
            {"id": lb.get("id"), "name": lb.get("name"), "color": lb.get("color"),
             "type": lb.get("type", "rectangle")}
            for lb in existing
        ]
        palette = ("#e60049", "#0bb4ff", "#50e991", "#e6d800", "#9b19f5",
                   "#ffa300", "#dc0ab4", "#b3d4ff", "#00bfa0", "#fabed4")
        for i, name in enumerate(names):
            if name in have:
                continue
            payload.append({"name": name, "color": palette[len(payload) % len(palette)],
                            "type": "rectangle"})
        if payload:
            self._patch(f"/api/projects/{project_id}", {"labels": payload})
        return self.list_labels(project_id)


    def resolve_label_map(self, project_id: int) -> dict[int, str]:
        """Return {label_id: label_name} for a project (robust to str labels)."""
        proj_labels: list[Any] = []
        try:
            proj_labels = self.get_project(project_id).get("labels") or []
        except Exception:
            pass
        mapping: dict[int, str] = {}
        for lb in proj_labels:
            if isinstance(lb, dict) and lb.get("id") is not None:
                mapping[int(lb["id"])] = str(lb.get("name", ""))
        if not mapping:
            for lb in self.list_labels(project_id):
                if lb.get("id") is not None:
                    mapping[int(lb["id"])] = str(lb.get("name", ""))
        return mapping

    # -- cloud storages -------------------------------------------------------

    def list_cloudstorages(self) -> list[dict[str, Any]]:
        """List cloud storages configured on the CVAT deployment."""
        data = self._get("/api/cloudstorages", {"page": 1, "page_size": 50})
        items = data.get("results") if isinstance(data, dict) else data
        return [cs for cs in (items or []) if isinstance(cs, dict)]

    def find_cloud_storage(self) -> Optional[int]:
        """Find the cloud storage id that mounts the SeaweedFS bucket.

        Matches by friendly name containing 'nereidforge' or 'seaweed'
        (case-insensitive). Returns None when no storage is configured.
        """
        for cs in self.list_cloudstorages():
            name = str(cs.get("display_name", "") or cs.get("name", "")).lower()
            if "nereidforge" in name or "seaweed" in name:
                return int(cs.get("id", 0) or 0)
        return None

    def delete_project(self, project_id: int) -> None:
        """Delete a CVAT project (cascades to tasks/jobs)."""
        self._delete(f"/api/projects/{project_id}")

    # -- tasks --------------------------------------------------------------

    def create_task(
        self, project_id: int, name: str,
        subset: str = "default", segment_size: int = 0,
    ) -> dict[str, Any]:
        """Create a task under a CVAT project.

        segment_size=0 means the whole video is one segment (one job).
        """
        body: dict[str, Any] = {
            "name": name,
            "project_id": project_id,
            "subset": subset,
        }
        if segment_size > 0:
            body["segment_size"] = segment_size
        return self._post("/api/tasks", body)

    def get_task(self, task_id: int) -> dict[str, Any]:
        """Get a single CVAT task by id."""
        return self._get(f"/api/tasks/{task_id}")

    def list_tasks(
        self, project_id: int, page: int = 1, page_size: int = 100,
    ) -> dict[str, Any]:
        """List tasks in a CVAT project with pagination."""
        return self._get("/api/tasks", {
            "project_id": project_id,
            "page": page,
            "page_size": page_size,
        })

    def delete_task(self, task_id: int) -> None:
        """Delete a CVAT task."""
        self._delete(f"/api/tasks/{task_id}")

    def list_jobs(self, task_id: int) -> list[dict[str, Any]]:
        """List jobs of a CVAT task (one segment -> one job)."""
        data = self._get("/api/jobs", {"task_id": task_id, "page_size": 100})
        results = data.get("results") if isinstance(data, dict) else data
        return [j for j in (results or []) if isinstance(j, dict)]

    def update_job(self, job_id: int, **fields: Any) -> dict[str, Any]:
        """Patch a CVAT job (e.g. stage/state).

        stage in {annotation, validation, acceptance}; state in
        {new, in progress, rejected, completed}. Marking every job completed
        drives the parent task's progress to 100.
        """
        return self._patch(f"/api/jobs/{job_id}", fields)

    # -- data upload --------------------------------------------------------

    def upload_data(
        self, task_id: int, server_files: list[str],
        cloud_storage_id: Optional[int] = None,
        image_quality: int = 70,
    ) -> dict[str, Any]:
        """Upload data to a CVAT task from server-side paths.

        server_files are relative paths under the CVAT share or cloud storage.
        For cloud storage, pass cloud_storage_id and paths relative to the
        bucket root. image_quality is required by CVAT for image/video tasks.

        ``use_zip_chunks`` only applies to image sets (it zips frame chunks for
        faster download). Video tasks extract frames server-side and must NOT
        send it (CVAT rejects/ignores it for video media), so it is omitted when
        any of the files is a video.
        """
        body: dict[str, Any] = {
            "server_files": server_files,
            "image_quality": image_quality,
            "use_cache": True,
        }
        video_exts = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v")
        is_video = any(str(f).lower().endswith(video_exts) for f in server_files)
        if not is_video:
            body["use_zip_chunks"] = True
        if cloud_storage_id:
            body["cloud_storage_id"] = cloud_storage_id
        # CVAT data upload is async; this call starts it.
        resp = self._http.post(
            f"{self._base_url()}/api/tasks/{task_id}/data",
            json=body, headers=self._headers(),
        )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"cvat data upload task={task_id} failed: {resp.status_code} {resp.text[:300]}"
            )
        return resp.json() if resp.content else {}

    def wait_task_ready(
        self, task_id: int, poll_interval: int = 5, max_wait: int = 300,
    ) -> bool:
        """Poll task status until it is not 'pending'/'processing'.

        Returns True when the task is ready (status='annotation' or similar).
        """
        deadline = time.time() + max_wait
        while time.time() < deadline:
            try:
                task = self.get_task(task_id)
                status = task.get("status", "")
                if status not in ("pending", "processing"):
                    return True
            except Exception as e:
                logger.warning("cvat task %s status poll error: %s", task_id, e)
            time.sleep(poll_interval)
        return False

    # -- annotations --------------------------------------------------------

    def get_annotations(self, task_id: int) -> dict[str, Any]:
        """Retrieve annotations for a CVAT task (tracks + shapes)."""
        return self._get(f"/api/tasks/{task_id}/annotations")

    def update_annotations(
        self, task_id: int, annotations: dict[str, Any],
        import_mode: str = "append", action: str = "create",
    ) -> dict[str, Any]:
        """Write annotations to a CVAT task.

        import_mode='append' preserves existing annotations; action='create'
        adds new shapes/tracks without overwriting.
        """
        return self._patch(
            f"/api/tasks/{task_id}/annotations?import_mode={import_mode}&action={action}",
            annotations,
        )

    def clear_annotations(self, task_id: int) -> None:
        """Delete all annotations for a task."""
        self._delete(f"/api/tasks/{task_id}/annotations")

    def replace_auto_annotations(
        self, task_id: int, tracks: list[dict[str, Any]],
        shapes: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Overwrite machine-generated annotations, preserving manual ones.

        Semi-auto pre-annotation is idempotent: re-running it must replace the
        previous machine tracks/shapes instead of appending duplicates. CVAT's
        append PATCH keeps everything, so we read the current annotation set,
        drop objects with source='auto' (our machine output), keep every
        manual object verbatim (with its id so CVAT updates rather than clones
        it), append the fresh machine objects, and PUT the whole set.
        """
        current = self.get_annotations(task_id)
        manual_tracks = [
            t for t in (current.get("tracks") or []) if t.get("source") != "auto"
        ]
        manual_shapes = [
            s for s in (current.get("shapes") or []) if s.get("source") != "auto"
        ]
        payload = {
            "tracks": manual_tracks + tracks,
            "shapes": manual_shapes + shapes,
            "tags": current.get("tags") or [],
        }
        return self._put(
            f"/api/tasks/{task_id}/annotations",
            payload,
            params={"action": "update"},
        )

    # -- export -------------------------------------------------------------

    def export_dataset(
        self, task_id: int, format_name: str = "CVAT for images 1.1",
    ) -> bytes:
        """Export a task's annotations as a dataset archive (zip).

        Blocks until the export is ready and returns the raw zip bytes.
        """
        # Start export
        resp = httpx.get(
            f"{self._base_url()}/api/tasks/{task_id}/dataset",
            headers=self._headers(),
            params={"format": format_name},
            timeout=httpx.Timeout(300),
        )
        if resp.status_code == 202:
            # Export is being prepared; poll until ready.
            time.sleep(3)
            resp = httpx.get(
                f"{self._base_url()}/api/tasks/{task_id}/dataset",
                headers=self._headers(),
                params={"format": format_name},
                timeout=httpx.Timeout(300),
            )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"cvat export task={task_id} failed: {resp.status_code} {resp.text[:300]}"
            )
        return resp.content

    def export_project_dataset(
        self, project_id: int, format_name: str = "CVAT for images 1.1",
    ) -> bytes:
        """Export a project's annotations as a dataset archive (zip)."""
        resp = httpx.get(
            f"{self._base_url()}/api/projects/{project_id}/dataset",
            headers=self._headers(),
            params={"format": format_name},
            timeout=httpx.Timeout(300),
        )
        if resp.status_code == 202:
            time.sleep(3)
            resp = httpx.get(
                f"{self._base_url()}/api/projects/{project_id}/dataset",
                headers=self._headers(),
                params={"format": format_name},
                timeout=httpx.Timeout(300),
            )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"cvat export project={project_id} failed: {resp.status_code} {resp.text[:300]}"
            )
        return resp.content

    # -- ML backend predict (shared with LS) --------------------------------

    def run_ml_backend_predict(
        self, ml_url: str, tasks: list[dict], project_id: int, label_config: str,
    ) -> list[dict[str, Any]]:
        """Call the NF ML backend /predict endpoint directly (no CVAT token).

        Reuses the same ML backend as Label Studio (ADR-010 D2).
        """
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

    # -- frames ---------------------------------------------------------------

    def get_task_frames(self, task_id: int, quality: str = "original") -> list[str]:
        """Return frame URLs for a video task.

        CVAT serves frames via /api/tasks/{id}/data?type=frame&quality=X&number=Y.
        Returns a list of URL strings (on the CVAT host) for each frame.
        """
        task = self.get_task(task_id)
        size = task.get("size", 0)
        stop_frame = task.get("stop_frame", 0)
        frame_count = stop_frame + 1 if stop_frame else size
        return [
            f"{self._public_url()}/api/tasks/{task_id}/data?type=frame&quality={quality}&number={i}"
            for i in range(frame_count)
        ]

    def download_frame(self, task_id: int, number: int, quality: str = "original") -> bytes:
        """Download a single frame as image bytes (authenticated).

        The ML backend cannot fetch CVAT frame URLs itself (no CVAT
        credentials), so the NF backend downloads frames and re-hosts
        them on SeaweedFS for the ML backend to read.

        CVAT's nginx rate-limits aggressive frame fetching with HTTP 429 + a
        Retry-After header (the editor prefetches many frames). Back off and
        retry a bounded number of times instead of surfacing the 429 as a hard
        500; this sync method runs in the FastAPI worker-thread pool, so a
        short sleep does not block the event loop.
        """
        url = f"{self._base_url()}/api/tasks/{task_id}/data"
        params = {"type": "frame", "quality": quality, "number": number}
        max_attempts = 4
        for attempt in range(1, max_attempts + 1):
            resp = self._http.get(url, headers=self._headers(), params=params)
            if resp.status_code == 429 and attempt < max_attempts:
                try:
                    wait_s = min(10.0, float(resp.headers.get("retry-after", "2")))
                except (TypeError, ValueError):
                    wait_s = min(2.0 * attempt, 10.0)
                logger.warning(
                    "cvat frame task=%s number=%s rate-limited (429), "
                    "retry %s/%s in %.1fs",
                    task_id, number, attempt, max_attempts - 1, wait_s,
                )
                time.sleep(wait_s)
                continue
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"cvat frame task={task_id} number={number} failed: "
                    f"{resp.status_code} {resp.text[:200]}"
                )
            return resp.content
        # Unreachable: the loop either returns or raises on the last attempt.
        raise RuntimeError(f"cvat frame task={task_id} number={number} failed after retries")


# -- label spec builder ------------------------------------------------------

def build_cvat_label_spec(schema: dict | None) -> list[dict[str, Any]]:
    """Build a CVAT JSON label spec from a project schema.

    Maps NF schema (labels + type) to CVAT label format:
    - rectanglelabels -> rectangle
    - polygonlabels -> polygon
    - keypointlabels -> points
    - brushlabels -> mask (CVAT doesn't have brush, fallback to polygon)

    Each label gets a name, color, and optional attributes.
    """
    if not schema:
        return _default_cvat_labels()

    raw_labels = schema.get("labels") or []
    if not raw_labels:
        return _default_cvat_labels()

    shape_type = schema.get("type", "rectanglelabels")
    cvat_type = _SHAPE_TYPE_MAP.get(shape_type, "rectangle")

    palette = [
        "#e60049", "#0bb4ff", "#50e991", "#e6d800", "#9b19f5",
        "#ffa300", "#dc0ab4", "#b3d4ff", "#00b464", "#ff6bac",
    ]

    labels = []
    for i, label in enumerate(raw_labels):
        if isinstance(label, str):
            labels.append({
                "name": label,
                "color": palette[i % len(palette)],
                "type": cvat_type,
                "attributes": [],
            })
        elif isinstance(label, dict):
            labels.append({
                "name": label.get("name", f"label_{i}"),
                "color": label.get("color", palette[i % len(palette)]),
                "type": label.get("type", cvat_type),
                "attributes": label.get("attributes", []),
            })
    return labels


# CVAT shape type mapping
_SHAPE_TYPE_MAP: dict[str, str] = {
    "rectanglelabels": "rectangle",
    "polygonlabels": "polygon",
    "keypointlabels": "points",
    "brushlabels": "mask",
}


def _default_cvat_labels() -> list[dict[str, Any]]:
    """Default COCO-style detection labels for CVAT."""
    names = "person,bicycle,car,motorcycle,airplane,bus,train,truck".split(",")
    palette = [
        "#e60049", "#0bb4ff", "#50e991", "#e6d800", "#9b19f5",
        "#ffa300", "#dc0ab4", "#b3d4ff",
    ]
    return [
        {"name": n, "color": palette[i % len(palette)], "type": "rectangle", "attributes": []}
        for i, n in enumerate(names)
    ]


# Module-level singleton.
cvat_client: CvatClient = CvatClient()
