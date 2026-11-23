# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Convert canonical NF dataset versions into Label Studio import payloads.

Every dataset upload is first normalized into the canonical layouts (see
``builders/README.md``); this module performs the SECOND,
project-scoped conversion: canonical version -> Label Studio tasks.

Inputs supported:

- ``yolo`` (detection/segmentation; flat ``images/+labels/`` or per-split
  ``train|val|test/{images,labels}``): labels become SAVED annotations.
  Five-field bbox lines map to RectangleLabels and polygon lines to
  PolygonLabels; regions are coerced when the project config asks for the
  other shape (polygon -> bounding envelope; bbox -> four-point polygon).
- ``imagenet`` (ImageFolder): the folder-derived class becomes a Choices
  annotation directly on the image.
- raw media (no format): plain unlabeled tasks.
- ``llm-sft`` / text corpora: JSONL passthrough, plus Choices annotations
  for ``text,label`` text-classification corpora.

All region coordinates are Label Studio percentages (0-100). Image pixel
dimensions are read once per image with PIL. Malformed label lines and
unknown class ids are skipped and counted instead of failing the import;
images without labels are kept as background/negative samples.
"""

from __future__ import annotations

import io
import json
import os
import posixpath

from app.services.dataset_zip import IMAGE_EXTS, _extract_classes, _split_token

# Control/object tag names emitted by build_label_config; kept in sync with
# app/clients/labelstudio.py.
CONTROL_NAME = "label"
IMAGE_NAME = "image"
TEXT_NAME = "text"

# Versions with at least this many images are converted/imported by an async
# worker task instead of inside the project-creation request. Overridable via
# the NF_LS_IMPORT_ASYNC_THRESHOLD environment variable.
ASYNC_THRESHOLD = int(os.getenv("NF_LS_IMPORT_ASYNC_THRESHOLD", "500"))


class ConversionReport:
    """Accumulate skipped-label diagnostics while building LS tasks."""

    def __init__(self) -> None:
        self.bad_lines = 0
        self.unknown_classes = 0
        self.warnings: list[str] = []

    def warn(self, text: str) -> None:
        """Record one human-readable warning (deduped)."""
        if text not in self.warnings:
            self.warnings.append(text)

    def text(self) -> str:
        """Render all diagnostics as one '; '-joined string."""
        if self.bad_lines:
            self.warn(f"{self.bad_lines} malformed label line(s) skipped")
        if self.unknown_classes:
            self.warn(f"{self.unknown_classes} label line(s) with unknown class id skipped")
        return "; ".join(self.warnings)


# -- media discovery --------------------------------------------------------

def _is_image(key: str) -> bool:
    """Return True for a supported image object key."""
    return key.lower().endswith(IMAGE_EXTS)


def discover_images(fmt: str, rel_keys: list[str]) -> list[dict]:
    """Locate every image in a version and describe its shape.

    Returns one dict per image: ``{key, split, rel, cls}`` where ``split`` is
    the canonical split token ("" for flat layouts), ``rel`` is the image path
    inside its images directory, and ``cls`` is the ImageFolder class (imagenet
    only). The version format decides whether ``images/<x>/<file>`` is an
    ImageFolder class tree or a plain image tree with subfolders.
    """
    out: list[dict] = []
    for key in rel_keys:
        if not _is_image(key):
            continue
        parts = key.split("/")
        token = _split_token(parts[0])
        if fmt == "imagenet":
            # <split>/<class>/<file...> or images/<class>/<file...>
            if token is not None and len(parts) >= 3:
                out.append({"key": key, "split": token, "rel": "/".join(parts[2:]),
                            "cls": parts[1]})
            elif parts[0] == "images" and len(parts) >= 3:
                out.append({"key": key, "split": "", "rel": "/".join(parts[2:]),
                            "cls": parts[1]})
            continue
        # yolo/coco/raw: images/<rel...> or <split>/images/<rel...>
        if parts[0] == "images" and len(parts) >= 2:
            out.append({"key": key, "split": "", "rel": "/".join(parts[1:]),
                        "cls": ""})
        elif token is not None and parts[1] == "images" and len(parts) >= 3:
            out.append({"key": key, "split": token, "rel": "/".join(parts[2:]),
                        "cls": ""})
    return out


def _label_key(image: dict) -> str:
    """Resolve the YOLO label object key for one discovered image."""
    stem = posixpath.splitext(image["rel"])[0]
    if image["split"]:
        return f"{image['split']}/labels/{stem}.txt"
    return f"labels/{stem}.txt"


# -- YOLO label parsing -----------------------------------------------------

def _parse_yolo_lines(text: bytes, classes: list[str],
                      report: ConversionReport) -> list[dict]:
    """Parse one YOLO label file into normalized (cls, bbox/polygon) records.

    Each returned record is ``{"cls": name, "bbox": [cx,cy,w,h]}`` or
    ``{"cls": name, "points": [[x,y], ...]}`` with normalized 0-1 values.
    """
    records: list[dict] = []
    for raw in text.decode("utf-8", errors="ignore").splitlines():
        fields = raw.split()
        if not fields:
            continue
        bbox = len(fields) == 5
        polygon = len(fields) >= 7 and (len(fields) - 1) % 2 == 0
        if not bbox and not polygon:
            report.bad_lines += 1
            continue
        try:
            cls_id = int(fields[0])
            values = [float(v) for v in fields[1:]]
        except ValueError:
            report.bad_lines += 1
            continue
        if cls_id < 0 or cls_id >= len(classes) or \
                not all(0.0 <= v <= 1.0 for v in values):
            report.unknown_classes += 1
            continue
        name = classes[cls_id]
        if bbox:
            records.append({"cls": name, "bbox": values})
        else:
            points = [[values[i], values[i + 1]]
                      for i in range(0, len(values) - 1, 2)]
            records.append({"cls": name, "points": points})
    return records


def _envelope(points: list[list[float]]) -> list[float]:
    """Return the axis-aligned bbox [cx,cy,w,h] enclosing polygon points."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    w, h = xmax - xmin, ymax - ymin
    return [(xmin + xmax) / 2, (ymin + ymax) / 2, w, h]


def _rect_polygon(bbox: list[float]) -> list[list[float]]:
    """Expand a YOLO bbox into its four corner points."""
    cx, cy, w, h = bbox
    x0, x1, y0, y1 = cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _clamp_pct(v: float) -> float:
    """Clamp one percentage coordinate into the 0-100 range."""
    return min(max(v, 0.0), 100.0)


def _region(record: dict, region_type: str, width: int, height: int) -> dict:
    """Serialize one parsed label record as an LS result region dict."""
    if region_type == "polygonlabels":
        points = record.get("points") or _rect_polygon(record["bbox"])
        value = {
            "points": [[_clamp_pct(x * 100), _clamp_pct(y * 100)]
                       for x, y in points],
            "polygonlabels": [record["cls"]],
            "closed": True,
        }
        kind = "polygonlabels"
    else:
        bbox = record.get("bbox") or _envelope(record["points"])
        cx, cy, w, h = bbox
        value = {
            "x": _clamp_pct((cx - w / 2) * 100),
            "y": _clamp_pct((cy - h / 2) * 100),
            "width": _clamp_pct(w * 100),
            "height": _clamp_pct(h * 100),
            "rotation": 0,
            "rectanglelabels": [record["cls"]],
        }
        kind = "rectanglelabels"
    return {
        "type": kind, "from_name": CONTROL_NAME, "to_name": IMAGE_NAME,
        "original_width": width, "original_height": height,
        "image_rotation": 0, "value": value,
    }


def _saved_annotation(result: list[dict]) -> dict:
    """Wrap result regions into one saved (non-prediction) LS annotation."""
    return {
        "result": result, "was_cancelled": False, "ground_truth": False,
        "lead_time": 0.0,
    }


def _choices_region(value: str, to_name: str) -> dict:
    """Build one Choices result region (image or text classification)."""
    return {
        "type": "choices", "from_name": CONTROL_NAME, "to_name": to_name,
        "value": {"choices": [value]},
    }


def _image_size(data: bytes) -> tuple[int, int]:
    """Read (width, height) from image bytes; (0,0) when undecodable."""
    try:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as im:
            return int(im.width), int(im.height)
    except Exception:
        return 0, 0


# -- schema derivation ------------------------------------------------------

def derive_schema(s3, filer: str, fmt: str, rel_keys: list[str]) -> dict:
    """Derive a project schema (labels + control type) from a version.

    YOLO labels and class names come from data.yaml; when any label file
    carries polygon lines the control is PolygonLabels, otherwise
    RectangleLabels. ImageFolder versions derive a Choices schema from
    class_names.json. Returns {} for raw/unreadable versions.
    """
    if fmt == "imagenet":
        try:
            names = json.loads(s3.read_object(f"{filer}/class_names.json"))
        except Exception:
            names = []
        return {"type": "choices", "labels": [str(n) for n in names]} \
            if names else {}
    if fmt in ("yolo", "coco"):
        try:
            classes = _extract_classes(s3.read_object(f"{filer}/data.yaml"))
        except Exception:
            return {}
        has_polygon = False
        for key in rel_keys:
            parts = key.split("/")
            if not key.endswith(".txt") or "labels" not in parts:
                continue
            try:
                for line in s3.read_object(f"{filer}/{key}").splitlines():
                    if len(line.split()) > 5:
                        has_polygon = True
                        break
            except Exception:
                continue
            if has_polygon:
                break
        return {
            "type": "polygonlabels" if has_polygon else "rectanglelabels",
            "labels": classes,
        }
    return {}


# -- public entry -----------------------------------------------------------

def build_ls_tasks(s3, ver: dict, schema: dict | None = None) -> dict:
    """Build Label Studio import tasks from one dataset version.

    ``ver`` is a version dict with ``uri`` and ``format``; ``schema`` is the
    target project schema (its ``type`` selects the region shape). Returns
    ``{"tasks": [...], "count": N, "warning": str}``.
    """
    report = ConversionReport()
    uri = (ver.get("uri") or "").rstrip("/")
    fmt = ver.get("format") or ""
    filer = s3.s3_uri_to_filer_path(uri)
    rel_keys = [key[len(filer) + 1:] for key in s3.walk(filer)]

    schema = schema if schema is not None else derive_schema(s3, filer, fmt, rel_keys)
    region_type = str(schema.get("type") or "rectanglelabels")
    images = discover_images(fmt, rel_keys)
    tasks: list[dict] = []

    if fmt == "imagenet":
        for image in images:
            tasks.append({
                "data": {"image": f"{uri}/{image['key']}"},
                "annotations": [_saved_annotation(
                    [_choices_region(image["cls"], IMAGE_NAME)])],
            })
        return {"tasks": tasks, "count": len(tasks), "warning": report.text()}

    classes: list[str] = []
    if fmt in ("yolo", "coco"):
        try:
            classes = _extract_classes(s3.read_object(f"{filer}/data.yaml"))
        except Exception:
            report.warn("data.yaml missing or unreadable; labels skipped")
    for image in images:
        data_payload = {"image": f"{uri}/{image['key']}"}
        records: list[dict] = []
        if classes:
            label_path = f"{filer}/{_label_key(image)}"
            try:
                records = _parse_yolo_lines(
                    s3.read_object(label_path), classes, report)
            except Exception:
                records = []
        if not records:
            # Background/negative image: task without an annotation.
            tasks.append({"data": data_payload})
            continue
        img_path = f"{filer}/{image['key']}"
        width, height = _image_size(s3.read_object(img_path))
        if not width or not height:
            report.warn("image dimensions unreadable; labeled regions skipped")
            tasks.append({"data": data_payload})
            continue
        result = [_region(r, region_type, width, height) for r in records]
        tasks.append({
            "data": data_payload,
            "annotations": [_saved_annotation(result)],
        })
    return {"tasks": tasks, "count": len(tasks), "warning": report.text()}


def build_ls_text_tasks(s3, ver: dict) -> dict:
    """Build text-classification LS tasks from an llm-sft/text version.

    Root-level JSONL records of the shape ``{"text": ..., "label": ...}``
    become tasks carrying saved Choices annotations. Other JSONL shapes pass
    through as plain data tasks.
    """
    report = ConversionReport()
    uri = (ver.get("uri") or "").rstrip("/")
    filer = s3.s3_uri_to_filer_path(uri)
    tasks: list[dict] = []
    for key in s3.walk(filer):
        if not key.endswith(".jsonl"):
            continue
        for line in s3.read_object(key).decode("utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                tasks.append({"data": {"text": line}})
                continue
            data = {k: v for k, v in obj.items() if k != "label"}
            label = obj.get("label")
            if label is None or "text" not in data:
                tasks.append({"data": data})
                continue
            result = [_choices_region(str(label), TEXT_NAME)]
            tasks.append({
                "data": data,
                "annotations": [_saved_annotation(result)],
            })
    return {"tasks": tasks, "count": len(tasks), "warning": report.text()}
