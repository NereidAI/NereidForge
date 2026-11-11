# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Normalize uploaded dataset zip archives into the canonical NF layouts.

External corpora arrive in several common export shapes. Import converts them
ONCE into the canonical layouts the trainers consume (see
``app/builders/README.md``):

- Detection / segmentation (format ``yolo``):
  * with splits:  ``train|val|test/{images,labels}/...`` + root ``data.yaml``
  * flat:         ``images/`` + ``labels/`` + root ``data.yaml``
    (``train.py`` applies the deterministic 80/20 split at run time)
  A ``data.yaml``-less export (e.g. the Ultralytics coco128 zip) is also
  recognized from its sibling ``images/`` + ``labels/`` trees; its nested
  subfolders are flattened and generic ``classN`` names are generated.
  Segmentation polygons are kept as YOLO polygon label lines
  (``cls x1 y1 x2 y2 ...``); bbox-only exports become 5-field lines.
- Image classification (format ``imagenet``): ``<split>/<class>/<file>`` or a
  flat ``images/<class>/<file>`` ImageFolder tree, plus ``class_names.json``.
- LLM/VLM (format ``llm-sft``): root-level ``*.jsonl`` plus optional images.
- Raw media (no format): flat ``images/`` / ``videos/``, browsable but not
  trainable until annotated.

``normalize_dataset_zip`` detects the archive flavor and dispatches to the
matching converter. Every converter is a pure in-memory function so the whole
ingestion path is unit-testable without S3 access.
"""

from __future__ import annotations

import io
import json
import posixpath
import re
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import yaml

# Media extensions accepted while scanning an archive.
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v")

# Directory/file names that can never be classification class names.
_SPECIAL_DIRS = {
    "images", "image", "imgs", "img", "labels", "label", "annotations",
    "annotation", "annotations_xml", "coco", "data", "texts", "videos",
    "dataset", "assets",
}


class DatasetZipError(ValueError):
    """Raised when an uploaded archive cannot be used as a dataset."""


# -- shared helpers ---------------------------------------------------------

def _safe_members(zf: ZipFile) -> dict[str, bytes]:
    """Return regular-file members keyed by cleaned POSIX paths.

    Absolute paths and any member containing a parent-directory segment are
    skipped (zip-slip guard); directories are not extracted.
    """
    out: dict[str, bytes] = {}
    for info in zf.infolist():
        if info.is_dir():
            continue
        name = posixpath.normpath(info.filename.replace("\\", "/"))
        parts = name.split("/")
        if name.startswith("/") or ".." in parts:
            continue
        out[name] = zf.read(info)
    return out


def _strip_wrapper(members: dict[str, bytes]) -> str:
    """Return the common wrapping directory prefix ('' when files sit at root).

    A single top folder that is itself a split token (train/valid/test) is not
    a wrapper: it carries split semantics the converters must preserve.
    """
    tops = {name.split("/", 1)[0] for name in members if "/" in name}
    rooted = {name for name in members if "/" not in name}
    if rooted or len(tops) != 1:
        return ""
    top = next(iter(tops))
    if _split_token(top) is not None:
        return ""
    if all(name == top or name.startswith(top + "/") for name in members):
        return top + "/"
    return ""


def _split_token(path: str) -> str | None:
    """Extract a train/val/test token from a path; None when no token appears."""
    lowered = path.lower()
    for seg in re.split(r"[/.\-_]", lowered):
        if "train" in seg:
            return "train"
        if seg in ("val", "valid", "validation", "validate", "dev"):
            return "val"
        if "test" in seg or seg == "eval":
            return "test"
    return None


def _canonical_split(token: str | None) -> str:
    """Map an exporter's split spelling onto the NF canonical directory name."""
    if token is None:
        return "train"
    return token


def _put_unique(
    files: dict[str, bytes],
    directory: str,
    file_name: str,
    payload: bytes,
    hint: str = "",
) -> str:
    """Store one object under directory/name with basename-collision handling.

    A second, different payload for the same target gets a ``<hint>-`` prefix
    so neither source file is silently lost. Returns the stored object key.
    """
    target = file_name if directory in ("", ".") else f"{directory}/{file_name}"
    existing = files.get(target)
    if existing is None or existing == payload:
        files[target] = payload
        return target
    stem, dot, ext = file_name.partition(".")
    alt_name = f"{hint}-{stem}.{ext}" if dot else f"{hint}-{file_name}"
    alt = alt_name if directory in ("", ".") else f"{directory}/{alt_name}"
    files[alt] = payload
    return alt


def _basename_index(members: dict[str, bytes], exts: tuple[str, ...]) -> dict[str, list[str]]:
    """Map lowercase basename -> all member keys carrying that name."""
    idx: dict[str, list[str]] = {}
    for key in members:
        if key.lower().endswith(exts):
            idx.setdefault(posixpath.basename(key).lower(), []).append(key)
    return idx


class _ClassRegistry:
    """Assign stable global class indices in first-seen order."""

    def __init__(self) -> None:
        self.names: list[str] = []
        self.index: dict[str, int] = {}

    def get(self, name: str) -> int:
        name = str(name)
        if name not in self.index:
            self.index[name] = len(self.names)
            self.names.append(name)
        return self.index[name]


def _build_data_yaml(classes: list[str], splits: set[str]) -> bytes:
    """Render the canonical data.yaml; flat and split layouts differ in paths."""
    if splits:
        lines = ["train: /data/train/images"]
        lines.append("val: /data/val/images" if "val" in splits
                     else "val: /data/train/images  # no val split exported")
        if "test" in splits:
            lines.append("test: /data/test/images")
    else:
        lines = ["train: /data/images", "val: /data/images"]
    lines.append("names:")
    for i, name in enumerate(classes):
        lines.append(f"  {i}: {name}")
    lines.append(f"nc: {len(classes)}")
    return ("\n".join(lines) + "\n").encode()


# -- YOLO -------------------------------------------------------------------

def _extract_classes(yaml_bytes: bytes) -> list[str]:
    """Read the ordered class names from a YOLO data.yaml file."""
    parsed = yaml.safe_load(io.BytesIO(yaml_bytes)) or {}
    names = parsed.get("names")
    if isinstance(names, dict):
        classes = [str(names[k]) for k in sorted(names, key=lambda x: int(x))]
    elif isinstance(names, (list, tuple)):
        classes = [str(n) for n in names]
    else:
        nc = parsed.get("nc")
        if isinstance(nc, int) and nc > 0:
            classes = [f"class{i}" for i in range(nc)]
        else:
            raise DatasetZipError("data.yaml declares neither names nor nc")
    return [c for c in classes if c]


def _normalize_yolo(members: dict[str, bytes], root: str, yaml_key: str) -> dict:
    """Normalize a YOLO export, preserving any per-split directories."""
    classes = _extract_classes(members[yaml_key])
    splits: dict[str, dict] = {}
    label_re = re.compile(rf"^{re.escape(root)}(?P<split>[^/]+)/labels/(?P<stem>[^/]+)\.txt$")
    image_re = re.compile(rf"^{re.escape(root)}(?P<split>[^/]+)/images/(?P<stem>.+)$")
    for key in members:
        m = label_re.match(key)
        if m:
            split = splits.setdefault(m.group("split"), {"labels": {}, "images": {}})
            split["labels"][m.group("stem")] = key
            continue
        m = image_re.match(key)
        if not m:
            continue
        stem = m.group("stem")
        if not stem.lower().endswith(IMAGE_EXTS):
            continue
        split = splits.setdefault(m.group("split"), {"labels": {}, "images": {}})
        split["images"][posixpath.splitext(stem)[0]] = (key, stem)

    files: dict[str, bytes] = {}
    canonical_splits: set[str] = set()
    warnings: list[str] = []
    if splits:
        # Exported train/valid/test splits are preserved verbatim.
        for raw_split, split in splits.items():
            token = _split_token(raw_split)
            if token is None:
                # Unknown directory next to data.yaml: keep its files but put
                # them in train so nothing is silently dropped.
                token = "train"
                warnings.append(f"unknown split '{raw_split}' merged into train")
            can = _canonical_split(token)
            canonical_splits.add(can)
            for stem, label_key in split["labels"].items():
                _put_unique(files, f"{can}/labels", f"{stem}.txt",
                            members[label_key], raw_split)
                image = split["images"].pop(stem, None)
                if image:
                    image_key, image_name = image
                    _put_unique(files, f"{can}/images", image_name,
                                members[image_key], raw_split)
            # Images without labels are negative/background samples.
            for image_key, image_name in split["images"].values():
                _put_unique(files, f"{can}/images", image_name,
                            members[image_key], raw_split)
        yaml_splits = canonical_splits
    else:
        # Pre-flat layout: root-level images/ + labels/ next to data.yaml.
        for key in members:
            if not key.startswith(root):
                continue
            rel = key[len(root):]
            if rel.startswith("labels/") and rel.endswith(".txt"):
                _put_unique(files, "labels", posixpath.basename(rel), members[key])
            elif rel.startswith("images/") and rel.lower().endswith(IMAGE_EXTS):
                _put_unique(files, "images", posixpath.basename(rel), members[key])
        yaml_splits = set()

    if not any(k.startswith("labels/") or "/labels/" in k for k in files
               if k.endswith(".txt")):
        raise DatasetZipError("data.yaml found but no YOLO label files are present")
    if "val" not in canonical_splits and canonical_splits:
        warnings.append("no validation split in the archive; val points at train")
    files["data.yaml"] = _build_data_yaml(classes, yaml_splits)
    return {
        "files": files, "format": "yolo", "classes": classes,
        "warning": "; ".join(warnings),
    }


# -- YOLO without data.yaml -------------------------------------------------

def _valid_yolo_label_line(text: str) -> bool:
    """Return True when one line is a well-formed YOLO bbox or polygon line.

    Accepted shapes: ``cls cx cy w h`` (5 fields) or ``cls x1 y1 ...`` with an
    even number of normalized coordinates and at least three points (7+
    fields). The class id is a non-negative integer and every coordinate must
    lie within ``[0.0, 1.0]``.
    """
    fields = text.split()
    bbox_shape = len(fields) == 5
    polygon_shape = len(fields) >= 7 and (len(fields) - 1) % 2 == 0
    if not bbox_shape and not polygon_shape:
        return False
    try:
        cls_id = int(fields[0])
        coords = [float(v) for v in fields[1:]]
    except ValueError:
        return False
    return cls_id >= 0 and all(0.0 <= v <= 1.0 for v in coords)


def _scan_anonymous_yolo(members: dict[str, bytes]) -> tuple[dict, dict, int] | None:
    """Collect flat/split YOLO trees from an archive without data.yaml.

    Returns ``(groups, label_sources, max_cls)`` where each group maps the
    canonical split name (``""`` for the flat shape) to its matched label/image
    stem indexes, ``label_sources`` maps every accepted label stem+group to its
    source member key, and ``max_cls`` is the highest class id seen. Returns
    None when the members do not form an unambiguous YOLO label tree.
    """
    raw_groups: dict[str, dict] = {}
    for key in members:
        parts = key.split("/")
        token = _split_token(parts[0]) if len(parts) >= 3 else None
        is_label = False
        if parts[0] == "labels" and len(parts) >= 2 and key.endswith(".txt"):
            group_key, rel, is_label = "", "/".join(parts[1:]), True
        elif parts[0] == "images" and len(parts) >= 2 \
                and key.lower().endswith(IMAGE_EXTS):
            group_key, rel = "", "/".join(parts[1:])
        elif token is not None and parts[1] == "labels" \
                and len(parts) >= 3 and key.endswith(".txt"):
            group_key, rel, is_label = _canonical_split(token), "/".join(parts[2:]), True
        elif token is not None and parts[1] == "images" \
                and len(parts) >= 3 and key.lower().endswith(IMAGE_EXTS):
            group_key, rel = _canonical_split(token), "/".join(parts[2:])
        else:
            continue
        group = raw_groups.setdefault(
            group_key, {"labels": {}, "images": {}})
        stem = posixpath.splitext(rel)[0]
        if is_label:
            group["labels"][stem] = key
        else:
            group["images"][stem] = (key, rel)

    scanned_label_groups = {k: g for k, g in raw_groups.items() if g["labels"]}
    if not scanned_label_groups:
        return None
    label_sources: dict[tuple[str, str], str] = {}
    max_cls = -1
    matched_any = False
    for group_key, group in scanned_label_groups.items():
        for stem, key in group["labels"].items():
            # Every non-empty label file must parse as YOLO; an unrelated text
            # tree under labels/ means this is not YOLO and detection falls
            # through to the other converters.
            try:
                text = members[key].decode("utf-8")
            except UnicodeDecodeError:
                return None
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                if not _valid_yolo_label_line(line):
                    return None
                max_cls = max(max_cls, int(line.split()[0]))
            label_sources[(group_key, stem)] = key
            if stem in group["images"]:
                matched_any = True
    if max_cls < 0 or not matched_any:
        return None
    return raw_groups, label_sources, max_cls


def _normalize_yolo_without_yaml(members: dict[str, bytes]) -> dict | None:
    """Normalize a YOLO images/labels tree that ships without data.yaml.

    Covered shapes are the flat ``images/`` + ``labels/`` tree (Ultralytics
    coco128 style, where images may sit in an extra ``train2017`` subfolder)
    and per-split ``train|val|test/{images,labels}`` trees. Nested subfolders
    are flattened because the runtime split is non-recursive; basename
    collisions keep a hint prefix so no file is lost. Class names cannot be
    recovered without data.yaml, so generic ``classN`` names are generated.
    """
    scanned = _scan_anonymous_yolo(members)
    if scanned is None:
        return None
    groups, label_sources, max_cls = scanned
    files: dict[str, bytes] = {}
    canonical_splits: set[str] = set()
    for group_key, group in groups.items():
        base_dir = group_key  # "" -> flat labels/ + images/
        labels_dir = "labels" if not base_dir else f"{base_dir}/labels"
        images_dir = "images" if not base_dir else f"{base_dir}/images"
        if base_dir:
            canonical_splits.add(base_dir)
        for stem, label_key in group["labels"].items():
            rel_hint = posixpath.dirname(stem) or group_key or "label"
            _put_unique(files, labels_dir, f"{posixpath.basename(stem)}.txt",
                        members[label_key], rel_hint)
            image = group["images"].pop(stem, None)
            if image:
                image_key, image_name = image
                _put_unique(files, images_dir, posixpath.basename(image_name),
                            members[image_key], rel_hint)
        # Images without label files are kept as background/negative samples.
        for image_key, image_name in group["images"].values():
            rel_hint = posixpath.dirname(posixpath.splitext(image_name)[0]) \
                or group_key or "img"
            _put_unique(files, images_dir, posixpath.basename(image_name),
                        members[image_key], rel_hint)
    classes = [f"class{i}" for i in range(max_cls + 1)]
    warning = (
        "no data.yaml found; generated generic class names "
        f"(class0..class{max_cls}) - rename after import if needed"
    )
    files["data.yaml"] = _build_data_yaml(classes, canonical_splits)
    return {
        "files": files, "format": "yolo", "classes": classes,
        "warning": warning,
    }


# -- CSV --------------------------------------------------------------------

_CSV_IMAGE_ALIASES = {"image", "img", "filename", "file", "file_name", "name", "path"}
_CSV_LABEL_ALIASES = {"label", "class", "class_name", "category", "tag"}
_CSV_XMIN_ALIASES = {"xmin", "x_min", "x1", "left"}
_CSV_YMIN_ALIASES = {"ymin", "y_min", "y1", "top"}
_CSV_XMAX_ALIASES = {"xmax", "x_max", "x2", "right"}
_CSV_YMAX_ALIASES = {"ymax", "y_max", "y2", "bottom"}


def _csv_header_map(fieldnames: list[str]) -> dict[str, str]:
    """Map canonical column roles -> actual CSV header names ('' when absent)."""
    lowered = {str(name).strip().lower(): name for name in fieldnames}

    def find(aliases: set[str]) -> str:
        return next((lowered[a] for a in aliases if a in lowered), "")

    return {
        "image": find(_CSV_IMAGE_ALIASES),
        "label": find(_CSV_LABEL_ALIASES),
        "text": next((lowered[a] for a in ("text", "content", "caption")
                      if a in lowered), ""),
        "xmin": find(_CSV_XMIN_ALIASES),
        "ymin": find(_CSV_YMIN_ALIASES),
        "xmax": find(_CSV_XMAX_ALIASES),
        "ymax": find(_CSV_YMAX_ALIASES),
        "width": find({"width", "w", "img_width"}),
        "height": find({"height", "h", "img_height"}),
    }


def _normalize_csv_rows(
    members: dict[str, bytes], rows: list[dict], cols: dict[str, str],
) -> tuple[dict, list[str]]:
    """Convert recognized CSV rows into a canonical version file map."""

    warnings: list[str] = []
    missing_images = 0
    image_idx = _basename_index(members, IMAGE_EXTS)
    classes = _ClassRegistry()
    files: dict[str, bytes] = {}

    def resolve_image(file_name: str) -> tuple[str | None, bytes | None]:
        # Match the referenced file against an archive image member.
        base = posixpath.basename(file_name).lower()
        hits = image_idx.get(base) or []
        if not hits:
            return None, None
        return hits[0], members[hits[0]]

    bbox_mode = all(cols[k] for k in ("xmin", "ymin", "xmax", "ymax"))
    grouped_labels: dict[str, list[str]] = {}
    for row in rows:
        file_name = str(row.get(cols["image"], "") or "").strip()
        label_name = str(row.get(cols["label"], "") or "").strip()
        if not file_name or not label_name:
            continue
        src_key, payload = resolve_image(file_name)
        if payload is None:
            missing_images += 1
            continue
        cls_id = classes.get(label_name)
        stored_name = posixpath.basename(src_key)
        if not bbox_mode:
            # image,label -> ImageFolder classification layout.
            _put_unique(files, f"images/{label_name}", stored_name,
                        payload, label_name)
            continue
        try:
            xmin = float(row[cols["xmin"]])
            ymin = float(row[cols["ymin"]])
            xmax = float(row[cols["xmax"]])
            ymax = float(row[cols["ymax"]])
        except ValueError:
            continue
        width = height = 0.0
        if cols["width"] and cols["height"]:
            try:
                width, height = float(row[cols["width"]]), float(row[cols["height"]])
            except ValueError:
                width = height = 0.0
        if not width or not height:
            # Read pixel dimensions from the referenced image member.
            try:
                from PIL import Image

                with Image.open(io.BytesIO(payload)) as im:
                    width, height = float(im.width), float(im.height)
            except Exception:
                width = height = 0.0
        if width and height and (xmax > 1.0 or ymax > 1.0):
            cx = (xmin + xmax) / 2 / width
            cy = (ymin + ymax) / 2 / height
            bw = (xmax - xmin) / width
            bh = (ymax - ymin) / height
        elif width and height:
            # Coordinates may be fractions even with dimensions available.
            cx = (xmin + xmax) / 2
            cy = (ymin + ymax) / 2
            bw, bh = xmax - xmin, ymax - ymin
        else:
            # No dimensions: assume already-normalized fractional coordinates.
            cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
            bw, bh = xmax - xmin, ymax - ymin
        line = f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
        stem = posixpath.splitext(stored_name)[0]
        grouped_labels.setdefault(stem, []).append(line)
        _put_unique(files, "images", stored_name, payload, "csv")
    if bbox_mode and grouped_labels:
        for stem, lines in grouped_labels.items():
            files[f"labels/{stem}.txt"] = ("\n".join(lines) + "\n").encode()
        files["data.yaml"] = _build_data_yaml(classes.names, set())
        fmt = "yolo"
    elif not bbox_mode and any(k.startswith("images/") for k in files):
        ordered = sorted(classes.names)
        files["class_names.json"] = json.dumps(ordered, ensure_ascii=False).encode()
        fmt = "imagenet"
    else:
        return None, None
    if missing_images:
        warnings.append(
            f"{missing_images} CSV row(s) referenced images missing from the zip and were skipped")
    return {"files": files, "format": fmt, "classes": classes.names,
            "warning": "; ".join(warnings)}, warnings


def _normalize_csv(members: dict[str, bytes]) -> dict | None:
    """Detect labeled CSV files (image-label / bbox / text-label) and convert.

    Header sniffing selects the shape: ``image,label`` -> ImageFolder,
    ``image,xmin,ymin,xmax,ymax,label`` (pixel coords with dimensions, or
    normalized fractions) -> YOLO, ``text,label`` -> root JSONL. CSVs with
    unrecognized headers are ignored so they keep the raw/text passthrough.
    """
    import csv as _csv

    for key, data in members.items():
        if not key.lower().endswith(".csv"):
            continue
        try:
            text = data.decode("utf-8", errors="ignore")
            reader = _csv.DictReader(io.StringIO(text))
            rows = list(reader)
            cols = _csv_header_map(list(reader.fieldnames or []))
        except Exception:
            continue
        if not rows:
            continue
        if cols["text"] and cols["label"] and not cols["image"]:
            # text,label -> llm-sft JSONL at the version root.
            lines = []
            for row in rows:
                obj = {"text": row[cols["text"]], "label": row[cols["label"]]}
                lines.append(json.dumps(obj, ensure_ascii=False))
            stem = posixpath.splitext(posixpath.basename(key))[0]
            return {
                "files": {f"{stem}.jsonl": ("\n".join(lines) + "\n").encode()},
                "format": "llm-sft", "classes": [], "warning": "",
            }
        if cols["image"] and cols["label"]:
            result, _ = _normalize_csv_rows(members, rows, cols)
            if result is not None:
                return result
    return None


# -- COCO -------------------------------------------------------------------

def _looks_like_coco(data: bytes) -> dict | None:
    """Parse a JSON blob and return it when it has the COCO top-level shape."""
    if not data.lstrip().startswith(b"{"):
        return None
    try:
        obj = json.loads(data.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    if isinstance(obj, dict) and isinstance(obj.get("images"), list) \
            and isinstance(obj.get("annotations"), list) \
            and isinstance(obj.get("categories"), list):
        return obj
    return None


def _find_image_member(
    members: dict[str, bytes],
    basename_idx: dict[str, list[str]],
    json_dir: str,
    file_name: str,
) -> str | None:
    """Resolve a COCO ``file_name`` to its actual zip member key."""
    candidates = [
        posixpath.normpath(posixpath.join(json_dir, file_name)),
        posixpath.basename(file_name),
    ]
    for cand in candidates:
        if cand in members:
            return cand
    hits = basename_idx.get(posixpath.basename(file_name).lower()) or []
    return hits[0] if len(hits) == 1 else None


def _coco_label_line(ann: dict, cls_id: int, width: float, height: float) -> str | None:
    """Serialize one COCO annotation as a YOLO bbox or polygon label line."""
    seg = ann.get("segmentation")
    poly: list[float] | None = None
    if isinstance(seg, list) and seg:
        # Largest polygon wins; RLE segm is a dict and falls back to bbox.
        points = max(seg, key=len)
        if len(points) >= 6:
            poly = [float(v) for v in points]
    if poly is not None and width > 0 and height > 0:
        norm = []
        for i in range(0, len(poly) - 1, 2):
            x = min(max(poly[i] / width, 0.0), 1.0)
            y = min(max(poly[i + 1] / height, 0.0), 1.0)
            norm.extend((x, y))
        if len(norm) >= 6:
            return f"{cls_id} " + " ".join(f"{v:.6f}" for v in norm)
    bb = ann.get("bbox")
    if not bb or len(bb) < 4 or width <= 0 or height <= 0:
        return None
    x, y, w, h = (float(v) for v in bb[:4])
    if w <= 0 or h <= 0:
        return None
    cx = min(max((x + w / 2) / width, 0.0), 1.0)
    cy = min(max((y + h / 2) / height, 0.0), 1.0)
    nw, nh = min(w / width, 1.0), min(h / height, 1.0)
    return f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def _normalize_coco(members: dict[str, bytes], json_keys: list[str]) -> dict:
    """Convert per-split COCO JSONs into canonical YOLO label directories."""
    basename_idx = _basename_index(members, IMAGE_EXTS)
    classes = _ClassRegistry()
    files: dict[str, bytes] = {}
    warnings: list[str] = []

    # Decide split vs flat: flat only when NO json path carries a split token
    # (e.g. one standalone root annotations.json).
    tokens = {_split_token(k) for k in json_keys}
    use_splits = any(t is not None for t in tokens)
    missing_images = 0

    for json_key in sorted(json_keys):
        doc = _looks_like_coco(members[json_key])
        if doc is None:
            continue
        json_dir = posixpath.dirname(json_key)
        token = _split_token(json_key)
        if not use_splits:
            img_dir, lab_dir = "images", "labels"
            hint = "coco"
        else:
            can = _canonical_split(token)
            img_dir, lab_dir = f"{can}/images", f"{can}/labels"
            hint = can
        local_cat = {int(c["id"]): str(c["name"]) for c in doc.get("categories", [])}
        # Resolve and copy images first; map source image id -> label stem.
        id_to_stem: dict[int, str] = {}
        for img in doc.get("images", []):
            file_name = str(img.get("file_name", ""))
            member_key = _find_image_member(members, basename_idx, json_dir, file_name)
            if member_key is None:
                missing_images += 1
                continue
            stored = _put_unique(
                files, img_dir, posixpath.basename(member_key),
                members[member_key], hint,
            )
            id_to_stem[int(img["id"])] = posixpath.splitext(
                posixpath.basename(stored))[0]
        # Group label lines per image so one file holds all its instances.
        labels: dict[str, list[str]] = {}
        for ann in doc.get("annotations", []):
            stem = id_to_stem.get(int(ann.get("image_id", -1)))
            cat_name = local_cat.get(int(ann.get("category_id", -1)))
            if stem is None or not cat_name:
                continue
            img = next((im for im in doc.get("images", [])
                        if int(im["id"]) == int(ann.get("image_id", -1))), {})
            line = _coco_label_line(
                ann, classes.get(cat_name),
                float(img.get("width", 0) or 0), float(img.get("height", 0) or 0),
            )
            if line:
                labels.setdefault(stem, []).append(line)
        for stem, lines in labels.items():
            files[f"{lab_dir}/{stem}.txt"] = ("\n".join(lines) + "\n").encode()

    label_count = sum(1 for k in files if k.endswith(".txt"))
    if not label_count:
        raise DatasetZipError("COCO JSON found but no usable annotations or images")
    splits_present = {k.split("/", 1)[0] for k in files if k.startswith(("train/", "val/", "test/"))}
    if use_splits and "val" not in splits_present:
        warnings.append("no validation split in the archive; val points at train")
    if missing_images:
        warnings.append(f"{missing_images} image(s) referenced by COCO JSON were missing and skipped")
    files["data.yaml"] = _build_data_yaml(classes.names, splits_present)
    return {
        "files": files, "format": "yolo", "classes": classes.names,
        "warning": "; ".join(warnings),
    }


# -- Pascal VOC -------------------------------------------------------------

def _xml_text(node: ET.Element | None, name: str, default: str = "") -> str:
    if node is None:
        return default
    child = node.find(name)
    return child.text.strip() if child is not None and child.text else default


def _voc_objects(root_el: ET.Element, width: float, height: float,
                 classes: _ClassRegistry) -> list[str]:
    """Convert VOC <object> entries (polygon preferred, else bndbox) to lines."""
    lines: list[str] = []
    for obj in root_el.findall("object"):
        name = _xml_text(obj, "name")
        if not name:
            continue
        poly_node = obj.find("polygon")
        pts: list[float] = []
        if poly_node is not None:
            xs = [float(c.text) for c in poly_node.findall("x")
                  if c.text and c.text.strip()]
            ys = [float(c.text) for c in poly_node.findall("y")
                  if c.text and c.text.strip()]
            if len(xs) == len(ys) and len(xs) >= 3 and width > 0 and height > 0:
                pts = []
                for x, y in zip(xs, ys):
                    pts.extend((
                        min(max(x / width, 0.0), 1.0),
                        min(max(y / height, 0.0), 1.0),
                    ))
        if pts:
            cls_id = classes.get(name)
            lines.append(f"{cls_id} " + " ".join(f"{v:.6f}" for v in pts))
            continue
        box = obj.find("bndbox")
        if box is None or width <= 0 or height <= 0:
            continue
        try:
            xmin = float(_xml_text(box, "xmin"))
            ymin = float(_xml_text(box, "ymin"))
            xmax = float(_xml_text(box, "xmax"))
            ymax = float(_xml_text(box, "ymax"))
        except ValueError:
            continue
        if xmax <= xmin or ymax <= ymin:
            continue
        cx, cy = (xmin + xmax) / 2 / width, (ymin + ymax) / 2 / height
        bw, bh = (xmax - xmin) / width, (ymax - ymin) / height
        lines.append(
            f"{classes.get(name)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
        )
    return lines


def _normalize_voc(members: dict[str, bytes], xml_keys: list[str]) -> dict | None:
    """Convert Pascal VOC annotation XMLs into canonical YOLO label dirs.

    Returns None when the XMLs carry no usable VOC objects (so the dispatcher
    can fall through to classification/raw handling).
    """
    basename_idx = _basename_index(members, IMAGE_EXTS)
    classes = _ClassRegistry()
    files: dict[str, bytes] = {}
    used_images: set[str] = set()
    warnings: list[str] = []
    use_splits = any(_split_token(k) is not None for k in xml_keys)
    produced = 0

    for xml_key in sorted(xml_keys):
        try:
            root_el = ET.fromstring(members[xml_key])
        except ET.ParseError:
            continue
        xml_dir = posixpath.dirname(xml_key)
        base_dir = posixpath.dirname(xml_dir.rstrip("/"))
        file_name = _xml_text(root_el, "filename")
        size = root_el.find("size")
        try:
            width = float(_xml_text(size, "width", "0"))
            height = float(_xml_text(size, "height", "0"))
        except ValueError:
            width = height = 0
        lines = _voc_objects(root_el, width, height, classes)
        if not lines or not file_name:
            continue
        produced += 1
        token = _split_token(xml_key)
        if use_splits:
            can = _canonical_split(token)
            img_dir, lab_dir, hint = f"{can}/images", f"{can}/labels", can
        else:
            img_dir, lab_dir, hint = "images", "labels", "voc"
        # Images live in ../images relative to annotations/, the split folder,
        # or the same folder; a unique basename match is the last resort.
        image_key = None
        for cand in (
            posixpath.normpath(posixpath.join(base_dir, "images", file_name)),
            posixpath.normpath(posixpath.join(xml_dir, file_name)),
        ):
            if cand in members:
                image_key = cand
                break
        if image_key is None:
            hits = basename_idx.get(file_name.lower()) or []
            image_key = hits[0] if len(hits) == 1 else None
        stem = posixpath.splitext(file_name)[0]
        _put_unique(files, lab_dir, f"{stem}.txt",
                    ("\n".join(lines) + "\n").encode(), hint)
        if image_key and image_key not in used_images:
            used_images.add(image_key)
            _put_unique(files, img_dir, posixpath.basename(image_key),
                        members[image_key], hint)

    if not produced:
        return None
    splits_present = {k.split("/", 1)[0] for k in files if k.startswith(("train/", "val/", "test/"))}
    if use_splits and "val" not in splits_present:
        warnings.append("no validation split in the archive; val points at train")
    files["data.yaml"] = _build_data_yaml(classes.names, splits_present)
    return {
        "files": files, "format": "yolo", "classes": classes.names,
        "warning": "; ".join(warnings),
    }


# -- ImageFolder / classification ------------------------------------------

def _class_dir_members(members: dict[str, bytes]) -> dict[str, list[str]]:
    """Map '<class>/<image>' canonical relative keys to source member keys.

    Recognizes Roboflow classification exports (``<split>/<class>/<file>``),
    a pre-flat ``images/<class>/<file>`` ImageFolder, and root-level
    ``<class>/<file>`` folders. The root-level shape is ambiguous with a
    single wrapping album folder, so it only counts as classification when
    at least two distinct class folders are present.
    """
    matched: dict[str, list[str]] = {"split": [], "flat": [], "root": []}
    root_classes: set[str] = set()
    for key in members:
        if not key.lower().endswith(IMAGE_EXTS):
            continue
        parts = key.split("/")
        if len(parts) >= 3 and _split_token(parts[0]) is not None:
            cls = parts[1]
            if cls and cls.lower() not in _SPECIAL_DIRS:
                matched["split"].append(key)
        elif len(parts) >= 3 and parts[0] == "images":
            cls = parts[1]
            if cls and cls.lower() not in _SPECIAL_DIRS:
                matched["flat"].append(key)
        elif len(parts) == 2:
            cls = parts[0]
            if cls and cls.lower() not in _SPECIAL_DIRS:
                root_classes.add(cls)
                matched["root"].append(key)
    if len(root_classes) < 2:
        # One bare folder is treated as a wrapping album, not a one-class
        # ImageFolder dataset; fall through to raw-media handling.
        matched["root"] = []
    else:
        matched["flat"].extend(matched.pop("root"))
    matched.pop("root", None)
    return matched


def _normalize_imagenet(members: dict[str, bytes]) -> dict | None:
    """Preserve an ImageFolder classification tree (optionally per split)."""
    found = _class_dir_members(members)
    if not found["split"] and not found["flat"]:
        return None
    files: dict[str, bytes] = {}
    class_names: set[str] = set()

    def store(parts: list[str], key: str) -> None:
        token = _split_token(parts[0])
        if token is not None:
            can = _canonical_split(token)
            cls, file_name = parts[1], "/".join(parts[2:])
        else:
            can, cls, file_name = "train", parts[1], "/".join(parts[2:])
        class_names.add(cls)
        _put_unique(files, f"{can}/{cls}", posixpath.basename(file_name),
                    members[key], can)

    if found["split"]:
        # Split layout is authoritative; fold any flat ImageFolder dirs into
        # train so the version has exactly one canonical shape.
        for key in found["split"]:
            store(key.split("/"), key)
        for key in found["flat"]:
            parts = key.split("/")
            store(["train"] + parts[1:] if parts[0] == "images" else ["train"] + parts, key)
    else:
        for key in found["flat"]:
            parts = key.split("/")
            rel = parts[1:] if parts[0] == "images" else parts
            cls = rel[0]
            class_names.add(cls)
            _put_unique(files, f"images/{cls}", posixpath.basename("/".join(rel[1:])),
                        members[key], cls)
    ordered = sorted(class_names)
    files["class_names.json"] = json.dumps(ordered, ensure_ascii=False).encode()
    return {
        "files": files, "format": "imagenet", "classes": ordered,
        "warning": "",
    }


# -- raw / dispatcher -------------------------------------------------------

def _normalize_raw(members: dict[str, bytes]) -> dict:
    """Flatten raw media/text files; the result is browsable but not trainable."""
    files: dict[str, bytes] = {}
    has_jsonl = False
    for key, payload in members.items():
        lower = key.lower()
        if lower.endswith(IMAGE_EXTS):
            _put_unique(files, "images", posixpath.basename(key), payload,
                        posixpath.basename(posixpath.dirname(key)) or "img")
        elif lower.endswith(VIDEO_EXTS):
            _put_unique(files, "videos", posixpath.basename(key), payload,
                        posixpath.basename(posixpath.dirname(key)) or "vid")
        elif lower.endswith(".jsonl"):
            # Sharegpt/VLM corpus: keep the file at the version root.
            _put_unique(files, ".", posixpath.basename(key), payload,
                        posixpath.basename(posixpath.dirname(key)) or "jsonl")
            has_jsonl = True
        elif lower.endswith((".txt", ".json", ".csv", ".md")):
            _put_unique(files, "texts", posixpath.basename(key), payload,
                        posixpath.basename(posixpath.dirname(key)) or "doc")
    if has_jsonl:
        # JSONL corpora are llm-sft; content detection finalizes the format.
        return {"files": files, "format": "llm-sft", "classes": [], "warning": ""}
    if not any(k.startswith(("images/", "videos/")) for k in files):
        raise DatasetZipError(
            "unsupported zip: expected YOLO (data.yaml), COCO (annotations JSON), "
            "Pascal VOC (XML), ImageFolder class folders, or raw media"
        )
    return {
        "files": files, "format": "", "classes": [],
        "warning": "no annotations found; imported as raw media (not trainable yet)",
    }


def normalize_dataset_zip(data: bytes) -> dict:
    """Detect an archive's flavor and normalize it into a canonical NF layout.

    Returns ``{"files": {object key: bytes}, "format", "classes", "warning"}``.
    Detection order: YOLO (data.yaml) -> COCO (images/annotations JSON) ->
    Pascal VOC (XML with objects) -> labeled CSV -> YOLO label trees without
    data.yaml -> ImageFolder classification -> raw/JSONL.
    """
    try:
        zf = ZipFile(io.BytesIO(data))
    except Exception as e:  # noqa: BLE001
        raise DatasetZipError(f"not a valid zip archive: {e}") from e
    members = _safe_members(zf)
    if not members:
        raise DatasetZipError("the zip archive is empty")
    prefix = _strip_wrapper(members)
    if prefix:
        members = {k[len(prefix):]: v for k, v in members.items()}

    yaml_keys = sorted(
        (k for k in members if k == "data.yaml" or k.endswith("/data.yaml")),
        key=lambda k: k.count("/"),
    )
    if yaml_keys:
        return _normalize_yolo(members, "", yaml_keys[0])

    coco_keys = [k for k, v in members.items()
                 if k.lower().endswith(".json") and _looks_like_coco(v) is not None]
    if coco_keys:
        return _normalize_coco(members, coco_keys)

    xml_keys = [k for k in members if k.lower().endswith(".xml")]
    if xml_keys:
        voc = _normalize_voc(members, xml_keys)
        if voc is not None:
            return voc

    csv_dataset = _normalize_csv(members)
    if csv_dataset is not None:
        return csv_dataset

    anonymous_yolo = _normalize_yolo_without_yaml(members)
    if anonymous_yolo is not None:
        return anonymous_yolo

    imagenet = _normalize_imagenet(members)
    if imagenet is not None:
        return imagenet

    return _normalize_raw(members)


# Backwards-compatible alias retained for callers/tests written before the
# multi-format dispatcher existed.
def normalize_yolo_zip(data: bytes) -> dict:
    """Normalize a zip known to contain a YOLO export."""
    return normalize_dataset_zip(data)
