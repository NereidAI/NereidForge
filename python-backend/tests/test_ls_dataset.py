# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Tests for canonical dataset version -> Label Studio task conversion.

Covers app/services/ls_dataset.py:

  1. YOLO split/flat versions produce saved RectangleLabels annotations with
     correct Label Studio percentage coordinates; background images and bad
     lines/class ids are handled.
  2. Polygon labels become PolygonLabels; cross-shape coercion
     (bbox -> rectangle polygon; polygon -> bounding envelope) works.
  3. ImageFolder versions produce image Choices annotations.
  4. Raw versions produce unlabeled tasks.
  5. text,label JSONL versions produce text Choices annotations.
  6. Schema derivation from data.yaml / class_names.json.
"""

import io
import json

import pytest

from app.services.ls_dataset import build_ls_tasks, build_ls_text_tasks, derive_schema

BUCKET = "nereidforge"


def _jpeg(color=(255, 0, 0), size=(100, 50)) -> bytes:
    """Build a tiny real JPEG with known pixel dimensions."""
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG")
    return buf.getvalue()


class FakeS3:
    """Minimal S3 stub over a {filer path: bytes} file map."""

    def __init__(self, files):
        self.files = files

    def s3_uri_to_filer_path(self, uri):
        if not uri.startswith("s3://"):
            return ""
        return "/buckets/" + uri[len("s3://"):].rstrip("/")

    def walk(self, filer, max_depth=None):
        prefix = filer.rstrip("/") + "/"
        return sorted(k for k in self.files if k.startswith(prefix))

    def read_object(self, filer_path):
        return self.files[filer_path]


def _s3(layout: dict[str, bytes]) -> FakeS3:
    """Build FakeS3 from {version-relative key: bytes}."""
    return FakeS3({f"/buckets/{BUCKET}/ds/v1/{k}": v for k, v in layout.items()})


def _ver(fmt: str) -> dict:
    return {"uri": f"s3://{BUCKET}/ds/v1/", "format": fmt}


# -- YOLO bbox --------------------------------------------------------------

def test_yolo_split_tasks_have_saved_rectangle_annotations():
    img = _jpeg(size=(100, 100))
    s3 = _s3({
        "data.yaml": b"names:\n  0: person\nnc: 1\n",
        "train/images/a.jpg": img,
        "train/labels/a.txt": b"0 0.5 0.5 0.2 0.2\n",
    })

    result = build_ls_tasks(s3, _ver("yolo"))

    task = result["tasks"][0]
    ann = task["annotations"][0]
    assert task["data"]["image"] == f"s3://{BUCKET}/ds/v1/train/images/a.jpg"
    region = ann["result"][0]
    assert region["type"] == "rectanglelabels"
    assert region["value"]["rectanglelabels"] == ["person"]
    assert region["original_width"] == 100
    assert region["value"]["x"] == pytest.approx(40.0)
    assert region["value"]["y"] == pytest.approx(40.0)
    assert region["value"]["width"] == pytest.approx(20.0)
    assert region["value"]["height"] == pytest.approx(20.0)
    assert ann["was_cancelled"] is False


def test_yolo_flat_and_background_images():
    s3 = _s3({
        "data.yaml": b"names:\n  0: cat\nnc: 1\n",
        "images/labeled.jpg": _jpeg(size=(200, 100)),
        "images/negative.jpg": _jpeg(color=(0, 255, 0), size=(200, 100)),
        "labels/labeled.txt": b"0 0.25 0.5 0.5 1.0\n",
    })

    result = build_ls_tasks(s3, _ver("yolo"))

    by_img = {t["data"]["image"].rsplit("/", 1)[-1]: t for t in result["tasks"]}
    labeled = by_img["labeled.jpg"]
    region = labeled["annotations"][0]["result"][0]
    # (cx .25 - w/2 .25) = 0; width 50%; height 100%
    assert region["value"]["x"] == pytest.approx(0.0)
    assert region["value"]["width"] == pytest.approx(50.0)
    assert region["value"]["height"] == pytest.approx(100.0)
    assert "annotations" not in by_img["negative.jpg"]


def test_bad_lines_and_unknown_class_ids_are_counted():
    s3 = _s3({
        "data.yaml": b"names:\n  0: cat\nnc: 1\n",
        "images/a.jpg": _jpeg(),
        "labels/a.txt": b"1 0.5 0.5 0.2 0.2\njunk line here\n0 0.5 0.5 0.2 0.2\n",
    })

    result = build_ls_tasks(s3, _ver("yolo"))

    assert "malformed" in result["warning"]
    assert "unknown class" in result["warning"]
    # The one valid line still produces an annotation region.
    regions = result["tasks"][0]["annotations"][0]["result"]
    assert len(regions) == 1


# -- polygons + coercion ----------------------------------------------------

def test_polygon_labels_become_polygonlabels_regions():
    s3 = _s3({
        "data.yaml": b"names:\n  0: thing\nnc: 1\n",
        "images/p.jpg": _jpeg(size=(100, 100)),
        "labels/p.txt": b"0 0.1 0.1 0.9 0.1 0.9 0.9\n",
    })

    result = build_ls_tasks(
        s3, _ver("yolo"), {"type": "polygonlabels", "labels": ["thing"]})

    region = result["tasks"][0]["annotations"][0]["result"][0]
    assert region["type"] == "polygonlabels"
    flat = [v for point in region["value"]["points"] for v in point]
    assert flat == pytest.approx([10, 10, 90, 10, 90, 90])
    assert region["value"]["polygonlabels"] == ["thing"]


def test_bbox_coerced_to_rectangle_polygon():
    s3 = _s3({
        "data.yaml": b"names:\n  0: cat\nnc: 1\n",
        "images/a.jpg": _jpeg(size=(100, 100)),
        "labels/a.txt": b"0 0.5 0.5 0.2 0.2\n",
    })

    result = build_ls_tasks(
        s3, _ver("yolo"), {"type": "polygonlabels", "labels": ["cat"]})

    points = result["tasks"][0]["annotations"][0]["result"][0]["value"]["points"]
    flat = [v for point in points for v in point]
    assert flat == pytest.approx([40, 40, 60, 40, 60, 60, 40, 60])


def test_polygon_coerced_to_bounding_envelope():
    s3 = _s3({
        "data.yaml": b"names:\n  0: cat\nnc: 1\n",
        "images/a.jpg": _jpeg(size=(100, 100)),
        "labels/a.txt": b"0 0.1 0.2 0.8 0.3 0.5 0.9\n",
    })

    result = build_ls_tasks(
        s3, _ver("yolo"), {"type": "rectanglelabels", "labels": ["cat"]})

    value = result["tasks"][0]["annotations"][0]["result"][0]["value"]
    assert value["x"] == pytest.approx(10.0)
    assert value["y"] == pytest.approx(20.0)
    assert value["width"] == pytest.approx(70.0)
    assert value["height"] == pytest.approx(70.0)


# -- imagenet / raw ---------------------------------------------------------

def test_imagenet_tasks_have_image_choices_annotations():
    s3 = _s3({
        "class_names.json": json.dumps(["cat", "dog"]).encode(),
        "train/cat/a.jpg": _jpeg(),
        "images/dog/b.png": _jpeg(color=(0, 0, 255)),
    })

    result = build_ls_tasks(s3, _ver("imagenet"))

    by_class = {}
    for t in result["tasks"]:
        region = t["annotations"][0]["result"][0]
        assert region["type"] == "choices"
        assert region["to_name"] == "image"
        by_class[region["value"]["choices"][0]] = t["data"]["image"]
    assert by_class["cat"].endswith("train/cat/a.jpg")
    assert by_class["dog"].endswith("images/dog/b.png")


def test_raw_version_produces_unlabeled_tasks():
    s3 = _s3({"images/nested/a.jpg": _jpeg(), "images/b.jpg": _jpeg()})

    result = build_ls_tasks(s3, _ver(""))

    assert len(result["tasks"]) == 2
    assert all("annotations" not in t for t in result["tasks"])


def test_text_label_jsonl_produces_text_choices():
    s3 = FakeS3({
        f"/buckets/{BUCKET}/ds/v1/data.jsonl":
            json.dumps({"text": "hello", "label": "greeting"}) .encode() + b"\n"
            + json.dumps({"text": "plain"}).encode(),
    })

    result = build_ls_text_tasks(s3, _ver("llm-sft"))

    labeled, plain = result["tasks"]
    region = labeled["annotations"][0]["result"][0]
    assert region["type"] == "choices" and region["to_name"] == "text"
    assert region["value"]["choices"] == ["greeting"]
    assert "annotations" not in plain


# -- schema derivation ------------------------------------------------------

def test_derive_schema_for_yolo_and_imagenet():
    s3 = _s3({
        "data.yaml": b"names:\n  0: helmet\n  1: head\nnc: 2\n",
        "labels/a.txt": b"0 0.5 0.5 0.2 0.2\n",
    })
    filer = f"/buckets/{BUCKET}/ds/v1"
    keys = ["data.yaml", "labels/a.txt"]

    schema = derive_schema(s3, filer, "yolo", keys)
    assert schema == {"type": "rectanglelabels", "labels": ["helmet", "head"]}

    s3 = _s3({"class_names.json": json.dumps(["cat"]).encode()})
    schema = derive_schema(s3, filer, "imagenet", ["class_names.json"])
    assert schema == {"type": "choices", "labels": ["cat"]}


def test_derive_schema_detects_polygon_control():
    s3 = _s3({
        "data.yaml": b"names:\n  0: mask\nnc: 1\n",
        "labels/a.txt": b"0 0.1 0.1 0.9 0.1 0.9 0.9\n",
    })
    filer = f"/buckets/{BUCKET}/ds/v1"

    schema = derive_schema(s3, filer, "yolo", ["data.yaml", "labels/a.txt"])

    assert schema["type"] == "polygonlabels"
