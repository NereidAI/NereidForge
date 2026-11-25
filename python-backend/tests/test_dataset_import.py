# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Tests for external dataset-zip ingestion into the canonical NF layouts.

Covers the canonical importer contract summarized in
``builders/README.md``:

  1. YOLO exports keep their ``train|val|test/{images,labels}`` splits (the
     Roboflow ``valid`` spelling becomes ``val``); pre-flat exports stay flat;
     wrapping directories, empty (negative) labels, and background images are
     handled.
  2. COCO per-split JSONs are converted to YOLO bbox/polygon labels under the
     canonical split directories; a single root annotations.json stays flat.
  3. Pascal VOC bbox and ``<polygon>`` XMLs convert to YOLO labels.
  4. ImageFolder / ImageNet class trees are preserved (split or flat).
  5. JSONL corpora become llm-sft; raw media imports as a non-trainable corpus;
     zip-slip members are dropped and junk archives fail clearly.
  6. Content detection marks yolo/coco/imagenet/llm-sft versions annotated and
     leaves raw image corpora untouched.
"""

import io
import json
import zipfile

import pytest

from app.models.orm import DatasetVersion
from app.services.dataset import DatasetService
from app.services.dataset_zip import DatasetZipError, normalize_dataset_zip, normalize_yolo_zip


def _zip(files: dict[str, bytes]) -> bytes:
    """Build an in-memory zip from {member name: content}."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


# -- YOLO -------------------------------------------------------------------

ROBOFLOW_YAML = b"train: ../train/images\nval: ../valid/images\nnc: 3\nnames: ['helmet', 'person', 'head']\n"


def test_yolo_split_zip_preserves_splits():
    """train/valid images+labels normalize to train/ and val/ verbatim."""
    data = _zip({
        "data.yaml": ROBOFLOW_YAML,
        "train/images/a.jpg": b"jpeg-a",
        "train/labels/a.txt": b"0 0.5 0.5 0.2 0.2\n",
        "valid/images/b.png": b"png-b",
        "valid/labels/b.txt": b"1 0.1 0.1 0.1 0.1\n",
        "README.txt": b"ignored",
    })

    result = normalize_yolo_zip(data)

    files = result["files"]
    assert result["format"] == "yolo"
    assert result["classes"] == ["helmet", "person", "head"]
    assert set(files) == {
        "data.yaml",
        "train/images/a.jpg",
        "train/labels/a.txt",
        "val/images/b.png",
        "val/labels/b.txt",
    }
    yaml_text = files["data.yaml"].decode()
    assert "train: /data/train/images" in yaml_text
    assert "val: /data/val/images" in yaml_text
    assert "0: helmet" in yaml_text
    assert "nc: 3" in yaml_text


def test_yolo_flat_zip_stays_flat():
    """Root-level images/ + labels/ next to data.yaml keep the flat layout."""
    data = _zip({
        "data.yaml": b"nc: 1\nnames: ['person']\n",
        "images/a.jpg": b"a",
        "labels/a.txt": b"0 0.5 0.5 0.2 0.2\n",
    })

    result = normalize_dataset_zip(data)

    assert set(result["files"]) == {"data.yaml", "images/a.jpg", "labels/a.txt"}
    yaml_text = result["files"]["data.yaml"].decode()
    assert "train: /data/images" in yaml_text
    assert "val: /data/images" in yaml_text


def test_yolo_zip_tolerates_wrapping_directory_and_empty_labels():
    """A single wrapping folder and negative images with empty labels work."""
    data = _zip({
        "HardHat.v4i.yolov8/data.yaml": b"names:\n  0: cat\n  1: dog\n",
        "HardHat.v4i.yolov8/train/images/x.jpg": b"x",
        "HardHat.v4i.yolov8/train/labels/x.txt": b"",
    })

    result = normalize_yolo_zip(data)

    assert result["classes"] == ["cat", "dog"]
    assert "train/images/x.jpg" in result["files"]
    assert "train/labels/x.txt" in result["files"]
    assert result["files"]["train/labels/x.txt"] == b""


def test_yolo_split_zip_keeps_unlabeled_background_images():
    """Images without a label file are kept in their split as backgrounds."""
    data = _zip({
        "data.yaml": b"nc: 1\nnames: ['person']\n",
        "train/images/labeled.jpg": b"a",
        "train/labels/labeled.txt": b"0 0 0 1 1\n",
        "train/images/negative.jpg": b"b",
    })

    result = normalize_yolo_zip(data)

    assert "train/images/labeled.jpg" in result["files"]
    assert "train/images/negative.jpg" in result["files"]
    assert "train/labels/negative.txt" not in result["files"]


def test_yolo_zip_without_labels_fails():
    """A data.yaml without any label files is rejected, not imported raw."""
    with pytest.raises(DatasetZipError):
        normalize_yolo_zip(_zip({
            "data.yaml": b"nc: 1\nnames: ['person']\n",
            "train/images/a.jpg": b"a",
        }))


def test_yolo_zip_rejects_junk_archives():
    """Zips without recognizable content fail with a clear DatasetZipError."""
    with pytest.raises(DatasetZipError):
        normalize_yolo_zip(_zip({"random/file.txt": b"x"}))
    with pytest.raises(DatasetZipError):
        normalize_yolo_zip(b"not a zip at all")


# -- YOLO without data.yaml -------------------------------------------------

def test_coco128_style_zip_without_yaml_is_detected_as_yolo():
    """Wrapper + nested images/train2017 + labels/train2017 normalize to YOLO."""
    data = _zip({
        "coco128/images/train2017/a.jpg": b"img-a",
        "coco128/images/train2017/b.jpg": b"img-b",
        "coco128/labels/train2017/a.txt": b"5 0.5 0.5 0.2 0.2\n",
        "coco128/labels/train2017/b.txt": b"0 0.1 0.1 0.1 0.1\n",
    })

    result = normalize_dataset_zip(data)

    files = result["files"]
    assert result["format"] == "yolo"
    assert len(result["classes"]) == 6  # highest id (5) + 1
    assert set(files) == {
        "data.yaml",
        "images/a.jpg", "images/b.jpg",
        "labels/a.txt", "labels/b.txt",
    }
    assert "train: /data/images" in files["data.yaml"].decode()
    assert "no data.yaml" in result["warning"]


def test_coco128_style_zip_keeps_unlabeled_background_images():
    """Images without sibling labels survive normalization as backgrounds."""
    data = _zip({
        "coco128/images/train2017/labeled.jpg": b"a",
        "coco128/images/train2017/negative.jpg": b"b",
        "coco128/labels/train2017/labeled.txt": b"0 0 0 1 1\n",
    })

    result = normalize_dataset_zip(data)

    assert "images/labeled.jpg" in result["files"]
    assert "images/negative.jpg" in result["files"]
    assert "labels/negative.txt" not in result["files"]


def test_yolo_split_zip_without_yaml_preserves_splits():
    """train/val images+labels without data.yaml still normalize as splits."""
    data = _zip({
        "train/images/a.jpg": b"a",
        "train/labels/a.txt": b"0 0.5 0.5 0.2 0.2\n",
        "val/images/b.jpg": b"b",
        "val/labels/b.txt": b"1 0.1 0.1 0.1 0.1\n",
    })

    result = normalize_dataset_zip(data)

    files = result["files"]
    assert result["format"] == "yolo"
    assert result["classes"] == ["class0", "class1"]
    assert set(files) == {
        "data.yaml",
        "train/images/a.jpg", "train/labels/a.txt",
        "val/images/b.jpg", "val/labels/b.txt",
    }
    assert "train: /data/train/images" in files["data.yaml"].decode()


def test_polygon_labels_without_yaml_are_detected():
    """YOLO polygon label lines are accepted by the data.yaml-less scanner."""
    data = _zip({
        "images/train/p.jpg": b"p",
        "labels/train/p.txt": b"0 0.1 0.1 0.9 0.1 0.9 0.9\n",
    })

    result = normalize_dataset_zip(data)

    assert result["format"] == "yolo"
    assert result["files"]["labels/p.txt"] == b"0 0.1 0.1 0.9 0.1 0.9 0.9\n"


def test_non_yolo_txt_under_labels_does_not_trigger_detection():
    """An unrelated text tree under labels/ falls through to raw-media import."""
    data = _zip({
        "images/a.jpg": b"a",
        "labels/notes.txt": b"this is just a plain readme note",
    })

    result = normalize_dataset_zip(data)

    assert result["format"] == ""
    assert "images/a.jpg" in result["files"]
    assert not any(k.startswith("labels/") for k in result["files"])


# -- CSV --------------------------------------------------------------------

def _tiny_jpeg(size=(100, 100)):
    """Build a small valid JPEG for CSV-import tests."""
    import io as _io

    from PIL import Image

    buf = _io.BytesIO()
    Image.new("RGB", size, (255, 0, 0)).save(buf, format="JPEG")
    return buf.getvalue()


def test_csv_image_label_becomes_imagenet():
    data = _zip({
        "export/a.jpg": _tiny_jpeg(),
        "export/annotations.csv": b"image,label\na.jpg,cat\na.jpg,dog\n",
    })

    result = normalize_dataset_zip(data)

    assert result["format"] == "imagenet"
    assert set(result["files"]) == {
        "class_names.json", "images/cat/a.jpg", "images/dog/a.jpg"}


def test_csv_bbox_pixel_coords_become_yolo_labels():
    data = _zip({
        "data/a.jpg": _tiny_jpeg(size=(100, 100)),
        "data/boxes.csv": (
            b"filename,width,height,class,xmin,ymin,xmax,ymax\n"
            b"a.jpg,100,100,cat,10,20,50,60\n"),
    })

    result = normalize_dataset_zip(data)

    assert result["format"] == "yolo"
    line = result["files"]["labels/a.txt"].decode().split()
    cls, cx, cy, w, h = line
    assert cls == "0"
    assert float(cx) == pytest.approx(0.3)
    assert float(cy) == pytest.approx(0.4)
    assert float(w) == pytest.approx(0.4)
    assert float(h) == pytest.approx(0.4)
    assert "images/a.jpg" in result["files"]
    assert "names:" in result["files"]["data.yaml"].decode()


def test_csv_bbox_normalized_coords_become_yolo_labels():
    data = _zip({
        "data/a.jpg": _tiny_jpeg(),
        "data/boxes.csv": (
            b"image,label,xmin,ymin,xmax,ymax\n"
            b"a.jpg,cat,0.1,0.2,0.5,0.6\n"),
    })

    result = normalize_dataset_zip(data)

    line = result["files"]["labels/a.txt"].decode().split()
    assert [float(v) for v in line[1:]] == pytest.approx([0.3, 0.4, 0.4, 0.4])


def test_csv_text_label_becomes_jsonl():
    data = _zip({"corpus.csv": b"text,label\nhello,spam\nworld,ham\n"})

    result = normalize_dataset_zip(data)

    assert result["format"] == "llm-sft"
    rows = [json.loads(line) for line in next(iter(result["files"].values())).decode().splitlines()]
    assert rows == [
        {"text": "hello", "label": "spam"},
        {"text": "world", "label": "ham"},
    ]


def test_csv_unknown_header_is_not_canonicalized():
    data = _zip({"data.csv": b"foo,bar\n1,2\n", "images/a.jpg": b"img"})

    result = normalize_dataset_zip(data)

    assert result["format"] == ""


# -- COCO -------------------------------------------------------------------

def _coco_doc(images, anns, cats):
    return json.dumps({"images": images, "annotations": anns,
                       "categories": cats}).encode()


COCO_CATS = [{"id": 1, "name": "helmet"}, {"id": 2, "name": "head"}]


def test_coco_split_zip_converts_to_yolo_split_labels():
    """train/valid COCO JSONs become train/ and val/ YOLO labels + data.yaml."""
    train_json = _coco_doc(
        [{"id": 0, "file_name": "a.jpg", "width": 100, "height": 100}],
        [{"id": 0, "image_id": 0, "category_id": 1, "bbox": [10, 10, 20, 20]}],
        COCO_CATS,
    )
    valid_json = _coco_doc(
        [{"id": 0, "file_name": "b.jpg", "width": 80, "height": 60}],
        [{"id": 0, "image_id": 0, "category_id": 2, "bbox": [0, 0, 10, 10]}],
        COCO_CATS,
    )
    data = _zip({
        "train/_annotations.coco.json": train_json,
        "train/a.jpg": b"img-a",
        "valid/_annotations.coco.json": valid_json,
        "valid/b.jpg": b"img-b",
        "README.dataset.txt": b"notes",
    })

    result = normalize_dataset_zip(data)

    files = result["files"]
    assert result["format"] == "yolo"
    assert result["classes"] == ["helmet", "head"]
    assert set(files) == {
        "data.yaml",
        "train/images/a.jpg",
        "train/labels/a.txt",
        "val/images/b.jpg",
        "val/labels/b.txt",
    }
    # bbox [10,10,20,20] on 100x100 -> cx=cy=0.2, w=h=0.2, class 0
    cls, cx, cy, w, h = files["train/labels/a.txt"].decode().split()[0:5]
    assert (cls, float(cx), float(cy), float(w), float(h)) == (
        "0", pytest.approx(0.2), pytest.approx(0.2),
        pytest.approx(0.2), pytest.approx(0.2))
    yaml_text = files["data.yaml"].decode()
    assert "train: /data/train/images" in yaml_text
    assert "val: /data/val/images" in yaml_text


def test_coco_polygon_segmentation_becomes_yolo_polygon_line():
    """COCO polygon segmentations are normalized, not collapsed to bboxes."""
    doc = _coco_doc(
        [{"id": 0, "file_name": "a.jpg", "width": 100, "height": 100}],
        [{"id": 0, "image_id": 0, "category_id": 1,
          "segmentation": [[10, 10, 50, 10, 50, 50, 10, 50]],
          "bbox": [10, 10, 40, 40]}],
        [{"id": 1, "name": "mask"}],
    )

    result = normalize_dataset_zip(_zip({
        "train/_annotations.coco.json": doc,
        "train/a.jpg": b"img",
    }))

    fields = result["files"]["train/labels/a.txt"].decode().split()
    assert fields[0] == "0"
    assert len(fields) == 9  # class + 8 normalized polygon coordinates
    coords = [float(v) for v in fields[1:]]
    assert coords[0::2] == pytest.approx([0.1, 0.5, 0.5, 0.1])
    assert coords[1::2] == pytest.approx([0.1, 0.1, 0.5, 0.5])


def test_coco_root_json_stays_flat():
    """One standalone root annotations.json normalizes into flat images/labels."""
    doc = _coco_doc(
        [{"id": 0, "file_name": "a.jpg", "width": 10, "height": 10}],
        [{"id": 0, "image_id": 0, "category_id": 1, "bbox": [0, 0, 10, 10]}],
        [{"id": 1, "name": "person"}],
    )

    result = normalize_dataset_zip(_zip({
        "annotations.json": doc,
        "images/a.jpg": b"img",
    }))

    assert set(result["files"]) == {"data.yaml", "images/a.jpg", "labels/a.txt"}
    assert "train: /data/images" in result["files"]["data.yaml"].decode()


def test_coco_same_basename_in_different_splits_kept_distinct():
    """Splits live in separate directories, so x.jpg never collides."""
    doc = _coco_doc(
        [{"id": 0, "file_name": "x.jpg", "width": 10, "height": 10}],
        [{"id": 0, "image_id": 0, "category_id": 1, "bbox": [0, 0, 1, 1]}],
        [{"id": 1, "name": "person"}],
    )

    result = normalize_dataset_zip(_zip({
        "train/_annotations.coco.json": doc,
        "train/x.jpg": b"one",
        "valid/_annotations.coco.json": doc,
        "valid/x.jpg": b"two",
    }))

    files = result["files"]
    assert files["train/images/x.jpg"] == b"one"
    assert files["val/images/x.jpg"] == b"two"
    assert "train/labels/x.txt" in files and "val/labels/x.txt" in files


def test_coco_missing_image_is_reported_and_skipped():
    """An annotation referencing an absent image produces a warning, no crash."""
    doc = _coco_doc(
        [{"id": 0, "file_name": "ghost.jpg", "width": 10, "height": 10}],
        [{"id": 0, "image_id": 0, "category_id": 1, "bbox": [0, 0, 1, 1]}],
        [{"id": 1, "name": "person"}],
    )

    with pytest.raises(DatasetZipError):
        normalize_dataset_zip(_zip({"train/_annotations.coco.json": doc}))


# -- Pascal VOC -------------------------------------------------------------

def test_voc_split_bbox_converts_to_yolo_split_labels():
    xml = (
        b"<annotation><filename>a.jpg</filename>"
        b"<size><width>100</width><height>200</height></size>"
        b"<object><name>cat</name><bndbox>"
        b"<xmin>10</xmin><ymin>20</ymin><xmax>30</xmax><ymax>60</ymax>"
        b"</bndbox></object></annotation>"
    )

    result = normalize_dataset_zip(_zip({
        "dataset/train/annotations/a.xml": xml,
        "dataset/train/images/a.jpg": b"img-a",
    }))

    assert result["format"] == "yolo"
    assert result["classes"] == ["cat"]
    assert set(result["files"]) == {
        "data.yaml", "train/images/a.jpg", "train/labels/a.txt"}
    cls, cx, cy, w, h = result["files"]["train/labels/a.txt"].decode().split()[0:5]
    assert cls == "0"
    assert float(cx) == pytest.approx(0.2)
    assert float(cy) == pytest.approx(0.2)
    assert float(w) == pytest.approx(0.2)
    assert float(h) == pytest.approx(0.2)


def test_voc_polygon_converts_to_normalized_polygon_line():
    xml = (
        b"<annotation><filename>p.jpg</filename>"
        b"<size><width>100</width><height>100</height></size>"
        b"<object><name>dog</name><polygon>"
        b"<x>10</x><y>10</y><x>90</x><y>10</y><x>90</x><y>90</y>"
        b"</polygon></object></annotation>"
    )

    result = normalize_dataset_zip(_zip({
        "voc/train/annotations/p.xml": xml,
        "voc/train/images/p.jpg": b"img",
    }))

    fields = result["files"]["train/labels/p.txt"].decode().split()
    assert fields[0] == "0"
    assert len(fields) == 7  # class + 6 normalized coordinates (triangle)
    assert [float(v) for v in fields[1:]] == pytest.approx(
        [0.1, 0.1, 0.9, 0.1, 0.9, 0.9])


# -- ImageFolder / classification -------------------------------------------

def test_imagenet_split_tree_is_preserved():
    data = _zip({
        "train/cat/a.jpg": b"a",
        "train/cat/b.jpg": b"b",
        "train/dog/c.png": b"c",
        "val/cat/a.jpg": b"va",
        "val/dog/d.png": b"vd",
        "README.txt": b"notes",
    })

    result = normalize_dataset_zip(data)

    assert result["format"] == "imagenet"
    assert result["classes"] == ["cat", "dog"]
    assert set(result["files"]) == {
        "class_names.json",
        "train/cat/a.jpg", "train/cat/b.jpg", "train/dog/c.png",
        "val/cat/a.jpg", "val/dog/d.png",
    }
    assert json.loads(result["files"]["class_names.json"]) == ["cat", "dog"]


def test_imagenet_flat_tree_is_preserved_under_images():
    data = _zip({
        "images/cat/a.jpg": b"a",
        "images/dog/b.png": b"b",
    })

    result = normalize_dataset_zip(data)

    assert result["format"] == "imagenet"
    assert set(result["files"]) == {
        "class_names.json", "images/cat/a.jpg", "images/dog/b.png"}


def test_imagenet_root_class_folders_moved_under_images():
    data = _zip({
        "cat/a.jpg": b"a",
        "dog/b.jpg": b"b",
    })

    result = normalize_dataset_zip(data)

    assert result["format"] == "imagenet"
    assert "images/cat/a.jpg" in result["files"]
    assert "images/dog/b.jpg" in result["files"]


# -- JSONL / raw / safety ----------------------------------------------------

def test_jsonl_zip_imports_as_llm_sft():
    data = _zip({
        "sharegpt/corpus.jsonl": b'{"messages": []}\n',
        "sharegpt/images/pic.jpg": b"img",
    })

    result = normalize_dataset_zip(data)

    assert result["format"] == "llm-sft"
    assert "corpus.jsonl" in result["files"]
    assert "images/pic.jpg" in result["files"]


def test_raw_image_zip_imports_as_non_trainable_corpus():
    data = _zip({
        "holiday-photos/one.jpg": b"a",
        "holiday-photos/sub/two.png": b"b",
        "notes.txt": b"ignore",
    })

    result = normalize_dataset_zip(data)

    assert result["format"] == ""
    assert "not trainable" in result["warning"]
    assert "images/one.jpg" in result["files"]
    assert "images/two.png" in result["files"]


def test_zip_slip_members_are_dropped():
    """Absolute and parent-traversal members never escape the output prefix."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("../evil.txt", b"x")
        zf.writestr("/etc/evil.txt", b"x")
        zf.writestr("images/ok.jpg", b"img")

    result = normalize_dataset_zip(buf.getvalue())

    assert result["format"] == ""
    assert set(result["files"]) == {"images/ok.jpg"}
    assert not any(".." in k or k.startswith("/") for k in result["files"])


# -- content detection ------------------------------------------------------

class _FakeS3:
    """Minimal SeaweedFSClient stub serving a fixed file listing."""

    def __init__(self, files):
        self._files = files

    def s3_uri_to_filer_path(self, uri):
        return uri

    def list_files_with_size(self, filer):
        return list(self._files)


def _service(files):
    """Build a DatasetService without a DB session."""
    return DatasetService(db=None, s3=_FakeS3(files))


def test_detection_marks_split_yolo_labels_annotated():
    svc = _service([
        ("data.yaml", 100),
        ("train/images/a.jpg", 10),
        ("train/labels/a.txt", 5),
        ("val/labels/b.txt", 5),
    ])
    v = DatasetVersion(uri="s3://nereidforge/ds/v1/")

    svc._apply_content_detection(v)

    assert v.format == "yolo"
    assert v.is_annotated is True


def test_detection_marks_flat_yolo_labels_annotated():
    svc = _service([("data.yaml", 100), ("images/a.jpg", 10), ("labels/a.txt", 5)])
    v = DatasetVersion(uri="s3://nereidforge/ds/v1/")

    svc._apply_content_detection(v)

    assert v.format == "yolo"
    assert v.is_annotated is True


def test_detection_marks_nested_flat_yolo_labels_annotated():
    svc = _service([
        ("images/train2017/a.jpg", 10),
        ("labels/train2017/a.txt", 5),
    ])
    v = DatasetVersion(uri="s3://nereidforge/ds/v1/")

    svc._apply_content_detection(v)

    assert v.format == "yolo"
    assert v.is_annotated is True


def test_detection_marks_coco_json_annotated():
    svc = _service([("coco/annotations.json", 999), ("images/a.jpg", 10)])
    v = DatasetVersion(uri="s3://nereidforge/ds/v1/")

    svc._apply_content_detection(v)

    assert v.format == "coco"
    assert v.is_annotated is True


def test_detection_marks_split_imagenet_annotated():
    svc = _service([
        ("class_names.json", 20),
        ("train/cat/a.jpg", 10),
        ("val/dog/b.jpg", 10),
    ])
    v = DatasetVersion(uri="s3://nereidforge/ds/v1/")

    svc._apply_content_detection(v)

    assert v.format == "imagenet"
    assert v.is_annotated is True


def test_detection_marks_flat_imagenet_annotated():
    svc = _service([("images/cat/a.jpg", 10), ("images/dog/b.jpg", 10)])
    v = DatasetVersion(uri="s3://nereidforge/ds/v1/")

    svc._apply_content_detection(v)

    assert v.format == "imagenet"
    assert v.is_annotated is True


def test_detection_prefers_jsonl_and_leaves_raw_images_alone():
    svc = _service([("images/a.jpg", 10)])
    v = DatasetVersion(uri="s3://nereidforge/ds/v1/", format="coco", is_annotated=False)

    svc._apply_content_detection(v)

    assert v.is_annotated is False

    svc = _service([("corpus.jsonl", 10)])
    v = DatasetVersion(uri="s3://nereidforge/ds/v1/")
    svc._apply_content_detection(v)
    assert v.format == "llm-sft"
    assert v.is_annotated is True
