# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Tests for the NF Paddle trainer that run without PaddleX installed.

They pin everything the platform and the job contract depend on: the announced
profile (no capability may be advertised without an implementation), dataset
layout handling, label order and COCO metric parsing. The PaddleX calls
themselves are exercised by the per-vendor image smoke test, because
installing PaddleX in the dev venv would contradict the image separation.
"""

import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import catalog as paddle_catalog  # noqa: E402
import dataset as paddle_dataset  # noqa: E402
import models as paddle_models  # noqa: E402
import nf_profile as paddle_profile  # noqa: E402
import trainer_metrics  # noqa: E402
from nereidforge_trainer import validate_profile  # noqa: E402

TIERS = {"core", "extended"}


def _committed():
    with open(os.path.join(os.path.dirname(HERE), "trainer-profile.json"), encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Announcement
# ---------------------------------------------------------------------------
def test_committed_profile_matches_generator():
    committed = _committed()
    generated = paddle_profile.profile()
    assert committed.pop("devices")
    generated.pop("devices")
    assert committed == generated


def test_profile_is_valid_and_only_announces_implemented_tasks():
    doc = _committed()
    assert validate_profile(doc) == []
    # The component implements detection only; advertising more is the bug this
    # repo already had once (four Paddle tasks announced, one implementation).
    assert [t["id"] for t in doc["tasks"]] == ["detection"]
    assert doc["tasks"][0]["default_arch"] == paddle_models.DEFAULT_ARCH
    assert all(t["tier"] in TIERS for t in doc["tasks"])


def test_every_architecture_declares_a_licence_and_a_size():
    for arch in _committed()["tasks"][0]["architectures"]:
        assert arch["size"] > 0, arch["id"]
        assert arch["license"] == "Apache-2.0", arch["id"]


def test_catalog_and_profile_agree():
    doc = _committed()
    entry = paddle_catalog.catalog()["tasks"][0]
    assert entry["default"] == doc["tasks"][0]["default_arch"]
    assert entry["architectures"] == [a["id"] for a in doc["tasks"][0]["architectures"]]


def test_entrypoints_exist_as_files():
    doc = _committed()
    for name, command in doc["entrypoints"].items():
        path = command.split()[-1].replace("/app/", "")
        assert os.path.isfile(os.path.join(os.path.dirname(HERE), path)), name


# ---------------------------------------------------------------------------
# Model naming / legacy ids
# ---------------------------------------------------------------------------
def test_default_architecture_maps_to_a_paddlex_model_name():
    assert paddle_models.model_name(paddle_models.DEFAULT_ARCH) == "PP-YOLOE_plus-S"
    assert paddle_models.resolve_model_name("PP-YOLOE_plus-L") == "PP-YOLOE_plus-L"


def test_legacy_yolo_ids_resolve_to_paddle_architectures():
    assert paddle_models.canonical_arch("yolov8n") == "ppyoloe_plus_s"
    assert paddle_models.resolve_model_name("yolov8n") == "PP-YOLOE_plus-S"
    assert paddle_models.canonical_arch("") == paddle_models.DEFAULT_ARCH


# ---------------------------------------------------------------------------
# Dataset layout
# ---------------------------------------------------------------------------
def _write_coco(root, *, train_ids=(1, 2), val_ids=(1, 2), names=("cat", "dog")):
    os.makedirs(os.path.join(root, "images"), exist_ok=True)
    os.makedirs(os.path.join(root, "annotations"), exist_ok=True)
    for name in ("a.jpg", "b.jpg"):
        with open(os.path.join(root, "images", name), "wb") as fh:
            fh.write(b"x")
    categories = [{"id": i, "name": n} for i, n in zip(train_ids, names)]
    payload = {
        "images": [{"id": 1, "file_name": "a.jpg"}, {"id": 2, "file_name": "b.jpg"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": train_ids[0],
                         "bbox": [1, 2, 3, 4], "area": 12, "iscrowd": 0}],
        "categories": categories,
    }
    for name in ("instance_train.json", "instance_val.json"):
        with open(os.path.join(root, "annotations", name), "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
    return payload


def test_coco_layout_is_detected_and_left_alone(tmp_path):
    root = tmp_path / "ds"
    _write_coco(str(root), train_ids=(0, 1))
    assert paddle_dataset.is_coco_layout(str(root))
    assert paddle_dataset.prepare_coco_dataset(str(root)) == str(root)


def test_flat_export_is_materialised_into_coco_layout(tmp_path):
    src = tmp_path / "flat"
    os.makedirs(os.path.join(src, "images"))
    for name in ("a.jpg", "b.png"):
        with open(os.path.join(src, "images", name), "wb") as fh:
            fh.write(b"x")
    payload = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "cat"}]}
    with open(os.path.join(src, "train.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh)

    target = tmp_path / "out"
    result = paddle_dataset.prepare_coco_dataset(str(src), target=str(target))
    assert result == str(target)
    assert os.path.isfile(os.path.join(str(target), "annotations", "instance_train.json"))
    assert os.path.isfile(os.path.join(str(target), "annotations", "instance_val.json"))
    assert sorted(os.listdir(os.path.join(str(target), "images"))) == ["a.jpg", "b.png"]


def test_missing_dataset_raises_with_a_clear_message(tmp_path):
    with pytest.raises(RuntimeError, match="no COCO dataset"):
        paddle_dataset.prepare_coco_dataset(str(tmp_path / "empty"))


def test_labels_are_ordered_by_category_id_and_densified(tmp_path):
    root = tmp_path / "ds"
    _write_coco(str(root), train_ids=(2, 5), names=("dog", "cat"))
    assert paddle_dataset.is_coco_layout(str(root))
    labels = paddle_dataset.load_coco_labels(str(root))
    assert labels == ["dog", "cat"]
    # 1-based (non-contiguous) ids are rewritten in place, keeping a backup copy.
    with open(os.path.join(str(root), "annotations", "instance_train.json"), encoding="utf-8") as fh:
        dense = json.load(fh)
    assert [c["id"] for c in dense["categories"]] == [0, 1]
    assert [a["category_id"] for a in dense["annotations"]] == [0]
    assert os.path.isfile(os.path.join(str(root), "annotations", "dense", "instance_train.json"))


def test_apply_label_map_is_idempotent():
    payload = {
        "categories": [{"id": 3, "name": "a"}, {"id": 7, "name": "b"}],
        "annotations": [{"category_id": 7}],
    }
    first = paddle_dataset.apply_label_map(json.loads(json.dumps(payload)))
    second = paddle_dataset.apply_label_map(first)
    assert second == first
    assert [c["id"] for c in first["categories"]] == [0, 1]
    assert first["annotations"][0]["category_id"] == 1


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
SAMPLE_LOG = """
[10/15 12:00:00] ppdet.utils.eval_utils INFO: Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.312
[10/15 12:00:00] ppdet.utils.eval_utils INFO: Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.512
[10/15 12:00:00] ppdet.utils.eval_utils INFO: Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.322
[10/15 12:00:00] ppdet.utils.eval_utils INFO: Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.101
[10/15 12:00:00] ppdet.utils.eval_utils INFO: Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.401
"""


def test_parse_metrics_extracts_the_coco_table():
    metrics = trainer_metrics.parse_metrics(SAMPLE_LOG)
    assert metrics["mAP50-95"] == 0.312
    assert metrics["mAP50"] == 0.512
    assert metrics["mAP75"] == 0.322
    assert metrics["mAP_small"] == 0.101
    assert metrics["AR100"] == 0.401


def test_parse_metrics_is_safe_on_empty_or_junk_output():
    assert trainer_metrics.parse_metrics("") == {}
    assert trainer_metrics.parse_metrics("no table here") == {}
    assert "no COCO metrics" in trainer_metrics.metrics_summary({})
    assert "mAP50-95" in trainer_metrics.metrics_summary({"mAP50-95": 0.5})


def test_row_alignment_survives_padding_variations():
    """PaddleDetection pads its table; the regex must not depend on spacing."""
    line = "Average Precision  (AP) @[ IoU=0.50:0.95 | area=  all  | maxDets=100 ] = 0.9876"
    assert trainer_metrics.parse_metrics(line)["mAP50-95"] == 0.9876


def test_train_entrypoint_logs_are_syntax_and_import_clean():
    """train.py/evaluate.py/export.py must compile (they import PaddleX lazily)."""
    import py_compile

    root = os.path.dirname(HERE)
    for name in ("train.py", "evaluate.py", "export.py", "nf_profile.py", "catalog.py",
                 "common.py", "dataset.py", "models.py", "trainer_metrics.py"):
        py_compile.compile(os.path.join(root, name), doraise=True)
