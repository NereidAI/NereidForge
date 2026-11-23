# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for optimizer calibration dataset inputs.

Canonical YOLO dataset versions are the primary NF detection contract. INT8
PTQ must consume them without a COCO export, preserve split identity and class
order, and retain COCO as a compatibility fallback.
"""

import importlib.util
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image


_OPTIMIZE_PATH = Path(__file__).parents[2] / "builders" / "optimizer" / "optimize.py"
_SPEC = importlib.util.spec_from_file_location("optimizer_optimize", _OPTIMIZE_PATH)
assert _SPEC and _SPEC.loader
optimize = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(optimize)


class _FakeS3:
    """_FakeS3 serves immutable objects and deterministic prefix listings."""

    def __init__(self, objects):
        self.objects = dict(objects)
        self.requested_keys = []

    def get_object(self, Bucket, Key):
        """get_object returns bytes or fails on the exact missing key."""
        self.requested_keys.append(Key)
        if Key not in self.objects:
            raise FileNotFoundError(Key)
        return {"Body": io.BytesIO(self.objects[Key])}

    def get_paginator(self, name):
        """get_paginator reuses this fake for list_objects_v2 pagination."""
        assert name == "list_objects_v2"
        return self

    def paginate(self, Bucket, Prefix):
        """paginate yields one sorted page of objects below Prefix."""
        keys = sorted(key for key in self.objects if key.startswith(Prefix))
        yield {"Contents": [{"Key": key} for key in keys]}


def _image_bytes(width, height):
    """_image_bytes creates an in-memory PNG with known source dimensions."""
    output = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(output, format="PNG")
    return output.getvalue()


def test_canonical_yolo_uses_train_and_val_with_source_pixel_boxes():
    """Canonical split YOLO input needs no COCO file and never mixes same stems."""
    prefix = "datasets/ds/v1"
    s3 = _FakeS3({
        f"{prefix}/data.yaml": b"names:\n  0: mask\n  1: no-mask\n",
        f"{prefix}/train/images/same.jpg": _image_bytes(100, 200),
        f"{prefix}/train/labels/same.txt": b"0 0.5 0.5 0.2 0.4\n",
        f"{prefix}/train/images/second.jpg": _image_bytes(100, 100),
        f"{prefix}/train/labels/second.txt": b"1 0.5 0.5 1.0 1.0\n",
        f"{prefix}/val/images/same.jpg": _image_bytes(100, 200),
        f"{prefix}/val/labels/same.txt": b"1 0.5 0.5 0.4 0.2\n",
        f"{prefix}/test/images/holdout.jpg": _image_bytes(100, 100),
        f"{prefix}/test/labels/holdout.txt": b"0 0.5 0.5 0.2 0.2\n",
    })

    calib, val, names = optimize.calibration_entries(
        s3, f"s3://nereidforge/{prefix}", calib_n=2, val_n=1,
    )

    assert names == ["mask", "no-mask"]
    assert [entry["key"] for entry in calib] == [
        f"{prefix}/train/images/same.jpg",
        f"{prefix}/train/images/second.jpg",
    ]
    assert [entry["key"] for entry in val] == [f"{prefix}/val/images/same.jpg"]
    np.testing.assert_allclose(calib[0]["boxes"][0], [[40, 60, 60, 140]])
    np.testing.assert_allclose(val[0]["boxes"][1], [[30, 80, 70, 120]])
    assert not any(key.endswith("annotations.json") for key in s3.requested_keys)


def test_flat_yolo_preserves_sequential_calibration_validation_fallback():
    """Flat images/labels input retains the existing sequential entry split."""
    prefix = "datasets/flat/v1"
    s3 = _FakeS3({
        f"{prefix}/data.yaml": b"names: [mask, no-mask]\n",
        f"{prefix}/images/a.jpg": _image_bytes(10, 20),
        f"{prefix}/labels/a.txt": b"0 0.5 0.5 0.2 0.5\n",
        f"{prefix}/images/b.jpg": _image_bytes(10, 20),
        f"{prefix}/labels/b.txt": b"1 0.5 0.5 0.2 0.5\n",
    })

    calib, val, names = optimize.calibration_entries(
        s3, f"s3://nereidforge/{prefix}", calib_n=1, val_n=1,
    )

    assert names == ["mask", "no-mask"]
    assert [entry["key"] for entry in calib] == [f"{prefix}/images/a.jpg"]
    assert [entry["key"] for entry in val] == [f"{prefix}/images/b.jpg"]
    np.testing.assert_allclose(calib[0]["boxes"][0], [[4, 5, 6, 15]])


def test_ov_map50_consumes_letterbox_metadata_and_restores_source_boxes(tmp_path):
    """The mAP gate accepts the five metadata values returned by _letterbox."""
    image_path = tmp_path / "source.png"
    Image.new("RGB", (200, 100), "white").save(image_path)
    entry = {
        "key": "source.png",
        "boxes": {0: np.asarray([[10, 20, 30, 60]], dtype=np.float64)},
    }

    def tensor_cache(item):
        """tensor_cache mirrors the optimizer's tensor/metadata cache contract."""
        tensor, *metadata = optimize._letterbox(str(image_path), 100)
        item["_meta"] = metadata
        return tensor

    class _Compiled:
        """_Compiled returns a stable output; decoding is controlled below."""

        def infer_new_request(self, inputs):
            """infer_new_request checks that one image tensor reaches inference."""
            assert inputs[0].shape == (1, 3, 100, 100)
            return {"output": np.zeros((1, 1), dtype=np.float32)}

    class _NSEval:
        """_NSEval checks the exact source-pixel box produced after unmapping."""

        @staticmethod
        def decode_detections(raw, nc, conf, iou):
            """decode_detections returns the GT box in letterboxed coordinates."""
            return (
                np.asarray([[5, 35, 15, 55]], dtype=np.float64),
                np.asarray([0.9]),
                np.asarray([0]),
            )

        @staticmethod
        def detection_metrics(preds, gts, nc, names):
            """detection_metrics rejects incorrect metadata unpacking or scaling."""
            np.testing.assert_allclose(preds[0][0][0][0], gts[0][0][0])
            return {"mAP50": 1.0}

    score = optimize._ov_map50(
        _Compiled(), [entry], ["mask"], 100, _NSEval, tensor_cache,
    )

    assert score == 1.0


def test_nncf_validation_uses_compiled_model_without_recompiling(monkeypatch):
    """NNCF 3.x OpenVINO callbacks receive an already compiled model."""
    ov_model = SimpleNamespace(
        input=lambda index: SimpleNamespace(get_any_name=lambda: "images"),
    )
    baseline_compiled = object()
    nncf_compiled = object()
    quantized_model = object()
    plain_quantized_model = object()
    compile_calls = []
    calib = [{"bucket": "b", "key": f"calib-{i}"} for i in range(4)]
    val = [{"bucket": "b", "key": f"val-{i}"} for i in range(2)]
    callback_scores = iter([0.84, 1.0, 0.825, 0.99, 0.849])
    monkeypatch.setattr(
        optimize, "calibration_entries", lambda *args: (calib, val, ["mask"]),
    )
    def fake_map50(compiled, *args):
        """fake_map50 separates baseline, one-item probes, and full validations."""
        return 0.84 if compiled is baseline_compiled else next(callback_scores)

    monkeypatch.setattr(optimize, "_ov_map50", fake_map50)
    monkeypatch.setattr(
        optimize,
        "_count_fake_quantize",
        lambda model: 10 if model is plain_quantized_model else 8,
    )

    class _Core:
        """_Core permits only the explicit FP32 baseline compilation."""

        def compile_model(self, model, device):
            """compile_model records calls and rejects callback recompilation."""
            compile_calls.append((model, device))
            assert model is ov_model, "NNCF's CompiledModel must be used directly"
            return baseline_compiled

    class _Dataset:
        """_Dataset preserves raw entries for the two-argument NNCF callback."""

        def __init__(self, entries, transform):
            self.entries = entries
            self.transform = transform

        def __iter__(self):
            """__iter__ mirrors NNCF's raw-entry iterable, without get_data()."""
            return iter(self.entries)

    def quantize_with_accuracy_control(model, calib_ds, val_ds, validation_fn, **kwargs):
        """Reproduce NNCF's full validations interleaved with one-item probes."""
        assert model is ov_model
        assert validation_fn(nncf_compiled, val_ds) == 0.84
        assert validation_fn(nncf_compiled, [val[0]]) == 1.0
        assert validation_fn(nncf_compiled, val_ds) == 0.825
        assert validation_fn(nncf_compiled, [val[0]]) == 0.99
        assert validation_fn(nncf_compiled, val_ds) == 0.849
        return quantized_model

    fake_nncf = SimpleNamespace(
        Dataset=_Dataset,
        QuantizationPreset=SimpleNamespace(MIXED="mixed"),
        quantize_with_accuracy_control=quantize_with_accuracy_control,
        quantize=lambda *args, **kwargs: plain_quantized_model,
    )
    monkeypatch.setitem(sys.modules, "nncf", fake_nncf)
    monkeypatch.setitem(sys.modules, "openvino", SimpleNamespace(Core=_Core))
    monkeypatch.setitem(sys.modules, "nseval", SimpleNamespace())

    result, info = optimize.nncf_static_quantize(
        SimpleNamespace(), {"calib_uri": "s3://b/prefix"}, ov_model, 640,
    )

    assert result is quantized_model
    assert compile_calls == [(ov_model, "CPU")]
    assert info["gate1"]["baseline_map50"] == 0.84
    assert info["gate1"]["candidate_map50"] == 0.849
    assert info["gate1"]["full_quantized_map50"] == 0.825
    assert info["gate1"]["passed"] is True
    assert info["quantizer_accuracy_aware"]["layers_reverted_to_fp16"] == 2


def test_coco_remains_compatibility_fallback():
    """COCO input keeps its original entry shape when no YOLO manifest exists."""
    prefix = "datasets/coco/v1"
    annotation = {
        "categories": [{"id": 0, "name": "mask"}],
        "images": [{"id": 7, "file_name": "a.jpg"}],
        "annotations": [{"image_id": 7, "category_id": 0, "bbox": [1, 2, 3, 4]}],
    }
    s3 = _FakeS3({
        f"{prefix}/annotations.json": json.dumps(annotation).encode(),
        f"{prefix}/images/a.jpg": _image_bytes(10, 20),
    })

    calib, val, names = optimize.calibration_entries(
        s3, f"s3://nereidforge/{prefix}", calib_n=1, val_n=1,
    )

    assert names == ["mask"]
    assert len(calib) == len(val) == 1
    assert calib[0]["image_id"] == 7
    np.testing.assert_allclose(calib[0]["boxes"][0], [[1, 2, 4, 6]])
