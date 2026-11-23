# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for ONNX detection graphs supported by image trials."""

import base64
import copy
from pathlib import Path
from types import SimpleNamespace
import sys

import cv2
from flask import Flask
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import onnx_predict  # noqa: E402
import try_predict  # noqa: E402


@pytest.fixture
def trial_client():
    """Expose the real Flask trial routes against fake ONNX sessions."""
    app = Flask(__name__)
    try_predict.register_try_routes(app)
    return app.test_client()


def _image_base64(image: np.ndarray) -> str:
    """Encode a test image exactly as the HTTP client contract expects."""
    return base64.b64encode(cv2.imencode(".png", image)[1].tobytes()).decode("ascii")


YUNET_NAMES = [
    *(f"cls_{stride}" for stride in (8, 16, 32)),
    *(f"obj_{stride}" for stride in (8, 16, 32)),
    *(f"bbox_{stride}" for stride in (8, 16, 32)),
    *(f"kps_{stride}" for stride in (8, 16, 32)),
]


class FakeSession:
    """Provide named ONNX tensors and capture feeds without onnxruntime."""

    def __init__(self, values, input_shape=None, input_name="images") -> None:
        self.values = {name: np.asarray(value) for name, value in values.items()}
        self.input_name = input_name
        self.input_shape = list(input_shape or [1, 3, 4, 4])
        self.feeds = []

    def get_inputs(self):
        """Describe the single image input like an ONNX Runtime session."""
        return [
            SimpleNamespace(
                name=self.input_name,
                shape=self.input_shape,
                type="tensor(float)",
            )
        ]

    def get_outputs(self):
        """Describe output tensors in their declared insertion order."""
        return [
            SimpleNamespace(name=name, shape=list(value.shape), type="tensor(float)")
            for name, value in self.values.items()
        ]

    def run(self, output_names, feeds):
        """Serve requested values and retain the feed for preprocessing checks."""
        self.feeds.append(feeds)
        names = list(self.values) if output_names is None else output_names
        return [self.values[name] for name in names]


class FakeYuNetSession:
    """Serve one synthetic YuNet face without requiring a model fixture."""

    def __init__(self) -> None:
        self._outputs = []
        for stride in (8, 16, 32):
            anchors = (32 // stride) ** 2
            self._outputs.extend(
                [
                    np.zeros((1, anchors, 1), dtype=np.float32),
                    np.zeros((1, anchors, 1), dtype=np.float32),
                    np.zeros((1, anchors, 4), dtype=np.float32),
                    np.zeros((1, anchors, 10), dtype=np.float32),
                ]
            )
        # Reorder the per-stride tensors into YuNet's output-name groups.
        self._values = {
            name: self._outputs[index]
            for name, index in zip(
                YUNET_NAMES,
                [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11],
                strict=True,
            )
        }
        # Anchor (row=1, col=2) at stride 8 decodes to [8,0,24,16].
        self._values["cls_8"][0, 6, 0] = 0.81
        self._values["obj_8"][0, 6, 0] = 1.0
        self._values["bbox_8"][0, 6, 2] = np.log(2.0)
        self._values["bbox_8"][0, 6, 3] = np.log(2.0)

    def get_inputs(self):
        """Return the static NCHW input expected by the synthetic graph."""
        return [SimpleNamespace(name="input", shape=[1, 3, 32, 32])]

    def get_outputs(self):
        """Expose the same twelve named heads as OpenCV Zoo YuNet."""
        shapes = {name: value.shape for name, value in self._values.items()}
        return [SimpleNamespace(name=name, shape=shapes[name]) for name in YUNET_NAMES]

    def run(self, output_names, _feeds):
        """Return requested tensors while remaining compatible with the old caller."""
        names = YUNET_NAMES if output_names is None else output_names
        return [self._values[name] for name in names]


def test_yunet_raw_outputs_decode_face_box(monkeypatch: pytest.MonkeyPatch) -> None:
    """YuNet must produce a drawable face box instead of parsing cls_8 as YOLO."""
    session = FakeYuNetSession()
    monkeypatch.setattr(onnx_predict, "_cached_session", lambda _path: session)
    monkeypatch.setattr(onnx_predict, "read_onnx_manifest", lambda _path: {})
    monkeypatch.setattr(onnx_predict, "read_onnx_names", lambda _path: None)

    detections = onnx_predict.detect_onnx(
        "yunet.onnx", np.zeros((32, 32, 3), dtype=np.uint8), conf_threshold=0.25
    )

    assert len(detections) == 1
    assert detections[0]["xyxy"] == pytest.approx([8.0, 0.0, 24.0, 16.0])
    assert detections[0]["score"] == pytest.approx(0.9)
    assert detections[0]["cls"] == 0
    assert detections[0]["name"] == "face"


def test_yunet_auto_and_manual_registry_execution_match() -> None:
    """Automatic YuNet matching and explicit selection share one decoder."""
    session = FakeYuNetSession()
    image = np.zeros((32, 32, 3), dtype=np.uint8)

    automatic = onnx_predict.run_onnx_trial(session, image)
    manual = onnx_predict.run_onnx_trial(
        session,
        image,
        decoder={"mode": "builtin", "decoder_id": "yunet"},
    )

    assert automatic["decoder"] == {
        "mode": "auto",
        "id": "yunet",
        "task": "detection",
    }
    assert manual["decoder"]["id"] == "yunet"
    assert automatic["detections"] == manual["detections"]
    assert automatic["detections"][0]["xyxy"] == pytest.approx([8.0, 0.0, 24.0, 16.0])


def test_legacy_yunet_auto_and_manual_registry_execution_match() -> None:
    """The official loc/conf/iou YuNet contract shares the same built-in decoder."""
    loc = np.zeros((58, 14), dtype=np.float32)
    conf = np.zeros((58, 2), dtype=np.float32)
    iou = np.zeros((58, 1), dtype=np.float32)
    conf[0, 1] = 0.81
    iou[0, 0] = 1.0
    session = FakeSession(
        {"loc": loc, "conf": conf, "iou": iou},
        input_shape=[1, 3, 32, 32],
        input_name="input",
    )
    image = np.zeros((32, 32, 3), dtype=np.uint8)

    automatic = onnx_predict.run_onnx_trial(session, image)
    manual = onnx_predict.run_onnx_trial(
        session,
        image,
        decoder={"mode": "builtin", "decoder_id": "yunet"},
    )

    assert automatic["decoder"] == {
        "mode": "auto",
        "id": "yunet",
        "task": "detection",
    }
    assert automatic["detections"] == manual["detections"]
    assert len(automatic["detections"]) == 1
    assert automatic["detections"][0]["xyxy"] == pytest.approx([-1.0, -1.0, 9.0, 9.0])
    assert automatic["detections"][0]["score"] == pytest.approx(0.9)


def test_unknown_auto_requires_mapping_but_inspection_exposes_candidates() -> None:
    """A YOLO-shaped tensor is not evidence that the graph is a detector."""
    session = FakeSession({"output": np.zeros((1, 84, 100), dtype=np.float32)})

    inspection = onnx_predict.inspect_decoders(session)

    assert inspection["auto_decoder"] is None
    assert inspection["mapping_required"] is True
    assert {item["id"] for item in inspection["decoders"]} >= {
        "yunet",
        "yolo",
        "mask_rcnn",
        "semantic_logits",
    }
    with pytest.raises(
        onnx_predict.DecoderMappingRequired, match="^decoder mapping required$"
    ):
        onnx_predict.run_onnx_trial(session, np.zeros((20, 30, 3), dtype=np.uint8))


def test_custom_detection_rows_restore_letterbox_coordinates_and_nms() -> None:
    """Explicit row fields decode boxes and remove duplicates at the request IoU."""
    session = FakeSession(
        {
            "prediction": np.array(
                [[[8, 0, 24, 32, 0.9, 1], [8, 0, 24, 32, 0.8, 1]]],
                dtype=np.float32,
            )
        },
        input_shape=[1, 3, 32, 32],
    )
    decoder = {
        "mode": "custom",
        "task": "detection",
        "input": {
            "tensor": "images",
            "width": 32,
            "height": 32,
            "layout": "NCHW",
            "color": "BGR",
            "resize": "letterbox",
            "scale": 1 / 255,
        },
        "output": {
            "prediction": "prediction",
            "layout": "rows",
            "bbox_start": 0,
            "bbox_encoding": "xyxy",
            "score_index": 4,
            "class_index": 5,
            "coordinate_space": "pixels",
            "score_activation": "none",
        },
    }

    result = onnx_predict.run_onnx_trial(
        session,
        np.zeros((40, 20, 3), dtype=np.uint8),
        decoder=decoder,
        conf_threshold=0.25,
        iou_threshold=0.5,
        names=["zero", "one"],
    )

    assert result["kind"] == "detection"
    assert result["decoder"] == {
        "mode": "custom",
        "id": "custom_detection",
        "task": "detection",
    }
    assert len(result["detections"]) == 1
    assert result["detections"][0] == {
        "xyxy": pytest.approx([0.0, 0.0, 20.0, 40.0]),
        "score": pytest.approx(0.9),
        "cls": 1,
        "name": "one",
    }
    assert session.feeds[0]["images"].shape == (1, 3, 32, 32)


def test_custom_detection_columns_support_cxcywh_normalized_and_sigmoid() -> None:
    """Column matrices use declared indices and stretch back on both axes."""
    columns = np.array([0.5, 0.5, 0.5, 0.5, 0.0, 2.0], dtype=np.float32)
    session = FakeSession(
        {"scores": columns.reshape(1, 6, 1)}, input_shape=[1, 3, 10, 10]
    )
    decoder = {
        "mode": "custom",
        "task": "detection",
        "input": {
            "tensor": "images",
            "width": 10,
            "height": 10,
            "layout": "NCHW",
            "color": "RGB",
            "resize": "stretch",
            "scale": 1.0,
        },
        "output": {
            "prediction": "scores",
            "layout": "columns",
            "bbox_start": 0,
            "bbox_encoding": "cxcywh",
            "score_index": 4,
            "class_index": 5,
            "coordinate_space": "normalized",
            "score_activation": "sigmoid",
        },
    }

    result = onnx_predict.run_onnx_trial(
        session, np.zeros((20, 40, 3), dtype=np.uint8), decoder=decoder
    )

    assert result["detections"][0]["xyxy"] == pytest.approx([10.0, 5.0, 30.0, 15.0])
    assert result["detections"][0]["score"] == pytest.approx(0.5)
    assert result["detections"][0]["cls"] == 2


def test_custom_classification_honors_class_axis_activation_labels_and_topk() -> None:
    """Classification removes only batch/singleton axes and activates on the declared axis."""
    session = FakeSession(
        {"logits": np.array([[[1.0], [2.0], [3.0]]], dtype=np.float32)}
    )
    decoder = {
        "mode": "custom",
        "task": "classification",
        "input": {
            "tensor": "images",
            "width": 4,
            "height": 4,
            "layout": "NCHW",
            "color": "RGB",
            "resize": "stretch",
            "scale": 1 / 255,
        },
        "output": {
            "output": "logits",
            "class_axis": 1,
            "activation": "softmax",
        },
        "labels": ["cat", "dog", "fox"],
    }

    result = onnx_predict.run_onnx_trial(
        session,
        np.zeros((8, 6, 3), dtype=np.uint8),
        decoder=decoder,
        top_k=2,
    )

    assert result["kind"] == "classification"
    assert result["decoder"]["id"] == "custom_classification"
    assert [item["cls"] for item in result["predictions"]] == [2, 1]
    assert [item["name"] for item in result["predictions"]] == ["fox", "dog"]
    expected = np.exp([1.0, 2.0, 3.0]) / np.exp([1.0, 2.0, 3.0]).sum()
    assert result["predictions"][0]["prob"] == pytest.approx(expected[2])
    assert len(result["predictions"]) == 2


def test_custom_semantic_logits_use_declared_axis_and_nearest_original_geometry() -> (
    None
):
    """Semantic logits become an original-resolution class-id mask."""
    logits = np.zeros((1, 2, 2, 2), dtype=np.float32)
    logits[0, 0, 0, 0] = 2
    logits[0, 1, 0, 1] = 2
    logits[0, 1, 1, 0] = 2
    logits[0, 0, 1, 1] = 2
    session = FakeSession({"logits": logits}, input_shape=[1, 3, 2, 2])
    decoder = {
        "mode": "custom",
        "task": "semantic_segmentation",
        "input": {
            "tensor": "images",
            "width": 2,
            "height": 2,
            "layout": "NCHW",
            "color": "RGB",
            "resize": "stretch",
            "scale": 1 / 255,
        },
        "output": {
            "output": "logits",
            "format": "logits",
            "class_axis": 1,
        },
    }

    result = onnx_predict.run_onnx_trial(
        session, np.zeros((4, 4, 3), dtype=np.uint8), decoder=decoder
    )

    assert result["kind"] == "semantic_segmentation"
    assert result["decoder"]["id"] == "custom_semantic_segmentation"
    np.testing.assert_array_equal(
        result["mask"],
        np.array([[0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 0, 0], [1, 1, 0, 0]]),
    )


def test_custom_instance_segmentation_restores_masks_without_detection_fallback() -> (
    None
):
    """Named custom tensors retain per-instance masks through the common DTO input."""
    session = FakeSession(
        {
            "boxes": np.array([[[0, 0, 4, 4]]], dtype=np.float32),
            "scores": np.array([[0.9]], dtype=np.float32),
            "labels": np.array([[1]], dtype=np.int64),
            "masks": np.array([[[[[-10, 10], [10, -10]]]]], dtype=np.float32).reshape(
                1, 1, 2, 2
            ),
        },
        input_shape=[1, 3, 4, 4],
    )
    decoder = {
        "mode": "custom",
        "task": "instance_segmentation",
        "input": {
            "tensor": "images",
            "width": 4,
            "height": 4,
            "layout": "NCHW",
            "color": "RGB",
            "resize": "stretch",
            "scale": 1 / 255,
        },
        "output": {
            "boxes": "boxes",
            "scores": "scores",
            "labels": "labels",
            "masks": "masks",
            "bbox_encoding": "xyxy",
            "coordinate_space": "pixels",
            "mask_activation": "sigmoid",
            "mask_threshold": 0.5,
        },
    }

    result = onnx_predict.run_onnx_trial(
        session,
        np.zeros((4, 4, 3), dtype=np.uint8),
        decoder=decoder,
        names=["background", "object"],
    )

    assert result["kind"] == "instance_segmentation"
    assert len(result["detections"]) == 1
    detection = result["detections"][0]
    assert detection["name"] == "object"
    assert detection["xyxy"] == pytest.approx([0, 0, 4, 4])
    np.testing.assert_array_equal(
        detection["mask"],
        np.array(
            [
                [0, 0, 255, 255],
                [0, 0, 255, 255],
                [255, 255, 0, 0],
                [255, 255, 0, 0],
            ],
            dtype=np.uint8,
        ),
    )


def test_named_mask_rcnn_contract_auto_matches_without_shape_inference() -> None:
    """The complete boxes/scores/labels/masks names are an automatic contract."""
    session = FakeSession(
        {
            "boxes": np.array([[[0, 0, 4, 4]]], dtype=np.float32),
            "scores": np.array([[0.75]], dtype=np.float32),
            "labels": np.array([[0]], dtype=np.int64),
            "masks": np.ones((1, 1, 4, 4), dtype=np.float32),
        },
        input_shape=[1, 3, 4, 4],
    )

    inspection = onnx_predict.inspect_decoders(session)
    result = onnx_predict.run_onnx_trial(session, np.zeros((4, 4, 3), dtype=np.uint8))

    assert inspection["auto_decoder"] == {
        "id": "mask_rcnn",
        "task": "instance_segmentation",
    }
    assert result["kind"] == "instance_segmentation"
    assert len(result["detections"]) == 1
    assert np.all(result["detections"][0]["mask"] == 255)


def test_torchvision_classification_auto_uses_manifest_and_manual_matches() -> None:
    """A classification manifest, rather than a rank-2 shape, selects torchvision."""
    session = FakeSession(
        {"logits": np.array([[0.0, 1.0, 2.0]], dtype=np.float32)},
        input_shape=[1, 3, 2, 2],
    )
    manifest = {"task": "classification", "input": {"size": 2}}
    image = np.zeros((2, 2, 3), dtype=np.uint8)

    automatic = onnx_predict.run_onnx_trial(session, image, manifest=manifest)
    manual = onnx_predict.run_onnx_trial(
        session,
        image,
        decoder={"mode": "builtin", "decoder_id": "torchvision_classification"},
    )

    assert automatic["decoder"]["id"] == "torchvision_classification"
    assert automatic["predictions"] == manual["predictions"]
    assert [item["cls"] for item in automatic["predictions"]] == [2, 1, 0]


def test_semantic_logits_auto_requires_native_manifest_and_manual_uses_nchw() -> None:
    """Semantic shape alone is insufficient; native semantic_seg metadata opts in."""
    logits = np.zeros((1, 2, 2, 2), dtype=np.float32)
    logits[:, 1, :, 1] = 1.0
    session = FakeSession({"logits": logits}, input_shape=[1, 3, 2, 2])
    image = np.zeros((2, 2, 3), dtype=np.uint8)

    automatic = onnx_predict.run_onnx_trial(
        session, image, manifest={"task": "semantic_seg", "input": {"size": 2}}
    )
    manual = onnx_predict.run_onnx_trial(
        session,
        image,
        decoder={"mode": "builtin", "decoder_id": "semantic_logits"},
    )

    assert automatic["decoder"]["id"] == "semantic_logits"
    np.testing.assert_array_equal(automatic["mask"], manual["mask"])
    np.testing.assert_array_equal(automatic["mask"], np.array([[0, 1], [0, 1]]))
    aliased = onnx_predict.inspect_decoders(session, {"task": "semantic_segmentation"})
    assert aliased["auto_decoder"] is None


def test_named_detection_contract_auto_matches_exact_tensor_names() -> None:
    """Boxes/scores/labels without masks select the named detection decoder."""
    session = FakeSession(
        {
            "boxes": np.array([[[0, 0, 4, 4]]], dtype=np.float32),
            "scores": np.array([[0.8]], dtype=np.float32),
            "labels": np.array([[0]], dtype=np.int64),
        },
        input_shape=[1, 3, 4, 4],
    )

    result = onnx_predict.run_onnx_trial(session, np.zeros((4, 4, 3), dtype=np.uint8))

    assert result["decoder"]["id"] == "named_detection"
    assert result["detections"][0]["xyxy"] == pytest.approx([0, 0, 4, 4])
    assert result["detections"][0]["score"] == pytest.approx(0.8)


def test_yolo_auto_requires_explicit_family_and_matches_manual_decoder() -> None:
    """An ultralytics family hint enables the verified YOLO tensor contract."""
    prediction = np.array([[[16.0], [16.0], [16.0], [16.0], [0.9]]], dtype=np.float32)
    session = FakeSession({"output": prediction}, input_shape=[1, 3, 32, 32])
    image = np.zeros((32, 32, 3), dtype=np.uint8)

    automatic = onnx_predict.run_onnx_trial(session, image, source="ultralytics")
    manual = onnx_predict.run_onnx_trial(
        session, image, decoder={"mode": "builtin", "decoder_id": "yolo"}
    )

    assert automatic["decoder"]["id"] == "yolo"
    assert automatic["detections"] == manual["detections"]
    assert automatic["detections"][0]["xyxy"] == pytest.approx([8, 8, 24, 24])
    assert automatic["detections"][0]["score"] == pytest.approx(0.9)


def test_http_inspect_succeeds_when_auto_predict_requires_mapping(
    trial_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inspection exposes tensors while unknown auto execution returns the exact 422."""
    session = FakeSession({"output": np.zeros((1, 84, 100), dtype=np.float32)})
    monkeypatch.setattr(
        try_predict,
        "_load_onnx_session",
        lambda _ref: (session, "/models/unknown.onnx"),
    )
    monkeypatch.setattr(onnx_predict, "read_onnx_manifest", lambda _path: {})
    monkeypatch.setattr(onnx_predict, "read_onnx_names", lambda _path: None)
    image = _image_base64(np.zeros((8, 12, 3), dtype=np.uint8))

    inspected = trial_client.post(
        "/try/inspect",
        json={"model_path": "unknown.onnx", "input_shape": [1, 3, 8, 12]},
    )
    predicted = trial_client.post(
        "/try/predict",
        json={"model_path": "unknown.onnx", "image_base64": image},
    )

    assert inspected.status_code == 200
    assert inspected.json["artifact"] == "unknown.onnx"
    assert inspected.json["inputs"] == [
        {"name": "images", "shape": [1, 3, 4, 4], "dtype": "tensor(float)"}
    ]
    assert inspected.json["outputs"][0]["name"] == "output"
    assert inspected.json["mapping_required"] is True
    assert inspected.json["auto_decoder"] is None
    assert inspected.json["names"] == {}
    assert predicted.status_code == 422
    assert predicted.json == {
        "error": "decoder mapping required",
        "code": "decoder_mapping_required",
    }


def test_http_yunet_auto_manual_parity_and_request_iou(
    trial_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The YuNet route shares auto/manual decoding and honors request NMS IoU."""
    session = FakeYuNetSession()
    session._values["cls_8"][0, 7, 0] = 0.64
    session._values["obj_8"][0, 7, 0] = 1.0
    session._values["bbox_8"][0, 7, 2] = np.log(2.0)
    session._values["bbox_8"][0, 7, 3] = np.log(2.0)
    monkeypatch.setattr(
        try_predict,
        "_load_onnx_session",
        lambda _ref: (session, "/models/yunet.onnx"),
    )
    monkeypatch.setattr(onnx_predict, "read_onnx_manifest", lambda _path: {})
    monkeypatch.setattr(onnx_predict, "read_onnx_names", lambda _path: None)
    base = {
        "model_path": "yunet.onnx",
        "image_base64": _image_base64(np.zeros((32, 32, 3), dtype=np.uint8)),
        "conf": 0.25,
    }

    automatic = trial_client.post("/try/predict", json={**base, "iou": 0.5})
    manual = trial_client.post(
        "/try/predict",
        json={
            **base,
            "iou": 0.5,
            "decoder": {"mode": "builtin", "decoder_id": "yunet"},
        },
    )
    suppressed = trial_client.post("/try/predict", json={**base, "iou": 0.2})

    assert automatic.status_code == manual.status_code == 200
    assert automatic.json["kind"] == "detection"
    assert automatic.json["decoder"] == {
        "mode": "auto",
        "id": "yunet",
        "task": "detection",
    }
    assert automatic.json["boxes"] == manual.json["boxes"]
    assert len(automatic.json["boxes"]) == 2
    assert len(suppressed.json["boxes"]) == 1


def test_http_custom_classification_contains_legacy_top5_and_requested_topk(
    trial_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The HTTP DTO retains five legacy entries and a separate requested top-k list."""
    session = FakeSession(
        {"logits": np.array([[0.0, 1.0, 2.0]], dtype=np.float32)},
        input_shape=[1, 3, 4, 4],
    )
    monkeypatch.setattr(
        try_predict,
        "_load_onnx_session",
        lambda _ref: (session, "/models/classifier.onnx"),
    )
    monkeypatch.setattr(onnx_predict, "read_onnx_manifest", lambda _path: {})
    monkeypatch.setattr(onnx_predict, "read_onnx_names", lambda _path: None)
    decoder = {
        "mode": "custom",
        "task": "classification",
        "input": {
            "tensor": "images",
            "width": 4,
            "height": 4,
            "layout": "NCHW",
            "color": "RGB",
            "resize": "stretch",
            "scale": 1 / 255,
        },
        "output": {"output": "logits", "class_axis": 1, "activation": "softmax"},
        "labels": ["cat", "dog", "fox"],
    }

    response = trial_client.post(
        "/try/predict",
        json={
            "model_path": "classifier.onnx",
            "image_base64": _image_base64(np.zeros((4, 4, 3), dtype=np.uint8)),
            "decoder": decoder,
            "top_k": 2,
        },
    )

    assert response.status_code == 200
    assert response.json["kind"] == "classification"
    assert response.json["decoder"] == {
        "mode": "custom",
        "id": "custom_classification",
        "task": "classification",
    }
    assert len(response.json["probs"]["top5"]) == 3
    assert response.json["probs"]["topk"] == [
        {"class_index": 2, "label": "fox", "score": 0.66524},
        {"class_index": 1, "label": "dog", "score": 0.24473},
    ]


def test_http_custom_semantic_returns_original_resolution_rgba_overlay(
    trial_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Semantic responses contain a decodable RGBA PNG, legend, and no detections."""
    logits = np.zeros((1, 2, 2, 2), dtype=np.float32)
    logits[0, 1, 0, 1] = 2
    logits[0, 1, 1, 0] = 2
    session = FakeSession({"logits": logits}, input_shape=[1, 3, 2, 2])
    monkeypatch.setattr(
        try_predict,
        "_load_onnx_session",
        lambda _ref: (session, "/models/semantic.onnx"),
    )
    monkeypatch.setattr(onnx_predict, "read_onnx_manifest", lambda _path: {})
    monkeypatch.setattr(onnx_predict, "read_onnx_names", lambda _path: None)
    decoder = {
        "mode": "custom",
        "task": "semantic_segmentation",
        "input": {
            "tensor": "images",
            "width": 2,
            "height": 2,
            "layout": "NCHW",
            "color": "RGB",
            "resize": "stretch",
            "scale": 1 / 255,
        },
        "output": {"output": "logits", "format": "logits", "class_axis": 1},
        "labels": ["background", "foreground"],
    }

    response = trial_client.post(
        "/try/predict",
        json={
            "model_path": "semantic.onnx",
            "image_base64": _image_base64(np.zeros((4, 4, 3), dtype=np.uint8)),
            "decoder": decoder,
        },
    )

    assert response.status_code == 200
    assert response.json["kind"] == "semantic_segmentation"
    assert response.json["boxes"] == []
    assert response.json["polygons"] == []
    assert response.json["probs"] is None
    assert response.json["mask_legend"] == [
        {"cls": 0, "label": "background", "color": [0, 0, 0, 0]},
        {"cls": 1, "label": "foreground", "color": [31, 119, 180, 128]},
    ]
    png = base64.b64decode(response.json["mask_overlay"])
    overlay = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
    assert overlay.shape == (4, 4, 4)
    assert overlay[0, 0].tolist() == [0, 0, 0, 0]
    assert overlay[0, 2].tolist() == [180, 119, 31, 128]


def test_http_custom_instance_retains_polygons(
    trial_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Custom instance output reaches the existing polygon DTO conversion."""
    session = FakeSession(
        {
            "boxes": np.array([[[0, 0, 4, 4]]], dtype=np.float32),
            "scores": np.array([[0.9]], dtype=np.float32),
            "labels": np.array([[0]], dtype=np.int64),
            "masks": np.ones((1, 1, 2, 2), dtype=np.float32),
        },
        input_shape=[1, 3, 4, 4],
    )
    monkeypatch.setattr(
        try_predict,
        "_load_onnx_session",
        lambda _ref: (session, "/models/instance.onnx"),
    )
    monkeypatch.setattr(onnx_predict, "read_onnx_manifest", lambda _path: {})
    monkeypatch.setattr(onnx_predict, "read_onnx_names", lambda _path: ["object"])
    decoder = {
        "mode": "custom",
        "task": "instance_segmentation",
        "input": {
            "tensor": "images",
            "width": 4,
            "height": 4,
            "layout": "NCHW",
            "color": "RGB",
            "resize": "stretch",
            "scale": 1 / 255,
        },
        "output": {
            "boxes": "boxes",
            "scores": "scores",
            "labels": "labels",
            "masks": "masks",
            "bbox_encoding": "xyxy",
            "coordinate_space": "pixels",
            "mask_activation": "none",
            "mask_threshold": 0.5,
        },
    }

    response = trial_client.post(
        "/try/predict",
        json={
            "model_path": "instance.onnx",
            "image_base64": _image_base64(np.zeros((4, 4, 3), dtype=np.uint8)),
            "decoder": decoder,
        },
    )

    assert response.status_code == 200
    assert response.json["kind"] == "instance_segmentation"
    assert response.json["decoder"]["id"] == "custom_instance_segmentation"
    assert len(response.json["boxes"]) == 1
    assert len(response.json["polygons"]) == 1
    assert response.json["polygons"][0]["label"] == "object"
    assert len(response.json["polygons"][0]["points"]) >= 3


def test_http_segmentation_mapping_errors_never_fall_back_to_detection(
    trial_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A detection graph selected as Mask R-CNN fails instead of returning boxes."""
    session = FakeSession(
        {
            "boxes": np.array([[[0, 0, 4, 4]]], dtype=np.float32),
            "scores": np.array([[0.9]], dtype=np.float32),
            "labels": np.array([[0]], dtype=np.int64),
        },
        input_shape=[1, 3, 4, 4],
    )
    monkeypatch.setattr(
        try_predict,
        "_load_onnx_session",
        lambda _ref: (session, "/models/detector.onnx"),
    )
    monkeypatch.setattr(onnx_predict, "read_onnx_manifest", lambda _path: {})
    monkeypatch.setattr(onnx_predict, "read_onnx_names", lambda _path: None)

    response = trial_client.post(
        "/try/predict",
        json={
            "model_path": "detector.onnx",
            "image_base64": _image_base64(np.zeros((4, 4, 3), dtype=np.uint8)),
            "decoder": {"mode": "builtin", "decoder_id": "mask_rcnn"},
        },
    )

    assert response.status_code == 422
    assert response.json["code"] == "invalid_decoder_mapping"
    assert "masks" in response.json["error"]
    assert "kind" not in response.json


def test_yunet_auto_requires_every_named_head_even_with_architecture_hint() -> None:
    """An architecture label cannot complete a partial YuNet output signature."""
    values = {
        name: np.zeros((1, 1, 1), dtype=np.float32)
        for name in YUNET_NAMES
        if name != "kps_32"
    }
    session = FakeSession(values, input_shape=[1, 3, 32, 32])

    inspection = onnx_predict.inspect_decoders(session, architecture="yunet")

    assert inspection["auto_decoder"] is None
    assert (
        next(item for item in inspection["decoders"] if item["id"] == "yunet")[
            "matched"
        ]
        is False
    )
    with pytest.raises(onnx_predict.DecoderMappingRequired):
        onnx_predict.run_onnx_trial(
            session,
            np.zeros((32, 32, 3), dtype=np.uint8),
            architecture="yunet",
        )


def test_custom_mapping_rejects_missing_fields_and_fixed_dimension_mismatch() -> None:
    """Malformed mappings fail validation instead of receiving inferred defaults."""
    session = FakeSession(
        {"prediction": np.zeros((1, 1, 6), dtype=np.float32)},
        input_shape=[1, 3, 4, 4],
    )
    decoder = {
        "mode": "custom",
        "task": "detection",
        "input": {
            "tensor": "images",
            "width": 4,
            "height": 4,
            "layout": "NCHW",
            "color": "RGB",
            "resize": "stretch",
            "scale": 1.0,
        },
        "output": {
            "prediction": "prediction",
            "layout": "rows",
            "bbox_start": 0,
            "bbox_encoding": "xyxy",
            "score_index": 4,
            "class_index": 5,
            "coordinate_space": "pixels",
            "score_activation": "none",
        },
    }
    missing = copy.deepcopy(decoder)
    del missing["output"]["score_index"]
    wrong_size = copy.deepcopy(decoder)
    wrong_size["input"]["width"] = 8
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    with pytest.raises(onnx_predict.DecoderMappingError, match="score_index"):
        onnx_predict.run_onnx_trial(session, image, decoder=missing)
    with pytest.raises(
        onnx_predict.DecoderMappingError, match="does not match graph width"
    ):
        onnx_predict.run_onnx_trial(session, image, decoder=wrong_size)


def test_custom_semantic_class_ids_preserve_ignore_index() -> None:
    """Class-id output needs no class axis and keeps ignored pixels identifiable."""
    session = FakeSession(
        {"mask": np.array([[[0, 255], [1, 0]]], dtype=np.int64)},
        input_shape=[1, 3, 2, 2],
    )
    decoder = {
        "mode": "custom",
        "task": "semantic_segmentation",
        "input": {
            "tensor": "images",
            "width": 2,
            "height": 2,
            "layout": "NCHW",
            "color": "BGR",
            "resize": "stretch",
            "scale": 1.0,
        },
        "output": {
            "output": "mask",
            "format": "class_ids",
            "ignore_index": 255,
        },
    }

    result = onnx_predict.run_onnx_trial(
        session, np.zeros((2, 2, 3), dtype=np.uint8), decoder=decoder
    )

    np.testing.assert_array_equal(result["mask"], np.array([[0, 255], [1, 0]]))
    assert result["ignore_index"] == 255


def _yolo_output_contract(
    output_format: str = "raw_cxcywh",
    layout: str = "rows",
    coordinate_space: str = "pixels",
) -> dict:
    """Declare the otherwise ambiguous YOLO tensor contract in manifest form."""
    return {
        "outputs": {
            "prediction": {
                "format": output_format,
                "layout": layout,
                "coordinate_space": coordinate_space,
            }
        }
    }


def test_yolo_two_class_raw_output_is_not_mistaken_for_end_to_end() -> None:
    """Six raw fields mean four coordinates plus two class scores, not score/class."""
    session = FakeSession(
        {"output": np.array([[[16, 0, 16, 16, 0.2, 0.9]]], dtype=np.float32)},
        input_shape=[1, 3, 32, 32],
    )

    result = onnx_predict.run_onnx_trial(
        session,
        np.zeros((32, 32, 3), dtype=np.uint8),
        decoder={"mode": "builtin", "decoder_id": "yolo"},
        names=["zero", "one"],
    )

    assert len(result["detections"]) == 1
    assert result["detections"][0]["score"] == pytest.approx(0.9)
    assert result["detections"][0]["cls"] == 1


def test_yolo_end_to_end_requires_and_honors_an_explicit_manifest_contract() -> None:
    """Post-NMS xyxy/score/class remains available without shape-based ambiguity."""
    session = FakeSession(
        {"output": np.array([[[8, 0, 24, 16, 0.9, 1]]], dtype=np.float32)},
        input_shape=[1, 3, 32, 32],
    )

    result = onnx_predict.run_onnx_trial(
        session,
        np.zeros((32, 32, 3), dtype=np.uint8),
        manifest=_yolo_output_contract("end2end_xyxy"),
        decoder={"mode": "builtin", "decoder_id": "yolo"},
    )

    assert result["detections"] == [{
        "xyxy": pytest.approx([8, 0, 24, 16]),
        "score": pytest.approx(0.9),
        "cls": 1,
        "name": None,
    }]


@pytest.mark.parametrize("orientation", ["rows", "columns"])
def test_yolo_small_layouts_use_the_label_count_contract(orientation: str) -> None:
    """Candidate counts below the field count do not reverse the tensor orientation."""
    values = np.zeros((1, 10, 84), dtype=np.float32)
    if orientation == "columns":
        values = values.transpose(0, 2, 1)
    for index in range(10):
        coordinates = [16, 16, 16, 16]
        if orientation == "rows":
            values[0, index, :4] = coordinates
            values[0, index, 83] = 0.9
        else:
            values[0, :4, index] = coordinates
            values[0, 83, index] = 0.9
    session = FakeSession({"output": values}, input_shape=[1, 3, 32, 32])

    result = onnx_predict.run_onnx_trial(
        session,
        np.zeros((32, 32, 3), dtype=np.uint8),
        decoder={"mode": "builtin", "decoder_id": "yolo"},
        names=[f"class-{index}" for index in range(80)],
    )

    assert len(result["detections"]) == 1
    assert result["detections"][0]["cls"] == 79
    assert result["detections"][0]["xyxy"] == pytest.approx([8, 8, 24, 24])


@pytest.mark.parametrize("shape", [(1, 0, 84), (1, 84, 0)])
def test_yolo_empty_layouts_return_no_detections(shape: tuple[int, int, int]) -> None:
    """Empty raw outputs preserve their declared field axis and return an empty list."""
    session = FakeSession(
        {"output": np.zeros(shape, dtype=np.float32)}, input_shape=[1, 3, 32, 32]
    )

    result = onnx_predict.run_onnx_trial(
        session,
        np.zeros((32, 32, 3), dtype=np.uint8),
        decoder={"mode": "builtin", "decoder_id": "yolo"},
        names=[f"class-{index}" for index in range(80)],
    )

    assert result["detections"] == []


def test_yolo_ambiguous_small_layout_requires_a_declared_contract() -> None:
    """Without labels or a manifest, two plausible small layouts are not guessed."""
    session = FakeSession(
        {"output": np.zeros((1, 10, 84), dtype=np.float32)},
        input_shape=[1, 3, 32, 32],
    )

    with pytest.raises(onnx_predict.DecoderMappingError, match="layout"):
        onnx_predict.run_onnx_trial(
            session,
            np.zeros((32, 32, 3), dtype=np.uint8),
            decoder={"mode": "builtin", "decoder_id": "yolo"},
        )


def test_yolo_small_pixel_coordinates_do_not_imply_normalized_space() -> None:
    """Coordinate values never select a different geometric contract."""
    session = FakeSession(
        {"output": np.array([[[0.5, 0.5, 0.25, 0.25, 0.2, 0.9]]], dtype=np.float32)},
        input_shape=[1, 3, 32, 32],
    )

    result = onnx_predict.run_onnx_trial(
        session,
        np.zeros((32, 32, 3), dtype=np.uint8),
        decoder={"mode": "builtin", "decoder_id": "yolo"},
        names=["zero", "one"],
    )

    assert result["detections"][0]["xyxy"] == pytest.approx(
        [0.375, 0.375, 0.625, 0.625]
    )


def test_yolo_normalized_boxes_undo_letterbox_before_source_scaling() -> None:
    """Normalized raw coordinates cover the letterboxed content, including its offset."""
    session = FakeSession(
        {"output": np.array([[[0.5, 0.5, 0.5, 1.0, 0.2, 0.9]]], dtype=np.float32)},
        input_shape=[1, 3, 32, 32],
    )

    result = onnx_predict.run_onnx_trial(
        session,
        np.zeros((40, 20, 3), dtype=np.uint8),
        manifest=_yolo_output_contract(coordinate_space="normalized"),
        decoder={"mode": "builtin", "decoder_id": "yolo"},
        names=["zero", "one"],
    )

    assert result["detections"][0]["xyxy"] == pytest.approx([0, 0, 20, 40])


def test_detect_wrapper_preserves_manifest_size_for_dynamic_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default wrapper argument must not override a dynamic graph's manifest size."""
    session = FakeSession(
        {"output": np.array([[[16, 16, 16, 16, 0.2, 0.9]]], dtype=np.float32)},
        input_shape=[1, 3, "height", "width"],
    )
    monkeypatch.setattr(onnx_predict, "_cached_session", lambda _path: session)
    monkeypatch.setattr(
        onnx_predict,
        "read_onnx_manifest",
        lambda _path: {"task": "detection", "input": {"size": 32}},
    )
    monkeypatch.setattr(onnx_predict, "read_onnx_names", lambda _path: ["zero", "one"])

    detections = onnx_predict.detect_onnx(
        "dynamic.onnx", np.zeros((32, 32, 3), dtype=np.uint8)
    )

    assert len(detections) == 1
    assert session.feeds[0]["images"].shape == (1, 3, 32, 32)


def test_classification_wrapper_keeps_logits_plus_features_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first output remains logits when an auxiliary features tensor follows it."""
    session = FakeSession(
        {
            "output": np.array([[0.0, 1.0, 2.0]], dtype=np.float32),
            "features": np.zeros((1, 4), dtype=np.float32),
        },
        input_shape=[1, 3, 2, 2],
    )
    monkeypatch.setattr(onnx_predict, "_cached_session", lambda _path: session)
    monkeypatch.setattr(
        onnx_predict, "read_onnx_manifest", lambda _path: {"task": "classification"}
    )
    monkeypatch.setattr(onnx_predict, "read_onnx_names", lambda _path: None)

    predictions = onnx_predict.classify_onnx(
        "classifier.onnx", np.zeros((2, 2, 3), dtype=np.uint8)
    )

    assert [item["cls"] for item in predictions] == [2, 1, 0]


def _edge_custom_decoder(task: str, output: dict, resize: str = "stretch") -> dict:
    """Build a complete custom request so each edge test changes only output semantics."""
    return {
        "mode": "custom",
        "task": task,
        "input": {
            "tensor": "images",
            "width": 4,
            "height": 4,
            "layout": "NCHW",
            "color": "RGB",
            "resize": resize,
            "scale": 1 / 255,
        },
        "output": output,
    }


@pytest.mark.parametrize("decoder_kind", ["custom", "named_detection", "mask_rcnn"])
@pytest.mark.parametrize("label", [1.0, 1.9, -1.0, np.nan, np.inf])
def test_retained_detection_class_ids_require_finite_non_negative_integers(
    decoder_kind: str, label: float
) -> None:
    """Integral floats are valid labels; lossy or non-finite coercion is rejected."""
    values = {
        "boxes": np.array([[[0, 0, 4, 4]]], dtype=np.float32),
        "scores": np.array([[0.9]], dtype=np.float32),
        "labels": np.array([[label]], dtype=np.float32),
        "masks": np.ones((1, 1, 4, 4), dtype=np.float32),
        "prediction": np.array([[[0, 0, 4, 4, 0.9, label]]], dtype=np.float32),
    }
    session = FakeSession(values, input_shape=[1, 3, 4, 4])
    if decoder_kind == "custom":
        decoder = _edge_custom_decoder("detection", {
            "prediction": "prediction",
            "layout": "rows",
            "bbox_start": 0,
            "bbox_encoding": "xyxy",
            "score_index": 4,
            "class_index": 5,
            "coordinate_space": "pixels",
            "score_activation": "none",
        })
    else:
        decoder = {"mode": "builtin", "decoder_id": decoder_kind}
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    if label == 1.0:
        result = onnx_predict.run_onnx_trial(session, image, decoder=decoder)
        assert result["detections"][0]["cls"] == 1
    else:
        with pytest.raises(onnx_predict.DecoderMappingError, match="class ID"):
            onnx_predict.run_onnx_trial(session, image, decoder=decoder)


@pytest.mark.parametrize("label", [1.0, 1.9, -0.9, np.nan, np.inf])
def test_semantic_class_ids_validate_before_integer_conversion(label: float) -> None:
    """Semantic class maps reject fractional/negative/non-finite IDs before argmax/casting."""
    session = FakeSession(
        {"mask": np.array([[label]], dtype=np.float32)}, input_shape=[1, 3, 4, 4]
    )
    decoder = _edge_custom_decoder(
        "semantic_segmentation", {"output": "mask", "format": "class_ids"}
    )
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    if label == 1.0:
        result = onnx_predict.run_onnx_trial(session, image, decoder=decoder)
        np.testing.assert_array_equal(result["mask"], np.ones((4, 4), dtype=np.int64))
    else:
        with pytest.raises(onnx_predict.DecoderMappingError, match="class ID"):
            onnx_predict.run_onnx_trial(session, image, decoder=decoder)


@pytest.mark.parametrize(
    ("shape", "class_axis"),
    [((1, 2, 2), None), ((1, 1, 2, 2), 1)],
)
def test_semantic_class_id_batch_and_singleton_channel_forms(
    shape: tuple[int, ...], class_axis: int | None
) -> None:
    """NHW needs no axis; N1HW removes only its declared singleton class axis."""
    values = np.array([0, 255, 1, 0], dtype=np.int64).reshape(shape)
    session = FakeSession({"mask": values}, input_shape=[1, 3, 2, 2])
    output = {"output": "mask", "format": "class_ids", "ignore_index": 255}
    if class_axis is not None:
        output["class_axis"] = class_axis
    decoder = _edge_custom_decoder("semantic_segmentation", output)
    decoder["input"].update(width=2, height=2)

    result = onnx_predict.run_onnx_trial(
        session, np.zeros((2, 2, 3), dtype=np.uint8), decoder=decoder
    )

    np.testing.assert_array_equal(result["mask"], np.array([[0, 255], [1, 0]]))
    assert result["ignore_index"] == 255


@pytest.mark.parametrize("activation", ["softmax", "none"])
def test_http_zero_class_classification_is_a_mapping_error(
    trial_client, monkeypatch: pytest.MonkeyPatch, activation: str
) -> None:
    """Zero classes never cause a 500 or a synthetic class-0 Top-1 response."""
    session = FakeSession(
        {"logits": np.zeros((1, 0), dtype=np.float32)}, input_shape=[1, 3, 4, 4]
    )
    monkeypatch.setattr(
        try_predict, "_load_onnx_session", lambda _ref: (session, "/models/empty.onnx")
    )
    monkeypatch.setattr(onnx_predict, "read_onnx_manifest", lambda _path: {})
    monkeypatch.setattr(onnx_predict, "read_onnx_names", lambda _path: None)
    decoder = _edge_custom_decoder("classification", {
        "output": "logits", "class_axis": 1, "activation": activation,
    })

    response = trial_client.post("/try/predict", json={
        "model_path": "empty.onnx",
        "image_base64": _image_base64(np.zeros((4, 4, 3), dtype=np.uint8)),
        "decoder": decoder,
    })

    assert response.status_code == 422
    assert response.json["code"] == "invalid_decoder_mapping"
    assert "class" in response.json["error"]


def test_http_zero_class_semantic_logits_is_a_mapping_error(
    trial_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Argmax is never attempted on a zero-length semantic class axis."""
    session = FakeSession(
        {"logits": np.zeros((1, 0, 2, 2), dtype=np.float32)},
        input_shape=[1, 3, 4, 4],
    )
    monkeypatch.setattr(
        try_predict, "_load_onnx_session", lambda _ref: (session, "/models/empty.onnx")
    )
    monkeypatch.setattr(onnx_predict, "read_onnx_manifest", lambda _path: {})
    monkeypatch.setattr(onnx_predict, "read_onnx_names", lambda _path: None)
    decoder = _edge_custom_decoder("semantic_segmentation", {
        "output": "logits", "format": "logits", "class_axis": 1,
    })

    response = trial_client.post("/try/predict", json={
        "model_path": "empty.onnx",
        "image_base64": _image_base64(np.zeros((4, 4, 3), dtype=np.uint8)),
        "decoder": decoder,
    })

    assert response.status_code == 422
    assert response.json["code"] == "invalid_decoder_mapping"


def test_custom_input_rejects_additional_required_graph_inputs_before_run() -> None:
    """Single-image mapping cannot silently omit another required graph input."""
    session = FakeSession(
        {"prediction": np.zeros((1, 1, 6), dtype=np.float32)},
        input_shape=[1, 3, 4, 4],
    )
    original_inputs = session.get_inputs
    session.get_inputs = lambda: [
        *original_inputs(),
        SimpleNamespace(name="aux", shape=[1], type="tensor(float)"),
    ]
    decoder = _edge_custom_decoder("detection", {
        "prediction": "prediction",
        "layout": "rows",
        "bbox_start": 0,
        "bbox_encoding": "xyxy",
        "score_index": 4,
        "class_index": 5,
        "coordinate_space": "pixels",
        "score_activation": "none",
    })

    with pytest.raises(onnx_predict.DecoderMappingError, match="exactly one"):
        onnx_predict.run_onnx_trial(
            session, np.zeros((4, 4, 3), dtype=np.uint8), decoder=decoder
        )
    assert session.feeds == []


def test_builtin_rejects_nonexistent_declared_input_tensor() -> None:
    """A misspelled manifest input never falls back to an unrelated graph input."""
    session = FakeSession(
        {"logits": np.array([[0.0, 1.0]], dtype=np.float32)},
        input_shape=[1, 3, 4, 4],
    )
    manifest = {"task": "classification", "input": {"name": "missing", "size": 4}}

    with pytest.raises(onnx_predict.DecoderMappingError, match="declared input"):
        onnx_predict.run_onnx_trial(
            session, np.zeros((4, 4, 3), dtype=np.uint8), manifest=manifest
        )
    assert session.feeds == []


def test_semantic_builtin_rejects_nonexistent_declared_logits_tensor() -> None:
    """A declared semantic tensor name is authoritative and never silently replaced."""
    session = FakeSession(
        {"logits": np.zeros((1, 2, 2, 2), dtype=np.float32)},
        input_shape=[1, 3, 2, 2],
    )
    manifest = {
        "task": "semantic_seg",
        "input": {"size": 2},
        "outputs": {"logits": {"tensor": "missing", "layout": "NCHW"}},
    }

    with pytest.raises(onnx_predict.DecoderMappingError, match="declared logits"):
        onnx_predict.run_onnx_trial(
            session, np.zeros((2, 2, 3), dtype=np.uint8), manifest=manifest
        )
    assert session.feeds == []


def test_custom_output_fields_are_validated_before_inference() -> None:
    """Missing output fields fail deterministically even when the graph cannot run."""
    session = FakeSession(
        {"prediction": np.zeros((1, 1, 6), dtype=np.float32)},
        input_shape=[1, 3, 4, 4],
    )

    def fail_run(_outputs, _feeds):
        raise AssertionError("inference must not start with an invalid output mapping")

    session.run = fail_run
    decoder = _edge_custom_decoder("detection", {
        "prediction": "prediction",
        "layout": "rows",
        "bbox_start": 0,
        "bbox_encoding": "xyxy",
        "class_index": 5,
        "coordinate_space": "pixels",
        "score_activation": "none",
    })

    with pytest.raises(onnx_predict.DecoderMappingError, match="score_index"):
        onnx_predict.run_onnx_trial(
            session, np.zeros((4, 4, 3), dtype=np.uint8), decoder=decoder
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("top_k", 1.9, "invalid top_k"),
        ("top_k", "1.9", "invalid top_k"),
        ("conf", np.nan, "invalid conf or iou"),
        ("conf", np.inf, "invalid conf or iou"),
        ("iou", np.nan, "invalid conf or iou"),
        ("iou", -np.inf, "invalid conf or iou"),
    ],
)
def test_http_rejects_lossy_or_non_finite_numeric_options_before_loading(
    trial_client,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: float,
    message: str,
) -> None:
    """Invalid request scalars are stable 400s and never reach model inference."""
    def fail_load(_ref):
        raise AssertionError("invalid numeric options must fail before model loading")

    monkeypatch.setattr(try_predict, "_load_onnx_session", fail_load)
    response = trial_client.post("/try/predict", json={
        "model_path": "unused.onnx",
        "image_base64": _image_base64(np.zeros((4, 4, 3), dtype=np.uint8)),
        field: value,
    })

    assert response.status_code == 400
    assert response.json == {"error": message}


@pytest.mark.parametrize("key", ["bad", "1.5", "-1"])
def test_http_rejects_invalid_name_indices_as_client_errors(
    trial_client, monkeypatch: pytest.MonkeyPatch, key: str
) -> None:
    """Malformed label-map keys return 400 instead of an internal conversion error."""
    session = FakeSession(
        {"logits": np.array([[0.0, 1.0]], dtype=np.float32)},
        input_shape=[1, 3, 4, 4],
    )
    monkeypatch.setattr(
        try_predict, "_load_onnx_session", lambda _ref: (session, "/models/names.onnx")
    )
    monkeypatch.setattr(onnx_predict, "read_onnx_manifest", lambda _path: {})
    monkeypatch.setattr(onnx_predict, "read_onnx_names", lambda _path: None)

    response = trial_client.post("/try/predict", json={
        "model_path": "names.onnx",
        "image_base64": _image_base64(np.zeros((4, 4, 3), dtype=np.uint8)),
        "names": {key: "label"},
    })

    assert response.status_code == 400
    assert response.json == {"error": "names keys must be non-negative integer indices"}
