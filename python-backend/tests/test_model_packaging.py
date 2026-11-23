# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Tests for production nereid-model packaging orchestration and decoding.

Covers the container-side pure helpers (anchor decode, device comparison,
coordinate unmap, static-IO gate, reproducible tar), the versioned
request/result contract, and the single edge-verified Job profile. OpenVINO
itself is never imported here (the full container run is verified on-cluster).
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
BUILDERS = REPO / "builders"
if str(BUILDERS) not in sys.path:
    sys.path.insert(0, str(BUILDERS))

from app.clients.package_jobs import PackageJobRunner  # noqa: E402
from app.services.package_contract import (  # noqa: E402
    EDGE_VERIFIED,
    INT8,
    FP16,
    FP32,
    PACKAGE_REQUEST_SCHEMA,
    PACKAGE_RESULT_SCHEMA,
    build_package_request,
    parse_package_result,
    precision_enum,
)
from common import model_package as mp  # noqa: E402


def test_yolov8_decode_applies_confidence_nms_and_pixel_boxes():
    """Anchor decode keeps letterbox-pixel xyxy boxes after per-class NMS."""
    nc = 2
    s = 640.0
    raw = np.zeros((1, 4 + nc, 3), dtype=np.float32)
    # Anchor 0: mask box centered at (320,320) px, 128 px, score 0.9.
    raw[0, 0, 0] = 0.5 * s
    raw[0, 1, 0] = 0.5 * s
    raw[0, 2, 0] = 0.2 * s
    raw[0, 3, 0] = 0.2 * s
    raw[0, 4, 0] = 0.9
    # Anchor 1: duplicate mask box at the same location (NMS-suppressed).
    raw[0, :4, 1] = raw[0, :4, 0]
    raw[0, 4, 1] = 0.8
    # Anchor 2: no-mask box in another corner, score 0.7.
    raw[0, 0, 2] = 0.2 * s
    raw[0, 1, 2] = 0.2 * s
    raw[0, 2, 2] = 0.1 * s
    raw[0, 3, 2] = 0.1 * s
    raw[0, 5, 2] = 0.7

    dets = mp.decode_yolov8(raw, nc=nc, conf=0.25, iou_thr=0.45)

    assert len(dets) == 2
    by_class = {d["class_id"]: d for d in dets}
    assert set(by_class) == {0, 1}
    assert by_class[0]["score"] == 0.9
    # Raw decoded boxes stay in letterbox-pixel space (descriptor says so).
    assert by_class[0]["box"] == [256.0, 256.0, 384.0, 384.0]


def test_compare_dets_rejects_count_and_score_drift():
    """Device comparison accepts identical input and rejects count/score drift."""
    labels = ["mask", "no-mask"]
    ref = [{"class_id": 0, "score": 0.9, "box": [0.1, 0.1, 0.2, 0.2]}]
    delta, iou = mp.compare_dets(ref, ref, labels, score_delta_tol=0.02, box_iou_min=0.9)
    assert (delta, iou) == (0.0, 1.0)

    drifted = [{"class_id": 0, "score": 0.8, "box": [0.101, 0.101, 0.201, 0.201]}]
    try:
        mp.compare_dets(ref, drifted, labels, score_delta_tol=0.02, box_iou_min=0.9)
    except RuntimeError as e:
        assert "score drift" in str(e)
    else:
        raise AssertionError("score drift must reject the device package")

    try:
        mp.compare_dets(ref, [], labels, score_delta_tol=0.02, box_iou_min=0.9)
    except RuntimeError as e:
        assert "detection count mismatch" in str(e)
    else:
        raise AssertionError("detection count drift must reject the device package")


def test_source_normalized_box_matches_unmap_formula():
    """Letterbox-pixel xyxy unmap is the inverse of the evaluator letterbox.

    Sample verified by hand: 1200x630 frame letterboxed into 640 gives
    r=0.5333 and pad_y=152; a box covering the canvas center maps back into
    the normalized source frame (regression for the NI UnmapBox contract).
    """
    w0, h0, r, pad_x, pad_y = 1200, 630, min(640 / 1200, 640 / 630), 0, 152
    # A 100 px letterbox box around the padded-frame center.
    cx, cy = 320.0, 320.0
    box = [cx - 50, cy - 50, cx + 50, cy + 50]
    x1, y1, x2, y2 = mp.source_normalized_box(box, (w0, h0, r, pad_x, pad_y))
    assert 0.0 <= x1 <= x2 <= 1.0 and 0.0 <= y1 <= y2 <= 1.0
    # Unmap arithmetic: (box - pad) / r then divide by source size.
    assert x1 == round((box[0] - pad_x) / r / w0, 6)
    assert y1 == round((box[1] - pad_y) / r / h0, 6)


def test_static_io_gate_rejects_dynamic_tensors():
    """D7: any dynamic input or output dimension blocks packaging."""
    static = [{"name": "images", "shape": [1, 3, 640, 640]}]
    out = [{"name": "output0", "shape": [1, 6, 8400]}]
    mp.assert_static_io(static, out)  # no raise

    for bad_in, bad_out in [
        ([{"name": "images", "shape": [-1, 3, 640, 640]}], out),
        (static, [{"name": "output0", "shape": [1, 6, -1]}]),
    ]:
        try:
            mp.assert_static_io(bad_in, bad_out)
        except mp.PackageError as e:
            assert e.code == "UNSUPPORTED_IR"
        else:
            raise AssertionError("dynamic tensors must fail with UNSUPPORTED_IR")


def _write_face_onnx(path: Path, family: str, batch=1) -> None:
    """Create a small graph with the exact tensor contract under test."""
    import onnx
    from onnx import helper, numpy_helper

    is_yunet = family == "yunet"
    input_name = "input" if is_yunet else "data"
    spatial = 64 if is_yunet else 112
    graph_input = helper.make_tensor_value_info(
        input_name, onnx.TensorProto.FLOAT, [batch, 3, spatial, spatial],
    )
    nodes = []
    outputs = []
    contracts = []
    if is_yunet:
        for stride in (8, 16, 32):
            anchors = (spatial // stride) ** 2
            contracts.extend([
                (f"cls_{stride}", [1, anchors, 1]),
                (f"obj_{stride}", [1, anchors, 1]),
                (f"bbox_{stride}", [1, anchors, 4]),
                (f"kps_{stride}", [1, anchors, 10]),
            ])
    else:
        contracts.append(("fc1", [1, 128]))
    for name, shape in contracts:
        nodes.append(helper.make_node(
            "Constant", [], [name],
            value=numpy_helper.from_array(np.zeros(shape, dtype=np.float32)),
        ))
        outputs.append(helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape))
    graph = helper.make_graph(nodes, family, [graph_input], outputs)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    onnx.checker.check_model(model)
    onnx.save(model, path)


def _face_package_spec(family: str) -> dict:
    """Build the parsed request fields needed by direct ONNX descriptor tests."""
    return {
        "request_id": "task-face",
        "precision": "FP32",
        "model": {
            "id": f"m-{family}", "version": "2026.11", "revision": "mv-face",
            "task": "detection" if family == "yunet" else "embedding",
        },
        "source": {
            "onnx_uri": f"s3://bucket/models/{family}/model.onnx",
            "family_hint": family,
        },
        "policy": {"conf": 0.25, "iou": 0.3},
    }


def _assert_onnx_triton_contract(descriptor: dict, model_name: str) -> None:
    """_assert_onnx_triton_contract checks managed Triton without OVMS constraints."""
    assert descriptor["artifact"]["format"] == "onnx"
    assert descriptor["runtime"] == {
        "backend": "kserve",
        "model_name": model_name,
        "repository_path": f"repository/{model_name}",
        "server": {"family": "triton"},
    }
    assert "hardware" not in descriptor


def test_yunet_onnx_descriptor_preserves_all_heads_and_five_point_semantics(tmp_path):
    """YuNet uses its twelve probability heads, never detection.anchors."""
    artifact = tmp_path / "yunet.onnx"
    _write_face_onnx(artifact, "yunet")

    descriptor = mp.build_onnx_descriptor(
        _face_package_spec("yunet"), str(artifact), "repository/m-yunet/1/model.onnx",
    )

    _assert_onnx_triton_contract(descriptor, "m-yunet")
    assert descriptor["schema"] == "nereid-model/1"
    assert descriptor["artifact"] == {
        "format": "onnx",
        "precision": "FP32",
        "files": [{
            "path": "repository/m-yunet/1/model.onnx",
            "bytes": artifact.stat().st_size,
            "sha256": mp.sha256_file(str(artifact)),
        }],
    }
    assert descriptor["runtime"]["repository_path"] == "repository/m-yunet"
    assert descriptor["io"]["inputs"] == [{
        "name": "input", "shape": [1, 3, 64, 64], "layout": "NCHW",
        "dtype": "FP32", "color_order": "BGR", "role": "source_image",
        "preprocess": {
            "resize": "letterbox", "scale": "1", "mean": [0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0], "pad_value": 0,
        },
    }]
    assert len(descriptor["io"]["outputs"]) == 12
    by_name = {item["name"]: item for item in descriptor["io"]["outputs"]}
    assert by_name["cls_8"]["role"] == "cls"
    assert by_name["cls_8"]["stride"] == 8
    assert by_name["bbox_16"]["shape"] == [1, 16, 4]
    assert by_name["kps_32"]["shape"] == [1, 4, 10]
    decoder = descriptor["decoder"]
    assert decoder["kind"] == "detection.yunet"
    assert decoder["activation"] == "probability"
    assert decoder["score"]["formula"] == "sqrt(clamp(cls,0,1)*clamp(obj,0,1))"
    assert decoder["boxes"]["coordinate_space"] == "model-input-pixels"
    assert decoder["landmarks"]["count"] == 5
    assert decoder["landmarks"]["order"] == [
        "right_eye", "left_eye", "nose_tip", "right_mouth_corner", "left_mouth_corner",
    ]
    assert [item["stride"] for item in decoder["strides"]] == [8, 16, 32]
    assert descriptor["verification"]["numeric_status"] == "not_run"


def test_sface_onnx_descriptor_declares_alignment_and_normalized_128d_vector(tmp_path):
    """SFace consumes one aligned 112x112 face and exposes raw fc1 plus decoder semantics."""
    artifact = tmp_path / "sface.onnx"
    _write_face_onnx(artifact, "sface")

    descriptor = mp.build_onnx_descriptor(
        _face_package_spec("sface"), str(artifact), "repository/m-sface/1/model.onnx",
    )

    _assert_onnx_triton_contract(descriptor, "m-sface")
    model_input = descriptor["io"]["inputs"][0]
    assert model_input["name"] == "data"
    assert model_input["shape"] == [1, 3, 112, 112]
    assert model_input["role"] == "five_point_aligned_face"
    assert model_input["color_order"] == "RGB"
    assert model_input["preprocess"]["scale"] == "1"
    assert model_input["preprocess"]["alignment"]["template"] == [
        [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
        [41.5493, 92.3655], [70.7299, 92.2041],
    ]
    assert descriptor["io"]["outputs"] == [{
        "name": "fc1", "shape": [1, 128], "layout": "NC", "dtype": "FP32",
        "role": "face_embedding", "dimensions": 128,
        "value_semantics": "raw_sface_embedding",
    }]
    assert descriptor["decoder"] == {
        "kind": "embeddings.vector", "output": "fc1", "dimensions": 128,
        "l2_normalize": True, "max_dims": 128,
        "value_semantics": "l2_normalized_128d_face_embedding",
    }
    assert descriptor["verification"]["numeric_status"] == "not_run"


def test_onnx_descriptor_rejects_dynamic_batch(tmp_path):
    """Direct face packages never silently convert a dynamic-batch graph."""
    artifact = tmp_path / "sface-dynamic.onnx"
    _write_face_onnx(artifact, "sface", batch="batch")
    try:
        mp.build_onnx_descriptor(
            _face_package_spec("sface"), str(artifact), "repository/m-sface/1/model.onnx",
        )
    except mp.PackageError as exc:
        assert exc.code == "UNSUPPORTED_ONNX"
    else:
        raise AssertionError("dynamic batch must be rejected")


class _FakePackageS3:
    """Store package sources and products by bucket/key without a live service."""

    def __init__(self):
        self.objects = {}

    @staticmethod
    def _key(bucket, key):
        return f"{bucket}/{key}"

    def get_object(self, Bucket, Key):
        return {"Body": io.BytesIO(self.objects[self._key(Bucket, Key)])}

    def download_file(self, Bucket, Key, destination):
        Path(destination).write_bytes(self.objects[self._key(Bucket, Key)])

    def upload_file(self, source, Bucket, Key):
        self.objects[self._key(Bucket, Key)] = Path(source).read_bytes()

    def put_object(self, Bucket, Key, Body, ContentType=""):
        self.objects[self._key(Bucket, Key)] = Body


def test_direct_onnx_run_publishes_standard_self_contained_package(tmp_path, monkeypatch):
    """The direct path publishes identical staging/tar bytes and truthful result metadata."""
    artifact = tmp_path / "sface.onnx"
    _write_face_onnx(artifact, "sface")
    s3 = _FakePackageS3()
    source_uri = "s3://bucket/models/m-abc/2026.11/model.onnx"
    s3.objects["bucket/models/m-abc/2026.11/model.onnx"] = artifact.read_bytes()
    kwargs = _minimal_request_kwargs(precision=FP32)
    kwargs["model"].update({"version": "2026.11", "task": "embedding"})
    kwargs["source"] = {"onnx_uri": source_uri, "family_hint": "sface"}
    kwargs["output"] = {
        "package_uri": "s3://bucket/packages/m-abc/nereid-model-fp32.tar.gz",
        "sidecar_uri": "s3://bucket/packages/m-abc/nereid-model-fp32.sidecar.json",
        "result_uri": "s3://bucket/packages/m-abc/results/task-123.result.json",
        "staging_uri": "s3://bucket/packages/m-abc/staging/fp32/",
    }
    request = build_package_request(**kwargs)
    s3.objects["bucket/requests/task-123.json"] = json.dumps(request).encode()
    monkeypatch.setattr(mp, "s3_client", lambda: s3)

    mp.run("s3://bucket/requests/task-123.json")

    package_bytes = s3.objects["bucket/packages/m-abc/nereid-model-fp32.tar.gz"]
    with tarfile.open(fileobj=io.BytesIO(package_bytes), mode="r:gz") as archive:
        assert set(archive.getnames()) == {
            "nereid-model.json", "repository/m-abc/1/model.onnx",
        }
        descriptor = json.load(archive.extractfile("nereid-model.json"))
        packaged_payload = archive.extractfile("repository/m-abc/1/model.onnx").read()
    assert packaged_payload == artifact.read_bytes()
    _assert_onnx_triton_contract(descriptor, "m-abc")
    assert descriptor["model"] == {
        "id": "m-abc", "version": "2026.11", "revision": "67530",
        "task": "embedding", "family": "sface",
    }
    assert descriptor["artifact"]["files"][0]["sha256"] == hashlib.sha256(
        artifact.read_bytes(),
    ).hexdigest()
    assert descriptor["verification"]["numeric_status"] == "not_run"
    assert s3.objects[
        "bucket/packages/m-abc/staging/fp32/repository/m-abc/1/model.onnx"
    ] == packaged_payload
    sidecar = json.loads(s3.objects["bucket/packages/m-abc/nereid-model-fp32.sidecar.json"])
    result = json.loads(s3.objects["bucket/packages/m-abc/results/task-123.result.json"])
    package_sha = hashlib.sha256(package_bytes).hexdigest()
    assert sidecar["package_sha256"] == package_sha
    assert sidecar["descriptor"] == descriptor
    assert result["status"] == "succeeded"
    assert result["package"] == {
        "uri": kwargs["output"]["package_uri"],
        "bytes": len(package_bytes), "sha256": package_sha,
    }
    assert result["verification"]["numeric_status"] == "not_run"


def test_ir_run_retains_ovms_runtime_and_package_semantics(tmp_path, monkeypatch):
    """test_ir_run_retains_ovms_runtime_and_package_semantics preserves the IR contract."""
    kwargs = _minimal_request_kwargs()
    kwargs["source"]["supporting_artifacts"] = []
    request = build_package_request(**kwargs)
    s3 = _FakePackageS3()
    s3.objects.update({
        "bucket/requests/task-123.json": json.dumps(request).encode(),
        "bucket/models/m.xml": b"<xml/>",
        "bucket/models/m.bin": b"weights",
        "bucket/datasets/coco/positive.jpg": b"sample",
    })
    monkeypatch.setattr(mp, "s3_client", lambda: s3)
    monkeypatch.setattr(
        mp, "resolve_labels_and_sample",
        lambda *_: (["mask"], "s3://bucket/datasets/coco/positive.jpg"),
    )

    raw = np.array([[[320.0], [320.0], [128.0], [128.0], [0.9]]], dtype=np.float32)

    class _Compiled:
        def infer_new_request(self, _inputs):
            return {"output0": raw.copy()}

    class _Core:
        def read_model(self, model_path):
            return model_path

        def compile_model(self, _model, _device):
            return _Compiled()

    fake_ov = types.SimpleNamespace(Core=_Core, __version__="2025.1")
    inputs = [{"name": "images", "shape": [1, 3, 640, 640], "dtype": "FP32"}]
    outputs = [{"name": "output0", "shape": [1, 5, 1], "dtype": "FP32"}]
    monkeypatch.setattr(
        mp, "inspect_ir", lambda *_: (inputs, outputs, ["mask"], fake_ov),
    )
    monkeypatch.setitem(
        sys.modules, "cv2",
        types.SimpleNamespace(imread=lambda _: np.zeros((640, 640, 3), dtype=np.uint8)),
    )
    monkeypatch.setattr(
        mp, "letterbox",
        lambda _image, size: (
            np.zeros((1, 3, size, size), dtype=np.float32), (640, 640, 1.0, 0, 0),
        ),
    )
    work = tmp_path / "ir-work"
    work.mkdir()
    monkeypatch.setattr(mp.tempfile, "mkdtemp", lambda **_kwargs: str(work))

    mp.run("s3://bucket/requests/task-123.json")

    package_bytes = s3.objects["bucket/packages/m.tar.gz"]
    with tarfile.open(fileobj=io.BytesIO(package_bytes), mode="r:gz") as archive:
        members = {name: archive.extractfile(name).read() for name in archive.getnames()}
    assert set(members) == {
        "nereid-model.json",
        "labels.txt",
        "repository/m-abc/1/model.bin",
        "repository/m-abc/1/model.xml",
        "samples/positive.jpg",
    }
    descriptor = json.loads(members["nereid-model.json"])
    assert descriptor["artifact"]["format"] == "openvino-ir"
    assert descriptor["artifact"]["precision"] == "FP16"
    assert descriptor["runtime"] == {
        "backend": "kserve",
        "model_name": "m-abc",
        "repository_path": "repository/m-abc",
        "server": {"family": "ovms", "min_version": "2025.1"},
    }
    assert descriptor["hardware"] == {"constraints": {"openvino": ">=2025.1"}}
    assert descriptor["model"] == {
        "id": "m-abc",
        "version": "job-1-ir-fp16",
        "revision": "67530",
        "task": "detection",
        "family": "yolov8",
        "labels_file": "labels.txt",
    }
    assert descriptor["io"] == {
        "inputs": [{
            "name": "images", "shape": [1, 3, 640, 640], "layout": "NCHW",
            "dtype": "FP32", "color_order": "RGB",
            "preprocess": {
                "resize": "letterbox", "scale": "1/255", "mean": [0.0, 0.0, 0.0],
                "std": [1.0, 1.0, 1.0], "pad_value": 114,
            },
        }],
        "outputs": [{
            "name": "output0", "shape": [1, 5, 1], "layout": "NCA", "dtype": "FP32",
        }],
    }
    assert descriptor["decoder"] == {
        "kind": "detection.anchors", "output": "output0", "threshold": 0.25,
        "nms": {"iou": 0.45, "agnostic": False, "top_k": 300},
    }
    for item in descriptor["artifact"]["files"]:
        payload = members[item["path"]]
        assert item["bytes"] == len(payload)
        assert item["sha256"] == hashlib.sha256(payload).hexdigest()


def test_reproducible_tar_is_byte_stable_and_normalized(tmp_path):
    """D10: tar bytes are independent of mtime/uid/gid and member order is fixed."""
    root = tmp_path / "pkg"
    (root / "sub").mkdir(parents=True)
    for rel, body in [("a.txt", b"alpha"), ("sub/b.bin", b"\x00\x01\x02")]:
        p = root / rel
        p.write_bytes(body)
        import os
        os.utime(p, (1_700_000_000, 1_700_000_000))
    rels = ["a.txt", "sub/b.bin"]
    first = mp.reproducible_tar(str(root), rels)
    # Rebuild after "touching" the files; bytes must stay identical.
    import os
    for rel in rels:
        os.utime(root / rel, (1_000_000_000, 1_000_000_000))
    second = mp.reproducible_tar(str(root), rels)
    assert first == second

    with tarfile.open(fileobj=io.BytesIO(first), mode="r:gz") as tar:
        for member in tar.getmembers():
            assert member.mtime == 0
            assert member.uid == member.gid == 0
            assert member.uname == member.gname == ""


def test_non_finite_output_is_rejected():
    """NaN/Inf model output is a verification failure, never a shipped package."""
    compiled = types.SimpleNamespace(
        infer_new_request=lambda inputs: {"out": np.array([[[np.nan]]], dtype=np.float32)}
    )
    try:
        mp.run_infer(compiled, np.zeros((1, 3, 8, 8), dtype=np.float32))
    except RuntimeError as e:
        assert "non-finite" in str(e)
    else:
        raise AssertionError("NaN output must fail verification")


def test_packager_selects_first_non_crowd_positive_coco_sample():
    """The fixed golden sample is the first non-crowd positive annotation."""
    ann = {
        "images": [
            {"id": 1, "file_name": "crowd.jpg"},
            {"id": 2, "file_name": "images/positive.jpg"},
        ],
        "annotations": [
            {"image_id": 1, "iscrowd": 1, "bbox": [1, 2, 30, 40]},
            {"image_id": 2, "iscrowd": 0, "bbox": [4, 5, 20, 25]},
        ],
    }

    assert mp.first_annotated_image("s3://bucket/dataset/coco", ann) == (
        "s3://bucket/dataset/coco/images/positive.jpg"
    )


class _FakeDatasetS3:
    """Minimal S3 dataset stub with deterministic listing and in-memory objects."""

    def __init__(self, prefix, objects):
        self.prefix = prefix.rstrip("/") + "/"
        self.objects = {self.prefix + key: value for key, value in objects.items()}

    def list_objects_v2(self, Bucket, Prefix, **kwargs):
        """Return every matching key; production pagination is not needed here."""
        keys = sorted(key for key in self.objects if key.startswith(Prefix))
        return {"Contents": [{"Key": key} for key in keys], "IsTruncated": False}

    def get_object(self, Bucket, Key):
        """Expose one object through boto3's streaming response shape."""
        return {"Body": io.BytesIO(self.objects[Key])}


def test_packager_resolves_canonical_yolo_labels_and_positive_sample():
    """Canonical split YOLO input needs no synthetic COCO annotation export."""
    dataset_uri = "s3://bucket/datasets/ds-mask/v1"
    s3 = _FakeDatasetS3("datasets/ds-mask/v1", {
        "data.yaml": b"train: /data/train/images\nval: /data/val/images\nnames:\n  0: mask\n  1: no-mask\nnc: 2\n",
        "test/images/unlabeled.jpg": b"test",
        "train/images/empty.jpg": b"empty",
        "train/labels/empty.txt": b"\n",
        "val/images/positive.jpg": b"positive",
        "val/labels/positive.txt": b"1 0.5 0.5 0.25 0.5\n",
    })

    labels, sample_uri = mp.resolve_labels_and_sample(s3, dataset_uri)

    assert labels == ["mask", "no-mask"]
    assert sample_uri == f"{dataset_uri}/val/images/positive.jpg"


def test_packager_retains_coco_fallback_when_yolo_manifest_is_absent():
    """Legacy COCO versions keep their existing labels and sample selection."""
    dataset_uri = "s3://bucket/datasets/ds-coco/v1"
    s3 = _FakeDatasetS3("datasets/ds-coco/v1", {
        "annotations.json": (
            b'{"images":[{"id":7,"file_name":"positive.jpg"}],'
            b'"annotations":[{"image_id":7,"bbox":[1,2,30,40]}],'
            b'"categories":[{"id":1,"name":"mask"}]}'
        ),
        "images/positive.jpg": b"positive",
    })

    labels, sample_uri = mp.resolve_labels_and_sample(s3, dataset_uri)

    assert labels == ["mask"]
    assert sample_uri == f"{dataset_uri}/images/positive.jpg"


def test_packager_normalizes_family_hint_without_core_decoder_knowledge():
    """Architecture hints collapse onto the decoder families the job supports."""
    assert mp.normalize_family("yolov8n.pt") == "yolov8"
    assert mp.normalize_family("YOLOv5s") == "yolov5"
    assert mp.normalize_family("") == "yolov8"


def test_packager_rejects_bad_precision_spec():
    """parse_spec validates the precision enum before downloading anything."""
    try:
        mp.parse_spec({"schema": mp.REQUEST_SCHEMA, "precision": "FP32"})
    except mp.PackageError as e:
        assert e.code == "BAD_REQUEST"
    else:
        raise AssertionError("FP32 is not a packaged precision (fallback only)")


def test_precision_enum_normalizes_api_form():
    """The API uses lowercase keys; the descriptor enum is uppercase."""
    assert precision_enum("fp16") == FP16
    assert precision_enum("INT8") == INT8
    try:
        precision_enum("gpu")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown precision must be rejected")


def _minimal_request_kwargs(precision=FP16):
    """Shared valid request kwargs for the contract tests."""
    return dict(
        request_id="task-123",
        tenant_id="default",
        model={
            "id": "m-abc",
            "version": "job-1-ir-fp16",
            "revision": "67530",
            "task": "detection",
        },
        source={
            "ir_xml_uri": "s3://bucket/models/m.xml",
            "ir_bin_uri": "s3://bucket/models/m.bin",
            "dataset_version_uri": "s3://bucket/datasets/coco/",
            "family_hint": "yolov8",
            "supporting_artifacts": [
                {"type": "optimization_metrics", "uri": "s3://bucket/m.metrics.json"}
            ],
        },
        precision=precision,
        output={
            "package_uri": "s3://bucket/packages/m.tar.gz",
            "sidecar_uri": "s3://bucket/packages/m.sidecar.json",
            "result_uri": "s3://bucket/packages/task-123.result.json",
            "staging_uri": "s3://bucket/packages/staging/fp16/",
        },
    )


def test_package_request_contract_is_versioned_and_contains_only_uris():
    """The request carries immutable S3 URIs plus the single edge profile."""
    request = build_package_request(**_minimal_request_kwargs())

    assert request["schema"] == PACKAGE_REQUEST_SCHEMA
    assert request["precision"] == FP16
    assert request["hardware"] == {
        "profile": EDGE_VERIFIED,
        "require_numeric_verification": True,
        "verify_devices": ["CPU", "GPU"],
    }
    assert request["policy"] == {
        "confidence_threshold": 0.25,
        "nms_iou_threshold": 0.45,
        "score_delta": 0.02,
        "box_iou": 0.9,
    }
    assert "sample_uri" not in request["source"]
    assert "labels" not in request["source"]
    assert "imgsz" not in request
    assert request["output"]["staging_uri"].endswith("/")


def test_package_request_keeps_ir_precision_rules_and_validates_uri():
    """Existing IR packages remain FP16/INT8 with numeric verification."""
    kwargs = _minimal_request_kwargs()
    kwargs["precision"] = "fp32"
    try:
        build_package_request(**kwargs)
    except ValueError:
        pass
    else:
        raise AssertionError("FP32 IR packaging must be rejected")

    kwargs = _minimal_request_kwargs()
    kwargs["source"]["ir_xml_uri"] = "http://example.local/m.xml"
    try:
        build_package_request(**kwargs)
    except ValueError as e:
        assert "ir_xml_uri" in str(e)
    else:
        raise AssertionError("request source must use immutable S3 URIs")


def test_onnx_package_request_is_fp32_structural_and_dataset_independent():
    """A direct ONNX package carries no IR/dataset fields and requests no GPU."""
    kwargs = _minimal_request_kwargs(precision=FP32)
    kwargs["source"] = {
        "onnx_uri": "s3://bucket/models/sface/v1/model.onnx",
        "family_hint": "sface",
    }

    request = build_package_request(**kwargs)

    assert request["precision"] == FP32
    assert request["source"] == {
        "onnx_uri": "s3://bucket/models/sface/v1/model.onnx",
        "family_hint": "sface",
        "supporting_artifacts": [],
    }
    assert request["hardware"] == {
        "profile": "onnx-structural",
        "require_numeric_verification": False,
        "verify_devices": [],
    }
    assert "dataset_version_uri" not in request["source"]
    parsed = mp.parse_spec(request)
    assert parsed["source"]["onnx_uri"].endswith("/model.onnx")
    assert parsed["precision"] == FP32


def test_parse_package_result_exposes_precision_and_staging():
    """The generic result summary carries precision, tar and staging locations."""
    parsed = parse_package_result({
        "schema": PACKAGE_RESULT_SCHEMA,
        "request_id": "task-123",
        "precision": "INT8",
        "status": "succeeded",
        "error": {"code": "", "message": ""},
        "package": {
            "uri": "s3://bucket/m.tar.gz",
            "bytes": 123,
            "sha256": "a" * 64,
        },
        "sidecar_uri": "s3://bucket/m.sidecar.json",
        "staging": {"uri": "s3://bucket/packages/staging/int8/", "files": []},
        "verification": {
            "passed": True,
            "detection_count": 3,
            "devices": [
                {"device": "intel:CPU", "status": "passed"},
                {"device": "intel:GPU", "status": "passed"},
            ],
        },
    })

    assert parsed["status"] == "succeeded"
    assert parsed["precision"] == INT8
    assert parsed["package"] == {
        "uri": "s3://bucket/m.tar.gz", "bytes": 123, "sha256": "a" * 64,
    }
    assert parsed["staging_uri"] == "s3://bucket/packages/staging/int8/"
    assert parsed["verification"]["detection_count"] == 3


def test_parse_failed_package_result_preserves_machine_code():
    """A failed result keeps the machine-readable error code for the record."""
    parsed = parse_package_result({
        "schema": PACKAGE_RESULT_SCHEMA,
        "request_id": "task-123",
        "precision": "INT8",
        "status": "failed",
        "error": {"code": "NUMERIC_MISMATCH", "message": "GPU score drift"},
        "verification": {"passed": False, "devices": []},
    })

    assert parsed["status"] == "failed"
    assert parsed["error"]["code"] == "NUMERIC_MISMATCH"
    assert not parsed["package"]


class _FakeBatchApi:
    """In-memory BatchV1Api stub capturing submitted Job manifests."""

    def __init__(self):
        self.created = []

    def create_namespaced_job(self, namespace, body, **kwargs):
        self.created.append((namespace, body))
        return types.SimpleNamespace(metadata=types.SimpleNamespace(name=body["metadata"]["name"]))


def test_package_job_runner_uses_the_single_edge_verified_profile():
    """Every precision runs on the XPU image with xe:1 and root (pit 174/177)."""
    batch = _FakeBatchApi()
    runner = PackageJobRunner(namespace="nereidforge", batch_api=batch)

    runner.submit("s3://bucket/task-fp16.request.json", EDGE_VERIFIED, job_name="pkg-fp16")

    body = batch.created[0][1]
    container = body["spec"]["template"]["spec"]["containers"][0]

    assert container["image"].endswith("train-evaluator-xpu:0.0.0-dev")
    assert container["command"] == ["/opt/xpu/bin/python", "/app/model_package.py"]
    assert container["args"] == ["--spec-uri", "s3://bucket/task-fp16.request.json"]
    assert container["securityContext"] == {"runAsUser": 0, "runAsGroup": 0}
    assert container["resources"]["limits"]["gpu.intel.com/xe"] == "1"
    assert body["spec"]["template"]["spec"]["restartPolicy"] == "Never"


def test_onnx_package_job_uses_cpu_structural_profile_without_gpu():
    """Structural ONNX packaging neither requests nor claims an Intel GPU."""
    batch = _FakeBatchApi()
    runner = PackageJobRunner(namespace="nereidforge", batch_api=batch)

    runner.submit(
        "s3://bucket/task-fp32.request.json", "onnx-structural", job_name="pkg-fp32",
    )

    body = batch.created[0][1]
    container = body["spec"]["template"]["spec"]["containers"][0]
    assert container["image"].endswith("train-evaluator:0.0.0-dev")
    assert container["command"] == ["python", "/app/model_package.py"]
    assert "gpu.intel.com/xe" not in container["resources"]["requests"]
    assert "gpu.intel.com/xe" not in container["resources"]["limits"]
    assert "securityContext" not in container
