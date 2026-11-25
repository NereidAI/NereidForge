# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Model-format auto-detection for the deploy-time ONNX conversion.

Only the *coarse family* (torch / tensorflow / paddle) is detected here, using
magic bytes and zip entry names on a small ranged S3 GET -- no ML framework is
imported. The supported torch exporter (torchvision / generic torch / custom)
is selected inside the convert pod. Removed AGPL Ultralytics and legacy YOLO
source trees are rejected rather than packaged into the converter.

A version-metadata `source` value always overrides detection:
    torchvision|torch|custom -> torch
    tensorflow|tf|keras      -> tensorflow
    paddle                   -> paddle
    auto|""                  -> detect from bytes
"""
from __future__ import annotations

import io
import logging
import posixpath
import zipfile

logger = logging.getLogger(__name__)

FORMAT_PYTORCH_CHECKPOINT = "pytorch-checkpoint"
FORMAT_KERAS = "keras"
FORMAT_TFLITE = "tflite"
FORMAT_GGUF = "gguf"
FORMAT_ASCEND_OM = "ascend-om"
FORMAT_PADDLE = "paddle"

# ARTIFACT_FORMATS is the authoritative upload/capability registry. Suffixes
# describe storage and layout only; capabilities name the workflows that the
# current product actually implements, so registration never implies deployability.
ARTIFACT_FORMATS = (
    {
        "id": FORMAT_PYTORCH_CHECKPOINT,
        "label": "PyTorch checkpoint",
        "extensions": (".pt", ".pth"),
        "primary_extensions": (".pt", ".pth"),
        "capabilities": ("download", "train", "convert", "trial", "deploy"),
        "compiled": False,
    },
    {
        "id": FORMAT_KERAS,
        "label": "Keras model",
        "extensions": (".h5", ".hdf5", ".keras"),
        "primary_extensions": (".h5", ".hdf5", ".keras"),
        "capabilities": ("download", "convert", "deploy"),
        "compiled": False,
    },
    {
        "id": "onnx",
        "label": "ONNX model",
        "extensions": (".onnx",),
        "primary_extensions": (".onnx",),
        "capabilities": ("download", "trial", "optimize", "deploy", "package"),
        "compiled": False,
    },
    {
        "id": "openvino-ir",
        "label": "OpenVINO IR",
        "extensions": (".xml", ".bin"),
        "primary_extensions": (".xml",),
        "capabilities": ("download", "deploy", "package"),
        "compiled": False,
    },
    {
        "id": FORMAT_TFLITE,
        "label": "TensorFlow Lite model",
        "extensions": (".tflite",),
        "primary_extensions": (".tflite",),
        "capabilities": ("download",),
        "compiled": False,
    },
    {
        "id": "tensorrt-plan",
        "label": "TensorRT engine",
        "extensions": (".plan", ".engine", ".trt"),
        "primary_extensions": (".plan", ".engine", ".trt"),
        "capabilities": ("download",),
        "compiled": True,
    },
    {
        "id": FORMAT_ASCEND_OM,
        "label": "Ascend OM model",
        "extensions": (".om",),
        "primary_extensions": (".om",),
        "capabilities": ("download",),
        "compiled": True,
    },
    {
        "id": FORMAT_GGUF,
        "label": "GGUF model",
        "extensions": (".gguf",),
        "primary_extensions": (".gguf",),
        "capabilities": ("download",),
        "compiled": False,
    },
    {
        "id": FORMAT_PADDLE,
        "label": "Paddle model",
        "extensions": (".pdmodel", ".pdiparams"),
        "primary_extensions": (".pdmodel",),
        "capabilities": ("download", "convert", "deploy"),
        "compiled": False,
    },
)
ARCHIVE_SUFFIXES = (".zip", ".tar.gz", ".tgz")


def artifact_format_catalog() -> dict:
    """artifact_format_catalog returns JSON-safe upload formats and real capabilities."""
    return {
        "formats": [
            {
                **definition,
                "extensions": list(definition["extensions"]),
                "primary_extensions": list(definition["primary_extensions"]),
                "capabilities": list(definition["capabilities"]),
            }
            for definition in ARTIFACT_FORMATS
        ],
        "archive_suffixes": list(ARCHIVE_SUFFIXES),
    }


def artifact_format_definition(format_id: str) -> dict | None:
    """artifact_format_definition resolves one stable format identifier."""
    return next((item for item in ARTIFACT_FORMATS if item["id"] == format_id), None)


def artifact_capability_permitted(metadata: dict, capability: str) -> bool:
    """artifact_capability_permitted applies recorded capabilities only to new versions.

    Missing or non-list metadata is the legacy compatibility path. A present
    list is authoritative, including an empty list, for every workflow owner.
    """
    capabilities = metadata.get("artifact_capabilities")
    return not isinstance(capabilities, list) or capability in capabilities


def artifact_format_for_primary(path: str) -> dict | None:
    """artifact_format_for_primary matches only primary suffixes, never companions."""
    lowered = path.lower()
    return next(
        (
            item for item in ARTIFACT_FORMATS
            if lowered.endswith(tuple(item["primary_extensions"]))
        ),
        None,
    )


def normalize_artifact_relative_path(value: str) -> str:
    """normalize_artifact_relative_path returns a portable safe relative object key.

    Absolute paths, parent traversal, backslashes, NULs, and empty results are
    rejected before a local extraction path or S3 key can be constructed.
    """
    raw = str(value or "").strip()
    if not raw or "\x00" in raw or "\\" in raw or raw.startswith("/"):
        raise ValueError("unsafe artifact path")
    normalized = posixpath.normpath(raw)
    if normalized in ("", ".", "..") or normalized.startswith("../"):
        raise ValueError("unsafe artifact path")
    return normalized


# File suffixes the conversion pipeline knows how to turn into ONNX.
CONVERTIBLE_SUFFIXES = (
    ".pt", ".pth", ".bin",
    ".h5", ".keras", ".zip",
    ".pdmodel",
)

_TORCH_SOURCES = {"torchvision", "torch", "custom"}
_TF_SOURCES = {"tensorflow", "tf", "keras"}

_HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
# How many leading bytes to fetch for magic/entry-name detection. A torch
# checkpoint stores archive/data.pkl first, so its pickle tokens are in here;
# .keras/SavedModel zips keep their JSON/protobuf entries early too.
_PROBE_BYTES = 8 * 1024 * 1024


def is_convertible(uri: str) -> bool:
    """True when the artifact suffix can be converted to ONNX.

    Note: OpenVINO IR weights also use the ``.bin`` suffix, but an IR version's
    primary artifact is always the ``.xml`` graph (the .bin is a companion), so
    callers must test `serving_format` first when both could apply.
    """
    path = (uri or "").split("?", 1)[0].rstrip("/").lower()
    return any(path.endswith(s) for s in CONVERTIBLE_SUFFIXES)


# Serving-side artifact formats. These decide (1) which runtime can serve the
# artifact and (2) how the predictor downloads it (single file vs an IR pair).
FORMAT_ONNX = "onnx"
FORMAT_OPENVINO_IR = "openvino-ir"          # .xml graph + same-basename .bin
FORMAT_TENSORRT_PLAN = "tensorrt-plan"     # serialized TensorRT engine

_TRT_PLAN_SUFFIXES = (".plan", ".engine", ".trt")


def serving_format(uri: str) -> str | None:
    """Classify an already-deployable artifact by its URI suffix.

    Returns None for training/source formats (.pt, ...) -- those go through
    ONNX conversion before reaching serving.
    """
    path = (uri or "").split("?", 1)[0].rstrip("/").lower()
    if path.endswith(".xml"):
        return FORMAT_OPENVINO_IR
    if path.endswith(_TRT_PLAN_SUFFIXES):
        return FORMAT_TENSORRT_PLAN
    if path.endswith(".onnx"):
        return FORMAT_ONNX
    return None


def ir_companion_bin(xml_uri: str) -> str | None:
    """Resolve the weights file URI for an OpenVINO IR graph (foo.xml -> foo.bin)."""
    if not xml_uri or serving_format(xml_uri) != FORMAT_OPENVINO_IR:
        return None
    head, sep, query = xml_uri.partition("?")
    return head[: -len(".xml")] + ".bin" + (sep and f"?{query}")


def _family_from_declared(source: str) -> str | None:
    s = (source or "").strip().lower()
    if s in ("", "auto"):
        return None
    if s in _TORCH_SOURCES:
        return "torch"
    if s in _TF_SOURCES:
        return "tensorflow"
    if s == "paddle":
        return "paddle"
    raise ValueError(
        f"unknown conversion source '{s}'; allowed: auto, {sorted(_TORCH_SOURCES | _TF_SOURCES | {'paddle'})}"
    )


def _family_from_bytes(name: str, head: bytes) -> str:
    suffix = name.lower()

    if suffix.endswith(".pdmodel") or suffix.endswith(".pdiparams"):
        return "paddle"

    if suffix.endswith((".h5",)) or head.startswith(_HDF5_MAGIC):
        # Keras HDF5 (and any HDF5 container carrying a keras model).
        return "tensorflow"

    if head.startswith((b"PK\x03\x04", b"PK\x05\x06")):
        # Zip container: modern Keras (.keras), a SavedModel bundle (.zip),
        # or a torch.save() archive. Entry names decide. The probe only
        # covers the first bytes, so the central directory (zip tail) is
        # usually absent on large artifacts: try namelist() for small zips,
        # then scan local file-header name fields in the raw head bytes.
        names: list[str] = []
        try:
            with zipfile.ZipFile(io.BytesIO(head)) as zf:
                names = zf.namelist()
        except zipfile.BadZipFile:
            names = []
        if not names:
            tokens = (b"config.json", b"saved_model.pb", b"data.pkl")
            names = [t.decode() for t in tokens if t in head]
        if any(n.endswith("config.json") and ("model" in n or "keras" in n) for n in names):
            return "tensorflow"
        if any(n.rstrip("/").endswith("saved_model.pb") for n in names):
            return "tensorflow"
        if any("data.pkl" in n for n in names):
            return "torch"
        # The probe window may cover no recognizable entry; decide from the
        # suffix instead of failing the deploy.
        if suffix.endswith((".keras", ".zip")):
            return "tensorflow"
        if suffix.endswith((".pt", ".pth", ".bin")):
            return "torch"

    # Raw torch pickle streams and anything .pt-ish that is not zipped.
    return "torch"


def detect_family(s3_client, bucket: str, key: str, declared: str = "") -> str:
    """Resolve the converter family for an S3 object.

    Args:
        s3_client: a boto3 S3 client able to read the artifact.
        bucket/key: the model object (the key also carries the file name).
        declared: version metadata `source` ("auto"/"" -> detect).
    Returns:
        "torch" | "tensorflow" | "paddle".
    """
    forced = _family_from_declared(declared)
    if forced:
        return forced

    name = key.rsplit("/", 1)[-1]
    try:
        head = s3_client.get_object(
            Bucket=bucket, Key=key, Range=f"bytes=0-{_PROBE_BYTES - 1}"
        )["Body"].read()
    except Exception as e:  # detection must not crash the deploy on read errors
        logger.warning("format probe read failed for s3://%s/%s: %s; falling back to suffix", bucket, key, e)
        head = b""
    family = _family_from_bytes(name, head)
    logger.info("model format auto-detected: s3://%s/%s -> %s", bucket, key, family)
    return family
