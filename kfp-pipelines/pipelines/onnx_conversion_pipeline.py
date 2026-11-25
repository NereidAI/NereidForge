# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""ONNX Conversion Pipeline

Convert model artifacts on S3 to ONNX. The coarse converter `family`
(torch / tensorflow / paddle) is detected from the artifact bytes by the
backend before submission and selects the matching branch + converter image,
so no framework image ever carries another framework's stack. Within the
torch branch, `source` (auto/torchvision/torch/custom) selects the exporter
inside convert_step; Paddle inference models always use convert-pp.
"""

import importlib.util
import os

from kfp import dsl

from images import nf_image

VALIDATOR_IMAGE = nf_image("runtime")


def _load_component(filename: str):
    """Load a standalone KFP component file from ../components by path."""
    path = os.path.join(
        os.path.dirname(__file__), "..", "components", filename
    )
    spec = importlib.util.spec_from_file_location(
        f"nf_component_{filename.rsplit('.', 1)[0]}", path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


convert_pt_to_onnx = _load_component("convert_step.py").convert_pt_to_onnx
convert_tf_to_onnx = _load_component("convert_tf_step.py").convert_tf_to_onnx
convert_paddle_to_onnx = _load_component("convert_paddle_step.py").convert_paddle_to_onnx


@dsl.component(base_image=VALIDATOR_IMAGE)
def validate_s3_credentials(s3_access_key: str, s3_secret_key: str) -> None:
    """Reject blank object-store credentials before any converter starts."""
    if not (s3_access_key or "").strip() or not (s3_secret_key or "").strip():
        raise ValueError("s3_access_key and s3_secret_key must be non-empty")


@dsl.pipeline(
    name="onnx-conversion",
    description="Convert PyTorch / TensorFlow / Paddle models to ONNX format",
)
def onnx_conversion_pipeline(
    src_uri: str,
    dest_uri: str,
    s3_access_key: str,
    s3_secret_key: str,
    s3_endpoint: str = "http://seaweedfs-s3.nereidforge.svc.cluster.local:8333",
    artifact_id: str = "onnx-convert",
    family: str = "torch",
    source: str = "auto",
    architecture: str = "",
    custom_uri: str = "",
    imgsz: int = 640,
    opset_version: int = 17,
):
    """Convert a model artifact to ONNX via the family-specific converter."""
    credential_check = validate_s3_credentials(
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key,
    )
    credential_check.set_display_name("Validate S3 credentials")
    credential_check.set_caching_options(False)

    # Exactly one branch runs; each branch uploads the same dest_uri, so the
    # predictor and backend only ever see the resulting ONNX.
    with dsl.If(family == "torch", name="torch-family"):
        step = convert_pt_to_onnx(
            src_uri=src_uri,
            dest_uri=dest_uri,
            s3_endpoint=s3_endpoint,
            s3_access_key=s3_access_key,
            s3_secret_key=s3_secret_key,
            artifact_id=artifact_id,
            source=source,
            architecture=architecture,
            custom_uri=custom_uri,
            imgsz=imgsz,
            opset_version=opset_version,
        )
        step.after(credential_check)
        step.set_display_name("Convert PyTorch to ONNX")
        # Writes a shared dest_uri in S3: a cache hit after the object was
        # deleted would silently strand the deploy with no artifact.
        step.set_caching_options(False)

    with dsl.Elif(family == "tensorflow", name="tensorflow-family"):
        step = convert_tf_to_onnx(
            src_uri=src_uri,
            dest_uri=dest_uri,
            s3_endpoint=s3_endpoint,
            s3_access_key=s3_access_key,
            s3_secret_key=s3_secret_key,
            artifact_id=artifact_id,
            opset_version=opset_version,
        )
        step.after(credential_check)
        step.set_display_name("Convert TensorFlow/Keras to ONNX")
        step.set_caching_options(False)

    with dsl.Else(name="paddle-family"):
        step = convert_paddle_to_onnx(
            src_uri=src_uri,
            dest_uri=dest_uri,
            s3_endpoint=s3_endpoint,
            s3_access_key=s3_access_key,
            s3_secret_key=s3_secret_key,
            artifact_id=artifact_id,
            opset_version=opset_version,
        )
        step.after(credential_check)
        step.set_display_name("Convert PaddlePaddle to ONNX")
        step.set_caching_options(False)
