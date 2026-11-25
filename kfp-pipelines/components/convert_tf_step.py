# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""
Convert TensorFlow/Keras models to ONNX (KFP component).

Supported inputs on S3:
    - modern Keras v3 single files:  .keras (zip bundle)
    - legacy HDF5 Keras models:      .h5
    - SavedModel directories:        .zip containing saved_model.pb at the
      bundle root (or one directory level down)

The component runs in the dedicated convert-tf image so the torch convert
image never carries TensorFlow (per-vendor/per-family image separation).
Nothing is fetched at runtime -- tensorflow, tf2onnx and boto3 are baked in.
"""

import logging

from kfp import dsl
from kfp.dsl import OutputPath

from images import nf_image

logger = logging.getLogger(__name__)

DEFAULT_CONVERT_TF_IMAGE = nf_image("trainer/convert-tf")


@dsl.component(
    base_image=DEFAULT_CONVERT_TF_IMAGE,
)
def convert_tf_to_onnx(
    src_uri: str,
    dest_uri: str,
    s3_endpoint: str,
    s3_access_key: str,
    s3_secret_key: str,
    artifact_id: str,
    opset_version: int = 17,
    output_path: OutputPath(str) = None,
) -> None:
    """Convert a Keras/SavedModel artifact on S3 to ONNX and upload it back."""
    import logging as _log
    import shutil
    import subprocess
    import zipfile
    from pathlib import Path

    import boto3
    import botocore

    _logger = _log.getLogger(__name__)

    def _parse_s3_uri(uri: str) -> tuple:
        if not uri.startswith("s3://"):
            raise ValueError(f"Invalid S3 URI: {uri}")
        rest = uri[5:]
        bucket = rest.split("/", 1)[0]
        key = rest.split("/", 1)[1] if "/" in rest else ""
        return bucket, key

    def _find_saved_model(root: Path) -> Path:
        """Locate the directory that actually contains saved_model.pb."""
        if (root / "saved_model.pb").exists():
            return root
        hits = list(root.rglob("saved_model.pb"))
        if not hits:
            raise FileNotFoundError(
                "the .zip bundle contains no saved_model.pb (not a SavedModel export)"
            )
        return hits[0].parent

    s3 = boto3.client(
        "s3",
        endpoint_url=s3_endpoint,
        aws_access_key_id=s3_access_key,
        aws_secret_access_key=s3_secret_key,
        region_name="us-east-1",
        config=botocore.config.Config(signature_version="s3v4"),
    )

    src_bucket, src_key = _parse_s3_uri(src_uri)
    dest_bucket, dest_key = _parse_s3_uri(dest_uri)

    work_dir = Path(f"/work/{artifact_id}")
    work_dir.mkdir(parents=True, exist_ok=True)
    src_path = work_dir / Path(src_key).name
    onnx_path = work_dir / "model.onnx"

    try:
        _logger.info(f"Downloading {src_uri} to {src_path}")
        s3.download_file(src_bucket, src_key, str(src_path))
        suffix = src_path.suffix.lower()

        if suffix in (".h5", ".keras"):
            # Single-file Keras models go through a SavedModel export first so
            # every form shares one conversion path (tf2onnx.from_keras is
            # broken against Keras 3, see Dockerfile.convert-tf notes).
            # .h5 is the legacy Keras-2 HDF5 format -> tf_keras; .keras is the
            # native Keras-3 zip container and loads with tf.keras.
            import tensorflow as tf

            export_dir = work_dir / "savedmodel"
            if suffix == ".h5":
                import tf_keras

                _logger.info(f"Loading legacy HDF5 Keras model from {src_path}")
                model = tf_keras.models.load_model(str(src_path), compile=False)
                tf.saved_model.save(model, str(export_dir))
            else:
                _logger.info(f"Loading Keras-3 model from {src_path}")
                model = tf.keras.models.load_model(str(src_path), compile=False)
                model.export(str(export_dir))
            model_dir = _find_saved_model(export_dir)
            _logger.info(f"Converting exported SavedModel {model_dir} via tf2onnx CLI")
            subprocess.run(
                [
                    "python", "-m", "tf2onnx.convert",
                    "--saved-model", str(model_dir),
                    "--output", str(onnx_path),
                    "--opset", str(opset_version),
                ],
                check=True,
            )

        elif suffix == ".zip":
            # SavedModel directory bundle.
            bundle = work_dir / "bundle"
            bundle.mkdir(exist_ok=True)
            with zipfile.ZipFile(src_path) as zf:
                zf.extractall(bundle)
            model_dir = _find_saved_model(bundle)
            _logger.info(f"Converting SavedModel {model_dir} via tf2onnx CLI")
            subprocess.run(
                [
                    "python", "-m", "tf2onnx.convert",
                    "--saved-model", str(model_dir),
                    "--output", str(onnx_path),
                    "--opset", str(opset_version),
                ],
                check=True,
            )
        else:
            raise ValueError(
                f"unsupported TensorFlow artifact suffix '{suffix}'; "
                "use .h5, .keras or a .zip of a SavedModel directory"
            )

        if not onnx_path.exists() or onnx_path.stat().st_size == 0:
            raise FileNotFoundError("tf2onnx produced no ONNX output")

        _logger.info(f"Uploading {onnx_path} to {dest_uri}")
        s3.upload_file(str(onnx_path), dest_bucket, dest_key)
        s3.head_object(Bucket=dest_bucket, Key=dest_key)
        _logger.info(f"TensorFlow ONNX conversion completed: {dest_uri}")

        if output_path:
            Path(output_path).write_text(dest_uri)

    except Exception as e:
        _logger.error(f"TensorFlow ONNX conversion failed: {e}")
        raise
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
