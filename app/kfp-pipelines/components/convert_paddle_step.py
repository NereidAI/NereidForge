# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""
Convert PaddlePaddle inference models to ONNX (KFP component).

Input on S3 is the inference model structure file `<name>.pdmodel`; its
parameter file `<name>.pdiparams` (same key prefix) is downloaded alongside
when present. Conversion uses paddle2onnx.export and runs in the dedicated
convert-pp image so the torch convert image never carries PaddlePaddle.
Nothing is fetched at runtime -- paddlepaddle, paddle2onnx and boto3 are baked.
"""

import logging

from kfp import dsl
from kfp.dsl import OutputPath

from images import nf_image

logger = logging.getLogger(__name__)

DEFAULT_CONVERT_PP_IMAGE = nf_image("trainer/convert-pp")


@dsl.component(
    base_image=DEFAULT_CONVERT_PP_IMAGE,
)
def convert_paddle_to_onnx(
    src_uri: str,
    dest_uri: str,
    s3_endpoint: str,
    s3_access_key: str,
    s3_secret_key: str,
    artifact_id: str,
    opset_version: int = 17,
    output_path: OutputPath(str) = None,
) -> None:
    """Convert a Paddle inference model (.pdmodel/.pdiparams) on S3 to ONNX."""
    import logging as _log
    import shutil
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
    model_path = work_dir / Path(src_key).name
    if model_path.suffix != ".pdmodel":
        raise ValueError(
            f"Paddle conversion expects a .pdmodel structure file, got {model_path.name}"
        )
    onnx_path = work_dir / "model.onnx"

    # The parameter file shares the model's key prefix (.pdmodel -> .pdiparams).
    params_key = src_key[: -len(".pdmodel")] + ".pdiparams"
    params_path = work_dir / (model_path.stem + ".pdiparams")

    try:
        _logger.info(f"Downloading {src_uri} to {model_path}")
        s3.download_file(src_bucket, src_key, str(model_path))
        try:
            s3.download_file(src_bucket, params_key, str(params_path))
            _logger.info(f"Downloaded params {params_key}")
            params_arg = str(params_path)
        except botocore.exceptions.ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in ("404", "NoSuchKey"):
                # Parameter-less models (e.g. fused inference exports) are valid.
                _logger.info("no .pdiparams sidecar found; converting structure only")
                params_arg = ""
            else:
                raise

        import paddle2onnx

        _logger.info("Running paddle2onnx.export")
        # paddle2onnx 2.x keyword names (model_filename/params_filename).
        paddle2onnx.export(
            model_filename=str(model_path),
            params_filename=params_arg,
            save_file=str(onnx_path),
            opset_version=opset_version,
            enable_onnx_checker=True,
        )

        if not onnx_path.exists() or onnx_path.stat().st_size == 0:
            raise FileNotFoundError("paddle2onnx produced no ONNX output")

        _logger.info(f"Uploading {onnx_path} to {dest_uri}")
        s3.upload_file(str(onnx_path), dest_bucket, dest_key)
        s3.head_object(Bucket=dest_bucket, Key=dest_key)
        _logger.info(f"Paddle ONNX conversion completed: {dest_uri}")

        if output_path:
            Path(output_path).write_text(dest_uri)

    except Exception as e:
        _logger.error(f"Paddle ONNX conversion failed: {e}")
        raise
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
