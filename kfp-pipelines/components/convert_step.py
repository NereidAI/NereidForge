# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""
Convert PyTorch and custom checkpoints to ONNX in the torch-only image.

Dispatch on a `source` string so the component handles every supported torch
export path without importing another framework:

    source=torchvision    -> torchvision.models.get_model(architecture) +
                             load_state_dict + torch.onnx.export
    source=torch          -> generic torch state_dict + architecture
    source=custom         -> user-supplied converter launcher (see `custom_uri`)

TensorFlow and Paddle inference artifacts are handled by separate components
and images (convert_tf_step.py / convert_paddle_step.py); the pipeline selects
exactly one branch by `family`.

`architecture` is required for torchvision (`resnet18`, `efficientnet_b0`, ...)
since a bare state_dict losslessly describes no network shape.

All families converge on ONNX -- the platform uses ONNX as the single model
exchange layer, so downstream serving (Triton / OpenVINO) and the NF trial /
pre-annotation backend are fully runtime-agnostic.

NO ultralytics/YOLO - AGPL-3.0 removed
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from kfp import dsl
from kfp.dsl import InputPath, OutputPath

from images import nf_image

logger = logging.getLogger(__name__)

# Conversion lives in its own image so training jobs do not inherit the ONNX
# export stack and the component can retain a single framework dependency set.
DEFAULT_CONVERT_IMAGE = nf_image("trainer/convert")


@dsl.component(
    base_image=DEFAULT_CONVERT_IMAGE,
    # NOTE: no packages_to_install -- the convert image bakes every dependency
    # (boto3, onnx, onnxscript, torch, torchvision).
    # Runtime pip install would require network access, which offline
    # deployments do not have.
)
def convert_pt_to_onnx(
    src_uri: str,
    dest_uri: str,
    s3_endpoint: str,
    s3_access_key: str,
    s3_secret_key: str,
    artifact_id: str,
    source: str = "auto",
    architecture: str = "",
    custom_uri: str = "",
    imgsz: int = 640,
    opset_version: int = 17,
    output_path: OutputPath(str) = None,
) -> None:
    """
    convert_pt_to_onnx converts a PyTorch or custom checkpoint to ONNX format.

    Args:
        src_uri: Source model S3 URI (s3://bucket/key.pt)
        dest_uri: Destination ONNX S3 URI (s3://bucket/key.onnx)
        s3_endpoint: S3 endpoint URL (e.g., http://seaweedfs-s3.nereidforge.svc.cluster.local:8333)
        s3_access_key: S3 access key
        s3_secret_key: S3 secret key
        artifact_id: Unique artifact identifier for temp file naming
        source: Model family (auto/torchvision/torch/custom);
                "auto" inspects the checkpoint to route appropriately
        architecture: Network name for torchvision (resnet18, ...); ignored elsewhere
        custom_uri: s3:// launcher for source=custom (module that exposes convert(ckpt, out, ...))
        imgsz: Input image size for export (default 640)
        opset_version: ONNX opset version (default 17)
        output_path: Output path for the destination URI
    """
    import logging as _log
    import shutil
    from pathlib import Path

    import boto3
    import botocore

    # Inline the allowed-sources set here (not module-level) because KFP's
    # lightweight component compilation strips module-level constants.
    # TensorFlow and Paddle run in their own components before this branch is
    # selected, keeping each converter's framework dependencies isolated.
    allowed_sources = {"torchvision", "torch", "custom"}

    _logger = _log.getLogger(__name__)

    # --- nested helpers (all must live inside this body; KFP strips module-level) ---

    def _parse_s3_uri(uri: str) -> tuple:
        """Parse s3://bucket/key into (bucket, key)."""
        if not uri.startswith("s3://"):
            raise ValueError(f"Invalid S3 URI: {uri}")
        rest = uri[5:]  # strip s3://
        bucket = rest.split("/", 1)[0]
        key = rest.split("/", 1)[1] if "/" in rest else ""
        return bucket, key

    def _pickle_tokens(pt_path) -> bytes:
        """Return the raw pickle global-name bytes of a torch/paddle checkpoint.

        torch.save writes a zip whose first entry is archive/data.pkl; very
        old checkpoints are raw pickles. Pickle stores class module/qualnames
        as raw UTF strings, so byte-substring matching identifies the network
        family without unpickling (safe + framework-light).
        """
        import io as _io
        import zipfile as _zipfile

        with open(pt_path, "rb") as f:
            head = f.read(4)
        if head[:2] == b"PK":
            with _zipfile.ZipFile(pt_path) as zf:
                pkl_name = next(
                    (n for n in zf.namelist() if n.endswith("data.pkl")), None
                )
                if pkl_name:
                    return zf.read(pkl_name)
        with open(pt_path, "rb") as f:
            return f.read()

    def _detect_torch_source(pt_path) -> str:
        """_detect_torch_source selects an exporter without unpickling checkpoints.

        Ultralytics artifacts remain rejected because their source tree was
        removed for license compliance. Paddle tokens identify a routing error:
        those artifacts must use the dedicated Paddle family and image.
        """
        tok = _pickle_tokens(pt_path)
        if b"paddlex" in tok or b"paddle" in tok:
            raise ValueError(
                f"{pt_path.name} is a Paddle artifact; submit it with "
                "family=paddle so convert-pp performs the conversion"
            )
        if b"ultralytics" in tok:
            raise RuntimeError(
                f"{pt_path.name} is an ultralytics checkpoint (AGPL-3.0). "
                "Ultralytics YOLO has been removed from NereidForge due to AGPL-3.0 "
                "license incompatibility. Please use PP-YOLOE (PaddleDetection) "
                "or another Apache-2.0 licensed model instead."
            )
        if b"torchvision" in tok:
            return "torchvision"
        return "torch"

    def _export_torch(pt_path, onnx_path, source, architecture, imgsz, opset_version):
        """_export_torch loads either supported state_dict route and exports ONNX.

        Torchvision and generic Torch checkpoints share the same architecture
        validation and serialization invariants; keeping one exporter prevents
        their framework compatibility fixes from drifting apart.
        """
        import torch
        from torchvision import models as tv_models

        arch = (architecture or "").strip().lower()
        if not arch:
            raise ValueError(f"source={source} requires 'architecture' (e.g. resnet18)")
        ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt.get("model", ckpt))
        if isinstance(state, torch.nn.Module):
            model = state
            model.eval()
        else:
            if not hasattr(tv_models, arch):
                raise ValueError(
                    f"architecture '{arch}' not found in torchvision.models; "
                    f"use a torchvision name (resnet18, efficientnet_b0, ...)"
                )
            try:
                model = getattr(tv_models, arch)()
            except TypeError:
                model = getattr(tv_models, arch)(num_classes=80)
            model.load_state_dict(state)
            model.eval()

        dummy = torch.zeros(1, 3, imgsz, imgsz)
        torch.onnx.export(
            model, dummy, str(onnx_path),
            input_names=["input"], output_names=["output"],
            opset_version=opset_version, dynamic_axes={"input": {0: "batch"}},
            dynamo=True, external_data=False, optimize=False,
        )
        return str(onnx_path)

    def _export_custom(pt_path, onnx_path, custom_uri, s3, opset_version):
        """Run a user-supplied conversion launcher.

        custom_uri must point to an s3:// .py exposing convert(src, out, opset) -> str
        returning the produced ONNX path.
        """
        if not custom_uri:
            raise ValueError("source=custom requires 'custom_uri'")
        if not custom_uri.startswith("s3://"):
            raise ValueError("custom_uri must be an s3:// launcher path")
        bucket, key = custom_uri[5:].split("/", 1)
        local = Path("/work/custom_launcher.py")
        s3.download_file(bucket, key, str(local))

        import importlib.util
        spec = importlib.util.spec_from_file_location("custom_launcher", local)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if not hasattr(mod, "convert"):
            raise ValueError("custom launcher must expose convert(src, out, opset)")
        return mod.convert(str(pt_path), str(onnx_path), opset_version)

    # --- end of nested helpers ---

    src_bucket, src_key = _parse_s3_uri(src_uri)
    dest_bucket, dest_key = _parse_s3_uri(dest_uri)

    # S3 client
    s3 = boto3.client(
        "s3",
        endpoint_url=s3_endpoint,
        aws_access_key_id=s3_access_key,
        aws_secret_access_key=s3_secret_key,
        region_name="us-east-1",
        config=botocore.config.Config(signature_version="s3v4"),
    )

    work_dir = Path(f"/work/{artifact_id}")
    work_dir.mkdir(parents=True, exist_ok=True)
    src_path = work_dir / Path(src_key).name
    onnx_path = work_dir / "model.onnx"

    source = (source or "auto").strip().lower()
    if source not in ("", "auto") and source not in allowed_sources:
        raise ValueError(
            f"unsupported model source '{source}'; allowed: {sorted(allowed_sources)}"
        )

    try:
        # 1. Download checkpoint from S3 (auto-detection needs the local file).
        _logger.info(f"Downloading {src_uri} to {src_path}")
        s3.download_file(src_bucket, src_key, str(src_path))

        # 1b. When no family was declared, inspect the checkpoint's pickle
        # tokens now that it is local.
        if source in ("", "auto"):
            source = _detect_torch_source(src_path)
            _logger.info(f"auto-detected torch source={source}")

        # 2. Export to ONNX via the family-specific exporter.
        _logger.info(
            f"Exporting source={source} architecture={architecture or '-'} imgsz={imgsz} "
            f"opset={opset_version}"
        )
        if source in ("torchvision", "torch"):
            src_onnx = _export_torch(
                src_path, onnx_path, source, architecture, imgsz, opset_version
            )
        elif source == "custom":
            src_onnx = _export_custom(src_path, onnx_path, custom_uri, s3, opset_version)
        else:
            raise NotImplementedError(
                f"source='{source}' conversion is not wired in this image yet; "
                f"allowed: {sorted(allowed_sources)}"
            )

        if src_onnx is None or not Path(src_onnx).exists():
            raise FileNotFoundError(f"No ONNX file produced for source={source}")

        # 3. Normalize to the final artifact path and upload.
        if Path(src_onnx).resolve() != onnx_path.resolve():
            shutil.copy2(src_onnx, onnx_path)
        _logger.info(f"Uploading {onnx_path} to {dest_uri}")
        s3.upload_file(str(onnx_path), dest_bucket, dest_key)
        s3.head_object(Bucket=dest_bucket, Key=dest_key)
        _logger.info(f"ONNX conversion completed: {dest_uri}")

        if output_path:
            Path(output_path).write_text(dest_uri)

    except Exception as e:
        _logger.error(f"ONNX conversion failed: {e}")
        raise
    finally:
        try:
            shutil.rmtree(work_dir, ignore_errors=True)
        except Exception:
            pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--src-uri", required=True)
    parser.add_argument("--dest-uri", required=True)
    parser.add_argument("--s3-endpoint", required=True)
    parser.add_argument("--s3-access-key", required=True)
    parser.add_argument("--s3-secret-key", required=True)
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--source", default="auto")
    parser.add_argument("--architecture", default="")
    parser.add_argument("--custom-uri", default="")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset-version", type=int, default=17)
    args = parser.parse_args()

    convert_pt_to_onnx(
        src_uri=args.src_uri,
        dest_uri=args.dest_uri,
        s3_endpoint=args.s3_endpoint,
        s3_access_key=args.s3_access_key,
        s3_secret_key=args.s3_secret_key,
        artifact_id=args.artifact_id,
        source=args.source,
        architecture=args.architecture,
        custom_uri=args.custom_uri,
        imgsz=args.imgsz,
        opset_version=args.opset_version,
    )
