#!/usr/bin/env python3
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""
TensorRT Optimizer Plugin CLI Entry Point

Usage:
    python plugin.py --input-uri <uri> --output-dir <dir> --config-json <json> --artifact-id <id>
"""

import argparse
import json
import logging
import sys
import traceback
import urllib.parse
from pathlib import Path
from typing import Dict, Any

import boto3

from converter import TensorRTConverter
from precision import PrecisionValidator
from benchmark import BenchmarkRunner


# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TensorRT Optimizer Plugin")
    parser.add_argument("--input-uri", required=True, help="Input model URI (s3:// or file://)")
    parser.add_argument("--output-dir", required=True, help="Output directory for artifacts")
    parser.add_argument("--config-json", required=True, help="Configuration JSON string")
    parser.add_argument("--artifact-id", required=True, help="Artifact ID for tracking")
    return parser.parse_args()


def download_file(uri: str, dest_path: Path) -> None:
    """Download file from URI (s3:// or file://) to local path."""
    parsed = urllib.parse.urlparse(uri)
    
    if parsed.scheme == "file":
        # Handle file:// URIs
        src_path = Path(parsed.path)
        if src_path.exists():
            import shutil
            shutil.copy2(src_path, dest_path)
        else:
            raise FileNotFoundError(f"Local file not found: {src_path}")
    elif parsed.scheme == "s3":
        # Handle s3:// URIs
        s3 = boto3.client('s3')
        bucket = parsed.netloc
        key = parsed.path.lstrip('/')
        s3.download_file(bucket, key, str(dest_path))
    else:
        raise ValueError(f"Unsupported URI scheme: {parsed.scheme}")


def upload_file(local_path: Path, uri: str) -> None:
    """Upload file from local path to URI (s3:// or file://)."""
    parsed = urllib.parse.urlparse(uri)
    
    if parsed.scheme == "file":
        dest_path = Path(parsed.path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(local_path, dest_path)
    elif parsed.scheme == "s3":
        s3 = boto3.client('s3')
        bucket = parsed.netloc
        key = parsed.path.lstrip('/')
        s3.upload_file(str(local_path), bucket, key)
    else:
        raise ValueError(f"Unsupported URI scheme: {parsed.scheme}")


def load_config(config_json: str) -> Dict[str, Any]:
    """Parse and validate configuration JSON."""
    try:
        config = json.loads(config_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid config JSON: {e}")
    
    # Validate required fields
    if "precision" not in config:
        raise ValueError("config must contain 'precision' field")
    
    precision = config["precision"]
    if precision not in ["fp32", "fp16", "int8"]:
        raise ValueError(f"Invalid precision: {precision}. Must be fp32, fp16, or int8")
    
    if precision == "int8" and "calibration_uri" not in config:
        raise ValueError("INT8 precision requires 'calibration_uri' in config")
    
    # Set defaults
    config.setdefault("workspace_size", 1073741824)  # 1GB
    config.setdefault("batch_size", 1)
    
    return config


def write_json(filepath: Path, data: Dict[str, Any]) -> None:
    """Write JSON data to file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def main() -> int:
    args = parse_args()
    
    artifact_id = args.artifact_id
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load configuration
        config = load_config(args.config_json)
        precision = config["precision"]
        workspace_size = config["workspace_size"]
        batch_size = config["batch_size"]
        calibration_uri = config.get("calibration_uri")
        
        logger.info(json.dumps({
            "artifact_id": artifact_id,
            "plugin": "nereid-tensorrt",
            "version": "1.0.0",
            "event": "start",
            "precision": precision,
            "input_uri": args.input_uri
        }))
        
        # Download input model
        onnx_path = output_dir / "model.onnx"
        download_file(args.input_uri, onnx_path)
        logger.info(json.dumps({
            "artifact_id": artifact_id,
            "plugin": "nereid-tensorrt",
            "version": "1.0.0",
            "event": "model_downloaded",
            "path": str(onnx_path)
        }))
        
        # Download calibration data if needed
        calibration_path = None
        if calibration_uri:
            calibration_path = output_dir / "calibration.data"
            download_file(calibration_uri, calibration_path)
            logger.info(json.dumps({
                "artifact_id": artifact_id,
                "plugin": "nereid-tensorrt",
                "version": "1.0.0",
                "event": "calibration_downloaded",
                "path": str(calibration_path)
            }))
        
        # Convert ONNX to TensorRT engine
        converter = TensorRTConverter(
            workspace_size=workspace_size,
            batch_size=batch_size
        )
        
        plan_path = output_dir / "model.plan"
        converter.convert(
            onnx_path=onnx_path,
            plan_path=plan_path,
            precision=precision,
            calibration_path=calibration_path
        )
        
        logger.info(json.dumps({
            "artifact_id": artifact_id,
            "plugin": "nereid-tensorrt",
            "version": "1.0.0",
            "event": "conversion_complete",
            "plan_path": str(plan_path)
        }))
        
        # Precision validation
        precision_validator = PrecisionValidator()
        precision_report = precision_validator.validate(
            onnx_path=onnx_path,
            plan_path=plan_path,
            precision=precision,
            calibration_path=calibration_path
        )
        
        write_json(output_dir / "precision_report.json", precision_report)
        logger.info(json.dumps({
            "artifact_id": artifact_id,
            "plugin": "nereid-tensorrt",
            "version": "1.0.0",
            "event": "precision_validation_complete",
            "report": precision_report
        }))
        
        # Benchmark (optional)
        benchmark_runner = BenchmarkRunner()
        benchmark_report = benchmark_runner.run(
            plan_path=plan_path,
            precision=precision
        )
        
        write_json(output_dir / "benchmark_report.json", benchmark_report)
        logger.info(json.dumps({
            "artifact_id": artifact_id,
            "plugin": "nereid-tensorrt",
            "version": "1.0.0",
            "event": "benchmark_complete",
            "report": benchmark_report
        }))
        
        # Write metadata.json (artifact-spec §4)
        metadata = {
            "artifact_id": artifact_id,
            "plugin": "nereid-tensorrt",
            "plugin_version": "1.0.0",
            "input_format": "onnx",
            "output_format": "plan",
            "precision": precision,
            "hardware_target": "nvidia-tensorrt",
            "workspace_size": workspace_size,
            "batch_size": batch_size,
            "files": {
                "model": "model.plan",
                "metadata": "metadata.json",
                "precision_report": "precision_report.json",
                "benchmark_report": "benchmark_report.json"
            }
        }
        write_json(output_dir / "metadata.json", metadata)
        
        logger.info(json.dumps({
            "artifact_id": artifact_id,
            "plugin": "nereid-tensorrt",
            "version": "1.0.0",
            "event": "success",
            "output_dir": str(output_dir)
        }))
        
        return 0
        
    except Exception as e:
        error_msg = traceback.format_exc()
        logger.error(json.dumps({
            "artifact_id": artifact_id,
            "plugin": "nereid-tensorrt",
            "version": "1.0.0",
            "event": "error",
            "error": str(e),
            "traceback": error_msg
        }))
        return 1


if __name__ == "__main__":
    sys.exit(main())
