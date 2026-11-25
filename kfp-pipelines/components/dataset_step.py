# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""
Dataset Step Component

Fetches dataset version from Dataset Registry, performs video frame extraction preprocessing.
"""

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

from kfp import dsl
from kfp.dsl import InputPath, OutputPath

from images import nf_image

logger = logging.getLogger(__name__)


@dsl.component(
    base_image=nf_image("runtime"),
    packages_to_install=["requests>=2.31.0", "tenacity>=8.2.0", "opencv-python-headless>=4.8.0"],
)
def dataset_step(
    input_uri: str,
    output_dir: str,
    config_json: str,
    artifact_id: str,
    dataset_name: str,
    dataset_version: str,
    registry_url: str = "http://dataset-registry.nereidforge.svc.cluster.local:8080",
    registry_token: str = "",
    frame_interval: int = 30,
    max_frames: int = 1000,
    dataset_output_path: OutputPath(str) = None,
    manifest_output_path: OutputPath(str) = None,
) -> None:
    """
    Fetch dataset from registry and extract frames from videos.

    Args:
        input_uri: Input URI (can be overridden by registry)
        output_dir: Output directory for extracted frames
        config_json: JSON configuration
        artifact_id: Unique artifact identifier
        dataset_name: Dataset name in registry
        dataset_version: Dataset version
        registry_url: Dataset Registry API URL
        registry_token: Authentication token
        frame_interval: Extract 1 frame every N frames
        max_frames: Maximum frames to extract per video
        dataset_output_path: Output path for dataset location
        manifest_output_path: Output path for dataset manifest
    """
    import json
    import logging
    import sys
    from pathlib import Path
    import requests
    from tenacity import retry, stop_after_attempt, wait_exponential
    import cv2

    logger = logging.getLogger("dataset_step")

    config = json.loads(config_json)
    headers = {"Authorization": f"Bearer {registry_token}"} if registry_token else {}

    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
    def fetch_dataset_version(name: str, version: str) -> dict:
        url = f"{registry_url}/api/v1/datasets/{name}/versions/{version}"
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()

    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
    def download_file(url: str, dest: Path) -> None:
        response = requests.get(url, headers=headers, stream=True, timeout=60)
        response.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    logger.info(f"Fetching dataset {dataset_name}:{dataset_version}")
    print(f"[dataset_step] input_uri={input_uri!r} output_dir={output_dir!r}", flush=True)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    manifest = {
        "dataset_name": dataset_name,
        "dataset_version": dataset_version,
        "artifact_id": artifact_id,
        "source_uri": input_uri,
        "extracted_frames": [],
    }

    # Pass through S3 URIs that already reference YOLO/COCO data in SeaweedFS,
    # bypassing the registry.
    if input_uri and input_uri.startswith("s3://"):
        print(f"[dataset_step] passthrough {input_uri}", flush=True)
        if dataset_output_path:
            Path(dataset_output_path).write_text(input_uri)
        if manifest_output_path:
            with open(manifest_output_path, "w") as f:
                json.dump(manifest, f, indent=2)
        print(f"[dataset_step] passthrough done", flush=True)
        return

    # Get dataset version info
    try:
        dataset_info = fetch_dataset_version(dataset_name, dataset_version)
        logger.info(f"Dataset info: {dataset_info}")
    except requests.RequestException as e:
        logger.error(f"Failed to fetch dataset: {e}")
        sys.exit(1)

    # Determine source URI
    source_uri = dataset_info.get("uri", input_uri)
    files = dataset_info.get("files", [])

    # Process each video file
    for file_info in files:
        if not file_info.get("path", "").endswith((".mp4", ".avi", ".mov", ".mkv")):
            continue

        file_url = file_info.get("url")
        if not file_url:
            continue

        local_video = output_path / "videos" / Path(file_info["path"]).name
        download_file(file_url, local_video)

        # Extract frames
        frames_dir = output_path / "frames" / local_video.stem
        frames_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(local_video))
        frame_count = 0
        extracted = 0

        while cap.isOpened() and extracted < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                frame_path = frames_dir / f"frame_{extracted:06d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                manifest["extracted_frames"].append({
                    "video": file_info["path"],
                    "frame_index": frame_count,
                    "path": str(frame_path.relative_to(output_path)),
                })
                extracted += 1

            frame_count += 1

        cap.release()
        logger.info(f"Extracted {extracted} frames from {file_info['path']}")

    # Write outputs
    if dataset_output_path:
        Path(dataset_output_path).write_text(str(output_path))

    if manifest_output_path:
        Path(manifest_output_path).write_text(json.dumps(manifest, indent=2))

    logger.info(f"Dataset processing completed. Output: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-uri", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--registry-url", default="http://dataset-registry.nereidforge.svc.cluster.local:8080")
    parser.add_argument("--registry-token", default="")
    parser.add_argument("--frame-interval", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=1000)
    args = parser.parse_args()

    dataset_step(
        input_uri=args.input_uri,
        output_dir=args.output_dir,
        config_json=args.config_json,
        artifact_id=args.artifact_id,
        dataset_name=args.dataset_name,
        dataset_version=args.dataset_version,
        registry_url=args.registry_url,
        registry_token=args.registry_token,
        frame_interval=args.frame_interval,
        max_frames=args.max_frames,
    )
