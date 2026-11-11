# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""
Training Pipeline

dataset -> train (Q07 simplification: the old checkpoint/export/register/event
steps were removed; training artifacts are written straight to SeaweedFS by the
trainer and model registration runs in the NereidForge worker after the job).
"""

import json
from typing import Dict, List, NamedTuple, Optional

from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Model, Metrics, Artifact

# Import components
from components.dataset_step import dataset_step
from components.train_step import train_step
from images import nf_image


TrainingOutput = NamedTuple(
    "TrainingOutput",
    [
        ("model_uri", str),
        ("export_uri", str),
        ("registration_uri", str),
        ("checkpoint_uri", str),
    ],
)


@dsl.pipeline(
    name="nereidforge-training-pipeline",
    description="Training pipeline: dataset -> train (artifacts registered by the worker)",
    pipeline_root="s3://nereidforge-pipelines/{{workflow.uid}}",
)
def training_pipeline(
    # Pipeline parameters
    dataset_name: str = "coco-2017",
    dataset_version: str = "v1.0.0",
    model_name: str = "yolov8n",
    model_version: str = "1.0.0",
    artifact_id: str = "{{workflow.uid}}",
    trainjob_name: str = "train-{{workflow.uid}}",
    namespace: str = "nereidforge",
    # Registry URLs
    dataset_registry_url: str = "http://dataset-registry.nereidforge.svc.cluster.local:8080",
    hub_api_url: str = "http://hub.nereidforge.svc.cluster.local:8080",
    hub_token: str = "",
    seaweedfs_url: str = "http://seaweedfs.nereidforge.svc.cluster.local:8888",
    event_bus_url: str = "http://event-bus.nereidforge.svc.cluster.local:8080",
    # Configuration (static defaults, artifact_id passed separately)
    # train_config is always supplied by the backend worker as a per-job S3 key
    # (s3://nereidforge/train-input/<job_id>/train_config.json) so concurrent
    # runs never clobber each other; this default only guards a direct run.
    train_config: str = 's3://nereidforge/train-input/unknown/train_config.json',
    export_formats: str = '["onnx", "gguf"]',
    quantization: str = "fp16",
    # Common config (without artifact_id - components receive it separately)
    common_config: str = '{"pipeline": "training"}',
    # Resource defaults
    cpu_request: str = "4",
    cpu_limit: str = "8",
    memory_request: str = "8Gi",
    memory_limit: str = "16Gi",
    gpu_request: str = "1",
    gpu_limit: str = "1",
    # Output directories
    dataset_output_dir: str = "s3://nereidforge/yolo-train/",
    train_output_dir: str = "s3://nereidforge/models/{{workflow.uid}}/",
    checkpoint_output_dir: str = "s3://nereidforge/checkpoints/{{workflow.uid}}/",
    export_output_dir: str = "s3://nereidforge/exports/{{workflow.uid}}/",
    register_output_dir: str = "s3://nereidforge/registry/{{workflow.uid}}/",
    event_output_dir: str = "s3://nereidforge/events/{{workflow.uid}}/",
) -> TrainingOutput:
    """
    Training pipeline with standardized component interface.

    All components use: --input-uri --output-dir --config-json --artifact-id
    """

    # ============================================================
    # Step 1: Dataset Preparation
    # ============================================================
    dataset_task = dataset_step(
        input_uri=dataset_output_dir,
        output_dir=dataset_output_dir,
        config_json=common_config,
        artifact_id=artifact_id,
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        registry_url=dataset_registry_url,
        registry_token=hub_token,
        frame_interval=30,
        max_frames=1000,
    )
    dataset_task.set_display_name("Dataset Preparation")
    dataset_task.set_caching_options(False)

    # Apply resource requirements
    dataset_task.set_cpu_request(cpu_request)
    dataset_task.set_cpu_limit(cpu_limit)
    dataset_task.set_memory_request(memory_request)
    dataset_task.set_memory_limit(memory_limit)

    # ============================================================
    # Step 2: Training
    # ============================================================
    train_task = train_step(
        input_uri=dataset_task.outputs["dataset_output_path"],
        output_dir=train_output_dir,
        config_json=train_config,
        artifact_id=artifact_id,
        trainjob_name=trainjob_name,
        namespace=namespace,
        poll_interval=30,
        timeout_seconds=86400,
        seaweedfs_url=seaweedfs_url,
        # Resolved here (compile/upload time) because the serialized body
        # cannot import images.py; train_config.image still wins at run time.
        trainer_image=nf_image("trainer/torch-cpu"),
    )
    train_task.set_display_name("Training")
    train_task.after(dataset_task)
    train_task.set_caching_options(False)

    # GPU resources for training
    train_task.set_cpu_request(cpu_request)
    train_task.set_cpu_limit(cpu_limit)
    train_task.set_memory_request(memory_request)
    train_task.set_memory_limit(memory_limit)
    # The TrainJob controls GPU resources through train_config; the KFP step does not
    # request GPUs, preventing CPU training from becoming Unschedulable without GPU nodes.

    # The legacy checkpoint/export/register/event steps were removed in the Q07 simplification.
    # train.py writes training artifacts (.pt/.onnx and metrics) directly to SeaweedFS.
    # The NereidForge worker registers the model after the job; KFP does not register it separately.

    # Return outputs
    return TrainingOutput(
        model_uri=train_task.outputs["model_output_path"],
        export_uri=train_task.outputs["model_output_path"],
        registration_uri=train_task.outputs["model_output_path"],
        checkpoint_uri=train_task.outputs["model_output_path"],
    )


# Compile if run directly
if __name__ == "__main__":
    from kfp import compiler

    compiler.Compiler().compile(
        training_pipeline,
        "training_pipeline.yaml",
    )
    print("Training pipeline compiled to training_pipeline.yaml")
