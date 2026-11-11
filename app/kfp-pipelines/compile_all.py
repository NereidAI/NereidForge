#!/usr/bin/env python3
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""
Compile All Pipelines

Compiles all 3 KFP v2 pipelines and writes them to
app/kfp-pipelines/dist/kfp-pipelines.yaml as a ConfigMap with 3 keys.

Note: the platform actually delivers pipelines to KFP via upload_pipelines.py
(KFP API), so this ConfigMap is only a local artifact convenience, not consumed
by the cluster.
"""

import json
import os
import sys
import yaml
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from kfp import compiler

# Import pipeline functions
from pipelines.training_pipeline import training_pipeline
from pipelines.optimization_pipeline import optimization_pipeline
from pipelines.evaluation_pipeline import evaluation_pipeline


def compile_pipeline_to_yaml(pipeline_func, output_path: str) -> str:
    """Compile a pipeline function to YAML string."""
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        temp_path = tmp.name

    try:
        compiler.Compiler().compile(pipeline_func, temp_path)
        with open(temp_path, "r") as f:
            yaml_content = f.read()
        return yaml_content
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def main():
    """Main compilation function."""
    # Output directory (local build artifact; not consumed by the cluster)
    dist_dir = Path(__file__).parent / "dist"
    dist_dir.mkdir(parents=True, exist_ok=True)

    output_file = dist_dir / "kfp-pipelines.yaml"

    print("Compiling NereidForge KFP pipelines...")

    # Compile all three pipelines
    print("  Compiling training_pipeline...")
    training_yaml = compile_pipeline_to_yaml(training_pipeline, "training_pipeline.yaml")

    print("  Compiling optimization_pipeline...")
    optimization_yaml = compile_pipeline_to_yaml(optimization_pipeline, "optimization_pipeline.yaml")

    print("  Compiling evaluation_pipeline...")
    evaluation_yaml = compile_pipeline_to_yaml(evaluation_pipeline, "evaluation_pipeline.yaml")

    # Create ConfigMap
    configmap = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": "nereidforge-kfp-pipelines",
            "namespace": "nereidforge",
            "labels": {
                "app.kubernetes.io/name": "nereidforge",
                "app.kubernetes.io/component": "kfp-pipelines",
            },
        },
        "data": {
            "training_pipeline.yaml": training_yaml,
            "optimization_pipeline.yaml": optimization_yaml,
            "evaluation_pipeline.yaml": evaluation_yaml,
        },
    }

    # Write ConfigMap
    with open(output_file, "w") as f:
        yaml.dump(configmap, f, default_flow_style=False, sort_keys=False)

    print(f"\n✓ All pipelines compiled to {output_file}")
    print(f"  Keys: training_pipeline.yaml, optimization_pipeline.yaml, evaluation_pipeline.yaml")

    # Verify output
    with open(output_file, "r") as f:
        content = f.read()
        print(f"\nConfigMap size: {len(content)} bytes")

    return 0


if __name__ == "__main__":
    sys.exit(main())
