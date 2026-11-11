# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Compilation guard for KFP pipeline component bodies.

Pit 192: a lightweight ``@dsl.component`` body is serialized as plain Python
source and re-executed in the executor pod without the module that defined it.
A signature default such as ``trainer_image: str = DEFAULT_TRAINER_IMAGE``
therefore raised ``NameError: name 'DEFAULT_TRAINER_IMAGE' is not defined`` at
run time, even though the compiled spec carried the resolved image: the
training run died in the ``train-step`` executor before creating any TrainJob.

These tests compile the real pipelines and inspect the serialized bodies, so a
future default that names a module-level constant fails here instead of on a
cluster.
"""

import sys
import tempfile
from pathlib import Path

import pytest
import yaml

# The KFP pipeline sources live in a sibling app (app/kfp-pipelines) and are
# not on the backend import path.
_KFP_PIPELINES = Path(__file__).resolve().parents[2] / "kfp-pipelines"
if str(_KFP_PIPELINES) not in sys.path:
    sys.path.insert(0, str(_KFP_PIPELINES))

pytest.importorskip("kfp", reason="kfp SDK required to compile pipelines")

from components.run_katib_experiment_step import run_katib_experiment  # noqa: E402
from kfp import compiler  # noqa: E402
from pipelines.evaluation_pipeline import evaluate_step, evaluation_pipeline  # noqa: E402
from pipelines.onnx_conversion_pipeline import (  # noqa: E402
    onnx_conversion_pipeline,
    validate_s3_credentials,
)
from pipelines.optimization_pipeline import (  # noqa: E402
    baseline_step,
    gate_step,
    optimization_pipeline,
    optimize_step,
)
from pipelines.optimization_search_pipeline import optimization_search_pipeline  # noqa: E402
from pipelines.training_pipeline import training_pipeline  # noqa: E402

PIPELINES = {
    "training": training_pipeline,
    "optimization": optimization_pipeline,
    "evaluation": evaluation_pipeline,
    "onnx-conversion": onnx_conversion_pipeline,
    "optimization-search": optimization_search_pipeline,
}

# Module-level names a serialized body must never mention: the image resolver,
# DEFAULT_* constants, and the former Katib TERMINAL set only exist at compile time.
FORBIDDEN_IN_BODY = ("nf_image", "DEFAULT_", "TERMINAL")


def _compile(name: str) -> dict:
    """Compile one pipeline to its IR document."""
    pipeline_func = PIPELINES[name]
    with tempfile.TemporaryDirectory(prefix="nf-pipeline-test-") as tmp:
        path = Path(tmp) / f"{name}.yaml"
        compiler.Compiler().compile(pipeline_func=pipeline_func, package_path=str(path))
        return yaml.safe_load(path.read_text())


@pytest.mark.parametrize("name", sorted(PIPELINES))
def test_component_bodies_do_not_reference_module_names(name):
    """Every executor body must be self-contained (no compile-time names)."""
    doc = _compile(name)
    offenders = []
    for label, executor in doc["deploymentSpec"]["executors"].items():
        body = "\n".join(executor.get("container", {}).get("command", []) or [])
        hits = sorted({token for token in FORBIDDEN_IN_BODY if token in body})
        if hits:
            offenders.append((label, hits))
    assert not offenders, f"{name} bodies reference compile-time names: {offenders}"


@pytest.mark.parametrize("name", sorted(set(PIPELINES) - {"training"}))
def test_pipeline_s3_credentials_are_required(name):
    """Credential pipeline parameters must not carry usable or empty defaults."""
    parameters = _compile(name)["root"]["inputDefinitions"]["parameters"]
    for credential_name in ("s3_access_key", "s3_secret_key"):
        assert parameters[credential_name]["parameterType"] == "STRING"
        assert "defaultValue" not in parameters[credential_name]


BLANK_CREDENTIAL_CASES = [
    (
        evaluate_step,
        {
            "model_uri": "",
            "baseline_model_uri": "",
            "dataset_uri": "",
            "output_uri": "",
            "report_uri": "",
            "artifact_id": "",
            "config_json": "{}",
            "status_output_path": "unused",
        },
    ),
    (
        baseline_step,
        {
            "model_uri": "",
            "dataset_uri": "",
            "output_uri": "",
            "config_json": "{}",
            "metrics": "unused",
        },
    ),
    (
        optimize_step,
        {
            "target": {},
            "model_uri": "",
            "output_uri": "",
            "config_json": "{}",
            "status": "unused",
        },
    ),
    (
        gate_step,
        {
            "target": {},
            "baseline_metrics_json": "{}",
            "dataset_uri": "",
            "output_uri": "",
            "config_json": "{}",
            "gates_json": "{}",
            "status": "unused",
        },
    ),
    (run_katib_experiment, {"experiment_json": "{}"}),
    (validate_s3_credentials, {}),
]


@pytest.mark.parametrize(
    ("component", "kwargs"),
    BLANK_CREDENTIAL_CASES,
    ids=("evaluate", "baseline", "optimize", "gate", "katib", "onnx-validator"),
)
@pytest.mark.parametrize("blank_name", ("s3_access_key", "s3_secret_key"))
def test_components_reject_blank_s3_credentials(component, kwargs, blank_name):
    """Every object-store component rejects either blank credential before work."""
    kwargs = dict(kwargs)
    kwargs.update(s3_access_key="test-access", s3_secret_key="test-secret")
    kwargs[blank_name] = " \t"
    with pytest.raises(ValueError, match="s3_access_key and s3_secret_key must be non-empty"):
        component.python_func(**kwargs)


def test_onnx_conversion_validates_credentials_before_condition():
    """No converter branch can start before the credential gate succeeds."""
    tasks = _compile("onnx-conversion")["root"]["dag"]["tasks"]
    branch_groups = [
        task
        for task in tasks.values()
        if task.get("componentRef", {}).get("name", "").startswith("comp-condition-branches-")
    ]
    assert len(branch_groups) == 1
    assert "validate-s3-credentials" in branch_groups[0]["dependentTasks"]


def test_training_pipeline_pins_trainer_image_as_literal():
    """The train step receives a resolved image, not a resolver call."""
    doc = _compile("training")
    tasks = doc["root"]["dag"]["tasks"]
    train_tasks = [
        task for task in tasks.values()
        if task.get("componentRef", {}).get("name") == "comp-train-step"
    ]
    assert train_tasks, "training pipeline lost its train-step task"
    ref = train_tasks[0]["inputs"]["parameters"]["trainer_image"]
    constant = ref.get("runtimeValue", {}).get("constant", "")
    assert isinstance(constant, str) and "/" in constant and ":" in constant, ref
    assert "nf_image" not in str(ref)
