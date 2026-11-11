# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Kale notebook compile + submit engine for Notebook jobs (ADR-013 Case 2).

The worker embeds the ``kubeflow-kale`` wheel (``kale`` console script =
``kale.cli:main``) to compile a Kale-annotated ``.ipynb`` into a native KFP v2
pipeline, post-processes the compiled spec, and submits the run.

Post-processing lives here so every rule from ADR-013 is enforced at one seam:

* **D5 / offline** — the KFP "pip-clone" executor bootstrap (``pip install
  kfp ... kubeflow-kale ...``) is stripped; all dependencies are baked into
  the step (dual-use) image, so executors never reach a package index.
* **run env** — every executor gets the NF run environment (S3 endpoint +
  credentials, bucket) plus ``NF_JOB_ID`` / ``NF_TENANT_ID`` /
  ``NF_MODEL_DIR`` so the notebook can upload its produced artifact to
  ``s3://<bucket>/models/{job_id}/`` (with an optional ``metrics.json``),
  without touching the KFP artifact store.
* **D8 / governance** — every executor image is validated against the
  approved-image allowlist before the pipeline is uploaded; arbitrary
  ``image:`` cell tags are overwritten by the approved step image at compile
  time (``auto_tag``), so the allowlist cannot be bypassed by a notebook.
* **D6 / registration contract** — the compiled pipeline's
  ``outputDefinitions`` are returned as a typed-artifact manifest so the
  worker can register produced models without the largest-blob scan.

The KFP SDK client is used purely for pipeline upload + run submission, in the
same single-user/trusted-internal mode as the previous spike
(``app/kfp-pipelines/notebook_submit.py``); the identity header comes from the
kfp context file written per call.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import yaml

from app.core.images import nf_image
logger = logging.getLogger("nereidforge.notebook_runtime")

# Single NF dual-use image = authoring workbench AND Kale step image
# (ADR-013 D4): JupyterLab + ML stack + kfp dependency closure + kale. On the
# interim 3.11 base until I-13 refreshes it to Python 3.12.
DEFAULT_STEP_IMAGE = nf_image("prediction/jupyter-base", "py3.12")

# Approved-step-image allowlist (ADR-013 D8). Expanded via the
# NF_NOTEBOOK_ALLOWED_IMAGES env var (comma-separated full refs). Besides the
# dual-use image, clusters running NF-deployed platform images may add the
# runtime image (kfp closure) — but it has no kale/pandas, so steps that
# import them fail; keep the default unless you know what you are doing.
_ENV_ALLOWED = os.environ.get("NF_NOTEBOOK_ALLOWED_IMAGES", "").strip()
ALLOWED_STEP_IMAGES = frozenset(
    ref.strip()
    for ref in _ENV_ALLOWED.split(",") if ref.strip()
).union({DEFAULT_STEP_IMAGE})

# KFP context lives next to $HOME (same convention as the KFP CLI). Kept on a
# module constant so tests can point it at a temp dir.
_HOME_CONFIG_KFP = Path(os.environ.get("HOME") or "/tmp") / ".config" / "kfp"


# Paths inside a run that are never candidate model artifacts.
_NON_MODEL_SUFFIXES = (".metrics.json", ".csv", ".log", ".txt", ".json")


def auto_tag(nb_path: str, image: str) -> None:
    """Ensure the notebook carries minimal kale tags; force the step image.

    Untagged notebooks get the first code cell marked ``imports`` and the
    remaining code cells merged into a single ``step:main``, so a plain
    notebook runs too. Every ``image:`` cell tag is replaced with the approved
    step image (kale 2.2.0 reads the base image from cell tags, not from its
    CLI ``--docker_image`` flag, which ``NotebookConfig`` rejects).
    """
    import nbformat

    nb = nbformat.read(nb_path, as_version=4)
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    tagged = any(
        any(t.startswith("step:") or t == "imports" for t in (c.metadata.get("tags") or []))
        for c in code_cells
    )
    if not tagged:
        for idx, cell in enumerate(code_cells):
            cell.metadata.tags = ["imports"] if idx == 0 else ["step:main"]
    if image:
        for cell in code_cells:
            tags = list(cell.metadata.get("tags") or [])
            tags = [t for t in tags if not t.startswith("image:")]
            tags.append(f"image:{image}")
            cell.metadata.tags = tags
    nbformat.write(nb, nb_path)


def _executors(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Yield every executor container dict from the compiled pipeline doc."""
    if not docs:
        return []
    ds = (docs[0].get("deploymentSpec") or {}) or {}
    executors = (ds.get("executors") or {}) or {}
    return [e.get("container") for e in executors.values() if isinstance(e, dict) and e.get("container")]


def strip_pip_bootstrap(container: dict[str, Any]) -> None:
    """Remove the KFP pip-clone wrapper from an executor command.

    KFP compiles Python components with a bootstrap like
    ``sh -c "pip install kfp ... && \"$0\" \"$@\"" sh -ec <executor> ...``.
    Air-gapped runs must never reach a package index, so the first three
    command elements (the wrapper) are dropped, leaving the plain executor.
    """
    cmd = container.get("command")
    if not isinstance(cmd, list) or len(cmd) < 6:
        return
    if cmd[0] == "sh" and cmd[1] == "-c" and '"$0" "$@"' in str(cmd[2]):
        container["command"] = cmd[3:]


def inject_env(container: dict[str, Any], env: dict[str, str]) -> None:
    """Append run env to an executor, de-duplicating on the variable name."""
    existing = container.get("env") or []
    names: set[str] = set()
    kept: list[dict[str, Any]] = []
    for e in existing:
        if not isinstance(e, dict):
            continue
        names.add(str(e.get("name")))
        # NF-owned keys win over kale defaults (e.g. HOME stays as baked).
        if str(e.get("name")) not in env:
            kept.append(e)
    for key, value in env.items():
        if key not in names:
            kept.append({"name": key, "value": str(value)})
    container["env"] = kept


def validate_images(docs: list[dict[str, Any]]) -> None:
    """Reject any executor image outside the approved allowlist (D8)."""
    offenders = sorted({
        str(c.get("image")) for c in _executors(docs) if str(c.get("image")) not in ALLOWED_STEP_IMAGES
    })
    if offenders:
        raise ValueError(
            "notebook pipeline uses non-approved step image(s) "
            f"{offenders}; allowed: {sorted(ALLOWED_STEP_IMAGES)}"
        )


def extract_outputs(docs: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Return the typed-artifact manifest declared by the compiled pipeline.

    Each component's ``outputDefinitions.artifacts`` becomes
    ``{"artifact": <name>, "type": <schemaTitle>}`` (e.g. ``system.HTML``).
    """
    if not docs:
        return []
    manifest: list[dict[str, str]] = []
    components = (docs[0].get("components") or {}) or {}
    for comp in components.values():
        artifacts = ((comp.get("outputDefinitions") or {}) or {}).get("artifacts") or {}
        for name, spec in artifacts.items():
            st = (((spec or {}).get("artifactType") or {}) or {}).get("schemaTitle")
            manifest.append({"artifact": str(name), "type": str(st or "")})
    return manifest


def _kale_binary() -> str:
    """Resolve the kale console script (installed with the backend package)."""
    found = shutil.which("kale")
    if found:
        return found
    installed = shutil.which(str(Path(sys.executable).parent / "kale"))
    if installed:
        return installed
    raise RuntimeError("kale CLI not found; install kubeflow-kale in the backend image")


def compile_pipeline(
    nb_bytes: bytes,
    *,
    run_name: str,
    step_image: str,
    workdir: Path,
    env_extra: Optional[dict[str, str]] = None,
) -> tuple[Path, list[dict[str, str]]]:
    """Compile a notebook to a post-processed KFP v2 pipeline; returns (yaml, manifest).

    Runs the kale CLI in ``workdir`` (kale only accepts RELATIVE output
    paths), then strips the pip bootstrap and injects the run env into every
    executor. The produced YAML is written to ``workdir/<run_name>.pipeline.yaml``
    and the typed-output manifest is returned.
    """
    write = workdir / "nb.ipynb"
    write.write_bytes(nb_bytes)
    auto_tag(str(write), image=step_image)

    env = dict(os.environ)
    env.setdefault("HOSTNAME", "nf-kale-host")
    env.setdefault("KALE_NAMESPACE_FALLBACK", "nereidforge")
    for k, v in (env_extra or {}).items():
        env[str(k)] = str(v)

    outdir = "out"
    cmd = [
        _kale_binary(),
        "--nb", str(write),
        "--output_path", outdir,
        "--pipeline_name", run_name,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env, cwd=str(workdir))
    yaml_files = list((workdir / outdir).glob("*.pipeline.yaml"))
    if r.returncode != 0 or not yaml_files:
        raise RuntimeError(f"kale compile failed: {r.stderr[-800:]}")

    raw = yaml_files[0].read_text()
    docs = list(yaml.safe_load_all(raw))
    for container in _executors(docs):
        strip_pip_bootstrap(container)
    validate_images(docs)
    for container in _executors(docs):
        inject_env(container, env_extra or {})
    manifest = extract_outputs(docs)

    out = workdir / f"{run_name}.pipeline.yaml"
    with open(out, "w") as f:
        yaml.safe_dump_all(docs, f, default_flow_style=False, sort_keys=False)
    logger.info("compiled %s -> %s (artifacts=%s)", run_name, out.name, manifest)
    return out, manifest


def submit_pipeline(
    compiled: Path,
    *,
    run_name: str,
    kfp_host: str,
    kfp_namespace: str,
    kfp_userid: str,
    exp_path: Path = _HOME_CONFIG_KFP,
    experiment_name: str = "NereidForge Notebooks",
    kfp_experiment_id: Optional[str] = None,
) -> dict[str, str]:
    """Upload the compiled pipeline and submit a run; returns run metadata.

    The SDK resolves the user identity from ``$HOME/.config/kfp/context.json``
    (written per call, ``kubeflow-userid`` header). All KFP calls stay scoped
    to the user's run namespace, never the platform control namespace, so the
    userid's RBAC is satisfied (the control-plane ``kubeflow`` namespace is
    -- instead -- where the KFP API itself runs, and users cannot list
    experiments there). When the platform already knows the experiment (the
    ``kfp_experiment_id`` we reuse for training runs), we skip experiment
    discovery entirely -- the backend then places the run where the user's
    default namespace mapping says.
    """
    from kfp import client as kfp_client

    ctx_dir = exp_path
    ctx_dir.mkdir(parents=True, exist_ok=True)
    (ctx_dir / "context.json").write_text(json.dumps({
        "namespace": kfp_namespace,
        "client_authentication_header_name": "kubeflow-userid",
        "client_authentication_header_value": kfp_userid,
    }))

    kc = kfp_client.Client(host=kfp_host, verify_ssl=False, namespace=kfp_namespace)
    suffix = time.strftime("%Y%m%d-%H%M%S")
    pipeline = kc.upload_pipeline(
        pipeline_package_path=str(compiled),
        pipeline_name=f"{run_name}-{suffix}",
        namespace=kfp_namespace,
    )
    version = kc.upload_pipeline_version(
        pipeline_package_path=str(compiled),
        pipeline_version_name=suffix,
        pipeline_id=pipeline.pipeline_id,
    )

    experiment_id = kfp_experiment_id
    if not experiment_id:
        # No shared experiment configured: resolve (or create) one by name in
        # the user's own namespace so the UI can link runs to a notebook group.
        try:
            experiment = kc.get_experiment(
                experiment_name=experiment_name, namespace=kfp_namespace
            )
            experiment_id = experiment.experiment_id
        except Exception:
            experiment = kc.create_experiment(name=experiment_name, namespace=kfp_namespace)
            experiment_id = experiment.experiment_id
    if not experiment_id:
        raise RuntimeError("kfp experiment resolution failed for notebook run")

    run = kc.run_pipeline(
        experiment_id=experiment_id,
        job_name=f"{run_name}-{suffix}",
        pipeline_id=pipeline.pipeline_id,
        version_id=version.pipeline_version_id,
    )
    return {"run_id": run.run_id, "pipeline_id": pipeline.pipeline_id}


def compile_and_submit(
    *,
    job_id: str,
    tenant_id: str,
    nb_bytes: bytes,
    step_image: str,
    kfp_host: str,
    kfp_namespace: str,
    kfp_userid: str,
    s3_env: dict[str, str],
    kfp_experiment_id: Optional[str] = None,
) -> dict[str, Any]:
    """Compile + submit a notebook run for a job; returns run metadata + manifest."""
    if step_image not in ALLOWED_STEP_IMAGES:
        raise ValueError(
            f"step image {step_image} is not in the approved notebook allowlist"
        )
    run_name = f"nb-{job_id}"
    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)
        compiled, outputs = compile_pipeline(
            nb_bytes,
            run_name=run_name,
            step_image=step_image,
            workdir=workdir,
            env_extra=s3_env,
        )
        meta = submit_pipeline(
            compiled,
            run_name=run_name,
            kfp_host=kfp_host,
            kfp_namespace=kfp_namespace,
            kfp_userid=kfp_userid,
            kfp_experiment_id=kfp_experiment_id,
        )
    meta["outputs"] = outputs
    logger.info("notebook job %s submitted as run %s (outputs=%s)", job_id, meta["run_id"], outputs)
    return meta
