# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Compile a Kale-annotated notebook and submit it as a KFP v2 run.

Called by the NF worker (the repo-root .venv owns kale + the kfp SDK). Contract:

    python notebook_submit.py <notebook.ipynb> <kfp_host> <run_name> \
        [--image <docker_image>] [--experiment <name>]

Prints a JSON line {"run_id": ...} on success (or {"error": ...} + exit 1).
Untagged notebooks are auto-tagged: the first code cell becomes `imports`,
all remaining code cells merge into a single `step:main` — so a plain
Jupyter notebook works without any Kale annotations.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def auto_tag(nb_path: str, image: str = "") -> None:
    """Ensure the notebook has minimal kale tags (idempotent-ish).

    Untagged notebooks get the first code cell marked ``imports`` and the
    remaining code cells merged into a single ``step:main``. When an explicit
    base image is requested it is injected as an ``image:<repo:tag>`` cell tag
    on every code cell (kale 2.2.0 reads the base image from cell tags, not
    from its ``--docker_image`` CLI flag, which is rejected by NotebookConfig).
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
            if idx == 0:
                cell.metadata.tags = ["imports"]
            else:
                cell.metadata.tags = ["step:main"]
    if image:
        for cell in code_cells:
            tags = list(cell.metadata.get("tags") or [])
            tags = [t for t in tags if not t.startswith("image:")]
            tags.append(f"image:{image}")
            cell.metadata.tags = tags
    nbformat.write(nb, nb_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook")
    ap.add_argument("kfp_host")
    ap.add_argument("run_name")
    ap.add_argument("--image", default="")
    ap.add_argument("--experiment", default="NereidForge Notebooks")
    args = ap.parse_args()

    auto_tag(args.notebook, image=args.image)

    with tempfile.TemporaryDirectory() as tmpdir:
        # kale only accepts RELATIVE output paths; run it with cwd=tmpdir and
        # copy the notebook there first so relative refs resolve.
        import shutil as _shutil

        nb_local = str(Path(tmpdir) / "nb.ipynb")
        _shutil.copy(args.notebook, nb_local)
        outdir = "out"
        # 1) compile with the kale CLI (python API is unstable across versions)
        kale_bin = str(Path(sys.executable).parent / "kale")
        env = dict(os.environ)
        # kale assumes it runs INSIDE a notebook pod: it reads HOSTNAME and
        # the in-cluster namespace file during PodDefaults discovery. On a
        # dev box those are absent -> provide stand-ins (the discovery
        # result is only advisory labels).
        env.setdefault("HOSTNAME", "nf-kale-host")
        env.setdefault("KALE_NAMESPACE_FALLBACK", "nereidforge")

        cmd = [
            kale_bin,
            "--nb", nb_local,
            "--output_path", outdir,
            "--pipeline_name", args.run_name,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env, cwd=tmpdir)
        yaml_files = [p for p in (Path(tmpdir) / outdir).glob("*.pipeline.yaml")]
        if r.returncode != 0 or not yaml_files:
            print(json.dumps({"error": f"kale compile failed: {r.stderr[-800:]}"}))
            sys.exit(1)
        compiled = str(yaml_files[0])

        # 2) upload as a pipeline version + submit a run
        from kfp import client as kfp_client
        from kfp_server_api.api_client import ApiClient

        # multi-user KFP requires the user identity header on every call;
        # inject it as a default header on the generated ApiClient.
        # multi-user KFP: identity header via the kfp context file
        user = os.environ.get("KFP_USERID", "admin@nereidforge.local")
        ctx_dir = Path.home() / ".config" / "kfp"
        ctx_dir.mkdir(parents=True, exist_ok=True)
        (ctx_dir / "context.json").write_text(json.dumps({
            "namespace": os.environ.get("KFP_NAMESPACE", "nereidforge"),
            "client_authentication_header_name": "kubeflow-userid",
            "client_authentication_header_value": user,
        }))
        kc = kfp_client.Client(host=args.kfp_host, verify_ssl=False,
                               namespace=os.environ.get("KFP_NAMESPACE", "nereidforge"))
        try:
            exp = kc.get_experiment(experiment_name=args.experiment)
        except Exception:
            exp = kc.create_experiment(name=args.experiment)
        suffix = time.strftime("nb-%Y%m%d-%H%M%S")
        # a fresh pipeline per notebook run keeps name collisions away
        p = kc.upload_pipeline(pipeline_package_path=compiled, pipeline_name=f"{args.run_name}-{suffix}")
        v = kc.upload_pipeline_version(
            pipeline_package_path=compiled,
            pipeline_version_name=suffix,
            pipeline_id=p.pipeline_id,
        )
        run = kc.run_pipeline(
            experiment_id=exp.experiment_id,
            job_name=f"{args.run_name}-{suffix}",
            pipeline_id=p.pipeline_id,
            version_id=v.pipeline_version_id,
        )
        print(json.dumps({"run_id": run.run_id, "pipeline_id": p.pipeline_id}))


if __name__ == "__main__":
    main()
