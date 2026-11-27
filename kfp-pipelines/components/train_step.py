# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""
Training Step Component

Creates a kubeflow-trainer TrainJob (trainer.kubeflow.org/v1alpha1), watches until Succeeded/Failed.
Exits with code 0 on success, non-zero on failure.

Supports CPU training and the vendor GPU resources declared in train_config
(nvidia.com/gpu, amd.com/gpu, gpu.intel.com/xe, or gpu.intel.com/i915).

Invariant: the function body is serialized as plain source and re-executed in
the executor pod without this module, so it must not reference module-level
names (constants resolved by `nf_image`, loggers, imports outside the body).
Every default it needs is either a literal or an explicit component input.
"""

import logging

from kfp import dsl
from kfp.dsl import OutputPath

from images import nf_image

logger = logging.getLogger(__name__)

@dsl.component(
    base_image=nf_image("runtime"),
    packages_to_install=["kubernetes>=30.0.0", "tenacity>=8.2.0", "pyyaml>=6.0"],
)
def train_step(
    input_uri: str,
    output_dir: str,
    config_json: str,
    artifact_id: str,
    trainjob_name: str,
    namespace: str = "nereidforge",
    runtime_name: str = "torch-distributed",
    # NOTE the light-weight component body is serialized as plain Python
    # source and re-executed without this module, so a default naming a
    # module-level constant (the registry resolver's result or any upper-case
    # default constant) raises NameError at run time even though the compiled
    # spec inlines the value (pit 192). The pipeline therefore passes the
    # resolved trainer image explicitly; the body only reads the parameter.
    trainer_image: str = "",
    poll_interval: int = 30,
    timeout_seconds: int = 86400,
    seaweedfs_url: str = "http://seaweedfs.nereidforge.svc.cluster.local:8888",
    model_output_path: OutputPath(str) = None,
    metrics_output_path: OutputPath(str) = None,
) -> None:
    """Create and monitor a kubeflow-trainer TrainJob CR."""
    import json
    import logging
    import sys
    import time
    import urllib.request
    from pathlib import Path
    from kubernetes import client, config
    from kubernetes.client.rest import ApiException

    logger = logging.getLogger("train_step")

    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()

    custom_api = client.CustomObjectsApi()

    GROUP = "trainer.kubeflow.org"
    VERSION = "v1alpha1"
    PLURAL = "trainjobs"

    # config_json may be inline JSON or an s3:// pointer to the uploaded
    # train_config.json (Q07: hyperparameters are read from seaweedfs).
    if config_json.startswith("s3://"):
        s3_uri = config_json
        filer_path = "/buckets/" + s3_uri[len("s3://"):].strip("/")
        if seaweedfs_url:
            url = seaweedfs_url.rstrip("/") + filer_path
            try:
                with urllib.request.urlopen(url, timeout=30) as resp:
                    config_json = resp.read().decode("utf-8")
                logger.info("train_step: loaded train_config from %s", url)
            except Exception as e:
                logger.error("train_step: failed to read train_config from %s: %s", url, e)
                sys.exit(1)

    train_config = json.loads(config_json)

    # --- Build a CRD-valid TrainJob spec ---
    # kubeflow-trainer v1alpha1: spec.trainer is an OBJECT:
    #   {image, command, args, env, numNodes, numProcPerNode, resourcesPerNode, ...}
    resources = train_config.get("resourcesPerNode", {}) or {}
    requests = resources.get("requests", {}) or {}
    limits = resources.get("limits", {}) or {}
    # Detect the accelerator generically: each vendor uses its own extended
    # resource key (nvidia.com/gpu | amd.com/gpu | gpu.intel.com/xe). The
    # backend profile already placed the right key in requests/limits; here we
    # only surface whichever one is present for logging.
    _GPU_KEYS = ("nvidia.com/gpu", "amd.com/gpu", "gpu.intel.com/xe", "gpu.intel.com/i915")
    gpu = "0"
    gpu_key = ""
    for k in _GPU_KEYS:
        if k in requests:
            gpu = str(requests[k])
            gpu_key = k
            break
    if gpu in ("", "None"):
        gpu = "0"

    command = train_config.get("command", [])
    args = train_config.get("args", [])

    # A custom training job points at a user-supplied script URI (I-11). When
    # no explicit command/args are given, the trainer container fetches the
    # script over the filer's anonymous public-read HTTP endpoint and runs it,
    # so a user image needs no fixed entrypoint contract. Explicit command/args
    # in train_config always win (that is the "custom entrypoint" channel).
    custom_script = str(train_config.get("custom_script_uri") or "")
    if not command and not args and custom_script:
        if custom_script.startswith("s3://"):
            script_url = (seaweedfs_url.rstrip("/") + "/buckets/"
                          + custom_script[len("s3://"):].strip("/"))
        elif custom_script.startswith("http://") or custom_script.startswith("https://"):
            script_url = custom_script
        else:
            logger.error("train_step: custom_script_uri must be s3:// or http(s)://, got %s", custom_script)
            sys.exit(1)
        # Python is the only runtime the fetch relies on (stdlib urllib), which
        # any ML training image is expected to ship. runpy sets __name__ to
        # __main__ so if-block guards in the user script behave as expected.
        _FETCH_AND_RUN = (
            "import os,runpy,sys,urllib.request\n"
            "url=sys.argv[1]\n"
            "f='/tmp/nf_custom_train.py'\n"
            "open(f,'wb').write(urllib.request.urlopen(url).read())\n"
            "os.chmod(f,0o755)\n"
            "sys.argv=['/tmp/nf_custom_train.py']\n"
            "runpy.run_path(f, run_name='__main__')\n"
        )
        logger.info("train_step: running custom script from %s", script_url)
        command = ["python"]
        args = ["-c", _FETCH_AND_RUN, script_url]

    # Without explicit command/args the trainer runs the image's entrypoint
    # script (/app/train.py by default); see the script map below.
    if not command:
        command = ["python"]
    if not args:
        # train_config.script selects the entrypoint inside the trainer image
        # (train | evaluate | export | catalog; ADR-014 D6). Images that own
        # their ENTRYPOINT can set script="" to use it as-is. The legacy
        # `timm_train` value from old jobs resolves to the unified trainer.
        script = str(train_config.get("script", "") or "train")
        script_map = {
            "train": "/app/train.py",
            "timm_train": "/app/train.py",
            "evaluate": "/app/evaluate.py",
            "export": "/app/export.py",
            "catalog": "/app/catalog.py",
            "optimize": "/app/optimize.py",
        }
        args = [script_map.get(script, "/app/train.py")]

    # Base env the entrypoints rely on. Vendor/accelerator and S3/NPROC values
    # come from train_config["env"] (built by the backend single-source-of-truth)
    # and are merged by name without duplicating the built-in keys.
    base_env = [
        {"name": "INPUT_URI", "value": input_uri},
        {"name": "OUTPUT_DIR", "value": output_dir},
        {"name": "CONFIG_JSON", "value": config_json},
        {"name": "ARTIFACT_ID", "value": artifact_id},
    ]
    extra_env = train_config.get("env", []) or []
    if isinstance(extra_env, dict):
        extra_env = [{"name": str(k), "value": str(v)} for k, v in extra_env.items()]
    seen = {e["name"] for e in base_env}
    for e in extra_env:
        if not isinstance(e, dict) or "name" not in e:
            continue
        if e["name"] not in seen:
            base_env.append({"name": str(e["name"]), "value": str(e.get("value", ""))})
            seen.add(e["name"])

    # The backend normally pins the trainer image in train_config; the
    # component parameter is the fallback when a config omits it. An empty
    # result means neither side resolved one, which is a hard error: a TrainJob
    # with an empty image would be rejected by the API server anyway.
    trainer_image_ref = str(train_config.get("image") or trainer_image or "").strip()
    if not trainer_image_ref:
        logger.error(
            "train_step: no trainer image in train_config and no trainer_image "
            "parameter; refusing to create an image-less TrainJob"
        )
        sys.exit(1)

    trainer_spec = {
        "image": trainer_image_ref,
        "command": command,
        "args": args,
        "env": base_env,
        "numNodes": int(train_config.get("numNodes", 1)),
        "numProcPerNode": int(train_config.get("numProcPerNode", 1)),
        "resourcesPerNode": {
            "requests": requests,
            "limits": limits,
        },
    }

    trainjob_spec = {
        "apiVersion": f"{GROUP}/{VERSION}",
        "kind": "TrainJob",
        "metadata": {
            "name": trainjob_name,
            "namespace": namespace,
            "labels": {
                "nereidforge/artifact-id": artifact_id,
                "nereidforge/pipeline": "training",
            },
        },
        "spec": {
            "trainer": trainer_spec,
            "runtimeRef": {
                "apiGroup": "trainer.kubeflow.org",
                "kind": "ClusterTrainingRuntime",
                "name": runtime_name,
            },
        },
    }

    # Runtime patches target the torch-distributed runtime's "node" replicated
    # job pod template. The TrainJob validator requires the patch manager to be
    # a domain-prefixed path (a bare "nereidforge" is rejected).
    #
    # A memory-backed /dev/shm emptyDir is always mounted because DataLoader
    # worker processes hand tensors back over POSIX shared memory. Without an
    # explicit mount the container only gets Docker's 64 MB tmpfs, which makes
    # the trainer's _safe_workers() clamp num_workers to 0: image decoding and
    # augmentation then run serially in the main process and the GPU stalls
    # between batches (observed ~35 W idle-floor power, minutes per epoch). The
    # trainer enables worker processes only when /dev/shm has at least 1 GiB
    # free, so the volume must be larger than that threshold.
    pod_spec: dict = {
        "volumes": [{
            "name": "shm",
            "emptyDir": {"medium": "Memory", "sizeLimit": "2Gi"},
        }],
        "containers": [{
            "name": "node",
            "volumeMounts": [{
                "name": "shm",
                "mountPath": "/dev/shm",
            }],
        }],
    }
    # Optional node pinning and vendor tolerations come from train_config
    # (backend accelerator profile plus an optional user hostname pin).
    node_selector = train_config.get("node_selector") or {}
    tolerations = train_config.get("tolerations") or []
    if node_selector:
        pod_spec["nodeSelector"] = node_selector
    if tolerations:
        pod_spec["tolerations"] = tolerations
    # NOTE: trainer v2.3.0 TrainJob has no spec.trainer.imagePullPolicy support
    # and runtimePatches only allow pod-level volumes plus container
    # env/name/securityContext/volumeMounts — imagePullPolicy cannot be set to
    # Always. After pushing a new trainer image with the same tag, delete
    # completed trainer pods and `k3s crictl rmi` the old image on the GPU node
    # to force a re-pull.
    trainjob_spec["spec"]["runtimePatches"] = [{
        "manager": "nereidforge.io/trainjob",
        "trainingRuntimeSpec": {
            "template": {
                "spec": {
                    "replicatedJobs": [{
                        "name": "node",
                        "template": {
                            "spec": {
                                "template": {
                                    "spec": pod_spec,
                                },
                            },
                        },
                    }],
                },
            },
        },
    }]

    logger.info("Creating TrainJob: %s in namespace %s (vendor=%s gpu=%s%s)",
                trainjob_name, namespace,
                train_config.get("accelerator", "cpu"), gpu,
                f" via {gpu_key}" if gpu_key else "")
    logger.debug("TrainJob spec: %s", json.dumps(trainjob_spec, indent=2))

    try:
        custom_api.create_namespaced_custom_object(
            group=GROUP, version=VERSION, namespace=namespace,
            plural=PLURAL, body=trainjob_spec,
        )
        logger.info("TrainJob %s created successfully", trainjob_name)
    except ApiException as e:
        if e.status == 409:
            logger.info("TrainJob %s already exists, continuing...", trainjob_name)
        else:
            logger.error("Failed to create TrainJob: %s", e.body or e)
            sys.exit(1)

    # --- Watch TrainJob status ---
    start_time = time.time()
    while time.time() - start_time < timeout_seconds:
        try:
            trainjob = custom_api.get_namespaced_custom_object(
                group=GROUP, version=VERSION, namespace=namespace,
                plural=PLURAL, name=trainjob_name,
            )
            status = trainjob.get("status", {})
            conditions = status.get("conditions", [])
            state = "Unknown"
            for cond in conditions:
                ctype = cond.get("type")
                if ctype in ("Succeeded", "Complete") and cond.get("status") == "True":
                    state = "Succeeded"
                elif ctype in ("Failed",) and cond.get("status") == "True":
                    state = "Failed"
            logger.info("TrainJob %s state: %s", trainjob_name, state)

            if state == "Succeeded":
                logger.info("TrainJob %s completed successfully", trainjob_name)
                model_uri = status.get("modelUri", output_dir)
                if model_output_path:
                    Path(model_output_path).write_text(model_uri)
                metrics = status.get("metrics", {})
                if metrics_output_path:
                    Path(metrics_output_path).write_text(json.dumps(metrics))
                sys.exit(0)
            elif state == "Failed":
                logger.error("TrainJob %s failed: %s", trainjob_name,
                             status.get("failureReason", "Unknown"))
                sys.exit(1)
        except ApiException as e:
            logger.error("Error checking TrainJob status: %s", e.body or e)
            sys.exit(1)
        time.sleep(poll_interval)

    logger.error("TrainJob %s timed out", trainjob_name)
    sys.exit(1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-uri", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--trainjob-name", required=True)
    parser.add_argument("--namespace", default="nereidforge")
    parser.add_argument("--runtime-name", default="torch-distributed")
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument("--timeout-seconds", type=int, default=86400)
    args = parser.parse_args()
    train_step(
        input_uri=args.input_uri,
        output_dir=args.output_dir,
        config_json=args.config_json,
        artifact_id=args.artifact_id,
        trainjob_name=args.trainjob_name,
        namespace=args.namespace,
        runtime_name=args.runtime_name,
        poll_interval=args.poll_interval,
        timeout_seconds=args.timeout_seconds,
    )
