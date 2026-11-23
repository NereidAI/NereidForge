# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""KServe inference service management via KFP (no direct cluster access).

Mirrors the Go K8sService (internal/service/k8s.go) semantics but performs NO
cluster operations from the NereidForge container. Deploying/undeploying an
InferenceService (+ Traefik ingress) is executed by a KFP cluster-op pipeline
(see kfp-pipelines/pipelines/cluster_deploy_pipeline.py); listing and
status come from the local DB, not kubectl.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from app.clients import cluster_ops
from app.core.config import config, require_s3_credentials
from app.services.model_format import (
    FORMAT_OPENVINO_IR,
    ir_companion_bin,
    serving_format,
)

log = logging.getLogger(__name__)


def sanitize_k8s_name(s: str) -> str:
    """Sanitize a string into a valid Kubernetes name."""
    return re.sub(r"[^a-zA-Z0-9.-]", "-", s).lower()


# Knative derives the Configuration "<isvc>-predictor" from the
# InferenceService name; DNS-1035 caps both at 63 chars, so the ISVC name
# itself must stay <= 53. Optimized versions like
# "model-<20-char-id>-job-c25c5ab16221-32035-ir-fp32" easily exceed that
# (the predictor Configuration then fails admission: "not a DNS 1035 label").
ISVC_NAME_MAX = 53


def serving_isvc_name(model_id: str, version: str) -> str:
    """Build the deterministic, length-bounded KServe InferenceService name.

    Short pairs keep the natural ``model-<id>-<version>`` form. Long pairs are
    truncated with a stable 8-char hash of (id, version); the version TAIL is
    preserved (e.g. ``...-32035-ir-fp32``) so the deployed service still reads
    sensibly, and the hash keeps deploy/undeploy/serving-status derivations
    identical and collision-free.
    """
    import hashlib

    mid = sanitize_k8s_name(model_id)
    ver = sanitize_k8s_name(version)
    name = f"model-{mid}-{ver}"
    if len(name) <= ISVC_NAME_MAX:
        return name
    digest = hashlib.sha1(f"{mid}/{ver}".encode()).hexdigest()[:8]
    # Fixed pieces: "model-" 6, two separators 2, hash 8 -> 16 chars.
    ver_tail = ver[-12:]
    mid_max = ISVC_NAME_MAX - 16 - len(ver_tail)
    mid_part = mid[:max(1, mid_max)].rstrip(".-")
    return f"model-{mid_part}-{ver_tail}-{digest}"


def _load_onnx_model(model_path: str):
    """Load an ONNX model and return the model proto."""
    try:
        import onnx
        return onnx.load(model_path)
    except Exception:
        return None


def inspect_onnx_model(model_path: str) -> dict:
    """Inspect an ONNX model and extract input/output tensor info.

    Returns a dict with:
    - inputs: list of {name, shape, dtype}
    - outputs: list of {name, shape, dtype}
    - opset_version: int
    - ir_version: int
    """
    model = _load_onnx_model(model_path)
    if not model:
        return {"inputs": [], "outputs": [], "opset_version": 0, "ir_version": 0}

    def _shape_to_list(shape_proto) -> list:
        dims = []
        for d in shape_proto.dim:
            if d.dim_param:
                dims.append(d.dim_param)  # dynamic dimension
            elif d.dim_value:
                dims.append(d.dim_value)
            else:
                dims.append(None)
        return dims

    def _dtype_to_str(dtype_int: int) -> str:
        # ONNX TensorProto.DataType enum
        dtype_map = {
            1: "FLOAT", 2: "UINT8", 3: "INT8", 4: "UINT16",
            5: "INT16", 6: "INT32", 7: "INT64", 8: "STRING",
            9: "BOOL", 10: "FLOAT16", 11: "DOUBLE",
            12: "UINT32", 13: "UINT64", 14: "COMPLEX64", 15: "COMPLEX128",
            16: "BFLOAT16", 17: "INT4", 18: "UINT4", 19: "FLOAT8E4M3FN",
            20: "FLOAT8E4M3FNUZ", 21: "FLOAT8E5M2", 22: "FLOAT8E5M2FNUZ",
        }
        return dtype_map.get(dtype_int, f"TYPE_{dtype_int}")

    inputs = []
    for inp in model.graph.input:
        shape = _shape_to_list(inp.type.tensor_type.shape)
        dtype = _dtype_to_str(inp.type.tensor_type.elem_type)
        inputs.append({"name": inp.name, "shape": shape, "dtype": dtype})

    outputs = []
    for out in model.graph.output:
        shape = _shape_to_list(out.type.tensor_type.shape)
        dtype = _dtype_to_str(out.type.tensor_type.elem_type)
        outputs.append({"name": out.name, "shape": shape, "dtype": dtype})

    opset_version = 0
    for opset in model.opset_import:
        if opset.domain == "" or opset.domain == "ai.onnx":
            opset_version = opset.version
            break

    return {
        "inputs": inputs,
        "outputs": outputs,
        "opset_version": opset_version,
        "ir_version": model.ir_version,
    }


def generate_triton_pbtxt(model_info: dict, model_name: str = "model") -> str:
    """Generate Triton config.pbtxt from ONNX model info.

    Args:
        model_info: dict from inspect_onnx_model()
        model_name: name of the model in Triton model repository

    Returns:
        config.pbtxt content as string
    """
    inputs = model_info.get("inputs", [])
    outputs = model_info.get("outputs", [])

    def _fmt_shape(shape) -> str:
        if not shape:
            return "[-1]"
        dims = []
        for d in shape:
            if d is None or (isinstance(d, str) and not d.isdigit()):
                dims.append("-1")
            else:
                dims.append(str(d))
        return "[" + ",".join(dims) + "]"

    def _triton_dtype(dtype: str) -> str:
        # Map ONNX dtype to Triton dtype
        dtype_map = {
            "FLOAT": "TYPE_FP32",
            "FLOAT16": "TYPE_FP16",
            "BFLOAT16": "TYPE_BF16",
            "INT8": "TYPE_INT8",
            "UINT8": "TYPE_UINT8",
            "INT16": "TYPE_INT16",
            "UINT16": "TYPE_UINT16",
            "INT32": "TYPE_INT32",
            "UINT32": "TYPE_UINT32",
            "INT64": "TYPE_INT64",
            "UINT64": "TYPE_UINT64",
            "BOOL": "TYPE_BOOL",
            "DOUBLE": "TYPE_FP64",
        }
        return dtype_map.get(dtype, "TYPE_FP32")

    lines = [
        f'name: "{model_name}"',
        'backend: "onnxruntime"',
        "max_batch_size: 0",
        "",
        "input [",
    ]

    for inp in inputs:
        lines.append("  {")
        lines.append(f'    name: "{inp["name"]}"')
        lines.append(f'    data_type: {_triton_dtype(inp["dtype"])}')
        lines.append(f'    dims: {_fmt_shape(inp["shape"])}')
        lines.append("  }")

    lines.append("]")
    lines.append("")
    lines.append("output [")
    for out in outputs:
        lines.append("  {")
        lines.append(f'    name: "{out["name"]}"')
        lines.append(f'    data_type: {_triton_dtype(out["dtype"])}')
        lines.append(f'    dims: {_fmt_shape(out["shape"])}')
        lines.append("  }")
    lines.append("]")

    return "\n".join(lines)


def download_model_from_s3(model_uri: str, dest_path: str) -> bool:
    """Download a model from S3, rejecting incomplete credentials before boto3."""
    if not model_uri.startswith("s3://"):
        return False
    access_key, secret_key = require_s3_credentials(
        config.storage.s3_access_key,
        config.storage.s3_secret_key,
        scope="storage.s3",
    )
    try:
        import boto3

        rest = model_uri[len("s3://"):]
        bucket, key = rest.split("/", 1)
        s3 = boto3.client(
            "s3",
            endpoint_url=config.serving.download_url_base.rstrip("/"),
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name="us-east-1",
        )
        s3.download_file(bucket, key, dest_path)
        return True
    except Exception:
        return False


def generate_runtime_config(runtime_type: str, model_uri: str, model_name: str = "model") -> dict:
    """Generate runtime-specific configuration for the given model.

    Returns a dict with:
    - serve_cmd: the command to run in the container
    - config_files: dict of {filename: content} for any config files
    - env: dict of environment variables
    - metadata: dict of metadata to store in model version
    """
    # Download model for inspection if needed
    import tempfile
    import os
    model_info = {"inputs": [], "outputs": []}

    # For Triton, we need model info
    is_triton = "triton" in runtime_type.lower()

    if is_triton and model_uri:
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            if download_model_from_s3(model_uri, tmp_path):
                model_info = inspect_onnx_model(tmp_path)
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    runtime_lower = runtime_type.lower()

    if "triton" in runtime_lower:
        # Generate Triton config.pbtxt
        pbtxt = generate_triton_pbtxt(model_info, "model")
        return {
            "serve_cmd": (
                "mkdir -p /models/model/1 && "
                f'curl -sf -o /models/model/1/model.onnx "{{download_url}}" && '
                f"printf \'{pbtxt.replace(chr(92), chr(92)*2).replace(chr(39), chr(39)+chr(34)+chr(39)+chr(34)+chr(39)).replace(chr(10), chr(92)+chr(110))}\' > /models/model/config.pbtxt && "
                f"ls -la /models/model/1/ && "
                f"exec tritonserver --model-repository=/models --model-control-mode=explicit "
                f"--load-model=model --http-port=8000 --grpc-port=8001"
            ),
            "config_files": {"config.pbtxt": pbtxt},
            "metadata": {
                "runtime": "triton",
                "model_info": model_info,
                "generated_pbtxt": pbtxt,
            },
        }
    elif "openvino" in runtime_lower:
        return {
            "serve_cmd": (
                "mkdir -p /tmp/models/model/1 && "
                'curl -sf -o /tmp/models/model/1/model.onnx "{download_url}" && '
                "ls -la /tmp/models/model/1/ && "
                "exec /ovms/bin/ovms --model_path=/tmp/models/model --model_name=model "
                "--port=9001 --rest_port=8080"
            ),
            "config_files": {},
            "metadata": {
                "runtime": "openvino",
                "model_info": model_info,
            },
        }
    else:
        # Default fallback
        return {
            "serve_cmd": (
                "mkdir -p /models/model/1 && "
                'curl -sf -o /models/model/1/model.onnx "{download_url}" && '
                "ls -la /models/model/1/ && "
                "exec python -m http.server 8080"
            ),
            "config_files": {},
            "metadata": {"runtime": runtime_type},
        }


@dataclass
class InferenceServiceSpec:
    """Spec for creating an inference service."""
    name: str
    namespace: str
    model_uri: str
    runtime_image: str = ""
    runtime_type: str = "openvino"
    gpu: str = "0"
    cpu_request: str = "1"
    memory_request: str = "2Gi"
    replicas: int = 1
    env: dict = field(default_factory=dict)
    domain: str = ""
    ingress_class: str = "traefik"
    convert_to_onnx: bool = False  # True when the model is a .pt needing ONNX export


class K8sService:
    """KServe deploy/undeploy facade backed by KFP cluster-op pipelines.

    No kubectl / kubernetes calls happen here: NF containers have no cluster
    access (by design). All mutations go through KFP, lists via the DB.
    """

    def __init__(self, namespace: str = "nereidforge") -> None:
        self.namespace = namespace

    # -- manifest builders ---------------------------------------------------

    def _isvc_manifest(self, spec: InferenceServiceSpec) -> dict:
        """Build the KServe InferenceService manifest."""
        ns = spec.namespace or self.namespace
        if not spec.name:
            raise ValueError("name is required")
        cpu_req = spec.cpu_request or "1"
        mem_req = spec.memory_request or "2Gi"
        gpu = spec.gpu or "0"
        runtime_type = spec.runtime_type or "openvino"
        rt_lower = runtime_type.lower()
        is_triton = "triton" in rt_lower
        is_openvino = "openvino" in rt_lower
        # Resolve the requested accelerator once for the whole manifest:
        # extended resource (scheduling + /dev injection), OVMS target device
        # and the NEO-enabled XPU image all derive from the same selection.
        from app.services.accelerators import ACCELERATOR_PROFILES, normalize_vendor

        vendor = normalize_vendor(gpu) or "cpu"
        accel = ACCELERATOR_PROFILES.get(vendor, ACCELERATOR_PROFILES["cpu"])
        gpu_resource = accel["resource"]
        is_intel_gpu = vendor == "intel" and is_openvino and gpu_resource
        image = spec.runtime_image or ""
        if not image:
            img_key = "triton" if is_triton else ("openvino" if is_openvino else rt_lower)
            image = config.serving.runtime_images.get(img_key) or list(config.serving.runtime_images.values())[0]
        if is_intel_gpu:
            image = config.serving.runtime_images.get("openvino_xpu") or image

        # Convert s3:// model URI to a SeaweedFS HTTP URL the container can curl.
        def _to_http(uri: str) -> str:
            if not uri.startswith("s3://"):
                return uri
            return config.serving.download_url_base.rstrip("/") + "/" + uri[len("s3://"):]

        download_url = _to_http(spec.model_uri)
        artifact_fmt = serving_format(spec.model_uri)
        if artifact_fmt == FORMAT_OPENVINO_IR and is_triton:
            raise ValueError(
                "OpenVINO IR (.xml/.bin) artifacts require the openvino runtime, "
                "not Triton"
            )

        # OVMS device selection: the OpenVINO GPU plugin requires an explicit
        # --target_device; CPU is the OVMS default.
        ov_device = " --target_device GPU" if is_intel_gpu else ""
        if is_triton:
            pbtxt = config.serving.triton_config_pbtxt
            esc = pbtxt.replace("\\", "\\\\").replace("'", "'\"'\"'")
            esc = "\\n".join(line for line in esc.split("\n"))
            serve_cmd = (
                "mkdir -p /models/model/1 && "
                f'curl -sf -o /models/model/1/model.onnx "{download_url}" && '
                f'printf \'{esc}\' > /models/model/config.pbtxt && '
                f"ls -la /models/model/1/ && "
                f"exec tritonserver --model-repository=/models --model-control-mode=explicit"
                f" --load-model=model --http-port={config.serving.serving_port} --grpc-port=9001"
            )
        elif artifact_fmt == FORMAT_OPENVINO_IR:
            # IR is a (model.xml graph, model.bin weights) pair -- OVMS loads
            # an IR directory when both files share a basename; both must be
            # fetched (the .bin URI is the same-basename S3 companion).
            bin_url = _to_http(ir_companion_bin(spec.model_uri) or spec.model_uri)
            serve_cmd = (
                "mkdir -p /tmp/models/model/1 && "
                f'curl -sf -o /tmp/models/model/1/model.xml "{download_url}" && '
                f'curl -sf -o /tmp/models/model/1/model.bin "{bin_url}" && '
                "ls -la /tmp/models/model/1/ && "
                f"exec /ovms/bin/ovms --model_path=/tmp/models/model --model_name=model"
                f" --port=9001 --rest_port={config.serving.serving_port}{ov_device}"
            )
        else:
            serve_cmd = (
                "mkdir -p /tmp/models/model/1 && "
                f'curl -sf -o /tmp/models/model/1/model.onnx "{download_url}" && '
                "ls -la /tmp/models/model/1/ && "
                f"exec /ovms/bin/ovms --model_path=/tmp/models/model --model_name=model"
                f" --port=9001 --rest_port={config.serving.serving_port}{ov_device}"
            )
        limits: dict = {"cpu": cpu_req, "memory": mem_req}
        requests: dict = {"cpu": cpu_req, "memory": mem_req}
        if gpu_resource:
            # Only limits is required for extended resources; KServe/Knative
            # also copies it to requests for scheduling. The Intel device
            # plugin injects /dev/dri once gpu.intel.com/xe is granted.
            limits[gpu_resource] = "1"
        return {
            "apiVersion": "serving.kserve.io/v1beta1",
            "kind": "InferenceService",
            "metadata": {
                "name": spec.name,
                "namespace": ns,
                "labels": {"nereidforge/runtime": runtime_type},
            },
            "spec": {
                "predictor": {
                    "serviceAccountName": "kserve",
                    "containers": [
                        {
                            "name": "kserve-container",
                            "image": image,
                            "command": ["/bin/sh", "-c", serve_cmd],
                            "ports": [{"containerPort": 8080, "protocol": "TCP"}],
                            "resources": {
                                "requests": requests,
                                "limits": limits,
                            },
                            # See _gpu_security_context: OVMS as uid 5000 makes
                            # NEO enumerate zero GPUs; Intel GPU deploys run root.
                            **self._gpu_security_context(is_intel_gpu),
                        }
                    ],
                }
            },
        }

    @staticmethod
    def _gpu_security_context(intel_gpu: bool) -> dict:
        """Container securityContext for an Intel GPU deploy.

        OVMS runs as uid 5000 in the upstream image; with that uid the NEO
        OpenCL runtime (26.x) silently enumerates ZERO platforms even though
        the device plugin hands it a chowned /dev/dri node — the same failure
        reproduces in the trainer image forced to uid 5000, while uid 1000
        and root both enumerate the Arc. Root is the upstream-supported OVMS
        run mode; CPU deploys and the CPU image stay untouched (B580-verified).
        """
        if intel_gpu:
            return {"securityContext": {"runAsUser": 0, "runAsGroup": 0}}
        return {}

    @staticmethod
    def _ingress_manifest(name: str, ns: str, domain: str) -> dict:
        """Traefik IngressRoute manifest routing the domain to the predictor."""
        # Initial target is a deterministic placeholder; the actual service
        # does not exist yet at apply time. The cluster_deploy_pipeline
        # repoints this IngressRoute to the active revision's private Knative
        # ServerlessService (found by serving.knative.dev/revision label, not
        # by name) once the InferenceService is Ready
        # (apply_manifests_step.reconcile_ingress_targets), so redeploys
        # (which create revision 00002, ...) never keep serving the stale one.
        return {
            "apiVersion": "traefik.io/v1alpha1",
            "kind": "IngressRoute",
            "metadata": {"name": name, "namespace": ns},
            "spec": {
                "entryPoints": ["web"],
                "routes": [
                    {
                        "match": f"Host(`{domain}`)",
                        "kind": "Rule",
                        "services": [
                            {"name": f"{name}-predictor-00001-private", "port": 80}
                        ],
                    }
                ],
            },
        }

    @staticmethod
    def _isvc_delete_manifest(name: str, ns: str) -> dict:
        return {
            "apiVersion": "serving.kserve.io/v1beta1",
            "kind": "InferenceService",
            "metadata": {"name": name, "namespace": ns},
        }

    @staticmethod
    def _ingress_delete_manifest(name: str, ns: str) -> dict:
        return {
            "apiVersion": "traefik.io/v1alpha1",
            "kind": "IngressRoute",
            "metadata": {"name": name, "namespace": ns},
        }

    # -- cluster mutations (delegated to KFP) -------------------------------

    def create_inference_service(self, spec: InferenceServiceSpec) -> str:
        """Deploy an InferenceService (+ optional Traefik ingress) via KFP."""
        ns = spec.namespace or self.namespace
        manifests = [self._isvc_manifest(spec)]
        if spec.domain:
            manifests.append(self._ingress_manifest(spec.name, ns, spec.domain))
        log.info("deploy via KFP: name=%s ns=%s manifests=%d", spec.name, ns, len(manifests))
        run_id = cluster_ops.submit_pipeline(
            "cluster_deploy_pipeline",
            params={
                "manifests_json": json.dumps(manifests),
                "wait_isvc_name": spec.name,
                "wait_isvc_namespace": ns,
                "timeout_minutes": 30,
            },
            run_name=f"deploy-{spec.name}",
            namespace=ns,
            wait=True,
            timeout=1800,
            poll_interval=20,
        )
        return f"deployed via kfp run {run_id}"

    def delete_inference_service(self, name: str, namespace: str, wait: bool = False) -> str:
        """Delete the InferenceService + ingress via KFP (idempotent).

        Defaults to wait=False: it only submits the KFP run (~1-3s) and returns
        the run id immediately; the cluster deletion proceeds asynchronously in
        KFP. Pass wait=True to block until the run finishes.
        """
        ns = namespace or self.namespace
        manifests = [
            self._isvc_delete_manifest(name, ns),
            self._ingress_delete_manifest(name, ns),
        ]
        log.info("undeploy via KFP: name=%s ns=%s wait=%s", name, ns, wait)
        return cluster_ops.submit_pipeline(
            "cluster_undeploy_pipeline",
            params={"manifests_json": json.dumps(manifests)},
            run_name=f"undeploy-{name}",
            namespace=ns,
            wait=wait,
            timeout=600,
            poll_interval=15,
        )

    def convert_to_onnx(
        self,
        src_uri: str,
        dest_uri: str,
        ns: str = "nereidforge",
        family: str = "torch",
        source: str = "auto",
        architecture: str = "",
    ) -> str:
        """Convert any supported model artifact to ONNX via the KFP onnx-conversion pipeline.

        `family` selects the converter image/branch (torch/tensorflow/paddle;
        detected from the artifact bytes by services.model_format) and
        `source` selects the torch exporter family
        (auto/paddlex/torchvision/torch) -- see convert_step.py.
        `architecture` names the network for torchvision state_dicts.
        """
        import uuid

        access_key, secret_key = require_s3_credentials(
            config.storage.s3_access_key,
            config.storage.s3_secret_key,
            scope="storage.s3",
        )
        s3_endpoint = config.serving.download_url_base.rstrip("/")
        params = {
            "src_uri": src_uri,
            "dest_uri": dest_uri,
            "s3_endpoint": s3_endpoint,
            "s3_access_key": access_key,
            "s3_secret_key": secret_key,
            "artifact_id": f"onnx-{uuid.uuid4().hex[:8]}",
            "family": family,
            "source": source,
            "architecture": architecture,
            "imgsz": 640,
            "opset_version": 17,
        }
        log.info(
            "onnx convert via KFP: src=%s dest=%s family=%s source=%s architecture=%s",
            src_uri, dest_uri, family, source, architecture,
        )
        cluster_ops.submit_pipeline(
            "onnx_conversion_pipeline",
            params=params,
            run_name=f"onnx-convert-{uuid.uuid4().hex[:8]}",
            namespace=ns,
            wait=True,
            timeout=1800,
            poll_interval=30,
        )
        return dest_uri

    def convert_pt_to_onnx(
        self,
        src_uri: str,
        dest_uri: str,
        ns: str = "nereidforge",
        source: str = "auto",
        architecture: str = "",
    ) -> str:
        """Backward-compatible alias for torch-family ONNX conversion."""
        return self.convert_to_onnx(
            src_uri, dest_uri, ns, family="torch",
            source=source, architecture=architecture,
        )

    def list_serving_runtimes(self) -> list[dict]:
        """List serving runtimes from static config (no cluster reads)."""
        items = []
        for name, image in config.serving.runtime_images.items():
            model_type = "onnx" if name in ("openvino", "triton") else (
                "pytorch" if name in ("torch",) else "huggingface" if name == "huggingface" else ""
            )
            items.append({"name": f"kserve-{name}", "model_type": model_type, "image": image, "disabled": False})
        if items:
            return items
        return [
            {"name": "kserve-tritonserver", "model_type": "tensorrt"},
            {"name": "kserve-torchserve", "model_type": "pytorch"},
            {"name": "kserve-huggingfaceserver", "model_type": "huggingface"},
        ]

# ---------------------------------------------------------------------------
# Interactive notebook workbench: restricted direct cluster access.
#
# Unlike the KServe helpers above (executed through a KFP cluster-op pipeline),
# the notebook workbench manages Kubeflow Notebook CRs + workspace PVCs via the
# kubernetes python SDK using the pod's own ServiceAccount, which is bound to a
# namespaced Role limited to exactly these resources (see the nereidforge-backend
# chart notebook-rbac template). Config resolution: in-cluster SA token first
# (prod), then $KUBECONFIG or /etc/rancher/k3s/k3s.yaml (dev on the host).
# ---------------------------------------------------------------------------

import os  # noqa: E402

from kubernetes import client as _k8s_client  # noqa: E402
from kubernetes import config as _k8s_config  # noqa: E402
from kubernetes.client.rest import ApiException  # noqa: E402

# Notebook CRD coordinates (kubeflow notebook-controller).
NOTEBOOK_GROUP = "kubeflow.org"
NOTEBOOK_VERSION = "v1"
NOTEBOOK_PLURAL = "notebooks"


def _load_kube_config_once(client: Any) -> bool:
    """Load in-cluster/kubeconfig credentials into an SDK-holder object.

    Shared by the restricted SDK clients below. The holder stores `_loaded`,
    `_available`, `_custom` and `_core` attributes. Returns True when the
    kubernetes API is usable. Mirrors the original notebook-client loading
    order: in-cluster SA token first (prod), then $KUBECONFIG or the k3s
    kubeconfig (dev on the host). Any failure downgrades to "unavailable"
    instead of crashing API startup on non-cluster boxes.
    """
    if client._loaded:
        return client._available
    client._loaded = True
    try:
        token_file = "/var/run/secrets/kubernetes.io/serviceaccount/token"
        if os.path.exists(token_file):
            _k8s_config.load_incluster_config()
        else:
            kubeconfig = os.environ.get("KUBECONFIG") or "/etc/rancher/k3s/k3s.yaml"
            _k8s_config.load_kube_config(config_file=kubeconfig)
        client._custom = _k8s_client.CustomObjectsApi()
        client._core = _k8s_client.CoreV1Api()
        client._available = True
        log.info("kubernetes client ready (kube config loaded)")
    except Exception as e:  # noqa: BLE001 - any config failure => disabled
        log.warning("kubernetes client unavailable: %s", e)
        client._available = False
    return client._available


class K8sServingVerifyError(RuntimeError):
    """Post-mutation cluster verification failed (object missing/not ready)."""


class K8sServingUnavailable(RuntimeError):
    """Verification could not run because the kubernetes API is unreachable."""


class K8sServingReader:
    """Read-only KServe InferenceService + predictor pod status.

    Enriches the Deployed Model Services list with live readiness, the
    scheduler node and predictor image, and the latest rollout error. All
    calls are best effort: every cluster failure degrades to None/empty so
    the API can still answer from DB-only metadata. Results are cached for
    a few seconds because the list endpoint fans out one call per service.
    """

    # KServe condition types that explain why a predictor is not ready.
    _FAILURE_CONDITIONS = ("Ready", "PredictorReady", "LatestDeploymentReady", "IngressReady")

    def __init__(self, ttl_seconds: float = 10.0) -> None:
        self._loaded = False
        self._available = False
        self._custom: Optional[_k8s_client.CustomObjectsApi] = None
        self._core: Optional[_k8s_client.CoreV1Api] = None
        self._ttl = ttl_seconds
        self._cache: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}

    @property
    def available(self) -> bool:
        return _load_kube_config_once(self)

    def _isvc(self, name: str, ns: str) -> Optional[dict[str, Any]]:
        """Fetch one InferenceService CR; None when missing/unreachable."""
        if not self.available:
            return None
        try:
            return self._custom.get_namespaced_custom_object(
                "serving.kserve.io", "v1beta1", ns, "inferenceservices", name
            )
        except ApiException as e:
            if e.status != 404:
                log.warning("read inferenceservice %s/%s failed: %s", ns, name, e)
        except Exception as e:  # noqa: BLE001 - reader must never raise
            log.warning("read inferenceservice %s/%s failed: %s", ns, name, e)
        return None

    def isvc_presence(self, name: str, ns: str) -> Optional[bool]:
        """Answer whether an InferenceService exists right now.

        Returns True (present), False (confirmed 404) or None (cluster API
        unreachable / reader not configured) — callers must not treat a
        transient API error as a successful deletion.
        """
        if not self.available:
            return None
        try:
            self._custom.get_namespaced_custom_object(
                "serving.kserve.io", "v1beta1", ns, "inferenceservices", name
            )
            return True
        except ApiException as e:
            if e.status == 404:
                return False
            log.warning("probe inferenceservice %s/%s failed: %s", ns, name, e)
        except Exception as e:  # noqa: BLE001 - reader must never raise
            log.warning("probe inferenceservice %s/%s failed: %s", ns, name, e)
        return None

    def namespace_exists(self, ns: str) -> Optional[bool]:
        """True/False for a namespace, None when the cluster API is unreachable."""
        if not self.available or self._core is None:
            return None
        try:
            self._core.read_namespace(ns)
            return True
        except ApiException as e:
            if e.status == 404:
                return False
            log.warning("read namespace %s failed: %s", ns, e)
        except Exception as e:  # noqa: BLE001 - reader must never raise
            log.warning("read namespace %s failed: %s", ns, e)
        return None

    def _predictor_pod(self, name: str, ns: str) -> Optional[dict[str, str]]:
        """Return {node, image} of a running predictor pod, if any."""
        if not self.available:
            return None
        try:
            pods = self._core.list_namespaced_pod(
                ns,
                label_selector=(
                    f"serving.kserve.io/inferenceservice={name},component=predictor"
                ),
            )
        except Exception as e:  # noqa: BLE001 - reader must never raise
            log.warning("list predictor pods %s/%s failed: %s", ns, name, e)
            return None
        # Prefer a running pod, otherwise take the newest pod so pending/image
        # errors still surface through its node assignment.
        candidates = sorted(
            pods.items,
            key=lambda p: p.status.start_time or p.metadata.creation_timestamp,
            reverse=True,
        )
        for pod in candidates:
            if pod.status.phase in ("Running", "Pending"):
                container = next(
                    (c for c in (pod.spec.containers or []) if c.name == "kserve-container"),
                    (pod.spec.containers or [None])[0],
                )
                return {
                    "node": pod.spec.node_name or "",
                    "image": getattr(container, "image", "") if container else "",
                }
        return None

    def describe(self, name: str, ns: str, *, force: bool = False) -> dict[str, Any]:
        """Live status bundle for one InferenceService (short-TTL cached).

        `force=True` bypasses the TTL cache — verification loops polling a
        mutating rollout must read the live object instead of a stale entry.
        """
        import time as _time

        key = (name, ns)
        now = _time.time()
        cached = self._cache.get(key)
        if not force and cached and now - cached[0] < self._ttl:
            return cached[1]
        result: dict[str, Any] = {
            "ready": False, "error": "", "node": "", "image": "", "created_at": "",
        }
        cr = self._isvc(name, ns)
        if cr is not None:
            result["created_at"] = str((cr.get("metadata") or {}).get("creationTimestamp") or "")
            for cond in (cr.get("status") or {}).get("conditions") or []:
                if cond.get("type") == "Ready" and str(cond.get("status")) == "True":
                    result["ready"] = True
                if (
                    str(cond.get("status")) == "False"
                    and cond.get("type") in self._FAILURE_CONDITIONS
                    and not result["error"]
                ):
                    result["error"] = str(cond.get("message") or cond.get("reason") or "")
        pod = self._predictor_pod(name, ns)
        if pod:
            result["node"] = pod["node"]
            result["image"] = pod["image"]
        self._cache[key] = (now, result)
        return result

    # -- Post-mutation verification gates -----------------------------------

    def wait_until_ready(
        self, name: str, ns: str, timeout: float = 120.0, interval: float = 5.0
    ) -> dict[str, Any]:
        """Verify an InferenceService really exists and is Ready.

        Independent post-deploy gate: trusting the KFP run result alone is not
        enough — a cached/skipped apply reports success while the object was
        never created, and a flaky pipeline failure may hide a good rollout.
        Polls with cache-bypassing reads until Ready=True or `timeout`.

        Raises K8sServingUnavailable when the cluster never answers, and
        K8sServingVerifyError when the object is missing or still not Ready at
        the deadline (message carries the KServe condition reason).
        """
        import time as _time

        if not self.available:
            raise K8sServingUnavailable(f"kubernetes API unavailable; cannot verify {ns}/{name}")
        deadline = _time.time() + timeout
        last = {}
        unknown_polls = 0
        while _time.time() < deadline:
            presence = self.isvc_presence(name, ns)
            if presence is None:
                unknown_polls += 1
            else:
                unknown_polls = 0
                if presence:
                    last = self.describe(name, ns, force=True)
                    if last.get("ready"):
                        return last
            _time.sleep(interval)
        if unknown_polls and not last:
            raise K8sServingUnavailable(
                f"kubernetes API did not answer while verifying {ns}/{name}"
            )
        presence = self.isvc_presence(name, ns)
        if presence is False:
            raise K8sServingVerifyError(
                f"deploy pipeline reported success but InferenceService "
                f"{ns}/{name} does not exist (apply skipped or object deleted)"
            )
        reason = str(last.get("error", "") or "") or "KServe Ready condition is not True"
        raise K8sServingVerifyError(
            f"InferenceService {ns}/{name} is not Ready after {timeout:.0f}s: {reason}"
        )

    def wait_until_absent(
        self, name: str, ns: str, timeout: float = 300.0, interval: float = 5.0
    ) -> None:
        """Verify an InferenceService is really gone after an undeploy.

        Only a confirmed 404 counts as absent; transient API errors (None)
        keep polling so a deletion failure can never be mistaken for success.
        Raises K8sServingUnavailable / K8sServingVerifyError on deadline.
        """
        import time as _time

        if not self.available:
            raise K8sServingUnavailable(f"kubernetes API unavailable; cannot verify {ns}/{name}")
        deadline = _time.time() + timeout
        last_presence: Optional[bool] = None
        unknown = 0
        while _time.time() < deadline:
            last_presence = self.isvc_presence(name, ns)
            if last_presence is False:
                self._cache.pop((name, ns), None)
                return
            if last_presence is None:
                unknown += 1
            else:
                unknown = 0
            _time.sleep(interval)
        if last_presence is None and unknown:
            raise K8sServingUnavailable(
                f"kubernetes API did not answer while verifying deletion of {ns}/{name}"
            )
        raise K8sServingVerifyError(
            f"undeploy pipeline reported success but InferenceService "
            f"{ns}/{name} still exists after {timeout:.0f}s"
        )


# Shared read-only client instance for the API/worker processes.
serving_reader = K8sServingReader()


class K8sNotebookError(RuntimeError):
    """Raised when a notebook-related kubernetes API call fails."""


class K8sNotebookClient:
    """Restricted kubernetes SDK wrapper for Notebook CR + PVC lifecycle."""

    def __init__(self) -> None:
        # Lazy, one-time config load; failures downgrade to "unavailable"
        # instead of crashing API startup on non-cluster dev boxes.
        self._loaded = False
        self._available = False
        self._custom: Optional[_k8s_client.CustomObjectsApi] = None
        self._core: Optional[_k8s_client.CoreV1Api] = None

    def _ensure(self) -> None:
        """Load kube config and build the SDK APIs exactly once."""
        if self._loaded:
            return
        self._loaded = True
        try:
            token_file = "/var/run/secrets/kubernetes.io/serviceaccount/token"
            if os.path.exists(token_file):
                _k8s_config.load_incluster_config()
            else:
                kubeconfig = os.environ.get("KUBECONFIG") or "/etc/rancher/k3s/k3s.yaml"
                _k8s_config.load_kube_config(config_file=kubeconfig)
            self._custom = _k8s_client.CustomObjectsApi()
            self._core = _k8s_client.CoreV1Api()
            self._available = True
            log.info("notebook k8s client ready (kube config loaded)")
        except Exception as e:  # noqa: BLE001 - any config failure => disabled
            log.warning("notebook k8s client unavailable: %s", e)
            self._available = False

    @property
    def available(self) -> bool:
        """Return True when the kubernetes API is reachable/configured."""
        self._ensure()
        return self._available

    def _apis(self) -> tuple[_k8s_client.CustomObjectsApi, _k8s_client.CoreV1Api]:
        """Return the SDK APIs, raising when cluster access is unavailable."""
        self._ensure()
        if not self._available or self._custom is None or self._core is None:
            raise K8sNotebookError(
                "interactive notebook workbench is unavailable: "
                "kubernetes access is not configured for this deployment"
            )
        return self._custom, self._core

    @staticmethod
    def _notfound(exc: ApiException) -> bool:
        """True for the SDK's 404 response."""
        return exc.status == 404

    # -- Notebook CRs --------------------------------------------------------

    def get_notebook(self, namespace: str, name: str) -> Optional[dict[str, Any]]:
        """Fetch one Notebook CR as a dict; None when it does not exist."""
        custom, _ = self._apis()
        try:
            return custom.get_namespaced_custom_object(
                NOTEBOOK_GROUP, NOTEBOOK_VERSION, namespace, NOTEBOOK_PLURAL, name
            )
        except ApiException as e:
            if self._notfound(e):
                return None
            raise K8sNotebookError(f"get notebook failed: {e.reason}") from e

    def list_notebooks(self, namespace: str) -> list[dict[str, Any]]:
        """List all Notebook CRs in one namespace."""
        custom, _ = self._apis()
        try:
            resp = custom.list_namespaced_custom_object(
                NOTEBOOK_GROUP, NOTEBOOK_VERSION, namespace, NOTEBOOK_PLURAL
            )
            return list(resp.get("items", []) or [])
        except ApiException as e:
            raise K8sNotebookError(f"list notebooks failed: {e.reason}") from e

    def create_notebook(self, namespace: str, body: dict[str, Any]) -> dict[str, Any]:
        """Create a Notebook CR from a full manifest dict."""
        custom, _ = self._apis()
        try:
            return custom.create_namespaced_custom_object(
                NOTEBOOK_GROUP, NOTEBOOK_VERSION, namespace, NOTEBOOK_PLURAL, body
            )
        except ApiException as e:
            raise K8sNotebookError(f"create notebook failed: {e.reason}") from e

    def replace_notebook(self, namespace: str, name: str, body: dict[str, Any]) -> dict[str, Any]:
        """PUT (replace) a Notebook CR; used for stop/start command swaps.

        A merge patch cannot safely replace one entry of the `containers`
        array, so callers GET the raw object, mutate it, and PUT it back
        (resourceVersion gives optimistic concurrency).
        """
        custom, _ = self._apis()
        try:
            return custom.replace_namespaced_custom_object(
                NOTEBOOK_GROUP, NOTEBOOK_VERSION, namespace, NOTEBOOK_PLURAL, name, body
            )
        except ApiException as e:
            raise K8sNotebookError(f"replace notebook failed: {e.reason}") from e

    def patch_notebook_annotations(
        self, namespace: str, name: str, annotations: dict[str, Any]
    ) -> dict[str, Any]:
        """Merge-patch metadata.annotations (null value removes a key)."""
        custom, _ = self._apis()
        body = {"metadata": {"annotations": annotations}}
        try:
            return custom.patch_namespaced_custom_object(
                NOTEBOOK_GROUP,
                NOTEBOOK_VERSION,
                namespace,
                NOTEBOOK_PLURAL,
                name,
                body,
                _request_timeout=15,
            )
        except ApiException as e:
            raise K8sNotebookError(f"patch notebook failed: {e.reason}") from e

    def delete_notebook(self, namespace: str, name: str) -> bool:
        """Delete a Notebook CR; returns False when it was already gone."""
        custom, _ = self._apis()
        try:
            custom.delete_namespaced_custom_object(
                NOTEBOOK_GROUP, NOTEBOOK_VERSION, namespace, NOTEBOOK_PLURAL, name
            )
            return True
        except ApiException as e:
            if self._notfound(e):
                return False
            raise K8sNotebookError(f"delete notebook failed: {e.reason}") from e

    # -- PVCs ----------------------------------------------------------------

    def create_pvc(self, namespace: str, body: dict[str, Any]) -> dict[str, Any]:
        """Create a workspace PersistentVolumeClaim (409 = already exists)."""
        _, core = self._apis()
        try:
            return core.create_namespaced_persistent_volume_claim(namespace, body)
        except ApiException as e:
            # Already-exists is fine: notebook workspaces survive CR delete.
            if e.status == 409:
                return {}
            raise K8sNotebookError(f"create pvc failed: {e.reason}") from e

    def delete_pvc(self, namespace: str, name: str) -> bool:
        """Delete a PVC; returns False when it did not exist."""
        _, core = self._apis()
        try:
            core.delete_namespaced_persistent_volume_claim(name, namespace)
            return True
        except ApiException as e:
            if self._notfound(e):
                return False
            raise K8sNotebookError(f"delete pvc failed: {e.reason}") from e


# -- Training pod logs ------------------------------------------------------

# Kubeflow Trainer (JobSet-backed) CRD coordinates.
TRAINJOB_GROUP = "trainer.kubeflow.org"
TRAINJOB_VERSION = "v1alpha1"
TRAINJOB_PLURAL = "trainjobs"

# Label train_step.py stamps on every TrainJob it creates; its value is the
# NereidForge job id, which makes pod discovery independent of naming schemes.
ARTIFACT_LABEL = "nereidforge/artifact-id"
# JobSet propagates this label onto every trainer pod (value = TrainJob name).
JOBSET_NAME_LABEL = "jobset.sigs.k8s.io/jobset-name"


class K8sTrainLogError(RuntimeError):
    """Raised when reading trainer pod logs fails."""


class K8sTrainLogClient:
    """Read-only access to TrainJob pods and their logs in a training namespace.

    Training runs are created by KFP with the tenant's `default-editor` SA in
    the shared system namespace; the NF backend SA only needs list/get on
    trainjobs+pods and get on pods/log (see training-logs-rbac.yaml).
    """

    def __init__(self, namespace: str = "nereidforge") -> None:
        # Mirror K8sNotebookClient's lazy config load so a non-cluster dev box
        # downgrades to unavailable instead of failing at construction.
        self.namespace = namespace
        self._loaded = False
        self._available = False
        self._custom: Optional[_k8s_client.CustomObjectsApi] = None
        self._core: Optional[_k8s_client.CoreV1Api] = None

    def _ensure(self) -> None:
        """Load kube config and build the SDK APIs exactly once."""
        if self._loaded:
            return
        self._loaded = True
        try:
            token_file = "/var/run/secrets/kubernetes.io/serviceaccount/token"
            if os.path.exists(token_file):
                _k8s_config.load_incluster_config()
            else:
                kubeconfig = os.environ.get("KUBECONFIG") or "/etc/rancher/k3s/k3s.yaml"
                _k8s_config.load_kube_config(config_file=kubeconfig)
            self._custom = _k8s_client.CustomObjectsApi()
            self._core = _k8s_client.CoreV1Api()
            self._available = True
            log.info("training-log k8s client ready")
        except Exception as e:  # noqa: BLE001 - config failure => disabled
            log.warning("training-log k8s client unavailable: %s", e)
            self._available = False

    @property
    def available(self) -> bool:
        """Return True when the kubernetes API is reachable/configured."""
        self._ensure()
        return self._available

    def _apis(self) -> tuple[_k8s_client.CustomObjectsApi, _k8s_client.CoreV1Api]:
        """Return the SDK APIs, raising when cluster access is unavailable."""
        self._ensure()
        if not self._available or self._custom is None or self._core is None:
            raise K8sTrainLogError("kubernetes access unavailable")
        return self._custom, self._core

    def find_trainjobs(self, artifact_id: str) -> list[dict[str, Any]]:
        """Return TrainJob CRs labeled for one NF job id.

        Each item carries name/state (Succeeded/Failed/Running/"")/created.
        Discovery is label-driven (nereidforge/artifact-id=<job id>).
        """
        custom, _ = self._apis()
        try:
            resp = custom.list_namespaced_custom_object(
                TRAINJOB_GROUP,
                TRAINJOB_VERSION,
                self.namespace,
                TRAINJOB_PLURAL,
                label_selector=f"{ARTIFACT_LABEL}={artifact_id}",
            )
        except ApiException as e:
            raise K8sTrainLogError(f"list trainjobs failed: {e.reason}") from e
        out: list[dict[str, Any]] = []
        for item in resp.get("items", []):
            meta = item.get("metadata", {}) or {}
            state = ""
            for cond in (item.get("status", {}) or {}).get("conditions", []) or []:
                if cond.get("status") != "True":
                    continue
                ctype = cond.get("type", "")
                if ctype in ("Succeeded", "Complete"):
                    state = "Succeeded"
                elif ctype == "Failed":
                    state = "Failed"
                elif ctype in ("Running",) and not state:
                    state = "Running"
            out.append(
                {
                    "name": meta.get("name", ""),
                    "state": state,
                    "created": meta.get("creationTimestamp", ""),
                }
            )
        out.sort(key=lambda x: (x["created"], x["name"]))
        return out

    def delete_trainjobs(self, artifact_id: str) -> list[str]:
        """Delete every TrainJob CR labeled for one NF job id.

        Deletion cascades (background GC) to the JobSet, per-node Jobs and
        trainer pods. Returns the deleted TrainJob names. Raises on API error
        so the caller can keep its DB row and allow a retry.
        """
        custom, _ = self._apis()
        deleted: list[str] = []
        for tj in self.find_trainjobs(artifact_id):
            name = tj.get("name", "")
            if not name:
                continue
            try:
                custom.delete_namespaced_custom_object(
                    TRAINJOB_GROUP,
                    TRAINJOB_VERSION,
                    self.namespace,
                    TRAINJOB_PLURAL,
                    name,
                    body=_k8s_client.V1DeleteOptions(propagation_policy="Background"),
                )
                deleted.append(name)
            except ApiException as e:
                if e.status == 404:
                    continue
                raise K8sTrainLogError(f"delete trainjob {name} failed: {e.reason}") from e
        return deleted

    def find_trainer_pods(self, artifact_id: str) -> list[dict[str, Any]]:
        """Return trainer pods created for one NF job id.

        Discovery is label-driven: train_step labels the TrainJob with
        nereidforge/artifact-id=<job id>, JobSet labels the pods with
        jobset-name=<TrainJob name>. Returns dicts with identity, phase, and
        trainer-container start/finish timestamps.
        """
        custom, core = self._apis()
        try:
            resp = custom.list_namespaced_custom_object(
                TRAINJOB_GROUP,
                TRAINJOB_VERSION,
                self.namespace,
                TRAINJOB_PLURAL,
                label_selector=f"{ARTIFACT_LABEL}={artifact_id}",
            )
        except ApiException as e:
            raise K8sTrainLogError(f"list trainjobs failed: {e.reason}") from e
        job_names = [
            (item.get("metadata", {}) or {}).get("name", "")
            for item in resp.get("items", [])
        ]
        job_names = [n for n in job_names if n]
        if not job_names:
            return []
        out: list[dict[str, Any]] = []
        for job_name in job_names:
            try:
                pods = core.list_namespaced_pod(
                    self.namespace,
                    label_selector=f"{JOBSET_NAME_LABEL}={job_name}",
                )
            except ApiException as e:
                raise K8sTrainLogError(f"list pods failed: {e.reason}") from e
            for p in pods.items:
                ctrs = (p.spec.containers or [])
                container = ctrs[0].name if ctrs else ""
                statuses = p.status.container_statuses or []
                status = next((s for s in statuses if s.name == container), None)
                terminated = status.state.terminated if status else None
                running = status.state.running if status else None
                # Container bounds exclude queueing and image scheduling delay.
                # They remain available after completion and cover BYOT images
                # whose metrics.json omits the optional train_time_s field.
                started_at = (
                    (terminated and terminated.started_at)
                    or (running and running.started_at)
                    or p.status.start_time
                )
                finished_at = terminated.finished_at if terminated else None
                out.append(
                    {
                        "name": p.metadata.name,
                        "phase": p.status.phase or "",
                        "container": container,
                        "job": job_name,
                        "node": p.spec.node_name or "",
                        "started": started_at.isoformat() if started_at else "",
                        "finished": finished_at.isoformat() if finished_at else "",
                    }
                )
        # Oldest pod first so multi-attempt logs read in chronological order.
        out.sort(key=lambda x: (x["started"], x["name"]))
        return out

    def read_pod_logs(
        self,
        pod_name: str,
        container: str = "",
        tail_lines: int = 400,
        since_seconds: Optional[int] = None,
    ) -> str:
        """Return the last tail_lines of stdout/stderr for one trainer pod."""
        _, core = self._apis()
        try:
            return core.read_namespaced_pod_log(
                name=pod_name,
                namespace=self.namespace,
                container=container or None,
                tail_lines=tail_lines,
                since_seconds=since_seconds,
                timestamps=False,
                _request_timeout=20,
            )
        except ApiException as e:
            if e.status == 404:
                return ""
            raise K8sTrainLogError(f"read pod logs failed: {e.reason}") from e
