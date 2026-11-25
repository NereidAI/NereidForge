# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Interactive notebook workbench service (Kubeflow Notebook CRD).

Manages Jupyter notebook SERVERS on top of the cluster's notebook-controller
(NOTE: distinct from notebook batch jobs, which run .ipynb files through
Kale/KFP). One Notebook CR = one Jupyter pod with its own PVC:

    create  -> Notebook CR + workspace PVC (no Ingress; access is proxied)
    stop    -> swap the container command to `sleep infinity` (the pod idles
               while the PVC and all notebook state survive)
    start   -> restore the original command from a CR annotation
    delete  -> remove the CR (PVC kept unless delete_pvc=true)

Access goes through the NF API reverse proxy (/notebook-servers/{name}/...)
because there is no wildcard DNS for per-notebook hosts; the proxy streams
HTTP and websockets to the notebook's in-cluster service. Cluster access is
narrowly scoped (app.clients.k8s + namespaced Role on the backend SA).
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

from app.clients.k8s import K8sNotebookClient, K8sNotebookError, sanitize_k8s_name
from app.core.images import nf_image

logger = logging.getLogger(__name__)

# Default tenant's notebook namespace (other tenants map to nereidforge-<t>).
DEFAULT_NB_NAMESPACE = "nereidforge"

# Jupyter workbench image (keeps its own py3.12 tag by product decision).
DEFAULT_IMAGE = nf_image("prediction/jupyter-base", "py3.12")

# GPU resource keys allowed on notebook create (anything else is rejected).
GPU_RESOURCE_KEYS = {
    "nvidia": "nvidia.com/gpu",
    "amd": "amd.com/gpu",
    "intel": "gpu.intel.com/xe",
}

# Native notebook-controller pause annotation: when present the controller
# scales the notebook StatefulSet to 0 (PVC kept); removing it scales back.
# Verified on notebook-controller v1.11.0 (the newer
# "notebooks.kubeflow.org/stop-annotation" key is NOT honored by this build).
STOP_ANNOTATION = "kubeflow-resource-stopped"


def notebook_namespace(tenant_id: str) -> str:
    """Resolve the notebook namespace for a tenant id."""
    if not tenant_id or tenant_id == "default":
        return DEFAULT_NB_NAMESPACE
    return f"nereidforge-{sanitize_k8s_name(tenant_id)}"


def _nb_svc_name(name: str) -> str:
    """notebook-controller creates a Service named after the notebook.

    Newer controller releases (verified on the deployed version) name it
    `<name>`; older releases used `<name>-service`.
    """
    return name


class NotebookUnavailable(K8sNotebookError):
    """Raised when cluster access is not configured (maps to HTTP 501)."""


class NotebookServerService:
    """CRUD + lifecycle for interactive notebook servers."""

    def __init__(self, client: Optional[K8sNotebookClient] = None) -> None:
        # Injectable client keeps tests/unit callers independent of kube config.
        self.k8s = client or K8sNotebookClient()

    def _check_available(self) -> None:
        """Raise a clear 501-style error when k8s access is not configured."""
        if not self.k8s.available:
            raise NotebookUnavailable(
                "interactive notebook workbench is unavailable: "
                "kubernetes access is not configured for this deployment"
            )

    # -- CRUD ----------------------------------------------------------------

    def create(
        self,
        tenant_id: str,
        name: str,
        image: str,
        cpu: str,
        memory: str,
        gpus: int,
        gpu_type: str,
        pvc_size: str,
        jupyter_token: str,
        created_by: str = "anonymous",
    ) -> dict[str, Any]:
        """Create a Notebook CR + workspace PVC; returns the API summary."""
        self._check_available()
        ns = notebook_namespace(tenant_id)
        nb_name = sanitize_k8s_name(name)
        if not nb_name:
            raise ValueError("invalid notebook name")
        if self.k8s.get_notebook(ns, nb_name) is not None:
            raise ValueError(f"notebook {nb_name} already exists")

        image = image or DEFAULT_IMAGE
        pvc_size = pvc_size or "10Gi"
        base_url = f"/notebook-servers/{nb_name}"

        container: dict[str, Any] = {
            "name": nb_name,
            "image": image,
            "env": [
                {"name": "JUPYTER_ENABLE_LAB", "value": "yes"},
                {"name": "JUPYTER_TOKEN", "value": jupyter_token or "nereidforge"},
            ],
            # base_url equals the external proxy prefix, so Jupyter emits
            # asset/API links that resolve through the proxy with no rewrite.
            "command": [
                "sh", "-c",
                "jupyter lab --notebook-dir=/home/jovyan --ip=0.0.0.0 --no-browser "
                "--allow-root --port=8888 --NotebookApp.token='' "
                "--NotebookApp.password='' --ServerApp.disable_check_xsrf=True "
                f"--ServerApp.base_url={base_url} ",
            ],
            "ports": [{"containerPort": 8888, "name": "notebook-port", "protocol": "TCP"}],
            "resources": {
                "requests": {"cpu": cpu or "500m", "memory": memory or "1Gi"},
                "limits": {"cpu": cpu or "500m", "memory": memory or "1Gi"},
            },
            "volumeMounts": [{"mountPath": "/home/jovyan", "name": "workspace"}],
        }
        if gpus and int(gpus) > 0:
            # Accept either a vendor alias (nvidia/amd/intel) or a raw key.
            limit_key = GPU_RESOURCE_KEYS.get(gpu_type, gpu_type)
            if limit_key not in GPU_RESOURCE_KEYS.values():
                raise ValueError(f"unsupported gpu resource: {gpu_type}")
            container["resources"]["limits"][limit_key] = str(int(gpus))

        nb = {
            "apiVersion": "kubeflow.org/v1",
            "kind": "Notebook",
            "metadata": {
                "name": nb_name,
                "namespace": ns,
                "labels": {"app": nb_name, "nereidforge/created-by": created_by or "anonymous"},
            },
            "spec": {
                "template": {
                    "spec": {
                        # default-editor is provisioned by KF Profiles in tenant ns.
                        "serviceAccountName": "default-editor",
                        "containers": [container],
                        "volumes": [
                            {
                                "name": "workspace",
                                "persistentVolumeClaim": {
                                    "claimName": f"{nb_name}-workspace"
                                },
                            }
                        ],
                    }
                }
            },
        }
        pvc = {
            "apiVersion": "v1",
            "kind": "PersistentVolumeClaim",
            "metadata": {"name": f"{nb_name}-workspace", "namespace": ns},
            "spec": {
                "accessModes": ["ReadWriteOnce"],
                "resources": {"requests": {"storage": pvc_size}},
            },
        }

        # PVC first: the notebook-controller's pod cannot start without it.
        self.k8s.create_pvc(ns, pvc)
        self.k8s.create_notebook(ns, nb)
        return {"name": nb_name, "status": "creating", "url_prefix": f"/notebook-servers/{nb_name}"}

    def list(self, tenant_id: str) -> list[dict[str, Any]]:
        """List notebook servers with resource/config summary."""
        self._check_available()
        ns = notebook_namespace(tenant_id)
        out: list[dict[str, Any]] = []
        for it in self.k8s.list_notebooks(ns):
            md = it.get("metadata", {}) or {}
            name = md.get("name", "")
            spec = ((it.get("spec", {}) or {}).get("template", {}) or {}).get("spec", {})
            ctrs = spec.get("containers", []) or [{}]
            c = ctrs[0]
            res = (c.get("resources") or {}).get("limits", {}) or {}
            stopped = bool((md.get("annotations", {}) or {}).get(STOP_ANNOTATION))
            conditions = it.get("status", {}).get("conditions", []) or []
            ready = any(
                str(cc.get("type", "")).lower() == "ready"
                and str(cc.get("status", "")).lower() == "true"
                for cc in conditions
            )
            out.append(
                {
                    "name": name,
                    "image": c.get("image", ""),
                    "cpu": res.get("cpu", ""),
                    "memory": res.get("memory", ""),
                    "gpus": {k: v for k, v in res.items() if "gpu" in k or k.endswith("/xe")},
                    "state": "stopped" if stopped else ("running" if ready else "starting"),
                    "created_at": md.get("creationTimestamp", ""),
                    "created_by": (md.get("labels", {}) or {}).get("nereidforge/created-by", ""),
                    "url_prefix": f"/notebook-servers/{name}",
                }
            )
        return out

    def get(self, tenant_id: str, name: str) -> Optional[dict[str, Any]]:
        """Get one notebook server summary."""
        for nb in self.list(tenant_id):
            if nb["name"] == sanitize_k8s_name(name):
                return nb
        return None

    def stop(self, tenant_id: str, name: str) -> bool:
        """Pause the notebook: annotate so the controller scales to 0.

        The workspace PVC and all notebook state are preserved; unlike a
        container command swap, the native annotation makes the controller
        actually scale the StatefulSet down.
        """
        self._check_available()
        ns = notebook_namespace(tenant_id)
        nb_name = sanitize_k8s_name(name)
        raw = self.k8s.get_notebook(ns, nb_name)
        if raw is None:
            return False
        anns = (raw.get("metadata", {}) or {}).get("annotations", {}) or {}
        if not anns.get(STOP_ANNOTATION):
            self.k8s.patch_notebook_annotations(ns, nb_name, {STOP_ANNOTATION: "true"})
        return True

    def start(self, tenant_id: str, name: str) -> bool:
        """Resume a paused notebook by removing the stop annotation."""
        self._check_available()
        ns = notebook_namespace(tenant_id)
        nb_name = sanitize_k8s_name(name)
        raw = self.k8s.get_notebook(ns, nb_name)
        if raw is None:
            return False
        anns = (raw.get("metadata", {}) or {}).get("annotations", {}) or {}
        if STOP_ANNOTATION in anns:
            # A null value in a JSON merge patch deletes the annotation key.
            self.k8s.patch_notebook_annotations(ns, nb_name, {STOP_ANNOTATION: None})
        return True

    def delete(self, tenant_id: str, name: str, delete_pvc: bool = False) -> bool:
        """Delete the notebook CR (and its PVC when requested)."""
        self._check_available()
        ns = notebook_namespace(tenant_id)
        nb_name = sanitize_k8s_name(name)
        if self.k8s.get_notebook(ns, nb_name) is None:
            return False
        self.k8s.delete_notebook(ns, nb_name)
        if delete_pvc:
            try:
                self.k8s.delete_pvc(ns, f"{nb_name}-workspace")
            except K8sNotebookError as e:
                logger.warning("pvc delete failed for %s: %s", nb_name, e)
        return True

    # -- proxy ---------------------------------------------------------------

    def proxy_target(self, tenant_id: str, name: str) -> tuple[str, Optional[str]]:
        """Return (http_base_url, jupyter_token) for a notebook's svc.

        Prod resolves in-cluster DNS directly; on a dev box (API on the host)
        NEREIDFORGE_NOTEBOOK_DEV_FWD maps "name=hostPort" pairs to local
        kubectl port-forwards.
        """
        self._check_available()
        ns = notebook_namespace(tenant_id)
        nb_name = sanitize_k8s_name(name)
        raw = self.k8s.get_notebook(ns, nb_name)
        if raw is None:
            raise K8sNotebookError(f"notebook {nb_name} not found")
        ctr = raw["spec"]["template"]["spec"]["containers"][0]
        token = next(
            (e.get("value") for e in (ctr.get("env") or []) if e.get("name") == "JUPYTER_TOKEN"),
            None,
        )
        dev_fwd = os.environ.get("NEREIDFORGE_NOTEBOOK_DEV_FWD", "") or os.environ.get(
            "NOTEBOOK_DEV_FWD", ""
        )
        if dev_fwd:
            mapping = dict(kv.split("=", 1) for kv in dev_fwd.split(",") if "=" in kv)
            if nb_name in mapping:
                return f"http://127.0.0.1:{mapping[nb_name]}", token
        return f"http://{_nb_svc_name(nb_name)}.{ns}.svc.cluster.local:80", token
