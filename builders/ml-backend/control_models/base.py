# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

import os
import logging
from typing import Any, Optional

from pydantic import BaseModel
from typing import List, Dict, ClassVar

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import DATA_UNDEFINED_NAME
from label_studio_sdk.label_interface.control_tags import ControlTag
from label_studio_sdk.label_interface import LabelInterface


# use matplotlib plots for debug
DEBUG_PLOT = os.getenv("DEBUG_PLOT", "false").lower() in ["1", "true"]
MODEL_SCORE_THRESHOLD = float(os.getenv("MODEL_SCORE_THRESHOLD", 0.5))
DEFAULT_MODEL_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MODEL_ROOT = os.getenv("MODEL_ROOT", DEFAULT_MODEL_ROOT)
os.makedirs(MODEL_ROOT, exist_ok=True)
# if true, allow to use custom model path from the control tag in the labeling config
ALLOW_CUSTOM_MODEL_PATH = os.getenv("ALLOW_CUSTOM_MODEL_PATH", "true").lower() in [
    "1",
    "true",
]
# SeaweedFS S3 gateway. The Service is named `seaweedfs-s3`; a bare
# `seaweedfs` name does not resolve inside the cluster.
S3_ENDPOINT_DEFAULT = "http://seaweedfs-s3.nereidforge.svc.cluster.local:8333"

# Media cache lives in the container writable layer (emptied on pod restart),
# but a long-lived pod pre-annotating many images would otherwise accumulate
# files without bound. Downloaded images go to a dedicated scratch dir (kept
# separate from MODEL_ROOT, which holds model weights and is never pruned),
# are capped by size, and evicted oldest-first.
MEDIA_CACHE_DIR = os.getenv("MEDIA_CACHE_DIR", "/tmp/ls-media")
os.makedirs(MEDIA_CACHE_DIR, exist_ok=True)
MEDIA_CACHE_MAX_BYTES = int(os.getenv("MEDIA_CACHE_MAX_BYTES", str(2 * 1024 * 1024 * 1024)))

# Inference is ONNX-only: the backend never imports a training framework.
# onnx_predict.py owns the session cache (bounded by onnxruntime itself), so
# there is no second model-handle LRU here.
logger = logging.getLogger(__name__)


def _prune_media_cache() -> None:
    """Evict oldest cached media files until under the size cap.

    Scans MEDIA_CACHE_DIR (the scratch dir for downloaded task images); model
    weights under MODEL_ROOT are never touched. Called after each media
    download so the container disk cannot fill up over a long pre-annotation run.
    """
    try:
        entries = []
        total = 0
        for name in os.listdir(MEDIA_CACHE_DIR):
            path = os.path.join(MEDIA_CACHE_DIR, name)
            if not os.path.isfile(path):
                continue
            size = os.path.getsize(path)
            total += size
            entries.append((os.path.getmtime(path), size, path))
        if total <= MEDIA_CACHE_MAX_BYTES:
            return
        entries.sort(key=lambda e: e[0])  # oldest mtime first
        for _mtime, size, path in entries:
            if total <= MEDIA_CACHE_MAX_BYTES:
                break
            try:
                os.remove(path)
                total -= size
                logger.info("media cache evict: %s (%d bytes freed)", path, size)
            except OSError:
                pass
    except Exception:
        # Cache pruning is best-effort and must never break inference.
        logger.debug("media cache prune failed", exc_info=True)


def get_bool(attr, attr_name, default="false"):
    return attr.get(attr_name, default).lower() in ["1", "true", "yes"]


class ControlModel(BaseModel):
    """
    Represents a control tag in Label Studio, which is associated with a specific type of labeling task
    and is used to generate predictions using an ONNX model artifact.

    The backend is deliberately framework-free at inference time: models are
    consumed as ONNX graphs plus the NF manifest (`metadata.json`), never as a
    trainer-specific checkpoint handle.

    Attributes:
        type (str): Type of the control, e.g., RectangleLabels, Choices, etc.
        control (ControlTag): The actual control element from the Label Studio configuration.
        from_name (str): The name of the control tag, used to link the control to the data.
        to_name (str): The name of the data field that this control is associated with.
        value (str): The value name from the object that this control operates on, e.g., an image or text field.
        model (object): Always None; kept so existing constructors stay compatible.
        model_path (str): Reference to the ONNX artifact (s3://, MODEL_ROOT-relative or absolute).
        model_score_threshold (float): Threshold for prediction scores; predictions below this value will be ignored.
        label_map (Optional[Dict[str, str]]): A mapping of model labels to Label Studio labels.
    """

    type: ClassVar[str]
    control: ControlTag
    from_name: str
    to_name: str
    value: str
    # Always None: the ONNX file path lives in `_resolved_model_path`.
    model: Optional[Any] = None
    model_path: ClassVar[str]
    model_score_threshold: float = 0.5
    label_map: Optional[Dict[str, str]] = {}
    label_studio_ml_backend: LabelStudioMLBase
    project_id: Optional[str] = None

    def __init__(self, **data):
        super().__init__(**data)

    @classmethod
    def is_control_matched(cls, control) -> bool:
        """Check if the control tag matches the model type.
        Args:
            control (ControlTag): The control tag from the Label Studio Interface.
        """
        raise NotImplementedError("This method should be overridden in derived classes")

    @staticmethod
    def get_from_name_for_label_map(
        label_interface: LabelInterface, target_name: str
    ) -> str:
        """Get the 'from_name' attribute for the label map building."""
        return target_name

    @classmethod
    def create(cls, mlbackend: LabelStudioMLBase, control: ControlTag):
        """Factory method to create an instance of a specific control model class.
        Args:
            mlbackend (LabelStudioMLBase): The ML backend instance.
            control (ControlTag): The control tag from the Label Studio Interface.
        """
        from_name = control.name
        to_name = control.to_name[0]
        value = control.objects[0].value_name

        # if skip is true, don't process this control
        if get_bool(control.attr, "model_skip", "false"):
            logger.info(
                f"Skipping control tag '{control.tag}' with name '{from_name}', model_skip=true found"
            )
            return None
        # read threshold attribute from the control tag, e.g.: <RectangleLabels model_score_threshold="0.5">
        model_score_threshold = float(
            control.attr.get("model_score_threshold")
            or control.attr.get(
                "score_threshold"
            )  # not recommended option, use `model_score_threshold`
            or MODEL_SCORE_THRESHOLD
        )
        # read `model_path` attribute from the control tag
        model_path = (
            ALLOW_CUSTOM_MODEL_PATH and control.attr.get("model_path")
        ) or cls.model_path

        # Inference is ONNX-only. Resolve the file (s3:// downloads handled
        # inside) and expose the label space from the manifest/embedded names.
        if not isinstance(model_path, str) or not model_path:
            raise ValueError(
                f"no ONNX model selected for control '{from_name}'. Upload a "
                "model in the NF UI and select it for this project."
            )
        if not model_path.lower().endswith(".onnx"):
            raise ValueError(
                f"unsupported model artifact '{model_path}': the ML backend "
                "consumes ONNX exports only (export the model in the NF UI)."
            )
        from onnx_predict import read_onnx_manifest, read_onnx_names

        resolved = cls.resolve_local_model_file(model_path)
        if not resolved:
            raise ValueError(f"onnx model not available: {model_path}")
        manifest = read_onnx_manifest(resolved)
        onnx_names = read_onnx_names(resolved) or list(manifest.get("labels") or [])
        cfg_label_names = (
            list(control.labels_attrs.keys()) if hasattr(control, "labels_attrs") else []
        )
        label_map = {
            n: n for n in onnx_names if control.labels_attrs and n in control.labels_attrs
        }
        if not label_map and cfg_label_names:
            # Class-index order is the fallback contract (documented in the
            # per-control predictors, which map cls -> config label position).
            label_map = {n: n for n in cfg_label_names}
        instance = cls(
            control=control,
            from_name=from_name,
            to_name=to_name,
            value=value,
            model=None,
            model_score_threshold=model_score_threshold,
            label_map=label_map,
            label_studio_ml_backend=mlbackend,
            project_id=mlbackend.project_id,
        )
        instance._resolved_model_path = resolved  # type: ignore[attr-defined]
        instance._model_manifest = manifest  # type: ignore[attr-defined]
        return instance

    @classmethod
    def resolve_local_model_file(cls, filename: str) -> str:
        """Resolve a model reference to a local file (s3:// downloads, MODEL_ROOT)."""
        env_path = os.getenv("MODEL_PATH", "")
        target = filename
        if env_path and not filename.startswith("s3://"):
            if env_path.startswith("s3://"):
                local = os.path.join(MODEL_ROOT, "model.onnx")
                if cls._download_s3_model(env_path, local):
                    cls._download_sidecar_manifest(env_path, local)
                    target = local
            else:
                target = env_path
        if target.startswith("s3://"):
            import hashlib

            digest = hashlib.sha1(target.encode()).hexdigest()[:16]
            # Preserve the original extension so onnxruntime accepts the path.
            suffix = os.path.splitext(target[5:].split("/", 1)[-1])[1][:16] or ".onnx"
            local = os.path.join(MODEL_ROOT, f"s3-{digest}{suffix}")
            if cls._download_s3_model(target, local):
                cls._download_sidecar_manifest(target, local)
                return local
            return ""
        if os.path.isabs(target):
            return target if os.path.exists(target) else ""
        path = os.path.join(MODEL_ROOT, target)
        return path if os.path.exists(path) else ""

    @staticmethod
    def _as_s3_uri(uri: str) -> str:
        """Normalize an S3 reference to the canonical `s3://bucket/key` form.

        Accepts both the plain `s3://` form and the
        `http://<s3-endpoint>/<bucket>/<key>` form that the NereidForge
        backend produces for in-cluster access. Presigned URLs carry their
        signature in a `?X-Amz-...` query string, which must be stripped before
        the path is treated as an object key (otherwise boto3 HeadObject looks
        up `...jpg?X-Amz-...` and 404s). Returns "" when the input is not an
        S3 reference at all.
        """
        if not isinstance(uri, str):
            return ""
        # Drop any query/fragment: presigned GET URLs append ?X-Amz-* signing
        # parameters that are not part of the object key.
        from urllib.parse import urlsplit

        parts = urlsplit(uri)
        path = parts.path
        if uri.startswith("s3://"):
            # urlsplit puts the bucket in netloc and the key in path.
            bucket = parts.netloc
            key = path.lstrip("/")
            if bucket and key:
                return f"s3://{bucket}/{key}"
            return ""
        endpoint = os.getenv("S3_ENDPOINT", S3_ENDPOINT_DEFAULT).rstrip("/")
        if endpoint:
            ep_host = urlsplit(endpoint).netloc
            if parts.netloc and parts.netloc == ep_host:
                bucket, sep, key = path.lstrip("/").partition("/")
                if bucket and sep and key:
                    return f"s3://{bucket}/{key}"
        return ""

    @classmethod
    def _download_s3_object(cls, s3_uri: str, local: str) -> bool:
        """Download any object from SeaweedFS S3 to a local path.

        Signed requests are mandatory: SeaweedFS runs with an S3 config and
        rejects unsigned reads with 403, so anonymous access is not an option.
        Credentials come from the S3_ACCESS_KEY/S3_SECRET_KEY env.
        Returns True when the file exists locally afterwards.
        """
        if os.path.exists(local):
            return True
        try:
            import boto3

            rest = s3_uri[len("s3://"):]
            bucket, key = rest.split("/", 1)
            s3 = boto3.client(
                "s3",
                endpoint_url=os.getenv("S3_ENDPOINT", S3_ENDPOINT_DEFAULT),
                aws_access_key_id=os.getenv("S3_ACCESS_KEY", ""),
                aws_secret_access_key=os.getenv("S3_SECRET_KEY", ""),
                region_name="us-east-1",
            )
            os.makedirs(os.path.dirname(local), exist_ok=True)
            s3.download_file(bucket, key, local)
            logger.info(f"Downloaded s3 object from {s3_uri}")
            return True
        except Exception as e:
            logger.warning(f"Failed to download object from {s3_uri}: {e}")
            if os.path.exists(local):
                os.remove(local)
            return False

    @classmethod
    def _download_s3_model(cls, s3_uri: str, local: str) -> bool:
        """Download model weights from SeaweedFS S3 (name kept for sam_segment.py)."""
        return cls._download_s3_object(s3_uri, local)

    @classmethod
    def _download_sidecar_manifest(cls, s3_uri: str, local_model: str) -> None:
        """Best-effort fetch of the artifact manifest next to an ONNX file.

        Artifacts are stored as `<prefix>/model.onnx` + `<prefix>/metadata.json`
        (artifact-spec). The local copies share the model stem so
        `read_onnx_manifest()` finds `<stem>.metadata.json` without guessing.
        """
        if not s3_uri.startswith("s3://"):
            return
        local_md = os.path.splitext(local_model)[0] + ".metadata.json"
        if os.path.isfile(local_md):
            return
        prefix = s3_uri.rsplit("/", 1)[0]
        stem = os.path.basename(s3_uri).rsplit(".", 1)[0]
        for name in (f"{stem}.metadata.json", "metadata.json"):
            if cls._download_s3_object(f"{prefix}/{name}", local_md):
                return

    @classmethod
    def resolve_s3_media(cls, uri: str) -> str:
        """Fetch media stored in SeaweedFS S3 with a signed request.

        LabelStudioMLBase.get_local_path() downloads with a plain unsigned
        HTTP GET, which SeaweedFS answers with 403 once an S3 config with
        credentials is installed. Everything reachable over the S3 endpoint
        is therefore fetched here instead.

        Returns a local path, or "" when the URI is not an S3 reference or the
        download failed (callers fall back to the Label Studio downloader).
        """
        s3_uri = cls._as_s3_uri(uri)
        if not s3_uri:
            return ""
        import hashlib

        digest = hashlib.sha1(s3_uri.encode()).hexdigest()[:16]
        suffix = os.path.splitext(s3_uri)[1][:16] or ".bin"
        local = os.path.join(MEDIA_CACHE_DIR, f"media-{digest}{suffix}")
        if not cls._download_s3_object(s3_uri, local):
            return ""
        # Bound the on-disk media cache so a long-lived pod never fills its disk.
        _prune_media_cache()
        return local

    def debug_plot(self, image):
        if not DEBUG_PLOT:
            return

        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 10))
        plt.imshow(image[..., ::-1])
        plt.axis("off")
        plt.title(self.type)
        plt.show()

    def predict_regions(self, path) -> List[Dict]:
        """Predict regions in the image using the ONNX model.
        Args:
            path (str): Path to the file with media
        """
        raise NotImplementedError("This method should be overridden in derived classes")

    def fit(self, event, data, **kwargs):
        """Fit the model."""
        logger.warning("The fit method is not implemented for this control model")
        return False

    def get_path(self, task):
        task_path = task["data"].get(self.value) or task["data"].get(
            DATA_UNDEFINED_NAME
        )
        if task_path is None:
            raise ValueError(
                f"Can't load path using key '{self.value}' from task {task}"
            )
        if not isinstance(task_path, str):
            raise ValueError(f"Path should be a string, but got {task_path}")

        # Media in SeaweedFS must be fetched with a signed request; the Label
        # Studio downloader below uses an unsigned GET and gets a 403.
        local = self.resolve_s3_media(task_path)
        if local:
            logger.debug(f"load_image (s3): {task_path} => {local}")
            return local

        # try path as local file or try to load it from Label Studio instance/download via http
        path = (
            task_path
            if os.path.exists(task_path)
            else self.label_studio_ml_backend.get_local_path(task_path, task_id=task.get("id"))
        )
        logger.debug(f"load_image: {task_path} => {path}")
        return path

    def model_label_name(self, name: Optional[str], cls_index: int) -> Optional[str]:
        """Resolve the model-side label for a prediction.

        Prefers the name embedded in the ONNX graph/manifest; falls back to the
        config label order (the documented contract when a graph carries no
        names), which is also how the NF trainer exports class indices.
        """
        if name:
            return name
        labels = list((self.label_map or {}).keys())
        if 0 <= int(cls_index) < len(labels):
            return labels[int(cls_index)]
        return None

    def ls_label(self, name: Optional[str], cls_index: int) -> Optional[str]:
        """Map a model-side label to the Label Studio label, or None."""
        model_name = self.model_label_name(name, cls_index)
        if not model_name:
            return None
        return (self.label_map or {}).get(model_name)

    def __str__(self):
        """Return a string with full representation of the control tag."""
        return (
            f"{self.type} from_name={self.from_name}, "
            f"label_map={self.label_map}, model_score_threshold={self.model_score_threshold}"
        )

    class Config:
        arbitrary_types_allowed = True
        protected_namespaces = ("__.*__", "_.*")  # Excludes 'model_'
