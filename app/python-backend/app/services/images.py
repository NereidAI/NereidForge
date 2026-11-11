# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Container image registry service: user-manageable train/prediction images.

Images live in the hub project under category prefixes (nereidforge/train/*,
nereidforge/prediction/*). This service manages the NF-side catalog:
list/create/update/delete entries, seed built-ins, and resolve which image a
training job or ML backend should use. It never talks to the registry itself
(pushing happens via builders/ scripts or manually).

The container registry endpoint/project come from system_config
(container_registry_host / container_registry_project, editable in
Settings → Container Registry); credentials live there too (admin-only
read). The Model Hub in Settings is a DIFFERENT thing (Kubeflow Model
Registry for model weights, not container images).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.images import GHCR, RELEASE_TAG, nf_image
from app.core.util import iso, new_id
from app.models.orm import ContainerImage
from app.services import trainer_profiles

logger = logging.getLogger(__name__)

CATEGORIES = ("train", "prediction")
# Default registry host/project for user-registered bare repos. Product
# images always use the public GHCR root (root treaty §18); an air-gapped site
# retargets pulls at the node containerd mirror, never via an edited manifest.
_root_parts = GHCR.rsplit("/", 1)
DEFAULT_REGISTRY_HOST = _root_parts[0]
DEFAULT_REGISTRY_PROJECT = _root_parts[1]
HUB_PREFIX = GHCR


def _repo(nested: str, tag: str = RELEASE_TAG) -> str:
    """_repo returns the flat GHCR repository (without a tag) for a built-in catalog entry."""
    return nf_image(nested, tag).rsplit(":", 1)[0]


def _registry_prefix(db: Session) -> str:
    """_registry_prefix resolves the hub project prefix from system_config with a built-in fallback."""
    try:
        row = db.execute(
            select(__import__("app.models.orm", fromlist=["SystemConfig"]).SystemConfig).where(
                __import__("app.models.orm", fromlist=["SystemConfig"]).SystemConfig.key == "platform"
            )
        ).scalar()
        cfg = (row.value if row else {}) or {}
        host = str(cfg.get("container_registry_host") or DEFAULT_REGISTRY_HOST).rstrip("/")
        project = str(cfg.get("container_registry_project") or DEFAULT_REGISTRY_PROJECT).strip("/")
        return f"{host}/{project}"
    except Exception:
        return f"{DEFAULT_REGISTRY_HOST}/{DEFAULT_REGISTRY_PROJECT}"


# Built-in images NF ships with. Repositories are the flat GHCR names resolved
# by nf_image(). Vision trainers are `trainer/torch-<vendor>` (torchvision + timm + smp) and
# `trainer/paddle-<vendor>`; every vendor stays its own image (ADR-012/014).
#
# Capabilities are NOT listed here any more: each component announces them in
# its own `trainer-profile.json` and the seed derives `task_types` /
# `entrypoint_hint` from that announcement (ADR-015). The seeds below only say
# where an image lives and which context it is built from — the same shape a
# user-registered image has, so nothing in this file can drift from an image.
_TORCH_TASKS: list[str] = []   # filled from the component announcement
_PADDLE_TASKS: list[str] = []  # filled from the component announcement
_TORCH_ENTRY = ""
_PADDLE_ENTRY = ""


def _announcement(build_context: str) -> tuple[dict[str, Any], list[str], str]:
    """_announcement returns (profile, task_types, entrypoint_hint) for a build context."""
    profile = trainer_profiles.builtin_profile(build_context) or {}
    return profile, trainer_profiles.task_ids(profile), trainer_profiles.entrypoint_hint(profile)


def _seed(
    name: str,
    build_context: str,
    *,
    tasks: Optional[list[str]] = None,
    hint: str = "",
    **fields: Any,
) -> dict[str, Any]:
    """Build one built-in seed spec from the component's own announcement.

    `tasks`/`hint` are explicit fallbacks for components that do not yet ship
    a `trainer-profile.json` (LLM images); the announcement always wins once a
    component publishes one, so no task list is ever hardcoded against a
    component that speaks for itself.
    """
    profile, announced_tasks, announced_hint = _announcement(build_context)
    return {
        "name": name,
        "build_context": build_context,
        "task_types": announced_tasks or list(tasks or []),
        "entrypoint_hint": announced_hint or hint,
        "profile": profile,
        **fields,
    }

BUILTIN_SEED = [
    _seed(
        "trainer-torch-cpu (torchvision + timm, CPU)",
        "builders/trainer-torch",
        category="train",
        repository=_repo("trainer/torch-cpu"),
        tag=RELEASE_TAG,
        description="PyTorch CPU: torchvision + timm + smp — classification, detection, instance/semantic segmentation, pose",
    ),
    _seed(
        "trainer-torch-cu (torchvision + timm, NVIDIA)",
        "builders/trainer-torch/Dockerfile.cu",
        category="train",
        repository=_repo("trainer/torch-cu"),
        tag=RELEASE_TAG,
        description="CUDA PyTorch + torchvision/timm: all vision tasks on NVIDIA GPUs",
    ),
    _seed(
        "trainer-torch-rocm (torchvision + timm, AMD)",
        "builders/trainer-torch/Dockerfile.rocm",
        category="train",
        repository=_repo("trainer/torch-rocm"),
        tag=RELEASE_TAG,
        description="ROCm/HIP PyTorch + torchvision/timm: all vision tasks on AMD RDNA GPUs",
    ),
    _seed(
        "trainer-torch-xpu (torchvision + timm, Intel)",
        "builders/trainer-torch/Dockerfile.xpu",
        category="train",
        repository=_repo("trainer/torch-xpu"),
        tag=RELEASE_TAG,
        description="Native torch.xpu + torchvision/timm: all vision tasks on Intel Arc",
    ),
    _seed(
        "trainer-paddle-cpu (PaddleDetection, CPU)",
        "builders/trainer-paddle",
        category="train",
        repository=_repo("trainer/paddle-cpu"),
        tag=RELEASE_TAG,
        description="PaddlePaddle CPU + PaddleX: object detection (permissive stack)",
    ),
    _seed(
        "trainer-paddle-cu (PaddleDetection, NVIDIA)",
        "builders/trainer-paddle/Dockerfile.cu",
        category="train",
        repository=_repo("trainer/paddle-cu"),
        tag=RELEASE_TAG,
        description="CUDA PaddlePaddle + PaddleX: object detection on NVIDIA GPUs (GPU wheel, not PyPI CPU build)",
    ),
    _seed(
        "trainer-llm-cu (LLaMA-Factory, NVIDIA)",
        "builders/trainer-llm-cu",
        tasks=["llm", "vlm"],
        hint="python /app/llm_train.py",
        category="train",
        repository=_repo("trainer/llm-cu"),
        tag=RELEASE_TAG,
        description="CUDA PyTorch + LLaMA-Factory: LLM/VLM SFT & LoRA on NVIDIA",
    ),
    _seed(
        "trainer-llm-rocm (LLaMA-Factory, AMD)",
        "builders/trainer-llm-rocm",
        tasks=["llm", "vlm"],
        hint="python /app/llm_train.py",
        category="train",
        repository=_repo("trainer/llm-rocm"),
        tag=RELEASE_TAG,
        description="ROCm/HIP PyTorch + LLaMA-Factory: LLM/VLM SFT & LoRA on AMD RDNA",
    ),
    _seed(
        "trainer-llm-xpu (transformers/peft, Intel)",
        "builders/trainer-llm-xpu",
        tasks=["llm", "vlm"],
        hint="python /app/llm_train.py (LLM) or /app/vlm_train.py (VLM)",
        category="train",
        repository=_repo("trainer/llm-xpu"),
        tag=RELEASE_TAG,
        description="Native torch.xpu transformers + peft QLoRA/VLM-LoRA on Intel Arc",
    ),
    _seed(
        "jupyter-base (notebook workbench, ML + Kale)",
        "builders/notebook-base",
        category="prediction",
        repository=_repo("prediction/jupyter-base", "py3.12"),
        tag="py3.12",
        description="JupyterLab + numpy/pandas/scikit/torch(CUDA-free)/opencv/paddle + kale+kfp: notebook workbench & Kale step (dual-use) image",
    ),
    _seed(
        "ml-backend (ONNX predict)",
        "builders/ml-backend",
        category="prediction",
        repository=_repo("ml-backend/ml-backend"),
        tag=RELEASE_TAG,
        description="Label Studio ML backend: ONNX detection/segmentation/classification/interactive, s3:// model_path",
    ),
]

# Trainer repositories shipped before the 2026-10 split (ADR-014 D-01). Jobs,
# clones and seeded DB rows may still carry these names; they resolve to the
# equivalent current image instead of failing to pull a deleted tag.
LEGACY_TRAINER_REPOS = {
    "trainer": "trainer/torch-cpu",
    "trainer/timm": "trainer/torch-cpu",
    "trainer/cv": "trainer/torch-cpu",
    "trainer/cv-cu": "trainer/torch-cu",
    "trainer/cv-rocm": "trainer/torch-rocm",
    "trainer/cv-xpu": "trainer/torch-xpu",
    # paddle-xpu never existed (zero xpu paddlepaddle builds); keep old saved
    # jobs scheduleable by routing Intel to torch-xpu (ADR-014 correction).
    "trainer/paddle-xpu": "trainer/torch-xpu",
}


def canonical_trainer_ref(image: str) -> str:
    """Map a legacy trainer reference to the current repository, keeping tag.

    Accepts both layouts (nested `.../nereidai/trainer/cv-cu:v1` and flat
    `ghcr.io/nereidai/trainer-cv-cu:v1`). Unknown/custom images pass through
    untouched.
    """
    ref = str(image or "").strip()
    if not ref:
        return ref
    tag = ""
    last_slash = ref.rfind("/")
    last_colon = ref.rfind(":")
    if last_colon > last_slash:
        ref, tag = ref[:last_colon], ref[last_colon:]
    for old, new in LEGACY_TRAINER_REPOS.items():
        for suffix in (f"/{old}", "/" + old.replace("/", "-")):
            if ref.endswith(suffix):
                base = nf_image(new).rsplit(":", 1)[0]
                return f"{base}{tag or ':' + RELEASE_TAG}"
    return str(image).strip()


def _img_dict(img: ContainerImage) -> dict[str, Any]:
    """Serialize a ContainerImage to the API response shape.

    The announcement (`profile`) and the facts derived from it travel together:
    `task_types` is what the platform schedules by, `default_archs` is what the
    UI preselects, and `extended` lists the capabilities that need an
    entitlement (ADR-015) so the UI can show them as locked instead of hiding
    them.
    """
    profile = img.profile or {}
    return {
        "id": img.id,
        "name": img.name,
        "category": img.category,
        "repository": img.repository,
        "tag": img.tag,
        "full_ref": f"{img.repository}:{img.tag}",
        "description": img.description,
        "build_context": img.build_context,
        "entrypoint_hint": img.entrypoint_hint,
        "task_types": img.task_types or [],
        "profile": profile,
        "profile_at": iso(img.profile_at),
        "profile_error": img.profile_error or "",
        "default_archs": {
            task: arch
            for task in (img.task_types or [])
            if (arch := trainer_profiles.default_arch(profile, task))
        },
        "extended": trainer_profiles.extended_capabilities(profile),
        "is_builtin": img.is_builtin,
        "created_by": img.created_by,
        "created_at": iso(img.created_at),
        "updated_at": iso(img.updated_at),
    }


class ImageService:
    """Manages the container image catalog (train / prediction)."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def seed_builtins(self, tenant_id: str) -> int:
        """Insert built-in images that are missing (idempotent, upgrades tags).

        Built-in rows whose repository is no longer shipped by the code
        (renamed/moved in a release) are retired here; otherwise a moved
        entry seeds a new row while the stale one stays forever (built-ins
        cannot be deleted through the API).
        """
        n = 0
        wanted = {spec["repository"] for spec in BUILTIN_SEED}
        for row in self.db.scalars(
            select(ContainerImage).where(
                ContainerImage.tenant_id == tenant_id,
                ContainerImage.is_builtin.is_(True),
            )
        ).all():
            if row.repository not in wanted:
                self.db.delete(row)
        for spec in BUILTIN_SEED:
            repo = spec["repository"]
            row = self.db.scalar(
                select(ContainerImage).where(
                    ContainerImage.tenant_id == tenant_id,
                    ContainerImage.repository == repo,
                )
            )
            if not row:
                self.db.add(ContainerImage(
                    id=new_id("img"),
                    tenant_id=tenant_id,
                    created_by="system",
                    profile_at=datetime.now(timezone.utc) if spec.get("profile") else None,
                    **spec,
                ))
                n += 1
            elif row.is_builtin:
                # keep builtin tag/description fresh with the code, and refresh
                # the capability announcement from the component (ADR-015)
                row.tag = spec["tag"]
                row.description = spec["description"]
                row.build_context = spec["build_context"]
                self._apply_profile(row, spec.get("profile") or {})
        self.db.commit()
        return n

    @staticmethod
    def _apply_profile(row: ContainerImage, profile: dict[str, Any], error: str = "") -> None:
        """Store an announcement and derive the platform facts from it.

        An empty/unusable announcement clears the derived facts instead of
        keeping stale ones: the image is then known to announce nothing, which
        is the honest state (the UI falls back to its own defaults).
        """
        normalised = trainer_profiles.normalise_profile(profile) if profile else {}
        row.profile = normalised
        row.profile_error = error or ""
        row.profile_at = datetime.now(timezone.utc) if normalised else None
        row.task_types = trainer_profiles.task_ids(normalised)
        hint = trainer_profiles.entrypoint_hint(normalised)
        if hint:
            row.entrypoint_hint = hint

    def refresh_profile(self, tenant_id: str, id: str) -> Optional[dict[str, Any]]:
        """Re-read a built-in component's announcement from the repository.

        Built-ins announce themselves through the file next to their build
        context; user images can POST their own document instead. Never raises
        for a missing file — it records why and keeps the row usable.
        """
        row = self.db.scalar(
            select(ContainerImage).where(
                ContainerImage.id == id, ContainerImage.tenant_id == tenant_id
            )
        )
        if not row:
            return None
        if not row.is_builtin:
            raise ValueError("only built-in images announce their profile from the repository")
        profile = trainer_profiles.load_profile(row.build_context)
        path = trainer_profiles.profile_path(row.build_context)
        if profile:
            self._apply_profile(row, profile)
        else:
            self._apply_profile(row, {}, error=f"no usable announcement at {path or row.build_context}")
        self.db.commit()
        self.db.refresh(row)
        return _img_dict(row)

    def set_profile(self, tenant_id: str, id: str, profile: Any) -> Optional[dict[str, Any]]:
        """Store an announcement supplied by the registrant (user images)."""
        row = self.db.scalar(
            select(ContainerImage).where(
                ContainerImage.id == id, ContainerImage.tenant_id == tenant_id
            )
        )
        if not row:
            return None
        if profile is None:
            self._apply_profile(row, {}, error="")
        else:
            error = ""
            if isinstance(profile, dict):
                schema = str(profile.get("schema") or "")
                if schema and schema != trainer_profiles.PROFILE_SCHEMA:
                    error = f"unknown profile schema {schema!r}"
            elif isinstance(profile, str):
                import json as _json

                try:
                    profile = _json.loads(profile)
                except _json.JSONDecodeError as exc:
                    profile, error = {}, f"profile is not valid JSON: {exc}"
            if not trainer_profiles.normalise_profile(profile).get("tasks"):
                error = error or "profile announces no usable task"
            self._apply_profile(row, profile if isinstance(profile, dict) else {}, error=error)
        self.db.commit()
        self.db.refresh(row)
        return _img_dict(row)

    def capabilities(self, tenant_id: str) -> dict[str, Any]:
        """Aggregate everything the registered images announce, with locks.

        This is the UI's single source for the train dialog: task list,
        architectures per task, default arch, devices and the locked/upgrade
        state of every extended node — all derived from announcements, never
        from platform hardcoding (ADR-015).
        """
        from app.services import entitlements as entitlements_svc

        granted = entitlements_svc.granted(self.db)
        rows = self.db.scalars(
            select(ContainerImage)
            .where(ContainerImage.tenant_id == tenant_id, ContainerImage.category == "train")
            .order_by(ContainerImage.name)
        ).all()
        images: list[dict[str, Any]] = []
        for row in rows:
            d = _img_dict(row)
            profile = d.get("profile") or {}
            granted_set = set(granted)

            def _lock(node: dict[str, Any]) -> dict[str, Any]:
                requires = trainer_profiles.requirements_for(
                    profile,
                    task=node.get("_task", ""),
                    arch=node.get("id", "") if node.get("_kind") == "arch" else "",
                    capability=node.get("id", "") if node.get("_kind") == "capability" else "",
                )
                missing = [k for k in requires if k not in granted_set]
                return {k: v for k, v in node.items() if not k.startswith("_")} | {
                    "locked": bool(missing), "requires": requires, "missing": missing,
                }

            tasks = []
            for task in profile.get("tasks") or []:
                archs = [
                    _lock({**a, "_task": task.get("id", ""), "_kind": "arch"})
                    for a in (task.get("architectures") or [])
                ]
                tasks.append(_lock({
                    "id": task.get("id", ""), "label": task.get("label", ""),
                    "default_arch": task.get("default_arch", ""),
                    "architectures": archs, "_task": task.get("id", ""),
                }))
            capabilities = [
                _lock({**c, "_kind": "capability"})
                for c in (profile.get("capabilities") or [])
            ]
            images.append({
                "id": d["id"], "name": d["name"], "repository": d["repository"],
                "tag": d["tag"], "full_ref": d["full_ref"], "is_builtin": d["is_builtin"],
                "devices": profile.get("devices") or [], "tasks": tasks,
                # Unannounced images still use the legacy catalog task list in
                # the training form; dropping it makes custom images vanish.
                "task_types": d["task_types"],
                "capabilities": capabilities,
                "announced": bool(profile),
                "profile_error": d.get("profile_error", ""),
            })
        return {"entitlements": granted, "images": images}

    def list(
        self, tenant_id: str, category: str = "", task_type: str = "",
    ) -> list[dict[str, Any]]:
        """List all images, optionally filtered by category and task type."""
        items, _ = self.list_paged(tenant_id, category, task_type, page=1, page_size=10_000)
        return items

    def list_paged(
        self,
        tenant_id: str,
        category: str = "",
        task_type: str = "",
        search: str = "",
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[dict[str, Any]], int]:
        """List images with server-side pagination.

        Pagination is applied in memory because the task_type match is a JSON
        array containment check kept in Python; catalogs are small (built-ins
        plus user registrations), so a full query is cheap.
        """
        q = select(ContainerImage).where(ContainerImage.tenant_id == tenant_id)
        if category:
            q = q.where(ContainerImage.category == category)
        rows = self.db.scalars(q.order_by(ContainerImage.category, ContainerImage.name)).all()
        needle = search.strip().lower()
        matched: list[dict[str, Any]] = []
        for r in rows:
            d = _img_dict(r)
            if task_type and task_type not in d["task_types"]:
                continue
            if needle and needle not in d["name"].lower():
                continue
            matched.append(d)
        total = len(matched)
        if page < 1:
            page = 1
        if page_size < 1:
            page_size = 20
        start = (page - 1) * page_size
        return matched[start:start + page_size], total

    def get(self, tenant_id: str, id: str) -> Optional[dict[str, Any]]:
        """Get one image by id."""
        row = self.db.scalar(
            select(ContainerImage).where(
                ContainerImage.id == id, ContainerImage.tenant_id == tenant_id
            )
        )
        return _img_dict(row) if row else None

    def create(
        self, tenant_id: str, spec: dict[str, Any], created_by: str,
    ) -> dict[str, Any]:
        """Register an image. Auto-prefixes bare repos with the hub project."""
        prefix = _registry_prefix(self.db)
        category = str(spec.get("category") or "").strip()
        if category not in CATEGORIES:
            raise ValueError(f"category must be one of {CATEGORIES}")
        name = str(spec.get("name") or "").strip()
        if not name:
            raise ValueError("name is required")
        repository = str(spec.get("repository") or "").strip()
        if not repository:
            raise ValueError("repository is required")
        # bare repo names (train/my-img) get the configured registry prefix
        if "/" not in repository.replace("://", ""):
            repository = f"{prefix}/{category}/{repository}"
        elif not repository.startswith(("hub.", "registry.", "harbor.")) and "://" not in repository:
            # relative multi-segment path (train/my-img) also prefixed
            repository = f"{prefix}/{repository}"
        tag = str(spec.get("tag") or "latest").strip()
        dup = self.db.scalar(
            select(ContainerImage).where(
                ContainerImage.tenant_id == tenant_id,
                ContainerImage.repository == repository,
                ContainerImage.tag == tag,
            )
        )
        if dup:
            raise ValueError(f"image {repository}:{tag} already registered")
        profile = spec.get("profile")
        # A build_context in this repository means the component announces
        # itself; an explicit document always wins (user-registered images).
        announced = trainer_profiles.builtin_profile(str(spec.get("build_context") or "")) if not profile else None
        profile = profile if profile is not None else announced
        row = ContainerImage(
            id=new_id("img"),
            tenant_id=tenant_id,
            name=name,
            category=category,
            repository=repository,
            tag=tag,
            description=str(spec.get("description") or ""),
            build_context=str(spec.get("build_context") or ""),
            entrypoint_hint=str(spec.get("entrypoint_hint") or ""),
            task_types=list(spec.get("task_types") or []),
            is_builtin=False,
            created_by=created_by or "anonymous",
        )
        if profile:
            # Component-announced (a build_context in this repo) or pasted by
            # the registrant: derive task_types/defaults from the announcement.
            self._apply_profile(row, profile)
        self.db.add(row)
        self.db.commit()
        self.db.refresh(row)
        return _img_dict(row)

    def update(self, tenant_id: str, id: str, patch: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Update editable fields of an image entry."""
        row = self.db.scalar(
            select(ContainerImage).where(
                ContainerImage.id == id, ContainerImage.tenant_id == tenant_id
            )
        )
        if not row:
            return None
        for k in ("name", "description", "tag", "entrypoint_hint", "task_types"):
            if k in patch and patch[k] is not None:
                setattr(row, k, patch[k])
        new_context = str(patch.get("build_context") or "").strip()
        if new_context:
            row.build_context = new_context
            # A repository context means the component announces itself;
            # re-read the announcement when the row doesn't carry a pasted one.
            if not row.profile:
                announced = trainer_profiles.builtin_profile(new_context)
                if announced:
                    self._apply_profile(row, announced)
        # category/repository changes allowed only for non-builtin entries
        if not row.is_builtin:
            if patch.get("category") in CATEGORIES:
                row.category = patch["category"]
            if patch.get("repository"):
                row.repository = str(patch["repository"]).strip()
        self.db.commit()
        self.db.refresh(row)
        return _img_dict(row)

    def delete(self, tenant_id: str, id: str) -> bool:
        """Delete an image entry (built-ins refuse)."""
        row = self.db.scalar(
            select(ContainerImage).where(
                ContainerImage.id == id, ContainerImage.tenant_id == tenant_id
            )
        )
        if not row:
            return False
        if row.is_builtin:
            raise ValueError("built-in images cannot be deleted (only edited)")
        self.db.delete(row)
        self.db.commit()
        return True



    # -- base model pool (LLM finetune) ---------------------------------------

    def list_base_models(self) -> list[dict[str, Any]]:
        """List base models under s3://nereidforge/base-models/*.

        A folder qualifies when it contains a config.json (HF-style model
        dir) — the LLM trainer downloads the whole folder as the base model.
        """
        from app.clients.s3 import SeaweedFSClient

        s3 = SeaweedFSClient()
        try:
            entries = s3.list_entries("/buckets/nereidforge/base-models")
        except Exception:
            return []
        out = []
        for e in entries:
            if not e.is_dir:
                continue
            name = e.name.rstrip("/")
            try:
                s3.read_object(f"/buckets/nereidforge/base-models/{name}/config.json")
                out.append({
                    "name": name,
                    "uri": "s3://nereidforge/base-models/{}/".format(name),
                    "config_ok": True,
                })
            except Exception:
                continue
        return sorted(out, key=lambda x: x["name"])
