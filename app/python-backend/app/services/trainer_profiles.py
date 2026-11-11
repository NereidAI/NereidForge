# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Trainer profile registry: read what a component announces (ADR-015).

The platform stores capabilities, it does not invent them. A component ships
`trainer-profile.json` (`nf.trainer-profile/1`) in its build context; this module
finds that file for built-in images, normalises it, and derives the platform
facts that used to be hardcoded lists (`task_types`, default architecture).

Design rules (keep them, they are the point of the mechanism):

* never raise for a missing/odd profile — an image without an announcement is
  still usable, it just contributes no capability facts;
* never trust the file's shape: `task_ids()` returns only ids the platform
  knows, `default_arch()` only strings;
* the file is read from the build context in *this* repository, which is how
  built-in components register themselves. User-registered images may paste or
  upload their own document through the API.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

PROFILE_SCHEMA = "nf.trainer-profile/1"
PROFILE_FILENAME = "trainer-profile.json"

#: Build-context prefixes whose profile lives under app/<prefix>/. Built-in
#: seeds use `builders/...` or `kfp-pipelines/docker/...`.
_CONTEXT_ROOTS = ("builders", "kfp-pipelines/docker")

#: Task vocabulary the platform can schedule/display. A component may announce
#: anything; only known ids become platform facts.
KNOWN_TASKS = (
    "classification",
    "multilabel",
    "detection",
    "instance_seg",
    "semantic_seg",
    "pose",
    "face_detection",
    "face_recognition",
    "tracking",
    "llm",
    "vlm",
    "notebook",
)

TIERS = ("core", "extended")


def repo_root() -> str:
    """Repository root, resolved from this file (app/python-backend/app/services)."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", "..", ".."))


def profile_path(build_context: str) -> str:
    """Absolute path of the profile a build context announces (may not exist).

    `build_context` is stored without the `app/` prefix (e.g.
    `builders/trainer-torch`, `builders/trainer-torch/Dockerfile.cu`); the
    profile always sits next to the component, i.e. in the first directory
    under the context root.
    """
    ctx = (build_context or "").strip().lstrip("/")
    if not ctx:
        return ""
    parts = ctx.split("/")
    if parts and parts[0] == "app":
        parts = parts[1:]
    if not parts:
        return ""
    root, rest = parts[0], parts[1:]
    if root not in _CONTEXT_ROOTS:
        # Unknown layout: no announcement to read (never guess).
        return ""
    # Drop a Dockerfile/… file name so only directories remain.
    directories = [p for p in rest if not p.startswith("Dockerfile")]
    return os.path.join(repo_root(), "app", root, *directories, PROFILE_FILENAME)


def load_profile(build_context: str) -> dict[str, Any]:
    """Read the announced profile for a build context; {} when unavailable."""
    path = profile_path(build_context)
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            document = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("unreadable trainer profile %s: %s", path, exc)
        return {}
    if not isinstance(document, dict):
        return {}
    schema = str(document.get("schema") or "")
    if schema != PROFILE_SCHEMA:
        # Forward compatible: keep the document, but flag the unknown schema so
        # the API/UI can show "announced by a newer component".
        logger.info("trainer profile %s uses schema %r", path, schema)
    return document


def normalise_profile(document: Any) -> dict[str, Any]:
    """Coerce an arbitrary profile document into the stored shape (never raises)."""
    if not isinstance(document, dict):
        return {}
    tasks: list[dict[str, Any]] = []
    for entry in document.get("tasks") or []:
        if not isinstance(entry, dict):
            continue
        task_id = str(entry.get("id") or "").strip().lower()
        if not task_id:
            continue
        architectures = []
        for arch in entry.get("architectures") or []:
            if isinstance(arch, dict) and str(arch.get("id") or "").strip():
                architectures.append(arch)
            elif isinstance(arch, str) and arch.strip():
                architectures.append({"id": arch.strip()})
        tasks.append({
            "id": task_id,
            "label": str(entry.get("label") or task_id),
            "tier": _tier(entry),
            "requires": _requires(entry),
            "default_arch": str(entry.get("default_arch") or ""),
            "architectures": architectures,
        })

    capabilities = []
    for entry in document.get("capabilities") or []:
        if isinstance(entry, dict) and str(entry.get("id") or "").strip():
            capabilities.append({
                "id": str(entry["id"]).strip(),
                "label": str(entry.get("label") or entry["id"]),
                "tier": _tier(entry),
                "requires": _requires(entry),
            })

    return {
        "schema": str(document.get("schema") or ""),
        "name": str(document.get("name") or ""),
        "framework": document.get("framework") if isinstance(document.get("framework"), dict) else {},
        "devices": [str(d) for d in (document.get("devices") or []) if str(d).strip()],
        "tier": _tier(document),
        "requires": _requires(document),
        "tasks": tasks,
        "capabilities": capabilities,
        "entrypoints": document.get("entrypoints") if isinstance(document.get("entrypoints"), dict) else {},
        "env": [str(e) for e in (document.get("env") or [])],
        "artifacts": [str(a) for a in (document.get("artifacts") or [])],
        "config_schema": document.get("config_schema") if isinstance(document.get("config_schema"), dict) else {},
        "license": document.get("license") if isinstance(document.get("license"), dict) else {},
    }


def task_ids(profile: dict[str, Any]) -> list[str]:
    """Task ids the platform knows, in announcement order (deduplicated)."""
    seen: list[str] = []
    for entry in profile.get("tasks") or []:
        task_id = str(entry.get("id") or "")
        if task_id in KNOWN_TASKS and task_id not in seen:
            seen.append(task_id)
    return seen


def default_arch(profile: dict[str, Any], task: str = "") -> str:
    """Announced default architecture for *task* (or the first task)."""
    wanted = (task or "").strip().lower()
    for entry in profile.get("tasks") or []:
        if wanted and str(entry.get("id") or "") != wanted:
            continue
        default = str(entry.get("default_arch") or "")
        if default:
            return default
        architectures = entry.get("architectures") or []
        if architectures:
            return str(architectures[0].get("id") or "")
    return ""


def architectures(profile: dict[str, Any], task: str = "") -> list[dict[str, Any]]:
    """Announced architectures for *task* (all tasks when task is empty)."""
    wanted = (task or "").strip().lower()
    result: list[dict[str, Any]] = []
    for entry in profile.get("tasks") or []:
        if wanted and str(entry.get("id") or "") != wanted:
            continue
        result.extend(a for a in (entry.get("architectures") or []) if isinstance(a, dict))
    return result


def tier_for(profile: dict[str, Any], *, task: str = "", arch: str = "",
             capability: str = "") -> str:
    """Most specific announced tier (architecture > task/capability > profile)."""
    if capability:
        for entry in profile.get("capabilities") or []:
            if str(entry.get("id") or "") == capability:
                return _tier(entry)
        return _tier(profile)
    if task:
        for entry in profile.get("tasks") or []:
            if str(entry.get("id") or "") != task:
                continue
            if arch:
                for candidate in entry.get("architectures") or []:
                    if str(candidate.get("id") or "") == arch:
                        return _tier(candidate)
            return _tier(entry)
        return _tier(profile)
    return _tier(profile)


def requirements_for(profile: dict[str, Any], *, task: str = "", arch: str = "",
                     capability: str = "") -> list[str]:
    """Entitlement keys for a capability (profile-level + most specific node)."""
    required = list(_requires(profile))
    node: dict[str, Any] = {}
    if capability:
        for entry in profile.get("capabilities") or []:
            if str(entry.get("id") or "") == capability:
                node = entry
                break
    elif task:
        for entry in profile.get("tasks") or []:
            if str(entry.get("id") or "") != task:
                continue
            if arch:
                for candidate in entry.get("architectures") or []:
                    if str(candidate.get("id") or "") == arch:
                        node = candidate
                        break
            if not node:
                node = entry
            break
    required.extend(_requires(node))
    ordered: list[str] = []
    for key in required:
        if key not in ordered:
            ordered.append(key)
    return ordered


def missing_entitlements(profile: dict[str, Any], granted: Any, *, task: str = "",
                         arch: str = "", capability: str = "") -> list[str]:
    """Entitlement keys required by the announcement but not granted."""
    have = {str(item) for item in (granted or [])}
    return [
        key for key in requirements_for(profile, task=task, arch=arch, capability=capability)
        if key not in have
    ]


def entrypoints(profile: dict[str, Any]) -> list[str]:
    """Namespace part of each announced entrypoint, e.g. ['python','evaluate',...]."""
    values = profile.get("entrypoints") or {}
    names = list(values) if isinstance(values, dict) else []
    return [str(n) for n in names]


def entrypoint_hint(profile: dict[str, Any], framework: str = "") -> str:
    """Human-readable entrypoint hint for the catalog, derived from the profile."""
    values = profile.get("entrypoints") or {}
    if not isinstance(values, dict) or not values:
        return ""
    train = str(values.get("train") or "")
    extras = [key for key in values if key != "train" and key != "profile"]
    hint = train or ""
    if extras:
        hint = f"{hint} (SCRIPT={','.join(['train', *extras])})" if hint else ""
    return hint.strip()


def _tier(node: Any) -> str:
    if isinstance(node, dict):
        value = str(node.get("tier") or "").strip().lower()
        if value in TIERS:
            return value
    return "core"


def _requires(node: Any) -> list[str]:
    if not isinstance(node, dict):
        return []
    raw = node.get("requires")
    if isinstance(raw, str):
        return [raw.strip()] if raw.strip() else []
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    return []


def profile_summary(profile: dict[str, Any]) -> dict[str, Any]:
    """Compact, API-friendly view of an announcement."""
    if not profile:
        return {}
    return {
        "schema": profile.get("schema", ""),
        "name": profile.get("name", ""),
        "framework": profile.get("framework") or {},
        "devices": profile.get("devices") or [],
        "tier": profile.get("tier") or "core",
        "requires": profile.get("requires") or [],
        "tasks": [
            {
                "id": entry.get("id", ""),
                "label": entry.get("label", ""),
                "tier": entry.get("tier", "core"),
                "requires": entry.get("requires") or [],
                "default_arch": entry.get("default_arch", ""),
                "architectures": entry.get("architectures") or [],
            }
            for entry in profile.get("tasks") or []
        ],
        "capabilities": profile.get("capabilities") or [],
        "artifacts": profile.get("artifacts") or [],
        "license": profile.get("license") or {},
    }


def extended_capabilities(profile: dict[str, Any]) -> list[dict[str, Any]]:
    """Everything the component announces as `extended` (the sellable surface)."""
    result: list[dict[str, Any]] = []
    if _tier(profile) == "extended":
        result.append({"id": profile.get("name") or "profile", "kind": "profile",
                       "requires": _requires(profile)})
    for entry in profile.get("tasks") or []:
        if _tier(entry) == "extended":
            result.append({"id": entry.get("id", ""), "kind": "task",
                           "label": entry.get("label", ""), "requires": _requires(entry)})
        for arch in entry.get("architectures") or []:
            if _tier(arch) == "extended":
                result.append({"id": arch.get("id", ""), "kind": "architecture",
                               "label": arch.get("label", ""), "task": entry.get("id", ""),
                               "requires": _requires(arch)})
    for entry in profile.get("capabilities") or []:
        if _tier(entry) == "extended":
            result.append({"id": entry.get("id", ""), "kind": "capability",
                           "label": entry.get("label", ""), "requires": _requires(entry)})
    return result


def builtin_profile(build_context: str) -> Optional[dict[str, Any]]:
    """Normalised announcement of a built-in component (None when it announces nothing)."""
    document = load_profile(build_context)
    if not document:
        return None
    return normalise_profile(document)
