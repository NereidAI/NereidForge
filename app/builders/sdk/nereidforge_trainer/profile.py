# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Capability profile validation and lookups (`nf.trainer-profile/1`).

The image announces; the platform registers (ADR-015). This module only helps
component authors and CI: it validates the document and answers the two
questions the platform asks — *which tier is this capability* and *which
entitlements does it need*. The platform itself reads the JSON as plain data and
never imports this package.
"""

from __future__ import annotations

import json
from typing import Any, Iterable

PROFILE_SCHEMA = "nf.trainer-profile/1"
TIERS = ("core", "extended")
DEVICE_VALUES = ("cpu", "cuda", "rocm", "xpu")
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


def validate_profile(doc: Any) -> list[str]:
    """Return a list of human-readable problems; empty means valid.

    Deliberately permissive about *unknown* extra fields: a newer component may
    announce more than this SDK version understands. It is strict about the
    fields the platform consumes.
    """
    errors: list[str] = []
    if not isinstance(doc, dict):
        return ["profile must be a JSON object"]

    if doc.get("schema") != PROFILE_SCHEMA:
        errors.append(f"schema must be {PROFILE_SCHEMA!r}, got {doc.get('schema')!r}")
    if not str(doc.get("name") or "").strip():
        errors.append("name is required (e.g. 'vision/torch')")

    framework = doc.get("framework")
    if not isinstance(framework, dict) or not str(framework.get("name") or "").strip():
        errors.append("framework.name is required")

    devices = doc.get("devices")
    if not isinstance(devices, list) or not devices:
        errors.append("devices must be a non-empty list")
    else:
        for device in devices:
            if device not in DEVICE_VALUES:
                errors.append(f"unknown device {device!r} (expected one of {DEVICE_VALUES})")

    if _tier(doc) == "extended" and not doc.get("requires"):
        errors.append("an extended profile must declare 'requires'")

    tasks = doc.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        errors.append("tasks must be a non-empty list")
        tasks = []
    seen_tasks: set[str] = set()
    for task in tasks:
        if not isinstance(task, dict):
            errors.append("every task must be an object")
            continue
        task_id = str(task.get("id") or "")
        if not task_id:
            errors.append("task.id is required")
            continue
        if task_id not in KNOWN_TASKS:
            errors.append(f"unknown task id {task_id!r}")
        if task_id in seen_tasks:
            errors.append(f"duplicate task id {task_id!r}")
        seen_tasks.add(task_id)
        if not str(task.get("label") or "").strip():
            errors.append(f"task {task_id}: label is required")

        archs = task.get("architectures")
        if not isinstance(archs, list) or not archs:
            errors.append(f"task {task_id}: architectures must be a non-empty list")
            continue
        arch_ids = [str(a.get("id") or "") for a in archs if isinstance(a, dict)]
        if len(arch_ids) != len(archs):
            errors.append(f"task {task_id}: every architecture must be an object")
        if any(not a for a in arch_ids):
            errors.append(f"task {task_id}: architecture.id is required")
        if len(arch_ids) != len(set(arch_ids)):
            errors.append(f"task {task_id}: duplicate architecture id")
        default_arch = str(task.get("default_arch") or "")
        if not default_arch:
            errors.append(f"task {task_id}: default_arch is required")
        elif default_arch not in arch_ids:
            errors.append(f"task {task_id}: default_arch {default_arch!r} not in architectures")
        for arch in archs:
            if not isinstance(arch, dict):
                continue
            if _tier(arch) == "extended" and not arch.get("requires"):
                errors.append(f"task {task_id}: extended architecture {arch.get('id')!r} needs 'requires'")
            if not str(arch.get("license") or "").strip():
                errors.append(f"task {task_id}: architecture {arch.get('id')!r} must declare 'license'")

    for cap in doc.get("capabilities") or []:
        if not isinstance(cap, dict) or not str(cap.get("id") or ""):
            errors.append("every capability needs an id")
            continue
        if _tier(cap) == "extended" and not cap.get("requires"):
            errors.append(f"capability {cap['id']!r} is extended and needs 'requires'")

    entrypoints = doc.get("entrypoints")
    if not isinstance(entrypoints, dict) or not str(entrypoints.get("train") or "").strip():
        errors.append("entrypoints.train is required")

    license_doc = doc.get("license")
    if not isinstance(license_doc, dict) or not str(license_doc.get("code") or "").strip():
        errors.append("license.code is required (the component's own licence)")

    return errors


def load_profile(path: str) -> dict[str, Any]:
    """Read a profile document from disk."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def task_entry(profile: dict[str, Any], task: str) -> dict[str, Any] | None:
    """Return the announced entry for *task*, or None."""
    for entry in profile.get("tasks") or []:
        if isinstance(entry, dict) and entry.get("id") == task:
            return entry
    return None


def capability_entry(profile: dict[str, Any], capability: str) -> dict[str, Any] | None:
    for entry in profile.get("capabilities") or []:
        if isinstance(entry, dict) and entry.get("id") == capability:
            return entry
    return None


def architecture_entry(profile: dict[str, Any], task: str, arch: str) -> dict[str, Any] | None:
    entry = task_entry(profile, task)
    if not entry:
        return None
    for candidate in entry.get("architectures") or []:
        if isinstance(candidate, dict) and candidate.get("id") == arch:
            return candidate
    return None


def _tier(node: Any) -> str:
    if isinstance(node, dict) and str(node.get("tier") or "").strip():
        return str(node["tier"]).strip().lower()
    return "core"


def effective_tier(
    profile: dict[str, Any],
    *,
    task: str = "",
    arch: str = "",
    capability: str = "",
) -> str:
    """Most specific announced tier: architecture > task/capability > profile."""
    if capability:
        entry = capability_entry(profile, capability)
        return _tier(entry) if entry else _tier(profile)
    if task:
        entry = task_entry(profile, task)
        if not entry:
            return _tier(profile)
        if arch:
            arch_entry = architecture_entry(profile, task, arch)
            if arch_entry:
                return _tier(arch_entry)
        return _tier(entry)
    return _tier(profile)


def profile_requirements(
    profile: dict[str, Any],
    *,
    task: str = "",
    arch: str = "",
    capability: str = "",
) -> list[str]:
    """Entitlement keys for a capability.

    Profile-level requirements apply to everything the component announces; the
    most specific node (architecture > task/capability) adds its own. A core
    architecture inside an extended task therefore stays usable, which is the
    point of declaring tiers at both levels.
    """
    required = [str(item) for item in (profile.get("requires") or [])]
    node: Any = None
    if capability:
        node = capability_entry(profile, capability)
    elif task:
        node = architecture_entry(profile, task, arch) if arch else task_entry(profile, task)
    if isinstance(node, dict):
        required.extend(str(item) for item in (node.get("requires") or []))
    seen: list[str] = []
    for key in required:
        if key not in seen:
            seen.append(key)
    return seen


def missing_entitlements(
    profile: dict[str, Any],
    entitlements: Iterable[str],
    *,
    task: str = "",
    arch: str = "",
    capability: str = "",
) -> list[str]:
    """Keys required by the capability but absent from the granted set."""
    granted = {str(item) for item in entitlements or []}
    return [
        key
        for key in profile_requirements(profile, task=task, arch=arch, capability=capability)
        if key not in granted
    ]


def emit(profile: dict[str, Any]) -> str:
    """Serialise a profile the way components commit it (stable, sorted keys)."""
    errors = validate_profile(profile)
    if errors:
        raise ValueError("invalid trainer profile: " + "; ".join(errors))
    return json.dumps(profile, indent=2, sort_keys=False) + "\n"
