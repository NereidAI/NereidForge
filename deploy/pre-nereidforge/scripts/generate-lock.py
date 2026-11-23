#!/usr/bin/env python3
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Generate the pre-NereidForge source/package lock using only the stdlib."""

import hashlib
import json
from pathlib import Path
import re
import sys


CHART_NAMES = ("cvat", "labelstudio", "postgresql", "seaweedfs", "valkey")


def chart_metadata(path):
    values = {}
    pattern = re.compile(r"^(name|version|appVersion):[ \t]*([^#]+?)[ \t]*$")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match and match.group(1) not in values:
            values[match.group(1)] = match.group(2).strip().strip("\"'")
    missing = {"name", "version", "appVersion"} - values.keys()
    if missing:
        raise SystemExit(f"{path}: missing chart metadata {sorted(missing)}")
    return values


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def shell_assignment(text, name):
    match = re.search(rf'^{name}="([^"]+)"$', text, re.MULTILINE)
    if not match:
        raise SystemExit(f"Unable to resolve Kubeflow baseline variable {name}")
    return match.group(1)


def main():
    if len(sys.argv) != 5:
        raise SystemExit("Usage: generate-lock.py <module-tree> <version> <source-commit> <output>")
    root = Path(sys.argv[1]).resolve()
    version = sys.argv[2]
    source_commit = sys.argv[3]
    output = Path(sys.argv[4])
    if not source_commit or any(character.isspace() for character in source_commit):
        raise SystemExit("A non-empty, whitespace-free source commit is required")

    discovered = sorted(path.parent.name for path in (root / "charts").glob("*/Chart.yaml"))
    if tuple(discovered) != CHART_NAMES:
        raise SystemExit(f"Expected exactly five pre charts {CHART_NAMES}, found {discovered}")
    charts = []
    for name in CHART_NAMES:
        relative = f"charts/{name}"
        metadata = chart_metadata(root / relative / "Chart.yaml")
        if metadata["name"] != name:
            raise SystemExit(f"Chart identity mismatch in {relative}: {metadata['name']}")
        charts.append({"path": relative, **metadata})

    fetch = (root / "scripts" / "04_fetch-kubeflow-manifests.sh").read_text(encoding="utf-8")
    repository = shell_assignment(fetch, "REPO_URL")
    tag = shell_assignment(fetch, "KUBEFLOW_VERSION")
    patch_readme = (root / "platform" / "kubeflow-patches" / "README.md").read_text(encoding="utf-8")
    commit_match = re.search(r"commit `([0-9a-f]{7,40})`", patch_readme)
    if not commit_match or tag not in patch_readme:
        raise SystemExit("Kubeflow patch README does not confirm the fetch baseline")
    baseline = {"repository": repository, "tag": tag, "commit": commit_match.group(1)}

    patch_root = root / "platform" / "kubeflow-patches"
    files = []
    for path in sorted(item for item in patch_root.rglob("*") if item.is_file()):
        files.append({"path": path.relative_to(root).as_posix(), "sha256": sha256(path)})
    if not any(item["path"].endswith("/patch.diff") for item in files):
        raise SystemExit("Kubeflow patch provenance is empty")

    document = {
        "schemaVersion": 1,
        "version": version,
        "sourceCommit": source_commit,
        "charts": charts,
        "kubeflow": {
            "baseline": baseline,
            "patches": {"path": "platform/kubeflow-patches", "files": files},
        },
    }
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
