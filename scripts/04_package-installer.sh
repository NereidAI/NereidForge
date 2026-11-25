#!/bin/bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# Deprecated local-only compatibility path: build a repository-shaped online
# installer from tracked source files. NF release workflows never call this
# script and NF GitHub Releases do not attach its archive. New installations use
# the three NF OCI Helm charts (the consolidated nereidforge chart deploys API,
# worker and frontend in one release) against a separately published
# pre-NereidForge release. Generated release.env metadata still supports
# existing local users. No cluster or registry is contacted.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

VERSION="${RELEASE_TAG:-$(nf_version)}"
if ! [[ "$VERSION" =~ ^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(-[0-9A-Za-z-]+(\.[0-9A-Za-z-]+)*)?$ ]]; then
    echo "RELEASE_TAG must be bare SemVer without build metadata or a commit-qualified development version: $VERSION" >&2
    exit 2
fi
if [[ "$VERSION" == *-* ]]; then
    IFS='.' read -ra prerelease_identifiers <<< "${VERSION#*-}"
    for identifier in "${prerelease_identifiers[@]}"; do
        if [[ "$identifier" =~ ^[0-9]+$ && "${#identifier}" -gt 1 && "$identifier" == 0* ]]; then
            echo "Numeric SemVer pre-release identifiers must not contain leading zeroes" >&2
            exit 2
        fi
    done
fi

check_cmd git helm tar sha256sum python3

OUTPUT_DIR="${INSTALLER_OUTPUT_DIR:-$ROOT_DIR/dist}"
PREFIX="NereidForge-$VERSION"
ARCHIVE="${INSTALLER_ARCHIVE:-$OUTPUT_DIR/nereidforge-installer-$VERSION.tar.gz}"
STAGING="$(mktemp -d)"
trap 'rm -rf "$STAGING"' EXIT
TREE="$STAGING/$PREFIX"
mkdir -p "$TREE" "$OUTPUT_DIR"

# Copy tracked paths from the working tree so pre-commit verification exercises
# the actual edits. Ignored credentials, build output, and .git never enter the
# archive. A release runner is clean, so this is equivalent to git archive there.
(
    cd "$ROOT_DIR"
    git ls-files -z | tar --null -T - -cf -
) | tar -xf - -C "$TREE"

# The installer scripts render these directories directly rather than running
# helm package, so inject product versions into the staged chart sources too.
# Third-party wrappers retain their independent Chart.yaml metadata.
PRODUCT_CHARTS=(
    charts/ml-backend
    charts/nereidforge
    charts/tenant-check
)
for chart in "${PRODUCT_CHARTS[@]}"; do
    sed -i -E "0,/^version:[[:space:]]*\\S/s/^version:[[:space:]]*\\S.*$/version: $VERSION/" "$TREE/$chart/Chart.yaml"
    sed -i -E "0,/^appVersion:[[:space:]]*\\S/s/^appVersion:[[:space:]]*\\S.*$/appVersion: \"$VERSION\"/" "$TREE/$chart/Chart.yaml"
done

# verify_product_versions proves that direct source-directory installation uses
# the same product metadata as the OCI packages, both before and after archiving.
verify_product_versions() {
    local tree="$1" chart metadata actual_version actual_app_version
    for chart in "${PRODUCT_CHARTS[@]}"; do
        metadata="$(helm show chart "$tree/$chart")"
        actual_version="$(awk '/^version:/ {print $2; exit}' <<< "$metadata")"
        actual_app_version="$(awk '/^appVersion:/ {print $2; exit}' <<< "$metadata" | tr -d '"')"
        test "$actual_version" = "$VERSION"
        test "$actual_app_version" = "$VERSION"
    done
}
verify_product_versions "$TREE"

SOURCE_COMMIT="$(git -C "$ROOT_DIR" rev-parse HEAD 2>/dev/null || printf 'unknown')"
cat > "$TREE/release.env" <<EOF
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0
# Generated installer metadata; this file is never tracked in the source tree.
RELEASE_TAG=$VERSION
SOURCE_COMMIT=$SOURCE_COMMIT
EOF

# Manifest generation is file-only: config.sh computes public references but
# does not pull images or access a cluster for this entry point.
RELEASE_TAG="$VERSION" JWT_SECRET=manifest-generation-unused \
    bash "$TREE/scripts/05_export-images.sh" \
    --manifest "$TREE/images-manifest.txt"

# verify_release_manifest rejects an incomplete or incorrectly tagged suite
# image set before a source archive can be presented as a complete installer.
verify_release_manifest() {
    python3 - "$TREE/images-manifest.txt" "$VERSION" <<'PY'
import pathlib
import sys

manifest = pathlib.Path(sys.argv[1])
version = sys.argv[2]
expected = {
    "inference-model-server-openvino-xpu",
    "ml-backend",
    "ml-backend-custom",
    "nereidforge-backend",
    "nereidforge-frontend",
    "nf-runtime",
    "prediction-jupyter-base",
    "train-evaluator",
    "train-evaluator-xpu",
    "train-optimizer",
    "trainer-convert",
    "trainer-convert-pp",
    "trainer-convert-tf",
    "trainer-llm-cu",
    "trainer-llm-rocm",
    "trainer-llm-xpu",
    "trainer-paddle-cpu",
    "trainer-paddle-cu",
    "trainer-torch-cpu",
    "trainer-torch-cu",
    "trainer-torch-rocm",
    "trainer-torch-xpu",
    "trainer-xpu-base",
}
references = {}
for raw_line in manifest.read_text().splitlines():
    line = raw_line.strip()
    if line.startswith("ghcr.io/nereidai/") and ":" in line:
        repository, tag = line.rsplit(":", 1)
        references[repository.removeprefix("ghcr.io/nereidai/")] = tag

missing = sorted(expected - references.keys())
unexpected = sorted(references.keys() - expected)
if missing or unexpected:
    raise SystemExit(f"suite image set mismatch; missing={missing}, unexpected={unexpected}")
for repository in sorted(expected):
    expected_tag = "py3.12" if repository == "prediction-jupyter-base" else version
    actual_tag = references[repository]
    if actual_tag != expected_tag:
        raise SystemExit(
            f"{repository} has tag {actual_tag!r}; expected {expected_tag!r}"
        )
print(f"Verified {len(expected)} suite-owned image references for {version}")
PY
}

verify_release_manifest
for required in \
    scripts/00_deploy-all.sh \
    scripts/03_package-charts.sh \
    deploy/pre-nereidforge/platform/kubeflow-manifests \
    deploy/pre-nereidforge/charts/cvat/Chart.lock \
    charts/nereidforge/Chart.yaml \
    secrets.yaml.template \
    LICENSE \
    NOTICE; do
    test -e "$TREE/$required" || {
        echo "Installer is missing required path: $required" >&2
        exit 1
    }
done
test ! -e "$TREE/.git"

rm -f "$ARCHIVE" "$ARCHIVE.sha256"
tar --sort=name --mtime="@${SOURCE_DATE_EPOCH:-0}" --owner=0 --group=0 \
    --numeric-owner -czf "$ARCHIVE" -C "$STAGING" "$PREFIX"
(
    cd "$(dirname "$ARCHIVE")"
    sha256sum "$(basename "$ARCHIVE")" > "$(basename "$ARCHIVE").sha256"
)

# Reparse the actual artifact without an inherited RELEASE_TAG. This catches an
# archive that silently falls back to 0.0.0-dev after its .git context is lost.
mkdir -p "$STAGING/verify"
tar -xzf "$ARCHIVE" -C "$STAGING/verify"
resolved="$(env -u RELEASE_TAG bash -c \
    'source "$1/scripts/lib/common.sh"; nf_version' \
    _ "$STAGING/verify/$PREFIX")"
test "$resolved" = "$VERSION" || {
    echo "Archived installer resolves version '$resolved', expected '$VERSION'" >&2
    exit 1
}
verify_product_versions "$STAGING/verify/$PREFIX"
# The frontend ingress host is the only chart value without a tracked default;
# the consolidated chart renders fully offline once that host is supplied.
helm template nereidforge \
    "$STAGING/verify/$PREFIX/charts/nereidforge" \
    --set frontend.ingress.host=forge.example.test \
    > "$STAGING/verified-application.yaml"
python3 - "$STAGING/verified-application.yaml" "$VERSION" <<'PY'
import pathlib
import re
import sys

text = pathlib.Path(sys.argv[1]).read_text()
version = sys.argv[2]
inline = re.compile(
    r"ghcr\.io/nereidai/[A-Za-z0-9][A-Za-z0-9._/-]*:[A-Za-z0-9_][A-Za-z0-9._-]*"
)
split = re.compile(
    r"repository:[ \t]*[\"']?"
    r"(ghcr\.io/nereidai/[A-Za-z0-9][A-Za-z0-9._/-]*)"
    r"[\"']?[ \t]*\r?\n[ \t]*tag:[ \t]*[\"']?"
    r"([A-Za-z0-9_][A-Za-z0-9._-]*)"
)
references = set(inline.findall(text))
references.update(repository + ":" + tag for repository, tag in split.findall(text))
expected = {
    f"ghcr.io/nereidai/nereidforge-backend:{version}",
    f"ghcr.io/nereidai/nereidforge-frontend:{version}",
    f"ghcr.io/nereidai/ml-backend:{version}",
    f"ghcr.io/nereidai/inference-model-server-openvino-xpu:{version}",
}
missing = sorted(expected - references)
mismatched = sorted(ref for ref in references if not ref.endswith(":" + version))
if missing or mismatched:
    raise SystemExit(f"archived application image mismatch; missing={missing}, mismatched={mismatched}")
print("Archived application inline and split suite-image references verified")
PY

printf 'Installer archive: %s\n' "$ARCHIVE"
printf 'SHA-256: %s.sha256\n' "$ARCHIVE"
printf 'Archived RELEASE_TAG: %s\n' "$resolved"
