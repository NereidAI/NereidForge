#!/usr/bin/env bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0
#
# License compliance gate for NereidForge deliverables (see LICENSE-POLICY.md).
#
# Scans what we ship for prohibited (copyleft) components:
#   1. runtime dependency manifests and container build files
#   2. first-party Python imports of a denylisted package
#   3. model/weight blobs and proprietary font assets tracked in git
#   4. copyleft license text in first-party license files
#   5. optional: an installed Python environment's dependency report (--env)
#   6. optional: vendored third-party license-file audit (--third-party-audit)
#
# The gate matches package names, imports and artifacts - never prose. A
# keyword scan of documentation would miss a renamed dependency and flag
# legitimate integration docs.
#
# Usage:
#   check-license-compliance.sh                    # static scan, tracked + untracked
#   check-license-compliance.sh --staged           # pre-commit, staged files only
#   check-license-compliance.sh --env .venv/bin/python
#   check-license-compliance.sh --third-party-audit
#
# Exit codes: 0 compliant, 1 violations found.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
DENYLIST="${REPO_ROOT}/infra/configs/license-denylist.txt"
FILE_MODE="repo"
ENV_PY=""
AUDIT_THIRD_PARTY=0
FAILURES=0
WARNINGS=0

usage() {
    sed -n '9,24p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --staged) FILE_MODE="staged"; shift ;;
        --env)
            [[ $# -ge 2 ]] || { echo "--env requires a Python interpreter path" >&2; exit 2; }
            ENV_PY="$2"; shift 2 ;;
        --third-party-audit) AUDIT_THIRD_PARTY=1; shift ;;
        -h|--help) usage; exit 0 ;;
        # pre-commit appends the changed file names; the staged list is used
        # instead, so unknown positional arguments are ignored on purpose.
        *) shift ;;
    esac
done

fail() { printf 'FAIL  %s\n' "$*"; FAILURES=$((FAILURES + 1)); }
warn() { printf 'WARN  %s\n' "$*"; WARNINGS=$((WARNINGS + 1)); }
info() { printf '      %s\n' "$*"; }

if [[ ! -f "$DENYLIST" ]]; then
    echo "denylist not found: $DENYLIST" >&2
    exit 2
fi

DENY_NAMES=()
DENY_LICENSES=()
while IFS='|' read -r name license reason; do
    name="${name%%#*}"
    name="$(echo "$name" | xargs)"
    [[ -z "$name" ]] && continue
    DENY_NAMES+=("$name")
    DENY_LICENSES+=("$(echo "$license" | xargs)")
done < "$DENYLIST"

# Vendored third-party trees carry their own upstream license and are governed
# by LICENSE-POLICY.md §6, not by the first-party dependency rules.
is_vendored() {
    case "$1" in
        infra/third_party/*|infra/charts/*/charts/*) return 0 ;;
        *) return 1 ;;
    esac
}

# A file whose dependency declarations bind what we ship.
is_dependency_file() {
    local base
    base="$(basename -- "$1")"
    case "$base" in
        requirements*.txt|constraints*.txt|pyproject.toml|setup.py|setup.cfg|Pipfile) return 0 ;;
        Dockerfile|Dockerfile.*|*.Dockerfile|Containerfile) return 0 ;;
        Chart.yaml|Chart.lock|docker-compose*.yaml|docker-compose*.yml) return 0 ;;
    esac
    [[ "$1" == *.sh ]]
}

# Strip whole-line and trailing comments so prose never triggers a match.
strip_comments() {
    sed -E -e 's/^[[:space:]]*#.*$//' -e 's/[[:space:]]+#.*$//' "$1" 2>/dev/null || true
}

collect_files() {
    if [[ "$FILE_MODE" == "staged" ]]; then
        git -C "$REPO_ROOT" diff --cached --name-only --diff-filter=ACMR
    else
        {
            git -C "$REPO_ROOT" ls-files
            git -C "$REPO_ROOT" ls-files --others --exclude-standard
        } | sort -u
    fi
}

mapfile -t FILES < <(collect_files)

echo "== NereidForge license compliance ($FILE_MODE scan, ${#FILES[@]} files) =="

# --- 1 + 2: denylisted packages in manifests, build files and imports ---------
for path in "${FILES[@]}"; do
    [[ -f "$REPO_ROOT/$path" ]] || continue
    is_vendored "$path" && continue

    if is_dependency_file "$path"; then
        for i in "${!DENY_NAMES[@]}"; do
            name="${DENY_NAMES[$i]}"
            escaped="${name//./\\.}"
            matches="$(strip_comments "$REPO_ROOT/$path" \
                | grep -inE "(^|[^[:alnum:]_.-])${escaped}([^[:alnum:]_.-]|$)" || true)"
            [[ -z "$matches" ]] && continue
            while IFS= read -r hit; do
                fail "$path:$hit -> '$name' (${DENY_LICENSES[$i]})"
            done <<< "$matches"
        done
    fi

    if [[ "$path" == *.py ]]; then
        for i in "${!DENY_NAMES[@]}"; do
            name="${DENY_NAMES[$i]}"
            escaped="${name//./\\.}"
            matches="$(grep -nE "^[[:space:]]*(from|import)[[:space:]]+${escaped}([[:space:].]|$)" \
                "$REPO_ROOT/$path" || true)"
            [[ -z "$matches" ]] && continue
            while IFS= read -r hit; do
                fail "$path:$hit -> imports '$name' (${DENY_LICENSES[$i]})"
            done <<< "$matches"
        done
    fi
done

# --- 3: model/weight blobs must never be tracked ------------------------------
# Test fixtures are the only carved-out location; shipped weights belong in the
# object store or model registry, not in git.
mapfile -t BLOBS < <(
    git -C "$REPO_ROOT" ls-files \
        | grep -Ei '\.(pt|pth|onnx|pdparams|pdopt|ckpt|safetensors|engine|plan|h5|tflite|mlmodel|gguf)$' \
        | grep -vE '/(testdata|fixtures)/' \
        || true
)
for blob in "${BLOBS[@]}"; do
    fail "$blob -> tracked model blob; move weights to S3/model registry"
done

# Fonts are assets with their own licenses; proprietary families (Arial,
# Helvetica, ...) must not be redistributed. An OFL/CC0 font needs an explicit
# license file next to it, which is what the exception below allows.
mapfile -t FONTS < <(
    git -C "$REPO_ROOT" ls-files \
        | grep -Ei '\.(ttf|otf|woff2?)$' \
        | grep -vE '/(testdata|fixtures)/' \
        || true
)
for font in "${FONTS[@]}"; do
    fail "$font -> tracked font asset; ship an OFL/CC0 font with its license, or drop it"
done

# --- 4: copyleft license text in first-party files ----------------------------
for marker in "GNU AFFERO GENERAL PUBLIC LICENSE" "GNU GENERAL PUBLIC LICENSE"; do
    mapfile -t LIC_FILES < <(
        git -C "$REPO_ROOT" grep -lF "$marker" -- '*LICENSE*' '*COPYING*' 2>/dev/null || true
    )
    for path in "${LIC_FILES[@]}"; do
        if is_vendored "$path"; then
            warn "$path -> vendored copyleft license text; must be listed in THIRD_PARTY_NOTICES.md"
        else
            fail "$path -> first-party file carries copyleft license text ($marker)"
        fi
    done
done

# --- 5: installed environment report -----------------------------------------
if [[ -n "$ENV_PY" ]]; then
    if [[ ! -x "$ENV_PY" ]]; then
        echo "env scan skipped: $ENV_PY is not executable" >&2
        FAILURES=$((FAILURES + 1))
    else
        echo "-- installed dependency report ($ENV_PY) --"
        set +e
        env_report="$("$ENV_PY" "$REPO_ROOT/infra/scripts/license_env_report.py" "$DENYLIST")"
        env_status=$?
        set -e
        printf '%s\n' "$env_report"
        [[ $env_status -ne 0 ]] && FAILURES=$((FAILURES + 1))
        env_warnings="$(printf '%s\n' "$env_report" \
            | sed -nE 's/.*blocked, ([0-9]+) need review.*/\1/p' | tail -n 1)"
        if [[ "$env_warnings" =~ ^[0-9]+$ ]]; then
            WARNINGS=$((WARNINGS + env_warnings))
        fi
    fi
fi

# --- 6: vendored third-party license-file audit -------------------------------
if [[ "$AUDIT_THIRD_PARTY" -eq 1 ]]; then
    echo "-- vendored third-party audit --"
    while IFS= read -r dir; do
        [[ -d "$REPO_ROOT/$dir" ]] || continue
        if ! find "$REPO_ROOT/$dir" -maxdepth 2 -type f \
            \( -iname 'LICENSE*' -o -iname 'COPYING*' \) -print -quit | grep -q .; then
            warn "$dir -> vendored tree has no license file"
        fi
    done < <(
        {
            find "$REPO_ROOT/infra/third_party" -mindepth 1 -maxdepth 1 -type d 2>/dev/null
            find "$REPO_ROOT/infra/charts" -path '*/charts/*' -mindepth 4 -maxdepth 4 -type d 2>/dev/null
        } | sed "s|^$REPO_ROOT/||" | sort -u
    )
fi

echo
if [[ "$FAILURES" -gt 0 ]]; then
    echo "RESULT: NOT COMPLIANT ($FAILURES failure(s), $WARNINGS warning(s))"
    echo "See LICENSE-POLICY.md for the rule and the L2 (user-supplied) path."
    exit 1
fi
if [[ "$WARNINGS" -gt 0 ]]; then
    echo "RESULT: REVIEW REQUIRED (0 failure(s), $WARNINGS warning(s))"
    echo "Resolve every review item against LICENSE-POLICY.md before release acceptance."
    exit 0
fi
echo "RESULT: compliant (0 warnings)"
