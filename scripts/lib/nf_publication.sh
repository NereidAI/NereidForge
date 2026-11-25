#!/bin/bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# Application-owned, fail-closed OCI chart publication helpers. An immutable
# tag is reused only after pulling it with the publisher configuration and
# comparing the local bytes. Only an explicit registry "not found" response
# means that a tag is absent; authentication, network, and other registry
# failures stop publication.
#
# The function and digest-variable names retain the existing publish-dev call
# sites during this first extraction. This file is self-contained: it imports
# no pre-NereidForge configuration, installer, or credential behavior.
set -euo pipefail

_pre_oras() {
    local command="$1"
    shift
    local -a registry_config=()
    if [[ -n "${ORAS_REGISTRY_CONFIG:-}" ]]; then
        registry_config=(--registry-config "$ORAS_REGISTRY_CONFIG")
    fi
    "${ORAS:-oras}" "$command" "${registry_config[@]}" "$@"
}

_pre_helm() {
    "${HELM:-helm}" "$@"
}

# Return 10 only for an explicit missing-manifest/name response. In particular,
# a missing local executable ("command not found") is not an absent OCI tag.
pre_oras_is_not_found_error() {
    local message="${1,,}"
    [[ -n "$message" ]] || return 1
    if [[ "$message" == *"command not found"* ||
          "$message" == *"no such file or directory"* ||
          "$message" == *"unauthorized"* ||
          "$message" == *"forbidden"* ||
          "$message" == *"authentication"* ||
          "$message" == *"authoriz"* ||
          "$message" == *"denied"* ||
          "$message" == *"permission"* ||
          "$message" == *"credential"* ||
          "$message" == *"connection"* ||
          "$message" == *"network"* ||
          "$message" == *"timeout"* ||
          "$message" == *"no such host"* ||
          "$message" == *"dial "* ||
          "$message" == *"refused"* ||
          "$message" == *"x509"* ||
          "$message" == *"certificate"* ||
          "$message" == *"tls"* ]]; then
        return 1
    fi
    [[ "$message" == *"not found"* ||
       "$message" == *"manifest unknown"* ||
       "$message" == *"name unknown"* ]]
}

# Set PRE_ORAS_DIGEST on success. Return 10 for absence and 1 for every other
# resolution failure, so callers cannot accidentally turn an error into a push.
pre_oras_resolve() {
    local reference="$1"
    local error_file output
    error_file="$(mktemp)"
    if output="$(_pre_oras resolve "$reference" 2>"$error_file")"; then
        rm -f "$error_file"
        PRE_ORAS_DIGEST="$output"
        if [[ -z "$PRE_ORAS_DIGEST" || "$PRE_ORAS_DIGEST" == *$'\n'* ]]; then
            printf 'Invalid empty or multiline digest while resolving %s\n' "$reference" >&2
            return 1
        fi
        return 0
    fi
    if pre_oras_is_not_found_error "$(cat "$error_file")"; then
        rm -f "$error_file"
        return 10
    fi
    cat "$error_file" >&2
    rm -f "$error_file"
    return 1
}

pre_oras_resolve_required() {
    local reference="$1" status
    if pre_oras_resolve "$reference"; then
        return 0
    else
        status=$?
        if [[ "$status" -eq 10 ]]; then
            printf 'Expected published OCI artifact is absent: %s\n' "$reference" >&2
        fi
        return 1
    fi
}

# Read-after-write on a brand-new GHCR package can briefly answer not-found
# while the registry propagates its first artifact. Retry only the absence
# result (status 10) with bounded backoff; every other resolution error fails
# closed immediately. The bounds are environment-overridable so offline tests
# run without sleeping.
pre_oras_wait_for_artifact() {
    local reference="$1"
    local attempts="${PRE_ORAS_READBACK_ATTEMPTS:-6}"
    local delay="${PRE_ORAS_READBACK_DELAY_SECONDS:-5}"
    local attempt=1 status
    [[ "$attempts" =~ ^[1-9][0-9]*$ ]] || {
        printf 'Invalid PRE_ORAS_READBACK_ATTEMPTS: %s\n' "$attempts" >&2
        return 2
    }
    [[ "$delay" =~ ^[0-9]+$ ]] || {
        printf 'Invalid PRE_ORAS_READBACK_DELAY_SECONDS: %s\n' "$delay" >&2
        return 2
    }
    while (( attempt <= attempts )); do
        if pre_oras_resolve "$reference"; then
            return 0
        else
            status=$?
        fi
        if [[ "$status" -ne 10 ]]; then
            return 1
        fi
        if (( attempt == attempts )); then
            printf 'Expected published OCI artifact is absent after %s read-back attempts: %s\n' \
                "$attempts" "$reference" >&2
            return 1
        fi
        printf 'Artifact %s not yet propagated; read-back attempt %s/%s in %ss\n' \
            "$reference" "$attempt" "$attempts" "$delay" >&2
        sleep "$delay"
        attempt=$((attempt + 1))
    done
}

# Helm names an OCI download after the requested tag, even when the archive's
# embedded Chart.yaml has a different (immutable) version.
pre_helm_pull_chart() {
    local repository="$1" requested_version="$2" destination="$3"
    local name="${repository##*/}"
    mkdir -p "$destination"
    if ! _pre_helm pull "oci://${repository}" --version "$requested_version" \
        --destination "$destination"; then
        printf 'Helm pull failed for %s:%s\n' "$repository" "$requested_version" >&2
        return 1
    fi
    PRE_HELM_PULLED_CHART="${destination}/${name}-${requested_version}.tgz"
    if [[ ! -s "$PRE_HELM_PULLED_CHART" ]]; then
        printf 'Helm pull did not produce requested-tag archive: %s\n' \
            "$PRE_HELM_PULLED_CHART" >&2
        return 1
    fi
}

# Restore an alias after a registry operation or verification failure. If the
# alias existed, restore its previous digest; otherwise leave it at the intended
# immutable digest. The caller still fails even when restoration succeeds.
pre_oras_restore_alias() {
    local repository="$1" immutable_digest="$2" floating="$3" previous_digest="$4"
    local desired_digest="${previous_digest:-$immutable_digest}"
    if ! _pre_oras tag "${repository}@${desired_digest}" "$floating"; then
        printf 'Failed to restore floating alias %s:%s to %s\n' \
            "$repository" "$floating" "$desired_digest" >&2
        return 1
    fi
    if ! pre_oras_resolve_required "${repository}:${floating}"; then
        printf 'Failed to verify restored floating alias %s:%s\n' \
            "$repository" "$floating" >&2
        return 1
    fi
    if [[ "$PRE_ORAS_DIGEST" != "$desired_digest" ]]; then
        printf 'Restored floating alias %s:%s resolved to %s, expected %s\n' \
            "$repository" "$floating" "$PRE_ORAS_DIGEST" "$desired_digest" >&2
        return 1
    fi
}

# Alias a floating tag to the verified digest, never to a mutable tag lookup.
# Capture any previous alias before mutation and restore it if tagging or the
# post-tag readback fails. A persistent registry violation still stops the run
# and requires operator recovery rather than allowing chart publication.
pre_oras_alias_digest() {
    local repository="$1" immutable_digest="$2" floating="$3"
    local previous_digest="" status
    if pre_oras_resolve "${repository}:${floating}"; then
        previous_digest="$PRE_ORAS_DIGEST"
    else
        status=$?
        if [[ "$status" -ne 10 ]]; then
            return 1
        fi
    fi
    if ! _pre_oras tag "${repository}@${immutable_digest}" "$floating"; then
        printf 'Failed to alias %s to verified digest %s\n' "$floating" "$immutable_digest" >&2
        pre_oras_restore_alias "$repository" "$immutable_digest" "$floating" "$previous_digest" || \
            printf 'Operator recovery is required for floating alias %s:%s\n' \
                "$repository" "$floating" >&2
        return 1
    fi
    if ! pre_oras_resolve_required "${repository}:${floating}"; then
        pre_oras_restore_alias "$repository" "$immutable_digest" "$floating" "$previous_digest" || \
            printf 'Operator recovery is required for floating alias %s:%s\n' \
                "$repository" "$floating" >&2
        return 1
    fi
    if [[ "$PRE_ORAS_DIGEST" != "$immutable_digest" ]]; then
        printf 'Floating alias %s:%s resolved to %s, expected %s\n' \
            "$repository" "$floating" "$PRE_ORAS_DIGEST" "$immutable_digest" >&2
        pre_oras_restore_alias "$repository" "$immutable_digest" "$floating" "$previous_digest" || \
            printf 'Operator recovery is required for floating alias %s:%s\n' \
                "$repository" "$floating" >&2
        return 1
    fi
}

# pre_publish_chart REPOSITORY PACKAGE IMMUTABLE [FLOATING]
# REPOSITORY excludes the oci:// scheme and includes the chart name.
pre_publish_chart() {
    if [[ "$#" -lt 3 || "$#" -gt 4 ]]; then
        printf 'Usage: pre_publish_chart REPOSITORY PACKAGE IMMUTABLE [FLOATING]\n' >&2
        return 2
    fi
    local repository="$1" package="$2" immutable="$3" floating="${4:-}"
    local present=0 status pull_dir immutable_digest
    if [[ ! -s "$package" ]]; then
        printf 'Chart package is missing or empty: %s\n' "$package" >&2
        return 1
    fi

    if pre_oras_resolve "${repository}:${immutable}"; then
        present=1
        immutable_digest="$PRE_ORAS_DIGEST"
    else
        status=$?
        if [[ "$status" -ne 10 ]]; then
            return 1
        fi
    fi
    if [[ "$present" -eq 0 ]]; then
        if ! _pre_helm push "$package" "oci://${repository%/*}"; then
            printf 'Helm push failed for immutable chart %s:%s\n' "$repository" "$immutable" >&2
            return 1
        fi
        if ! pre_oras_wait_for_artifact "${repository}:${immutable}"; then
            return 1
        fi
        immutable_digest="$PRE_ORAS_DIGEST"
    fi

    # Verify both reused and newly published content before moving an alias.
    if ! pull_dir="$(mktemp -d)"; then
        return 1
    fi
    if ! pre_helm_pull_chart "$repository" "$immutable" "$pull_dir"; then
        rm -rf "$pull_dir"
        return 1
    fi
    if ! cmp "$package" "$PRE_HELM_PULLED_CHART"; then
        printf 'Immutable chart collision or non-identical registry content: %s:%s\n' \
            "$repository" "$immutable" >&2
        rm -rf "$pull_dir"
        return 1
    fi
    rm -rf "$pull_dir"
    if ! pre_oras_resolve_required "${repository}:${immutable}"; then
        return 1
    fi
    if [[ "$PRE_ORAS_DIGEST" != "$immutable_digest" ]]; then
        printf 'Immutable chart moved during byte verification: %s:%s\n' \
            "$repository" "$immutable" >&2
        return 1
    fi
    if [[ -n "$floating" ]]; then
        if ! pre_oras_alias_digest "$repository" "$immutable_digest" "$floating"; then
            return 1
        fi
    fi
}
