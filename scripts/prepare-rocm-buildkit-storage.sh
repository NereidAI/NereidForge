#!/usr/bin/env bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

readonly minimum_available_kb=$((48 * 1024 * 1024))

filesystem_source() {
  local target="$1"
  local source
  if ! source="$(findmnt -n -o SOURCE --target "$target")"; then
    echo "::error::Unable to identify the filesystem containing $target" >&2
    return 1
  fi
  if [[ -z "$source" ]]; then
    echo "::error::No source device found for the filesystem containing $target" >&2
    return 1
  fi
  printf '%s\n' "$source"
}

sudo docker image prune --all --force
root_source="$(filesystem_source /)"
temporary_source="$(filesystem_source /mnt)"
storage_path=/mnt
move_storage=true
if [[ -z "$temporary_source" || "$temporary_source" == "$root_source" ]]; then
  storage_path=/
  move_storage=false
  echo "/mnt shares the root filesystem; using root capacity for ROCm BuildKit storage"
else
  echo "Using the independent /mnt filesystem for ROCm BuildKit storage"
fi

available_kb="$(df -Pk "$storage_path" | awk 'NR == 2 {print $4}')"
if [[ ! "$available_kb" =~ ^[0-9]+$ ]]; then
  echo "::error::Unable to determine free space at $storage_path for the ROCm build"
  exit 1
fi
if (( available_kb < minimum_available_kb )); then
  echo "::error::The ROCm build requires at least 48 GiB free at $storage_path"
  exit 1
fi

if [[ "$move_storage" == true ]]; then
  sudo systemctl stop docker.socket docker.service containerd.service
  sudo mkdir -p /mnt/nf-docker /mnt/nf-containerd /var/lib/docker /var/lib/containerd
  sudo mount --bind "/mnt/nf-docker" /var/lib/docker
  sudo mount --bind "/mnt/nf-containerd" /var/lib/containerd
  sudo systemctl start containerd.service docker.service
fi

df -h "$storage_path"
