#!/usr/bin/env bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

readonly minimum_available_kb=$((48 * 1024 * 1024))

sudo docker image prune --all --force
root_source="$(findmnt -n -o SOURCE /)"
temporary_source="$(findmnt -n -o SOURCE /mnt)"
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
