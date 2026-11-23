#!/usr/bin/env bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# Ensure host-side kubectl port-forwards needed when the FastAPI backend runs on
# the dev host (outside the cluster). Idempotent (skips ports already listening)
# and resilient (relaunches a forward that drops, e.g. on a pod roll / IP change).
#
# Forwards (must match configs/config.dev.yaml):
#   seaweedfs-s3  127.0.0.1:8833 -> nereidforge/seaweedfs-s3:8333   (S3)
#   ml-pipeline   127.0.0.1:18888 -> kubeflow/ml-pipeline:8888             (KFP)
#   model-registry 127.0.0.1:8889 -> default/model-registry-service:8080   (Hub)
set -uo pipefail

LOGDIR="${DEV_FWD_LOGDIR:-/tmp}"

# name:local_port:namespace:service:remote_port
FORWARDS=(
  "seaweedfs-s3:8833:nereidforge:seaweedfs-s3:8333"
  "ml-pipeline:18888:kubeflow:ml-pipeline:8888"
  "model-registry:8889:default:model-registry-service:8080"
  "labelstudio:8890:nereidforge:labelstudio:8080"
  "ml-backend:19090:nereidforge:ml-backend:9090"
)

port_listening() {
  ss -ltn 2>/dev/null | awk '{print $4}' | grep -q ":$1$"
}

start_fwd() {
  local name="$1" lp="$2" ns="$3" svc="$4" rp="$5"
  if port_listening "$lp"; then
    echo "[dev-fwd] $name 127.0.0.1:$lp already listening, skip"
    return 0
  fi
  echo "[dev-fwd] starting $name 127.0.0.1:$lp -> $ns/$svc:$rp"
  # Resilient wrapper: kubectl port-forward exits on pod roll / network blip; loop relaunches it.
  setsid nohup bash -c "while true; do sudo -n kubectl port-forward -n $ns svc/$svc $lp:$rp --address=127.0.0.1; echo '[dev-fwd] $name forward dropped, retrying in 3s'; sleep 3; done" \
    > "$LOGDIR/dev-fwd-$name.log" 2>&1 < /dev/null &
}

for f in "${FORWARDS[@]}"; do
  IFS=: read -r name lp ns svc rp <<< "$f"
  start_fwd "$name" "$lp" "$ns" "$svc" "$rp"
done

# Wait (up to ~30s) for every forward to be listening.
for _ in $(seq 1 30); do
  ok=1
  for f in "${FORWARDS[@]}"; do
    IFS=: read -r name lp ns svc rp <<< "$f"
    port_listening "$lp" || ok=0
  done
  [ "$ok" = 1 ] && break
  sleep 1
done

for f in "${FORWARDS[@]}"; do
  IFS=: read -r name lp ns svc rp <<< "$f"
  if port_listening "$lp"; then
    echo "[dev-fwd] OK   $name 127.0.0.1:$lp"
  else
    echo "[dev-fwd] FAIL $name 127.0.0.1:$lp (see $LOGDIR/dev-fwd-$name.log)"
  fi
done
