#!/bin/bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# gen-subcharts.sh - generate one helm subchart per Kubeflow component.
# Each subchart applies its upstream kustomize target(s) unchanged (upstream
# image references; air-gapped resolution uses the node containerd mirror) via
# a post-install/upgrade hook Job, driven by global values.
# Run from the kubeflow chart dir: bash hack/gen-subcharts.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHARTS="$ROOT/charts"
mkdir -p "$CHARTS"

# name|hookWeight|targets(csv)|desc
COMPS=(
  "istio|10|common/istio/istio-crds/base,common/istio/istio-namespace/base,common/istio/istio-install/overlays/insecure|istio"
  "kubeflow-namespace|5|common/kubeflow-namespace/base|kubeflow namespace"
  "kubeflow-roles|6|common/kubeflow-roles/base|kubeflow roles"
  "dex|20|common/dex/base,common/dex/overlays/istio|dex"
  "oauth2-proxy|25|common/oauth2-proxy/overlays/m2m-dex-only|oauth2-proxy"
  "kubeflow-istio-resources|27|common/istio/kubeflow-istio-resources/base|kubeflow istio resources"
  "knative|30|common/knative/knative-serving/base,common/knative/knative-serving-post-install-jobs/base,common/knative/knative-eventing/base,common/knative/knative-eventing-post-install-jobs/base|knative"
  "kfp|40|applications/pipeline/upstream/env/platform-agnostic-multi-user|KFP pipeline"
  "katib|45|applications/katib/upstream/installs/katib-with-kubeflow|katib"
  "kserve|50|applications/kserve/kserve|kserve"
  "dashboard|55|applications/dashboard/overlays/istio|dashboard"
  "notebooks|60|applications/notebooks-v1/overlays/istio|notebooks"
  "trainer|65|applications/trainer/overlays|trainer"
  "workspaces|70|applications/workspaces/overlays/istio|workspaces"
)

gen_one() {
  local name="$1" weight="$2" targets="$3" desc="$4"
  local d="$CHARTS/$name"
  # Only consuming Jobs receive credentials. secretKeyRef prevents Kubernetes
  # from expanding $(...) sequences inside literal operator-supplied values.
  local credential_env=""
  if [[ "$name" == "dex" ]]; then
    credential_env='            - name: DEX_STATIC_EMAIL
              value: {{ .Values.global.dex.staticEmail | quote }}
            - name: DEX_STATIC_PASSWORD_HASH
              valueFrom:
                secretKeyRef:
                  name: "{{ .Release.Name }}-kubeflow-credentials"
                  key: DEX_STATIC_PASSWORD_HASH'
  elif [[ "$name" == "kfp" ]]; then
    credential_env='            - name: SEAWEEDFS_S3_ACCESS_KEY
              valueFrom:
                secretKeyRef:
                  name: "{{ .Release.Name }}-kubeflow-credentials"
                  key: SEAWEEDFS_S3_ACCESS_KEY
            - name: SEAWEEDFS_S3_SECRET_KEY
              valueFrom:
                secretKeyRef:
                  name: "{{ .Release.Name }}-kubeflow-credentials"
                  key: SEAWEEDFS_S3_SECRET_KEY'
  fi
  mkdir -p "$d/templates" "$d/files"

  cat > "$d/Chart.yaml" <<EOF
apiVersion: v2
name: $name
description: Kubeflow $desc component (helm subchart of charts/kubeflow)
type: application
version: 0.1.0
EOF

  cat > "$d/values.yaml" <<EOF
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# Per-component subchart values. Global config (dex, tenant, manifests,
# kubeconfig, image) comes from the parent chart's .Values.global and is
# injected into the install hook at render time.
targets: "$targets"
hookWeight: $weight
EOF

  cat > "$d/templates/configmap.yaml" <<'CM_EOF'
apiVersion: v1
kind: ConfigMap
metadata:
  name: "{{ .Release.Name }}-{{ .Chart.Name }}-script"
  annotations:
    "helm.sh/hook": pre-install,pre-upgrade
    "helm.sh/hook-weight": "{{ .Values.hookWeight }}"
    "helm.sh/hook-delete-policy": before-hook-creation
data:
  apply.sh: |
{{ .Files.Get "files/apply.sh" | indent 4 }}
CM_EOF

  cat > "$d/templates/install-hook.yaml" <<HOOK_EOF
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

apiVersion: batch/v1
kind: Job
metadata:
  name: "{{ .Release.Name }}-install-{{ .Chart.Name }}"
  annotations:
    "helm.sh/hook": post-install,post-upgrade
    "helm.sh/hook-weight": "{{ .Values.hookWeight }}"
    "helm.sh/hook-delete-policy": before-hook-creation,hook-succeeded
spec:
  backoffLimit: 1
  activeDeadlineSeconds: 1800
  template:
    metadata:
      name: "{{ .Release.Name }}-install-{{ .Chart.Name }}"
    spec:
      restartPolicy: Never
      containers:
        - name: install
          image: "{{ .Values.global.image.repository }}:{{ .Values.global.image.tag }}"
          command: ["/bin/bash", "/scripts/apply.sh"]
          env:
            - name: MANIFESTS_DIR
              value: /manifests
${credential_env}
            - name: TARGETS
              value: {{ .Values.targets | quote }}
          volumeMounts:
            - name: manifests
              mountPath: /manifests
            - name: kubeconfig
              mountPath: /kube
              readOnly: true
            - name: scripts
              mountPath: /scripts
              readOnly: true
      volumes:
        - name: manifests
          hostPath:
            path: {{ .Values.global.manifests.hostPath | quote }}
            type: Directory
        - name: kubeconfig
          hostPath:
            path: {{ .Values.global.kubeconfig.hostPath | quote }}
            type: File
        - name: scripts
          configMap:
            name: "{{ .Release.Name }}-{{ .Chart.Name }}-script"
            defaultMode: 0755
HOOK_EOF

  cat > "$d/files/apply.sh" <<'APPLY_EOF'
#!/bin/bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# Per-component install hook: kustomize build the upstream target(s) keeping
# their upstream image references, then apply. Air-gapped nodes resolve the
# public image refs through the containerd registry mirror rendered by
# NereidOverview harbor-sync.sh (root treaty section 18), so no image rewrite
# and no pull secret are embedded in this chart.
set -euo pipefail

MANIFESTS_DIR="${MANIFESTS_DIR:-/manifests}"
DEX_STATIC_EMAIL="${DEX_STATIC_EMAIL:-admin@nereidforge.local}"
DEX_STATIC_PASSWORD_HASH="${DEX_STATIC_PASSWORD_HASH:-}"
SEAWEEDFS_S3_ACCESS_KEY="${SEAWEEDFS_S3_ACCESS_KEY:-}"
SEAWEEDFS_S3_SECRET_KEY="${SEAWEEDFS_S3_SECRET_KEY:-}"
TARGETS="${TARGETS:?TARGETS required}"

WFP_DIR="$MANIFESTS_DIR/.wfp-$(echo "$TARGETS" | md5sum | cut -c1-8)"
KUSTOMIZATION="$WFP_DIR/kustomization.yaml"
mkdir -p "$WFP_DIR"

python3 - "$MANIFESTS_DIR" "$TARGETS" "$WFP_DIR" <<'PY'
import sys, os
m, targets, wfp = sys.argv[1], sys.argv[2], sys.argv[3]
res = "\n".join(f"  - ../{t}" for t in targets.split(","))
open(os.path.join(wfp, "kustomization.yaml"), "w").write(
    "apiVersion: kustomize.config.k8s.io/v1beta1\n"
    "kind: Kustomization\n"
    "resources:\n" + res + "\n"
)
PY

rendered=$(mktemp)
if ! kustomize build "$WFP_DIR" > "$rendered"; then
  echo "[ERROR] kustomize build failed: $TARGETS" >&2
  rm -f "$rendered"
  exit 1
fi

# Replace complete YAML scalars with JSON strings, which are valid YAML and
# preserve quotes, backslashes, newlines, comment markers, and boolean-like keys.
python3 - "$rendered" <<'PY'
import json
import os
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as stream:
    text = stream.read()
for name in ("SEAWEEDFS_S3_ACCESS_KEY", "SEAWEEDFS_S3_SECRET_KEY"):
    token = "${" + name + "}"
    encoded = json.dumps(os.environ.get(name, ""))
    for scalar in (f'"{token}"', f"'{token}'", token):
        text = text.replace(scalar, encoded)
with open(path, "w", encoding="utf-8") as stream:
    stream.write(text)
PY

# Only the tenant-specific Dex identity is substituted; image references stay
# at their pristine upstream coordinates.
sed -i -E \
  -e "s#^  DEX_USER_PASSWORD: .*#  DEX_USER_PASSWORD: ${DEX_STATIC_PASSWORD_HASH}#" \
  -e "s#email: user@example.com#email: ${DEX_STATIC_EMAIL}#g" \
  "$rendered"

if [ ! -s "$rendered" ]; then
  echo "[INFO] (no resources to apply, skipping) $TARGETS"
  rm -f "$rendered"
  exit 0
fi

if ! kubectl apply -f "$rendered" 2>"$rendered.err"; then
  if grep -qE "262144|Too long" "$rendered.err"; then
    echo "[WARN] annotation limit, retrying with server-side apply"
    if ! kubectl apply --server-side=true --field-manager=wfp --force-conflicts -f "$rendered"; then
      echo "[ERROR] server-side apply failed: $TARGETS" >&2
      cat "$rendered.err" >&2
      exit 1
    fi
  else
    echo "[ERROR] kubectl apply failed: $TARGETS" >&2
    cat "$rendered.err" >&2
    exit 1
  fi
fi
rm -f "$rendered" "$rendered.err" "$KUSTOMIZATION"
echo "[OK] applied: $TARGETS"
APPLY_EOF

  # Only the Dex component consumes the password hash. Fail its hook before
  # rendering an empty Secret; unrelated components do not require the value.
  if [[ "$name" == "dex" ]]; then
    sed -i '/^DEX_STATIC_PASSWORD_HASH=/a : "${DEX_STATIC_PASSWORD_HASH:?Dex static password hash is required}"' \
      "$d/files/apply.sh"
  fi
  if [[ "$name" == "kfp" ]]; then
    sed -i '/^SEAWEEDFS_S3_SECRET_KEY=/a : "${SEAWEEDFS_S3_ACCESS_KEY:?SeaweedFS S3 access key is required for KFP}"\n: "${SEAWEEDFS_S3_SECRET_KEY:?SeaweedFS S3 secret key is required for KFP}"' \
      "$d/files/apply.sh"
  fi

  chmod +x "$d/files/apply.sh"
  echo "generated subchart: $name (weight=$weight, targets=$targets)"
}

for c in "${COMPS[@]}"; do
  IFS='|' read -r name weight targets desc <<< "$c"
  gen_one "$name" "$weight" "$targets" "$desc"
done

echo "done: ${#COMPS[@]} subcharts generated under $CHARTS"
