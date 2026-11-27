<!-- Copyright (C) 2026 YIQISOFT; SPDX-License-Identifier: Apache-2.0 -->

# kubeflow-patches

NereidForge customizations for `kubeflow/community-distribution`.

## Version baseline

- Upstream baseline: `kubeflow/community-distribution` tag **`26.03.1`** (no `v` prefix), commit `f09f3ee`.
- Fetch: `git clone --branch 26.03.1 --depth 1 https://github.com/kubeflow/community-distribution.git`.
- `../kubeflow-manifests` (the vendored directory) must equal pristine 26.03.1 plus these patches, additions, and removals byte for byte. Acceptance check: `diff -rq <rebuilt> ../kubeflow-manifests` must be empty.
- **Business version semantics:** the vendored tree represents the versions actually deployed by NereidForge, not necessarily the 26.03.1 upstream defaults. Selected newer versions (KFP 2.17.0, cert-manager v1.21.1, KServe v0.20.0, Trainer v2.3.0, Workspaces v2.0.0-beta.0, Jobset v0.12.0, oauth2-proxy v7.15.3, workflow-controller v4.0.5, and metacontroller v4.16.2) are restored explicitly through patches.
- Version assertions read the patched upstream manifests directly. For example, `fetch-kubeflow.sh` verifies the `kfp-api-server` tag in `applications/pipeline/upstream/base/pipeline/kustomization.yaml`.

## Layout

```
kubeflow-patches/
├── apply.sh          # Apply all patches, copy additions, then execute removals
├── removals.txt      # Files/directories present in 26.03.1 but absent from NF
├── patches/          # Per-domain patch.diff files (KCD 26.03.1 -> vendored)
│   ├── 01-podsecurity-privileged/    # Eight namespaces: restricted/baseline -> privileged
│   ├── 02-istio-1.30.3/              # Istio 1.30.1 -> 1.30.3, plus taint.name
│   ├── 03-argo-vendored/             # Argo kustomization: remote GitHub -> local vendor/
│   ├── 04-knative-postinstall/       # post-install-job kustomization resources: []
│   ├── 05-seaweedfs-external/        # Built-in SeaweedFS -> ExternalName for shared S3
│   ├── 06-pipeline-wait-db/          # API server wait-for-database/metadata initContainers
│   ├── 07-profile-controller/        # AWS_ENDPOINT_URL 8111 -> 9000; alpine/k8s 1.36.2
│   ├── 08-dex-user/                  # Dex static admin@nereidai.local user and password hash
│   ├── 09-hub-catalog/               # Hub v0.3.9 -> v0.3.14, MCP/agent catalog, external postgres
│   ├── 10-katib-startup/             # katib-mysql startup probe for first-boot stability
│   ├── 11-kfp-2.17.0/                # Pipeline tree 2.16.1 -> 2.17.0, including CRDs/config
│   ├── 12-kserve-0.20.0/             # KServe -> v0.20.0
│   ├── 13-trainer-2.3.0/             # Trainer -> v2.3.0, including the Jobset reference
│   ├── 14-workspaces-beta/           # Workspaces -> v2.0.0-beta.0
│   ├── 15-oauth2-7.15.3/             # oauth2-proxy -> v7.15.3
│   ├── 16-hub-options/               # Hub options, Istio authorization, and platform overlay
│   ├── 17-cert-manager-1.21.1/       # cert-manager -> v1.21.1
│   └── 18-misc-cosmetics/            # Istio/namespace/dashboard text and platform references
└── additions/       # NF-only files copied verbatim into the manifest tree
```

The complete difference ledger is: relative to 26.03.1, the vendored tree is the business snapshot with generated clutter removed (three `kustomization.yaml.orig` files, the `.wfp-kubeflow-new` deployment temporary directory, the empty `common/third-party` directory, and the nested `third-party/third-party` directory). NF semantics such as `nereidforge-admin` remain in their patches. Independent upstream-master evolution is included only where it belongs to a selected business version, such as the KFP 2.17.0 `dbHost` to `mysqlHost` rename carried by patch 11.

## Replay

```bash
# Assuming 26.03.1 has been cloned to ./kcd-latest
./apply.sh ./kcd-latest
# Acceptance: no differences from the vendored tree (apart from KCD top-level scaffolding)
diff -rq ./kcd-latest ../kubeflow-manifests
```

## Regenerating after an upstream or business-version change

```bash
# 1. Build a baseline by applying the existing patches to pristine 26.03.1.
# 2. Produce the intended final vendored snapshot without generated clutter.
# 3. Regenerate the affected per-domain patches from baseline -> final.
#    Binary business data belongs in additions, not patches.
# 4. Run the complete replay acceptance check above.
```

## Invariants

- The vendored tree must equal pristine 26.03.1 plus patches, additions, and removals. Every manual change must flow back into its canonical patch or addition.
- Every image keeps its pristine public upstream reference. Do not add image-transform components that rewrite registry hosts or paths.
- Online and air-gapped deployments consume the same manifests. Air-gapped nodes resolve the unchanged public references through node-level registry mirrors, swapping only the registry host while preserving the image path.
- The NF AuthorizationPolicy additions carry `${POD_CIDR}` and `${SERVICE_CIDR}` placeholders. `30_install-kubeflow.sh` requires both inputs and renders them only in its temporary manifest copy; the canonical addition and vendored copy must remain byte-identical.
- The Dex Secret carries only `${DEX_STATIC_PASSWORD_HASH}`. Replay normalization replaces the upstream hash without copying it into the patch ledger, and both installation paths require the operator-managed hash before rendering or applying Dex.
- The deployment script's historical Dex username `sed` was superseded by patch 08 and is an idempotent no-op after patching.
