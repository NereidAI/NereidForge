<!-- Copyright (C) 2026 YIQISOFT; SPDX-License-Identifier: Apache-2.0 -->

# Runbook: cutover from two NereidForge Helm releases to the consolidated release

## Scope and invariants

The former two application charts (`nereidforge-backend`, `nereidforge-frontend`)
were merged into one chart, `nereidforge`, deployed as a single Helm release
named **`nereidforge`** in the tenant namespace `nereidforge`. The release runs
three Deployments (API, worker, embedded SPA frontend) from one chart.

Invariants that do **not** change during cutover:

- Container image names and tags stay
  `ghcr.io/nereidai/nereidforge-backend:<tag>` and
  `ghcr.io/nereidai/nereidforge-frontend:<tag>`.
- The tenant namespace, externally owned Secrets, the internal CA bundle, the
  Ingress hostname/paths/TLS secret, the Service names, and the
  Deployment/HPA object names for `nereidforge-api`, `nereidforge-worker` and
  `nereidforge-frontend` all stay the same.
- The namespace carries `helm.sh/resource-policy: keep`, so uninstalling the
  old releases never deletes the namespace or non-Helm objects inside it.

Object names that **do** change (all rendered by the new chart on install):

| Kind | Old name (release `nereidforge-backend` / `nereidforge-frontend`) | New name (release `nereidforge`) |
|---|---|---|
| ServiceAccount | `nereidforge-backend`, `nereidforge-frontend` | `nereidforge` (single shared SA) |
| Secret (chart-managed) | `nereidforge-backend-secrets` | `nereidforge-secrets` |
| ConfigMap | `nereidforge-backend-config` | `nereidforge-config` |
| Role / RoleBinding | `nereidforge-backend-{notebooks,packager,trainjobs,trainlogs}` | `nereidforge-{notebooks,packager,trainjobs,trainlogs}` |
| post-install ConfigMap | `nereidforge-backend-post-install` | `nereidforge-post-install` |

Deployment label selectors change from `app.kubernetes.io/name=nereidforge-backend`
to `app.kubernetes.io/name=nereidforge` (the `component=<api|worker|frontend>`
label is unchanged). A Deployment's selector is immutable, so the three
Deployments cannot be patched in place; they are recreated during cutover.
Plan a short maintenance window for the reinstall path below.

## Preconditions and backup

1. Run the cutover from a checkout or installer that contains the consolidated
   chart and a matching published `nereidforge` OCI chart version.
2. Snapshot the live state so values and rollback stay available:

   ```bash
   helm -n nereidforge get values nereidforge-backend -a > backup-backend-values.yaml
   helm -n nereidforge get values nereidforge-frontend -a > backup-frontend-values.yaml
   helm -n nereidforge get manifest nereidforge-backend > backup-backend-manifest.yaml
   helm -n nereidforge get manifest nereidforge-frontend > backup-frontend-manifest.yaml
   kubectl get ns nereidforge -o yaml > backup-namespace.yaml
   ```

3. Confirm the externally owned objects exist (they are not Helm-managed by
   the application releases and survive uninstall):

   ```bash
   kubectl -n nereidforge get secret \
     nereidforge-service-credentials nereidforge-admin-credentials \
     pg-nereidforge-postgresql-credentials seaweedfs-s3-credentials \
     nereid-internal-ca
   ```

4. If the optional `harbor-registry-secret` exists and was adopted by the old
   backend release, export it first: an old-release uninstall deletes it
   because the old release owns it, while a default public install renders no
   replacement.

   ```bash
   kubectl -n nereidforge get secret harbor-registry-secret -o yaml \
     | grep -v '^\s*resourceVersion:\|^\s*uid:\|^\s*creationTimestamp:' \
     > backup-harbor-registry-secret.yaml
   ```

## Path A (recommended): same-namespace reinstall

This is the straightforward, deterministic path. It removes the two old
releases and installs the single new release in the kept namespace. Expect a
short API/UI interruption while the Deployments recreate; PVCs, Secrets and the
CNPG database are untouched.

1. Uninstall the two old releases. The namespace and every object without old
   release ownership remain; old-release-owned objects (old SA, ConfigMap,
   Roles, chart Secret, Services, Deployments, Ingress, HPAs) are removed.

   ```bash
   helm -n nereidforge uninstall nereidforge-frontend
   helm -n nereidforge uninstall nereidforge-backend
   kubectl -n nereidforge get all,sa,configmap,role,rolebinding,secret,ingress,hpa
   ```

2. Restore the optional registry secret when applicable:

   ```bash
   kubectl apply -f backup-harbor-registry-secret.yaml
   ```

3. Re-apply the cluster-wide GPU capacity RBAC. Its subject ServiceAccount is
   now the shared `nereidforge` SA instead of `nereidforge-backend`:

   ```bash
   kubectl apply -f scripts/manifests/gpu-capacity-rbac.yaml
   kubectl get clusterrolebinding nereidforge-node-reader -o yaml \
     | grep -A2 subjects
   ```

4. Install the consolidated release using the normal entry point, which keeps
   the same `--set` wiring (database, S3, SSO, LS/CVAT tokens, frontend host
   and five ingress routes) and re-renders the replicated PG credentials from
   the pre-owned CNPG bootstrap secret:

   ```bash
   bash scripts/40_nereidforge/app.sh nereidforge
   # or, for a direct OCI install with the validated handoff:
   helm upgrade --install nereidforge \
     oci://ghcr.io/nereidai/charts/nereidforge \
     --version "$NEREIDFORGE_VERSION" --namespace nereidforge \
     -f "$PRE_HANDOFF_DIR/values.generated.yaml" \
     -f "$NF_OPERATOR_VALUES_FILE" --wait --timeout 10m
   ```

   Post-install/post-upgrade hooks seed the platform administrator (without
   implicit password rotation when the retained admin Secret is present) and
   synchronize KFP pipeline definitions, exactly as the old backend release
   did.

5. Run the acceptance verifier and confirm the single release and three
   Deployments:

   ```bash
   bash scripts/60_verify/nf-check.sh nereidforge
   helm -n nereidforge list
   kubectl -n nereidforge get deploy -l app.kubernetes.io/name=nereidforge
   ```

## Path B (advanced): adopt kept objects and recreate only the Deployments

Use this only when preserving the existing Service ClusterIPs matters. The
immutable Deployment selectors still force Deployment recreation, so this path
is not zero-downtime; it only avoids Service/Ingress object replacement.

1. Relabel every object whose name is kept by the new chart (namespace, the
   two Services, the Ingress, both HPAs) with the new release ownership:

   ```bash
   for object in \
       "namespace/nereidforge" \
       "service/nereidforge-api" "service/nereidforge-frontend" \
       "ingress/nereidforge-frontend" \
       "hpa/nereidforge-api" "hpa/nereidforge-frontend"; do
       kubectl -n nereidforge label "$object" app.kubernetes.io/managed-by=Helm --overwrite
       kubectl -n nereidforge annotate "$object" \
         meta.helm.sh/release-name=nereidforge \
         meta.helm.sh/release-namespace=nereidforge --overwrite
   done
   ```

2. Delete the old Deployments explicitly (their immutable selectors cannot be
   adopted) and the old-release-owned objects whose names changed:

   ```bash
   kubectl -n nereidforge delete deploy nereidforge-api nereidforge-worker nereidforge-frontend
   kubectl -n nereidforge delete sa nereidforge-backend nereidforge-frontend
   kubectl -n nereidforge delete configmap nereidforge-backend-config \
     nereidforge-backend-post-install
   kubectl -n nereidforge delete secret nereidforge-backend-secrets
   kubectl -n nereidforge delete role,rolebinding -l app.kubernetes.io/instance=nereidforge-backend
   ```

3. Remove the old release history without touching the relabeled resources,
   then install the consolidated release and re-apply the GPU RBAC as in Path A
   steps 3-5:

   ```bash
   kubectl -n nereidforge delete secret -l owner=helm,name=nereidforge-backend
   kubectl -n nereidforge delete secret -l owner=helm,name=nereidforge-frontend
   helm upgrade --install nereidforge charts/nereidforge -n nereidforge \
     --create-namespace --wait --timeout 10m  # plus the documented --set flags
   ```

## Post-cutover checks

- `helm -n nereidforge list` shows exactly one application release,
  `nereidforge` (plus the independent `ml-backend` release).
- All three Deployments are ready and run under ServiceAccount `nereidforge`:

  ```bash
  kubectl -n nereidforge get deploy -o custom-columns=NAME:.metadata.name,SA:.spec.template.spec.serviceAccountName
  ```

- The Ingress still serves `/`, `/api`, `/auth/ns-callback`,
  `/v1/integration/ns` and `/notebook-servers` with the
  `nereidforge-frontend-tls` certificate.
- No orphaned `nereidforge-backend-*` or `nereidforge-frontend` SA/RBAC/ConfigMap
  objects remain; the notebook, packaging, training and train-logs Roles bind
  the new `nereidforge` SA.
- The platform admin login and the GPU capacity endpoint
  (`/api/v1/gpu-capacity`, backed by the updated ClusterRoleBinding) work.

## Rollback

1. Uninstall the consolidated release and re-apply the old GPU RBAC subject
   (`name: nereidforge-backend`) from the pre-change
   `scripts/manifests/gpu-capacity-rbac.yaml`.
2. Reinstall the two old charts at their previous versions, restoring the
   captured values:

   ```bash
   helm upgrade --install nereidforge-backend \
     oci://ghcr.io/nereidai/charts/nereidforge-backend \
     --version "<previous-version>" -n nereidforge -f backup-backend-values.yaml
   helm upgrade --install nereidforge-frontend \
     oci://ghcr.io/nereidai/charts/nereidforge-frontend \
     --version "<previous-version>" -n nereidforge -f backup-frontend-values.yaml
   ```

3. Recreate any object that cannot be restored from chart values using the
   manifest backups (`kubectl apply -f backup-*-manifest.yaml` after stripping
   Helm-managed metadata as needed).
