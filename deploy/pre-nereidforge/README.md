<!-- Copyright (C) 2026 YIQISOFT; SPDX-License-Identifier: Apache-2.0 -->

# Pre-NereidForge deployment assets

This module contains the supporting infrastructure and Kubeflow stage. Its
runtime scripts source only `scripts/lib/common.sh` and `scripts/lib/config.sh`
from this directory; the repository-root NereidForge libraries, product charts,
application images, JWT generation, and root values/secrets templates are not
part of the pre interface.

- `charts/` contains the five supporting-service Helm charts: CVAT, Label
  Studio, PostgreSQL, SeaweedFS, and Valkey.
- `platform/` contains the vendored Kubeflow manifests and their canonical patch
  provenance.
- `scripts/` contains infrastructure, Kubeflow, fetch/cache, verification, the
  non-secret NF Helm handoff/preflight, and local packaging scripts.
- `values/` contains pre-NereidForge chart value overlays.
- `tests/` contains stdlib regression tests for the local source/package
  interface.

## Local verification and ONLINE source packaging

Run from this directory (or use `make -C deploy/pre-nereidforge` from a full
checkout):

```bash
make verify
make package
make helm-package
```

`make verify` runs the stdlib tests, shell syntax checks, package/reparse tests,
and local Helm lint/template checks. It does **not** contact a registry or a
cluster, download chart dependencies, build images, or warm caches. CVAT is
rendered with its six tracked dependency archives. The PostgreSQL wrapper is
rendered with the external CNPG operator disabled and a metadata-only local
dependency placeholder because that operator chart is not vendored. When run from an
extracted installer, combined-checkout compatibility and package-rebuild tests
skip; the remaining version, interface, syntax, and chart checks still run.

`make package` stages only tracked files from this module, generates
`release.env` and `install-lock.json`, and writes these artifacts to `dist/` by
default:

```text
pre-nereidforge-installer-<version>.tar.gz
pre-nereidforge-installer-<version>.tar.gz.sha256
pre-nereidforge-installer-<version>.lock.json
```

The lock has `schemaVersion: 1` and records the pre version, source commit, all
five chart names/versions/appVersions, the Kubeflow community-distribution
baseline, and SHA-256 provenance for the tracked patch ledger. The same lock is
stored inside the archive. Packaging verifies the checksum, extracts the actual
archive outside the checkout, and sources its configuration with an empty
environment. The extracted copy must resolve the same version and module-local
chart/manifest paths without Git or repository-root files.

Overrides are optional:

```bash
# Local packaging example only; this does not publish a release.
PRE_RELEASE_TAG=0.0.1 PRE_OUTPUT_DIR=/tmp/pre-release make package
```

This is an **ONLINE source installer**, not an offline image bundle. Deployment
retains the existing upstream image references and runtime fetch/cache behavior.

`make helm-package` independently packages exactly `cvat`, `labelstudio`,
`postgresql`, `seaweedfs`, and `valkey`. It preserves their source chart and app
versions, stages CVAT's six tracked dependency archives, resolves other locked
dependencies in temporary storage, and writes chart packages plus an archive
under `dist/` by default. Package tar/gzip metadata is normalized so repeated
packaging of one checkout produces byte-identical charts for safe reruns. It
never uploads. This command can contact public
chart dependency repositories; unlike `make verify`, it is not a disconnected
check. The root `make helm-package` separately owns exactly three NF charts.

## Publication contract

A formal Git tag `pre-nereidforge/v<SemVer>` triggers only
`.github/workflows/pre-publish.yml`; the suffix becomes the bare module version
(for example, `pre-nereidforge/v0.0.1` maps to `0.0.1`). The workflow verifies
and packages the module, publishes exactly the five charts to the existing
`oci://ghcr.io/nereidai/charts/<name>` coordinates with independent upstream
versions, and anonymously pulls/renders each chart using isolated registry
configuration. The chart manifest records package SHA-256 and OCI manifest
digests. Each fixed upstream chart tag is first resolved and pulled with the
publisher credentials: a byte-identical archive is reused, a different archive
fails closed, and only an explicit not-found response permits a new push.
Formal pre tags use per-tag concurrency, so a pending rerun can replace only a
pending run for the same release. Before GHCR login, each run also performs an
oldest-run-first wait through the GitHub Actions API: it pages this workflow's
push runs and waits until every non-completed run with a lower run ID has
finished. Because an older run remains non-completed through chart publication
and release upload, distinct tags cannot race their shared fixed chart versions.
Only after those checks does it attach the installer, checksum, lock, and chart
manifest to the matching release. Existing published releases and existing
assets are never overwritten. It cannot publish NF images or charts.

Scoped pushes to `main` trigger `.github/workflows/pre-publish-dev.yml`. The
development channel publishes only pre artifacts: the five charts and an ORAS
installer artifact at `ghcr.io/nereidai/pre-nereidforge-installer`. Each has a
floating `0.0.0-dev` alias and an immutable `0.0.0-dev-<short-sha>` identity.
Development chart archives use the commit-qualified chart version while
retaining independent upstream appVersions; formal and ordinary local chart
packages retain upstream chart versions too. Immutable charts and installers
are pulled and compared byte-for-byte before reuse; mismatches and registry
errors fail without an overwrite. The installer is pushed only under its
immutable tag, then its floating alias is pointed at that verified digest.
ORAS chart aliases follow the same verified-digest rule, both identities are
anonymously pull-verified, and no moving GitHub Release is created. These pre
workflows do not trigger NF image publication or cache warming.

## Version selection

`pre_version()` and package resolution use only the pre release channel, in this
order:

1. An explicit, non-empty `PRE_RELEASE_TAG` (bare SemVer) wins.
2. A generated module `release.env` can supply `PRE_RELEASE_TAG` and
   `SOURCE_COMMIT`; an explicit environment tag still wins.
3. An exact Git tag named `pre-nereidforge/v<SemVer>` supplies the bare version.
4. Any other checkout uses `0.0.0-dev`.

A root NereidForge `v*` tag and the root `RELEASE_TAG` variable never select a
pre version. `PRE_ROOT` identifies this module; `REPO_ROOT` is used only for Git
metadata. Consequently, a normal nested worktree needs `.git` only at the
repository root. A copied source archive without generated `release.env` is
rejected instead of silently becoming a development installation.

No formal pre-NereidForge `v0.0.1` release has been published. The `0.0.1`
commands and tests are local fixtures only.

## Source execution and configuration

The existing entry points remain:

```bash
scripts/10_infra.sh
scripts/30_install-kubeflow.sh
scripts/40_finalize-handoff.sh
scripts/40_check-nf-handoff.sh
scripts/60_verify/infra-check.sh
scripts/60_verify/kubeflow-check.sh
scripts/60_verify/mr-check.sh
```

These commands perform real deployment or cluster verification; they are not
run by `make verify`. Site inputs retain their existing names, including
`BASE_DOMAIN`, `STORAGE_CLASS`, PostgreSQL and SeaweedFS settings/credentials,
Label Studio storage, `POD_CIDR`, `SERVICE_CIDR`, and the Dex identity/hash used
by Kubeflow installation and Model Registry verification. `KUBECTL`,
`OFFLINE_DIR`, and the upstream image overrides used by pre scripts also remain
available. Required values are still validated by the stage that consumes them;
no application JWT or NF application secret is generated by pre configuration.

The infrastructure entry point validates `boto3` and any credentials needed for
first-use initialization before creating the namespace or changing another
cluster resource. Existing canonical Secrets are checked by their owning stages
instead of requiring replacement site credentials.

The PostgreSQL stage retains
`nereidforge/pg-${PG_CLUSTER_NAME}-credentials` as the database identity. First
initialization requires `PG_ADMIN_PASSWORD` and creates a protected basic-auth
Secret through stdin with create-only semantics. Later runs reload that Secret;
a missing site password is accepted, while a conflicting non-empty value fails
before Secret or Helm mutation. The stored username must match `PG_DB_USER`.
The chart references this external Secret and has no `bootstrap.password`
value. Passwords never enter Helm values/arguments, connection-probe arguments,
logs, or the generated NF handoff.

The SeaweedFS stage retains `nereidforge/seaweedfs-s3-credentials` as the
object-store source of truth. On a first install it initializes that Kubernetes
Secret from the existing site inputs; on later runs it reuses the stored identity
without rotating it. Upgrades from the legacy SeaweedFS config preserve the same
identity. The Secret also carries `s3-config.json`, with anonymous reads limited
to the two serving prefixes. SeaweedFS, Label Studio, CVAT, and Kubeflow consume
this identity without putting it in Helm values or `--set` arguments. Before any
consumer starts, the stage idempotently creates and authenticates list access to
`mlpipeline`, `nereidforge`, `labelstudio`, and `cvat`; provisioning requires
`boto3` in the module-local virtual environment or the operator's Python.

`DEX_STATIC_EMAIL` defaults to `admin@nereidforge.local`, matching the vendored
Dex configuration, Kubeflow Profile, and RBAC. The Kubeflow stage rejects a
different identity or a malformed/non-bcrypt `DEX_STATIC_PASSWORD_HASH` before
reading cluster credentials. Its 18 ordered targets use one immutable render
per target and at most four apply attempts, 15 seconds apart by default, to
tolerate starting CRDs and admission webhooks. Annotation-size failures switch
to server-side apply. `KUBEFLOW_APPLY_ATTEMPTS` and
`KUBEFLOW_APPLY_RETRY_DELAY` can tune those limits; persistent failures retain
the target diagnostic and make the stage fail.

For standalone use, `PRE_ENV_FILE` selects a site env file and
`PRE_HANDOFF_DIR` selects the mode-0700 handoff directory. Defaults live under
a sibling `.deployment` tree. Full-stack callers can continue supplying
`NF_ENV_FILE` and `NF_HANDOFF_DIR`; pre maps these compatibility names to the
same paths. Existing service token/admin credential handoff filenames and
contents are unchanged.

After the infrastructure and Kubeflow stages both succeed, the Kubeflow entry
point finalizes `PRE_HANDOFF_DIR/values.generated.yaml`. This deterministic
`pre-nereidforge-handoff/v1` file contains only non-secret endpoints, Secret
names/keys, namespace ownership, ingress, and post-install settings. The same
file is valid for the consolidated NF nereidforge chart and the ML backend
chart. The finalizer
also retains exactly `NEREIDFORGE_LABELSTUDIO_TOKEN` and
`NEREIDFORGE_CVAT_TOKEN` in `nereidforge/nereidforge-service-credentials` with
server-side apply, preserving independently managed NF keys, and copies only
`ca.crt` to `nereidforge/nereid-internal-ca`.

The operator must retain `NEREIDFORGE_AUTH_JWT_SECRET` and the selected
`NEREIDFORGE_SSO_NS_CLIENT_SECRET` or `NEREIDFORGE_SSO_OIDC_CLIENT_SECRET` in the
runtime Secret, and `password` in `nereidforge-admin-credentials`. Pre never
creates or rotates those NF-owned credentials. Keep them out of both values
files and Helm `--set`. An empty non-secret overlay is written as `{}`.

Run the read-only preflight and then three OCI Helm releases in order from the
extracted pre installer:

```bash
export NF_OPERATOR_VALUES_FILE="$PRE_HANDOFF_DIR/operator-values.yaml"
bash scripts/40_check-nf-handoff.sh
export NEREIDFORGE_VERSION=<verified-formal-version>
# Consolidated application release: API, worker and the embedded SPA frontend.
helm upgrade --install nereidforge \
  oci://ghcr.io/nereidai/charts/nereidforge \
  --version "$NEREIDFORGE_VERSION" --namespace nereidforge \
  -f "$PRE_HANDOFF_DIR/values.generated.yaml" -f "$NF_OPERATOR_VALUES_FILE" \
  --wait --timeout 10m
helm upgrade --install ml-backend \
  oci://ghcr.io/nereidai/charts/nereidforge-ml-backend \
  --version "$NEREIDFORGE_VERSION" --namespace nereidforge \
  -f "$PRE_HANDOFF_DIR/values.generated.yaml" -f "$NF_OPERATOR_VALUES_FILE" \
  --wait --timeout 10m
```

The explicit production runbook uses two separate commands so a failure stops
the sequence. Existing sites that ran the retired `nereidforge-backend` and
`nereidforge-frontend` releases follow
[`docs/deployment/consolidated-helm-release-cutover.md`](../../docs/deployment/consolidated-helm-release-cutover.md)
for the same-namespace adoption or reinstall procedure. The nereidforge chart's
Helm hooks seed the administrator without
implicit password rotation and synchronize KFP pipelines. They are enabled by
the handoff, fail the Helm operation on error, and replace only their own
previous hook resources. `nereidforge-tenant-check` remains the third,
verification-only NF chart. The preflight performs no writes and supports only
the initial namespace-fixed tenant `nereidforge`.

An extracted installer is entered directly, for example:

```bash
tar -xzf pre-nereidforge-installer-<version>.tar.gz
cd pre-nereidforge-installer-<version>
# Set the required site inputs first, then invoke the desired scripts/ entry point.
```

The repository-root `scripts/00_deploy-all.sh` and `scripts/60_verify.sh`
compatibility paths still invoke these pre scripts and pass their established
environment for source-based development. Formal NF releases are now Helm-only
and require a compatible pre release; they no longer attach a full-stack source
installer or publish these five charts. Root exact-tag lookups exclude the
`pre-nereidforge/v*` namespace so a pre tag cannot be misread as an NF release.
NF `v*` and other legacy local tag behavior is unchanged and has focused
regression coverage.

## Remaining limitations

- The handoff covers the namespace-fixed initial NF tenant only. Broader
  multi-tenant wiring and application-internal credential redesign remain out of
  scope; every credential continues to live in an existing Kubernetes Secret.
- Independent formal/development publication workflows are implemented in
  source, but no formal `v0.0.1` tag, GitHub Release, or GHCR artifact has been
  published by this change.
- Local checks do not execute publication or establish registry availability,
  anonymous registry acceptance, runtime installation success, or cluster
  health. The explicit deployment/verification entry points still require a
  suitably configured cluster and operator inputs.
- No independent offline bundle is implemented in this stage. Existing legacy
  fetch/cache and full-stack offline compatibility scripts are preserved, but
  they are not a new pre offline deliverable.
