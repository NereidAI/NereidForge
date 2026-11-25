<!-- Copyright (C) 2026 YIQISOFT; SPDX-License-Identifier: Apache-2.0 -->

# NereidForge — model factory control plane

[![Manual cache warm](https://github.com/NereidAI/NereidForge/actions/workflows/ci.yml/badge.svg)](https://github.com/NereidAI/NereidForge/actions/workflows/ci.yml)
[![Dev Images](https://github.com/NereidAI/NereidForge/actions/workflows/publish-dev.yml/badge.svg)](https://github.com/NereidAI/NereidForge/actions/workflows/publish-dev.yml)
[![Release Images](https://github.com/NereidAI/NereidForge/actions/workflows/publish-images.yml/badge.svg)](https://github.com/NereidAI/NereidForge/actions/workflows/publish-images.yml)
[![DCO](https://github.com/NereidAI/NereidForge/actions/workflows/dco.yml/badge.svg)](https://github.com/NereidAI/NereidForge/actions/workflows/dco.yml)
[![License Compliance](https://github.com/NereidAI/NereidForge/actions/workflows/license-compliance.yml/badge.svg)](https://github.com/NereidAI/NereidForge/actions/workflows/license-compliance.yml)

[![License](https://img.shields.io/github/license/NereidAI/NereidForge)](LICENSE)
[![Architectures](https://img.shields.io/badge/arch-amd64%20%7C%20arm64-627185?logo=linux&logoColor=white)](https://github.com/NereidAI/NereidForge/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-%3E%3D3.14-3776AB?logo=python&logoColor=white)](python-backend/pyproject.toml)
[![Node](https://img.shields.io/badge/node-%3E%3D24-339933?logo=node.js&logoColor=white)](frontend/package.json)
[![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-fe519e?logo=conventionalcommits&logoColor=white)](https://www.conventionalcommits.org/)

[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)](frontend/package.json)
[![Ant Design](https://img.shields.io/badge/Ant%20Design-5-0170FE?logo=antdesign&logoColor=white)](frontend/package.json)
[![TypeScript](https://img.shields.io/badge/TypeScript-5-3178C6?logo=typescript&logoColor=white)](frontend/package.json)
[![Vite](https://img.shields.io/badge/Vite-5-646CFF?logo=vite&logoColor=white)](frontend/package.json)
[![Kubeflow](https://img.shields.io/badge/Kubeflow-integrated-4279F3?logo=kubeflow&logoColor=white)](https://www.kubeflow.org/)

[![GHCR backend](https://img.shields.io/badge/GHCR-backend-2496ED?logo=docker&logoColor=white)](https://github.com/NereidAI/NereidForge/pkgs/container/nereidforge-backend)
[![GHCR frontend](https://img.shields.io/badge/GHCR-frontend-2496ED?logo=docker&logoColor=white)](https://github.com/NereidAI/NereidForge/pkgs/container/nereidforge-frontend)

![NereidForge dashboard](docs/images/nereidforge.png)

NereidForge provides dataset and annotation workflows, training, optimization,
evaluation, model registration, and inference integrations. Delivery is divided
into three Modules: Pre-NereidForge infrastructure, NereidForge Builders, and
Application-only NereidForge. They currently share one repository while their
ownership and verification seams are established; separate repositories are a
later extraction decision, not a prerequisite for independent delivery. See
[ADR 0001](docs/three-delivery-modules.md) and the
[domain glossary](CONTEXT.md). Inactive legacy assets and externally owned CRDs
remain under `infra/`.

## Directory map

| Layer | Directory | Contents |
|----|------|------|
| **Application-only NereidForge** | `python-backend/` | FastAPI API, synchronous SQLAlchemy worker, and notebook runtime |
| | `frontend/` | React, TypeScript, Ant Design, and Vite frontend |
| | `kfp-pipelines/` | Training, optimization, and evaluation pipeline definitions, excluding builder-owned `docker/` |
| **NereidForge Builders** | `builders/` | Trainer, ML-backend, optimizer, evaluator, notebook, and shared SDK build contexts |
| | `kfp-pipelines/docker/` | Builder-owned runtime and conversion image build contexts |
| | `optimizer-plugins/` | Optional builder-owned optimization plugin images |
| **Application charts** | `charts/` | Backend, frontend, ML-backend, and tenant-check Helm charts |
| **Shared tooling** | `configs/` | Host-neutral runtime configuration, templates, and development monitoring |
| | `scripts/` | Full-stack compatibility, application, build, verification, and packaging entry points |
| **Pre-NereidForge deployment** | `deploy/pre-nereidforge/charts/` | Supporting-service Helm charts |
| | `deploy/pre-nereidforge/platform/` | Vendored Kubeflow manifests and public-safe integration patches |
| | `deploy/pre-nereidforge/scripts/` | Infrastructure, Kubeflow, and corresponding verification entry points |
| | `deploy/pre-nereidforge/values/` | Pre-NereidForge chart value overlays |
| **Legacy/external assets** | `infra/` | Inactive Kubeflow umbrella chart and externally owned cert-manager CRDs |
| **Public documentation** | `docs/product/` | Product overview and whitepaper |
| | `docs/api/openapi.yaml` | Hand-maintained HTTP API contract |
| | `builders/` | Trainer SDK, runtime, artifact, and dataset contracts |

## Development and verification

The backend requires Python 3.14 or newer. The frontend requires Node.js 24 or
newer. Prepare the shared development environment and run the standard gates:

```bash
make dev-build
make lint
make test
make license-check
```

Each delivery Module also has a focused offline gate. These commands do not
build the release image set, contact a registry, install into a cluster, or
publish artifacts:

```bash
make -C deploy/pre-nereidforge verify
make -C builders verify
make nf-verify
```

The builders gate includes its SDK, Paddle, XPU, inventory, and Dockerfile
contracts. The NF gate composes the builder-owned 21/16-image hardware
inventories with the two application images: hardware dev profiles publish 21
(`full`) or 16 (`cpu-xpu`) builder images, `backend-frontend` publishes only
the two application images, and the formal release gate still requires all 23
images. The pre gate verifies only infrastructure, its five charts, installer,
and handoff. The legacy aggregate `make test` remains as compatibility coverage
while assertions move to their owning Modules.

See [CONTRIBUTING.md](CONTRIBUTING.md) for Module-specific checks, DCO
requirements, source headers, and API-contract rules. Local dependencies are
defined in `docker-compose.dev.yml`; the Compose file includes no Grafana,
Loki, or Tempo server components. Set a unique `POSTGRES_PASSWORD` in an
untracked `.env` before starting Compose, and provide the same value through
`NEREIDFORGE_DATABASE_PASSWORD` to a host-run backend when it uses that
database. Object-store and JWT credentials likewise come from untracked local
configuration; no shared development secret is supplied.

### Development OCI publication

An NF development publication explicitly selects one of three image profiles.
`backend-frontend` builds and publishes only `nereidforge-backend` and
`nereidforge-frontend`; it schedules no builder, dependent-image, or chart jobs.
The two hardware profiles publish builder images only: `full` covers all 21
builder images and `cpu-xpu` the 16 core/CPU/Intel-XPU builder images, each
with the same three OCI Helm charts. Neither hardware profile builds the
application images; they are exclusive to `backend-frontend`.
`cpu-xpu` excludes only the five AMD/NVIDIA trainer images
(`trainer-torch-cu`, `trainer-paddle-cu`, `trainer-llm-cu`,
`trainer-torch-rocm`, and `trainer-llm-rocm`); Intel XPU images remain required.
Publication occurs only by explicitly dispatching `publish-dev` on `main`;
pushes, merges, and pull requests never start this workflow. The publication
matrix is never a pull-request check. Select only the intended scope:

```bash
# Two application images only; no builder images or charts.
gh workflow run publish-dev.yml --repo NereidAI/NereidForge --ref main \
  -f image_profile=backend-frontend

# 16-image core/CPU/Intel-XPU builder snapshot plus three charts.
gh workflow run publish-dev.yml --repo NereidAI/NereidForge --ref main \
  -f image_profile=cpu-xpu

# Full 21-builder-image snapshot plus three charts (the default).
gh workflow run publish-dev.yml --repo NereidAI/NereidForge --ref main \
  -f image_profile=full
```

Each run uses immutable `0.0.0-dev-<short-sha>` identities and floating
`0.0.0-dev` aliases. Same-commit immutable artifacts are anonymously verified
and reused without overwrite. Every image selected by the run's profile must
pass anonymous revision verification before any selected floating alias moves.
For `full` and `cpu-xpu`, chart publication starts only after that image gate;
`backend-frontend` publishes no charts. The run summary records the profile and
all 2, 16, or 21 verified digests. Excluded aliases are not evidence for the
selected snapshot. Registry failures and source-revision collisions fail closed.
Repeating a manual dispatch for the same commit and profile resumes missing
artifacts without rewriting existing ones; no formal release is created.

## Deployment

The full-stack deployment entry point is `scripts/00_deploy-all.sh`, which calls
the infrastructure and Kubeflow stages under `deploy/pre-nereidforge/scripts/`
before deploying NereidForge applications. Configuration is rendered from
`configs/unified/config.yaml.j2` and the host-neutral files under `configs/`.
Supply site-specific domains,
object-store settings, NFS settings, identity-provider URLs, and secrets
through the documented environment inputs or untracked operator overrides.
Production overlays must not contain real credentials. Deployment requires
operator-managed PostgreSQL, object-store, JWT, and Dex credentials; in
particular, `DEX_STATIC_PASSWORD_HASH` must be the bcrypt hash matching the
verification probe's `DEX_STATIC_PASSWORD`. No shared credential fallback is
provided.

Every image reference remains at its public coordinate: suite-built images use
`ghcr.io/nereidai/<flat-name>:<tag>`, and third-party images retain their
upstream registry references. Air-gapped operators can resolve the same
references through node-level registry mirrors without rewriting tracked
configuration.

### Chart packaging and release contract

Local chart packaging is split by ownership:

```bash
# Exactly three NF product charts; no upload.
make helm-package

# Exactly five supporting-service charts with independent upstream versions;
# dependencies are resolved in a temporary staging tree, and nothing is uploaded.
make -C deploy/pre-nereidforge helm-package
```

The legacy `infra/charts/kubeflow` Helm umbrella is not an active installation
path and is not packaged. Kubeflow remains an ordered Kustomize/platform
installer in the pre module, not an umbrella Helm chart.

A formal NF Git tag `v<SemVer>` publishes the 23 NF-owned images and only the
three charts under `charts/`. Its GitHub Release contains only
`nereidforge-images-<version>.txt` and `nereidforge-charts-<version>.txt`. NF is
installed through those OCI Helm charts against a compatible pre-NereidForge
release; NF releases do not publish pre charts or a repository-shaped
full-stack source installer.

A formal pre Git tag `pre-nereidforge/v<SemVer>` maps to the bare pre module
version. Its independent workflow publishes CVAT, Label Studio, PostgreSQL,
SeaweedFS, and Valkey under `oci://ghcr.io/nereidai/charts/`, preserving each
chart's upstream chart/app versions. The matching GitHub Release carries the
versioned pre installer, checksum, lock, and chart manifest. No formal pre
`v0.0.1` tag or release has been created by this source change.

The pre development channel is triggered by scoped main-branch changes. It
publishes an OCI installer artifact and all five charts with floating
`0.0.0-dev` and immutable `0.0.0-dev-<short-sha>` identities. Floating tags are
aliases of the immutable manifests; no moving GitHub Release is created.

The separate NF development workflow publishes either the two-image
`backend-frontend` application scope without charts, or a builder-only
hardware snapshot of 21 `full` images or 16 `cpu-xpu` images with exactly
three charts. The application images are never part of a hardware-profile run.
All profiles use the same floating/immutable identity contract. Image builds
never overwrite an existing immutable tag and push no floating tag. Every
selected immutable manifest must pass anonymous source-revision verification
before its floating alias moves; suite chart publication starts only after that
gate. Same-commit reruns reuse verified artifacts, and repackaged charts are
byte-identical. No formal NF release is required.

`make installer-package` and `scripts/04_package-installer.sh` remain as a
**deprecated local-only compatibility interface** for existing full-stack
source workflows. They are not called by NF publication and their archive is
not an NF release asset.

### Direct Helm installation handoff

A successful pre installation retains the two pre-owned Label Studio/CVAT token
keys in `nereidforge/nereidforge-service-credentials`, replicates only the public
CA certificate, and writes `values.generated.yaml` below `PRE_HANDOFF_DIR`. That
file contains endpoints, Secret names/keys, and image-independent chart values;
it contains no credentials. The operator retains the NF JWT, selected SSO client
secret, and administrator password in existing Kubernetes Secrets. Do not put
those values in either values file or pass them with Helm `--set`.

Create a non-secret `operator-values.yaml` (use `{}` when the generated defaults
need no override), add the NF-owned Secret keys without replacing the pre-owned
keys, and run the read-only preflight. Load `PRE_HANDOFF_DIR`,
`NF_OPERATOR_VALUES_FILE`, and `NEREIDFORGE_VERSION=0.0.0-dev` from the
persistent non-secret operator environment, then install in this order:

```bash
: "${PRE_HANDOFF_DIR:?Load the persistent operator environment}"
: "${NF_OPERATOR_VALUES_FILE:?Load the persistent operator environment}"
: "${NEREIDFORGE_VERSION:?Load the persistent operator environment}"
bash scripts/40_check-nf-handoff.sh  # run from the extracted pre installer

# Consolidated application release: API, worker and the embedded SPA frontend
# in one Helm chart; the frontend tier is configured under the frontend: block.
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

The nereidforge chart's post-install/post-upgrade hooks seed the administrator
without implicit password rotation and synchronize the existing KFP pipeline
definitions. Hook failures fail the Helm operation; they are not warning-only
steps. The third chart, `nereidforge-tenant-check`, remains a verification tool,
not an application release. No NF source installer is required or published.

These local/static checks do not establish GHCR publication success, anonymous
registry availability, installation acceptance, or cluster health.

## License and security

NereidForge is distributed under Apache-2.0. Read
[LICENSE-POLICY.md](LICENSE-POLICY.md) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before adding or distributing
dependencies. A clean static scan is not, by itself, a determination that every
built image or transitive dependency is compliant; image-level findings and
review items must be resolved explicitly.

Report vulnerabilities privately using the process in
[SECURITY.md](SECURITY.md).

## Acknowledgments

NereidForge builds on the following open-source projects through runtime
integrations, pipeline tooling, container images, or self-hosted frontend
assets:

- [Kubeflow](https://github.com/kubeflow) for pipeline orchestration, training, and model-registry integration.
- [Label Studio](https://github.com/heartexlabs/label-studio) for annotation workflows and ML-backend integration.
- [CVAT](https://github.com/cvat-ai/cvat) for image and video annotation service integration.
- [Netron](https://github.com/lutzroeder/netron) for the self-hosted model graph viewer.

Third-party projects retain their own licenses. NereidForge is an independent
integration and is not endorsed by these projects or their maintainers.
