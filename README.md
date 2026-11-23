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
evaluation, model registration, and inference integrations. First-party
application code lives at the repository root; deployment and integration
artifacts live under `infra/`. Vendored third-party stacks are isolated under
`infra/third_party/`.

## Directory map

| Layer | Directory | Contents |
|----|------|------|
| **Applications** | `python-backend/` | FastAPI API, synchronous SQLAlchemy worker, and notebook runtime |
| | `frontend/` | React, TypeScript, Ant Design, and Vite frontend |
| | `kfp-pipelines/` | Training, optimization, and evaluation pipeline components |
| | `optimizer-plugins/` | Manifest-driven optimization plugins |
| | `builders/` | Trainer, ML-backend, optimizer, evaluator, notebook, and shared SDK build contexts |
| **Application charts** | `charts/` | Backend, frontend, ML-backend, and tenant-check Helm charts |
| **Infrastructure** | `infra/charts/` | Supporting-service Helm charts |
| | `infra/scripts/` | Deployment, rendering, local verification, and image-build entry points |
| | `infra/configs/` | Host-neutral runtime configuration and templates |
| | `infra/third_party/` | Vendored upstream manifests and public-safe integration patches |
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

See [CONTRIBUTING.md](CONTRIBUTING.md) for component-specific checks, DCO
requirements, source headers, and API-contract rules. Local dependencies are
defined in `docker-compose.dev.yml`; the Compose file includes no Grafana,
Loki, or Tempo server components. Set a unique `POSTGRES_PASSWORD` in an
untracked `.env` before starting Compose, and provide the same value through
`NEREIDFORGE_DATABASE_PASSWORD` to a host-run backend when it uses that
database. Object-store and JWT credentials likewise come from untracked local
configuration; no shared development secret is supplied.

## Deployment

The deployment entry point is `infra/scripts/00_deploy-all.sh`, with separate
stages for infrastructure, Kubeflow integration, and NereidForge applications.
Configuration is rendered from `infra/configs/unified/config.yaml.j2` and the
host-neutral files under `infra/configs/`. Supply site-specific domains,
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

### Release packaging

Package the nine charts used by the active installation stages with:

```bash
make helm-package
```

The legacy `infra/charts/kubeflow` Helm umbrella is not an active installation
path and is not packaged. Kubeflow integration still uses the repository's
Kustomize stages and vendored manifests.

Build the repository-shaped installer archive with an explicit bare release
version when no exact Git tag is available:

```bash
RELEASE_TAG=0.1.0 make installer-package
```

The archive includes the deployment scripts, chart sources, configuration,
licenses, image manifest, and Kubeflow manifests. Its generated `release.env`
preserves the selected version after extraction without `.git`; an explicitly
exported `RELEASE_TAG` still takes precedence. The packager verifies all 23
suite-owned runtime/training image references and reparses the archive to prove
that it does not fall back to `0.0.0-dev`.

For a GitHub Release, `nereidforge-installer-<version>.tar.gz` is the **only
supported release installer**. GitHub's automatic **Source code** tar/zip
archives contain raw development chart metadata and no `release.env`; the
deployment scripts reject them rather than silently installing `0.0.0-dev`.
Use a real Git checkout for development or the custom installer for a release.

The rolling development workflow publishes only backend and frontend images.
It is not a complete Forge installation channel; use a formal release when the
ML backend and runtime/training images are required.

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
