<!-- Copyright (C) 2026 YIQISOFT; SPDX-License-Identifier: Apache-2.0 -->

# NereidForge — model factory control plane

![NereidForge dashboard](docs/images/nereidforge.png)

NereidForge provides dataset and annotation workflows, training, optimization,
evaluation, model registration, and inference integrations. First-party
application code lives under `app/`; deployment and integration artifacts live
under `infra/`. Vendored third-party stacks are isolated under
`infra/third_party/`.

## Directory map

| Layer | Directory | Contents |
|----|------|------|
| **Applications** | `app/python-backend/` | FastAPI API, synchronous SQLAlchemy worker, and notebook runtime |
| | `app/frontend/` | React, TypeScript, Ant Design, and Vite frontend |
| | `app/kfp-pipelines/` | Training, optimization, and evaluation pipeline components |
| | `app/optimizer-plugins/` | Manifest-driven optimization plugins |
| | `app/builders/` | Trainer, ML-backend, optimizer, evaluator, notebook, and shared SDK build contexts |
| **Application charts** | `app/charts/` | Backend, frontend, ML-backend, and tenant-check Helm charts |
| **Infrastructure** | `infra/charts/` | Supporting-service Helm charts |
| | `infra/scripts/` | Deployment, rendering, local verification, and image-build entry points |
| | `infra/configs/` | Host-neutral runtime configuration and templates |
| | `infra/third_party/` | Vendored upstream manifests and public-safe integration patches |
| **Public documentation** | `docs/product/` | Product overview and whitepaper |
| | `docs/api/openapi.yaml` | Hand-maintained HTTP API contract |
| | `app/builders/` | Trainer SDK, runtime, artifact, and dataset contracts |

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
