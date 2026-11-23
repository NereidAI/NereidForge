<!-- Copyright (C) 2026 YIQISOFT; SPDX-License-Identifier: Apache-2.0 -->

# Contributing to NereidForge

Thank you for contributing. Keep changes focused, explain the user-facing or operational impact, and include the verification that exercises the changed area.

## Development environment

The backend requires Python 3.14 or newer. The frontend requires Node.js 24 or newer and npm. Docker is needed only for local image builds.

From the repository root, prepare the shared virtual environment and frontend dependencies:

```bash
make dev-build
```

## Validation commands

Use the repository's existing targets rather than inventing project-specific commands:

```bash
make test
make lint
make license-check
make license-report
```

For frontend-only work, the scripts declared in `frontend/package.json` are:

```bash
cd frontend
npm test
npm run lint
npm run build
```

`npm run build` performs the TypeScript check and creates the production Vite bundle. Backend and frontend application images can be built locally with `make docker-build` from the repository root. This target creates local three-tag images and does not publish them.

Run the checks relevant to your change before opening a pull request. If an existing environment or baseline prevents a check from passing, report the exact command and failure instead of claiming success.

## Developer Certificate of Origin

NereidForge uses the Developer Certificate of Origin. Every commit must include a `Signed-off-by` trailer that exactly matches the commit author:

```bash
git commit -s
```

The resulting trailer has this form:

```text
Signed-off-by: Your Name <you@example.com>
```

The DCO workflow checks every non-merge commit in a pull request. Configure your Git author name and email before committing, and keep the author and sign-off identical.

## Source headers and language

New and modified first-party source files must carry the following copyright and SPDX identifiers in the file's native comment syntax:

```text
Copyright (C) 2026 YIQISOFT
SPDX-License-Identifier: Apache-2.0
```

Place the header at the beginning of the file while preserving required prologues such as shebangs, YAML document markers, Markdown front matter, and HTML doctypes. Markdown uses an HTML comment; Helm templates use a non-rendering Go-template comment; shell, YAML, TOML, requirements, and similar text formats use `#`; CSS and HTML use their own comment syntax.

Header coverage exceptions are:

- `LICENSE`, `NOTICE`, and `THIRD_PARTY_NOTICES.md`;
- JSON files, lockfiles, Helm `*.tgz` archives, images, and other binary assets;
- `frontend/public/netron/`, `frontend/public/label-studio/`, `infra/charts/kubeflow/`, `infra/third_party/`, `deploy/pre-nereidforge/platform/`, and `vendor/` directories;
- generated files only while their generator controls the header.

Runtime code, comments, documentation additions, and API response messages must be English. The bilingual files in `frontend/public/help/` are product functionality: preserve their Chinese-English paired content and HelpDrawer behavior. Maintain `frontend/src/locales/en.ts` and `frontend/src/locales/zh.ts` together when changing user-facing text; Chinese in the Chinese locale and language selector is intentional.

## Backend API contracts

When adding, changing, or removing a FastAPI endpoint or schema, update the route's English summary and typed response/error models in the same change. Update `docs/api/openapi.yaml` at the same time so the hand-maintained contract and generated `/openapi.json` describe the same API.

## Third-party licensing and assets

Read `LICENSE-POLICY.md`, `NOTICE`, and `THIRD_PARTY_NOTICES.md` before adding a dependency, base image, trainer, runtime, plugin, or model artifact. The default-supported L1 surface is permissive-only. User-supplied L2 components remain the user's licensing responsibility and must not silently become bundled deliverables.

The denylist and artifact checks run through:

```bash
make license-check
```

Do not commit third-party model weights or bake them into suite images. Default profiles reference weights at runtime; user-supplied artifacts retain their own license declarations.

Do not directly modify vendored third-party assets. Use the upstream project or the repository's established generator and patch process, preserve upstream license texts and notices, and record any integration change outside the vendored files.

## Pull requests

Keep unrelated refactors out of the change. Describe the behavior, tests, documentation, license impact, and any deployment implications. Include exact failure output for checks that could not run, and never push credentials, generated secrets, internal endpoints, or local operator state with the contribution.
