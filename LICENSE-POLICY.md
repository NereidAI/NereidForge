<!-- Copyright (C) 2026 YIQISOFT; SPDX-License-Identifier: Apache-2.0 -->

# NereidForge License Policy

> Status: adopted 2026-10-23. Supersedes the earlier `NO-YOLO-POLICY.md` draft.
> Scope: every deliverable produced from this repository — source code, container
> images, Helm charts, model artifacts, datasets, documentation.

## 1. Position

NereidForge is distributed under **Apache-2.0**. That license defines what *we*
ship, not what our users may run.

Training and inference implementations are pluggable by design: a trainer is a
container image that implements a small contract, and a served model is an
artifact described by a manifest. Because the platform ships no model
implementation of its own, it does not inherit the license of anything a user
chooses to run. Compatibility with third-party frameworks is a feature; support
for them is not.

This yields three tiers.

| Tier | Meaning | License rule | Platform behavior |
|------|---------|--------------|-------------------|
| **L1 — Default support** | Bundled, tested, documented, listed in the default catalog | Permissive only: Apache-2.0, MIT, MIT-0, BSD-2/3, ISC, PSF, Zlib | Official images, profiles, quickstart, UI defaults |
| **L2 — Compatible** | Runs through the same contract, but is neither shipped nor supported by us | Any license. The user owns the compliance obligation | Register and run; declare, record, display. Never blocked at runtime |
| **L3 — Never in deliverables** | Anything copyleft that would enter our repository or images | AGPL, SSPL, BUSL, Commons Clause, or an unknown license | CI blocks the change; no exceptions without an ADR |

The practical consequence: **we do not refuse to run third-party trainers, we
refuse to distribute them.**

## 2. Tier 1 (default support)

A component may be listed as default-supported only if all of the following
hold:

1. Its license is in the permissive set above, and so is every transitive
   dependency installed by our image or imported by our code.
2. It appears in a first-party dependency manifest (`requirements*.txt`,
   `pyproject.toml`, `Dockerfile`, chart values) — never downloaded ad hoc at
   build time from an unpinned source.
3. It ships without third-party model weights. Default profiles reference
   weights by URI at run time; no checkpoint is baked into an image.
4. It is described by capability in user-facing surfaces (see §5).

## 3. Tier 2 (compatibility, user-supplied)

Third-party trainers, inference runtimes, optimizer/evaluator plugins, and model
artifacts are **L2**. The platform:

- accepts any image reference the deployment can pull;
- requires a license declaration (`SPDX identifier` or `Proprietary`) in the
  profile/manifest;
- records an acknowledgement (user + timestamp) when the declared license is
  copyleft — this is an audit trail, not a gate;
- shows the declaration in the UI so downstream users can make their own call;
- does **not** redistribute the image or weights. Mirrors are the operator's
  responsibility and are not part of a release.

Two statements belong in the compatibility documentation:

1. The platform does not evaluate, grant, or transfer license rights for
   third-party components.
2. Re-exporting weights to another format (ONNX, TensorRT, GGUF, ...) does not
   change the license of the upstream model or of the code that produced it.

There is intentionally **no runtime license detection or blocking**. Detection
is unreliable, produces false positives, and would silently break the
compatibility promise. Enforcement lives at the boundary of what we ship.

### 3.1 Custom builds and licence resale (ADR-016)

A customer-specific trainer is an L2 component even when *we* write the code for
it. The delivery model keeps the boundary intact:

- the implementation lives in the **customer's** repository and image, built
  against the Apache-2.0 Trainer SDK (`app/builders/sdk/`);
- the copyleft (or commercial) framework and its weights are installed by the
  customer, under the customer's licence — we neither mirror nor redistribute
  them;
- the customer's `trainer-profile.json` declares the licence, the platform
  records and displays it, and nothing enters this repository or a release.

Acting as a **reseller or agent** for an upstream licence (for example YOLO) is
a commercial arrangement between the customer and the upstream vendor. It does
not grant us redistribution rights, does not change the L2 classification, and
must not be used to route upstream code or weights into our deliverables.

Brand names of customisation targets may appear in commercial material and in
this documentation (they are necessary to describe the engagement); they stay
out of default product surfaces per §5.

## 4. Tier 3 (never shipped)

The following must not exist in this repository, in any image we build, or in
any chart we publish:

| Prohibited | Example shapes |
|------------|----------------|
| Copyleft source or imports | a Python module importing a copyleft package |
| Copyleft dependencies | a copyleft package in a runtime `requirements*.txt` / `pyproject.toml` |
| Copyleft build inputs | a copyleft base image or a build-time download of one |
| Bundled third-party weights | checkpoints committed to git or copied into an image |
| Copyleft license text in first-party files | vendoring a copyleft implementation |

A "compatibility shim" that vendors a copyleft implementation is still a
violation. Compatibility is achieved through the contract, never through code
we copy.

Dev-only tooling (test runners, linters, scanners) is not shipped and may be
copyleft; it must never appear in a runtime image or runtime manifest.

## 5. Naming and positioning

Default surfaces — API identifiers, profile names, catalog entries, quickstart
commands, code defaults — use **capability names**: `pytorch/detection`,
`paddle/detection`, `custom`. They must not advertise or preset a third-party
model family.

Model-family names are allowed where the identity is technically or legally
required:

- compliance and third-party notices;
- interoperability troubleshooting;
- a user-supplied profile's own `displayName`.

Dataset label formats are formats, not code, and may be supported regardless of
which project popularised them. Prefer neutral names (`txt-normalized`) in new
code.

## 6. Vendored third-party components

Unmodified upstream trees (for example `infra/third_party/**` and NF-owned
subcharts under `infra/charts/kubeflow/charts/*`) keep their upstream license and
are never re-licensed. Helm dependency archives fetched at package time are still
part of the packaged deliverable: every dependency and every image it deploys
must satisfy this policy, even when a Helm condition disables it by default.
An operator-supplied copyleft chart may be documented as L2, but must not be a
dependency of an NF-packaged chart. When a tree really must be vendored, it must be:

1. listed in `THIRD_PARTY_NOTICES.md` with their upstream license;
2. accompanied by the upstream license text;
3. deployed as an independent service (process boundary) — never linked into
   first-party code;
4. optional, and disabled by default where the deployment does not need them.

A vendored copyleft component does not taint first-party code, but it does
appear in a customer's license inventory. Treat adding one as a decision that
needs maintainer sign-off, not as a routine dependency bump.

## 7. Enforcement

| Layer | What | Where |
|-------|------|-------|
| Pre-commit | denylisted packages, imports, weights, proprietary fonts, license text in staged files | `infra/scripts/check-license-compliance.sh --staged` |
| CI (push / PR) | same rules across the repository | `.github/workflows/license-compliance.yml` |
| CI (weekly) | installed-dependency license report for the backend | same workflow, `--env` mode |
| Manual / release | image-level SBOM + license scan | `syft` / `trivy` on built images (TODO: wire into `publish-images.yml`) |
| Review | §8 / §9 checklists | pull request review |

The gate deliberately matches **packages, imports and artifacts**, not prose. A
keyword scan over documentation would both miss real violations (a renamed
dependency) and flag legitimate integration docs.

Run locally:

```bash
make license-check          # static scan, whole repo
make license-report         # installed-env report for the unified venv
bash infra/scripts/check-license-compliance.sh --third-party-audit
```

## 8. Adding a dependency or base image

1. Check the license of the package **and** its transitive dependencies.
2. If it is permissive, add it to the owner manifest; if it is copyleft, stop —
   it is L2 at best and must not enter a shipped manifest.
3. Pin or bound the version; never install from an unpinned VCS URL at build
   time.
4. If it is an image base, record the registry and digest in the builder.
5. Run `make license-check` and confirm the CI job passes.

## 9. Adding a trainer, runtime or plugin

1. Build against the Trainer SDK (`nereidforge-trainer`), not against private
   platform imports; publish the image under your own registry.
2. Announce capabilities in `trainer-profile.json` (`nf.trainer-profile/1`) and
   validate it with `nf-trainer validate`; the platform registers that document
   instead of hardcoding a task list (ADR-015).
3. Declare `license.code` and `license.weights` (SPDX or `Proprietary`).
4. Declare accepted dataset formats and produced artifacts; do not assume the
   platform knows your framework.
5. Keep brand names out of platform identifiers; they belong in `displayName`.
6. Do not add the component to the default catalog unless §2 holds — being
   registered as an L2 image is always allowed.

## 10. Exceptions

An exception requires an ADR that states the component, its license, why it is
unavoidable, the distribution model (shipped vs. optional service), and the
exit plan. Maintainer sign-off is required. There are no standing exceptions:
each one is reviewed when the component is upgraded.

## 11. Non-goals

- This policy is not legal advice; it is the engineering rule we enforce.
- It does not restrict what customers run on their own clusters.
- It does not attempt to police user-supplied artifacts beyond the declarations
  described in §3.
