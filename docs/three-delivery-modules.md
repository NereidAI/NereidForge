<!-- Copyright (C) 2026 YIQISOFT; SPDX-License-Identifier: Apache-2.0 -->

# ADR 0001: Establish Three NereidForge Delivery Modules

- Status: Accepted
- Date: 2026-11-24

## Context

NereidForge currently combines infrastructure installation, builder-image
construction, application images, application charts, and their verification in
one checkout and partly shared workflows. This makes a builder change, an
infrastructure fix, and an application change share avoidable build, release,
and recovery risk.

The existing Pre-NereidForge installer and handoff already provide a useful
seam. Builder images already have stable public coordinates and trainer-profile
contracts. Application charts already consume the handoff and published images.
A physical repository split today would nevertheless break source-level
couplings and cross-module tests without improving those contracts first.

## Decision

Establish exactly three delivery Modules, initially in one repository.

### Pre-NereidForge

Its Interface accepts site infrastructure inputs, installs and verifies the
supporting infrastructure, and returns the versioned `pre-nereidforge-handoff/v1`
document. It owns five supporting charts, the pre installer, infrastructure
verification, and canonical infrastructure identities. It does not build or
publish NF application or builder artifacts.

### NereidForge Builders

Its Interface accepts a source revision and an explicit image profile, then
produces verified immutable builder-image references and trainer-profile
artifacts. It owns builder sources, builder inventories, builder verification,
and builder publication evidence. It has no cluster or Secret access and no
resident runtime.

### Application-only NereidForge

Its Interface accepts a verified Pre-NereidForge Handoff, retained application
identities, and a pinned builder-image manifest. It produces and verifies the
backend and frontend images, four application charts, and the suite release
composition. It owns runtime image selection and compatibility mappings. It
does not initialize infrastructure or implicitly build builder images during
installation.

There is no fourth shared-delivery Module. Shared behavior must have one owner
or become a versioned Interface.

## Release composition

The application Module composes suite releases without taking ownership of
builder implementations:

- `full`: 21 builder images plus 2 application images, exactly 23 images;
- `cpu-xpu`: 16 builder images plus 2 application images, exactly 18 images;
- exactly five AMD/NVIDIA trainer images are excluded from `cpu-xpu`;
- every selected immutable image passes inventory, reuse/collision, anonymous
  digest, platform, and source-revision checks before floating aliases move;
- all four NF charts wait for the complete selected image gate;
- formal publication continues to require all 23 images;
- Pre-NereidForge publication remains independent: five charts and its source
  installer, with no NF artifact gate.

While all three Modules share a checkout and revision, existing same-revision
OCI annotations remain valid. After repository extraction, the application must
consume a pinned builder manifest containing references, digests, revisions,
and platforms rather than assuming that builder and application revisions are
identical.

## Migration sequence

1. Preserve and port the authoritative Pre-NereidForge safety fixes.
2. Add builder-owned inventory and pull-request verification, including the
   existing XPU build-context and Torch-pin contracts.
3. Add application-owned delivery verification and an NF-owned narrow OCI
   publication helper; remove NF workflow imports from pre sources.
4. Make pre verification pre-only and move misplaced application/builder
   assertions to their owning suites.
5. Replace the backend's builder-source profile reads with a pinned, versioned
   trainer-profile bundle.
6. Move production sources only after their Interfaces and consumer tests pass.
7. Consider separate repositories only after each Module can be tested and
   released without another Module's source tree.

Legacy combined installers remain compatibility adapters during migration; they
are not a fourth Module and are not the target installation Interface.

## Consequences

- Pull requests gain an owning verification gate instead of relying on manual
  monolithic publication to detect builder regressions.
- Infrastructure, builder, and application failures can be repaired and
  published incrementally without changing public artifact coordinates.
- The first increment duplicates a small number of compatibility assertions
  while tests move to their owners. This is temporary and safer than deleting
  the aggregate gate before the replacements pass.
- Repository extraction is delayed until source-level imports and test
  dependencies are removed. The Module decision does not require three
  repositories immediately.
- No publication trigger, image name, chart name, installation handoff, or
  canonical Secret ownership changes in the first increment.
