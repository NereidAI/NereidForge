<!-- Copyright (C) 2026 YIQISOFT; SPDX-License-Identifier: Apache-2.0 -->

# NereidForge Domain Glossary

## Delivery Modules

### Pre-NereidForge

The infrastructure lifecycle Module. It installs and verifies the supporting
infrastructure required by NereidForge applications, owns the canonical
infrastructure identities, and produces the Pre-NereidForge Handoff after its
own acceptance gates pass.

### NereidForge Builders

The build-time Module. It produces and verifies versioned builder images and
trainer profiles for an explicit image profile. It has no resident runtime and
does not install infrastructure or applications.

### Application-only NereidForge

The application runtime Module. It consumes a verified Pre-NereidForge Handoff,
retained application identities, and published builder artifacts. It neither
initializes Pre-NereidForge infrastructure nor builds builder images as an
implicit installation side effect.

## Contracts

### Pre-NereidForge Handoff

The versioned, non-secret interface from Pre-NereidForge to Application-only
NereidForge. It identifies endpoints, Kubernetes Secret references, and public
configuration. Referenced canonical Secrets and credentials remain outside the
handoff.

### Image Profile

A named, explicit inventory of images that must be built and verified together.
`backend-frontend` contains only the two application images and publishes no
charts; `full` and `cpu-xpu` are complete suite publication profiles with their
four charts. One profile must not silently substitute for another, and the
application-only profile does not satisfy a Suite Release gate.

### Builder Manifest

The versioned Builder output consumed by Application-only NereidForge. It
identifies image references, digests, source revisions, and platforms. After
repository extraction, builder and application source revisions need not be the
same.

### Suite Release

The Application-owned composition of its four charts and two images with one
verified Builder Manifest. Composition does not transfer ownership of Builder
implementations or permit a partial Image Profile to satisfy the release gate.

### Canonical Identity

The authoritative retained credential or Secret for an existing installation.
Recovery must preserve it; a missing local copy or incomplete journal is not
permission to generate a replacement.
