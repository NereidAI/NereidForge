<!-- Copyright (C) 2026 YIQISOFT; SPDX-License-Identifier: Apache-2.0 -->

# Pre-NereidForge deployment assets

This tree groups the infrastructure and Kubeflow deployment stage:

- `charts/` contains the supporting-service Helm charts.
- `platform/` contains the vendored Kubeflow manifests and NF patch set.
- `scripts/` contains infrastructure, Kubeflow, fetch/cache, and related verification scripts.
- `values/` contains pre-NereidForge chart value overlays.

The full-stack compatibility entry points and combined offline tooling remain
under the repository-root `scripts/`. Pre-NereidForge scripts temporarily use
the root shared script libraries. Self-contained packaging and handoff are the
next restructuring stage; this tree is not yet an independently releasable
package.
