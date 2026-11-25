<!-- Copyright (C) 2026 YIQISOFT; SPDX-License-Identifier: Apache-2.0 -->

# ML Backend (Pre-annotation Service)

## Purpose
An ML Backend is a model service for Label Studio that powers **semi-automatic annotation**: it generates pre-annotation boxes for unlabeled tasks, which annotators then correct manually.

## Flow
1. Register an ML Backend: record its name, URL, and optionally a linked model.
2. In the annotation workbench, select the ML Backend (or the default service) from the dropdown next to the Semi-automatic annotation button.
3. Pre-annotations are written into Label Studio as predictions and are visible in the workbench.

## Prerequisites
- The ML Backend service must be reachable (inside the cluster or exposed externally).
- When a model is linked, its pre-annotations must match that model's class space.

## FAQ
- **Why do the pre-annotation labels not match?** The model's classes and the annotation configuration must be consistent.
- **Why is the service unhealthy?** The health check is a real HTTP probe against `/health`.

## Next Steps
Register a model, select the service in the annotation workbench, run semi-automatic annotation, review manually, then export for training.
