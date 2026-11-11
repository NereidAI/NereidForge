// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

/**
 * Single source of truth for self-built NereidForge image refs on the UI.
 *
 * All refs are public coordinates (root treaty §18): self-built images always
 * resolve to flat ghcr.io/nereidai/<category>-<name> names. Air-gapped
 * clusters resolve the same refs through the node-level containerd registry
 * mirror; the bundle carries no registry/layout switch.
 *
 * Every self-built image carries the tag-sourced release version (root
 * treaty §19.5): there is no VERSION file, the production Dockerfile bakes
 * the git tag's bare SemVer as VITE_IMAGE_TAG, and non-tag builds fall back
 * to the rolling 0.0.0-dev dev channel, so UI-selected values can never drift
 * from the backend catalog. The literal fallback also covers local `vite dev`.
 *
 * Exceptions (NOT routed through the release tag): jupyter-base keeps its
 * Python-version tag (py3.12, rebuild-in-place dual-use workbench image) and
 * mirrored upstream images keep their upstream tags.
 */

const REGISTRY = 'ghcr.io/nereidai';

const FLAT_ALIASES: Record<string, string> = {
  'ml-backend/ml-backend': 'ml-backend',
  'ml-backend/ml-backend-custom': 'ml-backend-custom',
  runtime: 'nf-runtime',
};

export const IMAGE_TAG: string = import.meta.env.VITE_IMAGE_TAG || '0.0.0-dev';

export const nfImage = (repo: string, tag?: string): string => {
  const resolvedTag = tag || IMAGE_TAG;
  return `${REGISTRY}/${FLAT_ALIASES[repo] ?? repo.replace(/\//g, '-')}:${resolvedTag}`;
};

/**
 * Fallback trainer-image dropdown options shown before the catalog loads.
 * Vision trainers are per-runtime-family and per-vendor (ADR-014 D-01):
 * `torch-*` covers every vision task, `paddle-*` (PaddleX) is the second
 * framework. Paddle ships cpu + cu only: no ROCm wheel and zero xpu
 * paddlepaddle builds exist on PyPI (corrected 2026-10).
 */
export const TRAINER_IMAGE_FALLBACKS: { value: string; label: string }[] = [
  { value: nfImage('trainer/torch-cpu'), label: 'Torch Vision · CPU (torchvision + timm)' },
  { value: nfImage('trainer/torch-cu'), label: 'Torch Vision · NVIDIA CUDA' },
  { value: nfImage('trainer/torch-rocm'), label: 'Torch Vision · AMD ROCm' },
  { value: nfImage('trainer/torch-xpu'), label: 'Torch Vision · Intel XPU' },
  { value: nfImage('trainer/paddle-cpu'), label: 'Paddle Vision · CPU (PaddleX detection)' },
  { value: nfImage('trainer/paddle-cu'), label: 'Paddle Vision · NVIDIA CUDA (PaddleX)' },
  { value: nfImage('trainer/llm-cu'), label: 'LLM / VLM · NVIDIA (LLaMA-Factory)' },
  { value: nfImage('trainer/llm-rocm'), label: 'LLM / VLM · AMD (LLaMA-Factory)' },
  { value: nfImage('trainer/llm-xpu'), label: 'LLM / VLM · Intel XPU (transformers QLoRA)' },
];

const withoutTag = (image: string): string => image.replace(/:[^:/]+$/, '');

// Recognized built-in trainer image base names (the segment after the
// category), grouped by runtime family. Canonical public refs use
// "/trainer-<base>"; legacy nested and prefixed-flat forms remain recognizable
// so an old saved job normalizes instead of dropping. Legacy cv/timm bases map
// onto the supported torch family.
const TRAINER_BASE_RE =
  /(?:\/trainer\/|\/trainer-|-trainer-)(torch(?:-cpu|-cu|-rocm|-xpu)?|paddle(?:-cpu|-cu|-xpu)?|cv(?:-cu|-rocm|-xpu)?|timm|llm(?:-cu|-rocm|-xpu)?)$/;

const VISION_TASKS = new Set([
  'classification', 'multilabel', 'detection', 'segmentation',
  'instance_seg', 'semantic_seg', 'keypoint', 'pose',
]);

const LLM_TASKS = new Set(['llm', 'vlm']);

export const compatibleTrainerImage = (
  image: string | undefined,
  taskType: string,
  accelerator: string,
): string => {
  const selected = (image || '').trim();
  if (!selected) return '';
  const repo = withoutTag(selected);
  const match = repo.match(TRAINER_BASE_RE);
  if (!match) return selected;
  const base = match[1];

  const vendorToken = accelerator === 'nvidia'
    ? '-cu'
    : accelerator === 'amd'
      ? '-rocm'
      : accelerator === 'intel'
        ? '-xpu'
        : '';

  let wanted = '';
  if (LLM_TASKS.has(taskType)) {
    // CPU has no LLM image: keep the family only when the vendor matches.
    if (vendorToken && base === `llm${vendorToken}`) wanted = base;
  } else if (VISION_TASKS.has(taskType)) {
    // Same runtime family the user picked (paddle stays paddle); legacy
    // cv/timm bases are normalized onto the torch family. Paddle ships no
    // ROCm/XPU wheels, so those vendors fall back to the auto image ('').
    const family = base.startsWith('paddle') ? 'paddle' : 'torch';
    if (!(family === 'paddle' && (accelerator === 'amd' || accelerator === 'intel'))) {
      wanted = `${family}${vendorToken || '-cpu'}`;
    }
  }
  if (!wanted) return '';

  // Every recognized built-in trainer returns through the sole public resolver,
  // including legacy nested or non-GHCR refs; never preserve their old registry.
  return nfImage(`trainer/${wanted}`);
};
