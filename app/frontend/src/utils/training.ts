// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

/**
 * Vision architecture defaults shared by the training form.
 *
 * Mirrors the trainer catalog (`app/builders/trainer-torch/models.py`) and the
 * backend's `JobService.DEFAULT_ARCHS` — one task vocabulary, one default per
 * task. A bare architecture id means "train from the framework's permissive
 * upstream backbone" (`pretrained`), not a vendored checkpoint: NF ships no
 * third-party weights (ADR-014 D5).
 */
import type { TrainingRuntime } from '@nftypes/api';

export const DEFAULT_ARCHS: Record<string, string> = {
  classification: 'resnet50',
  multilabel: 'resnet50',
  detection: 'fasterrcnn_mobilenet_v3_large_fpn',
  segmentation: 'maskrcnn_resnet50_fpn_v2',
  instance_seg: 'maskrcnn_resnet50_fpn_v2',
  semantic_seg: 'deeplabv3_mobilenet_v3_large',
  keypoint: 'keypointrcnn_resnet50_fpn',
  pose: 'keypointrcnn_resnet50_fpn',
};

/** Default architecture id for a UI task type; detection when unknown. */
export const defaultArchFor = (taskType: string): string =>
  DEFAULT_ARCHS[taskType] || DEFAULT_ARCHS.detection;

/** trainingDurationSeconds prefers trainer metrics, then terminated workload bounds. */
export const trainingDurationSeconds = (
  metrics: Record<string, unknown> | null | undefined,
  runtime: TrainingRuntime | null | undefined,
): number => {
  const reported = Number(metrics?.train_time_s);
  if (Number.isFinite(reported) && reported > 0) return reported;

  const pods = runtime?.pods ?? [];
  if (!pods.length || pods.some((pod) => !pod.started || !pod.finished)) return 0;
  const starts = pods.map((pod) => Date.parse(pod.started as string));
  const finishes = pods.map((pod) => Date.parse(pod.finished as string));
  if (![...starts, ...finishes].every(Number.isFinite)) return 0;
  const seconds = (Math.max(...finishes) - Math.min(...starts)) / 1000;
  return seconds > 0 ? seconds : 0;
};

/** Initialization choices exposed by the vision training form. */
export type InitializationMode = 'pretrained' | 'scratch' | 'checkpoint';

/** Normalized architecture and initialization selections for create/clone/detail views. */
export interface TrainingSelection {
  architecture: string;
  initialization: InitializationMode;
  startingWeights: string;
}

const CHECKPOINT_SUFFIXES = ['.pt', '.pth', '.ckpt', '.safetensors'];

/**
 * Return whether a legacy base_model value is a checkpoint rather than an
 * architecture id. This mirrors the backend compatibility rule; new jobs use
 * explicit architecture/base_weights_uri fields and do not rely on this test.
 */
export const isCheckpointReference = (value: string): boolean => {
  const candidate = value.trim().toLowerCase();
  return candidate.includes('/')
    || candidate.startsWith('s3:')
    || CHECKPOINT_SUFFIXES.some((suffix) => candidate.endsWith(suffix));
};

/**
 * Resolve current and legacy job config into the three independent concepts
 * shown by the UI. A checkpoint may intentionally leave architecture empty:
 * the backend then uses registered-version metadata, a recognized stock name,
 * or the task default in that order.
 */
export const trainingSelectionFromConfig = (
  config: Record<string, unknown>,
  taskType: string,
): TrainingSelection => {
  const explicitArchitecture = String(config.architecture || '').trim();
  const legacyValue = String(config.base_model || config.model || '').trim();
  const explicitWeights = String(config.base_weights_uri || '').trim();
  const legacyWeights = !explicitWeights && isCheckpointReference(legacyValue)
    ? legacyValue
    : '';
  const startingWeights = explicitWeights || legacyWeights;
  const initialization: InitializationMode = startingWeights
    ? 'checkpoint'
    : config.pretrained === false
      ? 'scratch'
      : 'pretrained';
  const legacyArchitecture = legacyValue && !isCheckpointReference(legacyValue)
    ? legacyValue
    : '';
  return {
    architecture: explicitArchitecture
      || legacyArchitecture
      || (startingWeights ? '' : defaultArchFor(taskType)),
    initialization,
    startingWeights,
  };
};
