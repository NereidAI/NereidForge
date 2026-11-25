// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

// Model-trial configuration and canvas rendering. The utilities deliberately
// require explicit output semantics; input dimensions only provide editable defaults.
import type {
  TrialDecoderConfig,
  TrialResultKind,
  TrialSchema,
  TrialTask,
  TryResult,
} from '@nftypes/api';

const PALETTE = ['#ff4d4f', '#1677ff', '#52c41a', '#fa8c16', '#722ed1', '#13c2c2'];

/** trialErrorMessage extracts readable API guidance without passing objects into React content. */
export const trialErrorMessage = (error: any, fallback: string): string => {
  const data = error?.response?.data;
  const detail = data?.detail;
  if (typeof detail === 'string' && detail) return detail;
  if (typeof detail?.error === 'string' && detail.error) return detail.error;
  if (Array.isArray(detail)) {
    const messages = detail.map((item) => item?.msg).filter(Boolean);
    if (messages.length) return messages.join('; ');
  }
  if (typeof data?.error === 'string' && data.error) return data.error;
  if (typeof error?.message === 'string' && error.message) return error.message;
  return fallback;
};

/** trialResultKind resolves legacy response names without treating unknown tasks as detection. */
export const trialResultKind = (result: TryResult): TrialResultKind => {
  if (result.kind) return result.kind;
  if (result.task === 'classify') return 'classification';
  if (result.task === 'segment') return result.mask_overlay ? 'semantic_segmentation' : 'instance_segmentation';
  if (result.task === 'detect') return 'detection';
  return result.task || 'unknown';
};

/** trialColor gives one stable class color across boxes, polygons and legends. */
export const trialColor = (classIndex: number, alpha = 1): string => {
  const hex = PALETTE[Math.abs(classIndex) % PALETTE.length];
  const channels = [1, 3, 5].map((offset) => parseInt(hex.slice(offset, offset + 2), 16));
  return `rgba(${channels[0]}, ${channels[1]}, ${channels[2]}, ${alpha})`;
};

/** createCustomDecoderConfig creates editable defaults from graph metadata, never an automatic output decoder. */
export const createCustomDecoderConfig = (schema: TrialSchema, task: TrialTask): TrialDecoderConfig => {
  const input = schema.inputs[0];
  const rawHint = schema.input_shape || [];
  const hinted = rawHint.length === 3 ? [1, ...rawHint] : rawHint;
  const shape = input?.shape?.length === 4 ? input.shape : hinted;
  const nchw = shape?.length === 4 && shape[1] === 3;
  const nhwc = shape?.length === 4 && shape[3] === 3;
  const numberDim = (value: number | string | null | undefined, fallback: number) =>
    typeof value === 'number' && value > 0 ? value : fallback;
  const hintedDim = (index: number) => numberDim(hinted[index], 640);
  const width = nchw
    ? numberDim(shape[3], hintedDim(3))
    : nhwc ? numberDim(shape[2], hintedDim(2)) : hintedDim(3);
  const height = nchw
    ? numberDim(shape[2], hintedDim(2))
    : nhwc ? numberDim(shape[1], hintedDim(1)) : hintedDim(2);
  const tensor = (name: string) => schema.outputs.find((item) => item.name === name)?.name || '';
  let output: Record<string, unknown>;
  if (task === 'detection') {
    output = {
      prediction: '',
      layout: 'rows', bbox_start: 0, bbox_encoding: 'xyxy', score_index: 4,
      class_index: 5, coordinate_space: 'pixels', score_activation: 'none',
    };
  } else if (task === 'instance_segmentation') {
    output = {
      boxes: tensor('boxes'), scores: tensor('scores'), labels: tensor('labels'), masks: tensor('masks'),
      bbox_encoding: 'xyxy', coordinate_space: 'pixels', mask_activation: 'sigmoid', mask_threshold: 0.5,
    };
  } else if (task === 'semantic_segmentation') {
    output = { output: tensor('logits'), format: 'logits', class_axis: 1, ignore_index: -1 };
  } else {
    output = { output: '', class_axis: 1, activation: 'softmax' };
  }
  const labels = Object.entries(schema.names || {})
    .sort(([a], [b]) => Number(a) - Number(b))
    .map(([, label]) => label);
  return {
    mode: 'custom',
    task,
    input: {
      tensor: input?.name || '', width, height, layout: nhwc ? 'NHWC' : 'NCHW',
      color: 'RGB', resize: 'letterbox', scale: 1 / 255,
      mean: [0, 0, 0], std: [1, 1, 1],
    },
    output: output as unknown as TrialDecoderConfig['output'],
    labels,
  };
};

/** patchTrialOutput merges fields and removes optional keys set to undefined before serialization. */
export const patchTrialOutput = (
  output: Record<string, unknown>,
  patch: Record<string, unknown>,
): Record<string, unknown> => {
  const next = { ...output, ...patch };
  Object.keys(next).forEach((key) => {
    if (next[key] === undefined) delete next[key];
  });
  return next;
};

/** trialDecoderError returns a stable validation reason before a trial can mutate saved configuration. */
export const trialDecoderError = (config: TrialDecoderConfig, schema: TrialSchema): string | null => {
  if (config.mode === 'auto') return schema.auto_decoder ? null : 'decoder mapping required';
  if (config.mode === 'builtin') {
    return config.decoder_id && schema.decoders.some((item) => item.id === config.decoder_id)
      ? null
      : 'builtin decoder required';
  }
  const input = config.input;
  if (!input?.tensor || !(input.width > 0) || !(input.height > 0)) return 'input mapping required';
  if (!config.output) return 'output mapping required';
  const output = config.output as unknown as Record<string, unknown>;
  const required = config.task === 'detection'
    ? ['prediction']
    : config.task === 'instance_segmentation'
      ? ['boxes', 'scores', 'labels', 'masks']
      : ['output'];
  return required.every((key) => typeof output[key] === 'string' && output[key]) ? null : 'output mapping required';
};

/** classificationTopK normalizes current and legacy probability payloads without repeating Top-1. */
export const classificationTopK = (result: TryResult) => {
  const probs = result.probs;
  if (!probs) return [] as Array<{ class_index: number; label: string; score: number }>;
  if (probs.topk) return probs.topk;
  return probs.top5.map(([label, score], index) => {
    const named = Object.entries(result.names || {}).find(([, value]) => value === label);
    return { class_index: named ? Number(named[0]) : index, label, score };
  });
};

/** drawTrialOverlay renders original-image coordinates onto a canvas scaled by CSS. */
export const drawTrialOverlay = (
  canvas: HTMLCanvasElement,
  result: TryResult,
  maskImage?: CanvasImageSource | null,
): void => {
  const width = Math.max(1, Math.round(result.width || 1));
  const height = Math.max(1, Math.round(result.height || 1));
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext('2d');
  if (!context) return;
  context.clearRect(0, 0, width, height);
  const kind = trialResultKind(result);
  if (maskImage && kind === 'semantic_segmentation') {
    context.drawImage(maskImage, 0, 0, width, height);
  }

  const polygons = kind === 'instance_segmentation' ? result.polygons || [] : [];
  for (const polygon of polygons) {
    if (polygon.points.length < 3) continue;
    context.beginPath();
    context.moveTo(polygon.points[0][0], polygon.points[0][1]);
    polygon.points.slice(1).forEach(([x, y]) => context.lineTo(x, y));
    context.closePath();
    context.fillStyle = trialColor(polygon.cls, 0.25);
    context.fill();
    context.strokeStyle = trialColor(polygon.cls);
    context.lineWidth = Math.max(2, width / 500);
    context.stroke();
  }

  const boxes = kind === 'detection' || kind === 'instance_segmentation' ? result.boxes || [] : [];
  for (const box of boxes) {
    const [x1, y1, x2, y2] = box.bbox;
    const color = trialColor(box.cls);
    const lineWidth = Math.max(2, width / 500);
    context.strokeStyle = color;
    context.lineWidth = lineWidth;
    context.strokeRect(x1, y1, x2 - x1, y2 - y1);
    const caption = `${box.label} ${box.score.toFixed(2)}`;
    const captionHeight = Math.max(16, width / 45);
    const captionWidth = Math.max(36, caption.length * captionHeight * 0.42);
    const captionY = Math.max(0, y1 - captionHeight);
    context.fillStyle = color;
    context.fillRect(x1, captionY, captionWidth, captionHeight);
    context.fillStyle = '#fff';
    context.font = `${Math.max(11, captionHeight * 0.7)}px sans-serif`;
    context.textBaseline = 'middle';
    context.fillText(caption, x1 + 4, captionY + captionHeight / 2);
  }
};
