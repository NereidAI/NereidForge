// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

// Verify task-specific trial configuration and canvas rendering at a pure seam.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';
import ts from 'typescript';

const entryPoint = fileURLToPath(new URL('../src/utils/modelTrial.ts', import.meta.url));

/** loadTrialUtils compiles the dependency-free trial utilities for Node tests. */
const loadTrialUtils = () => {
  const { outputText } = ts.transpileModule(readFileSync(entryPoint, 'utf8'), {
    compilerOptions: {
      module: ts.ModuleKind.CommonJS,
      target: ts.ScriptTarget.ES2020,
      esModuleInterop: true,
    },
  });
  const module = { exports: {} };
  const evaluate = new Function('module', 'exports', 'require', outputText);
  evaluate(module, module.exports, (id) => { throw new Error(`unexpected import: ${id}`); });
  return module.exports;
};

const schema = {
  artifact: 's3://models/face/v1/model.onnx',
  inputs: [{ name: 'images', shape: [1, 3, 640, 640], dtype: 'tensor(float)' }],
  outputs: [{ name: 'output0', shape: [1, 6, 2], dtype: 'tensor(float)' }],
  decoders: [],
  auto_decoder: null,
  mapping_required: true,
  names: { 0: 'face' },
  saved_decoder: null,
};

/** fakeCanvas records drawing operations without introducing a DOM test runner. */
const fakeCanvas = () => {
  const calls = [];
  const context = {
    fillStyle: '', strokeStyle: '', lineWidth: 1, font: '', textBaseline: '',
    clearRect: (...args) => calls.push(['clearRect', ...args]),
    drawImage: (...args) => calls.push(['drawImage', ...args]),
    beginPath: () => calls.push(['beginPath']),
    moveTo: (...args) => calls.push(['moveTo', ...args]),
    lineTo: (...args) => calls.push(['lineTo', ...args]),
    closePath: () => calls.push(['closePath']),
    fill: () => calls.push(['fill']),
    stroke: () => calls.push(['stroke']),
    strokeRect: (...args) => calls.push(['strokeRect', ...args]),
    fillRect: (...args) => calls.push(['fillRect', ...args]),
    fillText: (...args) => calls.push(['fillText', ...args]),
  };
  return { canvas: { width: 0, height: 0, getContext: () => context }, calls };
};

test('custom defaults use input metadata but require explicit output semantics', () => {
  const { createCustomDecoderConfig, trialDecoderError } = loadTrialUtils();
  const config = createCustomDecoderConfig(schema, 'detection');

  assert.deepEqual(config.input, {
    tensor: 'images', width: 640, height: 640, layout: 'NCHW', color: 'RGB',
    resize: 'letterbox', scale: 1 / 255, mean: [0, 0, 0], std: [1, 1, 1],
  });
  assert.equal(config.output.prediction, '');
  assert.equal(config.output.layout, 'rows');
  assert.equal(trialDecoderError(config, schema), 'output mapping required');
  const mapped = { ...config, output: { ...config.output, prediction: 'output0' } };
  assert.equal(trialDecoderError(mapped, schema), null);
  assert.equal(
    trialDecoderError({ mode: 'auto' }, schema),
    'decoder mapping required',
  );

  const dynamic = createCustomDecoderConfig({
    ...schema,
    inputs: [{ name: 'images', shape: [1, 3, 'height', 'width'], dtype: 'tensor(float)' }],
    input_shape: [1, 3, 512, 768],
  }, 'detection');
  assert.equal(dynamic.input.width, 768);
  assert.equal(dynamic.input.height, 512);

  const chwHint = createCustomDecoderConfig({
    ...schema,
    inputs: [{ name: 'images', shape: [1, 3, 'height', 'width'], dtype: 'tensor(float)' }],
    input_shape: [3, 512, 768],
  }, 'detection');
  assert.equal(chwHint.input.width, 768);
  assert.equal(chwHint.input.height, 512);

  const instance = createCustomDecoderConfig(schema, 'instance_segmentation');
  assert.equal(trialDecoderError(instance, schema), 'output mapping required');
});

test('semantic output patches omit NHW axes and retain explicit N1HW axes', () => {
  const { patchTrialOutput } = loadTrialUtils();
  const logits = { output: 'mask', format: 'logits', class_axis: 1 };
  const classIds = patchTrialOutput(logits, { format: 'class_ids', class_axis: undefined });

  assert.deepEqual(classIds, { output: 'mask', format: 'class_ids' });
  assert.deepEqual(
    patchTrialOutput(classIds, { class_axis: 1 }),
    { output: 'mask', format: 'class_ids', class_axis: 1 },
  );
  assert.deepEqual(
    patchTrialOutput(classIds, { format: 'logits', class_axis: 1 }),
    logits,
  );
});

test('trial errors prefer structured guidance over object-valued details', () => {
  const { trialErrorMessage } = loadTrialUtils();
  const nested = {
    response: { data: { detail: { error: 'decoder mapping required', code: 'decoder_mapping_required' } } },
  };
  assert.equal(trialErrorMessage(nested, 'fallback'), 'decoder mapping required');
  assert.equal(trialErrorMessage({ response: { data: { detail: 'plain detail' } } }, 'fallback'), 'plain detail');
  assert.equal(trialErrorMessage({ response: { data: { error: 'backend error' } } }, 'fallback'), 'backend error');
  assert.equal(trialErrorMessage(new Error('network down'), 'fallback'), 'network down');
});

test('canvas draws detection boxes and filled instance polygons in image coordinates', () => {
  const { drawTrialOverlay } = loadTrialUtils();
  const { canvas, calls } = fakeCanvas();
  drawTrialOverlay(canvas, {
    task: 'segment', kind: 'instance_segmentation', width: 200, height: 100,
    names: {}, boxes: [{ bbox: [10, 20, 60, 80], score: 0.9, cls: 1, label: 'person' }],
    polygons: [{ score: 0.9, cls: 1, label: 'person', points: [[10, 20], [60, 20], [60, 80]] }],
  });

  assert.equal(canvas.width, 200);
  assert.equal(canvas.height, 100);
  assert.ok(calls.some((call) => call[0] === 'strokeRect' && call.slice(1).join(',') === '10,20,50,60'));
  assert.ok(calls.some((call) => call[0] === 'lineTo' && call[1] === 60 && call[2] === 80));
  assert.ok(calls.some((call) => call[0] === 'fill'));
  assert.ok(calls.some((call) => call[0] === 'closePath'));
});

test('canvas composites semantic overlay and classification uses one ordered Top-K list', () => {
  const { drawTrialOverlay, classificationTopK } = loadTrialUtils();
  const { canvas, calls } = fakeCanvas();
  const overlay = { id: 'rgba-mask' };
  drawTrialOverlay(canvas, {
    task: 'segment', kind: 'semantic_segmentation', width: 320, height: 160,
    names: {}, mask_overlay: 'base64-data',
  }, overlay);
  assert.ok(calls.some((call) => call[0] === 'drawImage' && call[1] === overlay &&
    call.slice(2).join(',') === '0,0,320,160'));

  const unknown = fakeCanvas();
  drawTrialOverlay(unknown.canvas, {
    task: 'tracking', kind: 'tracking', width: 100, height: 100, names: {},
    boxes: [{ bbox: [0, 0, 10, 10], score: 1, cls: 0, label: 'not-a-detection' }],
  });
  assert.equal(unknown.calls.some((call) => call[0] === 'strokeRect'), false);

  assert.deepEqual(classificationTopK({
    probs: {
      class_index: 1, label: 'cat', score: 0.8,
      topk: [{ class_index: 1, label: 'cat', score: 0.8 }, { class_index: 0, label: 'dog', score: 0.2 }],
      top5: [['cat', 0.8], ['dog', 0.2]],
    },
  }), [
    { class_index: 1, label: 'cat', score: 0.8 },
    { class_index: 0, label: 'dog', score: 0.2 },
  ]);
});
