// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

// Verify duration fallback semantics independently of React rendering.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';
import ts from 'typescript';

const entryPoint = fileURLToPath(new URL('../src/utils/training.ts', import.meta.url));

/** loadTrainingUtils transpiles the pure training helpers for Node. */
const loadTrainingUtils = () => {
  const { outputText } = ts.transpileModule(readFileSync(entryPoint, 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
  });
  const module = { exports: {} };
  const evaluate = new Function('module', 'exports', 'require', outputText);
  evaluate(module, module.exports, (id) => {
    throw new Error(`unexpected runtime import: ${id}`);
  });
  return module.exports;
};

const runtime = {
  pods: [{
    started: '2026-11-02T09:16:34+00:00',
    finished: '2026-11-02T10:14:19+00:00',
  }],
};

test('trainer-reported duration remains authoritative', () => {
  const { trainingDurationSeconds } = loadTrainingUtils();
  assert.equal(trainingDurationSeconds({ train_time_s: 12.5 }, runtime), 12.5);
});

test('terminated trainer containers recover the helmet-retrain duration', () => {
  const { trainingDurationSeconds } = loadTrainingUtils();
  assert.equal(trainingDurationSeconds({}, runtime), 3465);
});

test('distributed duration spans the earliest start to the latest finish', () => {
  const { trainingDurationSeconds } = loadTrainingUtils();
  const distributed = {
    pods: [
      runtime.pods[0],
      { started: '2026-11-02T09:17:00+00:00', finished: '2026-11-02T10:15:00+00:00' },
    ],
  };
  assert.equal(trainingDurationSeconds(null, distributed), 3506);
});

test('running or incomplete workloads do not fabricate a duration', () => {
  const { trainingDurationSeconds } = loadTrainingUtils();
  assert.equal(trainingDurationSeconds({}, { pods: [{ started: runtime.pods[0].started }] }), 0);
});
