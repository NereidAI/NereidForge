// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

// Verify browser API routes against the backend's public /api/v1 contract.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { File } from 'node:buffer';
import ts from 'typescript';

const entryPoint = fileURLToPath(new URL('../src/api/endpoints.ts', import.meta.url));

/** loadEndpoints compiles endpoint definitions with a recording client. */
const loadEndpoints = (calls) => {
  const record = (method) => (url, ...args) => {
    calls.push({ method, url, args });
    return Promise.resolve({ data: {} });
  };
  const apiClient = {
    get: record('get'), post: record('post'), put: record('put'),
    patch: record('patch'), delete: record('delete'),
  };
  const { outputText } = ts.transpileModule(readFileSync(entryPoint, 'utf8'), {
    compilerOptions: {
      module: ts.ModuleKind.CommonJS,
      target: ts.ScriptTarget.ES2020,
      esModuleInterop: true,
    },
  });
  const module = { exports: {} };
  const require = (id) => {
    if (id === './client') return { __esModule: true, default: apiClient };
    throw new Error(`unexpected runtime import: ${id}`);
  };
  const evaluate = new Function('module', 'exports', 'require', outputText);
  evaluate(module, module.exports, require);
  return module.exports;
};

test('optimization detail uses the registered live-logs route', async () => {
  const calls = [];
  const { optimizationApi } = loadEndpoints(calls);

  await optimizationApi.getLogs('job-opt', { tail: 2000 });

  assert.deepEqual(calls, [{
    method: 'get',
    url: '/optimization/jobs/job-opt/logs',
    args: [{ params: { tail: 2000 } }],
  }]);
});

test('model trial carries artifact inspection and decoder configuration', async () => {
  const calls = [];
  const { modelApi } = loadEndpoints(calls);

  await modelApi.getTrySchema('face', 'v1', 'nested/yunet.onnx');
  const decoderConfig = {
    mode: 'builtin',
    decoder_id: 'yunet',
  };
  const image = new File(['image'], 'lena.jpg', { type: 'image/jpeg' });
  await modelApi.tryVersion('face', 'v1', {
    image,
    conf: 0.4,
    artifactPath: 'nested/yunet.onnx',
    decoderConfig,
    topK: 3,
  });

  assert.equal(calls[0].method, 'get');
  assert.equal(calls[0].url, '/models/face/versions/v1/try/schema?path=nested%2Fyunet.onnx');
  assert.equal(calls[0].args[0].timeout, 200000);
  assert.equal(calls[1].method, 'post');
  assert.equal(calls[1].url, '/models/face/versions/v1/try?path=nested%2Fyunet.onnx');
  const [body, config] = calls[1].args;
  assert.equal(body.get('image'), image);
  assert.equal(body.get('conf'), '0.4');
  assert.equal(body.get('decoder_config'), JSON.stringify(decoderConfig));
  assert.equal(body.get('top_k'), '3');
  assert.equal(config.timeout, 200000);
});

test('model version upload preserves every artifact file and discovers formats', async () => {
  const calls = [];
  const { modelApi } = loadEndpoints(calls);
  const xml = new File(['graph'], 'model.xml');
  const bin = new File(['weights'], 'model.bin');
  const metadata = {
    primary_file: 'model.xml',
    runtime_requirements: { openvino: '2026.1' },
  };

  await modelApi.getArtifactFormats();
  await modelApi.createVersion('face', {
    version: 'v2', description: 'IR', files: [xml, bin], metadata,
  });

  assert.deepEqual(
    calls.map(({ method, url }) => ({ method, url })),
    [
      { method: 'get', url: '/models/artifact-formats' },
      { method: 'post', url: '/models/face/versions' },
    ],
  );
  const [body, config] = calls[1].args;
  assert.deepEqual(body.getAll('file'), [xml, bin]);
  assert.equal(body.get('version'), 'v2');
  assert.equal(body.get('description'), 'IR');
  assert.equal(body.get('metadata'), JSON.stringify(metadata));
  assert.equal(config.timeout, 1800000);
});

test('model and version deletion use the super-admin-protected registry routes', async () => {
  const calls = [];
  const { modelApi } = loadEndpoints(calls);

  await modelApi.deleteVersion('face', '2021dec');
  await modelApi.delete('face');

  assert.deepEqual(calls, [
    { method: 'delete', url: '/models/face/versions/2021dec', args: [] },
    { method: 'delete', url: '/models/face', args: [] },
  ]);
});

test('FP32 edge packages use the read-only list and byte-download routes', async () => {
  const calls = [];
  const { modelApi } = loadEndpoints(calls);

  await modelApi.listPackages('face', '2021dec');
  await modelApi.downloadPackage('face', '2021dec', 'fp32');

  assert.deepEqual(calls, [
    { method: 'get', url: '/models/face/versions/2021dec/packages', args: [] },
    {
      method: 'get',
      url: '/models/face/versions/2021dec/package',
      args: [{ params: { precision: 'fp32' }, responseType: 'blob' }],
    },
  ]);
});

test('training data routes use the registered backend paths', async () => {
  const calls = [];
  const { imageApi, settingsApi } = loadEndpoints(calls);

  await imageApi.capabilities();
  await settingsApi.getGpuCapacity();

  assert.deepEqual(
    calls.map(({ method, url }) => ({ method, url })),
    [
      { method: 'get', url: '/capabilities' },
      { method: 'get', url: '/gpu-capacity' },
    ],
  );
});
