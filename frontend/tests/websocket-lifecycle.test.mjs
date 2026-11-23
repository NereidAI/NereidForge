// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import ts from 'typescript';

// Transpile only the framework-independent helper; no browser or React runtime
// is needed to verify its connection lifecycle with deterministic doubles.
const source = readFileSync(new URL('../src/hooks/jobSocket.ts', import.meta.url), 'utf8');
const { outputText } = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
});
const loaded = { exports: {} };
const execute = new Function('exports', 'module', 'require', 'window', 'URLSearchParams', outputText);
execute(loaded.exports, loaded, () => {
  throw new Error('The helper must not require runtime dependencies');
}, { location: { protocol: 'https:', host: 'forge.example' } }, URLSearchParams);
const {
  jobStatusSubscriptionKey,
  subscribeToJobStatuses,
} = loaded.exports;

class FakeSocket {
  constructor(url, sockets) {
    this.url = url;
    this.closeCalls = [];
    this.onmessage = null;
    this.onerror = null;
    this.onclose = null;
    sockets.push(this);
  }

  close(code, reason) {
    this.closeCalls.push({ code, reason });
    this.onclose?.({ code, reason });
  }
}

function createRuntime() {
  const sockets = [];
  const timers = new Map();
  let nextTimer = 0;
  const delays = [];
  const runtime = {
    createSocket: (url) => new FakeSocket(url, sockets),
    schedule: (callback, delay) => {
      const handle = ++nextTimer;
      timers.set(handle, callback);
      delays.push(delay);
      return handle;
    },
    cancel: (handle) => timers.delete(handle),
  };
  return {
    sockets,
    timers,
    delays,
    runtime,
    runNextTimer() {
      const [handle, callback] = timers.entries().next().value;
      timers.delete(handle);
      callback();
    },
  };
}

test('job keys are stable and sockets preserve auth and progress handling', () => {
  assert.equal(jobStatusSubscriptionKey(['a', 'b']), jobStatusSubscriptionKey([...['a', 'b']]));
  assert.notEqual(jobStatusSubscriptionKey(['a']), jobStatusSubscriptionKey(['a', 'b']));
  const fake = createRuntime();
  const progress = [];
  const cleanup = subscribeToJobStatuses(['job a'], {
    tenantId: 'tenant one',
    accessToken: 'secret-token',
    onProgress: (value) => progress.push(value),
  }, fake.runtime);
  const url = new URL(fake.sockets[0].url);
  assert.equal(url.protocol, 'wss:');
  assert.equal(url.host, 'forge.example');
  assert.equal(url.pathname, '/ws');
  assert.equal(url.searchParams.get('job_id'), 'job a');
  assert.equal(url.searchParams.get('tenant_id'), 'tenant one');
  assert.equal(url.searchParams.get('token'), 'secret-token');
  fake.sockets[0].onmessage({
    data: JSON.stringify({ type: 'job_progress', payload: { job_id: 'job a', progress: 25 } }),
  });
  assert.deepEqual(progress, [{ job_id: 'job a', progress: 25 }]);
  cleanup();
});

test('an abnormal close reconnects after three seconds and cleanup closes the replacement', () => {
  const fake = createRuntime();
  const cleanup = subscribeToJobStatuses(['job-a'], { onProgress: () => {} }, fake.runtime);
  fake.sockets[0].onclose({ code: 1006 });
  assert.deepEqual(fake.delays, [3000]);
  assert.equal(fake.timers.size, 1);
  fake.runNextTimer();
  assert.equal(fake.sockets.length, 2);
  cleanup();
  assert.deepEqual(fake.sockets[1].closeCalls, [{ code: 1000, reason: 'Client disconnect' }]);
  assert.equal(fake.timers.size, 0);
});

test('disposal cancels a pending reconnect without creating another socket', () => {
  const fake = createRuntime();
  const cleanup = subscribeToJobStatuses(['job-a'], { onProgress: () => {} }, fake.runtime);
  fake.sockets[0].onclose({ code: 1012 });
  assert.equal(fake.timers.size, 1);
  cleanup();
  cleanup();
  assert.equal(fake.timers.size, 0);
  assert.equal(fake.sockets.length, 1);
});

test('job-set replacement closes removed and obsolete connections', () => {
  const fake = createRuntime();
  const options = { onProgress: () => {} };
  const cleanupInitial = subscribeToJobStatuses(['job-a', 'job-b'], options, fake.runtime);
  cleanupInitial();
  assert.equal(fake.sockets.length, 2);
  assert.ok(fake.sockets.every((socket) => socket.closeCalls[0]?.code === 1000));

  const cleanupAdded = subscribeToJobStatuses(['job-a', 'job-b', 'job-c'], options, fake.runtime);
  assert.equal(fake.sockets.length, 5);
  cleanupAdded();
  const cleanupRemoved = subscribeToJobStatuses(['job-c'], options, fake.runtime);
  assert.equal(fake.sockets.length, 6);
  assert.match(fake.sockets[5].url, /job_id=job-c/);
  cleanupRemoved();
  assert.ok(fake.sockets.every((socket) => socket.closeCalls[0]?.code === 1000));
});

test('authentication replacement closes old sockets and uses new credentials', () => {
  const fake = createRuntime();
  const cleanupOld = subscribeToJobStatuses(['job-a'], {
    tenantId: 'old-tenant',
    accessToken: 'old-token',
    onProgress: () => {},
  }, fake.runtime);
  cleanupOld();
  const cleanupNew = subscribeToJobStatuses(['job-a'], {
    tenantId: 'new-tenant',
    accessToken: 'new-token',
    onProgress: () => {},
  }, fake.runtime);
  const url = new URL(fake.sockets[1].url);
  assert.equal(url.searchParams.get('tenant_id'), 'new-tenant');
  assert.equal(url.searchParams.get('token'), 'new-token');
  assert.equal(fake.sockets[0].closeCalls[0].code, 1000);
  cleanupNew();
});

test('a normal close does not schedule a reconnect', () => {
  const fake = createRuntime();
  const cleanup = subscribeToJobStatuses(['job-a'], { onProgress: () => {} }, fake.runtime);
  fake.sockets[0].onclose({ code: 1000 });
  assert.equal(fake.timers.size, 0);
  cleanup();
});
