// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

// Verify that access-token refresh never erases the stored refresh token.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

const require = createRequire(new URL('../package.json', import.meta.url));
const ts = require('typescript');
const storePath = fileURLToPath(new URL('../src/store/authStore.ts', import.meta.url));

const installStorage = () => {
  const values = new Map();
  globalThis.localStorage = {
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, String(value)),
    removeItem: (key) => values.delete(key),
    clear: () => values.clear(),
  };
};

const loadAuthStore = (refresh = async () => { throw new Error('unexpected refresh'); }) => {
  const { outputText } = ts.transpileModule(readFileSync(storePath, 'utf8'), {
    compilerOptions: {
      module: ts.ModuleKind.CommonJS,
      target: ts.ScriptTarget.ES2022,
      esModuleInterop: true,
    },
  });
  const module = { exports: {} };
  const localRequire = (id) => {
    if (id === '@api/endpoints') return { authApi: { refresh } };
    return require(id);
  };
  const evaluate = new Function('module', 'exports', 'require', outputText);
  evaluate(module, module.exports, localRequire);
  return module.exports;
};

const tokenWithExpiry = (exp) => {
  const encode = (value) => Buffer.from(JSON.stringify(value)).toString('base64');
  return `${encode({ alg: 'HS256', typ: 'JWT' })}.${encode({ exp })}.signature`;
};

const user = {
  id: 'usr-1', username: 'alice', email: '', roles: ['user'], tenant_id: 'default',
};

const loginTokens = () => ({
  access_token: tokenWithExpiry(Math.floor(Date.now() / 1000) + 3600),
  refresh_token: tokenWithExpiry(Math.floor(Date.now() / 1000) + 7200),
  token_type: 'Bearer',
  expires_in: 3600,
});

test('setTokens preserves the refresh token when only an access token is refreshed', () => {
  installStorage();
  const { useAuthStore } = loadAuthStore();
  const tokens = loginTokens();
  useAuthStore.getState().setAuth(user, tokens);

  useAuthStore.getState().setTokens('new-access-token');

  assert.equal(useAuthStore.getState().accessToken, 'new-access-token');
  assert.equal(useAuthStore.getState().refreshToken, tokens.refresh_token);
});

test('only the built-in default-tenant admin passes the presentation gate', () => {
  installStorage();
  const { useAuthStore } = loadAuthStore();
  const admin = {
    ...user, id: 'usr-admin', username: 'admin', roles: ['admin', 'user'], tenant_id: 'default',
  };
  useAuthStore.getState().setAuth(admin, loginTokens());
  assert.equal(useAuthStore.getState().isSuperAdmin(), true);

  for (const candidate of [
    { ...admin, username: 'operator' },
    { ...admin, tenant_id: 'acme' },
    { ...admin, roles: ['user'] },
    { ...admin, username: 'tenant-admin', tenant_id: 'acme' },
  ]) {
    useAuthStore.getState().setUser(candidate);
    assert.equal(useAuthStore.getState().isSuperAdmin(), false);
  }
});

test('initializeAuth preserves the refresh token with the real refresh response shape', async () => {
  installStorage();
  const { useAuthStore, initializeAuth } = loadAuthStore(async () => ({
    data: { access_token: 'new-access-token', token_type: 'Bearer', expires_in: 3600 },
  }));
  const tokens = loginTokens();
  useAuthStore.getState().setAuth(user, {
    ...tokens,
    access_token: tokenWithExpiry(0),
  });

  await initializeAuth();

  assert.equal(useAuthStore.getState().accessToken, 'new-access-token');
  assert.equal(useAuthStore.getState().refreshToken, tokens.refresh_token);
  assert.equal(useAuthStore.getState().isAuthenticated, true);
});
