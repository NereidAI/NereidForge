// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

import type { LoginResponse } from '@nftypes/api';

// Session payload delivered by the backend in the URL fragment after a
// successful SSO exchange (both ns and oidc modes converge here).
export type SsoSession = LoginResponse & { to?: string };

/** Browser entry URL for the SSO flow (backend 302s to NS or the IdP). */
export const ssoLoginUrl = (loginPath: string, to: string): string => {
  const base = loginPath || '/api/v1/auth/sso/login';
  return `${base}?to=${encodeURIComponent(isRelativePath(to) ? to : '/dashboard')}`;
};

/** Only same-origin relative paths may be used as post-login deep links. */
export const isRelativePath = (path: string | null | undefined): boolean => {
  if (!path) return false;
  return path.startsWith('/') && !path.startsWith('//') && !path.includes('\\');
};

/** Decode the `#session=...&to=...` fragment set by the backend callback. */
export const parseSessionFromHash = (hash: string): SsoSession | null => {
  const raw = hash.startsWith('#') ? hash.slice(1) : hash;
  if (!raw) return null;
  const params = new URLSearchParams(raw);
  const session = params.get('session');
  if (!session) return null;
  try {
    return JSON.parse(session) as SsoSession;
  } catch {
    return null;
  }
};

/** Remember the IdP/NS logout URL so logout can chain end-session. */
export const SSO_LOGOUT_KEY = 'nereidforge.sso.logout_url';

export const rememberSsoLogout = (logoutUrl: string | undefined): void => {
  if (logoutUrl) window.localStorage.setItem(SSO_LOGOUT_KEY, logoutUrl);
};

export const takeSsoLogout = (): string | null => {
  const url = window.localStorage.getItem(SSO_LOGOUT_KEY);
  if (url) window.localStorage.removeItem(SSO_LOGOUT_KEY);
  return url;
};
