// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

import React, { useEffect, useRef, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { Card, Spin, Result, Button } from 'antd';
import { useTranslation } from 'react-i18next';
import { authApi } from '@api/endpoints';
import { useAuthStore } from '@store/authStore';
import { isRelativePath, parseSessionFromHash, rememberSsoLogout } from '@/utils/sso';

// Landing route the backend redirects to after an SSO login. Two shapes reach
// the SPA:
//   - oidc mode: the IdP returns ?code=...&state=... here, and this page posts
//     them back to the API, which performs the token exchange.
//   - ns mode: the backend already exchanged the NS ticket and redirects here
//     with #session=... (token pair in the fragment, never in a query string).
// Both paths end in the same local session store.
const SSOCallback: React.FC = () => {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [error, setError] = useState('');
  const started = useRef(false);

  useEffect(() => {
    if (started.current) return;
    started.current = true;

    const finish = (session: { access_token: string; refresh_token: string; token_type: string; expires_in: number; user: any; to?: string }) => {
      useAuthStore.getState().setAuth(session.user, {
        access_token: session.access_token,
        refresh_token: session.refresh_token,
        token_type: session.token_type,
        expires_in: session.expires_in,
      });
      const to = isRelativePath(session.to) ? (session.to as string) : '/dashboard';
      navigate(to, { replace: true });
    };

    const fail = (code: string) => setError(code || 'oidc_failed');

    // ns mode: token pair arrives in the fragment.
    const session = parseSessionFromHash(window.location.hash);
    if (session) {
      finish(session);
      return;
    }

    const oidcError = searchParams.get('oidc_error') || searchParams.get('error');
    if (oidcError) {
      fail(oidcError);
      return;
    }

    // oidc mode: exchange the authorization code server-side.
    const code = searchParams.get('code') || '';
    const state = searchParams.get('state') || '';
    if (!code || !state) {
      fail('oidc_invalid_state');
      return;
    }
    authApi
      .ssoStatus()
      .then((status) => {
        const logoutUrl = status.data.logout_url;
        return authApi.ssoCallback(code, state).then((res) => ({ res, logoutUrl }));
      })
      .then(({ res, logoutUrl }) => {
        if (res.data.error) {
          fail(res.data.error);
          return;
        }
        rememberSsoLogout(logoutUrl);
        finish(res.data);
      })
      .catch(() => fail('oidc_failed'));
  }, [navigate, searchParams]);

  if (error) {
    return (
      <Card style={{ maxWidth: 400, margin: '100px auto', textAlign: 'center' }}>
        <Result
          status="error"
          title={t('auth.ssoFailedTitle')}
          subTitle={t(`auth.ssoError.${error}`, { defaultValue: t('auth.ssoFailed') })}
          extra={
            <Button type="primary" onClick={() => navigate('/login', { replace: true })}>
              {t('auth.backToLogin')}
            </Button>
          }
        />
      </Card>
    );
  }

  return (
    <Card style={{ maxWidth: 400, margin: '100px auto', textAlign: 'center' }}>
      <Spin size="large" />
      <p style={{ marginTop: 16, color: '#999' }}>{t('auth.ssoSigningIn')}</p>
    </Card>
  );
};

export default SSOCallback;
