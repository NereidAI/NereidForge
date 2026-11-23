// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuthStore } from '@store/authStore';
import { authApi } from '@api/endpoints';
import { takeSsoLogout } from '@/utils/sso';

export const useAuth = () => {
  const navigate = useNavigate();
  const {
    setAuth,
    isLoading,
    setLoading,
    user,
    tenantId,
    isAuthenticated,
    isAdmin,
    isSuperAdmin,
    hasRole,
    setTenant,
    logout: storeLogoutFn,
  } = useAuthStore();

  // Login with username + password (local JWT auth)
  const login = useCallback(
    async (username: string, password: string, namespace?: string, captchaId?: string, captchaAnswer?: number) => {
      setLoading(true);
      try {
        const response = await authApi.login({
          username,
          password,
          namespace: namespace || undefined,
          captcha_id: captchaId,
          captcha_answer: captchaAnswer,
        });
        setAuth(response.data.user, {
          access_token: response.data.access_token,
          refresh_token: response.data.refresh_token,
          token_type: response.data.token_type,
          expires_in: response.data.expires_in,
        });
        navigate('/dashboard');
      } finally {
        setLoading(false);
      }
    },
    [setLoading, setAuth, navigate]
  );

  // Logout
  const logout = useCallback(async () => {
    try {
      await authApi.logout();
    } catch {
      // ignore network errors on logout
    }
    const ssoLogoutUrl = takeSsoLogout();
    storeLogoutFn();
    if (ssoLogoutUrl) {
      // SSO session: chain NS/NI end-session (standard section 4.4.7).
      window.location.assign(ssoLogoutUrl);
      return;
    }
    navigate('/login');
  }, [storeLogoutFn, navigate]);

  return {
    user,
    tenantId,
    isAuthenticated,
    isAdmin: isAdmin(),
    isSuperAdmin: isSuperAdmin(),
    hasRole,
    roles: user?.roles ?? [],
    login,
    logout,
    isLoading,
    setTenant,
  };
};

export const useRequireAuth = () => {
  const navigate = useNavigate();
  const { isAuthenticated, isLoading } = useAuthStore();

  useEffect(() => {
    if (!isLoading && !isAuthenticated) {
      navigate('/login', { replace: true });
    }
  }, [isAuthenticated, isLoading, navigate]);

  return { isAuthenticated, isLoading };
};

export const useRequireAdmin = () => {
  const navigate = useNavigate();
  const { isAuthenticated, isLoading, isAdmin } = useAuthStore();

  useEffect(() => {
    if (!isLoading) {
      if (!isAuthenticated) {
        navigate('/login', { replace: true });
      } else if (!isAdmin()) {
        navigate('/dashboard', { replace: true });
      }
    }
  }, [isAuthenticated, isLoading, navigate, isAdmin]);

  return { isAuthenticated, isLoading, isAdmin: isAdmin() };
};
