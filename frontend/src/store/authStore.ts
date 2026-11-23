// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import type { User, AuthTokens } from '@nftypes/api';

interface AuthState {
  user: User | null;
  accessToken: string | null;
  refreshToken: string | null;
  tenantId: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  
  // Actions
  setAuth: (user: User, tokens: AuthTokens) => void;
  setTokens: (accessToken: string, refreshToken?: string) => void;
  setUser: (user: User | null) => void;
  setTenant: (tenantId: string) => void;
  logout: () => void;
  setLoading: (loading: boolean) => void;
  
  // Computed
  hasRole: (role: string) => boolean;
  isAdmin: () => boolean;
  isSuperAdmin: () => boolean;
}

const STORAGE_KEY = 'nereidforge-auth';

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      user: null,
      accessToken: null,
      refreshToken: null,
      tenantId: null,
      isAuthenticated: false,
      isLoading: false,

      setAuth: (user: User, tokens: AuthTokens) => {
        const tenantId = user.tenant_id;
        set({
          user,
          accessToken: tokens.access_token,
          refreshToken: tokens.refresh_token,
          tenantId,
          isAuthenticated: true,
          isLoading: false,
        });
      },

      setTokens: (accessToken: string, refreshToken?: string) => {
        // Access refresh responses intentionally omit refresh_token; preserve
        // the still-valid token instead of clearing the next refresh attempt.
        set({ accessToken, refreshToken: refreshToken ?? get().refreshToken });
      },

      setUser: (user: User | null) => {
        if (user) {
          set({ user, tenantId: user.tenant_id, isAuthenticated: true });
        } else {
          set({ user: null, isAuthenticated: false });
        }
      },

      setTenant: (tenantId: string) => {
        set({ tenantId });
        const { user } = get();
        if (user) {
          set({ user: { ...user, tenant_id: tenantId } });
        }
      },

      logout: () => {
        set({
          user: null,
          accessToken: null,
          refreshToken: null,
          tenantId: null,
          isAuthenticated: false,
          isLoading: false,
        });
      },

      setLoading: (isLoading: boolean) => {
        set({ isLoading });
      },

      hasRole: (role: string) => {
        const { user } = get();
        return user?.roles?.includes(role) ?? false;
      },

      isAdmin: () => {
        const { user } = get();
        return user?.roles?.includes('admin') ?? false;
      },

      isSuperAdmin: () => {
        const { user } = get();
        // This only controls presentation. The backend resolves the token's
        // stable user ID and remains authoritative for local/federated identity.
        return user?.username === 'admin'
          && user.tenant_id === 'default'
          && (user.roles?.includes('admin') ?? false);
      },
    }),
    {
      name: STORAGE_KEY,
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        user: state.user,
        accessToken: state.accessToken,
        refreshToken: state.refreshToken,
        tenantId: state.tenantId,
        isAuthenticated: state.isAuthenticated,
      }),
    }
  )
);

// Token expiration check
export const isTokenExpired = (token: string | null): boolean => {
  if (!token) return true;
  try {
    const payload = JSON.parse(atob(token.split('.')[1]));
    return payload.exp * 1000 < Date.now();
  } catch {
    return true;
  }
};

// Initialize auth on app start
export const initializeAuth = async (): Promise<void> => {
  const { accessToken, refreshToken, isAuthenticated, setLoading, logout } = useAuthStore.getState();

  if (!isAuthenticated || !accessToken) return;
  
  setLoading(true);
  
  // Check if access token is expired
  if (isTokenExpired(accessToken)) {
    // Try to refresh
    if (refreshToken && !isTokenExpired(refreshToken)) {
      try {
        const { authApi } = await import('@api/endpoints');
        const response = await authApi.refresh(refreshToken);
        useAuthStore.getState().setTokens(response.data.access_token);
      } catch {
        logout();
      }
    } else {
      logout();
    }
  }
  
  setLoading(false);
};
