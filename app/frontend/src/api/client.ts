// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

import axios, { AxiosError, AxiosInstance, InternalAxiosRequestConfig, AxiosRequestConfig } from 'axios';
import { useAuthStore } from '@store/authStore';
import type { AuthTokens, ApiError } from '@nftypes/api';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api/v1';

class ApiClient {
  private client: AxiosInstance;
  private refreshTokenPromise: Promise<string> | null = null;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.setupInterceptors();
  }

  private setupInterceptors(): void {
    // Request interceptor - add auth headers
    this.client.interceptors.request.use(
      (config: InternalAxiosRequestConfig) => {
        const { accessToken, tenantId } = useAuthStore.getState();

        if (accessToken) {
          config.headers.Authorization = `Bearer ${accessToken}`;
        }

        if (tenantId) {
          config.headers['X-Tenant-Id'] = tenantId;
        }

        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor - handle 401 and token refresh
    this.client.interceptors.response.use(
      (response) => response,
      async (error: AxiosError<ApiError>) => {
        const originalRequest = error.config as InternalAxiosRequestConfig & { _retry?: boolean };

        if (error.response?.status === 401 && !originalRequest._retry) {
          originalRequest._retry = true;

          try {
            const newAccessToken = await this.refreshAccessToken();
            originalRequest.headers.Authorization = `Bearer ${newAccessToken}`;
            return this.client(originalRequest);
          } catch (refreshError) {
            // Refresh failed, logout user
            useAuthStore.getState().logout();
            window.location.href = '/login';
            return Promise.reject(refreshError);
          }
        }

        return Promise.reject(error);
      }
    );
  }

  private async refreshAccessToken(): Promise<string> {
    // Prevent multiple simultaneous refresh attempts
    if (this.refreshTokenPromise) {
      return this.refreshTokenPromise;
    }

    const { refreshToken } = useAuthStore.getState();

    if (!refreshToken) {
      throw new Error('No refresh token available');
    }

    this.refreshTokenPromise = (async () => {
      try {
        const response = await axios.post<AuthTokens>(`${API_BASE_URL}/auth/refresh`, {
          refresh_token: refreshToken,
        });

        const { access_token } = response.data;
        useAuthStore.getState().setTokens(access_token);

        return access_token;
      } finally {
        this.refreshTokenPromise = null;
      }
    })();

    return this.refreshTokenPromise;
  }

  public getClient(): AxiosInstance {
    return this.client;
  }

  // Convenience methods
  public get<T>(url: string, params?: object) {
    const config: AxiosRequestConfig = { params };
    return this.client.get<T>(url, config);
  }

  public post<T>(url: string, data?: object, config?: AxiosRequestConfig) {
    return this.client.post<T>(url, data, config);
  }

  public put<T>(url: string, data?: object) {
    return this.client.put<T>(url, data);
  }

  public patch<T>(url: string, data?: object) {
    return this.client.patch<T>(url, data);
  }

  public delete<T>(url: string) {
    return this.client.delete<T>(url);
  }

  public async downloadFile(url: string, filename?: string): Promise<void> {
    // downloadUrl already carries the full /api/v1 prefix; use fetch (not the
    // axios baseURL, which would double the prefix) and attach the bearer
    // token manually, then trigger a browser download.
    const { accessToken, tenantId } = useAuthStore.getState();
    const resp = await fetch(url, {
      headers: {
        Authorization: accessToken ? `Bearer ${accessToken}` : '',
        'X-Tenant-Id': tenantId || 'default',
      },
    });
    if (!resp.ok) {
      throw new Error(`download failed (HTTP ${resp.status})`);
    }
    const blob = await resp.blob();
    const dl = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = dl;
    const cd = resp.headers.get('content-disposition') || '';
    const m = cd.match(/filename="?([^";]+)"?/);
    a.download = filename || (m ? m[1] : 'download');
    a.click();
    URL.revokeObjectURL(dl);
  }
}

export const apiClient = new ApiClient();
export default apiClient.getClient();
