// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

import type { JobProgressPayload, WebSocketMessage } from '@nftypes/api';

/** JobSocketOptions supplies shared authentication and progress handling. */
export interface JobSocketOptions {
  tenantId?: string | null;
  accessToken?: string | null;
  onProgress: (progress: JobProgressPayload) => void;
}

/** JobSocketRuntime permits deterministic socket and timer doubles in tests. */
export interface JobSocketRuntime {
  createSocket?: (url: string) => WebSocket;
  schedule?: (callback: () => void, delay: number) => ReturnType<typeof setTimeout>;
  cancel?: (handle: ReturnType<typeof setTimeout>) => void;
}

/** buildWebSocketUrl keeps single- and multi-job consumers on the same URL contract. */
export const buildWebSocketUrl = (
  jobId?: string,
  tenantId?: string | null,
  accessToken?: string | null,
): string => {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const baseUrl = `${protocol}//${window.location.host}/ws`;
  const params = new URLSearchParams();

  if (jobId) params.append('job_id', jobId);
  if (tenantId) params.append('tenant_id', tenantId);
  if (accessToken) params.append('token', accessToken);

  return `${baseUrl}?${params.toString()}`;
};

/** jobStatusSubscriptionKey gives inline job arrays a stable effect dependency. */
export const jobStatusSubscriptionKey = (jobIds: string[]): string => JSON.stringify(jobIds);

/** subscribeToJob opens one reconnecting socket and returns its idempotent cleanup. */
const subscribeToJob = (
  jobId: string,
  options: JobSocketOptions,
  runtime: JobSocketRuntime,
): (() => void) => {
  const createSocket = runtime.createSocket ?? ((url: string) => new WebSocket(url));
  const schedule = runtime.schedule ?? ((callback: () => void, delay: number) => setTimeout(callback, delay));
  const cancel = runtime.cancel ?? ((handle: ReturnType<typeof setTimeout>) => clearTimeout(handle));
  let disposed = false;
  let socket: WebSocket | null = null;
  let reconnectTimeout: ReturnType<typeof setTimeout> | null = null;

  const connect = () => {
    if (disposed || !jobId) return;
    const ws = createSocket(buildWebSocketUrl(jobId, options.tenantId, options.accessToken));
    socket = ws;
    ws.onmessage = (event) => {
      try {
        const message: WebSocketMessage = JSON.parse(event.data);
        if (message.type === 'job_progress') {
          options.onProgress(message.payload as JobProgressPayload);
        }
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
    ws.onclose = (event) => {
      socket = null;
      if (!disposed && event.code !== 1000) {
        reconnectTimeout = schedule(connect, 3000);
      }
    };
  };

  connect();
  return () => {
    if (disposed) return;
    disposed = true;
    if (reconnectTimeout) cancel(reconnectTimeout);
    reconnectTimeout = null;
    socket?.close(1000, 'Client disconnect');
    socket = null;
  };
};

/** subscribeToJobStatuses owns one socket per job and cleans up the whole set. */
export const subscribeToJobStatuses = (
  jobIds: string[],
  options: JobSocketOptions,
  runtime: JobSocketRuntime = {},
): (() => void) => {
  const cleanups = jobIds.map((jobId) => subscribeToJob(jobId, options, runtime));
  return () => cleanups.forEach((cleanup) => cleanup());
};
