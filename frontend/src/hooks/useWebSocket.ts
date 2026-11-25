// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useRef, useCallback, useState } from 'react';
import { useAuthStore } from '@store/authStore';
import type { WebSocketMessage, JobProgressPayload } from '@nftypes/api';
import {
  buildWebSocketUrl,
  jobStatusSubscriptionKey,
  subscribeToJobStatuses,
} from './jobSocket';

interface UseWebSocketOptions {
  jobId?: string;
  onMessage?: (message: WebSocketMessage) => void;
  onProgress?: (progress: JobProgressPayload) => void;
  onError?: (error: Event) => void;
  onClose?: (event: CloseEvent) => void;
  reconnect?: boolean;
  reconnectInterval?: number;
}

interface UseWebSocketReturn {
  connect: () => void;
  disconnect: () => void;
  send: (data: unknown) => void;
  isConnected: boolean;
  lastMessage: WebSocketMessage | null;
}

export const useWebSocket = (options: UseWebSocketOptions = {}): UseWebSocketReturn => {
  const {
    jobId,
    onMessage,
    onProgress,
    onError,
    onClose,
    reconnect = true,
    reconnectInterval = 3000,
  } = options;

  const { accessToken, tenantId } = useAuthStore();
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);

  const buildUrl = useCallback(
    () => buildWebSocketUrl(jobId, tenantId, accessToken),
    [jobId, tenantId, accessToken],
  );

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const url = buildUrl();
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      setIsConnected(true);
      console.log('WebSocket connected:', url);
    };

    ws.onmessage = (event) => {
      try {
        const message: WebSocketMessage = JSON.parse(event.data);
        setLastMessage(message);
        
        if (onMessage) onMessage(message);
        
        if (message.type === 'job_progress' && onProgress) {
          onProgress(message.payload as JobProgressPayload);
        }
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      if (onError) onError(error);
    };

    ws.onclose = (event) => {
      setIsConnected(false);
      console.log('WebSocket disconnected:', event.code, event.reason);
      if (onClose) onClose(event);

      if (reconnect && event.code !== 1000) {
        reconnectTimeoutRef.current = setTimeout(() => {
          connect();
        }, reconnectInterval);
      }
    };
  }, [buildUrl, onMessage, onProgress, onError, onClose, reconnect, reconnectInterval]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    
    if (wsRef.current) {
      wsRef.current.close(1000, 'Client disconnect');
      wsRef.current = null;
    }
    
    setIsConnected(false);
  }, []);

  const send = useCallback((data: unknown) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    } else {
      console.warn('WebSocket not connected, cannot send message');
    }
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      disconnect();
    };
  }, [disconnect]);

  // Reconnect when jobId or auth changes
  useEffect(() => {
    if (jobId) {
      connect();
    } else {
      disconnect();
    }

    return () => disconnect();
  }, [jobId, accessToken, tenantId, connect, disconnect]);

  return {
    connect,
    disconnect,
    send,
    isConnected,
    lastMessage,
  };
};

/** useJobStatus subscribes to a changing set of jobs without varying the hook count. */
export const useJobStatus = (jobIds: string[]) => {
  const [statuses, setStatuses] = useState<Record<string, JobProgressPayload>>({});
  const { accessToken, tenantId } = useAuthStore();

  const handleProgress = useCallback((progress: JobProgressPayload) => {
    setStatuses(prev => ({
      ...prev,
      [progress.job_id]: progress,
    }));
  }, []);

  // A value key keeps inline array literals from reconnecting healthy sockets.
  const jobIdsKey = jobStatusSubscriptionKey(jobIds);
  useEffect(() => {
    const ids = JSON.parse(jobIdsKey) as string[];
    return subscribeToJobStatuses(ids, {
      tenantId,
      accessToken,
      onProgress: handleProgress,
    });
  }, [jobIdsKey, accessToken, tenantId, handleProgress]);

  return statuses;
};
