// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { AxiosResponse } from 'axios';
import type { TableProps } from 'antd';
import { useTranslation } from 'react-i18next';

// Generic server-paginated table state. Every page change / page-size change /
// filter change triggers a new API request; filters always reset to page 1 so
// switching a filter while sitting on a later page can never show a blank page.
export interface PageEnvelope<T> {
  items: T[];
  total: number;
  page?: number;
  page_size?: number;
  total_pages?: number;
}

export interface UseServerTableOptions<T, F extends Record<string, unknown>> {
  // API call returning the standard {items,total,page,page_size} envelope.
  list: (params: { page: number; page_size: number } & F) => Promise<AxiosResponse<PageEnvelope<T>>>;
  // Server-side filters (search, category, status, ...). Identity of the
  // serialized value drives re-fetches, so callers can pass inline objects.
  filters?: F;
  defaultPageSize?: number;
  // Optional polling interval (keeps the currently viewed page fresh).
  pollMs?: number;
  // Pause polling/fetching when false (e.g. modal closed).
  enabled?: boolean;
  rowKey?: string | ((row: T) => string);
  // Side-channel override for the total reported by the API (deployments
  // endpoint returns {services: [...]} instead of the standard envelope).
  extract?: (data: unknown) => PageEnvelope<T>;
}

export interface ServerTable<T> {
  items: T[];
  total: number;
  page: number;
  pageSize: number;
  loading: boolean;
  setPage: (p: number) => void;
  reload: () => void;
  tableProps: {
    dataSource: T[];
    loading: boolean;
    rowKey: string | ((row: T) => string);
    pagination: NonNullable<TableProps<T>['pagination']>;
    onChange: NonNullable<TableProps<T>['onChange']>;
  };
}

export function useServerTable<T extends object, F extends Record<string, unknown> = Record<string, never>>(
  opts: UseServerTableOptions<T, F>,
): ServerTable<T> {
  const { t } = useTranslation();
  const defaultPageSize = opts.defaultPageSize ?? 10;
  const [items, setItems] = useState<T[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPageState] = useState(1);
  const [pageSize, setPageSize] = useState(defaultPageSize);
  const [loading, setLoading] = useState(false);

  const filtersKey = JSON.stringify(opts.filters ?? {});
  const filtersRef = useRef(opts.filters ?? ({} as F));
  filtersRef.current = opts.filters ?? ({} as F);
  const listRef = useRef(opts.list);
  listRef.current = opts.list;
  const extractRef = useRef(opts.extract);
  extractRef.current = opts.extract;

  // Changing any server-side filter jumps back to the first page.
  const firstFiltersKey = useRef(filtersKey);
  useEffect(() => {
    if (firstFiltersKey.current !== filtersKey) {
      firstFiltersKey.current = filtersKey;
      setPageState(1);
    }
  }, [filtersKey]);

  const fetchPage = useCallback(
    async (p: number, ps: number) => {
      if (opts.enabled === false) return;
      setLoading(true);
      try {
        const params = { page: p, page_size: ps, ...(filtersRef.current) } as {
          page: number;
          page_size: number;
        } & F;
        const res = await listRef.current(params);
        const env = extractRef.current ? extractRef.current(res.data) : (res.data as PageEnvelope<T>);
        setItems(env.items ?? []);
        setTotal(env.total ?? 0);
        // If the current page moved past the last page (rows deleted / filter
        // narrowed the result set), step back and reload once.
        const lastPage = Math.max(1, Math.ceil((env.total ?? 0) / ps));
        if (p > lastPage) {
          setPageState(lastPage);
        }
      } catch (e) {
        console.error('Failed to load table page:', e);
        setItems([]);
        setTotal(0);
      } finally {
        setLoading(false);
      }
    },
    [opts.enabled],
  );

  useEffect(() => {
    fetchPage(page, pageSize);
  }, [fetchPage, page, pageSize, filtersKey]);

  useEffect(() => {
    if (!opts.pollMs) return;
    const timer = setInterval(() => {
      fetchPage(page, pageSize);
    }, opts.pollMs);
    return () => clearInterval(timer);
  }, [opts.pollMs, fetchPage, page, pageSize, filtersKey]);

  const setPage = useCallback((p: number) => setPageState(p), []);
  const reload = useCallback(() => {
    fetchPage(page, pageSize);
  }, [fetchPage, page, pageSize]);

  const onChange = useCallback<NonNullable<TableProps<T>['onChange']>>(
    (pg) => {
      if (pg.pageSize && pg.pageSize !== pageSize) {
        setPageSize(pg.pageSize);
        setPageState(1);
      } else if (pg.current && pg.current !== page) {
        setPageState(pg.current);
      }
    },
    [page, pageSize],
  );

  const pagination = useMemo<NonNullable<TableProps<T>['pagination']>>(
    () => ({
      current: page,
      pageSize,
      total,
      showSizeChanger: true,
      pageSizeOptions: [10, 20, 50, 100],
      showTotal: (n) => t('common.totalItems', { count: n, total }),
    }),
    [page, pageSize, total, t],
  );

  return {
    items,
    total,
    page,
    pageSize,
    loading,
    setPage,
    reload,
    tableProps: {
      dataSource: items,
      loading,
      rowKey: opts.rowKey ?? 'id',
      pagination,
      onChange,
    },
  };
}
