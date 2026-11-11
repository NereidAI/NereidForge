// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

import React, { useState } from 'react';
import { Card, Table, Tag } from 'antd';
import { useTranslation } from 'react-i18next';
import { settingsApi } from '@api/endpoints';
import { useServerTable } from '@hooks/useServerTable';
import { useDebounce } from '@hooks/useDebounce';
import FilterSearchBar from '@components/FilterSearchBar';
import { formatDateTime } from '../../utils/format';

// Audit actions currently recorded by NF (static dropdown source of truth).
const AUDIT_ACTIONS = ['optimization.gate.override'] as const;

interface AuditEntry {
  id: string;
  tenant_id: string;
  user_id: string;
  action: string;
  resource_type: string;
  resource_id: string;
  details: Record<string, unknown>;
  created_at: string;
}

// Audit log viewer: admin can review high-risk operation traces
export const AuditLogs: React.FC = () => {
  const { t } = useTranslation();
  // Server-side action filter plus debounced user/resource-id search.
  const [actionFilter, setActionFilter] = useState('');
  const [searchInput, setSearchInput] = useState('');
  const search = useDebounce(searchInput.trim(), 300);
  const table = useServerTable<AuditEntry, { action?: string; search?: string }>({
    list: (params) => settingsApi.getAuditLogs(params),
    filters: {
      action: actionFilter || undefined,
      search: search || undefined,
    },
    defaultPageSize: 10,
  });

  const columns = [
    { title: t('pages.settings.audit.time'), dataIndex: 'created_at', key: 'created_at', render: (d: string) => formatDateTime(d) },
    { title: t('pages.settings.audit.user'), dataIndex: 'user_id', key: 'user_id' },
    {
      title: t('pages.settings.audit.action'),
      dataIndex: 'action',
      key: 'action',
      render: (a: string) => <Tag color={a.includes('delete') ? 'red' : 'blue'}>{a}</Tag>,
    },
    { title: t('pages.settings.audit.resource'), dataIndex: 'resource_type', key: 'resource_type' },
    { title: t('pages.settings.audit.resourceId'), dataIndex: 'resource_id', key: 'resource_id' },
    {
      title: t('pages.settings.audit.details'),
      dataIndex: 'details',
      key: 'details',
      ellipsis: true,
      render: (d: Record<string, unknown>) => (d ? JSON.stringify(d).slice(0, 80) : '-'),
    },
  ];

  return (
    <Card size="small">
      <FilterSearchBar
        filterPlaceholder={t('pages.settings.audit.filterAction')}
        filterValue={actionFilter}
        onFilterChange={setActionFilter}
        filterOptions={AUDIT_ACTIONS.map((v) => ({
          value: v, label: t(`pages.settings.audit.action_${v.replace(/\./g, '_')}`),
        }))}
        searchPlaceholder={t('pages.settings.audit.searchPlaceholder')}
        searchValue={searchInput}
        onSearchChange={setSearchInput}
      />
      <Table {...table.tableProps} columns={columns} scroll={{ x: 'max-content' }} />
    </Card>
  );
};

export default AuditLogs;
