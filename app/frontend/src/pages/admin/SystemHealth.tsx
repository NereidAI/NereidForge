// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

import React, { useCallback, useEffect, useState } from 'react';
import { Alert, Button, Card, Space, Spin, Tag, Tooltip, Typography } from 'antd';
import { CheckCircleFilled, ExclamationCircleFilled, ReloadOutlined, WarningFilled } from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import { settingsApi } from '../../api/endpoints';

interface PreflightCheck {
  name: string;
  status: 'ok' | 'warn' | 'error';
  message: string;
}

interface PreflightReport {
  summary: 'ok' | 'warn' | 'error';
  ok: boolean;
  error_count: number;
  warning_count: number;
  checks: PreflightCheck[];
}

// Read-only data-plane dependency checks (kubernetes/S3/KFP/runtime images/
// ML backends/inference domain). The backend runs the same probes at startup
// and logs the result; this panel surfaces drift before a deploy fails.
const SystemHealth: React.FC = () => {
  const { t } = useTranslation();
  const [data, setData] = useState<PreflightReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState('');

  const load = useCallback(async () => {
    setLoading(true);
    setErr('');
    try {
      const res = await settingsApi.preflight();
      setData(res.data);
    } catch (e: unknown) {
      const msg = (e as { response?: { data?: { detail?: string; message?: string } } })?.response?.data;
      setErr(msg?.detail || msg?.message || 'preflight request failed');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const icon = (s: PreflightCheck['status']) => {
    if (s === 'ok') return <CheckCircleFilled style={{ color: '#52c41a' }} />;
    if (s === 'warn') return <WarningFilled style={{ color: '#faad14' }} />;
    return <ExclamationCircleFilled style={{ color: '#ff4d4f' }} />;
  };

  const tag = (s: PreflightCheck['status']) => {
    if (s === 'ok') return <Tag color="green">{t('pages.preflight.ok')}</Tag>;
    if (s === 'warn') return <Tag color="orange">{t('pages.preflight.warn')}</Tag>;
    return <Tag color="red">{t('pages.preflight.error')}</Tag>;
  };

  return (
    <Card
      size="small"
      style={{ marginTop: 16 }}
      title={
        <Space>
          {t('pages.preflight.title')}
          {data && tag(data.summary)}
        </Space>
      }
      extra={<Button size="small" icon={<ReloadOutlined />} loading={loading} onClick={() => void load()}>{t('common.refresh')}</Button>}
    >
      <Alert
        type={data?.summary === 'error' ? 'error' : data?.summary === 'warn' ? 'warning' : 'success'}
        showIcon
        style={{ marginBottom: 12 }}
        message={
          data
            ? t('pages.preflight.summary', { errors: data.error_count, warnings: data.warning_count })
            : t('pages.preflight.running')
        }
      />
      {err && <Alert type="error" showIcon message={err} style={{ marginBottom: 12 }} />}
      {loading && !data ? (
        <Spin />
      ) : (
        (data?.checks ?? []).map((c) => (
          <div key={c.name} style={{ display: 'flex', alignItems: 'flex-start', gap: 8, padding: '6px 0', borderBottom: '1px solid #f0f0f0' }}>
            <span style={{ paddingTop: 3 }}>{icon(c.status)}</span>
            <Typography.Text strong style={{ minWidth: 130 }}>{c.name}</Typography.Text>
            <Tooltip title={c.message}>
              <Typography.Text type="secondary" ellipsis style={{ flex: 1, maxWidth: 640 }}>
                {c.message}
              </Typography.Text>
            </Tooltip>
          </div>
        ))
      )}
    </Card>
  );
};

export default SystemHealth;
