// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

import React, { useEffect, useState } from 'react';
import { Card, Table, Tag, Button, Space, message, Popconfirm, Modal, Form, Select, Input, Alert } from 'antd';
import { ReloadOutlined, StopOutlined, PlusOutlined, CloudServerOutlined } from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import { modelApi, settingsApi } from '@api/endpoints';
import { HelpButton } from '@components/HelpDrawer';
import PageHeader from '@components/PageHeader';
import FilterSearchBar from '@components/FilterSearchBar';
import type { Model } from '@nftypes/api';
import { formatDateTime } from '../utils/format';

interface DeployedService {
  name: string;
  namespace: string;
  state?: string;
  ready: boolean;
  url: string;
  domain: string;
  runtime?: string;
  image?: string;
  model_uri?: string;
  node?: string;
  age?: string;
  error?: string;
  created_at?: string;
}

// Manage all deployed model services (KServe): view + deploy/undeploy from here
const Deployments: React.FC = () => {
  const { t } = useTranslation();
  const [services, setServices] = useState<DeployedService[]>([]);
  const [loading, setLoading] = useState(false);
  const [models, setModels] = useState<Model[]>([]);
  const [deployOpen, setDeployOpen] = useState(false);
  const [deploying, setDeploying] = useState(false);
  const [undeployingName, setUndeployingName] = useState<string | null>(null);
  const [computeResources, setComputeResources] = useState<any[]>([]);
  const [errorDetail, setErrorDetail] = useState<DeployedService | null>(null);
  const [runtimes, setRuntimes] = useState<Array<{ name: string; model_type: string; image: string }>>([]);
  const [form] = Form.useForm();
  // Client-side filters: status dropdown + name search (all services are
  // already loaded in one list call).
  const [statusFilter, setStatusFilter] = useState('');
  const [searchInput, setSearchInput] = useState('');
  // Derive a stable status key per row for filtering.
  const rowStatus = (r: DeployedService): 'ready' | 'error' | 'deploying' =>
    r.ready ? 'ready' : r.error ? 'error' : 'deploying';
  const filteredServices = React.useMemo(() => {
    const needle = searchInput.trim().toLowerCase();
    return services.filter((r) =>
      (!statusFilter || rowStatus(r) === statusFilter)
      && (!needle || r.name.toLowerCase().includes(needle)),
    );
  }, [services, statusFilter, searchInput]);
  const selectedModelName = Form.useWatch('model_name', form);
  // Versions are fetched on demand when a model is selected. Do not rely on the
  // versions embedded in the models list: the hub detail call fills them with
  // a per-model request that silently yields [] on a transient hub error,
  // which showed up as an empty Version dropdown after picking a model.
  const [deployVersions, setDeployVersions] = useState<string[]>([]);
  const [loadingVersions, setLoadingVersions] = useState(false);
  useEffect(() => {
    let cancelled = false;
    setDeployVersions([]);
    if (!selectedModelName) return;
    setLoadingVersions(true);
    const embedded = (models.find((m) => m.name === selectedModelName)?.versions || [])
      .map((v) => v.version);
    modelApi.getVersions(selectedModelName, { page: 1, page_size: 100 })
      .then((res) => {
        if (!cancelled) {
          const fromApi = (res.data.items || []).map((v) => v.version);
          // Fallback to the embedded snapshot only when the paged call is empty.
          setDeployVersions(fromApi.length ? fromApi : embedded);
        }
      })
      .catch(() => {
        if (!cancelled) setDeployVersions(embedded);
      })
      .finally(() => { if (!cancelled) setLoadingVersions(false); });
    return () => { cancelled = true; };
  }, [selectedModelName, models]);

  const load = async () => {
    setLoading(true);
    try {
      const res = await settingsApi.getDeployments();
      setServices(res.data.services ?? []);
    } catch {
      setServices([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
    modelApi.getServingRuntimes().then((res) => setRuntimes(res.data.items ?? [])).catch(() => {});
  }, []);

  // Auto-refresh only while at least one service is deploying or not yet
  // ready; stops once every row is steady (ready/failed).
  const hasBusyService = services.some((s) => s.state === 'deploying' || (!s.ready && !s.error));
  useEffect(() => {
    if (!hasBusyService) return;
    const timer = setInterval(load, 10000);
    return () => clearInterval(timer);
  }, [hasBusyService, services.length]);

  // Open deploy modal: load model list + cluster resources
  const openDeploy = async () => {
    form.resetFields();
    setDeployOpen(true);
    try {
      const [mRes, rRes] = await Promise.all([
        modelApi.list({ page_size: 100 }),
        settingsApi.getComputeResources().catch(() => ({ data: { resources: [] } })),
      ]);
      setModels(mRes.data.items ?? []);
      setComputeResources(rRes.data.resources ?? []);
    } catch {
      setModels([]);
    }
  };

  const handleDeploy = async (values: any) => {
    setDeploying(true);
    try {
      await modelApi.deploy(values.model_name, values.version, 'kserve', {
        runtime_type: values.runtime_type,
        gpu: values.gpu || '0',
        domain: values.domain || undefined,
      });
      // Async: only the conversion + rollout task was enqueued.
      message.success(t('pages.models.deployStarted'));
      setDeployOpen(false);
      form.resetFields();
      load();
    } catch (e: any) {
      console.error(e);
      message.error(e?.response?.data?.message || t('pages.models.deployFailed'));
    } finally {
      setDeploying(false);
    }
  };

  const handleUndeploy = async (name: string, namespace: string) => {
    // undeploy runs a KFP pipeline server-side and can take ~1-3 minutes; keep
    // the row button spinning so the user knows it is in progress (not stuck).
    setUndeployingName(name);
    const hide = message.loading(t('pages.models.undeploying') + ' ...', 0);
    try {
      await modelApi.undeployByName(name, namespace);
      hide();
      message.success(t('pages.models.undeployed'));
      load();
    } catch (e: any) {
      hide();
      message.error(e?.response?.data?.detail || e?.response?.data?.message || t('pages.models.deployFailed'));
    } finally {
      setUndeployingName(null);
    }
  };

  const columns = [
    {
      title: t('pages.deployments.status'),
      key: 'status',
      width: 100,
      render: (_: unknown, r: DeployedService) => (
        r.ready
          ? <Tag color="green">{t('pages.models.deployedReady')}</Tag>
          : r.state === 'terminating'
            ? <Tag color="blue">{t('pages.models.undeploying')}</Tag>
            : r.error
              ? <Tag color="red" title={r.error}>{t('pages.deployments.error')}</Tag>
              : <Tag color="orange">{t('pages.models.deploying')}</Tag>
      ),
    },
    { title: t('pages.deployments.service'), dataIndex: 'name', key: 'name', ellipsis: true },
    { title: t('pages.deployments.namespace'), dataIndex: 'namespace', key: 'namespace', width: 180 },
    {
      title: t('pages.models.endpoint'),
      key: 'endpoint',
      render: (_: unknown, r: DeployedService) => (
        <span style={{ fontSize: 12 }}>
          {r.domain ? `http://${r.domain}` : (r.url || '-')}
        </span>
      ),
    },
    {
      title: t('pages.deployments.runtime'),
      dataIndex: 'runtime',
      key: 'runtime',
      width: 110,
      render: (v: string) => v || '-',
    },
    {
      title: t('pages.deployments.node'),
      dataIndex: 'node',
      key: 'node',
      width: 140,
      render: (v: string) => v || '-',
    },
    {
      title: t('pages.deployments.error'),
      key: 'error',
      width: 120,
      render: (_: unknown, r: DeployedService) => (
        r.error
          ? <Button size="small" danger type="link" icon={<CloudServerOutlined />} onClick={() => setErrorDetail(r)}>{t('pages.deployments.viewError')}</Button>
          : <span style={{ color: '#ccc' }}>-</span>
      ),
    },
    {
      title: t('pages.deployments.createdAt'),
      dataIndex: 'age',
      key: 'age',
      width: 170,
      render: (v: string) => (v ? formatDateTime(v) : '-'),
    },
    {
      title: t('common.actions'),
      key: 'actions',
      width: 120,
      render: (_: unknown, r: DeployedService) => (
        <Space>
          <Popconfirm title={t('pages.models.undeploy')} onConfirm={() => handleUndeploy(r.name, r.namespace)}>
            <Button size="small" danger icon={<StopOutlined />} loading={undeployingName === r.name} disabled={!!undeployingName || r.state === 'terminating'}>{r.state === 'terminating' ? t('pages.models.undeploying') : t('pages.models.undeploy')}</Button>
          </Popconfirm>
        </Space>
      ),
    },
  ];

  return (
    <Card>
      <PageHeader
        title={t('pages.deployments.title')}
        subtitle={t('pages.deployments.subtitle')}
        actions={[
          <HelpButton module="models" />,
          <Button type="primary" icon={<PlusOutlined />} onClick={openDeploy}>{t('pages.deployments.newDeploy')}</Button>,
          <Button icon={<ReloadOutlined />} onClick={load}>{t('common.refresh')}</Button>,
        ]}
      />
      <FilterSearchBar
        filterPlaceholder={t('pages.deployments.filterStatus')}
        filterValue={statusFilter}
        onFilterChange={setStatusFilter}
        filterOptions={[
          { value: 'ready', label: t('pages.models.deployedReady') },
          { value: 'deploying', label: t('pages.models.deploying') },
          { value: 'error', label: t('pages.deployments.error') },
        ]}
        searchPlaceholder={t('pages.deployments.searchByName')}
        searchValue={searchInput}
        onSearchChange={setSearchInput}
      />
      <Table
        columns={columns}
        dataSource={filteredServices}
        rowKey="name"
        loading={loading}
        pagination={{ pageSize: 10, showSizeChanger: true, pageSizeOptions: [10, 20, 50, 100] }}
        scroll={{ x: 'max-content' }}
      />

      <Modal
        title={`${t('pages.deployments.viewError')} - ${errorDetail?.name || ''}`}
        open={!!errorDetail}
        onCancel={() => setErrorDetail(null)}
        footer={<Button onClick={() => setErrorDetail(null)}>{t('common.close')}</Button>}
      >
        {errorDetail && (
          <div>
            <p><strong>Namespace:</strong> {errorDetail.namespace}</p>
            <p><strong>Runtime:</strong> {errorDetail.runtime || '-'}</p>
            <p><strong>Node:</strong> {errorDetail.node || '-'}</p>
            <p><strong>Model URI:</strong> <span style={{ wordBreak: 'break-all', fontSize: 12 }}>{errorDetail.model_uri || '-'}</span></p>
            <p><strong>Image:</strong> <span style={{ wordBreak: 'break-all', fontSize: 12 }}>{errorDetail.image || '-'}</span></p>
            <Alert
              type="error"
              showIcon
              message={t('pages.deployments.error')}
              description={<pre style={{ whiteSpace: 'pre-wrap', fontSize: 12, margin: 0 }}>{errorDetail.error || 'Unknown error'}</pre>}
            />
          </div>
        )}
      </Modal>

      <Modal title={t('pages.deployments.newDeploy')} open={deployOpen} onCancel={() => setDeployOpen(false)} footer={null} destroyOnHidden>
        <Form form={form} key={deployOpen ? 'open' : 'closed'} onFinish={handleDeploy} layout="vertical">
          <Form.Item name="model_name" label={t('pages.models.name')} rules={[{ required: true }]}>
            <Select
              showSearch
              optionFilterProp="label"
              placeholder={t('pages.models.name')}
              options={models.map((m) => ({ value: m.name, label: `${m.name} (${m.task_type || '-'})` }))}
              onChange={() => form.setFieldValue('version', undefined)}
            />
          </Form.Item>
          <Form.Item name="version" label={t('pages.models.version')} rules={[{ required: true }]}>
            <Select
              placeholder={t('pages.models.version')}
              loading={loadingVersions}
              notFoundContent={loadingVersions ? t('common.loading') : t('pages.models.version')}
              options={deployVersions.map((v) => ({ value: v, label: v }))}
            />
          </Form.Item>
          <Form.Item name="runtime_type" label={t('pages.models.runtime')} rules={[{ required: true }]}>
            <Select
              placeholder={t('pages.models.runtime')}
              options={runtimes.map((r) => ({
                value: r.name,
                label: `${r.name.replace(/^kserve-/, '')}${r.model_type ? ` (${r.model_type})` : ''}`,
              }))}
            />
          </Form.Item>
          <Form.Item name="gpu" label={t('pages.models.gpu')} initialValue="cpu">
            <Select
              options={[
                { value: 'cpu', label: 'CPU' },
                ...computeResources.filter(r => r.device !== 'cpu').map(r => ({
                  value: r.device,
                  label: `${r.name} · ${r.count} ${r.unit}${r.gpu_model ? ` (${r.gpu_model})` : ''}`,
                })),
              ]}
            />
          </Form.Item>
          <Form.Item name="domain" label={`${t('pages.models.domain')} (${t('pages.training.recommended')})`} extra={t('pages.models.domainHint')}>
            <Input placeholder={t('pages.deployments.namePlaceholder')} />
          </Form.Item>
          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit" icon={<CloudServerOutlined />} loading={deploying}>{t('pages.models.deployToKfp')}</Button>
              <Button onClick={() => setDeployOpen(false)}>{t('common.cancel')}</Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>
    </Card>
  );
};

export default Deployments;
