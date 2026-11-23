// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

import React, { useCallback, useEffect, useState } from 'react';
import { Card, Table, Modal, Form, Input, message, Tag, Tabs, Space, Popconfirm, Button, Select, Tooltip, Progress, Alert } from 'antd';
import { ReloadOutlined, PlusOutlined, PlayCircleOutlined, PauseCircleOutlined, DeleteOutlined, ExportOutlined } from '@ant-design/icons';

import { useTranslation } from 'react-i18next';
import { PageHeader } from '@components/PageHeader';
import apiClient from '@api/client';
import { useServerTable } from '@hooks/useServerTable';
import { useDebounce } from '@hooks/useDebounce';
import FilterSearchBar from '@components/FilterSearchBar';
import { useJobStatusOptions } from '@components/StatusBadge';
import { nfImage } from '../utils/images';

const JUPYTER_BASE_IMAGE = nfImage('prediction/jupyter-base', 'py3.12');

interface NotebookJob {
  id: string;
  name: string;
  status: string;
  progress: number;
  params: { notebook_uri?: string };
  results: Record<string, unknown>;
  error_message: string;
  created_at: string;
}

// Upload an .ipynb (staged to S3 notebooks/) then create+start a notebook job.

// ---- Interactive notebook servers (Kubeflow Notebook CRD workbench) --------

interface NbServer {
  name: string;
  image: string;
  cpu: string;
  memory: string;
  gpus: Record<string, string>;
  state: string;
  url_prefix: string;
  created_by: string;
}

const Servers: React.FC = () => {
  const { t } = useTranslation();
  const [servers, setServers] = useState<NbServer[]>([]);
  const [unavailable, setUnavailable] = useState(false);
  const [images, setImages] = useState<Array<{ full_ref: string; name: string }>>([]);
  const [open, setOpen] = useState(false);
  const [creating, setCreating] = useState(false);
  const [form] = Form.useForm();

  const fetchServers = useCallback(async () => {
    try {
      const res = await apiClient.get<{ items: NbServer[] }>('/notebook-servers');
      setServers(res.data.items || []);
      setUnavailable(false);
    } catch (e) {
      // 501 = deployment has no k8s access; keep the last list but show banner.
      const status = (e as { response?: { status?: number } })?.response?.status;
      if (status === 501) setUnavailable(true);
    }
  }, []);

  useEffect(() => {
    fetchServers();
    const t1 = setInterval(fetchServers, 5000);
    return () => clearInterval(t1);
  }, [fetchServers]);

  useEffect(() => {
    apiClient.get<{ items: Array<{ full_ref: string; name: string }> }>('/images?category=prediction')
      .then((r) => setImages(r.data.items || []))
      .catch(() => {});
  }, []);

  const openCreate = () => {
    form.resetFields();
    form.setFieldsValue({
      image: JUPYTER_BASE_IMAGE,
      cpu: '500m', memory: '1Gi', pvc_size: '10Gi', gpus: 0, gpu_type: 'nvidia',
    });
    setOpen(true);
  };

  const handleCreate = async () => {
    const values = await form.validateFields();
    setCreating(true);
    try {
      await apiClient.post('/notebook-servers', {
        name: values.name,
        image: values.image,
        cpu: values.cpu,
        memory: values.memory,
        gpus: Number(values.gpus || 0),
        gpu_type: values.gpu_type || 'nvidia',
        pvc_size: values.pvc_size,
        jupyter_token: values.jupyter_token || 'nereidforge',
      });
      message.success(t('pages.notebooks.serverCreated'));
      setOpen(false);
      fetchServers();
    } catch (e: unknown) {
      const msg = (e as { response?: { data?: { message?: string; detail?: string } } })?.response?.data;
      message.error(msg?.message || msg?.detail || t('common.error'));
    } finally {
      setCreating(false);
    }
  };

  const act = async (path: string, method: 'post' | 'delete' = 'post') => {
    try {
      await apiClient[method](`/notebook-servers/${path}`);
      fetchServers();
    } catch (e: unknown) {
      const msg = (e as { response?: { data?: { message?: string; detail?: string } } })?.response?.data;
      message.error(msg?.message || msg?.detail || t('common.error'));
    }
  };

  const cols = [
    { title: t('pages.notebooks.name'), dataIndex: 'name', key: 'name' },
    { title: 'Image', dataIndex: 'image', key: 'image', ellipsis: true, render: (v: string) => <span style={{ fontFamily: 'monospace', fontSize: 11 }}>{v}</span> },
    { title: 'CPU', dataIndex: 'cpu', key: 'cpu', width: 80 },
    { title: 'Memory', dataIndex: 'memory', key: 'memory', width: 90 },
    {
      title: 'GPU', key: 'gpu', width: 130,
      render: (_: unknown, r: NbServer) => Object.entries(r.gpus || {}).map(([k, v]) => (
        <Tag key={k}>{String(v)} {k.includes('nvidia') ? 'GPU' : k.split('/').pop()}</Tag>
      )),
    },
    {
      title: t('common.status'), dataIndex: 'state', key: 'state', width: 100,
      render: (s: string) => (
        <Tag color={s === 'running' ? 'green' : s === 'stopped' ? 'default' : 'blue'}>{s}</Tag>
      ),
    },
    {
      title: '', key: 'actions', width: 140,
      render: (_: unknown, r: NbServer) => (
        <Space size={4}>
          {r.state === 'running' && (
            <Tooltip title={t('pages.notebooks.open')}>
              {/* Bare prefix == jupyter base_url, no /api/v1 and no rewrite. */}
              <Button size="small" type="primary" icon={<ExportOutlined />} onClick={() => window.open(`${r.url_prefix}/lab`, '_blank')} />
            </Tooltip>
          )}
          {r.state === 'stopped'
            ? <Tooltip title={t('pages.notebooks.resume')}><Button size="small" icon={<PlayCircleOutlined />} onClick={() => act(`${r.name}/start`)} /></Tooltip>
            : <Tooltip title={t('pages.notebooks.pause')}><Button size="small" icon={<PauseCircleOutlined />} onClick={() => act(`${r.name}/stop`)} /></Tooltip>}
          <Popconfirm title={t('pages.notebooks.deleteConfirm')} onConfirm={() => act(r.name, 'delete')}>
            <Tooltip title={t('common.delete')}><Button size="small" danger icon={<DeleteOutlined />} /></Tooltip>
          </Popconfirm>
        </Space>
      ),
    },
  ];

  return (
    <Card
      title={t('pages.notebooks.serversTitle')}
      extra={<Button type="primary" icon={<PlusOutlined />} onClick={openCreate}>{t('pages.notebooks.createServer')}</Button>}
    >
      {unavailable && (
        <Alert type="warning" showIcon style={{ marginBottom: 12 }} message={t('pages.notebooks.serversUnavailable')} />
      )}
      <Table rowKey="name" size="small" columns={cols} dataSource={servers} pagination={{ pageSize: 10, showSizeChanger: true, pageSizeOptions: [10, 20, 50] }} />
      <Modal title={t('pages.notebooks.createServer')} open={open} onOk={handleCreate} confirmLoading={creating} onCancel={() => setOpen(false)} destroyOnHidden>
        <Form form={form} layout="vertical">
          <Form.Item name="name" label={t('pages.notebooks.name')} rules={[{ required: true, message: t('common.required') }]}>
            <Input placeholder="my-notebook" />
          </Form.Item>
          <Form.Item name="image" label="Image" rules={[{ required: true }]} extra={t('pages.notebooks.imageHint')}>
            <Select
              showSearch
              options={[
                ...images.map((im) => ({ value: im.full_ref, label: `${im.name} (${im.full_ref.split('/').pop()})` })),
                { value: JUPYTER_BASE_IMAGE, label: 'jupyter-base (py3.12)' },
              ]}
              filterOption={(input: string, option?: { label?: string; value?: string }) =>
                String(option?.label ?? '').toLowerCase().includes(input.toLowerCase())
                || String(option?.value ?? '').includes(input.toLowerCase())}
            />
          </Form.Item>
          <Space style={{ width: '100%' }} size="middle" wrap>
            <Form.Item name="cpu" label="CPU" style={{ width: 120 }}><Input placeholder="500m" /></Form.Item>
            <Form.Item name="memory" label="Memory" style={{ width: 120 }}><Input placeholder="1Gi" /></Form.Item>
            <Form.Item name="gpus" label="GPU" style={{ width: 90 }}><Input type="number" min={0} /></Form.Item>
            <Form.Item name="gpu_type" label={t('pages.notebooks.gpuType')} style={{ width: 170 }}>
              <Select
                options={[
                  { value: 'nvidia', label: t('pages.notebooks.gpuNvidia') },
                  { value: 'amd', label: t('pages.notebooks.gpuAmd') },
                  { value: 'intel', label: t('pages.notebooks.gpuIntel') },
                ]}
              />
            </Form.Item>
          </Space>
          <Form.Item name="pvc_size" label={t('pages.notebooks.pvcSize')}><Input placeholder="10Gi" style={{ width: 160 }} /></Form.Item>
          <Form.Item name="jupyter_token" label="Jupyter Token" extra={t('pages.notebooks.tokenHint')}>
            <Input.Password placeholder="nereidforge" />
          </Form.Item>
        </Form>
      </Modal>
    </Card>
  );
};

export const Notebooks: React.FC = () => {
  const { t } = useTranslation();
  // Server-side status filter plus debounced job-name search.
  const [statusFilter, setStatusFilter] = useState('');
  const [searchInput, setSearchInput] = useState('');
  const search = useDebounce(searchInput.trim(), 300);
  const statusOptions = useJobStatusOptions();
  // Server-side paginated notebook (Kale batch) jobs, polling the open page.
  const jobsTable = useServerTable<NotebookJob, { status?: string; search?: string }>({
    list: (params) => apiClient.get('/notebook/jobs', { params }),
    filters: {
      status: statusFilter || undefined,
      search: search || undefined,
    },
    defaultPageSize: 10,
    pollMs: 5000,
  });
  const { reload: fetchJobs } = jobsTable;
  const [modalOpen, setModalOpen] = useState(false);
  const [fileList, setFileList] = useState<File[]>([]);
  const [uploadProgress, setUploadProgress] = useState<number | null>(null);
  const [creating, setCreating] = useState(false);
  const [form] = Form.useForm();

  const handleSubmit = async () => {
    const values = await form.validateFields();
    if (fileList.length === 0) {
      message.warning(t('pages.notebooks.pickFile'));
      return;
    }
    setCreating(true);
    setUploadProgress(0);
    try {
      const f = fileList[0];
      const name = f.name.replace(/\.ipynb$/, '');
      const fd = new FormData();
      fd.append('files', f);
      await apiClient.post('/datasets/notebook-staging/upload', fd, {
        onUploadProgress: (e) => {
          if (e.total) setUploadProgress(Math.round((e.loaded / e.total) * 100));
        },
      });
      // dataset upload stages under s3://nereidforge/notebook-staging/v1/<file>
      const uri = `s3://nereidforge/notebook-staging/v1/${f.name}`;
      const created = await apiClient.post('/notebook/jobs', {
        name: values.name || name,
        params: { notebook_uri: uri },
      });
      await apiClient.post(`/notebook/jobs/${created.data.id}/start`);
      message.success(t('pages.notebooks.submitted'));
      setModalOpen(false);
      setFileList([]);
      setUploadProgress(null);
      form.resetFields();
      fetchJobs();
    } catch (e: unknown) {
      const msg = (e as { response?: { data?: { message?: string; detail?: string } } })?.response?.data;
      message.error(msg?.message || msg?.detail || t('common.error'));
    } finally {
      setCreating(false);
    }
  };

  const columns = [
    { title: t('pages.notebooks.name'), dataIndex: 'name', key: 'name' },
    {
      title: t('common.status'),
      dataIndex: 'status',
      key: 'status',
      render: (s: string) => <Tag color={s === 'completed' ? 'green' : s === 'failed' ? 'red' : 'blue'}>{s}</Tag>,
    },
    {
      title: t('pages.notebooks.progress'),
      dataIndex: 'progress',
      key: 'progress',
      width: 100,
      render: (p: number) => `${p || 0}%`,
    },
    {
      title: t('pages.notebooks.notebook'),
      dataIndex: ['params', 'notebook_uri'],
      key: 'uri',
      ellipsis: true,
      render: (u: string) => <span style={{ fontFamily: 'monospace', fontSize: 12 }}>{u}</span>,
    },
    {
      title: t('tables.createdAt'),
      dataIndex: 'created_at',
      key: 'created_at',
      width: 170,
      render: (v: string) => new Date(v).toLocaleString(),
    },
  ];

  return (
    <div>
      <PageHeader
        title={t('pages.notebooks.title')}
        subtitle={t('pages.notebooks.subtitle')}
      />
      <Tabs defaultActiveKey="servers" items={[
        { key: 'servers', label: t('pages.notebooks.tabServers'), children: <Servers /> },
        { key: 'jobs', label: t('pages.notebooks.tabJobs'), children: (
          <Card>
            <div style={{ marginBottom: 12, display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
              <Tooltip title={t('common.refresh')}><Button icon={<ReloadOutlined />} onClick={fetchJobs} /></Tooltip>
              <Button type="primary" icon={<PlusOutlined />} onClick={() => setModalOpen(true)}>{t('pages.notebooks.submitJob')}</Button>
            </div>
            <FilterSearchBar
              filterPlaceholder={t('pages.notebooks.filterStatus')}
              filterValue={statusFilter}
              onFilterChange={setStatusFilter}
              filterOptions={statusOptions}
              searchPlaceholder={t('pages.notebooks.searchPlaceholder')}
              searchValue={searchInput}
              onSearchChange={setSearchInput}
            />
            <Table {...jobsTable.tableProps} size="small" columns={columns} />
          </Card>
        ) },
      ]} />

      <Modal
        title={t('pages.notebooks.createTitle')}
        open={modalOpen}
        onOk={handleSubmit}
        confirmLoading={creating}
        onCancel={() => setModalOpen(false)}
        destroyOnHidden
      >
        <Form form={form} layout="vertical">
          <Form.Item name="name" label={t('pages.notebooks.jobName')}>
            <Input placeholder={t('pages.notebooks.jobNamePlaceholder')} />
          </Form.Item>
          <Form.Item label={t('pages.notebooks.file')} required extra={t('pages.notebooks.fileHint')}>
            <input
              type="file"
              accept=".ipynb"
              onChange={(e) => setFileList(e.target.files ? Array.from(e.target.files) : [])}
            />
            {uploadProgress !== null && uploadProgress > 0 && (
              <Progress percent={uploadProgress} status={uploadProgress >= 100 ? 'success' : 'active'} style={{ marginTop: 8 }} />
            )}
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

export default Notebooks;
