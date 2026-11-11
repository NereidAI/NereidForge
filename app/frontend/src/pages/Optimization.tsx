// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

import React, { useEffect, useState, useCallback } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { Card, Table, Button, Space, Modal, Form, Input, InputNumber, Select, Spin, Alert, Tooltip, Tag, Descriptions, Drawer, message, Switch, Progress } from 'antd';
import { DeleteOutlined, PlayCircleOutlined, StopOutlined, DownloadOutlined, WarningOutlined } from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import { optimizationApi, modelApi, datasetApi, settingsApi } from '@api/endpoints';
import { apiClient } from '@api/client';
import { PageHeader } from '@components/PageHeader';
import { HelpButton } from '@components/HelpDrawer';
import { StatusBadge, statusLabelKey, useJobStatusOptions } from '@components/StatusBadge';
import { useServerTable } from '@hooks/useServerTable';
import { useDebounce } from '@hooks/useDebounce';
import FilterSearchBar from '@components/FilterSearchBar';
import { useAuthStore } from '@store/authStore';
import type { OptimizationJob, GateReport, Model, Dataset, DatasetVersion } from '@nftypes/api';
import { formatDateTime } from '../utils/format';

// Quantization methods available per vendor; fp16 is meaningless on CPU edge
// (no fp16 arithmetic), so it is offered for INTEL(OpenVINO)/NVIDIA only.
const QUANT_METHODS = [
  { value: 'graph', label: 'Graph Opt (ORT)' },
  { value: 'int8-dynamic', label: 'INT8 Dynamic' },
  { value: 'int8-static', label: 'INT8 Static (calibrated)' },
  { value: 'fp16', label: 'FP16' },
];

// Display label for a vendor runtime (normalized backend value -> label).
const VENDOR_LABELS: Record<string, string> = {
  intel: 'INTEL',
  nvidia: 'NVIDIA',
  amd: 'AMD',
  huawei: 'HUAWEI',
  openvino: 'INTEL',
  tensorrt: 'NVIDIA',
  'onnx-rocm': 'AMD',
  ascend: 'HUAWEI',
  cpu: 'CPU',
};
const vendorLabel = (v: string): string => VENDOR_LABELS[v] || v || 'CPU';

// Worker-owned milestone ids are translated at the UI boundary. Unknown or
// legacy jobs keep the original percentage-only rendering.
const OPTIMIZATION_STAGE_KEYS: Record<string, string> = {
  baseline: 'pages.optimization.stageBaseline',
  optimizing: 'pages.optimization.stageOptimizing',
  validating: 'pages.optimization.stageValidating',
  finalizing: 'pages.optimization.stageFinalizing',
};
// optimizationStageKey maps a worker milestone id to its translation key.
const optimizationStageKey = (job: OptimizationJob): string => (
  OPTIMIZATION_STAGE_KEYS[String(job.metrics?.pipeline_stage || '')] || ''
);

// Vendor-first hardware targets. Availability is probed from the cluster's
// node GPU/NPU resources; unavailable vendors are greyed out, CPU is always
// selectable.
const VENDORS = [
  { value: 'cpu', label: 'CPU (ONNX Runtime)', quants: ['graph', 'int8-dynamic', 'int8-static'] },
  { value: 'intel', label: 'INTEL (OpenVINO)', quants: ['graph', 'fp16', 'int8-dynamic', 'int8-static'] },
  { value: 'nvidia', label: 'NVIDIA (TensorRT)', quants: ['fp16'] },
  { value: 'amd', label: 'AMD (ROCm)', quants: ['graph', 'fp16', 'int8-dynamic', 'int8-static'] },
  { value: 'huawei', label: 'HUAWEI (Ascend)', quants: ['graph', 'fp16', 'int8-dynamic', 'int8-static'] },
];

export const Optimization: React.FC = () => {
  const { t } = useTranslation();
  const navigate = useNavigate();
  // Server-side status filter plus debounced job-name search.
  const [statusFilter, setStatusFilter] = useState('');
  const [searchInput, setSearchInput] = useState('');
  const search = useDebounce(searchInput.trim(), 300);
  const statusOptions = useJobStatusOptions();
  // Server-side paginated job list with polling on the visible page.
  const jobsTable = useServerTable<OptimizationJob, { status?: string; search?: string }>({
    list: (params) => optimizationApi.list(params),
    filters: {
      status: statusFilter || undefined,
      search: search || undefined,
    },
    defaultPageSize: 10,
    pollMs: 5000,
  });
  const { reload: fetchJobs } = jobsTable;
  const [modalOpen, setModalOpen] = useState(false);
  const [models, setModels] = useState<Model[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [versions, setVersions] = useState<{ version: string; file_path?: string; metadata?: Record<string, unknown> }[]>([]);
  const [datasetVersions, setDatasetVersions] = useState<DatasetVersion[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [searchMode, setSearchMode] = useState(false);
  const [hardwareCaps, setHardwareCaps] = useState<Record<string, { available: boolean; detail?: string }>>({});
  const [form] = Form.useForm();
  const watchedDevices = Form.useWatch('target_devices', form) || [];
  // Hardware-aware vendor options: unavailable vendors are greyed out, CPU
  // stays selectable.
  const vendorOptions = VENDORS.map((v) => {
    const cap = v.value === 'cpu' ? { available: true, detail: '' } : hardwareCaps[v.value];
    const avail = cap?.available ?? false;
    const detail = cap?.detail || '';
    return {
      ...v,
      disabled: !avail,
      label: detail && !avail ? `${v.label} — ${detail}` : v.label,
      title: detail || undefined,
    };
  });
  // Quantization options filtered by the selected vendors. An empty selection
  // shows all methods.
  const quantOptions = watchedDevices.length > 0
    ? VENDORS.filter((d) => watchedDevices.includes(d.value))
        .flatMap((d) => d.quants)
        .filter((v, i, a) => a.indexOf(v) === i)
        .map((v) => QUANT_METHODS.find((q) => q.value === v)!).filter(Boolean)
    : QUANT_METHODS;

  const openCreate = async () => {
    form.resetFields();
    // Prefill gate thresholds with the spec defaults
    form.setFieldsValue({ max_map_drop_pct: 1, p99_latency_ms: 50, min_throughput_qps: 100, calib_n: 200 });
    setModalOpen(true);
    try {
      const [m, d, h] = await Promise.all([
        modelApi.list({ page_size: 100 }),
        datasetApi.list({ page_size: 100 }),
        settingsApi.getHardwareCapabilities().catch(() => ({ data: {} as Record<string, { available: boolean }> })),
      ]);
      setModels(m.data.items);
      setDatasets(d.data.items);
      setHardwareCaps(h.data);
    } catch (error) {
      console.error('Failed to load models/datasets/hardware:', error);
    }
  };

  const onModelChange = async (modelId: string) => {
    form.setFieldValue('source_version', undefined);
    setVersions([]);
    try {
      const res = await modelApi.getVersions(modelId, { page_size: 50 });
      setVersions(res.data.items);
    } catch (error) {
      console.error('Failed to load versions:', error);
    }
  };

  const onDatasetChange = async (datasetId: string) => {
    form.setFieldValue('dataset_version', undefined);
    setDatasetVersions([]);
    try {
      const res = await datasetApi.getVersions(datasetId, { page_size: 50 });
      setDatasetVersions(res.data.items);
    } catch (error) {
      console.error('Failed to load dataset versions:', error);
    }
  };

  const columns = [
    {
      title: t('pages.optimization.jobName'),
      dataIndex: 'name',
      key: 'name',
      render: (name: string, record: OptimizationJob) => (
        <Button type="link" style={{ padding: 0 }} onClick={() => navigate(`/optimization/${record.id}`)}>{name}</Button>
      ),
    },
    {
      title: t('pages.optimization.quantization'),
      key: 'quant',
      render: (_: unknown, record: OptimizationJob) => {
        const p = record.params || {};
        return `${vendorLabel(p.target_device)} · ${p.quant_method || '-'}${p.dtype ? ` (${p.dtype})` : ''}`;
      },
    },
    {
      title: t('pages.optimization.gate'),
      key: 'gate',
      render: (_: unknown, record: OptimizationJob) => {
        const gr = record.results?.gate_result;
        if (!gr) return '-';
        const color = gr === 'PASS' ? 'green' : gr === 'YELLOW' ? 'orange' : 'red';
        return <Tag color={color}>{gr}</Tag>;
      },
    },
    {
      title: t('pages.training.status'),
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => <StatusBadge status={status as never} />,
    },
    {
      title: t('common.progress'),
      dataIndex: 'progress',
      key: 'progress',
      width: 220,
      render: (progress: number | undefined, record: OptimizationJob) => {
        const pct = typeof progress === 'number' ? Math.max(0, Math.min(100, progress)) : 0;
        const status = record.status;
        const progStatus: 'success' | 'exception' | 'active' | 'normal' =
          status === 'completed' ? 'success' :
          status === 'failed' ? 'exception' :
          status === 'running' ? 'active' : 'normal';
        const idle = status === 'pending' || status === 'queued' || status === 'cancelled';
        const stageKey = status === 'running' ? optimizationStageKey(record) : '';
        return (
          <div>
            <Progress
              percent={idle ? 0 : pct}
              size="small"
              showInfo
              status={progStatus}
              strokeColor={idle ? '#d9d9d9' : undefined}
            />
            {stageKey && (
              <div
                style={{ color: '#666', fontSize: 12, lineHeight: 1.4 }}
                title={t('pages.optimization.milestoneHint')}
              >
                {t(stageKey)}
                {' · '}
                {t('pages.optimization.milestoneCount', {
                  completed: Number(record.metrics?.completed_stages || 0),
                  total: Number(record.metrics?.total_stages || 4),
                })}
              </div>
            )}
          </div>
        );
      },
    },
    {
      title: t('pages.optimization.sourceModel'),
      key: 'srcModel',
      render: (_: unknown, record: OptimizationJob) => {
        const p = record.params || {};
        const mid = (p.source_model_id || p.model_id || '') as string;
        const ver = (p.source_version || '') as string;
        return (
          <span style={{ fontSize: 12 }}>
            {mid ? mid.slice(0, 12) + '…' : '-'}
            {ver ? <span style={{ color: '#888' }}> · {ver}</span> : null}
          </span>
        );
      },
    },
    {
      title: t('pages.optimization.dataset'),
      key: 'srcDataset',
      render: (_: unknown, record: OptimizationJob) => {
        const p = record.params || {};
        const did = (p.dataset_id || '') as string;
        const dv = p.dataset_version;
        return (
          <span style={{ fontSize: 12 }}>
            {did ? did.slice(0, 12) + '…' : '-'}
            {dv ? <span style={{ color: '#888' }}> · v{dv}</span> : null}
          </span>
        );
      },
    },
    {
      title: 'mAP@50',
      key: 'map50',
      render: (_: unknown, record: OptimizationJob) => {
        const v = record.metrics?.mAP50;
        return typeof v === 'number' ? Number(v).toFixed(3) : '-';
      },
    },
    {
      title: t('pages.optimization.avgLatency'),
      key: 'latency',
      render: (_: unknown, record: OptimizationJob) => {
        const m = record.metrics || {};
        const bm = (m.benchmark || {}) as Record<string, number>;
        const ms = bm.avg_ms ?? m.avg_latency_ms;
        return ms ? `${Number(ms).toFixed(1)} ms` : '-';
      },
    },
    {
      title: t('pages.optimization.size'),
      key: 'size',
      render: (_: unknown, record: OptimizationJob) => {
        const m = record.metrics || {};
        return m.source_size_bytes && m.optimized_size_bytes
          ? `${formatBytes(Number(m.source_size_bytes))} → ${formatBytes(Number(m.optimized_size_bytes))}`
          : '-';
      },
    },
    {
      title: t('pages.optimization.speedup'),
      key: 'speedup',
      render: (_: unknown, record: OptimizationJob) => {
        const ratio = record.metrics?.speedup_ratio;
        return ratio ? `${Number(ratio).toFixed(2)}x` : '-';
      },
    },
    {
      title: t('common.errorMessage'),
      key: 'err',
      render: (_: unknown, record: OptimizationJob) => {
        const e = record.error_message;
        if (!e || record.status === 'completed') return '-';
        return (
          <Tooltip title={e}>
            <span style={{ color: '#cf1322', fontSize: 12, maxWidth: 180, display: 'inline-block', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {e}
            </span>
          </Tooltip>
        );
      },
    },
    {
      title: t('common.createdAt'),
      dataIndex: 'created_at',
      key: 'created_at',
      render: (date: string) => formatDateTime(date),
    },
    {
      title: t('common.actions'),
      key: 'actions',
      render: (_: unknown, record: OptimizationJob) => (
        <Space size={2}>
          <Tooltip title={record.status === 'running' ? t('common.stop') : t('common.start')}>
            <Button
              type="text"
              size="small"
              icon={record.status === 'running' ? <StopOutlined /> : <PlayCircleOutlined />}
              onClick={async () => {
                try {
                  if (record.status === 'running') await optimizationApi.stop(record.id);
                  else await optimizationApi.start(record.id);
                  fetchJobs();
                } catch (e) {
                  message.error(String((e as Error).message || e));
                }
              }}
            />
          </Tooltip>
          <Tooltip title={t('common.download')}>
            <Button
              type="text"
              size="small"
              icon={<DownloadOutlined />}
              disabled={record.status !== 'completed' && record.status !== 'failed'}
              onClick={() => apiClient.downloadFile(optimizationApi.downloadUrl(record.id))}
            />
          </Tooltip>
          <Tooltip title={t('common.delete')}>
            <Button type="text" size="small" danger icon={<DeleteOutlined />} onClick={async () => {
              if (window.confirm(t('common.confirmDelete') || `Delete ${record.name}?`)) {
                await optimizationApi.delete(record.id);
                fetchJobs();
              }
            }} />
          </Tooltip>
        </Space>
      ),
    },
  ];

  const handleSubmit = async (values: {
    name: string;
    description?: string;
    source_model_id: string;
    source_version?: string;
    dataset_id?: string;
    dataset_version?: number;
    target_devices: string[];
    quant_methods: string[];
    calib_n?: number;
    max_map_drop_pct?: number;
    p99_latency_ms?: number;
    min_throughput_qps?: number;
  }) => {
    // Targets = device x method cross product; each becomes one DAG branch.
    // In search mode the targets are left empty and Katib explores them.
    const targets: Array<{ target_device: string; quant_method: string; dtype?: string; calib_n?: number }> = [];
    if (!searchMode) {
      for (const dev of values.target_devices) {
        for (const m of values.quant_methods) {
          targets.push({
            target_device: dev,
            quant_method: m,
            dtype: m === 'fp16' ? 'fp16' : m.startsWith('int8') ? 'int8' : '',
            calib_n: values.calib_n,
          });
        }
      }
    }
    const payload = {
      name: values.name,
      description: values.description,
      source_model_id: values.source_model_id,
      source_version: values.source_version,
      dataset_id: values.dataset_id,
      dataset_version: values.dataset_version,
      targets,
      params: {
        task_type: 'detection',
        max_map_drop_pct: values.max_map_drop_pct,
        p99_latency_ms: values.p99_latency_ms,
        min_throughput_qps: values.min_throughput_qps,
        ...(searchMode ? { search_mode: 'true' } : {}),
      },
    };
    setSubmitting(true);
    try {
      await optimizationApi.create(payload);
      message.success(t('pages.optimization.batchCreated'));
      setModalOpen(false);
      fetchJobs();
    } catch (error) {
      message.error(String((error as Error).message || error));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div>
      <PageHeader
        title={t('pages.optimization.title')}
        onCreate={openCreate}
        onRefresh={fetchJobs}
        actions={[<HelpButton module="optimization" />]}
      />
      <Card>
        <FilterSearchBar
          filterPlaceholder={t('pages.optimization.filterStatus')}
          filterValue={statusFilter}
          onFilterChange={setStatusFilter}
          filterOptions={statusOptions}
          searchPlaceholder={t('pages.optimization.searchPlaceholder')}
          searchValue={searchInput}
          onSearchChange={setSearchInput}
        />
        <Table
          {...jobsTable.tableProps}
          columns={columns}
          scroll={{ x: 'max-content' }}
        />
      </Card>

      <Modal
        title={t('pages.optimization.createTitle')}
        open={modalOpen}
        onOk={() => form.validateFields().then(handleSubmit)}
        onCancel={() => setModalOpen(false)}
        confirmLoading={submitting}
        width={760}
      >
        <Form form={form} layout="vertical">
          <Form.Item name="name" label={t('pages.optimization.jobName')} rules={[{ required: true, message: t('common.required') }]}>
            <Input placeholder={t('pages.optimization.jobName')} />
          </Form.Item>
          <Form.Item name="description" label={t('pages.datasets.description')}>
            <Input.TextArea rows={2} />
          </Form.Item>
          <Space size={16} style={{ display: 'flex' }}>
            <Form.Item name="source_model_id" label={t('pages.optimization.sourceModel')} rules={[{ required: true, message: t('common.required') }]} style={{ width: 320 }}>
              <Select
                showSearch
                optionFilterProp="label"
                placeholder={t('pages.optimization.sourceModel')}
                onChange={onModelChange}
                options={models.map((m) => ({ value: m.id, label: `${m.name} (${m.task_type || ''})` }))}
              />
            </Form.Item>
            <Form.Item name="source_version" label={t('pages.optimization.sourceVersion')} style={{ width: 300 }}>
              <Select
                allowClear
                placeholder={t('pages.optimization.sourceVersionLatest')}
                options={versions.map((v) => {
                  const fp = (v.file_path || '').toLowerCase();
                  const fw = String((v.metadata as Record<string, unknown> | undefined)?.framework ?? '');
                  const isIr = fp.endsWith('.xml') || fw === 'openvino';
                  const capabilities = v.metadata?.artifact_capabilities;
                  const capabilityDenied = Array.isArray(capabilities) && !capabilities.includes('optimize');
                  const disabled = isIr || capabilityDenied;
                  const reason = isIr
                    ? t('pages.optimization.irSourceDisabled')
                    : t('pages.optimization.capabilitySourceDisabled');
                  const hint = isIr
                    ? t('pages.optimization.irSourceHint')
                    : t('pages.optimization.capabilitySourceHint');
                  return {
                    value: v.version,
                    label: disabled ? `${v.version} — ${reason}` : v.version,
                    disabled,
                    title: disabled ? hint : undefined,
                  };
                })}
              />
            </Form.Item>
          </Space>
          <Space size={16} style={{ display: 'flex' }}>
            <Form.Item name="dataset_id" label={t('pages.evaluation.dataset')} style={{ width: 320 }} extra={t('pages.optimization.datasetHint')} rules={[{ required: true, message: t('common.required') }]}>
              <Select
                showSearch
                optionFilterProp="label"
                placeholder={t('pages.evaluation.dataset')}
                onChange={onDatasetChange}
                options={datasets.map((d) => ({ value: d.id, label: d.name }))}
              />
            </Form.Item>
            <Form.Item name="dataset_version" label={t('pages.datasets.version')} style={{ width: 160 }} rules={[{ required: true, message: t('common.required') }]}>
              <Select
                placeholder={t('pages.datasets.version')}
                disabled={!datasetVersions.length}
                notFoundContent={t('pages.optimization.selectDatasetFirst')}
                options={datasetVersions.map((v) => ({ value: v.version, label: `v${v.version}${v.is_draft ? ' (draft)' : ''}` }))}
              />
            </Form.Item>
          </Space>
          <Form.Item label={t('pages.optimization.searchMode')} extra={t('pages.optimization.searchModeHint')}>
            <Switch checked={searchMode} onChange={setSearchMode} />
          </Form.Item>
          {!searchMode && (
            <>
              <Form.Item name="target_devices" label={t('pages.optimization.targetDevice')} rules={[{ required: true, message: t('common.required') }]} extra={t('pages.optimization.multiTargetHint')}>
                <Select mode="multiple" options={vendorOptions} />
              </Form.Item>
              <Form.Item name="quant_methods" label={t('pages.optimization.quantization')} rules={[{ required: true, message: t('common.required') }]}>
                <Select mode="multiple" options={quantOptions} />
              </Form.Item>
            </>
          )}
          <Form.Item name="calib_n" label={t('pages.optimization.calibN')} extra={t('pages.optimization.calibNHint')}>
            <InputNumber min={10} max={1000} style={{ width: 160 }} />
          </Form.Item>
          <Card size="small" title={t('pages.optimization.gateThresholds')} style={{ marginBottom: 8 }}>
            <Space size={16} style={{ display: 'flex' }}>
              <Form.Item name="max_map_drop_pct" label={t('pages.optimization.gateMapDrop')} style={{ marginBottom: 0 }}>
                <InputNumber min={0} step={0.5} addonAfter="%" style={{ width: 180 }} />
              </Form.Item>
              <Form.Item name="p99_latency_ms" label={t('pages.optimization.gateP99')} style={{ marginBottom: 0 }}>
                <InputNumber min={0} addonAfter="ms" style={{ width: 180 }} />
              </Form.Item>
              <Form.Item name="min_throughput_qps" label={t('pages.optimization.gateQps')} style={{ marginBottom: 0 }}>
                <InputNumber min={0} style={{ width: 140 }} />
              </Form.Item>
            </Space>
          </Card>
        </Form>
      </Modal>
    </div>
  );
};

// Optimization Detail Page: metrics + gate report + override + download
export const OptimizationDetail: React.FC = () => {
  const { t } = useTranslation();
  const { id: jobId } = useParams<{ id: string }>();
  const [job, setJob] = useState<OptimizationJob | null>(null);
  const [logs, setLogs] = useState('');
  const [loading, setLoading] = useState(true);
  const [overrideOpen, setOverrideOpen] = useState(false);
  const [reason, setReason] = useState('');
  const [perClassOpen, setPerClassOpen] = useState(false);
  const { user, isAdmin } = useAuthStore();
  const logsRef = React.useRef<HTMLPreElement>(null);

  const fetchJob = useCallback(async () => {
    try {
      if (!jobId) return;
      const [response, logsResponse] = await Promise.all([
        optimizationApi.get(jobId),
        optimizationApi.getLogs(jobId, { tail: 2000 }).catch(() => ({ data: { logs: '' } })),
      ]);
      setJob(response.data);
      setLogs(logsResponse.data?.logs || '');
    } catch (error) {
      console.error('Failed to fetch job:', error);
    } finally {
      setLoading(false);
    }
  }, [jobId]);

  useEffect(() => {
    fetchJob();
    const timer = setInterval(fetchJob, 5000);
    return () => clearInterval(timer);
  }, [fetchJob]);

  // Follow new executor output without pulling users away from earlier logs
  // they deliberately scrolled back to inspect.
  useEffect(() => {
    const el = logsRef.current;
    if (!el) return;
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
    if (nearBottom) el.scrollTop = el.scrollHeight;
  }, [logs]);

  if (loading) return <Spin size="large" style={{ display: 'block', margin: '100px auto' }} />;
  if (!job) return <Alert message="Job not found" type="error" />;

  const report: GateReport | undefined = job.results?.report;
  const stageKey = job.status === 'running' ? optimizationStageKey(job) : '';
  const milestoneCount = t('pages.optimization.milestoneCount', {
    completed: Number(job.metrics?.completed_stages || 0),
    total: Number(job.metrics?.total_stages || 4),
  });
  const metrics = job.metrics || {};
  const benchmark = (report?.metrics?.benchmark || metrics.benchmark || {}) as Record<string, number>;
  const perClass = (report?.metrics?.per_class || {}) as Record<string, { ap50?: number; precision?: number; recall?: number; gt?: number }>;
  const worstClasses = Object.entries(perClass)
    .sort((a, b) => (a[1].ap50 ?? 1) - (b[1].ap50 ?? 1))
    .slice(0, 5);
  const gateFailed = report?.gate_result === 'FAIL';
  const canOverride = gateFailed && job.status === 'failed' &&
    (isAdmin() || job.created_by === user?.username);
  const overridden = !!job.results?.gate_override;

  const submitOverride = async () => {
    if (!reason.trim()) {
      message.warning(t('pages.optimization.overrideReasonRequired'));
      return;
    }
    try {
      await optimizationApi.overrideGate(job.id, reason.trim());
      message.success(t('pages.optimization.overrideDone'));
      setOverrideOpen(false);
      setReason('');
      fetchJob();
    } catch (error) {
      message.error(String((error as Error).message || error));
    }
  };

  return (
    <div>
      <PageHeader
        title={job.name}
        subtitle={`${t('pages.training.status')}: ${t(statusLabelKey(job.status))} • ${stageKey
          ? `${t(stageKey)} • ${milestoneCount}`
          : `${t('common.progress')}: ${job.progress}%`}`}
        onRefresh={fetchJob}
      />
      <Card style={{ marginBottom: 16 }}>
        <Descriptions size="small" column={4} bordered>
          <Descriptions.Item label={t('pages.optimization.sourceModel')}>{(job.params?.source_model_id as string) || '-'}</Descriptions.Item>
          <Descriptions.Item label={t('pages.optimization.sourceVersion')}>{(job.params?.source_version as string) || t('pages.optimization.sourceVersionLatest')}</Descriptions.Item>
          <Descriptions.Item label={t('pages.optimization.targetDevice')}>{vendorLabel(job.params?.target_device as string)}</Descriptions.Item>
          <Descriptions.Item label={t('pages.optimization.quantization')}>{(job.params?.quant_method as string) || '-'}</Descriptions.Item>
          <Descriptions.Item label={t('pages.optimization.size')}>
            {metrics.source_size_bytes && metrics.optimized_size_bytes
              ? `${formatBytes(Number(metrics.source_size_bytes))} → ${formatBytes(Number(metrics.optimized_size_bytes))}`
              : '-'}
          </Descriptions.Item>
          <Descriptions.Item label={t('pages.optimization.speedup')}>
            {metrics.speedup_ratio ? `${Number(metrics.speedup_ratio).toFixed(2)}x` : '-'}
          </Descriptions.Item>
          <Descriptions.Item label={t('pages.optimization.avgLatency')}>
            {benchmark.avg_ms ? `${benchmark.avg_ms} ms` : metrics.avg_latency_ms ? `${metrics.avg_latency_ms} ms` : '-'}
          </Descriptions.Item>
          <Descriptions.Item label={t('pages.optimization.throughput')}>
            {benchmark.qps ? `${benchmark.qps} req/s` : '-'}
          </Descriptions.Item>
          <Descriptions.Item label={t('pages.optimization.executionNodes')}>
            {Array.isArray(job.config?.execution_nodes) && (job.config?.execution_nodes as string[]).length > 0
              ? (job.config?.execution_nodes as string[]).join(', ')
              : '-'}
          </Descriptions.Item>
        </Descriptions>
        {job.error_message && job.status !== 'completed' && (
          <Alert
            message={gateFailed ? t('pages.optimization.gateFailedTitle') : t('common.error')}
            description={job.error_message}
            type={gateFailed ? 'warning' : 'error'}
            style={{ marginTop: 16 }}
          />
        )}
      </Card>

      <Card
        title={t('common.logs')}
        style={{ marginBottom: 16 }}
        extra={job.status === 'running' || job.status === 'pending' ? (
          <span style={{ color: '#52c41a', fontSize: 12 }}>● {t('pages.optimization.logsLive')}</span>
        ) : null}
      >
        <pre
          ref={logsRef}
          style={{ background: '#1e1e1e', color: '#d4d4d4', padding: 16, borderRadius: 4, maxHeight: 480, overflow: 'auto', fontSize: 12, lineHeight: 1.5, whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}
        >
          {logs || t('pages.optimization.noLogs')}
        </pre>
      </Card>

      {report?.gates && (
        <Card title={t('pages.optimization.gateReport')} style={{ marginBottom: 16 }}
          extra={report.gate_result ? (
            <Tag color={report.gate_result === 'PASS' ? 'green' : report.gate_result === 'YELLOW' ? 'orange' : 'red'}>
              {report.gate_result}
            </Tag>
          ) : null}>
          <Table
            size="small"
            rowKey={(k) => String(k)}
            pagination={false}
            dataSource={Object.entries(report.gates).map(([k, v]) => ({ key: k, ...v }))}
            columns={[
              { title: t('pages.optimization.gateName'), dataIndex: 'key' },
              { title: t('pages.optimization.gateValue'), dataIndex: 'value' },
              { title: t('pages.optimization.gateThreshold'), dataIndex: 'threshold' },
              {
                title: t('pages.optimization.gateType'),
                dataIndex: 'hard',
                render: (hard: boolean) => <Tag>{hard ? t('pages.optimization.hardGate') : t('pages.optimization.softGate')}</Tag>,
              },
              {
                title: t('pages.optimization.gatePass'),
                dataIndex: 'passed',
                render: (passed: boolean, rec: any) => (
                  <Tag color={passed ? 'green' : rec.hard ? 'red' : 'orange'}>
                    {passed ? 'PASS' : rec.hard ? 'FAIL' : <><WarningOutlined /> YELLOW</>}
                  </Tag>
                ),
              },
            ]}
          />
          {report.delta && Object.keys(report.delta).length > 0 && (
            <Alert
              style={{ marginTop: 12 }}
              type="info"
              message={Object.entries(report.delta)
                .map(([k, v]) => `${k}: ${v > 0 ? '+' : ''}${v}`)
                .join('  ·  ')}
            />
          )}
          {canOverride && (
            <Button type="primary" danger style={{ marginTop: 12 }} onClick={() => setOverrideOpen(true)}>
              {t('pages.optimization.overrideButton')}
            </Button>
          )}
          {overridden && (
            <Alert
              style={{ marginTop: 12 }}
              type="success"
              message={t('pages.optimization.overrideRecorded')}
              description={`${job.results?.gate_override?.by}: ${job.results?.gate_override?.reason} → ${job.results?.gate_override?.registered_version}`}
            />
          )}
        </Card>
      )}

      <Card
        title={t('pages.optimization.perClass')}
        style={{ marginBottom: 16 }}
        extra={worstClasses.length > 0 ? <Button size="small" onClick={() => setPerClassOpen(true)}>{t('pages.optimization.showAll')}</Button> : null}
      >
        {worstClasses.length > 0 ? (
          <Table
            size="small"
            pagination={false}
            dataSource={worstClasses.map(([name, v]) => ({ key: name, name, ...v }))}
            columns={[
              { title: t('pages.optimization.className'), dataIndex: 'name' },
              { title: 'AP@50', dataIndex: 'ap50' },
              { title: 'P', dataIndex: 'precision' },
              { title: 'R', dataIndex: 'recall' },
              { title: 'GT', dataIndex: 'gt' },
            ]}
          />
        ) : (
          '-'
        )}
      </Card>

      <Card>
        <Space>
          <Button
            type="primary"
            icon={<DownloadOutlined />}
            disabled={!job.id}
            onClick={() => apiClient.downloadFile(optimizationApi.downloadUrl(job.id))}
          >
            {t('common.download')}
          </Button>
        </Space>
      </Card>

      <Modal
        title={t('pages.optimization.overrideTitle')}
        open={overrideOpen}
        onOk={submitOverride}
        onCancel={() => setOverrideOpen(false)}
        okText={t('pages.optimization.overrideConfirm')}
        okButtonProps={{ danger: true }}
      >
        <Alert type="warning" style={{ marginBottom: 12 }} message={t('pages.optimization.overrideWarning')} />
        <Input.TextArea rows={3} value={reason} onChange={(e) => setReason(e.target.value)} placeholder={t('pages.optimization.overrideReasonPlaceholder')} />
      </Modal>

      <Drawer title={t('pages.optimization.perClassAll')} open={perClassOpen} onClose={() => setPerClassOpen(false)} width={480}>
        <Table
          size="small"
          pagination={false}
          dataSource={Object.entries(perClass).map(([name, v]) => ({ key: name, name, ...v }))}
          columns={[
            { title: t('pages.optimization.className'), dataIndex: 'name' },
            { title: 'AP@50', dataIndex: 'ap50' },
            { title: 'P', dataIndex: 'precision' },
            { title: 'R', dataIndex: 'recall' },
          ]}
        />
      </Drawer>
    </div>
  );
};

const formatBytes = (bytes: number): string => {
  if (!bytes) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`;
};

export default Optimization;
