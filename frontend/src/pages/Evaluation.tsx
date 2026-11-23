// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

import React, { useEffect, useState, useCallback } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { Card, Table, Button, Space, Modal, Form, Input, Select, InputNumber, Spin, Alert, Descriptions, Tooltip, Tag, message } from 'antd';
import { DeleteOutlined, PlayCircleOutlined, StopOutlined, DownloadOutlined } from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import { evaluationApi, modelApi, datasetApi } from '@api/endpoints';
import { apiClient } from '@api/client';
import { PageHeader } from '@components/PageHeader';
import { HelpButton } from '@components/HelpDrawer';
import { StatusBadge, statusLabelKey, useJobStatusOptions } from '@components/StatusBadge';
import { useServerTable } from '@hooks/useServerTable';
import { useDebounce } from '@hooks/useDebounce';
import FilterSearchBar from '@components/FilterSearchBar';
import type { EvaluationJob, GateReport, Model, Dataset, DatasetVersion } from '@nftypes/api';
import { formatDateTime } from '../utils/format';

export const Evaluation: React.FC = () => {
  const { t } = useTranslation();
  const navigate = useNavigate();
  // Server-side status filter plus debounced job-name search.
  const [statusFilter, setStatusFilter] = useState('');
  const [searchInput, setSearchInput] = useState('');
  const search = useDebounce(searchInput.trim(), 300);
  const statusOptions = useJobStatusOptions();
  // Server-side paginated job list with polling on the visible page.
  const jobsTable = useServerTable<EvaluationJob, { status?: string; search?: string }>({
    list: (params) => evaluationApi.list(params),
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
  const [versions, setVersions] = useState<{ version: string }[]>([]);
  const [baselineVersions, setBaselineVersions] = useState<{ version: string }[]>([]);
  const [datasetVersions, setDatasetVersions] = useState<DatasetVersion[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [form] = Form.useForm();

  const selectedDataset = Form.useWatch('dataset_id', form);

  // Load dataset versions for the version dropdown once a dataset is chosen.
  useEffect(() => {
    if (!selectedDataset) {
      setDatasetVersions([]);
      form.setFieldValue('dataset_version', undefined);
      return;
    }
    datasetApi.getVersions(selectedDataset, { page_size: 100 }).then((res) => {
      setDatasetVersions(res.data.items || []);
    }).catch(() => setDatasetVersions([]));
  }, [selectedDataset, form]);

  const openCreate = async () => {
    form.resetFields();
    form.setFieldsValue({ task_type: 'detection', conf_threshold: 0.25 });
    setModalOpen(true);
    try {
      const [m, d] = await Promise.all([
        modelApi.list({ page_size: 100 }),
        datasetApi.list({ page_size: 100 }),
      ]);
      setModels(m.data.items);
      setDatasets(d.data.items);
    } catch (error) {
      console.error('Failed to load models/datasets:', error);
    }
  };

  const onModelChange = async (modelId: string) => {
    form.setFieldValue('model_version', undefined);
    setVersions([]);
    if (!modelId) return;
    try {
      const res = await modelApi.getVersions(modelId, { page_size: 50 });
      setVersions(res.data.items);
    } catch { /* keep empty */ }
  };

  const onBaselineChange = async (modelId: string) => {
    form.setFieldValue('baseline_model_version', undefined);
    setBaselineVersions([]);
    if (!modelId) return;
    try {
      const res = await modelApi.getVersions(modelId, { page_size: 50 });
      setBaselineVersions(res.data.items);
    } catch { /* keep empty */ }
  };

  const columns = [
    {
      title: t('pages.evaluation.jobName'),
      dataIndex: 'name',
      key: 'name',
      render: (name: string, record: EvaluationJob) => (
        <Button type="link" style={{ padding: 0 }} onClick={() => navigate(`/evaluation/${record.id}`)}>{name}</Button>
      ),
    },
    {
      title: t('pages.evaluation.model'),
      key: 'model',
      render: (_: unknown, r: EvaluationJob) => (r.params?.model_id as string) || '-',
    },
    {
      title: t('pages.evaluation.taskType'),
      key: 'task_type',
      render: (_: unknown, r: EvaluationJob) => (r.params?.task_type as string) || 'detection',
    },
    {
      title: 'mAP@50',
      key: 'map50',
      render: (_: unknown, r: EvaluationJob) => {
        const v = r.metrics?.mAP50 ?? r.results?.metrics?.mAP50;
        return v !== undefined ? Number(v).toFixed(4) : '-';
      },
    },
    {
      title: t('pages.training.status'),
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => <StatusBadge status={status as never} />,
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
      render: (_: unknown, record: EvaluationJob) => (
        <Space size={2}>
          <Tooltip title={record.status === 'running' ? t('common.stop') : t('common.start')}>
            <Button
              type="text"
              size="small"
              icon={record.status === 'running' ? <StopOutlined /> : <PlayCircleOutlined />}
              onClick={async () => {
                try {
                  if (record.status === 'running') await evaluationApi.stop(record.id);
                  else await evaluationApi.start(record.id);
                  fetchJobs();
                } catch (e) {
                  message.error(String((e as Error).message || e));
                }
              }}
            />
          </Tooltip>
          <Tooltip title={t('common.download')}>
            <Button type="text" size="small" icon={<DownloadOutlined />} disabled={record.status !== 'completed'} onClick={() => apiClient.downloadFile(evaluationApi.downloadUrl(record.id))} />
          </Tooltip>
          <Tooltip title={t('common.delete')}>
            <Button type="text" size="small" danger icon={<DeleteOutlined />} onClick={async () => {
              if (window.confirm(t('common.confirmDelete') || `Delete ${record.name}?`)) {
                await evaluationApi.delete(record.id);
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
    model_id: string;
    model_version?: string;
    baseline_model_id?: string;
    baseline_model_version?: string;
    dataset_id?: string;
    dataset_version?: number;
    task_type: string;
    conf_threshold?: number;
    max_images?: number;
  }) => {
    setSubmitting(true);
    try {
      await evaluationApi.create({
        name: values.name,
        description: values.description,
        model_id: values.model_id,
        model_version: values.model_version,
        baseline_model_id: values.baseline_model_id,
        baseline_model_version: values.baseline_model_version,
        dataset_id: values.dataset_id,
        dataset_version: values.dataset_version,
        params: {
          task_type: values.task_type,
          conf_threshold: values.conf_threshold,
          max_images: values.max_images,
        },
      });
      message.success(t('pages.evaluation.created'));
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
        title={t('pages.evaluation.title')}
        onCreate={openCreate}
        onRefresh={fetchJobs}
        actions={[<HelpButton module="evaluation" />]}
      />
      <Card>
        <FilterSearchBar
          filterPlaceholder={t('pages.evaluation.filterStatus')}
          filterValue={statusFilter}
          onFilterChange={setStatusFilter}
          filterOptions={statusOptions}
          searchPlaceholder={t('pages.evaluation.searchPlaceholder')}
          searchValue={searchInput}
          onSearchChange={setSearchInput}
        />
        <Table {...jobsTable.tableProps} columns={columns} scroll={{ x: 'max-content' }} />
      </Card>

      <Modal
        title={t('pages.evaluation.createTitle')}
        open={modalOpen}
        onOk={() => form.validateFields().then(handleSubmit)}
        onCancel={() => setModalOpen(false)}
        confirmLoading={submitting}
        width={760}
      >
        <Form form={form} layout="vertical">
          <Form.Item name="name" label={t('pages.evaluation.jobName')} rules={[{ required: true, message: t('common.required') }]}>
            <Input />
          </Form.Item>
          <Form.Item name="description" label={t('pages.datasets.description')}>
            <Input.TextArea rows={2} />
          </Form.Item>
          <Space size={16} style={{ display: 'flex' }}>
            <Form.Item name="model_id" label={t('pages.evaluation.model')} rules={[{ required: true, message: t('common.required') }]} style={{ width: 320 }}>
              <Select showSearch optionFilterProp="label" placeholder={t('pages.evaluation.model')} onChange={onModelChange} options={models.map((m) => ({ value: m.id, label: m.name }))} />
            </Form.Item>
            <Form.Item name="model_version" label={t('pages.evaluation.modelVersion')} style={{ width: 300 }}>
              <Select allowClear placeholder={t('pages.optimization.sourceVersionLatest')} options={versions.map((v) => ({ value: v.version, label: v.version }))} />
            </Form.Item>
          </Space>
          <Space size={16} style={{ display: 'flex' }}>
            <Form.Item name="baseline_model_id" label={t('pages.evaluation.baselineModel')} extra={t('pages.evaluation.baselineHint')} style={{ width: 320 }}>
              <Select allowClear showSearch optionFilterProp="label" placeholder={t('pages.evaluation.baselineNone')} onChange={onBaselineChange} options={models.map((m) => ({ value: m.id, label: m.name }))} />
            </Form.Item>
            <Form.Item name="baseline_model_version" label={t('pages.evaluation.baselineVersion')} style={{ width: 300 }}>
              <Select allowClear placeholder={t('pages.optimization.sourceVersionLatest')} options={baselineVersions.map((v) => ({ value: v.version, label: v.version }))} />
            </Form.Item>
          </Space>
          <Space size={16} style={{ display: 'flex' }}>
            <Form.Item name="dataset_id" label={t('pages.evaluation.dataset')} rules={[{ required: true, message: t('common.required') }]} style={{ width: 320 }}>
              <Select showSearch optionFilterProp="label" placeholder={t('pages.evaluation.dataset')} options={datasets.map((d) => ({ value: d.id, label: d.name }))} />
            </Form.Item>
            <Form.Item name="dataset_version" label={t('pages.datasets.version')} rules={[{ required: true, message: t('common.required') }]} style={{ width: 200 }}>
              <Select
                placeholder={t('pages.optimization.sourceVersionLatest')}
                options={datasetVersions.map((v) => ({ value: v.version, label: `v${v.version}${v.description ? ` · ${v.description}` : ''}` }))}
              />
            </Form.Item>
          </Space>
          <Space size={16} style={{ display: 'flex' }}>
            <Form.Item name="task_type" label={t('pages.evaluation.taskType')} rules={[{ required: true }]} style={{ width: 200 }}>
              <Select options={[{ value: 'detection', label: t('pages.evaluation.detection') }, { value: 'classification', label: t('pages.evaluation.classification') }]} />
            </Form.Item>
            <Form.Item name="conf_threshold" label={t('pages.evaluation.confThreshold')} style={{ width: 160 }}>
              <InputNumber min={0.01} max={1} step={0.05} style={{ width: '100%' }} />
            </Form.Item>
            <Form.Item name="max_images" label={t('pages.evaluation.maxImages')} extra={t('pages.evaluation.maxImagesHint')} style={{ width: 220 }}>
              <InputNumber min={0} style={{ width: '100%' }} />
            </Form.Item>
          </Space>
        </Form>
      </Modal>
    </div>
  );
};

// Evaluation Detail: full metrics + baseline delta + per-class
export const EvaluationDetail: React.FC = () => {
  const { t } = useTranslation();
  const { id: jobId } = useParams<{ id: string }>();
  const [job, setJob] = useState<EvaluationJob | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchJob = useCallback(async () => {
    try {
      if (!jobId) return;
      const response = await evaluationApi.get(jobId);
      setJob(response.data);
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

  if (loading) return <Spin size="large" style={{ display: 'block', margin: '100px auto' }} />;
  if (!job) return <Alert message="Job not found" type="error" />;

  const report: GateReport | undefined = job.results?.report;
  const metrics = report?.metrics || job.metrics || {};
  const baseline = report?.baseline_metrics;
  const delta = report?.delta || {};
  const perClass = (metrics.per_class || {}) as Record<string, { ap50?: number; precision?: number; recall?: number; gt?: number }>;
  const rows = Object.entries(perClass).map(([name, v]) => ({ key: name, name, ...v }));
  const isCls = (job.params?.task_type as string) === 'classification';

  return (
    <div>
      <PageHeader
        title={job.name}
        subtitle={`${t('pages.training.status')}: ${t(statusLabelKey(job.status))}`}
        onRefresh={fetchJob}
      />
      <Card style={{ marginBottom: 16 }}>
        <Descriptions size="small" column={4} bordered>
          {isCls ? (
            <>
              <Descriptions.Item label="Top-1">{metrics.top1_accuracy ?? '-'}</Descriptions.Item>
              <Descriptions.Item label="Top-5">{metrics.top5_accuracy ?? '-'}</Descriptions.Item>
            </>
          ) : (
            <>
              <Descriptions.Item label="mAP@50">{metrics.mAP50 ?? '-'}</Descriptions.Item>
              <Descriptions.Item label="mAP@50:95">{metrics.mAP50_95 ?? '-'}</Descriptions.Item>
              <Descriptions.Item label="P">{metrics.precision ?? '-'}</Descriptions.Item>
              <Descriptions.Item label="R">{metrics.recall ?? '-'}</Descriptions.Item>
            </>
          )}
          <Descriptions.Item label="F1">{metrics.f1_score ?? '-'}</Descriptions.Item>
          <Descriptions.Item label={t('pages.evaluation.imagesEvaluated')}>{metrics.images_evaluated ?? '-'}</Descriptions.Item>
          <Descriptions.Item label={t('pages.optimization.avgLatency')}>{metrics.avg_latency_ms ? `${metrics.avg_latency_ms} ms` : '-'}</Descriptions.Item>
          <Descriptions.Item label={t('pages.evaluation.baselineModel')}>{job.params?.baseline_model_id ? String(job.params.baseline_model_id) : '-'}</Descriptions.Item>
        </Descriptions>
        {baseline && Object.keys(delta).length > 0 && (
          <Alert
            style={{ marginTop: 12 }}
            type="info"
            message={t('pages.evaluation.deltaTitle')}
            description={Object.entries(delta).map(([k, v]) => (
              <Tag key={k} color={v >= 0 ? 'green' : 'red'}>{`${k}: ${v > 0 ? '+' : ''}${v}`}</Tag>
            ))}
          />
        )}
        {job.error_message && <Alert style={{ marginTop: 12 }} type="error" message={job.error_message} />}
      </Card>

      {rows.length > 0 && (
        <Card title={t(isCls ? 'pages.evaluation.perClass' : 'pages.optimization.perClass')} style={{ marginBottom: 16 }}>
          <Table
            size="small"
            pagination={rows.length > 10 ? { pageSize: 10 } : false}
            dataSource={rows}
            columns={[
              { title: t('pages.optimization.className'), dataIndex: 'name' },
              ...(isCls ? [] : [{ title: 'AP@50', dataIndex: 'ap50' }]),
              { title: 'P', dataIndex: 'precision' },
              { title: 'R', dataIndex: 'recall' },
              ...(isCls ? [] : [{ title: 'GT', dataIndex: 'gt' }]),
            ]}
          />
        </Card>
      )}

      <Card>
        <Button icon={<DownloadOutlined />} disabled={job.status !== 'completed'} onClick={() => apiClient.downloadFile(evaluationApi.downloadUrl(job.id))}>
          {t('pages.evaluation.downloadReport')}
        </Button>
      </Card>
    </div>
  );
};

export default Evaluation;
