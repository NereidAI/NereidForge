// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

import React, { useState, useEffect } from 'react';
import { Table, Button, Space, Modal, Form, Input, Select, Card, Upload, Tag, Tooltip, Typography, Progress, Spin, Empty, Alert, Popconfirm } from 'antd';
import { DownOutlined, CloudServerOutlined, UploadOutlined, StopOutlined, EyeOutlined, ExperimentOutlined, InboxOutlined, DeleteOutlined } from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import { useNavigate } from 'react-router-dom';
import { message } from 'antd';
import { useAuth } from '../hooks/useAuth';
import { modelApi, settingsApi } from '../api/endpoints';
import { useAuthStore } from '../store/authStore';
import { ArtifactFormatCatalog, Model, ModelVersion, ModelVersionFile } from '../types/api';
import PageHeader from '@components/PageHeader';
import FilterSearchBar from '@components/FilterSearchBar';
import { HelpButton } from '@components/HelpDrawer';
import StatusBadge from '../components/StatusBadge';
import { useServerTable } from '@hooks/useServerTable';
import { useDebounce } from '@hooks/useDebounce';
import TryModelModal from '../components/TryModelModal';
import EdgePackageModal from '../components/EdgePackageModal';
import { formatDateTime } from '../utils/format';

// Single source of truth for selectable model task types in the upload forms.
// `sam` = interactive segmentation (Segment Anything) chosen in the video
// annotator; distinct from `segmentation` (YOLOv8 instance-seg weights).
const TASK_TYPE_OPTIONS = ['detection', 'classification', 'segmentation', 'sam', 'text-generation', 'vlm'];

// Sources accepted by model_format.py before conversion dispatch. Removed YOLO
// repository exporters stay unavailable; explicit torch and torchvision remain
// distinct so deployment never guesses a checkpoint's framework semantics.
const SOURCE_OPTIONS = ['auto', 'torchvision', 'torch', 'custom'];

const Models: React.FC = () => {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const { tenantId, isSuperAdmin } = useAuth();
  // Server-side filters: task-type dropdown plus debounced name search.
  const [taskTypeFilter, setTaskTypeFilter] = useState('');
  const [searchInput, setSearchInput] = useState('');
  const search = useDebounce(searchInput.trim(), 300);
  // Server-side paginated registry list.
  const modelsTable = useServerTable<Model, { task_type?: string; search?: string }>({
    list: (params) => modelApi.list(params),
    defaultPageSize: 10,
    filters: { task_type: taskTypeFilter || undefined, search: search || undefined },
  });
  const { reload: loadModels } = modelsTable;
  const [selectedModel, setSelectedModel] = useState<Model | null>(null);
  // Deployment status lookup (per version)
  const [servingMap, setServingMap] = useState<Record<string, any>>({});

  // Fetch the serving state (not-deployed / deploying / ready / failed) for
  // every displayed version. Deploy is async: conversion can take 5-10 min,
  // so statuses are re-polled while any version is still deploying.
  const fetchServingStatuses = React.useCallback(
    async (modelName: string, vers: ModelVersion[]) => {
      const results = await Promise.allSettled(
        vers.map((v) => modelApi.servingStatus(modelName, v.version)),
      );
      const map: Record<string, any> = {};
      vers.forEach((v, i) => {
        const r = results[i];
        map[v.version] = r.status === 'fulfilled' ? r.value.data : { deployed: false };
      });
      setServingMap((prev) => ({ ...prev, ...map }));
    },
    [],
  );

  // Server-side paginated version table for the opened model (fetch gated on
  // the modal being open; the model name filter triggers reloads).
  const versionsTable = useServerTable<ModelVersion, { name: string }>({
    enabled: !!selectedModel,
    filters: { name: selectedModel?.name ?? '' },
    defaultPageSize: 10,
    list: async (params) => {
      const res = await modelApi.getVersions(params.name, { page: params.page, page_size: params.page_size });
      await fetchServingStatuses(params.name, res.data.items || []);
      return res;
    },
  });
  const { items: versions, reload: reloadVersions } = versionsTable;
  // Keep re-polling only while a displayed version is still deploying or the
  // created predictor is not yet ready; stops as soon as every row is steady.
  const hasBusyServing = Object.values(servingMap).some(
    (s: any) => s?.state === 'deploying' || (s?.deployed && s.ready === false),
  );
  useEffect(() => {
    if (!selectedModel || !hasBusyServing) return;
    const timer = setInterval(() => {
      void fetchServingStatuses(selectedModel.name, versions);
    }, 8000);
    return () => clearInterval(timer);
  }, [selectedModel, hasBusyServing, versions, fetchServingStatuses]);
  const [modalVisible, setModalVisible] = useState(false);
  const [uploadVisible, setUploadVisible] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [deployTarget, setDeployTarget] = useState<{ name: string; version: string } | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [artifactCatalog, setArtifactCatalog] = useState<ArtifactFormatCatalog | null>(null);
  const [runtimes, setRuntimes] = useState<Array<{ name: string; model_type: string; image: string }>>([]);
  const [deploying, setDeploying] = useState(false);
  const [deletingKey, setDeletingKey] = useState<string | null>(null);
  const [computeResources, setComputeResources] = useState<any[]>([]);
  // URL loaded into the embedded Netron viewer modal (null = closed).
  const [netronUrl, setNetronUrl] = useState<string | null>(null);
  // Registry version selected for the ad-hoc single-image trial modal.
  // filePath is set when the trial is launched on one specific file of the
  // version's artifact bundle; absent = service-default ONNX/.pt resolution.
  const [tryTarget, setTryTarget] = useState<{ version: ModelVersion; filePath?: string } | null>(null);
  // Version selected for production edge package generation/download.
  const [packageTarget, setPackageTarget] = useState<ModelVersion | null>(null);
  // Controlled expansion of version rows (artifact bundles). Controlled (not
  // the table's internal state) so expansion survives the 8s serving-status
  // polling, which replaces the dataSource array.
  const [expandedVersionKeys, setExpandedVersionKeys] = useState<string[]>([]);
  const [deployForm] = Form.useForm();
  const deployRuntimeValue = Form.useWatch('runtime_type', deployForm);
  const [registerForm] = Form.useForm();
  const [uploadForm] = Form.useForm();
  const uploadArtifactFormat = Form.useWatch('artifact_format', uploadForm);
  const acceptedArtifactSuffixes = artifactCatalog ? Array.from(new Set([
    ...artifactCatalog.formats.flatMap(format => format.extensions),
    ...artifactCatalog.archive_suffixes,
  ])).join(',') : undefined;
  const selectedArtifactDescriptor = artifactCatalog?.formats.find(
    format => format.id === uploadArtifactFormat,
  );

  useEffect(() => {
    modelApi.getServingRuntimes().then((res) => setRuntimes(res.data.items ?? [])).catch(() => {});
    modelApi.getArtifactFormats().then((res) => setArtifactCatalog(res.data)).catch(() => {});
  }, [tenantId]);

  const handleRowClick = (record: Model) => {
    // Reset the per-version deployment status lookup before the new model's
    // versions load (keys are version strings, not namespaced per model).
    setServingMap({});
    setExpandedVersionKeys([]);
    setSelectedModel(record);
  };

  const handleRegister = async (values: any) => {
    // Model registration mirrors Kubeflow Model Registry while keeping the
    // task type + framework dimensions the NF list/pre-annotation rely on.
    try {
      await modelApi.create({
        name: values.name,
        description: values.description,
        framework: values.framework || 'pytorch',
        task_type: values.task_type,
      });
      setModalVisible(false);
      registerForm.resetFields();
      loadModels();
    } catch (e) {
      console.error(e);
    }
  };

  // Manually upload one file, an IR-style file group, or one safe archive.
  const handleUpload = async (values: any) => {
    const files = (values.files?.fileList || [])
      .map((entry: any) => entry.originFileObj)
      .filter(Boolean) as File[];
    if (!files.length || !selectedModel) return;
    let runtimeRequirements: Record<string, unknown> | undefined;
    const rawRequirements = String(values.runtime_requirements || '').trim();
    if (rawRequirements) {
      try {
        const parsed = JSON.parse(rawRequirements);
        if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) throw new Error('object required');
        runtimeRequirements = parsed;
      } catch {
        message.error(t('pages.models.invalidRuntimeRequirements'));
        return;
      }
    }
    setUploading(true);
    setUploadProgress(0);
    try {
      const metadata = {
        // framework/task_type are model-level attributes (all versions of a
        // model share them); inherit from the model instead of re-asking.
        framework: selectedModel.framework || 'pytorch',
        task_type: selectedModel.task_type || 'detection',
        description: values.description,
        input_shape: values.input_shape ? String(values.input_shape).split(',').map(Number) : [],
        // Source hints help conversion/decoder matching but never override the
        // backend's validated artifact format or capability registry.
        ...(values.source ? { source: values.source } : {}),
        ...(values.architecture ? { architecture: String(values.architecture).trim() } : {}),
        ...(values.artifact_format && values.artifact_format !== 'auto' ? { artifact_format: values.artifact_format } : {}),
        ...(values.primary_file?.trim() ? { primary_file: values.primary_file.trim() } : {}),
        ...(runtimeRequirements ? { runtime_requirements: runtimeRequirements } : {}),
      };
      await modelApi.createVersion(selectedModel.name, {
        version: values.version,
        description: values.description,
        files,
        metadata,
      }, setUploadProgress);
      message.success(t('pages.models.versionUploaded'));
      setUploadVisible(false);
      uploadForm.resetFields();
      reloadVersions();
    } catch (e: any) {
      console.error(e);
      message.error(e?.response?.data?.detail || t('pages.models.uploadFailed'));
    } finally {
      setUploading(false);
      setUploadProgress(0);
    }
  };

  // Download model version artifact (authenticated streaming download).
  // With no filePath this streams the registered primary artifact; with one
  // it streams that specific sibling from the version's S3 artifact bundle.
  const handleDownloadArtifact = async (modelName: string, version: string, filePath?: string) => {
    try {
      const token = useAuthStore.getState().accessToken;
      const tenant = useAuthStore.getState().tenantId;
      const qs = filePath ? `?path=${encodeURIComponent(filePath)}` : '';
      const resp = await fetch(`/api/v1/models/${modelName}/versions/${version}/download${qs}`, {
        headers: { Authorization: token ? `Bearer ${token}` : '', 'X-Tenant-Id': tenant || 'default' },
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const blob = await resp.blob();
      // Use the server-provided Content-Disposition filename (real extension).
      let fname = `${modelName}-${version}.bin`;
      const cd = resp.headers.get('Content-Disposition');
      if (cd) {
        const m = cd.match(/filename="?([^";]+)"?/);
        if (m) fname = m[1];
      }
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = fname;
      document.body.appendChild(a);
      a.click();
      URL.revokeObjectURL(a.href);
    } catch (e) {
      console.error(e);
      message.error(t('pages.models.downloadFailed'));
    }
  };

  const hasArtifactCapability = (v: ModelVersion, capability: string): boolean => {
    const capabilities = v.metadata?.artifact_capabilities;
    // Versions created before capability recording retain their legacy actions.
    return !Array.isArray(capabilities) || capabilities.includes(capability);
  };

  const isOpenVinoIRVersion = (v: ModelVersion) => {
    const primary = v.file_path || '';
    const optimized = String(v.metadata?.optimized_uri || '');
    return primary.endsWith('.xml') || optimized.endsWith('.xml');
  };

  // Open a SeaweedFS (s3://) artifact in the embedded Netron viewer through
  // the same-origin auth-exempt /s3-object proxy so the iframe can fetch it.
  const openNetron = (uri: string) => {
    if (!uri.startsWith('s3://')) {
      message.warning(t('pages.models.netronUnsupported'));
      return;
    }
    const fname = (uri.split('/').pop() || '').toLowerCase();
    const supported = /\.(onnx|pt|pth|tflite|pb|keras|h5|gguf|caffemodel|mlmodel)$/.test(fname);
    if (!supported) {
      message.info(t('pages.models.netronUnsupported'));
      return;
    }
    const proxyUrl = `/api/v1/s3-object?uri=${encodeURIComponent(uri)}`;
    setNetronUrl(`/netron/index.html?url=${encodeURIComponent(proxyUrl)}`);
  };

  const handleDeploy = async (modelName: string, version: string) => {
    setDeployTarget({ name: modelName, version });
    // Load cluster compute resources (select based on available hardware)
    try {
      const res = await settingsApi.getComputeResources();
      setComputeResources(res.data.resources ?? []);
    } catch {
      setComputeResources([]);
    }
  };

  const handleDeploySubmit = async (values: any) => {
    if (!deployTarget) return;
    setDeploying(true);
    try {
      await modelApi.deploy(deployTarget.name, deployTarget.version, 'kserve', {
        runtime_type: values.runtime_type,
        gpu: values.gpu || '0',
        domain: values.domain || undefined,
      });
      // Async: the request only enqueues the conversion + rollout task.
      message.success(t('pages.models.deployStarted'));
      setDeployTarget(null);
      reloadVersions();
    } catch (e: any) {
      console.error(e);
      const detail = e?.response?.data?.message || t('pages.models.deployFailed');
      message.error(detail);
    } finally {
      setDeploying(false);
    }
  };

  const reportDeleteError = (e: any, fallback: string) => {
    const status = e?.response?.status;
    if (status === 403) {
      message.error(t('pages.models.deleteForbidden'));
      return;
    }
    if (status === 409) {
      message.error(t('pages.models.deleteLiveConflict'));
      return;
    }
    message.error(e?.response?.data?.detail || fallback);
  };

  const handleDeleteModel = async (record: Model) => {
    const key = `model:${record.id}`;
    setDeletingKey(key);
    try {
      await modelApi.delete(record.name);
      message.success(t('pages.models.modelDeleted'));
      if (selectedModel?.id === record.id) setSelectedModel(null);
      loadModels();
    } catch (e: any) {
      reportDeleteError(e, t('pages.models.modelDeleteFailed'));
    } finally {
      setDeletingKey((current) => current === key ? null : current);
    }
  };

  const handleDeleteVersion = async (record: ModelVersion) => {
    if (!selectedModel) return;
    const key = `version:${selectedModel.id}:${record.version}`;
    setDeletingKey(key);
    try {
      await modelApi.deleteVersion(selectedModel.name, record.version);
      message.success(t('pages.models.versionDeleted'));
      setExpandedVersionKeys((current) => current.filter((item) => item !== record.version));
      reloadVersions();
    } catch (e: any) {
      reportDeleteError(e, t('pages.models.versionDeleteFailed'));
    } finally {
      setDeletingKey((current) => current === key ? null : current);
    }
  };

  const columns = [
    { title: t('pages.models.name'), dataIndex: 'name', key: 'name', ellipsis: true },
    { title: t('pages.models.taskType'), dataIndex: 'task_type', key: 'task_type', ellipsis: true },
    { title: t('pages.models.framework'), dataIndex: 'framework', key: 'framework' },
    { title: t('pages.models.description'), dataIndex: 'description', key: 'description', ellipsis: true, render: (v: string) => v || '-' },
    {
      title: t('pages.models.latestVersion'),
      key: 'latest_version',
      render: (_: unknown, r: Model) => (r.versions?.[0]?.version) || '-',
    },
    {
      title: t('pages.models.versionCount'),
      key: 'version_count',
      render: (_: unknown, r: Model) => r.versions?.length || 0,
    },
    {
      title: t('pages.models.stage'),
      key: 'stage',
      render: (_: unknown, r: Model) => {
        const s = r.versions?.[0]?.status;
        return s ? <StatusBadge status={s} /> : '-';
      },
    },
    {
      title: t('pages.models.artifacts'),
      key: 'artifacts_count',
      render: (_: unknown, r: Model) => r.versions?.[0]?.artifacts_count ?? 0,
    },
    { title: t('pages.models.createdBy'), dataIndex: 'created_by', key: 'created_by', render: (v: string) => v || '-' },
    { title: t('pages.models.updatedAt'), dataIndex: 'updated_at', key: 'updated_at', render: (d: string) => formatDateTime(d) },
    ...(isSuperAdmin ? [{
      title: t('common.actions'),
      key: 'actions',
      render: (_: unknown, record: Model) => {
        const key = `model:${record.id}`;
        return (
          <Popconfirm
            title={t('pages.models.deleteModelConfirm', { name: record.name })}
            description={t('pages.models.deleteModelHint')}
            okText={t('common.delete')}
            cancelText={t('common.cancel')}
            okButtonProps={{ danger: true, loading: deletingKey === key }}
            onConfirm={() => handleDeleteModel(record)}
          >
            <Button
              type="link"
              size="small"
              danger
              icon={<DeleteOutlined />}
              loading={deletingKey === key}
              onClick={(e) => e.stopPropagation()}
            >
              {t('common.delete')}
            </Button>
          </Popconfirm>
        );
      },
    }] : []),
  ];

  return (
    <Card>
      <PageHeader title={t('pages.models.title')} subtitle={t('pages.models.subtitle')} actions={[
        <HelpButton module="models" />,
        <Button type="primary" onClick={() => { setModalVisible(true); }}>
          {t('pages.models.register')}
        </Button>
      ]} />
      <FilterSearchBar
        filterPlaceholder={t('pages.models.filterTaskType')}
        filterValue={taskTypeFilter}
        onFilterChange={setTaskTypeFilter}
        filterOptions={TASK_TYPE_OPTIONS.map((tt) => ({
          value: tt,
          label: t(`pages.models.tt_${tt.replace(/-/g, '_')}`),
        }))}
        searchPlaceholder={t('pages.models.searchByName')}
        searchValue={searchInput}
        onSearchChange={setSearchInput}
      />
      <Table
        {...modelsTable.tableProps}
        columns={columns}
        onRow={(record) => ({ onClick: () => handleRowClick(record) })}
        scroll={{ x: 'max-content' }}
      />
      <Modal
        title={<span>{selectedModel?.name || ''} - {t('pages.models.versions')} <Button type="primary" size="small" icon={<UploadOutlined />} onClick={() => { uploadForm.resetFields(); setUploadVisible(true); }}>{t('pages.models.uploadVersion')}</Button></span>}
        open={!!selectedModel}
        onCancel={() => setSelectedModel(null)}
        footer={null}
        width={1200}
        style={{ maxWidth: 'calc(100vw - 32px)' }}
        destroyOnHidden
      >
        <Table
          {...versionsTable.tableProps}
          rowKey="version"
          columns={[
            { title: t('pages.models.version'), dataIndex: 'version', key: 'version' },
            {
              // Source training job: links to the job that produced this
              // version, so timestamp-less job-id labels stay traceable.
              title: t('pages.models.sourceJob'),
              key: 'source_job',
              ellipsis: true,
              render: (_: unknown, r: ModelVersion) => {
                const jid = r.metadata?.job_id;
                if (!jid) return '-';
                const jname = r.metadata?.job_name || r.description || jid;
                return (
                  <Button type="link" size="small" style={{ padding: 0 }} onClick={() => navigate(`/training/${jid}`)}>
                    {jname}
                  </Button>
                );
              },
            },
            {
              title: t('pages.models.targetDevice'),
              key: 'target_device',
              render: (_: unknown, r: ModelVersion) => {
                const td = r.metadata?.target_device;
                const opt = r.metadata?.optimizer;
                if (td || opt) return td || opt;
                return r.metadata?.uploaded_by ? t('pages.models.manualUpload') : '-';
              },
            },
            { title: t('pages.models.artifacts'), key: 'artifacts_count', render: (_: unknown, r: ModelVersion) => r.artifacts_count ?? 0 },
            {
              title: t('pages.models.deployStatus'),
              key: 'deploy_status',
              render: (_: unknown, r: ModelVersion) => {
                const st = servingMap[r.version];
                if (!st) return <Tag>...</Tag>;
                if (st.state === 'failed') {
                  return (
                    <Tooltip title={st.error || t('pages.models.deployFailed')}>
                      <Tag color="red">{t('pages.models.deployFailed')}</Tag>
                    </Tooltip>
                  );
                }
                if (st.state === 'deploying') {
                  return <Tag color="orange">{t('pages.models.deploying')}</Tag>;
                }
                if (st.state === 'terminating') {
                  return <Tag color="blue">{t('pages.models.undeploying')}</Tag>;
                }
                if (st.state === 'undeploy_failed') {
                  return (
                    <Tooltip title={st.error || t('pages.models.deployFailed')}>
                      <Tag color="red">{t('pages.deployments.error')}</Tag>
                    </Tooltip>
                  );
                }
                return st.deployed ? (
                  <Tag color={st.ready === false ? 'orange' : 'green'}>{st.ready === false ? t('pages.models.deploying') : t('pages.models.deployedReady')}</Tag>
                ) : (
                  <Tag>{t('pages.models.notDeployed')}</Tag>
                );
              },
            },
            {
              title: t('pages.models.endpoint'),
              key: 'endpoint',
              render: (_: unknown, r: ModelVersion) => {
                const st = servingMap[r.version];
                if (st?.url && (st.deployed || st.state === 'deploying')) {
                  return <Typography.Text copyable={{ text: st.url }} ellipsis style={{ maxWidth: 200, fontSize: 12, color: st.state === 'deploying' ? '#faad14' : undefined }}>{st.url}</Typography.Text>;
                }
                return '-';
              },
            },
            {
              title: t('common.actions'),
              key: 'actions',
              // Exactly one lifecycle action per row (Deploy or Undeploy), so a
              // dropdown would add a click for nothing — render a link button.
              // Per-artifact actions live in the expanded bundle.
              render: (_: unknown, record: ModelVersion) => {
                const stRow = servingMap[record.version];
                // A version already deployed or still being deployed gets the
                // undeploy (cancel) action instead of a duplicate deploy.
                const isLive = !!(
                  stRow?.deployed
                  || stRow?.state === 'deploying'
                  || stRow?.state === 'undeploy_failed'
                  || record.status === 'deployed'
                  || record.status === 'deploying'
                );
                if (stRow?.state === 'terminating') {
                  return (
                    <Button type="link" size="small" danger icon={<StopOutlined />} disabled>
                      {t('pages.models.undeploying')}
                    </Button>
                  );
                }
                const lifecycleButton = isLive ? (
                  <Button
                    type="link"
                    size="small"
                    danger
                    icon={<StopOutlined />}
                    onClick={(e) => {
                      e.stopPropagation();
                      void (async () => {
                        try {
                          await modelApi.undeploy(selectedModel?.name || '', record.version);
                          message.success(t('pages.models.undeployed'));
                          reloadVersions();
                        } catch {
                          message.error(t('pages.models.deployFailed'));
                        }
                      })();
                    }}
                  >
                    {t('pages.models.undeploy')}
                  </Button>
                ) : !hasArtifactCapability(record, 'deploy') ? (
                  <Tooltip title={t('pages.models.registrationOnly')}>
                    <span>
                      <Button type="link" size="small" icon={<CloudServerOutlined />} disabled>
                        {t('pages.models.deployToKfp')}
                      </Button>
                    </span>
                  </Tooltip>
                ) : (
                  <Button
                    type="link"
                    size="small"
                    icon={<CloudServerOutlined />}
                    onClick={(e) => { e.stopPropagation(); handleDeploy(selectedModel?.name || '', record.version); }}
                  >
                    {t('pages.models.deployToKfp')}
                  </Button>
                );
                const deleteKey = `version:${selectedModel?.id || ''}:${record.version}`;
                const deleteButton = isSuperAdmin ? (
                  <Tooltip title={isLive ? t('pages.models.deleteUndeployFirst') : undefined}>
                    <span>
                      <Popconfirm
                        title={t('pages.models.deleteVersionConfirm', { version: record.version })}
                        description={t('pages.models.deleteVersionHint')}
                        okText={t('common.delete')}
                        cancelText={t('common.cancel')}
                        okButtonProps={{ danger: true, loading: deletingKey === deleteKey }}
                        disabled={isLive}
                        onConfirm={() => handleDeleteVersion(record)}
                      >
                        <Button
                          type="link"
                          size="small"
                          danger
                          icon={<DeleteOutlined />}
                          disabled={isLive}
                          loading={deletingKey === deleteKey}
                        >
                          {t('common.delete')}
                        </Button>
                      </Popconfirm>
                    </span>
                  </Tooltip>
                ) : null;
                return (
                  <Space size={0} onClick={(e) => e.stopPropagation()}>
                    {lifecycleButton}
                    {deleteButton}
                    {isOpenVinoIRVersion(record) && hasArtifactCapability(record, 'package') && (
                      <Button
                        type="link"
                        size="small"
                        icon={<InboxOutlined />}
                        onClick={(e) => {
                          e.stopPropagation();
                          setPackageTarget(record);
                        }}
                      >
                        {t('pages.models.edgePackage')}
                      </Button>
                    )}
                  </Space>
                );
              },
            },
          ]}
          scroll={{ x: 'max-content' }}
          // Expanding a row lists every file in the version's S3 prefix —
          // the row itself only shows the primary artifact (ONNX after deploy).
          // Expansion is controlled so it survives dataSource refreshes, and
          // clicking anywhere on the row toggles it (the tiny chevron alone is
          // easy to miss inside the modal).
          onRow={(r) => ({
            onClick: (e) => {
              const target = e.target as HTMLElement;
              // The antd expand chevron already toggles via
              // onExpandedRowsChange; don't double-toggle from the row click.
              if (target.closest('button, a, .ant-dropdown-trigger, input, .ant-table-row-expand-icon')) return;
              const key = r.version;
              setExpandedVersionKeys((prev) =>
                prev.includes(key) ? prev.filter((k) => k !== key) : [...prev, key],
              );
            },
            style: { cursor: 'pointer' },
          })}
          expandable={{
            expandedRowKeys: expandedVersionKeys,
            onExpandedRowsChange: (keys) => setExpandedVersionKeys(keys as string[]),
            expandedRowRender: (r: ModelVersion) => (
              <VersionFilesRow
                modelName={selectedModel?.name || ''}
                version={r.version}
                artifactCapabilities={r.metadata?.artifact_capabilities}
                onDownload={handleDownloadArtifact}
                onTry={(fileName) => setTryTarget({ version: r, filePath: fileName })}
                onVisualize={openNetron}
              />
            ),
            rowExpandable: () => true,
          }}
        />
      </Modal>
      <Modal title={t('pages.models.register')} open={modalVisible} onCancel={() => setModalVisible(false)} footer={null}>
        <Form form={registerForm} onFinish={handleRegister} layout="vertical">
          <Form.Item name="name" label={t('pages.models.name')} rules={[{ required: true }]}>
            <Input placeholder="e.g. yolov8-custom" />
          </Form.Item>
          <Form.Item name="task_type" label={t('pages.models.taskType')} initialValue="detection">
            <Select options={TASK_TYPE_OPTIONS.map(tt => ({ label: tt, value: tt }))} placeholder={t('pages.models.selectTaskType')} />
          </Form.Item>
          <Form.Item name="framework" label={t('pages.models.framework')}>
            <Select options={['pytorch', 'tensorflow', 'onnx', 'transformers'].map(t => ({ label: t, value: t }))} placeholder={t('pages.models.selectFramework')} />
          </Form.Item>
          <Form.Item name="description" label={t('pages.models.description')}>
            <Input.TextArea rows={3} placeholder={t('pages.models.descriptionPlaceholder')} />
          </Form.Item>
          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit">{t('common.confirm')}</Button>
              <Button onClick={() => setModalVisible(false)}>{t('common.cancel')}</Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      <Modal title={t('pages.models.uploadVersion')} open={uploadVisible} onCancel={() => setUploadVisible(false)} footer={null} destroyOnHidden zIndex={1200}>
        <Form form={uploadForm} onFinish={handleUpload} layout="vertical">
          <Alert
            type="info"
            showIcon
            style={{ marginBottom: 16 }}
            message={t('pages.models.uploadBundleHint')}
          />
          <Form.Item name="version" label={t('pages.models.version')} rules={[{ required: true }]}>
            <Input placeholder="v1.0.0" />
          </Form.Item>
          <Form.Item name="description" label={t('pages.models.description')}>
            <Input.TextArea rows={2} />
          </Form.Item>
          <Form.Item
            name="input_shape"
            label={t('pages.models.inputShape')}
            tooltip={t('pages.models.inputShapeHint')}
          >
            <Input placeholder="1,3,640,640" />
          </Form.Item>
          <Form.Item
            name="source"
            label={t('pages.models.source')}
            initialValue="auto"
            tooltip={t('pages.models.sourceHint')}
          >
            <Select
              options={(SOURCE_OPTIONS || []).map(s => ({ label: s, value: s }))}
              placeholder={t('pages.models.selectSource')}
            />
          </Form.Item>
          <Form.Item
            noStyle
            shouldUpdate={(prev, cur) => prev.source !== cur.source}
          >
            {({ getFieldValue }) => (
              getFieldValue('source') === 'torchvision' ? (
                <Form.Item name="architecture" label={t('pages.models.architecture')}>
                  <Input placeholder="resnet18, efficientnet_b0, ..." />
                </Form.Item>
              ) : null
            )}
          </Form.Item>
          <Form.Item
            name="artifact_format"
            label={t('pages.models.artifactFormat')}
            initialValue="auto"
          >
            <Select
              options={[
                { value: 'auto', label: t('pages.models.artifactFormatAuto') },
                ...(artifactCatalog?.formats || []).map(format => ({
                  value: format.id,
                  label: `${format.label} — ${format.capabilities.map(
                    capability => t(`pages.models.capability.${capability}`),
                  ).join(', ')}`,
                })),
              ]}
            />
          </Form.Item>
          {selectedArtifactDescriptor?.compiled && (
            <Alert
              type="warning"
              showIcon
              style={{ marginBottom: 16 }}
              message={t('pages.models.compiledArtifactHint')}
            />
          )}
          <Form.Item
            name="primary_file"
            label={`${t('pages.models.primaryFile')} (${t('pages.models.optional')})`}
            tooltip={t('pages.models.primaryFileHint')}
          >
            <Input placeholder="repository/demo/1/model.xml" />
          </Form.Item>
          <Form.Item
            name="runtime_requirements"
            label={`${t('pages.models.runtimeRequirements')} (${t('pages.models.optional')})`}
            tooltip={t('pages.models.runtimeRequirementsHint')}
          >
            <Input.TextArea rows={2} placeholder='{"tensorrt":"8.6","gpu_compute_cap":"8.6"}' />
          </Form.Item>
          <Form.Item
            name="files"
            label={t('pages.models.files')}
            extra={acceptedArtifactSuffixes}
            rules={[{ required: true }]}
          >
            <Upload beforeUpload={() => false} multiple accept={acceptedArtifactSuffixes}>
              <Button icon={<UploadOutlined />}>{t('pages.models.chooseFiles')}</Button>
            </Upload>
          </Form.Item>
          {uploadProgress > 0 && (
            <Form.Item>
              <Progress percent={uploadProgress} status={uploadProgress >= 100 ? 'success' : 'active'} />
            </Form.Item>
          )}
          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit" loading={uploading}>{t('common.confirm')}</Button>
              <Button onClick={() => setUploadVisible(false)} disabled={uploadProgress > 0 && uploadProgress < 100}>{t('common.cancel')}</Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      <Modal title={t('pages.models.deployToKfp')} open={!!deployTarget} onCancel={() => setDeployTarget(null)} footer={null} destroyOnHidden zIndex={1200}>
        {deployTarget && (() => {
          const v = versions.find(x => x.version === deployTarget.version);
          const fp = v?.file_path || '';
          const fname = fp.split('/').pop() || '-';
          const ext = fname.includes('.') ? fname.split('.').pop()?.toUpperCase() : '-';
          const size = v?.size_bytes ?? 0;
          const sizeStr = size >= 1e6
            ? `${(size / 1e6).toFixed(2)} MB`
            : size >= 1e3
              ? `${(size / 1e3).toFixed(1)} KB`
              : `${size} B`;
          return (
            <Card size="small" style={{ marginBottom: 16 }}>
              <Space direction="vertical" size={2}>
                <span><strong>{t('pages.models.version')}:</strong> {deployTarget.version}</span>
                <span><strong>{t('pages.models.artifactFile')}:</strong> {fname}</span>
                <span><strong>{t('pages.models.artifactFormat')}:</strong> {v?.metadata?.artifact_format || ext}</span>
                <span><strong>{t('pages.models.artifactSize')}:</strong> {sizeStr}</span>
              </Space>
            </Card>
          );
        })()}
        <Form form={deployForm} key={deployTarget ? deployTarget.version : 'closed'} onFinish={handleDeploySubmit} layout="vertical">
          {(() => {
            const dv = deployTarget ? versions.find(x => x.version === deployTarget.version) : undefined;
            const isIR = /\.xml$/i.test(dv?.file_path || '');
            return (
              <>
          <Form.Item
            name="runtime_type"
            label={t('pages.models.runtime')}
            rules={[{ required: true }]}
            initialValue={isIR ? 'kserve-openvino' : undefined}
          >
            <Select
              placeholder={t('pages.models.runtime')}
              disabled={isIR}
              options={runtimes.map((r) => ({
                value: r.name,
                label: `${r.name.replace(/^kserve-/, '')}${r.model_type ? ` (${r.model_type})` : ''}`,
              }))}
            />
          </Form.Item>
          {isIR && (
            <Alert
              type="info"
              showIcon
              style={{ marginBottom: 16 }}
              message={t('pages.models.irRuntimeLocked')}
            />
          )}
          {!isIR && deployRuntimeValue === 'kserve-openvino' && (
            <Alert
              type="warning"
              showIcon
              style={{ marginBottom: 16 }}
              message={t('pages.models.openvinoOnnxHint')}
            />
          )}
              </>
            );
          })()}
          <Form.Item name="gpu" label={t('pages.models.gpu')} extra={computeResources.length ? t('pages.models.computeResources') : undefined} initialValue="cpu">
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
            <Input placeholder="model-detect.example.com" />
          </Form.Item>
          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit" loading={deploying}>{t('common.confirm')}</Button>
              <Button onClick={() => setDeployTarget(null)}>{t('common.cancel')}</Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      {/* Embedded Netron model viewer (self-hosted under /netron/). */}
      <Modal
        title={t('pages.models.netronViewer')}
        open={!!netronUrl}
        onCancel={() => setNetronUrl(null)}
        footer={null}
        destroyOnClose
        width="90%"
        zIndex={1200}
        styles={{ body: { height: '80vh', padding: 0 } }}
      >
        {netronUrl && (
          <iframe
            src={netronUrl}
            title="Netron"
            style={{ width: '100%', height: '100%', border: 0 }}
          />
        )}
      </Modal>

      <EdgePackageModal
        open={!!packageTarget}
        modelName={selectedModel?.name || ''}
        version={packageTarget}
        onClose={() => setPackageTarget(null)}
      />

      {/* Ad-hoc single-image trial with optional Grad-CAM (no deployment). */}
      <TryModelModal
        model={selectedModel}
        version={tryTarget?.version || null}
        filePath={tryTarget?.filePath || null}
        onClose={() => setTryTarget(null)}
      />
    </Card>
  );
};

// Human-readable byte size for the version artifact file list.
const formatBytes = (n: number): string => {
  if (!n) return '0 B';
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)} GB`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)} MB`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)} KB`;
  return `${n} B`;
};

// Expanded row content for the Versions table: lazily lists EVERY file in the
// version's S3 prefix (the table row itself only shows the primary file_path,
// which is the ONNX after deploy — the original .pt is a sibling). Per-file
// artifact actions (Test with image / Visualize / Download) live here.
const VersionFilesRow: React.FC<{
  modelName: string;
  version: string;
  artifactCapabilities?: string[];
  onDownload: (modelName: string, version: string, filePath: string) => void;
  onTry: (fileName: string) => void;
  onVisualize: (uri: string) => void;
}> = ({ modelName, version, artifactCapabilities, onDownload, onTry, onVisualize }) => {
  const { t } = useTranslation();
  const [files, setFiles] = useState<ModelVersionFile[] | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let alive = true;
    modelApi.getVersionFiles(modelName, version)
      .then((res) => { if (alive) setFiles(res.data.files || []); })
      .catch(() => { if (alive) setFailed(true); });
    return () => { alive = false; };
  }, [modelName, version]);

  if (failed) return <span style={{ color: '#999' }}>{t('pages.models.versionFilesFailed')}</span>;
  if (!files) return <Spin size="small" />;
  if (files.length === 0) return <Empty description={t('pages.models.versionFilesEmpty')} image={Empty.PRESENTED_IMAGE_SIMPLE} />;

  return (
    <Space direction="vertical" style={{ width: '100%', padding: '4px 0' }} size={4}>
      {files.map((f) => {
        const name = f.name;
        const lower = name.toLowerCase();
        const canTry = /\.(pt|onnx)$/i.test(lower) && (
          !Array.isArray(artifactCapabilities) || artifactCapabilities.includes('trial')
        );
        const canView = /\.(onnx|pt|pth|tflite|pb|keras|h5|gguf|caffemodel|mlmodel)$/i.test(lower);
        return (
          <Space key={name} size={12} style={{ fontSize: 13 }}>
            <Typography.Text style={{ minWidth: 220 }}>{name}</Typography.Text>
            <Tag>{formatBytes(f.size)}</Tag>
            {f.modified_at && <Typography.Text type="secondary" style={{ fontSize: 12 }}>{f.modified_at}</Typography.Text>}
            {canTry && (
              <Button size="small" type="link" icon={<ExperimentOutlined />} onClick={(e) => { e.stopPropagation(); onTry(name); }}>
                {t('pages.models.tryModel')}
              </Button>
            )}
            {canView && (
              <Button size="small" type="link" icon={<EyeOutlined />} onClick={(e) => { e.stopPropagation(); onVisualize(f.uri); }}>
                {t('pages.models.visualize')}
              </Button>
            )}
            <Button size="small" type="link" icon={<DownOutlined />} onClick={(e) => { e.stopPropagation(); onDownload(modelName, version, name); }}>
              {t('pages.models.downloadArtifact')}
            </Button>
          </Space>
        );
      })}
    </Space>
  );
};

export const ModelVersions: React.FC = () => {
  const { t } = useTranslation();
  return (
    <Card title={t('pages.models.versions')}>
      <p>{t('pages.models.todo')}</p>
    </Card>
  );
};

export default Models;
