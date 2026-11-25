// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

import React, { useCallback, useEffect, useState } from 'react';
import { Card, Table, Button, Space, Modal, Form, Input, Upload, Popconfirm, message, Tag, Alert, Tooltip, Empty, Spin, Progress, Drawer, Tabs, Select, Descriptions, Tree } from 'antd';
import { UploadOutlined, EditOutlined, DeleteOutlined, EyeOutlined, DownloadOutlined, CopyOutlined, CheckOutlined, FileTextOutlined, FolderOutlined, FileOutlined, PictureOutlined, InfoCircleOutlined } from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import { useParams, useNavigate } from 'react-router-dom';
import { datasetApi } from '@api/endpoints';
import { PageHeader } from '@components/PageHeader';
import { HelpButton } from '@components/HelpDrawer';
import { StatusBadge } from '@components/StatusBadge';
import { useServerTable } from '@hooks/useServerTable';
import { useDebounce } from '@hooks/useDebounce';
import FilterSearchBar from '@components/FilterSearchBar';
import { useAuthStore } from '@store/authStore';
import { DatasetStatus } from '@nftypes/api';
import type { Dataset, DatasetVersion, VersionFile } from '@nftypes/api';
import { formatDateTime } from '../utils/format';

// Vite inlines these deployment limits at build time; the literals are local-
// development defaults and match the backend's StorageConfig defaults.
const MAX_UPLOAD_FILES = Number(import.meta.env.VITE_DATASET_UPLOAD_MAX_FILES || 200);
const MAX_UPLOAD_BYTES = Number(import.meta.env.VITE_DATASET_UPLOAD_MAX_BYTES || 1024 * 1024 * 1024);

export const Datasets: React.FC = () => {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const [modalOpen, setModalOpen] = useState(false);
  const [editingDataset, setEditingDataset] = useState<Dataset | null>(null);
  const [form] = Form.useForm();
  const [refsMap, setRefsMap] = useState<Record<string, string[]>>({});
  // Server-side status filter plus debounced name search.
  const [statusFilter, setStatusFilter] = useState('');
  const [searchInput, setSearchInput] = useState('');
  const search = useDebounce(searchInput.trim(), 300);

  // Server-side paginated dataset list; reference status (which jobs/projects
  // use a dataset) is loaded per visible page to disable unsafe deletes.
  const table = useServerTable<Dataset, { status?: string; search?: string }>({
    defaultPageSize: 10,
    filters: {
      status: statusFilter || undefined,
      search: search || undefined,
    },
    list: async (params) => {
      const response = await datasetApi.list(params);
      const refs: Record<string, string[]> = {};
      await Promise.allSettled(
        (response.data.items || []).map(async (ds) => {
          try {
            const r = await datasetApi.getReferences(ds.id);
            if (r.data.references?.length) refs[ds.id] = r.data.references;
          } catch { /* ignore */ }
        })
      );
      setRefsMap(refs);
      return response;
    },
  });
  const { reload: fetchDatasets } = table;

  const columns = [
    {
      title: t('pages.datasets.datasetName'),
      dataIndex: 'name',
      key: 'name',
      render: (name: string, record: Dataset) => (
        <span onClick={() => handleView(record)} style={{ cursor: 'pointer', color: '#1677ff' }}>
          {name}
        </span>
      ),
    },
    {
      title: t('pages.datasets.version'),
      dataIndex: 'version',
      key: 'version',
    },
    {
      title: t('pages.datasets.format'),
      dataIndex: 'format',
      key: 'format',
      render: (f: string) => <FormatTag format={f || ''} />,
    },
    {
      title: t('pages.datasets.size'),
      dataIndex: 'size_bytes',
      key: 'size_bytes',
      render: (size: number) => formatBytes(size),
    },
    {
      title: t('pages.datasets.status'),
      dataIndex: 'status',
      key: 'status',
      render: (status: DatasetStatus) => <StatusBadge status={status} />,
    },
    {
      title: t('pages.datasets.createdAt'),
      dataIndex: 'created_at',
      key: 'created_at',
      render: (date: string) => formatDateTime(date),
    },
    {
      title: t('common.actions'),
      key: 'actions',
      render: (_: unknown, record: Dataset) => {
        const refs = refsMap[record.id];
        return (
          <Space size={2}>
            <Tooltip title={t('common.edit')}>
              <Button type="text" size="small" icon={<EditOutlined />} onClick={() => handleEdit(record)} />
            </Tooltip>
            <Tooltip title={refs?.length ? `${t('pages.datasets.referencedTip')}: ${refs.join(', ')}` : t('common.delete')}>
              <Button
                type="text"
                size="small"
                danger
                icon={<DeleteOutlined />}
                disabled={!!refs?.length}
                onClick={() => handleDelete(record)}
              />
            </Tooltip>
          </Space>
        );
      },
    },
  ];

  const handleCreate = () => {
    setEditingDataset(null);
    form.resetFields();
    setModalOpen(true);
  };

  const handleEdit = (dataset: Dataset) => {
    setEditingDataset(dataset);
    form.setFieldsValue({
      name: dataset.name,
      description: dataset.description || '',
    });
    setModalOpen(true);
  };

  const handleView = (dataset: Dataset) => {
    // SPA navigate to version page (avoids full page reload that remounts Suspense)
    navigate(`/datasets/${dataset.id}/versions`);
  };

  const handleDelete = async (dataset: Dataset) => {
    if (window.confirm(t('common.confirmDelete') || `Delete ${dataset.name}?`)) {
      try {
        await datasetApi.delete(dataset.id);
        fetchDatasets();
      } catch (error) {
        console.error('Delete failed:', error);
      }
    }
  };

  const handleSubmit = async (values: { name: string; description: string }) => {
    try {
      if (editingDataset) {
        // Format is fixed after creation; only name/description are editable.
        await datasetApi.update(editingDataset.id, values);
      } else {
        await datasetApi.create(values);
      }
      setModalOpen(false);
      fetchDatasets();
    } catch (error) {
      console.error('Save failed:', error);
    }
  };

  return (
    <div>
      <PageHeader
        title={t('pages.datasets.title')}
        onCreate={handleCreate}
        onRefresh={fetchDatasets}
        actions={[<HelpButton module="datasets" />]}
      />
      <Card>
        <FilterSearchBar
          filterPlaceholder={t('pages.datasets.filterStatus')}
          filterValue={statusFilter}
          onFilterChange={setStatusFilter}
          filterOptions={[
            { label: t('pages.datasets.statusDraft'), value: DatasetStatus.DRAFT },
            { label: t('pages.datasets.statusActive'), value: DatasetStatus.ACTIVE },
            { label: t('pages.datasets.statusArchived'), value: DatasetStatus.ARCHIVED },
          ]}
          searchPlaceholder={t('pages.datasets.searchPlaceholder')}
          searchValue={searchInput}
          onSearchChange={setSearchInput}
        />
        <Table
          {...table.tableProps}
          columns={columns}
        scroll={{ x: 'max-content' }}
        />
      </Card>

      <Modal
        title={editingDataset ? t('common.edit') : t('common.create')}
        open={modalOpen}
        onOk={() => form.validateFields().then(handleSubmit)}
        onCancel={() => setModalOpen(false)}
      >
        <Form form={form} layout="vertical">
          <Form.Item
            name="name"
            label={t('pages.datasets.datasetName')}
            rules={[{ required: true, message: t('common.required') }]}
          >
            <Input placeholder={t('pages.datasets.datasetName')} />
          </Form.Item>
          <Form.Item
            name="description"
            label={t('pages.datasets.description')}
          >
            <Input.TextArea rows={3} placeholder={t('pages.datasets.description')} />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

const formatBytes = (bytes: number): string => {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

// Format tag shared by the dataset list and the version list. Empty format
// means the dataset is undecided until content detection runs on finalize.
const FormatTag: React.FC<{ format: string }> = ({ format: f }) => {
  const { t } = useTranslation();
  if (f === 'llm-sft') {
    return <Tag color="purple">{t('pages.datasets.formatLlmSft')}</Tag>;
  }
  if (f) return <>{f}</>;
  return <Tag>{t('pages.datasets.formatAuto')}</Tag>;
};

export const DatasetVersions: React.FC = () => {
  const { t } = useTranslation();
  const { id: datasetId } = useParams<{ id: string }>();
  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [versions, setVersions] = useState<DatasetVersion[]>([]);
  const [loading, setLoading] = useState(false);
  const [downloadPct, setDownloadPct] = useState<{ id: string; pct: number } | null>(null);
  const [uploadFiles, setUploadFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<number | null>(null);
  const [uploadOpen, setUploadOpen] = useState(false);
  // "Register S3 prefix" modal: attaches an already-uploaded S3 prefix as a
  // finalized trainable version without copying any bytes.
  const [registerOpen, setRegisterOpen] = useState(false);
  const [registering, setRegistering] = useState(false);
  const [registerForm] = Form.useForm();
  const [editOpen, setEditOpen] = useState(false);
  const [editingVersion, setEditingVersion] = useState<DatasetVersion | null>(null);
  const [editForm] = Form.useForm();
  const [previewOpen, setPreviewOpen] = useState(false);
  const [previewVersion, setPreviewVersion] = useState<DatasetVersion | null>(null);
  // Dataset browser drawer (HuggingFace-style): version switcher + files/card tabs.
  const [browseTab, setBrowseTab] = useState('files');
  const [versionFiles, setVersionFiles] = useState<VersionFile[]>([]);
  const [filesLoading, setFilesLoading] = useState(false);
  const [cardData, setCardData] = useState<{ raw: string; readme: string; frontmatter: Record<string, any> } | null>(null);
  const [cardLoading, setCardLoading] = useState(false);
  const [editingCard, setEditingCard] = useState(false);
  const [cardText, setCardText] = useState('');
  const [savingCard, setSavingCard] = useState(false);
  // Right-hand preview pane: the selected file plus its text content (media is
  // rendered straight from mediaURL(), so only text needs fetched content).
  const [previewFile, setPreviewFile] = useState<VersionFile | null>(null);
  const [previewText, setPreviewText] = useState<string | null>(null);
  const [fileLoading, setFileLoading] = useState(false);

  const fetchData = useCallback(async () => {
    if (!datasetId) return;
    setLoading(true);
    try {
      const [dsRes, vRes] = await Promise.all([
        datasetApi.get(datasetId),
        datasetApi.getVersions(datasetId, { page_size: 100 }),
      ]);
      setDataset(dsRes.data);
      setVersions(vRes.data.items);
    } catch (e) {
      console.error('Failed to fetch dataset versions:', e);
    } finally {
      setLoading(false);
    }
  }, [datasetId]);

  // Current draft (version record with is_draft=true)
  const currentDraft = versions.find(v => v.is_draft) || null;

  // Build version tree: roots = non-derived versions (no parent or parent not in list), derived versions nested as children
  const buildTree = (): DatasetVersion[] => {
    const roots = versions.filter(v => !v.parent_version_id || !versions.some(p => p.id === v.parent_version_id));
    const childrenOf = (id: string): DatasetVersion[] => versions.filter(v => v.parent_version_id === id);
    const attach = (node: DatasetVersion, depth: number): any => {
      const kids = childrenOf(node.id);
      if (kids.length) {
        return { ...node, children: kids.map(k => attach(k, depth + 1)) };
      }
      return node;
    };
    return roots.map(r => attach(r, 0));
  };
  const treeData = buildTree();

  // Create new draft
  const handleCreateDraft = async () => {
    if (!datasetId) return;
    try {
      await datasetApi.createDraft(datasetId, { description: t('pages.datasets.draftDesc') });
      fetchData();
    } catch (e) {
      console.error('Create draft failed:', e);
    }
  };

  // Finalize the draft.
  const handleFinalizeDraft = async (draftId: string) => {
    if (!datasetId) return;
    try {
      await datasetApi.finalizeDraft(datasetId, draftId);
      message.success(t('pages.datasets.draftFinalized'));
      fetchData();
    } catch (e) {
      console.error('Finalize draft failed:', e);
    }
  };

  // Copy version to new draft
  const handleCopyToDraft = async (versionId: string) => {
    if (!datasetId) return;
    try {
      await datasetApi.copyVersionToDraft(datasetId, versionId);
      message.success(t('pages.datasets.copiedToDraft'));
      fetchData();
    } catch (e) {
      console.error('Copy to draft failed:', e);
    }
  };

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const handleDownloadStream = async (version: DatasetVersion) => {
    try {
      const token = useAuthStore.getState().accessToken;
      const tenant = useAuthStore.getState().tenantId;
      const url = `/api/v1/datasets/${datasetId}/download?version=${version.version}`;
      const resp = await fetch(url, {
        headers: {
          Authorization: token ? `Bearer ${token}` : '',
          'X-Tenant-Id': tenant || 'default',
        },
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const contentLength = Number(resp.headers.get('Content-Length') || 0);
      const reader = resp.body!.getReader();
      const chunks: BlobPart[] = [];
      let received = 0;
      let chunk = await reader.read();
      while (!chunk.done) {
        const { value } = chunk;
        chunks.push(value);
        received += value.length;
        if (contentLength > 0) {
          setDownloadPct({ id: version.id, pct: Math.round((received / contentLength) * 100) });
        }
        chunk = await reader.read();
      }
      const blob = new Blob(chunks, { type: 'application/zip' });
      const dl = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = dl;
      a.download = `${dataset?.name || 'dataset'}-v${version.version}.zip`;
      a.click();
      URL.revokeObjectURL(dl);
      setDownloadPct(null);
    } catch (e) {
      console.error('Stream download failed:', e);
    }
  };

  const handleEditVersion = (v: DatasetVersion) => {
    setEditingVersion(v);
    editForm.setFieldsValue({ description: v.description });
    setEditOpen(true);
  };

  const handleEditVersionSubmit = async () => {
    if (!editingVersion || !datasetId) return;
    try {
      const values = await editForm.validateFields();
      await datasetApi.updateVersion(datasetId, editingVersion.version, { description: values.description });
      setEditOpen(false);
      fetchData();
    } catch (e) {
      console.error('Edit version failed:', e);
    }
  };

  const handleDeleteVersion = async (v: DatasetVersion) => {
    if (!datasetId) return;
    try {
      await datasetApi.deleteVersion(datasetId, v.version);
      fetchData();
    } catch (e) {
      console.error('Delete version failed:', e);
    }
  };

  // Delete draft (including seaweedfs data)
  const handleDeleteDraft = async (v: DatasetVersion) => {
    if (!datasetId) return;
    try {
      await datasetApi.deleteDraft(datasetId, v.id);
      message.success(t('pages.datasets.draftDeleted'));
      fetchData();
    } catch (e) {
      console.error('Delete draft failed:', e);
    }
  };

  // Upload files to current draft (without creating a new version)
  const handleUploadToDraft = async () => {
    if (!datasetId) return;
    try {
      await datasetApi.createDraft(datasetId, {}); // ensure a draft exists
      if (uploadFiles.length === 0) {
        message.warning(t('pages.datasets.chooseFiles'));
        return;
      }
      const totalBytes = uploadFiles.reduce((s, f) => s + f.size, 0);
      if (uploadFiles.length > MAX_UPLOAD_FILES) {
        message.error(t('pages.datasets.uploadTooMany', { count: MAX_UPLOAD_FILES }));
        return;
      }
      if (totalBytes > MAX_UPLOAD_BYTES) {
        message.error(t('pages.datasets.uploadTooLarge', { limit: formatBytes(MAX_UPLOAD_BYTES) }));
        return;
      }
      setUploading(true);
      setUploadProgress(0);
      try {
        await datasetApi.uploadFiles(datasetId!, uploadFiles, setUploadProgress);
      } catch (e) {
        console.error('Upload to draft failed:', e);
        message.error(t('pages.datasets.uploadFailed'));
        return;
      }
      setUploadOpen(false);
      setUploadFiles([]);
      setUploadProgress(null);
      fetchData();
    } catch (e) {
      console.error('Upload to draft failed:', e);
    } finally {
      setUploading(false);
    }
  };

  // Register an existing S3 prefix as a finalized trainable version (no copy).
  const handleRegisterPrefix = async () => {
    if (!datasetId) return;
    try {
      const values = await registerForm.validateFields();
      setRegistering(true);
      const res = await datasetApi.registerVersion(datasetId, {
        uri: values.uri.trim(),
        format: values.format || 'auto',
        description: values.description || '',
      });
      message.success(t('pages.datasets.registerOk', { v: res.data.version }));
      setRegisterOpen(false);
      registerForm.resetFields();
      fetchData();
    } catch (e: any) {
      if (e?.errorFields) return; // form validation error, antd already shows it
      console.error('Register prefix failed:', e);
      message.error(e?.response?.data?.message || t('pages.datasets.registerFailed'));
    } finally {
      setRegistering(false);
    }
  };

  const handlePreview = async (v: DatasetVersion) => {
    if (!datasetId) return;
    setPreviewVersion(v);
    setPreviewOpen(true);
    setBrowseTab('files');
    setPreviewFile(null);
    setPreviewText(null);
    setEditingCard(false);
    await Promise.all([
      loadVersionFiles(v),
      loadVersionCard(v),
    ]);
  };

  // Switch the drawer to a different version (draft or an older v).
  const switchBrowseVersion = async (versionKey: string) => {
    const v = versions.find((x) => browseVersionKey(x) === versionKey);
    if (!v) return;
    setPreviewVersion(v);
    setPreviewFile(null);
    setPreviewText(null);
    setEditingCard(false);
    await Promise.all([
      loadVersionFiles(v),
      loadVersionCard(v),
    ]);
  };

  const browseVersionKey = (v: DatasetVersion) => (v.is_draft ? `draft-${v.id}` : `v${v.version}`);

  // Load the version's full file list for the file-browser tab.
  const loadVersionFiles = async (v: DatasetVersion) => {
    if (!datasetId) return;
    setFilesLoading(true);
    try {
      const res = await datasetApi.getVersionFiles(datasetId, v.version);
      setVersionFiles(res.data.files || []);
    } catch {
      setVersionFiles([]);
    } finally {
      setFilesLoading(false);
    }
  };

  // Load the README.md dataset card (markdown + frontmatter).
  const loadVersionCard = async (v: DatasetVersion) => {
    if (!datasetId) return;
    setCardLoading(true);
    try {
      const res = await datasetApi.getVersionCard(datasetId, v.version);
      setCardData({ raw: res.data.raw || '', readme: res.data.readme || '', frontmatter: res.data.frontmatter || {} });
    } catch {
      setCardData({ raw: '', readme: '', frontmatter: {} });
    } finally {
      setCardLoading(false);
    }
  };

  // Enter card edit mode, pre-filling the textarea with the raw README.md.
  const startEditCard = () => {
    setCardText(cardData?.raw || '');
    setEditingCard(true);
  };

  // Persist the edited README.md and refresh the parsed card view.
  const saveCard = async () => {
    if (!datasetId || !previewVersion) return;
    setSavingCard(true);
    try {
      const res = await datasetApi.saveVersionCard(datasetId, previewVersion.version, cardText);
      setCardData({ raw: res.data.raw || '', readme: res.data.readme || '', frontmatter: res.data.frontmatter || {} });
      setEditingCard(false);
      message.success(t('pages.datasets.cardSaved'));
    } catch (e: any) {
      message.error(e?.response?.data?.message || t('common.error'));
    } finally {
      setSavingCard(false);
    }
  };

  // Select a file in the tree: media renders inline, text content is fetched.
  const openPreviewFile = async (f: VersionFile) => {
    if (!datasetId || !previewVersion) return;
    setPreviewFile(f);
    if (f.is_text) {
      setPreviewText(null);
      setFileLoading(true);
      try {
        const res = await datasetApi.getVersionFile(datasetId, previewVersion.version, f.path);
        setPreviewText(res.data.content);
      } catch {
        setPreviewText(null);
        message.error(t('common.error'));
      } finally {
        setFileLoading(false);
      }
    } else {
      // Media / binary: no text fetch; the pane renders <img>/<video> via mediaURL.
      setPreviewText(null);
      setFileLoading(false);
    }
  };

  // Delete an image/video (draft versions only), then refresh the file tree.
  const handleDeletePreviewImage = async (path: string) => {
    if (!datasetId || !previewVersion) return;
    try {
      await datasetApi.deleteVersionImage(datasetId, previewVersion.version, path);
      message.success(t('pages.datasets.imageDeleted'));
      setPreviewFile(null);
      setPreviewText(null);
      await loadVersionFiles(previewVersion);
      fetchData();
    } catch (e: any) {
      message.error(e?.response?.data?.message || t('common.error'));
    }
  };

  // Media URL proxied through s3-object so draft/old versions resolve correctly.
  const mediaURL = (path: string): string => {
    const base = previewVersion?.uri || '';
    const uri = base.endsWith('/') ? `${base}${path}` : `${base}/${path}`;
    return `/api/v1/s3-object?uri=${encodeURIComponent(uri)}`;
  };

  // Build an antd Tree data structure from the flat file list (group by folder).
  const fileTreeData = (): any[] => {
    const root: Record<string, any> = {};
    for (const f of versionFiles) {
      const parts = f.path.split('/');
      let node = root;
      parts.forEach((part, idx) => {
        const isLeaf = idx === parts.length - 1;
        const key = parts.slice(0, idx + 1).join('/');
        if (!node[part]) {
          node[part] = { __key: key, __leaf: isLeaf, __file: isLeaf ? f : null, children: {} };
        }
        node = node[part].children;
      });
    }
    const build = (children: Record<string, any>): any[] =>
      Object.entries(children).map(([name, val]) => {
        const leaf = val.__leaf;
        return {
          key: val.__key,
          title: leaf ? (
            <Space size={4}>
              <span>{name}</span>
              <span style={{ color: '#aaa', fontSize: 11 }}>{formatBytes(val.__file.size)}</span>
            </Space>
          ) : name,
          icon: leaf
            ? (val.__file.is_media ? <PictureOutlined /> : val.__file.is_text ? <FileTextOutlined /> : <FileOutlined />)
            : <FolderOutlined />,
          isLeaf: leaf,
          selectable: leaf,
          __file: leaf ? val.__file : null,
          children: leaf ? undefined : build(val.children),
        };
      });
    return build(root);
  };
  const columns = [
    {
      title: t('pages.datasets.version'),
      key: 'version',
      render: (_: unknown, r: DatasetVersion) => (
        <Space size={4}>
          {r.is_draft ? <Tag color="orange">{t('pages.datasets.draft')}</Tag> : <Tag color="blue">v{r.version}</Tag>}
          {r.format === 'llm-sft'
            ? <Tag color="purple">{t('pages.datasets.formatLlmSft')}</Tag>
            : r.is_annotated && <Tag color="green">{t('pages.datasets.annotated')}</Tag>}
        </Space>
      ),
    },
    {
      title: t('pages.datasets.description'),
      dataIndex: 'description',
      key: 'description',
      render: (d: string) => d || '-',
    },
    {
      title: t('pages.datasets.format'),
      dataIndex: 'format',
      key: 'format',
      render: (f: string) => <FormatTag format={f || ''} />,
    },
    { title: 'URI', dataIndex: 'uri', key: 'uri', ellipsis: true },
    {
      title: t('pages.datasets.fileCount'),
      dataIndex: 'file_count',
      key: 'file_count',
      render: (n: number) => n ?? 0,
    },
    {
      title: t('pages.datasets.size'),
      dataIndex: 'size_bytes',
      key: 'size_bytes',
      render: (size: number) => formatBytes(size),
    },
    { title: 'Commit', dataIndex: 'commit_hash', key: 'commit_hash', render: (h: string) => h?.slice(0, 8) || '-' },
    {
      title: t('pages.datasets.createdAt'),
      dataIndex: 'created_at',
      key: 'created_at',
      render: (d: string) => formatDateTime(d),
    },
    {
      title: t('common.actions'),
      key: 'actions',
      render: (_: unknown, r: DatasetVersion) => (
        <Space size={4}>
          <Tooltip title={t('pages.datasets.preview')}>
            <Button size="small" type="text" icon={<EyeOutlined />} onClick={() => handlePreview(r)} />
          </Tooltip>
          <Tooltip title={t('common.download')}>
            <Button
              size="small"
              type="text"
              icon={<DownloadOutlined />}
              loading={downloadPct?.id === r.id}
              onClick={() => handleDownloadStream(r)}
            >
              {downloadPct?.id === r.id ? `${downloadPct.pct}%` : ''}
            </Button>
          </Tooltip>
          {r.is_draft ? (
            <>
              <Tooltip title={t('pages.datasets.uploadToDraft')}>
                <Button size="small" type="text" icon={<UploadOutlined />} onClick={() => { setUploadFiles([]); setUploadOpen(true); }} />
              </Tooltip>
              <Tooltip title={t('pages.datasets.finalize')}>
                <Button size="small" type="primary" icon={<CheckOutlined />} onClick={() => handleFinalizeDraft(r.id)} />
              </Tooltip>
              <Tooltip title={t('common.edit')}>
                <Button size="small" type="text" icon={<EditOutlined />} onClick={() => handleEditVersion(r)} />
              </Tooltip>
              <Tooltip title={t('common.delete')}>
                <Popconfirm title={t('pages.datasets.deleteVersionConfirm')} onConfirm={() => handleDeleteDraft(r)}>
                  <Button size="small" type="text" danger icon={<DeleteOutlined />} />
                </Popconfirm>
              </Tooltip>
            </>
          ) : (
            <>
              <Tooltip title={t('pages.datasets.copyToDraft')}>
                <Button size="small" type="text" icon={<CopyOutlined />} onClick={() => handleCopyToDraft(r.id)} />
              </Tooltip>
              <Tooltip title={t('common.edit')}>
                <Button size="small" type="text" icon={<EditOutlined />} onClick={() => handleEditVersion(r)} />
              </Tooltip>
              {r.version !== dataset?.version && (
                <Tooltip title={t('common.delete')}>
                  <Popconfirm title={t('pages.datasets.deleteVersionConfirm')} onConfirm={() => handleDeleteVersion(r)}>
                    <Button size="small" type="text" danger icon={<DeleteOutlined />} />
                  </Popconfirm>
                </Tooltip>
              )}
            </>
          )}
        </Space>
      ),
    },
  ];

  return (
    <div>
      <PageHeader
        title={`${dataset?.name || 'Dataset'} - Versions`}
        subtitle={dataset?.description}
        breadcrumbs={[{ title: t('pages.datasets.title'), href: '/datasets' }]}
        onCreate={handleCreateDraft}
        onRefresh={fetchData}
        actions={[
          <Button key="register" icon={<FolderOutlined />} onClick={() => { registerForm.resetFields(); setRegisterOpen(true); }}>
            {t('pages.datasets.registerPrefix')}
          </Button>,
        ]}
      />
      {currentDraft && (
        <Alert
          type="info"
          showIcon
          style={{ marginBottom: 16 }}
          message={t('pages.datasets.draftHint')}
          description={t('pages.datasets.draftHintDesc')}
        />
      )}
      <Card>
        <Table
          columns={columns}
          dataSource={treeData}
          loading={loading}
          rowKey="id"
          pagination={false}
          defaultExpandAllRows
          expandable={{ indentSize: 24 }}
          scroll={{ x: 'max-content' }}
        />
      </Card>

      <Modal
        title={`${t('pages.datasets.editVersion')} v${editingVersion?.version}`}
        open={editOpen}
        onOk={handleEditVersionSubmit}
        onCancel={() => setEditOpen(false)}
        destroyOnHidden
      >
        <Form form={editForm} layout="vertical">
          <Form.Item name="description" label={t('pages.datasets.description')}>
            <Input.TextArea rows={3} placeholder={t('pages.datasets.description')} />
          </Form.Item>
        </Form>
      </Modal>

      <Modal
        title={t('pages.datasets.uploadToDraft')}
        open={uploadOpen}
        onOk={handleUploadToDraft}
        onCancel={() => { setUploadOpen(false); setUploadFiles([]); }}
        confirmLoading={uploading}
        destroyOnHidden
      >
        <Form layout="vertical">
          <Form.Item label={t('pages.datasets.uploadFiles')}>
            <Upload
              multiple
              beforeUpload={(file) => {
                if (uploading) return false;
                setUploadFiles((prev) => [...prev, file]);
                return false;
              }}
              onRemove={(file) => setUploadFiles((prev) => prev.filter((f) => f.name !== file.name))}
              fileList={uploadFiles.map((f) => ({ uid: f.name, name: f.name, status: 'done' as const }))}
              accept=".jpg,.jpeg,.png,.bmp,.webp,.mp4,.avi,.mov,.mkv,.webm,.flv,.wmv,.m4v,.zip,.json,.jsonl,.txt,.csv,.md,.xml"
            >
              <Button icon={<UploadOutlined />}>{t('pages.datasets.chooseFiles')}</Button>
            </Upload>
            <Alert
              type="info"
              showIcon={false}
              style={{ marginTop: 8 }}
              message={t('pages.datasets.uploadZipHint')}
            />
            {uploading && uploadProgress !== null && <Progress percent={uploadProgress} />}
          </Form.Item>
        </Form>
      </Modal>

      <Modal
        title={t('pages.datasets.registerPrefixTitle')}
        open={registerOpen}
        onOk={handleRegisterPrefix}
        onCancel={() => setRegisterOpen(false)}
        confirmLoading={registering}
        destroyOnHidden
      >
        <Form form={registerForm} layout="vertical" initialValues={{ format: 'auto' }}>
          <Form.Item
            name="uri"
            label={t('pages.datasets.registerUri')}
            rules={[{ required: true, message: t('common.required') }]}
            extra={t('pages.datasets.registerUriHint')}
          >
            <Input placeholder={t('pages.datasets.registerUriPlaceholder')} />
          </Form.Item>
          <Form.Item name="format" label={t('pages.datasets.registerFormat')}>
            <Select
              options={[
                { value: 'auto', label: t('pages.datasets.registerFormatAuto') },
                { value: 'llm-sft', label: t('pages.datasets.registerFormatLlmSft') },
                { value: 'coco', label: t('pages.datasets.registerFormatCoco') },
                { value: 'yolo', label: t('pages.datasets.registerFormatYolo') },
              ]}
            />
          </Form.Item>
          <Form.Item name="description" label={t('pages.datasets.description')}>
            <Input.TextArea rows={2} />
          </Form.Item>
        </Form>
      </Modal>

      <Drawer
        title={
          <Space>
            <span>{t('pages.datasets.browseTitle')}</span>
            <Select
              size="small"
              style={{ minWidth: 140 }}
              value={previewVersion ? browseVersionKey(previewVersion) : undefined}
              onChange={switchBrowseVersion}
              options={versions
                .slice()
                .sort((a, b) => b.version - a.version)
                .map((v) => ({
                  value: browseVersionKey(v),
                  label: v.is_draft
                    ? `${t('pages.datasets.draft')} (v${v.version})`
                    : `v${v.version}${v.is_annotated ? ' ✓' : ''}`,
                }))}
            />
          </Space>
        }
        open={previewOpen}
        onClose={() => setPreviewOpen(false)}
        width={1000}
        destroyOnClose
      >
        <Tabs
          activeKey={browseTab}
          onChange={setBrowseTab}
          items={[
            {
              key: 'files',
              label: <span><FolderOutlined /> {t('pages.datasets.tabFiles')}</span>,
              children: (
                <Spin spinning={filesLoading}>
                  <div style={{ display: 'flex', gap: 16, alignItems: 'flex-start' }}>
                    <div style={{ flex: '0 0 300px', borderRight: '1px solid #f0f0f0', paddingRight: 12, maxHeight: '64vh', overflow: 'auto' }}>
                      <Tree
                        showIcon
                        defaultExpandAll
                        treeData={fileTreeData()}
                        selectedKeys={previewFile ? [previewFile.path] : []}
                        onSelect={(_keys, info) => {
                          const f = (info.node as any)?.__file as VersionFile | undefined;
                          if (f) openPreviewFile(f);
                        }}
                      />
                    </div>
                    <div style={{ flex: 1, minWidth: 0 }}>
                      {fileLoading ? (
                        <Spin />
                      ) : !previewFile ? (
                        <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={t('pages.datasets.selectFile')} />
                      ) : previewFile.is_media ? (
                        (() => {
                          const isVideo = /\.(mp4|avi|mov|mkv|webm|flv|wmv|m4v)$/i.test(previewFile.path);
                          return (
                            <div>
                              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
                                <span style={{ fontWeight: 600, wordBreak: 'break-all' }}>{previewFile.path}</span>
                                {previewVersion?.is_draft && (
                                  <Popconfirm title={t('pages.datasets.deleteImageConfirm')} onConfirm={() => handleDeletePreviewImage(previewFile.path)}>
                                    <Button size="small" danger icon={<DeleteOutlined />}>{t('common.delete')}</Button>
                                  </Popconfirm>
                                )}
                              </div>
                              {isVideo ? (
                                <video src={mediaURL(previewFile.path)} controls style={{ maxWidth: '100%', maxHeight: '56vh', borderRadius: 4, border: '1px solid #f0f0f0' }} />
                              ) : (
                                <img src={mediaURL(previewFile.path)} alt={previewFile.path} style={{ maxWidth: '100%', maxHeight: '56vh', objectFit: 'contain', borderRadius: 4, border: '1px solid #f0f0f0' }} />
                              )}
                            </div>
                          );
                        })()
                      ) : previewFile.is_text ? (
                        <>
                          <div style={{ fontWeight: 600, marginBottom: 8, wordBreak: 'break-all' }}>{previewFile.path}</div>
                          <pre style={{ background: '#f6f8fa', padding: 12, borderRadius: 4, fontSize: 12, maxHeight: '56vh', overflow: 'auto', whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                            {previewText ?? ''}
                          </pre>
                        </>
                      ) : (
                        <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={t('pages.datasets.binaryNoPreview')} />
                      )}
                    </div>
                  </div>
                </Spin>
              ),
            },
            {
              key: 'card',
              label: <span><InfoCircleOutlined /> {t('pages.datasets.tabCard')}</span>,
              children: (
                <Spin spinning={cardLoading}>
                  <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: 8 }}>
                    {editingCard ? (
                      <Space>
                        <Button onClick={() => setEditingCard(false)}>{t('common.cancel')}</Button>
                        <Button type="primary" loading={savingCard} onClick={saveCard}>{t('common.save')}</Button>
                      </Space>
                    ) : (
                      <Button size="small" icon={<EditOutlined />} onClick={startEditCard}>{t('pages.datasets.editCard')}</Button>
                    )}
                  </div>
                  {editingCard ? (
                    <>
                      <Alert type="info" showIcon style={{ marginBottom: 8 }} message={t('pages.datasets.cardEditHint')} />
                      <Input.TextArea
                        value={cardText}
                        onChange={(e) => setCardText(e.target.value)}
                        autoSize={{ minRows: 16, maxRows: 30 }}
                        style={{ fontFamily: 'monospace', fontSize: 13 }}
                        placeholder={'---\ntask: object-detection\nclasses: [person, car]\n---\n# Dataset title\n\nDescription...'}
                      />
                    </>
                  ) : (
                    <>
                      {cardData && Object.keys(cardData.frontmatter).length > 0 && (
                        <Descriptions size="small" bordered column={1} style={{ marginBottom: 16 }}>
                          {Object.entries(cardData.frontmatter).map(([k, val]) => (
                            <Descriptions.Item key={k} label={k}>
                              {Array.isArray(val) ? val.join(', ') : typeof val === 'object' ? JSON.stringify(val) : String(val)}
                            </Descriptions.Item>
                          ))}
                        </Descriptions>
                      )}
                      {cardData?.readme ? (
                        <pre style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word', fontFamily: 'inherit', fontSize: 14, lineHeight: 1.6 }}>
                          {cardData.readme}
                        </pre>
                      ) : (
                        <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={t('pages.datasets.noCard')} />
                      )}
                    </>
                  )}
                </Spin>
              ),
            },
          ]}
        />
      </Drawer>
    </div>
  );
};

export default Datasets;
