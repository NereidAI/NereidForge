// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

import React, { useState } from 'react';
import {
  Button, Card, Form, Input, Modal, Progress, Radio, Select, Table, Tag, Typography, Upload, message,
} from 'antd';
import { DeleteOutlined, PlusOutlined, UploadOutlined } from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import { baseModelPoolApi } from '@api/endpoints';
import { useServerTable } from '@hooks/useServerTable';
import { useDebounce } from '@hooks/useDebounce';
import { BaseModelPoolEntry } from '@nftypes/api';
import PageHeader from '@components/PageHeader';
import FilterSearchBar from '@components/FilterSearchBar';
import { formatDateTime } from '../../utils/format';

// Tag colors for lifecycle statuses.
const STATUS_COLORS: Record<string, string> = {
  original: 'default', develop: 'blue', production: 'green',
};
// Tag colors for pool-entry sources.
const SOURCE_COLORS: Record<string, string> = {
  manual: 'geekblue', 's3-reconcile': 'orange', seed: 'purple',
};
// Tag colors for task tags.
const TASK_COLORS: Record<string, string> = {
  detection: 'blue', segmentation: 'green', classification: 'gold', llm: 'purple', vlm: 'magenta',
};

// Format byte counts for the size column.
function formatBytes(bytes: number): string {
  if (!bytes) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  let v = bytes;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i += 1;
  }
  return `${v.toFixed(v >= 100 || i === 0 ? 0 : 1)} ${units[i]}`;
}

// Base-model pool management: upload cv checkpoints / llm model directories.
const BaseModels: React.FC = () => {
  const { t } = useTranslation();
  const [uploadOpen, setUploadOpen] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [form] = Form.useForm();
  const uploadKind = Form.useWatch('kind', form);
  const [fileList, setFileList] = useState<File[]>([]);
  // Server-side filters: kind tabs plus debounced name search.
  const [kindFilter, setKindFilter] = useState('');
  const [searchInput, setSearchInput] = useState('');
  const search = useDebounce(searchInput.trim(), 300);

  const table = useServerTable<BaseModelPoolEntry, { kind?: string; search?: string }>({
    list: (params) => baseModelPoolApi.list(params),
    defaultPageSize: 10,
    filters: {
      kind: kindFilter || undefined,
      search: search || undefined,
    },
  });

  const handleDelete = async (row: BaseModelPoolEntry) => {
    await baseModelPoolApi.delete(row.id);
    message.success(t('pages.baseModels.deleted'));
    table.reload();
  };

  const handleUpload = async () => {
    const values = await form.validateFields();
    if (!fileList.length) {
      message.warning(t('pages.baseModels.noFiles'));
      return;
    }
    const uploadKind = values.kind as 'cv' | 'llm';
    if (uploadKind === 'llm') {
      const isZip = fileList.length === 1 && fileList[0].name.toLowerCase().endsWith('.zip');
      if (!isZip && !values.name) {
        message.warning(t('pages.baseModels.nameRequired'));
        return;
      }
    }
    setUploading(true);
    setProgress(0);
    try {
      await baseModelPoolApi.upload(
        {
          kind: uploadKind, name: values.name, files: fileList,
          framework: values.framework, task: values.task,
          status: values.status, description: values.description,
        },
        setProgress,
      );
      message.success(t('pages.baseModels.uploaded'));
      setUploadOpen(false);
      setFileList([]);
      table.reload();
    } catch (e: any) {
      message.error(e?.response?.data?.detail || t('pages.baseModels.uploadFailed'));
    } finally {
      setUploading(false);
    }
  };

  const columns = [
    {
      title: t('pages.baseModels.name'),
      dataIndex: 'name',
      key: 'name',
      ellipsis: true,
      render: (v: string, r: BaseModelPoolEntry) => (
        <Typography.Text copyable={r.kind === 'cv' ? { text: r.uri } : undefined}>{v}</Typography.Text>
      ),
    },
    {
      title: t('pages.baseModels.kind'),
      dataIndex: 'kind',
      key: 'kind',
      width: 90,
      render: (v: string) => <Tag color={v === 'cv' ? 'blue' : 'purple'}>{v.toUpperCase()}</Tag>,
    },
    {
      title: t('pages.baseModels.framework'),
      dataIndex: 'framework',
      key: 'framework',
      width: 110,
      render: (v: string) => (v ? v.charAt(0).toUpperCase() + v.slice(1) : '-'),
    },
    {
      title: t('pages.baseModels.task'),
      dataIndex: 'task',
      key: 'task',
      width: 130,
      render: (v: string) => <Tag color={TASK_COLORS[v] || 'default'}>{v}</Tag>,
    },
    {
      title: t('pages.baseModels.status'),
      dataIndex: 'status',
      key: 'status',
      width: 110,
      render: (v: string) => <Tag color={STATUS_COLORS[v] || 'default'}>{v}</Tag>,
    },
    {
      title: t('pages.baseModels.description'),
      dataIndex: 'description',
      key: 'description',
      width: 200,
      ellipsis: true,
      render: (v: string) => v || '-',
    },
    {
      title: t('pages.baseModels.size'),
      dataIndex: 'size_bytes',
      key: 'size_bytes',
      width: 110,
      render: (v: number) => formatBytes(v),
    },
    {
      title: t('pages.baseModels.source'),
      dataIndex: 'source',
      key: 'source',
      width: 130,
      render: (v: string) => <Tag color={SOURCE_COLORS[v] || 'default'}>{v}</Tag>,
    },
    {
      title: t('pages.baseModels.createdAt'),
      dataIndex: 'created_at',
      key: 'created_at',
      width: 170,
      render: (v: string) => formatDateTime(v),
    },
    {
      title: '',
      key: 'actions',
      width: 60,
      render: (_: unknown, r: BaseModelPoolEntry) => (
        <Button type="text" danger size="small" icon={<DeleteOutlined />} onClick={() => handleDelete(r)} />
      ),
    },
  ];

  return (
    <div>
      <PageHeader
        title={t('pages.baseModels.title')}
        subtitle={t('pages.baseModels.subtitle')}
        actions={[
          <Button key="upload" type="primary" icon={<PlusOutlined />} onClick={() => {
            form.resetFields();
            setFileList([]);
            setProgress(0);
            setUploadOpen(true);
          }}>
            {t('pages.baseModels.upload')}
          </Button>,
        ]}
      />
      <Card size="small">
        <FilterSearchBar
          filterPlaceholder={t('pages.baseModels.kind')}
          filterValue={kindFilter}
          onFilterChange={setKindFilter}
          filterOptions={[
            { label: t('pages.baseModels.kindCv'), value: 'cv' },
            { label: t('pages.baseModels.kindLlm'), value: 'llm' },
          ]}
          searchPlaceholder={t('pages.baseModels.searchPlaceholder')}
          searchValue={searchInput}
          onSearchChange={setSearchInput}
        />
        <Table {...table.tableProps} rowKey="id" columns={columns} scroll={{ x: 'max-content' }} />
      </Card>
      <Modal
        title={t('pages.baseModels.upload')}
        open={uploadOpen}
        onOk={handleUpload}
        confirmLoading={uploading}
        onCancel={() => !uploading && setUploadOpen(false)}
        destroyOnHidden
      >
        <Form
          form={form} layout="vertical"
          initialValues={{ kind: 'cv', framework: 'pytorch', task: 'detection', status: 'develop' }}
        >
          <Form.Item name="kind" label={t('pages.baseModels.kind')}>
            <Radio.Group onChange={(e) => form.setFieldValue('task', e.target.value === 'cv' ? 'detection' : 'llm')}>
              <Radio.Button value="cv">{t('pages.baseModels.kindCv')}</Radio.Button>
              <Radio.Button value="llm">{t('pages.baseModels.kindLlm')}</Radio.Button>
            </Radio.Group>
          </Form.Item>
          {uploadKind === 'llm' && (
            <Form.Item name="name" label={t('pages.baseModels.dirName')} extra={t('pages.baseModels.dirNameHint')}>
              <Input placeholder="qwen3" />
            </Form.Item>
          )}
          <Form.Item name="framework" label={t('pages.baseModels.framework')}>
            <Select
              options={[
                { label: 'PyTorch', value: 'pytorch' },
                { label: 'TensorFlow', value: 'tensorflow' },
                { label: 'ONNX', value: 'onnx' },
              ]}
            />
          </Form.Item>
          <Form.Item name="task" label={t('pages.baseModels.task')}>
            <Select
              options={(uploadKind === 'cv'
                ? ['detection', 'segmentation', 'classification']
                : ['llm', 'vlm']
              ).map((v) => ({ label: t(`pages.baseModels.task_${v}`), value: v }))}
            />
          </Form.Item>
          <Form.Item name="status" label={t('pages.baseModels.status')}>
            <Select
              options={['original', 'develop', 'production'].map((v) => (
                { label: t(`pages.baseModels.status_${v}`), value: v }
              ))}
            />
          </Form.Item>
          <Form.Item name="description" label={t('pages.baseModels.description')}>
            <Input.TextArea rows={2} placeholder={t('pages.baseModels.descriptionPlaceholder')} />
          </Form.Item>
          <Upload
            beforeUpload={(file) => {
              setFileList((prev) => [...prev, file]);
              return false;
            }}
            onRemove={(file) => setFileList((prev) => prev.filter((f) => f.name !== file.name))}
            multiple
            accept={uploadKind === 'cv'
              ? '.pt,.pth,.h5,.hdf5,.keras,.pb,.onnx'
              : '.zip,.json,.safetensors,.bin,.txt,.model'}
          >
            <Button icon={<UploadOutlined />}>{t('pages.baseModels.chooseFiles')}</Button>
          </Upload>
          {uploading && <Progress percent={progress} style={{ marginTop: 16 }} />}
        </Form>
      </Modal>
    </div>
  );
};

export default BaseModels;
