// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

import React, { useState } from 'react';
import {
  Card, Table, Button, Space, Modal, Form, Input, Select, Tag, Tooltip,
  message, Popconfirm, Typography,
} from 'antd';
import {
  DeleteOutlined, EditOutlined, RocketOutlined, ExperimentOutlined, SyncOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import { PageHeader } from '@components/PageHeader';
import { imageApi } from '@api/endpoints';
import { useServerTable } from '@hooks/useServerTable';
import { useDebounce } from '@hooks/useDebounce';
import FilterSearchBar from '@components/FilterSearchBar';

const { Text } = Typography;

interface ContainerImage {
  id: string;
  name: string;
  category: string;
  repository: string;
  tag: string;
  full_ref: string;
  description: string;
  build_context: string;
  entrypoint_hint: string;
  task_types: string[];
  profile?: { schema?: string; name?: string } & Record<string, unknown>;
  default_archs?: Record<string, string>;
  extended?: Array<{ id: string; kind: string }>;
  profile_error?: string;
  is_builtin: boolean;
  created_by: string;
  created_at: string;
}

const CATEGORY_META: Record<string, { color: string; icon: React.ReactNode }> = {
  train: { color: 'geekblue', icon: <ExperimentOutlined /> },
  prediction: { color: 'purple', icon: <RocketOutlined /> },
};

export const Images: React.FC = () => {
  const { t } = useTranslation();
  const [modalOpen, setModalOpen] = useState(false);
  const [editing, setEditing] = useState<ContainerImage | null>(null);
  const [categoryFilter, setCategoryFilter] = useState<string>('');
  const [searchInput, setSearchInput] = useState('');
  const search = useDebounce(searchInput.trim(), 300);
  const [form] = Form.useForm();

  // Server-side pagination + category filtering + name search; switching any
  // filter resets to page 1 inside the hook (avoids blank-page-on-filter).
  const table = useServerTable<ContainerImage, { category?: string; search?: string }>({
    list: (params) => imageApi.list(params),
    filters: {
      category: categoryFilter || undefined,
      search: search || undefined,
    },
    defaultPageSize: 10,
    rowKey: 'id',
  });
  const { reload } = table;

  const openCreate = () => {
    setEditing(null);
    form.resetFields();
    form.setFieldsValue({ category: 'train', tag: 'latest' });
    setModalOpen(true);
  };

  const openEdit = (img: ContainerImage) => {
    setEditing(img);
    form.setFieldsValue({
      name: img.name,
      tag: img.tag,
      description: img.description,
      repository: img.repository,
      entrypoint_hint: img.entrypoint_hint,
      build_context: img.build_context,
      task_types: img.task_types,
    });
    setModalOpen(true);
  };

  const handleSubmit = async () => {
    try {
      const values = await form.validateFields();
      // Announcement document is pasted as JSON text; it is not a column.
      const profileText = String(values._profile_text || '').trim();
      delete values._profile_text;
      let profileDoc: Record<string, unknown> | null = null;
      if (profileText) {
        try {
          profileDoc = JSON.parse(profileText);
        } catch {
          message.error(t('pages.images.profileInvalidJson'));
          return;
        }
      }
      let imageId = editing?.id;
      if (editing) {
        await imageApi.update(editing.id, values);
      } else {
        const created = await imageApi.create(values);
        imageId = created.data.id;
      }
      if (profileDoc && imageId) {
        await imageApi.setProfile(imageId, profileDoc);
      }
      message.success(editing ? t('pages.images.updated') : t('pages.images.created'));
      setModalOpen(false);
      reload();
    } catch (e: unknown) {
      const msg = (e as { response?: { data?: { detail?: string; message?: string } } })?.response?.data;
      if (msg) message.error(msg.detail || msg.message || t('common.error'));
    }
  };

  const handleRefreshProfile = async (img: ContainerImage) => {
    try {
      await imageApi.refreshProfile(img.id);
      message.success(t('pages.images.profileRefreshed'));
      reload();
    } catch (e: unknown) {
      const msg = (e as { response?: { data?: { detail?: string } } })?.response?.data;
      message.error(msg?.detail || t('common.error'));
    }
  };

  const handleDelete = async (img: ContainerImage) => {
    try {
      await imageApi.delete(img.id);
      message.success(t('pages.images.deleted'));
      reload();
    } catch (e: unknown) {
      const msg = (e as { response?: { data?: { detail?: string } } })?.response?.data;
      message.error(msg?.detail || t('common.error'));
    }
  };

  const columns = [
    {
      title: t('pages.images.name'),
      dataIndex: 'name',
      key: 'name',
      render: (name: string, r: ContainerImage) => (
        <Space>
          <span>{CATEGORY_META[r.category]?.icon}</span>
          <span>{name}</span>
          {r.is_builtin && <Tag color="gold">{t('pages.images.builtin')}</Tag>}
        </Space>
      ),
    },
    {
      title: t('pages.images.category'),
      dataIndex: 'category',
      key: 'category',
      width: 130,
      render: (c: string) => (
        <Tag color={CATEGORY_META[c]?.color}>{c}</Tag>
      ),
    },
    {
      title: t('pages.images.imageRef'),
      dataIndex: 'full_ref',
      key: 'full_ref',
      ellipsis: true,
      render: (ref: string) => (
        <Tooltip title={t('pages.images.copyRef')}>
          <Text
            copyable={{ text: ref, tooltips: [t('pages.images.copyRef'), t('pages.images.copied')] }}
            style={{ fontFamily: 'monospace', fontSize: 12 }}
          >
            {ref}
          </Text>
        </Tooltip>
      ),
    },
    {
      title: t('pages.images.taskTypes'),
      dataIndex: 'task_types',
      key: 'task_types',
      width: 220,
      render: (ts: string[], r: ContainerImage) => (
        <Space size={4} wrap>
          {(ts || []).map((x) => <Tag key={x} style={{ fontSize: 11 }}>{x}</Tag>)}
          {r.profile?.schema && (
            <Tooltip title={r.profile_error || t('pages.images.announcedHint')}>
              <Tag color="green" style={{ fontSize: 11 }}>
                {t('pages.images.announced')}{(r.extended?.length || 0) > 0 ? ` · ${r.extended?.length}` : ''}
              </Tag>
            </Tooltip>
          )}
        </Space>
      ),
    },
    {
      title: t('pages.images.buildContext'),
      dataIndex: 'build_context',
      key: 'build_context',
      width: 150,
      ellipsis: true,
      render: (bc: string) => bc ? <Text code style={{ fontSize: 11 }}>{bc}</Text> : <Text type="secondary">-</Text>,
    },
    {
      title: '',
      key: 'actions',
      width: 160,
      render: (_: unknown, r: ContainerImage) => (
        <Space size={4}>
          <Button size="small" type="text" icon={<EditOutlined />} onClick={() => openEdit(r)} />
          {r.is_builtin && r.build_context && (
            <Tooltip title={t('pages.images.profileRefresh')}>
              <Button
                size="small"
                type="text"
                icon={<SyncOutlined />}
                onClick={() => handleRefreshProfile(r)}
              />
            </Tooltip>
          )}
          {!r.is_builtin && (
            <Popconfirm title={t('pages.images.deleteConfirm')} onConfirm={() => handleDelete(r)}>
              <Button size="small" type="text" danger icon={<DeleteOutlined />} />
            </Popconfirm>
          )}
        </Space>
      ),
    },
  ];

  return (
    <div>
      <PageHeader
        title={t('pages.images.title')}
        subtitle={t('pages.images.subtitle')}
        onRefresh={reload}
        onCreate={openCreate}
      />
      <Card>
        <FilterSearchBar
          filterPlaceholder={t('pages.images.filterCategory')}
          filterValue={categoryFilter}
          onFilterChange={setCategoryFilter}
          filterOptions={[
            { value: 'train', label: t('pages.images.catTrain') },
            { value: 'prediction', label: t('pages.images.catPrediction') },
          ]}
          searchPlaceholder={t('pages.images.searchByName')}
          searchValue={searchInput}
          onSearchChange={setSearchInput}
        />
        <Table
          {...table.tableProps}
          size="small"
          columns={columns}
        />
      </Card>

      <Modal
        title={editing ? t('pages.images.editTitle') : t('pages.images.createTitle')}
        open={modalOpen}
        onOk={handleSubmit}
        onCancel={() => setModalOpen(false)}
        destroyOnHidden
      >
        <Form form={form} layout="vertical">
          <Form.Item name="name" label={t('pages.images.name')} rules={[{ required: true }]}>
            <Input placeholder={t('pages.images.namePlaceholder')} />
          </Form.Item>
          <Form.Item
            name="category"
            label={t('pages.images.category')}
            rules={[{ required: true }]}
            extra={t('pages.images.categoryHint')}
          >
            <Select
              disabled={!!editing?.is_builtin}
              options={[
                { value: 'train', label: t('pages.images.catTrain') },
                { value: 'prediction', label: t('pages.images.catPrediction') },
              ]}
            />
          </Form.Item>
          {!editing?.is_builtin && (
            <Form.Item
              name="repository"
              label={t('pages.images.repository')}
              rules={[{ required: true }]}
              extra={t('pages.images.repositoryHint')}
            >
              <Input placeholder="train/my-trainer" disabled={!!editing?.is_builtin} />
            </Form.Item>
          )}
          <Form.Item name="tag" label="Tag" rules={[{ required: true }]}>
            <Input placeholder="v1.0.0" style={{ width: 200 }} />
          </Form.Item>
          <Form.Item name="task_types" label={t('pages.images.taskTypes')}>
            <Select
              mode="tags"
              placeholder="detection, classification, llm..."
              style={{ width: '100%' }}
            />
          </Form.Item>
          <Form.Item name="build_context" label={t('pages.images.buildContext')} extra={t('pages.images.buildContextHint')}>
            <Input placeholder="builders/my-trainer" />
          </Form.Item>
          <Form.Item name="entrypoint_hint" label={t('pages.images.entrypoint')} extra={t('pages.images.entrypointHint')}>
            <Input placeholder="python /app/train.py" />
          </Form.Item>
          <Form.Item name="description" label={t('common.description')}>
            <Input.TextArea rows={2} />
          </Form.Item>
          <Form.Item
            name="_profile_text"
            label={t('pages.images.profileJson')}
            extra={t('pages.images.profileJsonHint')}
          >
            <Input.TextArea
              rows={6}
              placeholder='{"schema": "nf.trainer-profile/1", ...}'
              style={{ fontFamily: 'monospace', fontSize: 12 }}
            />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

export default Images;
