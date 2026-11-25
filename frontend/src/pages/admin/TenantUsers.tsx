// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

import React, { useState } from 'react';
import { Card, Table, Button, Modal, Form, Input, Select, Space, Tag, message, Popconfirm, Switch, List } from 'antd';
import { PlusOutlined, ReloadOutlined, DeleteOutlined, RocketOutlined } from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import { tenantApi, settingsApi } from '@api/endpoints';
import { useServerTable } from '@hooks/useServerTable';
import { useDebounce } from '@hooks/useDebounce';
import FilterSearchBar from '@components/FilterSearchBar';
import type { Tenant, User } from '@nftypes/api';
import { formatDateTime } from '../../utils/format';

const ROLE_OPTIONS = [
  { value: 'admin', label: 'admin' },
  { value: 'tenant-admin', label: 'tenant-admin' },
  { value: 'engineer', label: 'engineer' },
];

export const TenantsManagement: React.FC = () => {
  const { t } = useTranslation();
  // Debounced name search (server-side).
  const [searchInput, setSearchInput] = useState('');
  const search = useDebounce(searchInput.trim(), 300);
  // Server-side paginated tenant list.
  const table = useServerTable<Tenant, { search?: string }>({
    list: (params) => tenantApi.list(params),
    filters: { search: search || undefined },
    defaultPageSize: 10,
  });
  const { reload: load, loading } = table;
  const [createOpen, setCreateOpen] = useState(false);
  const [provSteps, setProvSteps] = useState<string[] | null>(null);
  const [provisioning, setProvisioning] = useState(false);
  const [form] = Form.useForm();

  const handleCreate = async () => {
    const values = await form.validateFields();
    try {
      await tenantApi.create(values);
      message.success(t('pages.settings.tenants.created'));
      setCreateOpen(false);
      form.resetFields();
      load();
    } catch {
      message.error(t('pages.settings.tenants.createFailed'));
    }
  };

  const handleToggle = async (tenant: Tenant) => {
    try {
      await tenantApi.update(tenant.id, { is_active: !tenant.is_active });
      load();
    } catch {
      message.error(t('common.error'));
    }
  };

  const handleDelete = async (id: string) => {
    try {
      await tenantApi.delete(id);
      message.success(t('common.deleteSuccess'));
      load();
    } catch {
      message.error(t('common.error'));
    }
  };

  const handleProvision = async (tenant: Tenant) => {
    setProvisioning(true);
    try {
      const res = await tenantApi.provision(tenant.id);
      setProvSteps(res.data.steps ?? []);
      message.success(t('pages.settings.tenants.provisioned'));
      load();
    } catch (e: any) {
      message.error(e?.response?.data?.message || t('pages.settings.tenants.provisionFailed'));
    } finally {
      setProvisioning(false);
    }
  };

  const columns = [
    { title: t('pages.settings.tenants.name'), dataIndex: 'name', key: 'name' },
    { title: t('pages.settings.tenants.displayName'), dataIndex: 'display_name', key: 'display_name' },
    { title: t('pages.settings.tenants.description'), dataIndex: 'description', key: 'description', ellipsis: true },
    {
      title: t('pages.settings.tenants.adminUser'),
      dataIndex: 'admin_username',
      key: 'admin_username',
      width: 120,
      render: (v: string, r: Tenant) => (
        <Tag color={r.name === 'default' ? 'gold' : 'blue'}>{r.name === 'default' ? `admin (${t('pages.settings.tenants.superAdmin')})` : v || '-'}</Tag>
      ),
    },
    {
      title: t('pages.settings.tenants.namespace'),
      dataIndex: 'namespace',
      key: 'namespace',
      width: 190,
      render: (v: string) => <span style={{ fontFamily: 'monospace', fontSize: 12 }}>{v}</span>,
    },
    {
      title: t('pages.settings.tenants.status'),
      dataIndex: 'is_active',
      key: 'is_active',
      render: (active: boolean) => <Tag color={active ? 'green' : 'red'}>{active ? t('common.enabled') : t('common.disabled')}</Tag>,
    },
    {
      title: t('common.createdAt'),
      dataIndex: 'created_at',
      key: 'created_at',
      render: (d: string) => formatDateTime(d),
    },
    {
      title: t('common.actions'),
      key: 'actions',
      render: (_: unknown, record: Tenant) => (
        <Space>
          <Button
            size="small"
            icon={<RocketOutlined />}
            onClick={() => handleProvision(record)}
            loading={provisioning}
            disabled={record.id === 'default'}
          >
            {t('pages.settings.tenants.provision')}
          </Button>
          <Switch checked={record.is_active} onChange={() => handleToggle(record)} checkedChildren={t('common.enabled')} unCheckedChildren={t('common.disabled')} />
          <Popconfirm title={t('pages.settings.tenants.deleteConfirm')} onConfirm={() => handleDelete(record.id)}>
            <Button danger size="small" icon={<DeleteOutlined />} />
          </Popconfirm>
        </Space>
      ),
    },
  ];

  return (
    <Card size="small">
      <Space style={{ marginBottom: 16 }}>
        <Button type="primary" icon={<PlusOutlined />} onClick={() => setCreateOpen(true)}>
          {t('pages.settings.tenants.create')}
        </Button>
        <Button icon={<ReloadOutlined />} onClick={load} loading={loading}>
          {t('common.refresh')}
        </Button>
      </Space>
      <FilterSearchBar
        searchPlaceholder={t('pages.settings.tenants.searchPlaceholder')}
        searchValue={searchInput}
        onSearchChange={setSearchInput}
      />
      <Table {...table.tableProps} columns={columns} scroll={{ x: 'max-content' }} />

      <Modal title={t('pages.settings.tenants.create')} open={createOpen} onOk={handleCreate} onCancel={() => setCreateOpen(false)} destroyOnHidden>
        <Form form={form} layout="vertical">
          <Form.Item name="name" label={t('pages.settings.tenants.name')} rules={[{ required: true }]}>
            <Input placeholder="tenant-a" />
          </Form.Item>
          <Form.Item name="display_name" label={t('pages.settings.tenants.displayName')} rules={[{ required: true }]}>
            <Input />
          </Form.Item>
          <Form.Item name="description" label={t('pages.settings.tenants.description')}>
            <Input.TextArea rows={2} />
          </Form.Item>
          <Form.Item
            name="admin_username"
            label={t('pages.settings.tenants.adminUser')}
            extra={t('pages.settings.tenants.adminUserHint')}
          >
            <Input placeholder="tenant-admin" />
          </Form.Item>
        </Form>
      </Modal>

      <Modal title={t('pages.settings.tenants.provision')} open={!!provSteps} onCancel={() => setProvSteps(null)} footer={<Button onClick={() => setProvSteps(null)}>{t('common.close')}</Button>} destroyOnHidden>
        <List
          size="small"
          dataSource={provSteps ?? []}
          renderItem={(item) => <List.Item>{item}</List.Item>}
        />
      </Modal>
    </Card>
  );
};

export const UsersManagement: React.FC<{ tenants: Tenant[] }> = ({ tenants }) => {
  const { t } = useTranslation();
  // Debounced name search (server-side).
  const [searchInput, setSearchInput] = useState('');
  const search = useDebounce(searchInput.trim(), 300);
  // Server-side paginated user list.
  const table = useServerTable<User, { search?: string }>({
    list: (params) => settingsApi.getUsers(params),
    filters: { search: search || undefined },
    defaultPageSize: 10,
  });
  const { reload: load, loading } = table;
  const [createOpen, setCreateOpen] = useState(false);
  const [form] = Form.useForm();

  const handleCreate = async () => {
    const values = await form.validateFields();
    try {
      await settingsApi.createUser(values);
      message.success(t('pages.settings.users.created'));
      setCreateOpen(false);
      form.resetFields();
      load();
    } catch {
      message.error(t('pages.settings.users.createFailed'));
    }
  };

  const handleToggle = async (user: User) => {
    try {
      await settingsApi.updateUser(user.id, { is_active: !user.is_active });
      load();
    } catch {
      message.error(t('common.error'));
    }
  };

  const columns = [
    { title: t('pages.settings.users.username'), dataIndex: 'username', key: 'username' },
    { title: t('pages.settings.users.fullName'), dataIndex: 'full_name', key: 'full_name' },
    { title: t('pages.settings.users.email'), dataIndex: 'email', key: 'email' },
    {
      title: t('pages.settings.users.roles'),
      dataIndex: 'roles',
      key: 'roles',
      render: (roles: string[]) => roles.map((r) => <Tag key={r}>{r}</Tag>),
    },
    { title: t('pages.settings.users.tenant'), dataIndex: 'tenant_id', key: 'tenant_id' },
    {
      title: t('pages.settings.users.status'),
      dataIndex: 'is_active',
      key: 'is_active',
      render: (active: boolean) => <Tag color={active ? 'green' : 'red'}>{active ? t('common.enabled') : t('common.disabled')}</Tag>,
    },
    {
      title: t('common.actions'),
      key: 'actions',
      render: (_: unknown, record: User) => (
        <Switch checked={record.is_active} onChange={() => handleToggle(record)} checkedChildren={t('common.enabled')} unCheckedChildren={t('common.disabled')} />
      ),
    },
  ];

  return (
    <Card size="small">
      <Space style={{ marginBottom: 16 }}>
        <Button type="primary" icon={<PlusOutlined />} onClick={() => setCreateOpen(true)}>
          {t('pages.settings.users.create')}
        </Button>
        <Button icon={<ReloadOutlined />} onClick={load} loading={loading}>
          {t('common.refresh')}
        </Button>
      </Space>
      <FilterSearchBar
        searchPlaceholder={t('pages.settings.users.searchPlaceholder')}
        searchValue={searchInput}
        onSearchChange={setSearchInput}
      />
      <Table {...table.tableProps} columns={columns} scroll={{ x: 'max-content' }} />

      <Modal title={t('pages.settings.users.create')} open={createOpen} onOk={handleCreate} onCancel={() => setCreateOpen(false)} destroyOnHidden>
        <Form form={form} layout="vertical">
          <Form.Item name="username" label={t('pages.settings.users.username')} rules={[{ required: true }]}>
            <Input autoComplete="off" />
          </Form.Item>
          <Form.Item name="full_name" label={t('pages.settings.users.fullName')}>
            <Input />
          </Form.Item>
          <Form.Item name="email" label={t('pages.settings.users.email')} rules={[{ type: 'email', required: true }]}>
            <Input />
          </Form.Item>
          <Form.Item name="password" label={t('auth.password')} rules={[{ required: true, min: 6 }]}>
            <Input.Password autoComplete="new-password" />
          </Form.Item>
          <Form.Item name="roles" label={t('pages.settings.users.roles')} rules={[{ required: true }]}>
            <Select mode="multiple" options={ROLE_OPTIONS} />
          </Form.Item>
          <Form.Item name="tenant_id" label={t('pages.settings.users.tenant')} rules={[{ required: true }]}>
            <Select
              showSearch
              optionFilterProp="label"
              options={tenants.map((t2) => ({ value: t2.id, label: `${t2.name} (${t2.id})` }))}
            />
          </Form.Item>
        </Form>
      </Modal>
    </Card>
  );
};
