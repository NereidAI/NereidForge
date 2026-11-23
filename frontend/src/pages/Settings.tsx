// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

import React, { useEffect, useState } from 'react';
import { Card, Tabs, Form, Input, InputNumber, Button, Alert, Radio, Select, Space, Switch, message } from 'antd';
import { useTranslation } from 'react-i18next';
import { useAuth } from '../hooks/useAuth';
import { settingsApi, tenantApi } from '../api/endpoints';
import PageHeader from '../components/PageHeader';
import { TenantsManagement, UsersManagement } from './admin/TenantUsers';
import { AuditLogs } from './admin/AuditLogs';
import SystemHealth from './admin/SystemHealth';
import type { Tenant } from '@nftypes/api';

const { TabPane } = Tabs;

const Settings: React.FC = () => {
  const { t } = useTranslation();
  const { roles } = useAuth();
  const isAdmin = roles.includes('admin');

  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [storageMode, setStorageMode] = useState<'cluster' | 'custom'>('cluster');
  const [tenants, setTenants] = useState<Tenant[]>([]);

  const [generalForm] = Form.useForm();
  const [storageForm] = Form.useForm();
  const [registryForm] = Form.useForm();
  const [trainingForm] = Form.useForm();

  useEffect(() => {
    if (!isAdmin) return;
    (async () => {
      try {
        const res = await settingsApi.getSystemConfig();
        const cfg = res.data;
        generalForm.setFieldsValue({
          platformName: cfg.name,
          maxGpusPerJob: cfg.max_gpus_per_job,
        });
        storageForm.setFieldsValue({
          backend: cfg.storage_backend || 'seaweedfs',
          bucket: cfg.storage_bucket,
        });
        const endpoint = typeof cfg.storage_endpoint === 'string' ? cfg.storage_endpoint : '';
        setStorageMode(endpoint ? 'custom' : 'cluster');
        storageForm.setFieldsValue({ endpoint });
        registryForm.setFieldsValue({
          host: cfg.container_registry_host,
          project: cfg.container_registry_project,
          username: cfg.container_registry_username,
          insecure: cfg.container_registry_insecure,
        });
        trainingForm.setFieldsValue({
          trainerCpuRequest: cfg.trainer_cpu_request,
          trainerCpuLimit: cfg.trainer_cpu_limit,
          trainerMemoryRequest: cfg.trainer_memory_request,
          trainerMemoryLimit: cfg.trainer_memory_limit,
          trainerGpuRequest: cfg.trainer_gpu_request,
        });
        const tRes = await tenantApi.list({ page_size: 200 });
        setTenants(tRes.data.items ?? []);
      } catch {
        message.error(t('pages.settings.loadFailed'));
      } finally {
        setLoading(false);
      }
    })();
  }, [isAdmin, generalForm, storageForm, registryForm, trainingForm, t]);

  const saveSection = async (payload: Record<string, unknown>) => {
    setSaving(true);
    try {
      await settingsApi.updateSystemConfig(payload);
      message.success(t('pages.settings.saved'));
    } catch {
      message.error(t('pages.settings.saveFailed'));
    } finally {
      setSaving(false);
    }
  };

  const handleGeneral = (values: { platformName?: string; maxGpusPerJob?: number }) => {
    saveSection({
      name: values.platformName,
      max_gpus_per_job: values.maxGpusPerJob,
    });
  };

  const handleStorage = (values: { backend?: string; bucket?: string; endpoint?: string }) => {
    const payload: Record<string, unknown> = {
      storage_backend: values.backend,
      storage_bucket: values.bucket,
      storage_endpoint: storageMode === 'custom' ? values.endpoint : '',
    };
    saveSection(payload);
  };

  const handleRegistry = (values: {
    host?: string; project?: string; username?: string; password?: string; insecure?: boolean;
  }) => {
    // password left empty = keep the stored one (never echoed back)
    const patch: Record<string, unknown> = {
      container_registry_host: values.host,
      container_registry_project: values.project,
      container_registry_username: values.username,
      container_registry_insecure: values.insecure,
    };
    if (values.password) patch.container_registry_password = values.password;
    saveSection(patch);
  };

  // Test uses the CURRENT form values (unsaved edits included); an empty
  // password field makes the backend fall back to the stored one.
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState<{ ok: boolean; message?: string; project?: string; repo_count?: number } | null>(null);
  const handleRegistryTest = async () => {
    const values = await registryForm.validateFields();
    setTesting(true);
    setTestResult(null);
    try {
      const res = await settingsApi.testRegistry({
        host: values.host,
        project: values.project,
        username: values.username,
        password: values.password,
        insecure: values.insecure,
      });
      setTestResult(res.data);
    } catch (e: unknown) {
      const msg = (e as { response?: { data?: { detail?: string; message?: string } } })?.response?.data;
      setTestResult({ ok: false, message: msg?.detail || msg?.message || 'request failed' });
    } finally {
      setTesting(false);
    }
  };

  const handleTraining = (values: {
    trainerCpuRequest?: string;
    trainerCpuLimit?: string;
    trainerMemoryRequest?: string;
    trainerMemoryLimit?: string;
    trainerGpuRequest?: string;
  }) => {
    saveSection({
      trainer_cpu_request: values.trainerCpuRequest,
      trainer_cpu_limit: values.trainerCpuLimit,
      trainer_memory_request: values.trainerMemoryRequest,
      trainer_memory_limit: values.trainerMemoryLimit,
      trainer_gpu_request: values.trainerGpuRequest,
    });
  };

  if (!isAdmin) {
    return (
      <Card>
        <PageHeader title={t('pages.settings.title')} subtitle={t('pages.settings.noPermission')} />
        <Alert message={t('pages.settings.noPermission')} type="warning" />
      </Card>
    );
  }

  return (
    <Card>
      <PageHeader title={t('pages.settings.title')} subtitle={t('pages.settings.subtitle')} />
      {loading ? (
        <div style={{ textAlign: 'center', padding: 64 }}>...</div>
      ) : (
        <Tabs defaultActiveKey="health">
          <TabPane tab={t('pages.settings.tabHealth')} key="health">
            <SystemHealth />
          </TabPane>

          <TabPane tab={t('pages.settings.tabGeneral')} key="general">
            <Form form={generalForm} layout="vertical" onFinish={handleGeneral} style={{ marginTop: 16 }}>
              <Form.Item name="platformName" label={t('pages.settings.general.platformName')} rules={[{ required: true }]}>
                <Input placeholder="NereidForge" />
              </Form.Item>
              <Form.Item name="maxGpusPerJob" label={t('pages.settings.general.maxGpusPerJob')}>
                <InputNumber min={1} max={64} style={{ width: '100%' }} />
              </Form.Item>
              <Form.Item>
                <Button type="primary" htmlType="submit" loading={saving}>{t('pages.settings.general.save')}</Button>
              </Form.Item>
            </Form>
          </TabPane>

          <TabPane tab={t('pages.settings.tabStorage')} key="storage">
            <Form form={storageForm} layout="vertical" onFinish={handleStorage} style={{ marginTop: 16 }}>
              <Form.Item name="backend" label={t('pages.settings.storage.backend')}>
                <Select
                  options={[
                    { value: 'seaweedfs', label: 'SeaweedFS (S3)' },
                    { value: 'minio', label: 'MinIO (S3)' },
                  ]}
                />
              </Form.Item>
              <Form.Item name="bucket" label={t('pages.settings.storage.bucket')}>
                <Input placeholder="nereidforge" />
              </Form.Item>
              <Form.Item label={t('pages.settings.storage.endpointMode')}>
                <Radio.Group value={storageMode} onChange={(e) => setStorageMode(e.target.value)}>
                  <Radio value="cluster">{t('pages.settings.storage.endpointCluster')}</Radio>
                  <Radio value="custom">{t('pages.settings.storage.endpointCustom')}</Radio>
                </Radio.Group>
              </Form.Item>
              {storageMode === 'custom' ? (
                <Form.Item
                  name="endpoint"
                  label={t('pages.settings.storage.endpoint')}
                  rules={[{ required: true, message: t('pages.settings.storage.endpointRequired') }]}
                >
                  <Input placeholder={t('pages.settings.storage.endpointPlaceholder')} />
                </Form.Item>
              ) : (
                <Alert type="info" showIcon message={t('pages.settings.storage.endpointClusterDesc')} />
              )}
              <Form.Item>
                <Button type="primary" htmlType="submit" loading={saving}>{t('pages.settings.storage.save')}</Button>
              </Form.Item>
            </Form>
          </TabPane>

          {/* KFP connection is deployment config (configs/config.{dev,prod}.yaml
              kfp.host — cluster svc in prod, port-forward in dev), same policy
              as the Model Registry: not user-editable at runtime. Pipeline
              submission health surfaces on the Training page / API logs. */}

          {/* Model Registry (Kubeflow hub) intentionally has NO settings tab:
              its address is deployment config (configs/config.{dev,prod}.yaml
              hub.host — cluster-internal svc in prod, port-forward in dev),
              not a user-facing runtime setting. Health shows on the Models
              page and in API startup logs. */}

          <TabPane tab={t('pages.settings.tabRegistry')} key="registry">
            <Form form={registryForm} layout="vertical" onFinish={handleRegistry} style={{ marginTop: 16 }}>
              <Alert type="info" showIcon message={t('pages.settings.registry.hint')} style={{ marginBottom: 16 }} />
              <Form.Item name="host" label={t('pages.settings.registry.host')} rules={[{ required: true }]}>
                <Input placeholder="ghcr.io" />
              </Form.Item>
              <Form.Item name="project" label={t('pages.settings.registry.project')} rules={[{ required: true }]}>
                <Input placeholder="nereidai" />
              </Form.Item>
              <Form.Item name="username" label={t('pages.settings.registry.username')}>
                <Input placeholder="admin" autoComplete="off" />
              </Form.Item>
              <Form.Item
                name="password"
                label={t('pages.settings.registry.password')}
                extra={t('pages.settings.registry.passwordHint')}
              >
                <Input.Password placeholder="••••••••" autoComplete="new-password" />
              </Form.Item>
              <Form.Item name="insecure" label={t('pages.settings.registry.insecure')} valuePropName="checked">
                <Switch />
              </Form.Item>
              <Form.Item>
                <Space>
                  <Button type="primary" htmlType="submit" loading={saving}>{t('pages.settings.registry.save')}</Button>
                  <Button loading={testing} onClick={handleRegistryTest}>
                    {t('pages.settings.registry.test')}
                  </Button>
                </Space>
                {testResult && (
                  <Alert
                    type={testResult.ok ? 'success' : 'error'}
                    showIcon
                    style={{ marginTop: 12 }}
                    message={
                      testResult.ok
                        ? `${t('pages.settings.registry.testOk')} — ${testResult.project}${typeof testResult.repo_count === 'number' ? ` · ${testResult.repo_count} ${t('pages.settings.registry.reposUnit')}` : ''}`
                        : `${t('pages.settings.registry.testFail')}${testResult.message ? `：${testResult.message}` : ''}`
                    }
                  />
                )}
              </Form.Item>
            </Form>
          </TabPane>

          <TabPane tab={t('pages.settings.tabTraining')} key="training">
            <Form form={trainingForm} layout="vertical" onFinish={handleTraining} style={{ marginTop: 16 }}>
              <Alert type="info" showIcon message={t('pages.settings.training.hint')} style={{ marginBottom: 16 }} />
              <Form.Item name="trainerCpuRequest" label={t('pages.settings.training.trainerCpuRequest')}>
                <Input placeholder="4" />
              </Form.Item>
              <Form.Item name="trainerCpuLimit" label={t('pages.settings.training.trainerCpuLimit')}>
                <Input placeholder="8" />
              </Form.Item>
              <Form.Item name="trainerMemoryRequest" label={t('pages.settings.training.trainerMemoryRequest')}>
                <Input placeholder="8Gi" />
              </Form.Item>
              <Form.Item name="trainerMemoryLimit" label={t('pages.settings.training.trainerMemoryLimit')}>
                <Input placeholder="16Gi" />
              </Form.Item>
              <Form.Item name="trainerGpuRequest" label={t('pages.settings.training.trainerGpuRequest')}>
                <InputNumber min={0} max={8} style={{ width: '100%' }} />
              </Form.Item>
              <Form.Item>
                <Button type="primary" htmlType="submit" loading={saving}>{t('pages.settings.training.save')}</Button>
              </Form.Item>
            </Form>
          </TabPane>

          <TabPane tab={t('pages.settings.tabTenants')} key="tenants">
            <Tabs defaultActiveKey="tenants" size="small">
              <TabPane tab={t('pages.settings.tenants.title')} key="tenants">
                <TenantsManagement />
              </TabPane>
              <TabPane tab={t('pages.settings.users.title')} key="users">
                <UsersManagement tenants={tenants} />
              </TabPane>
            </Tabs>
          </TabPane>

          <TabPane tab={t('pages.settings.tabAudit')} key="audit">
            <AuditLogs />
          </TabPane>

          <TabPane tab={t('pages.settings.tabPlugins')} key="plugins">
            <Alert message={t('pages.settings.plugins.hint')} type="info" />
          </TabPane>
        </Tabs>
      )}
    </Card>
  );
};

export default Settings;
