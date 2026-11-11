// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

import React, { useState, useEffect } from 'react';
import { Card, Table, Button, Tag, Space, Modal, Form, Input, Select, InputNumber, Alert, message } from 'antd';
import { useTranslation } from 'react-i18next';
import { useAuth } from '../hooks/useAuth';
import { mlBackendApi, modelApi } from '../api/endpoints';
import { HelpButton } from '@components/HelpDrawer';
import PageHeader from '../components/PageHeader';
import { useServerTable } from '@hooks/useServerTable';
import { useDebounce } from '@hooks/useDebounce';
import FilterSearchBar from '@components/FilterSearchBar';
import { statusLabelKey } from '@components/StatusBadge';
import { MLBackendStatus } from '@nftypes/api';
import type { MLBackend, Model } from '@nftypes/api';
import { formatDateTime } from '../utils/format';

const MLBackend: React.FC = () => {
  const { t } = useTranslation();
  const { tenantId } = useAuth();
  // Server-side status filter plus debounced name search.
  const [statusFilter, setStatusFilter] = useState('');
  const [searchInput, setSearchInput] = useState('');
  const search = useDebounce(searchInput.trim(), 300);
  const statusOptions = Object.values(MLBackendStatus).map((v) => ({
    value: v, label: t(statusLabelKey(v)),
  }));
  // Server-side paginated backend list.
  const backendsTable = useServerTable<MLBackend, { status?: string; search?: string }>({
    list: (params) => mlBackendApi.list(params),
    filters: {
      status: statusFilter || undefined,
      search: search || undefined,
    },
    defaultPageSize: 10,
  });
  const { reload: loadDeployments } = backendsTable;
  const [models, setModels] = useState<Model[]>([]);
  const [modalVisible, setModalVisible] = useState(false);
  const [switchVisible, setSwitchVisible] = useState(false);
  const [switchingBackend, setSwitchingBackend] = useState<MLBackend | null>(null);
  const [switchForm] = Form.useForm();
  const [deployForm] = Form.useForm();
  // Watch model_name selection to cascade version dropdown (useWatch triggers re-render)
  const selectedModelName = Form.useWatch('model_name', deployForm);
  const selectedModel = models.find((m) => m.name === selectedModelName);
  // Switch modal also cascades version dropdown
  const switchSelectedModelName = Form.useWatch('model_name', switchForm);
  const switchSelectedModel = models.find((m) => m.name === switchSelectedModelName);
  
  // Custom backend creation
  const [customModalVisible, setCustomModalVisible] = useState(false);
  const [customFromRegistryVisible, setCustomFromRegistryVisible] = useState(false);
  const [customForm] = Form.useForm();
  const [customRegistryForm] = Form.useForm();
  // Form.useWatch is required: getFieldValue() during render does NOT
  // subscribe to changes, so the Version Select stayed on its initial empty
  // options after a model was picked.
  const registrySelectedModelName = Form.useWatch('model_name', customRegistryForm);
  const registrySelectedModel = models.find((m) => m.name === registrySelectedModelName);

  const fallbackCopy = (text: string) => {
    try {
      const ta = document.createElement('textarea');
      ta.value = text;
      ta.style.position = 'fixed';
      ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.select();
      document.execCommand('copy');
      document.body.removeChild(ta);
      message.success(t('pages.mlBackend.copied'));
    } catch (e) {
      console.error(e);
      message.error(t('pages.mlBackend.copyFailed'));
    }
  };

  useEffect(() => {
    loadModels();
  }, [tenantId]);

  const loadModels = async () => {
    try {
      const res = await modelApi.list({ page_size: 100 });
      setModels(res.data.items);
    } catch (e) {
      console.error(e);
    }
  };

  const handleDeploy = async (values: any) => {
    try {
      // annotations_schema is a JSON string, parse to object with fallback default
      let schema: Record<string, unknown> | undefined;
      if (values.annotations_schema) {
        try {
          schema = JSON.parse(values.annotations_schema);
        } catch {
          schema = undefined;
        }
      }
      await mlBackendApi.create({
        model_name: values.model_name,
        model_version: values.version,
        name: values.model_name + '-' + (values.version || 'v1'),
        model_type: values.model_type || 'cnn',
        confidence_threshold: values.confidence_threshold ?? 0.6,
        annotations_schema: schema,
        capabilities: [],
        url: `http://ml-backend-${values.model_type || 'cnn'}:9090`,
      });
      setModalVisible(false);
      deployForm.resetFields();
      loadDeployments();
    } catch (e) {
      console.error(e);
    }
  };

  const handleCreateCustom = async (values: any) => {
    try {
      await mlBackendApi.createCustom({
        model_name: values.model_name,
        model_version: values.model_version,
        s3_path: values.s3_path,
        backend_type: values.backend_type,
        framework: values.framework,
      });
      setCustomModalVisible(false);
      customForm.resetFields();
      message.success(t('pages.mlBackend.customCreated'));
      loadDeployments();
    } catch (e) {
      console.error(e);
      message.error(t('pages.mlBackend.customCreateFailed'));
    }
  };

  const handleCreateCustomFromRegistry = async (values: any) => {
    try {
      await mlBackendApi.createCustomFromRegistry({
        model_name: values.model_name,
        model_version: values.model_version,
      });
      setCustomFromRegistryVisible(false);
      customRegistryForm.resetFields();
      message.success(t('pages.mlBackend.customCreated'));
      loadDeployments();
    } catch (e) {
      console.error(e);
      message.error(t('pages.mlBackend.customCreateFailed'));
    }
  };

  const openSwitch = (backend: MLBackend) => {
    setSwitchingBackend(backend);
    switchForm.setFieldsValue({
      model_name: backend.model_name || undefined,
      model_version: backend.model_version || undefined,
    });
    setSwitchVisible(true);
  };

  const handleSwitchModel = async (values: any) => {
    if (!switchingBackend) return;
    try {
      await mlBackendApi.update(switchingBackend.id, {
        model_name: values.model_name,
        model_version: values.model_version,
      });
      setSwitchVisible(false);
      message.success(t('pages.mlBackend.switched'));
      loadDeployments();
    } catch (e) {
      console.error(e);
      message.error(t('pages.mlBackend.switchFailed'));
    }
  };

  // Map backend status (worker writes connected/error/unreachable/...) to a
  // tag colour and a localized label. The old render only knew healthy/
  // unhealthy, so a live "connected" backend wrongly showed as orange.
  const statusMeta = (s: string): { color: string; key: string } => {
    switch ((s || '').toLowerCase()) {
      case 'connected':
      case 'healthy':
        return { color: 'green', key: 'statusConnected' };
      case 'error':
      case 'unhealthy':
        return { color: 'red', key: 'statusError' };
      case 'unreachable':
        return { color: 'red', key: 'statusUnreachable' };
      case 'creating':
      case 'created':
        return { color: 'orange', key: 'statusCreating' };
      default:
        return { color: 'default', key: 'statusUnknown' };
    }
  };

  // Persist an inline confidence-threshold edit (allowed even on the shared
  // built-in backend; only its URL/model are locked).
  const saveConfidence = async (record: MLBackend, value: number | null) => {
    if (value == null) return;
    const v = Math.min(1, Math.max(0.01, Number(value)));
    if (Math.abs(v - Number(record.confidence_threshold ?? 0.25)) < 1e-9) return;
    try {
      await mlBackendApi.update(record.id, { confidence_threshold: v });
      message.success(t('pages.mlBackend.confidenceUpdated'));
      loadDeployments();
    } catch (e) {
      console.error(e);
      message.error(t('pages.mlBackend.confidenceFailed'));
    }
  };

  const columns = [
    { title: t('pages.mlBackend.name'), dataIndex: 'name', key: 'name', ellipsis: true },
    {
      // The shared built-in backend has no pinned model (weights are chosen
      // per project / on-demand); show that plainly instead of "YOLO / -".
      title: t('pages.mlBackend.model'),
      key: 'model',
      render: (_: unknown, record: MLBackend) =>
        record.model_version
          ? `${record.model_name}@${record.model_version}`
          : <Tag color="blue">{t('pages.mlBackend.builtInEngine')}</Tag>,
    },
    {
      title: t('pages.mlBackend.status'),
      dataIndex: 'status',
      key: 'status',
      render: (v: string) => {
        const m = statusMeta(v || 'unknown');
        return <Tag color={m.color}>{t(`pages.mlBackend.${m.key}`)}</Tag>;
      },
    },
    {
      title: t('pages.mlBackend.lastCheck'),
      dataIndex: 'last_health_check',
      key: 'last_health_check',
      render: (v: string) => (v ? formatDateTime(v) : '-'),
    },
    {
      title: t('pages.mlBackend.confidenceThreshold'),
      dataIndex: 'confidence_threshold',
      key: 'confidence_threshold',
      render: (v: number, record: MLBackend) => (
        <InputNumber
          size="small"
          min={0.01}
          max={1}
          step={0.05}
          defaultValue={v ?? 0.25}
          style={{ width: 84 }}
          onPressEnter={(e) => saveConfidence(record, Number((e.target as HTMLInputElement).value))}
          onBlur={(e) => saveConfidence(record, Number(e.target.value))}
        />
      ),
    },
    { title: t('pages.mlBackend.createdAt'), dataIndex: 'created_at', key: 'created_at', render: (v: string) => formatDateTime(v) },
    {
      title: t('pages.mlBackend.actions'),
      key: 'actions',
      render: (record: any) => {
        // Check if this is the shared built-in backend (fixed in-cluster URL,
        // either the short svc name or the FQDN). It cannot be switched/deleted.
        const isSystemPreset = /ml-backend\.nereidforge(\.svc\.cluster\.local)?:9090/.test(record.url || '') &&
                               !record.url?.includes('custom');
        
        return (
          <Space>
            <Button type="link" onClick={() => {
              const text = record.url || '';
              if (navigator.clipboard && navigator.clipboard.writeText) {
                navigator.clipboard.writeText(text).then(() => {
                  message.success(t('pages.mlBackend.copied'));
                }).catch(() => {
                  fallbackCopy(text);
                });
              } else {
                fallbackCopy(text);
              }
            }}>{t('pages.mlBackend.copyUrl')}</Button>
            <Button type="link" onClick={async () => {
              try {
                const res = await mlBackendApi.healthCheck(record.id);
                message.success(t('pages.mlBackend.healthOk', { status: res.data.status, latency: res.data.latency_ms }));
                loadDeployments();
              } catch (e) { console.error(e); message.error(t('pages.mlBackend.healthFail')); }
            }}>{t('pages.mlBackend.check')}</Button>
            {!isSystemPreset && (
              <>
                <Button type="link" onClick={() => openSwitch(record)}>{t('pages.mlBackend.switchModel')}</Button>
                <Button type="link" danger onClick={async () => {
                  if (window.confirm(t('pages.mlBackend.deleteConfirm') || `Delete ${record.name}?`)) {
                    try {
                      await mlBackendApi.delete(record.id);
                      message.success(t('pages.mlBackend.deleted'));
                      loadDeployments();
                    } catch (e) {
                      console.error(e);
                      message.error(t('pages.mlBackend.deleteFailed'));
                    }
                  }
                }}>{t('common.delete')}</Button>
              </>
            )}
          </Space>
        );
      },
    },
  ];

  return (
    <Card>
      <PageHeader
        title={t('pages.mlBackend.title')}
        subtitle={t('pages.mlBackend.subtitle')}
        actions={[
          <HelpButton module="mlbackend" />,
          <Button onClick={() => { customRegistryForm.resetFields(); setCustomFromRegistryVisible(true); }}>{t('pages.mlBackend.createFromRegistry')}</Button>,
          <Button onClick={() => { customForm.resetFields(); setCustomModalVisible(true); }}>{t('pages.mlBackend.createCustom')}</Button>,
          <Button type="primary" onClick={() => { deployForm.resetFields(); setModalVisible(true); }}>{t('pages.mlBackend.deployService')}</Button>
        ]}
      />
      <Alert message={t('pages.mlBackend.deployHint')} type="info" style={{ marginBottom: 16 }} />
      <FilterSearchBar
        filterPlaceholder={t('pages.mlBackend.filterStatus')}
        filterValue={statusFilter}
        onFilterChange={setStatusFilter}
        filterOptions={statusOptions}
        searchPlaceholder={t('pages.mlBackend.searchPlaceholder')}
        searchValue={searchInput}
        onSearchChange={setSearchInput}
      />
      <Table
        {...backendsTable.tableProps}
        columns={columns}
      scroll={{ x: 'max-content' }}
      />
      <Modal title={t('pages.mlBackend.deployService')} open={modalVisible} onCancel={() => setModalVisible(false)} footer={null}>
        <Form form={deployForm} onFinish={handleDeploy} layout="vertical">
          <Form.Item name="model_name" label={t('pages.mlBackend.modelName')} rules={[{ required: true }]}>
            <Select
              showSearch
              optionFilterProp="label"
              placeholder={t('pages.mlBackend.selectModel')}
              options={models.map((m) => ({ value: m.name, label: `${m.name} (${m.task_type || '-'})` }))}
              onChange={() => deployForm.setFieldValue('version', undefined)}
            />
          </Form.Item>
          <Form.Item name="version" label={t('pages.mlBackend.version')} rules={[{ required: true }]}>
            <Select
              placeholder={t('pages.mlBackend.versionPh')}
              options={(selectedModel?.versions || []).map((v) => ({ value: v.version, label: v.version }))}
            />
          </Form.Item>
          <Form.Item name="model_type" label={t('pages.mlBackend.modelType')} rules={[{ required: true }]} initialValue="cnn">
            <Select options={['cnn', 'llm', 'vlm'].map(t2 => ({ label: t2.toUpperCase(), value: t2 }))} placeholder={t('pages.mlBackend.selectType')} />
          </Form.Item>
          <Form.Item name="confidence_threshold" label={t('pages.mlBackend.confidenceThreshold')} rules={[{ required: true }]} initialValue={0.6}>
            <InputNumber min={0} max={1} step={0.05} style={{ width: '100%' }} />
          </Form.Item>
          <Form.Item name="annotations_schema" label={t('pages.mlBackend.annotationsSchema')}>
            <Input.TextArea rows={4} placeholder='{"labels": ["person", "car"]}' />
          </Form.Item>
          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit">{t('pages.mlBackend.deploy')}</Button>
              <Button onClick={() => setModalVisible(false)}>{t('pages.mlBackend.cancel')}</Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>
      <Modal title={t('pages.mlBackend.switchModel')} open={switchVisible} onCancel={() => setSwitchVisible(false)} footer={null}>
        <p style={{ color: '#888', fontSize: 13 }}>{t('pages.mlBackend.switchHint')}</p>
        <Form form={switchForm} onFinish={handleSwitchModel} layout="vertical">
          <Form.Item name="model_name" label={t('pages.mlBackend.model')} rules={[{ required: true }]}>
            <Select
              showSearch
              optionFilterProp="label"
              placeholder={t('pages.mlBackend.selectModel')}
              options={models.map(m => ({ label: `${m.name} (${m.task_type || '-'})`, value: m.name }))}
              onChange={() => switchForm.setFieldValue('model_version', undefined)}
            />
          </Form.Item>
          <Form.Item name="model_version" label={t('pages.mlBackend.version')} rules={[{ required: true }]}>
            <Select
              placeholder={t('pages.mlBackend.versionPh')}
              options={(switchSelectedModel?.versions || []).map((v) => ({ value: v.version, label: v.version }))}
            />
          </Form.Item>
          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit">{t('pages.mlBackend.switch')}</Button>
              <Button onClick={() => setSwitchVisible(false)}>{t('pages.mlBackend.cancel')}</Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>
      <Modal title={t('pages.mlBackend.createCustom')} open={customModalVisible} onCancel={() => setCustomModalVisible(false)} footer={null}>
        <Form form={customForm} onFinish={handleCreateCustom} layout="vertical">
          <Form.Item name="model_name" label={t('pages.mlBackend.modelName')} rules={[{ required: true }]}>
            <Input placeholder={t('pages.mlBackend.modelNamePh')} />
          </Form.Item>
          <Form.Item name="model_version" label={t('pages.mlBackend.version')} rules={[{ required: true }]}>
            <Input placeholder={t('pages.mlBackend.versionPh')} />
          </Form.Item>
          <Form.Item name="s3_path" label={t('pages.mlBackend.s3Path')} rules={[{ required: true }]}>
            <Input placeholder="s3://bucket/path/to/model.pt" />
          </Form.Item>
          <Form.Item name="backend_type" label={t('pages.mlBackend.backendType')} rules={[{ required: true }]}>
            <Select options={['det', 'seg', 'ocr', 'pose', 'obb'].map(t2 => ({ label: t2.toUpperCase(), value: t2 }))} placeholder={t('pages.mlBackend.selectType')} />
          </Form.Item>
          <Form.Item name="framework" label={t('pages.mlBackend.framework')} rules={[{ required: true }]}>
            <Select options={['ultralytics', 'paddleocr', 'pytorch', 'tensorflow', 'onnxruntime'].map(t2 => ({ label: t2, value: t2 }))} placeholder={t('pages.mlBackend.selectFramework')} />
          </Form.Item>
          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit">{t('pages.mlBackend.create')}</Button>
              <Button onClick={() => setCustomModalVisible(false)}>{t('pages.mlBackend.cancel')}</Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>
      <Modal title={t('pages.mlBackend.createFromRegistry')} open={customFromRegistryVisible} onCancel={() => setCustomFromRegistryVisible(false)} footer={null}>
        <Form form={customRegistryForm} onFinish={handleCreateCustomFromRegistry} layout="vertical">
          <Form.Item name="model_name" label={t('pages.mlBackend.modelName')} rules={[{ required: true }]}>
            <Select
              showSearch
              optionFilterProp="label"
              placeholder={t('pages.mlBackend.selectModel')}
              options={models.map((m) => ({ value: m.name, label: `${m.name} (${m.task_type || '-'})` }))}
              onChange={() => customRegistryForm.setFieldValue('model_version', undefined)}
            />
          </Form.Item>
          <Form.Item name="model_version" label={t('pages.mlBackend.version')} rules={[{ required: true }]}>
            <Select
              placeholder={t('pages.mlBackend.versionPh')}
              options={(registrySelectedModel?.versions || []).map((v) => ({ value: v.version, label: v.version }))}
            />
          </Form.Item>
          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit">{t('pages.mlBackend.create')}</Button>
              <Button onClick={() => setCustomFromRegistryVisible(false)}>{t('pages.mlBackend.cancel')}</Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>
    </Card>
  );
};

export default MLBackend;
