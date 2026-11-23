// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

import React, { useEffect, useState } from 'react';
import { Card, Table, Button, Space, Modal, Form, Input, Select, Divider, Tooltip, message } from 'antd';
import { EditOutlined, DeleteOutlined, PlayCircleOutlined, StopOutlined, ThunderboltOutlined } from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import { useNavigate } from 'react-router-dom';
import { annotationApi, datasetApi, settingsApi } from '@api/endpoints';
import { PageHeader } from '@components/PageHeader';
import { HelpButton } from '@components/HelpDrawer';
import { StatusBadge } from '@components/StatusBadge';
import { useServerTable } from '@hooks/useServerTable';
import { useDebounce } from '@hooks/useDebounce';
import FilterSearchBar from '@components/FilterSearchBar';
import type { AnnotationTask, AnnotationTaskStatus, Dataset, DatasetVersion, User } from '@nftypes/api';
import { formatDateTime } from '../utils/format';

const CV_TASK_TYPES = [
  { value: 'detection', key: 'taskDetection' },
  { value: 'segmentation', key: 'taskSegmentation' },
  { value: 'classification', key: 'taskClassification' },
  { value: 'keypoint', key: 'taskKeypoint' },
  { value: 'rotation', key: 'taskRotation' },
  { value: 'mask', key: 'taskMask' },
  { value: 'video-tracking', key: 'taskVideoTracking' },
  { value: 'ocr', key: 'taskOCR' },
] as const;

const NLP_TASK_TYPES = [
  { value: 'text-classification', key: 'taskTextClassification' },
  { value: 'ner', key: 'taskNER' },
  { value: 'text-generation', key: 'taskTextGeneration' },
  { value: 'summarization', key: 'taskSummarization' },
  { value: 'chat-rlhf', key: 'taskChatRLHF' },
] as const;

const TASK_TYPE_KEY: Record<string, string> = {
  detection: 'taskDetection',
  segmentation: 'taskSegmentation',
  classification: 'taskClassification',
  keypoint: 'taskKeypoint',
  rotation: 'taskRotation',
  mask: 'taskMask',
  'video-tracking': 'taskVideoTracking',
  'text-classification': 'taskTextClassification',
  ner: 'taskNER',
  'text-generation': 'taskTextGeneration',
  'chat-rlhf': 'taskChatRLHF',
  summarization: 'taskSummarization',
};

const ANNOTATION_TOOLS = ['label-studio', 'cvat', 'doccano'] as const;

// Annotation label types (maps to backend schema.type -> LS label config)
const LABEL_TYPES = [
  { value: 'rectanglelabels', label: 'Rectangle (bounding box)' },
  { value: 'polygonlabels', label: 'Polygon (segmentation)' },
  { value: 'keypointlabels', label: 'KeyPoint' },
  { value: 'choices', label: 'Choices (classification)' },
  { value: 'obb', label: 'OBB (rotated box)' },
  { value: 'brushlabels', label: 'Brush (mask)' },
];

export const Annotations: React.FC = () => {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const [assigneeFilter, setAssigneeFilter] = useState<string | undefined>(undefined);
  // Debounced task-name search (server-side, combined with the assignee filter).
  const [searchInput, setSearchInput] = useState('');
  const search = useDebounce(searchInput.trim(), 300);
  // Server-side paginated project list; the filters reset to page 1.
  const tasksTable = useServerTable<AnnotationTask, { assignee_id?: string; search?: string }>({
    list: (params) => annotationApi.list(params),
    filters: { assignee_id: assigneeFilter || undefined, search: search || undefined },
    defaultPageSize: 10,
  });
  const { reload: fetchTasks } = tasksTable;
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [versions, setVersions] = useState<DatasetVersion[]>([]);
  const [users, setUsers] = useState<User[]>([]);
  const [modalOpen, setModalOpen] = useState(false);
  const [editingTask, setEditingTask] = useState<AnnotationTask | null>(null);
  const [form] = Form.useForm();
  const watchDatasetId = Form.useWatch('dataset_id', form);
  const watchTaskType = Form.useWatch('task_type', form);
  const watchVersion = Form.useWatch('dataset_version', form);
  const watchLabels = Form.useWatch('labels', form);

  // Auto-set annotation_tool based on task_type (ADR-010: CVAT only for video-tracking)
  useEffect(() => {
    if (watchTaskType === 'video-tracking') {
      form.setFieldValue('annotation_tool', 'cvat');
    } else {
      form.setFieldValue('annotation_tool', 'label-studio');
    }
  }, [watchTaskType, form]);

  // Load version list after dataset selection for the dataset_version dropdown
  useEffect(() => {
    if (!watchDatasetId) {
      setVersions([]);
      return;
    }
    datasetApi.getVersions(watchDatasetId, { page_size: 100 }).then((res) => {
      const all = (res.data.items || []).filter((v) => !v.is_draft);
      // video-tracking requires a version with videos (ADR-010 D1):
      // narrow the picker to video-bearing versions and warn when none.
      const items = watchTaskType === 'video-tracking'
        ? all.filter((v) => v.has_videos)
        : all;
      setVersions(items);
      if (watchTaskType === 'video-tracking' && all.length > 0 && items.length === 0) {
        message.warning(t('pages.annotations.noVideoVersions'));
      }
      // Select first version by default when creating; keep pre-filled version when editing
      if (!editingTask && !form.getFieldValue('dataset_version')) {
        form.setFieldValue('dataset_version', items[0]?.version);
      }
    }).catch(() => {
      setVersions([]);
    });
  }, [watchDatasetId, watchTaskType, editingTask, form, t]);

  // Prefill the label control type from the selected version's canonical
  // format (create mode, while the user has not typed labels). Class names
  // are intentionally left blank: the backend derives them from the version
  // (data.yaml / class_names.json) during project creation.
  useEffect(() => {
    if (editingTask || watchLabels || !watchDatasetId || !watchVersion) {
      return;
    }
    const ver = versions.find((v) => v.version === watchVersion);
    if (!ver) {
      return;
    }
    const nextType = ver.format === 'imagenet' ? 'choices'
      : ver.format === 'llm-sft' ? 'choices'
      : 'rectanglelabels';
    form.setFieldValue('label_type', nextType);
  }, [editingTask, watchLabels, watchDatasetId, watchVersion, versions, form]);

  const fetchDatasets = async () => {
    try {
      const response = await datasetApi.list({ page_size: 100 });
      setDatasets(response.data.items);
    } catch (error) {
      console.error('Failed to fetch datasets:', error);
    }
  };

  // Active annotators selectable as assignee (large page size; a tenant has
  // far fewer users than this).
  const fetchUsers = async () => {
    try {
      const response = await settingsApi.getUsers({ page_size: 200 });
      setUsers((response.data.items || []).filter((u) => u.is_active !== false));
    } catch (error) {
      console.error('Failed to fetch users:', error);
    }
  };

  useEffect(() => {
    fetchDatasets();
    fetchUsers();
  }, []);

  const columns = [
    {
      title: t('pages.annotations.taskName'),
      dataIndex: 'name',
      key: 'name',
      render: (name: string, record: AnnotationTask) => (
        <a onClick={() => handleView(record)} style={{ color: '#1677ff' }}>{name}</a>
      ),
    },
    {
      title: t('pages.annotations.taskType'),
      dataIndex: 'task_type',
      key: 'task_type',
      render: (taskType: string | undefined) => {
        if (!taskType) return '-';
        return TASK_TYPE_KEY[taskType] ? t(`pages.annotations.${TASK_TYPE_KEY[taskType]}`) : taskType;
      },
    },
    {
      title: t('pages.annotations.annotationTool'),
      dataIndex: 'annotation_tool',
      key: 'annotation_tool',
      render: (tool: string | undefined) => {
        if (!tool) return '-';
        const toolMap: Record<string, string> = {
          'label-studio': t('pages.annotations.toolLabelStudio'),
          'cvat': t('pages.annotations.toolCVAT'),
          'doccano': t('pages.annotations.toolDoccano'),
        };
        return toolMap[tool] || tool;
      },
    },
    {
      title: t('pages.annotations.dataset'),
      dataIndex: 'dataset_id',
      key: 'dataset_id',
      ellipsis: true,
      render: (id: string | undefined, record: AnnotationTask) => {
        if (!id) return '-';
        // show the dataset's human name + version as <dataset>/<version>
        const ds = datasets.find((d) => d.id === id);
        const label = `${ds?.name || id}/${record.dataset_version ?? 1}`;
        return (
          <Button type="link" size="small" style={{ padding: 0 }} onClick={() => navigate(`/datasets/${id}/versions`)}>
            {label}
          </Button>
        );
      },
    },
    {
      title: t('pages.annotations.status'),
      dataIndex: 'status',
      key: 'status',
      render: (status: AnnotationTaskStatus) => <StatusBadge status={status} />,
    },
    {
      title: t('pages.annotations.progress'),
      dataIndex: 'progress',
      key: 'progress',
      render: (progress: number, record: AnnotationTask) => (
        <div style={{ width: 120 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }}>
            <span>{progress}%</span>
            <span>{record.annotated_items}/{record.total_items}</span>
          </div>
        </div>
      ),
    },
    {
      title: t('pages.annotations.assignee'),
      dataIndex: 'assignee_id',
      key: 'assignee_id',
      render: (id: string | undefined, record: AnnotationTask) => {
        if (!id) return <span style={{ color: '#bbb' }}>-</span>;
        const u = users.find((x) => x.id === id);
        return record.assignee_name || u?.full_name || u?.username || `${id.slice(0, 8)}...`;
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
      render: (_: unknown, record: AnnotationTask) => (
        <Space size={2}>
          <Tooltip title={t('common.edit')}>
            <Button type="text" size="small" icon={<EditOutlined />} onClick={() => handleEdit(record)} />
          </Tooltip>
          <Tooltip title={record.status === 'pending' ? t('common.start') : t('common.stop')}>
            <Button
              type="text"
              size="small"
              icon={record.status === 'pending' ? <PlayCircleOutlined /> : <StopOutlined />}
              onClick={() => handleToggle(record)}
            />
          </Tooltip>
          <Tooltip title={t('pages.annotations.studio.semiAuto')}>
            <Button type="text" size="small" icon={<ThunderboltOutlined />} onClick={() => handleStart(record)} />
          </Tooltip>
          <Tooltip title={t('common.delete')}>
            <Button type="text" size="small" danger icon={<DeleteOutlined />} onClick={() => handleDelete(record)} />
          </Tooltip>
        </Space>
      ),
    },
  ];

  const handleCreate = () => {
    setEditingTask(null);
    form.resetFields();
    setModalOpen(true);
  };

  const handleEdit = (task: AnnotationTask) => {
    setEditingTask(task);
    const schema = (task.schema || {}) as Record<string, unknown>;
    const labels = Array.isArray(schema.labels) ? schema.labels.join(', ') : '';
    form.setFieldsValue({
      name: task.name,
      description: task.description || '',
      task_type: task.task_type || 'detection',
      annotation_tool: task.annotation_tool || 'label-studio',
      labels,
      label_type: (schema.type as string) || 'rectanglelabels',
      dataset_id: task.dataset_id,
      dataset_version: task.dataset_version,
      assignee_id: task.assignee_id || undefined,
    });
    setModalOpen(true);
  };

  const handleView = async (task: AnnotationTask) => {
    // Pending LS projects are only provisioned on start; auto-start so the
    // studio opens directly without a separate Start click.
    if (task.status === 'pending') {
      try {
        await annotationApi.start(task.id);
      } catch (error) {
        console.error('Start failed:', error);
      }
    }
    navigate(`/annotations/${task.id}`);
  };

  const handleStart = async (task: AnnotationTask) => {
    try {
      await annotationApi.start(task.id);
      navigate(`/annotations/${task.id}`);
    } catch (error) {
      console.error('Start failed:', error);
    }
  };

  const handleToggle = async (task: AnnotationTask) => {
    try {
      if (task.status === 'pending') {
        await annotationApi.start(task.id);
      } else {
        await annotationApi.cancel(task.id);
      }
      fetchTasks();
    } catch (error) {
      console.error('Toggle failed:', error);
    }
  };

  const handleDelete = async (task: AnnotationTask) => {
    if (window.confirm(t('common.confirmDelete') || `Delete ${task.name}?`)) {
      try {
        await annotationApi.delete(task.id);
        fetchTasks();
      } catch (error) {
        console.error('Delete failed:', error);
      }
    }
  };

  const handleSubmit = async (values: {
    name: string;
    description: string;
    task_type: string;
    annotation_tool: string;
    labels?: string;
    label_type?: string;
    dataset_id: string;
    dataset_version: number;
    assignee_id?: string;
  }) => {
    // Build schema from form values: labels (comma-separated) + type
    const schema: Record<string, unknown> = {
      task_type: values.task_type,
    };
    if (values.labels && values.labels.trim()) {
      const labels = values.labels.split(',').map((s) => s.trim()).filter(Boolean);
      if (labels.length) schema['labels'] = labels;
    }
    if (values.label_type) schema['type'] = values.label_type;
    const payload = {
      name: values.name,
      description: values.description,
      task_type: values.task_type,
      annotation_tool: values.annotation_tool,
      schema,
      dataset_id: values.dataset_id,
      dataset_version: Number(values.dataset_version),
      assignee_id: values.assignee_id || '',
    };
    try {
      if (editingTask) {
        await annotationApi.update(editingTask.id, payload);
      } else {
        await annotationApi.create(payload);
      }
      setModalOpen(false);
      fetchTasks();
    } catch (error) {
      console.error('Save failed:', error);
    }
  };

  return (
    <div>
      <PageHeader
        title={t('pages.annotations.title')}
        onCreate={handleCreate}
        onRefresh={fetchTasks}
      actions={[<HelpButton module="annotations" />]}
      />
      <Card>
        <FilterSearchBar
          filterPlaceholder={t('pages.annotations.filterAssignee')}
          filterValue={assigneeFilter}
          onFilterChange={setAssigneeFilter}
          filterOptions={[
            { value: 'any', label: t('pages.annotations.assigneeAssigned') },
            { value: 'none', label: t('pages.annotations.assigneeUnassigned') },
            ...users.map((u) => ({
              value: u.id,
              label: u.full_name ? `${u.full_name} (${u.username})` : u.username,
            })),
          ]}
          searchPlaceholder={t('pages.annotations.searchPlaceholder')}
          searchValue={searchInput}
          onSearchChange={setSearchInput}
        />
        <Table
          {...tasksTable.tableProps}
          columns={columns}
          scroll={{ x: 'max-content' }}
        />
      </Card>

      <Modal
        title={editingTask ? t('common.edit') : t('common.create')}
        open={modalOpen}
        onOk={() => form.validateFields().then(handleSubmit)}
        onCancel={() => setModalOpen(false)}

        width={600}
      >
        <Form form={form} layout="vertical">
          <Form.Item
            name="name"
            label={t('pages.annotations.taskName')}
            rules={[{ required: true, message: t('common.required') }]}
          >
            <Input placeholder={t('pages.annotations.taskName')} />
          </Form.Item>
          <Form.Item
            name="description"
            label={t('pages.datasets.description')}
          >
            <Input.TextArea rows={2} placeholder={t('pages.datasets.description')} />
          </Form.Item>

          <Divider plain orientation="left" style={{ fontSize: 13, margin: '12px 0' }}>
            {t('pages.annotations.taskType')}
          </Divider>

          <Form.Item
            name="task_type"
            label={t('pages.annotations.taskType')}
            rules={[{ required: true, message: t('common.required') }]}
          >
            <Select placeholder={t('pages.annotations.selectTaskType')}>
              <Select.OptGroup label={t('pages.annotations.cvTasks')}>
                {CV_TASK_TYPES.map(v => (
                  <Select.Option key={v.value} value={v.value}>{t(`pages.annotations.${v.key}`)}</Select.Option>
                ))}
              </Select.OptGroup>
              <Select.OptGroup label={t('pages.annotations.llmTasks')}>
                {NLP_TASK_TYPES.map(v => (
                  <Select.Option key={v.value} value={v.value}>{t(`pages.annotations.${v.key}`)}</Select.Option>
                ))}
              </Select.OptGroup>
              <Select.OptGroup label={t('pages.annotations.customConfig')}>
                <Select.Option key="custom" value="custom">{t('pages.annotations.customConfigLabel')}</Select.Option>
              </Select.OptGroup>
            </Select>
          </Form.Item>

          <Form.Item
            name="annotation_tool"
            label={t('pages.annotations.annotationTool')}
            rules={[{ required: true, message: t('common.required') }]}
          >
            <Select
              placeholder={t('pages.annotations.selectAnnotationTool')}
              options={ANNOTATION_TOOLS.map(v => ({ value: v, label: t(`pages.annotations.tool${v === 'label-studio' ? 'LabelStudio' : v === 'cvat' ? 'CVAT' : 'Doccano'}`) }))}
              disabled
            />
          </Form.Item>

          <Form.Item
            name="labels"
            label={t('pages.annotations.labels')}
            extra={t('pages.annotations.labelsHint')}
          >
            <Input placeholder={t('pages.annotations.labelsPlaceholder')} />
          </Form.Item>

          <Form.Item
            name="label_type"
            label={t('pages.annotations.labelType')}
            rules={[{ required: true, message: t('common.required') }]}
          >
            <Select placeholder={t('pages.annotations.labelType')} options={LABEL_TYPES.map((v) => ({ value: v.value, label: v.label }))} />
          </Form.Item>

          <Form.Item
            name="dataset_id"
            label={t('pages.annotations.dataset')}
            rules={[{ required: true, message: t('common.required') }]}
          >
            <Select
              placeholder={t('pages.annotations.dataset')}
              options={datasets.map(d => ({ label: d.name, value: d.id }))}
            />
          </Form.Item>
          <Form.Item
            name="dataset_version"
            label={t('pages.datasets.version')}
            rules={[{ required: true, message: t('common.required') }]}
          >
            <Select
              placeholder={t('pages.datasets.version')}
              options={versions.map(v => ({ label: `${t('pages.datasets.version')} ${v.version} (${v.file_count} files)`, value: v.version }))}
              loading={versions.length === 0 && !!watchDatasetId}
            />
          </Form.Item>
          <Form.Item
            name="assignee_id"
            label={t('pages.annotations.assignee')}
          >
            <Select
              allowClear
              showSearch
              optionFilterProp="label"
              placeholder={t('pages.annotations.assigneePlaceholder')}
              options={users.map((u) => ({
                value: u.id,
                label: u.full_name ? `${u.full_name} (${u.username})` : u.username,
              }))}
            />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

export default Annotations;
