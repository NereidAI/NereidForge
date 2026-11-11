// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

import React, { useEffect, useState, useCallback, useMemo } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { Card, Table, Button, Space, Modal, Form, Input, Select, Spin, Tabs, Progress, Alert, Row, Col, Statistic, InputNumber, Tooltip, Descriptions, message, Popconfirm, Collapse, Slider, Radio, AutoComplete, Switch, Divider, Tag } from 'antd';
import type { FormInstance } from 'antd';
import { DeleteOutlined, PlayCircleOutlined, StopOutlined, ReloadOutlined, DownloadOutlined, RedoOutlined, EyeOutlined, RocketOutlined } from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import { trainingApi, modelApi, datasetApi, imageApi, settingsApi } from '@api/endpoints';
import { PageHeader } from '@components/PageHeader';
import { HelpButton } from '@components/HelpDrawer';
import { statusLabelKey, useJobStatusOptions } from '@components/StatusBadge';
import { TrainingCurves } from '@components/TrainingCurves';
import { useServerTable } from '@hooks/useServerTable';
import { useDebounce } from '@hooks/useDebounce';
import FilterSearchBar from '@components/FilterSearchBar';
import { formatDateTime } from '../utils/format';
import { TRAINER_IMAGE_FALLBACKS, compatibleTrainerImage } from '../utils/images';
import { defaultArchFor, trainingDurationSeconds, trainingSelectionFromConfig } from '../utils/training';
import type { InitializationMode } from '../utils/training';
import {
  announcedArchOptions,
  announcedDefaultArch,
  matchingImages as announcedMatching,
  taskLock,
} from '../utils/capabilities';
import type { CapabilityImage, GpuVendorCapacity } from '@api/endpoints';
import type { TrainingJob, TrainingConfig, TrainingMetrics, TrainingRuntime, Model, Dataset, DatasetVersion, BaseModelPoolEntry } from '@nftypes/api';

// Augmentation presets — selecting one fills every slider; the advanced panel
// lets users override individual values. Only shown for the image
// detection/segmentation task types (classification and LLM/VLM use their own
// concepts). The trainer applies the geometric subset (hflip/vflip/rotation/
// scale) plus colour jitter; mosaic/mixup/copy-paste need multi-image
// collation and are accepted but currently unused (ADR-014 follow-up).
type AugKey =
  | 'fliplr' | 'flipud' | 'degrees' | 'shear' | 'translate' | 'scale' | 'perspective'
  | 'hsv_h' | 'hsv_s' | 'hsv_v' | 'mosaic' | 'mixup' | 'copy_paste' | 'close_mosaic'
  | 'patience' | 'freeze' | 'weight_decay'
  | 'optimizer' | 'lr0' | 'lrf' | 'cos_lr' | 'warmup_epochs' | 'warmup_momentum'
  | 'warmup_bias_lr' | 'multi_scale' | 'dropout' | 'seed' | 'deterministic'
  | 'amp' | 'save_period';

type AugValue = number | string | boolean;

// A registered model version with the resolved trainable checkpoint (see
// GET /models/trainable-versions). checkpoints only exist for trainable
// versions; ONNX-only versions carry an empty checkpoint_uri.
interface TrainableVersion {
  model_id: string;
  model_name: string;
  version: string;
  task_type: string;
  trainable: boolean;
  checkpoint_uri: string;
  size_bytes: number;
}

// Control widgets used by the parameter panel: range sliders, numeric inputs,
// on/off switches and dropdown selects.
type AugWidget = 'slider' | 'number' | 'switch' | 'select';

interface AugFieldDef {
  key: AugKey;
  widget: AugWidget;
  // Augmentation knobs live behind "advanced"; production controls are always
  // visible because they directly affect the shipped model.
  group: 'augmentation' | 'production';
  min?: number;
  max?: number;
  step?: number;
  options?: string[];
}

const AUG_PRESETS: Record<'off' | 'light' | 'medium' | 'strong', Record<AugKey, AugValue>> = {
  off: { fliplr: 0, flipud: 0, degrees: 0, shear: 0, translate: 0, scale: 0, perspective: 0, hsv_h: 0, hsv_s: 0, hsv_v: 0, mosaic: 0, mixup: 0, copy_paste: 0, close_mosaic: 0, patience: 100, freeze: 0, weight_decay: 0.0005, optimizer: 'auto', lr0: 0.01, lrf: 0.01, cos_lr: false, warmup_epochs: 3, warmup_momentum: 0.8, warmup_bias_lr: 0.1, multi_scale: false, dropout: 0, seed: 0, deterministic: true, amp: true, save_period: 0 },
  light: { fliplr: 0.5, flipud: 0, degrees: 0, shear: 0, translate: 0.05, scale: 0.3, perspective: 0, hsv_h: 0.01, hsv_s: 0.4, hsv_v: 0.3, mosaic: 1, mixup: 0, copy_paste: 0, close_mosaic: 10, patience: 50, freeze: 0, weight_decay: 0.0005, optimizer: 'auto', lr0: 0.01, lrf: 0.01, cos_lr: false, warmup_epochs: 3, warmup_momentum: 0.8, warmup_bias_lr: 0.1, multi_scale: false, dropout: 0, seed: 0, deterministic: true, amp: true, save_period: 0 },
  medium: { fliplr: 0.5, flipud: 0, degrees: 0, shear: 0, translate: 0.1, scale: 0.5, perspective: 0, hsv_h: 0.015, hsv_s: 0.7, hsv_v: 0.4, mosaic: 1, mixup: 0, copy_paste: 0, close_mosaic: 10, patience: 50, freeze: 0, weight_decay: 0.0005, optimizer: 'auto', lr0: 0.01, lrf: 0.01, cos_lr: false, warmup_epochs: 3, warmup_momentum: 0.8, warmup_bias_lr: 0.1, multi_scale: false, dropout: 0, seed: 0, deterministic: true, amp: true, save_period: 0 },
  strong: { fliplr: 0.5, flipud: 0, degrees: 5, shear: 2, translate: 0.15, scale: 0.7, perspective: 0, hsv_h: 0.02, hsv_s: 0.8, hsv_v: 0.5, mosaic: 1, mixup: 0.15, copy_paste: 0, close_mosaic: 15, patience: 60, freeze: 0, weight_decay: 0.0005, optimizer: 'auto', lr0: 0.01, lrf: 0.01, cos_lr: true, warmup_epochs: 3, warmup_momentum: 0.8, warmup_bias_lr: 0.1, multi_scale: false, dropout: 0.05, seed: 0, deterministic: true, amp: true, save_period: 0 },
};

// Field definitions mirroring AUG_SPECS in train.py (validation still happens
// server-side; these bounds only drive the widgets).
const AUG_FIELDS: AugFieldDef[] = [
  { key: 'fliplr', widget: 'slider', group: 'augmentation', min: 0, max: 1, step: 0.05 },
  { key: 'flipud', widget: 'slider', group: 'augmentation', min: 0, max: 1, step: 0.05 },
  { key: 'degrees', widget: 'slider', group: 'augmentation', min: 0, max: 15, step: 1 },
  { key: 'shear', widget: 'slider', group: 'augmentation', min: 0, max: 10, step: 1 },
  { key: 'translate', widget: 'slider', group: 'augmentation', min: 0, max: 0.3, step: 0.01 },
  { key: 'scale', widget: 'slider', group: 'augmentation', min: 0, max: 0.9, step: 0.05 },
  { key: 'perspective', widget: 'slider', group: 'augmentation', min: 0, max: 0.01, step: 0.001 },
  { key: 'hsv_h', widget: 'slider', group: 'augmentation', min: 0, max: 0.1, step: 0.005 },
  { key: 'hsv_s', widget: 'slider', group: 'augmentation', min: 0, max: 1, step: 0.05 },
  { key: 'hsv_v', widget: 'slider', group: 'augmentation', min: 0, max: 1, step: 0.05 },
  { key: 'mosaic', widget: 'slider', group: 'augmentation', min: 0, max: 1, step: 0.05 },
  { key: 'mixup', widget: 'slider', group: 'augmentation', min: 0, max: 0.5, step: 0.05 },
  { key: 'copy_paste', widget: 'slider', group: 'augmentation', min: 0, max: 1, step: 0.05 },
  { key: 'close_mosaic', widget: 'slider', group: 'augmentation', min: 0, max: 50, step: 1 },
  { key: 'freeze', widget: 'slider', group: 'augmentation', min: 0, max: 24, step: 1 },
  { key: 'weight_decay', widget: 'slider', group: 'augmentation', min: 0, max: 0.01, step: 0.0001 },
  // Production controls: early stopping, optimizer/LR, scale robustness and
  // reproducibility/precision switches.
  { key: 'patience', widget: 'number', group: 'production', min: 0, max: 300, step: 5 },
  { key: 'optimizer', widget: 'select', group: 'production', options: ['auto', 'SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp'] },
  { key: 'lr0', widget: 'number', group: 'production', min: 0, max: 1, step: 0.001 },
  { key: 'lrf', widget: 'number', group: 'production', min: 0.0001, max: 1, step: 0.001 },
  { key: 'cos_lr', widget: 'switch', group: 'production' },
  { key: 'warmup_epochs', widget: 'number', group: 'production', min: 0, max: 10, step: 0.5 },
  { key: 'warmup_momentum', widget: 'number', group: 'production', min: 0, max: 0.95, step: 0.05 },
  { key: 'warmup_bias_lr', widget: 'number', group: 'production', min: 0, max: 1, step: 0.01 },
  { key: 'multi_scale', widget: 'switch', group: 'production' },
  { key: 'dropout', widget: 'number', group: 'production', min: 0, max: 0.7, step: 0.01 },
  { key: 'seed', widget: 'number', group: 'production', min: 0, max: 999999999, step: 1 },
  { key: 'deterministic', widget: 'switch', group: 'production' },
  { key: 'amp', widget: 'switch', group: 'production' },
  { key: 'save_period', widget: 'number', group: 'production', min: 0, max: 100, step: 1 },
];

const SLIDER_FIELDS = AUG_FIELDS.filter((f) => f.widget === 'slider');
const PRODUCTION_FIELDS = AUG_FIELDS.filter((f) => f.group === 'production');

// Common vision architecture ids offered for classification jobs (free-text
// custom ids are also accepted). torchvision ids and timm ids share one
// namespace: the trainer falls back to timm when the name is not in the
// torchvision catalog (ADR-014 D1).
const CLASSIFICATION_ARCHS = [
  'resnet18', 'resnet50', 'efficientnet_b0', 'mobilenetv3_large_100',
  'convnext_tiny', 'vit_base_patch16_224', 'vit_small_patch16_224',
];

// Per-task-type sensible defaults applied when switching the task selector,
// so CV / classification / LLM / VLM never inherit each other's hyperparams.
function applyTaskTypeDefaults(form: FormInstance, tt: string) {
  const patch: Record<string, unknown> = { custom_image: '', workers: undefined };
  if (tt === 'llm' || tt === 'vlm') {
    if (!form.getFieldValue('accelerator') || form.getFieldValue('accelerator') === 'cpu') {
      patch.accelerator = 'nvidia';
    }
    patch.base_model = undefined;
    patch.architecture = undefined;
    patch.initialization = undefined;
    patch.starting_weights = undefined;
    patch.gpu_count = 1;
    patch.epochs = tt === 'vlm' ? 5 : 3;
    patch.batch = tt === 'vlm' ? 2 : 4;
    patch.llm_learning_rate = 2e-4;
    patch.llm_grad_accum = 4;
    patch.llm_cutoff = tt === 'vlm' ? 512 : 1024;
    patch.llm_rank = 16;
    patch.llm_alpha = 32;
    patch.llm_dropout = 0.05;
    patch.llm_template = 'qwen';
    patch.llm_quant = 0;
  } else if (tt === 'classification') {
    patch.accelerator = 'cpu';
    patch.gpu_count = undefined;
    patch.base_model = undefined;
    patch.architecture = 'resnet50';
    patch.initialization = 'pretrained';
    patch.starting_weights = undefined;
    patch.epochs = 50;
    patch.batch = 32;
    patch.imgsz = 224;
    patch.split_ratio = 0.8;
  } else {
    // detection / segmentation / pose — same torch image, task picks the head
    patch.accelerator = 'cpu';
    patch.gpu_count = undefined;
    patch.base_model = undefined;
    patch.architecture = defaultArchFor(tt);
    patch.initialization = 'pretrained';
    patch.starting_weights = undefined;
    patch.epochs = 100;
    patch.batch = 16;
    patch.imgsz = 640;
    patch.split_ratio = 0.8;
    patch.aug_preset = 'medium';
    AUG_FIELDS.forEach((f) => { patch[`aug_${f.key}`] = AUG_PRESETS.medium[f.key]; });
  }
  form.setFieldsValue(patch);
}

// SFT corpus formats (sharegpt JSONL) usable only by LLM/VLM jobs.
const SFT_DATASET_FORMATS = new Set(['llm-sft', 'jsonl']);

// Whether a dataset corpus is selectable for a given training task type:
// - llm/vlm  -> only llm-sft/jsonl corpora
// - classification -> imagenet corpora or corpora annotated for classification
// - detection/segmentation -> coco/yolo corpora whose annotation-derived
//   task types include it; a coco/yolo corpus with no inferred type is a
//   generic CV corpus usable for both detection and segmentation.
function datasetMatchesTaskType(
  d: Pick<Dataset, 'format' | 'task_types'>,
  taskType: string,
): boolean {
  const fmt = d.format || 'coco';
  const isSft = SFT_DATASET_FORMATS.has(fmt);
  const isLlmTask = taskType === 'llm' || taskType === 'vlm';
  if (isLlmTask) return isSft;
  if (isSft) return false;
  if (taskType === 'classification') {
    return fmt === 'imagenet' || (d.task_types ?? []).includes('classification');
  }
  if (taskType === 'detection' || taskType === 'segmentation') {
    const types = d.task_types ?? [];
    if (fmt === 'imagenet') return false;
    return types.length === 0 || types.includes(taskType);
  }
  return !isSft;
}

export const Training: React.FC = () => {
  const { t } = useTranslation();
  const navigate = useNavigate();
  // Server-side status filter plus debounced job-name search.
  const [statusFilter, setStatusFilter] = useState('');
  const [searchInput, setSearchInput] = useState('');
  const search = useDebounce(searchInput.trim(), 300);
  const statusOptions = useJobStatusOptions();
  // Server-side paginated job list; polling keeps the visible page fresh.
  const jobsTable = useServerTable<TrainingJob, { status?: string; search?: string }>({
    list: (params) => trainingApi.list(params),
    filters: {
      status: statusFilter || undefined,
      search: search || undefined,
    },
    defaultPageSize: 10,
    pollMs: 5000,
  });
  const { reload: fetchJobs } = jobsTable;
  const [models, setModels] = useState<Model[]>([]);
  const [trainImages, setTrainImages] = useState<Array<{ full_ref: string; name: string; task_types: string[] }>>([]);
  // Capabilities announced by registered trainer images (ADR-015); the dialog
  // derives image/architecture options and lock states from these, never from
  // a hardcoded platform list.
  const [capImages, setCapImages] = useState<CapabilityImage[]>([]);
  const [baseModels, setBaseModels] = useState<Array<{ name: string; uri: string }>>([]);
  // CV checkpoints in the managed base-model pool (base-models/cv/*.pt),
  // shown only when checkpoint initialization is selected.
  const [cvPoolEntries, setCvPoolEntries] = useState<BaseModelPoolEntry[]>([]);
  // Registered model versions usable as the starting checkpoint for
  // continued / incremental training. Non-trainable versions (ONNX-only)
  // are kept so the selector can show them disabled.
  const [trainableVersions, setTrainableVersions] = useState<TrainableVersion[]>([]);
  // Real GPU/vGPU capacity advertised by node device plugins (shared-dev-num);
  // empty means node reads are unavailable and GPU count stays unbounded.
  const [gpuCapacity, setGpuCapacity] = useState<Record<string, GpuVendorCapacity>>({});
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [versions, setVersions] = useState<DatasetVersion[]>([]);
  const [modalOpen, setModalOpen] = useState(false);
  const [editingJob, setEditingJob] = useState<TrainingJob | null>(null);
  const [form] = Form.useForm();
  const selectedDataset = Form.useWatch('dataset_id', form);
  // The dataset picker is filtered by the selected task type: SFT corpora
  // only show for LLM/VLM, and CV corpora are narrowed by the annotation
  // task type they were created for (detection/segmentation/classification).
  const selectedTaskType = Form.useWatch('task_type', form);
  const formTaskType = selectedTaskType || 'detection';
  const isLlmFormTask = formTaskType === 'llm' || formTaskType === 'vlm';
  const visibleDatasets = datasets.filter((d) => datasetMatchesTaskType(d, formTaskType));

  // Load annotated versions after dataset selection (only show annotated ones for training)
  useEffect(() => {
    if (!selectedDataset) {
      setVersions([]);
      return;
    }
    datasetApi.getVersions(selectedDataset, { page_size: 100 }).then((res) => {
      const all = res.data.items || [];
      setVersions(all.filter((v) => v.is_annotated) as typeof versions);
    }).catch(() => setVersions([]));
  }, [selectedDataset]);

  const fetchModels = async () => {
    try {
      const res = await modelApi.list({ page_size: 100 });
      setModels(res.data.items);
    } catch (e) { console.error(e); }
  };

  // trainer image options come from the image catalog (System → Images);
  // fall back to the hardcoded built-ins when the catalog is unavailable.
  const fetchTrainImages = async () => {
    try {
      const res = await imageApi.list({ category: "train" });
      setTrainImages(res.data.items || []);
    } catch (e) { console.error(e); }
    // Announced capabilities drive the option lists; tolerate its absence
    // (older backends / offline dev): the legacy catalog rows remain usable.
    try {
      const res = await imageApi.capabilities();
      setCapImages(res.data.images || []);
    } catch (e) { /* announcements are optional */ }
  };

  const fetchDatasets = async () => {
    try {
      const res = await datasetApi.list({ page_size: 100 });
      setDatasets(res.data.items);
    } catch (e) { console.error(e); }
  };

  useEffect(() => {
    fetchModels();
    fetchDatasets();
    fetchTrainImages();
    import('@api/client').then(({ default: apiClient }) => {
      apiClient.get<{ items: Array<{ name: string; uri: string }> }>('/base-models').then((r) => setBaseModels(r.data.items || [])).catch(() => {});
      apiClient.get<{ items: BaseModelPoolEntry[] }>('/base-model-pool?kind=cv&page_size=100').then((r) => setCvPoolEntries(r.data.items || [])).catch(() => {});
      apiClient.get<{ items: TrainableVersion[] }>('/models/trainable-versions').then((r) => setTrainableVersions(r.data.items || [])).catch(() => {});
      settingsApi.getGpuCapacity().then((r) => setGpuCapacity(r.data.vendors || {})).catch(() => {});
    });
  }, []);

  // Options for the "continue from my model" groups, keyed by form task type.
  // ONNX-only versions are rendered disabled with an explanatory label and
  // native title tooltip.
  const continueOptions = useMemo(() => {
    const byTask: Record<string, TrainableVersion[]> = {
      detection: [], segmentation: [], classification: [], llm: [], vlm: [],
    };
    for (const v of trainableVersions) {
      (byTask[v.task_type] ||= []).push(v);
    }
    const toOption = (v: TrainableVersion) =>
      v.trainable
        ? { value: v.checkpoint_uri, label: `${v.model_name} · ${v.version}` }
        : {
            value: `__untrainable__${v.model_id}__${v.version}`,
            label: `${v.model_name} · ${v.version} · ${t('pages.training.onnxOnlySuffix')}`,
            disabled: true,
            title: t('pages.training.onnxOnlyHint'),
          };
    return {
      detection: byTask.detection.map(toOption),
      segmentation: byTask.segmentation.map(toOption),
      classification: byTask.classification.map(toOption),
      llm: [...byTask.llm, ...byTask.vlm].map(toOption),
    };
  }, [trainableVersions, t]);

  // Lookup from the resolved checkpoint URI back to its registered version,
  // so picking a "continue" option can attach the source model id to the job.
  const trainableByURI = useMemo(
    () => new Map(trainableVersions.filter((v) => v.trainable).map((v) => [v.checkpoint_uri, v])),
    [trainableVersions],
  );

  // Sync the hidden source id and optional target model when a registered
  // checkpoint is selected. Architecture-only initialization has no source
  // model and never passes through this checkpoint-specific handler.
  const syncStartingWeights = (value: string) => {
    const picked = trainableByURI.get(value);
    if (picked) {
      form.setFieldValue('base_model_id', picked.model_id);
      form.setFieldValue('model_id', picked.model_id);
      return;
    }
    form.setFieldValue('base_model_id', '');
    const stem = value.trim().replace(/\.[a-z]+$/i, '');
    const stock = models.find((m) => m.name === stem);
    form.setFieldValue('model_id', stock?.id || undefined);
  };

  // Once the pool loads, upgrade a cloned legacy weights file name to the
  // matching pool S3 URI so the checkpoint selector shows a real option.
  useEffect(() => {
    if (!cvPoolEntries.length) return;
    const cur = String(form.getFieldValue('starting_weights') || '');
    if (!cur || cur.startsWith('s3://') || cur.includes('/')) return;
    const match = cvPoolEntries.find((e) => e.name === cur);
    if (match) {
      form.setFieldValue('starting_weights', match.uri);
      form.setFieldValue('base_model_id', '');
    }
  }, [cvPoolEntries, form]);

  const columns = [
    {
      title: t('pages.training.jobName'),
      dataIndex: 'name',
      key: 'name',
      render: (name: string, record: TrainingJob) => (
        <a onClick={() => handleView(record)}>{name}</a>
      ),
    },
    {
      title: t('pages.training.model'),
      dataIndex: 'model_id',
      key: 'model_id',
      ellipsis: true,
      render: (_: unknown, record: TrainingJob) => {
        // Prefer the artifact actually produced by this training run; fall
        // back to the source/base model when no model was registered yet.
        const producedId = record.output_model_id;
        if (producedId) {
          const m = models.find((x) => x.id === producedId);
          const label = record.output_model_name || m?.name || producedId.slice(0, 12);
          return <Tooltip title={record.output_model_version ? `v${record.output_model_version}` : undefined}><span>{label}</span></Tooltip>;
        }
        const srcId = record.model_id;
        if (!srcId) return '-';
        const m = models.find((x) => x.id === srcId);
        return m?.name || srcId.slice(0, 12);
      },
    },
    {
      title: t('pages.training.dataset'),
      dataIndex: 'dataset_id',
      key: 'dataset_id',
      ellipsis: true,
      render: (id: string | undefined, record: TrainingJob) => {
        if (!id) return '-';
        const d = datasets.find((x) => x.id === id);
        return d ? `${d.name} (v${record.dataset_version})` : id.slice(0, 12);
      },
    },
    {
      title: t('pages.training.trainerImage'),
      dataIndex: 'trainer_image',
      key: 'trainer_image',
      ellipsis: true,
      render: (img: string | undefined) => {
        if (!img) return '-';
        const short = img.split('/').pop() || img;
        return <Tooltip title={img}><span>{short}</span></Tooltip>;
      },
    },
    {
      title: t('pages.training.status'),
      key: 'status',
      width: 180,
      render: (_: unknown, record: TrainingJob) => {
        const status = record.status;
        const pct = typeof record.progress === 'number' ? Math.max(0, Math.min(100, record.progress)) : 0;
        // Map job status to antd Progress status:
        // running -> active (animated), completed -> success, failed -> exception,
        // pending/queued/cancelled -> normal (no animation).
        const progStatus: 'success' | 'exception' | 'active' | 'normal' =
          status === 'completed' ? 'success' :
          status === 'failed' ? 'exception' :
          status === 'running' ? 'active' : 'normal';
        // Idle (pending/queued) and cancelled bars render gray and static so
        // they don't look like an in-flight run.
        const idle = status === 'pending' || status === 'queued' || status === 'cancelled';
        // The status text lives in the hover tooltip instead of a separate tag.
        return (
          <Space direction="vertical" size={2} style={{ width: '100%' }}>
            <Tooltip title={t(statusLabelKey(status))}>
              <Progress
                percent={idle ? 0 : pct}
                size="small"
                showInfo
                status={progStatus}
                strokeColor={idle ? '#d9d9d9' : undefined}
              />
            </Tooltip>
            {record.current_epoch != null && record.total_epochs != null && record.total_epochs > 0 && (
              <span style={{ fontSize: 12, color: '#888' }}>
                {t('pages.training.epochProgress', { current: record.current_epoch, total: record.total_epochs })}
              </span>
            )}
          </Space>
        );
      },
    },
    {
      title: t('pages.training.metrics'),
      dataIndex: 'metrics',
      key: 'metrics',
      width: 160,
      render: (metrics: TrainingMetrics | undefined) => {
        if (!metrics) return '-';
        const m = metrics as unknown as Record<string, unknown>;
        // VLM/LLM jobs report a single training-loss value instead of
        // detection metrics.
        if (typeof m.final_loss === 'number') {
          return (
            <Tooltip title={t('pages.training.metricsTipLoss')}>
              <span style={{ fontSize: 12 }}>Loss {m.final_loss.toFixed(4)}</span>
            </Tooltip>
          );
        }
        // Four key detection metrics, always rendered as a 2x2 grid. Training
        // jobs persist "mAP50-95" (hyphen) while evaluator jobs use
        // "mAP50_95" (underscore); accept both.
        const cells: Array<{ label: string; color: string; v: unknown }> = [
          { label: 'mAP50-95', color: '#722ed1', v: m['mAP50-95'] ?? m.mAP50_95 },
          { label: 'R', color: '#fa8c16', v: m.recall },
        ];
        return (
          <Space size={12}>
            {cells.map(({ label, color, v }) => (
              <span key={label} style={{ fontSize: 12 }}>
                <span style={{ color, fontWeight: 600 }}>{label} </span>
                {typeof v === 'number' ? `${Math.round(v * 100)}%` : '-'}
              </span>
            ))}
          </Space>
        );
      },
    },
    {
      title: t('common.createdAt'),
      dataIndex: 'created_at',
      key: 'created_at',
      width: 170,
      render: (date: string) => formatDateTime(date),
    },
    {
      title: t('common.actions'),
      key: 'actions',
      width: 140,
      render: (_: unknown, record: TrainingJob) => (
        <Space size={2}>
          <Tooltip title={t('pages.training.retrain')}>
            <Button type="text" size="small" icon={<RedoOutlined />} onClick={() => handleRetrain(record)} />
          </Tooltip>
          <Tooltip title={record.status === 'running' ? t('common.stop') : t('common.start')}>
            <Button
              type="text"
              size="small"
              icon={record.status === 'running' ? <StopOutlined /> : <PlayCircleOutlined />}
              onClick={() => handleToggle(record)}
            />
          </Tooltip>
          <Tooltip title={t('common.restart')}>
            <Button type="text" size="small" icon={<ReloadOutlined />} onClick={() => handleRestart(record)} />
          </Tooltip>
          <Tooltip title={t('common.delete')}>
            <Button type="text" size="small" danger icon={<DeleteOutlined />} onClick={() => handleDelete(record)} />
          </Tooltip>
        </Space>
      ),
    },
  ];

  const handleCreate = () => {
    setEditingJob(null);
    form.resetFields();
    // Default to detection with the "medium" augmentation preset.
    applyTaskTypeDefaults(form, 'detection');
    form.setFieldsValue({ task_type: 'detection', aug_preset: 'medium' });
    setModalOpen(true);
  };

  const handleView = (job: TrainingJob) => {
    navigate(`/training/${job.id}`);
  };

  // Retrain (clone): pre-fill the form with the original job's params/config, then create a new training job on save.
  const handleRetrain = (job: TrainingJob) => {
    // Clone into a clean store so fields absent from the source cannot leak in
    // from a previously opened job, including fields inside collapsed panels.
    form.resetFields();
    const cfg = (job.config || {}) as unknown as Record<string, unknown>;
    const tt = (cfg.task_type as string) || 'detection';
    const isLlmJob = tt === 'llm' || tt === 'vlm';
    const llm = (cfg.llm || {}) as Record<string, unknown>;
    const selection = trainingSelectionFromConfig(cfg, tt);
    const cfgAccelerator = (cfg.accelerator as string) || (isLlmJob ? 'nvidia' : 'cpu');
    const cfgGpuCount = typeof cfg.gpu_count === 'number'
      ? cfg.gpu_count
      : typeof cfg.num_proc_per_node === 'number'
        ? cfg.num_proc_per_node
        : cfgAccelerator === 'cpu' ? undefined : 1;
    form.setFieldsValue({
      name: `${job.name}-retrain`,
      description: '',
      model_id: job.model_id || job.output_model_id || undefined,
      dataset_id: job.dataset_id,
      dataset_version: job.dataset_version,
      task_type: tt,
      accelerator: cfgAccelerator,
      gpu_count: cfgGpuCount,
      epochs: typeof cfg.epochs === 'number' ? cfg.epochs : undefined,
      imgsz: typeof cfg.imgsz === 'number' ? cfg.imgsz : undefined,
      batch: typeof cfg.batch === 'number' ? cfg.batch : undefined,
      workers: typeof cfg.workers === 'number' ? cfg.workers : undefined,
      base_model: isLlmJob ? (cfg.base_model as string) || undefined : undefined,
      base_model_id: (cfg.base_model_id as string) || undefined,
      architecture: isLlmJob ? undefined : selection.architecture,
      initialization: isLlmJob ? undefined : selection.initialization,
      starting_weights: isLlmJob ? undefined : selection.startingWeights || undefined,
      split_ratio: typeof cfg.split_ratio === 'number' ? cfg.split_ratio : undefined,
      custom_image: compatibleTrainerImage(
        (cfg as { trainer_image?: string }).trainer_image || (job as unknown as { trainer_image?: string }).trainer_image,
        tt,
        cfgAccelerator,
      ),
      custom_script_uri: (cfg as { custom_script_uri?: string }).custom_script_uri || '',
      // LLM/VLM LoRA hyperparameters from the source job's llm{} block.
      llm_learning_rate: typeof llm.learning_rate === 'number' ? llm.learning_rate : undefined,
      llm_grad_accum: typeof llm.gradient_accumulation_steps === 'number' ? llm.gradient_accumulation_steps : undefined,
      llm_cutoff: typeof llm.cutoff_len === 'number' ? llm.cutoff_len : undefined,
      llm_rank: typeof llm.lora_rank === 'number' ? llm.lora_rank : undefined,
      llm_alpha: typeof llm.lora_alpha === 'number' ? llm.lora_alpha : undefined,
      llm_dropout: typeof llm.lora_dropout === 'number' ? llm.lora_dropout : undefined,
      llm_template: (llm.template as string) || 'qwen',
      llm_quant: typeof llm.quantization_bit === 'number' ? llm.quantization_bit : 0,
    });
    // Restore augmentation preset + individual sliders when the source job had them.
    const aug = cfg.augmentation as Partial<Record<AugKey, number>> & { preset?: string } | undefined;
    if (aug) {
      const preset = (aug.preset as 'off' | 'light' | 'medium' | 'strong') || 'medium';
      const vals = AUG_PRESETS[preset];
      const patch: Record<string, unknown> = { aug_preset: preset };
      AUG_FIELDS.forEach((f) => {
        patch[`aug_${f.key}`] = (typeof aug[f.key] === 'number'
          || typeof aug[f.key] === 'boolean'
          || typeof aug[f.key] === 'string')
          ? (aug[f.key] as AugValue) : vals[f.key];
      });
      form.setFieldsValue(patch);
    }
    setEditingJob(null);
    setModalOpen(true);
  };

  const handleToggle = async (job: TrainingJob) => {
    try {
      if (job.status === 'running') {
        await trainingApi.stop(job.id);
      } else {
        await trainingApi.start(job.id);
      }
      fetchJobs();
    } catch (error) {
      console.error('Toggle failed:', error);
    }
  };

  const handleRestart = async (job: TrainingJob) => {
    try {
      await trainingApi.restart(job.id);
      fetchJobs();
    } catch (error) {
      console.error('Restart failed:', error);
    }
  };

  const handleDelete = async (job: TrainingJob) => {
    if (window.confirm(t('common.confirmDelete') || `Delete ${job.name}?`)) {
      try {
        await trainingApi.delete(job.id);
        fetchJobs();
      } catch (error) {
        console.error('Delete failed:', error);
      }
    }
  };

  const handleSubmit = async (values: {
    name: string;
    description: string;
    model_id?: string;
    dataset_id: string;
    dataset_version: number;
    epochs?: number;
    imgsz?: number;
    batch?: number;
    workers?: number;
    base_model?: string;
    architecture?: string;
    initialization?: InitializationMode;
    starting_weights?: string;
    split_ratio?: number;
    task_type?: string;
    custom_image?: string;
    custom_image_manual?: string;
    custom_script_uri?: string;
    base_model_id?: string;
    accelerator?: string;
    gpu_count?: number;
    llm_learning_rate?: number;
    llm_grad_accum?: number;
    llm_cutoff?: number;
    llm_rank?: number;
    llm_alpha?: number;
    llm_dropout?: number;
    llm_template?: string;
    llm_quant?: number;
  }) => {
    try {
      const tt = values.task_type || 'detection';
      const isVision = tt === 'detection' || tt === 'segmentation';
      const isCls = tt === 'classification';
      const isLlm = tt === 'llm' || tt === 'vlm';
      const accelerator = isLlm ? (values.accelerator || 'nvidia') : (values.accelerator || 'cpu');
      const isGpuJob = accelerator !== 'cpu';
      const gpuCount = isGpuJob
        ? Math.max(1, values.gpu_count ?? 1)
        : 0;
      const initialization = values.initialization || 'pretrained';
      const architecture = (values.architecture || '').trim();
      const startingWeights = (values.starting_weights || '').trim();
      const config: TrainingConfig = {
        task_type: tt,
        // GPU vendor drives the per-vendor trainer image + resource key (ADR-012).
        // LLM/VLM has no CPU image; detection/segmentation may run on CPU.
        accelerator,
        gpu_count: gpuCount,
        epochs: values.epochs ?? (isLlm ? (tt === 'vlm' ? 5 : 3) : 100),
        batch: values.batch ?? (isLlm ? (tt === 'vlm' ? 2 : 4) : 16),
        // Keep the legacy base_model value for older consumers while emitting
        // explicit fields for the backend and trainer. Checkpoint mode may
        // leave architecture empty to request metadata/stock/default resolution.
        base_model: isLlm
          ? values.base_model ?? ''
          : initialization === 'checkpoint'
            ? startingWeights
            : architecture,
      } as TrainingConfig;
      if (!isLlm) {
        config.architecture = architecture;
        config.pretrained = initialization === 'pretrained';
        if (initialization === 'checkpoint') {
          config.base_weights_uri = startingWeights;
        }
      }
      const sourceModelId = (values.base_model_id || '').trim();
      if (sourceModelId) {
        config.base_model_id = sourceModelId;
      }
      if (!isLlm) {
        // Image-task hyperparameters (timm uses imgsz as the input resolution).
        config.imgsz = values.imgsz ?? (isCls ? 224 : 640);
        config.split_ratio = values.split_ratio ?? 0.8;
        if (typeof values.workers === 'number') {
          config.workers = values.workers;
        }
      } else {
        // LLM/VLM LoRA block — full contract of backend JobService._llm_args.
        const quant = values.llm_quant ?? 0;
        config.llm = {
          template: values.llm_template || 'qwen',
          cutoff_len: values.llm_cutoff ?? (tt === 'vlm' ? 512 : 1024),
          num_train_epochs: values.epochs ?? (tt === 'vlm' ? 5 : 3),
          learning_rate: values.llm_learning_rate ?? 2e-4,
          per_device_train_batch_size: values.batch ?? (tt === 'vlm' ? 2 : 4),
          gradient_accumulation_steps: values.llm_grad_accum ?? 4,
          lora_rank: values.llm_rank ?? 16,
          lora_alpha: values.llm_alpha ?? 32,
          lora_dropout: values.llm_dropout ?? 0.05,
          ...(quant > 0 ? { quantization_bit: quant } : {}),
        };
      }
      // Collapse flat aug_* form fields into config.augmentation for YOLO jobs.
      if (isVision) {
        // Include augmentation fields that are still inside an unopened panel.
        const all = form.getFieldsValue(true);
        const augmentation: Record<string, number | string | boolean> = {
          preset: (all.aug_preset as string) || 'medium',
        };
        AUG_FIELDS.forEach((f) => {
          const v = all[`aug_${f.key}`];
          if (typeof v === 'number' || typeof v === 'boolean'
              || (typeof v === 'string' && v !== '')) {
            augmentation[f.key] = v;
          }
        });
        config.augmentation = augmentation as TrainingConfig['augmentation'];
      }
      const chosenImage = (values.custom_image_manual || values.custom_image || '').trim();
      const payload = {
        name: values.name,
        description: values.description,
        model_id: values.model_id,
        dataset_id: values.dataset_id,
        dataset_version: values.dataset_version,
        config,
        custom_image: chosenImage || undefined,
        custom_script_uri: (values.custom_script_uri || '').trim() || undefined,
      };
      if (editingJob) {
        await trainingApi.update(editingJob.id, payload);
        setModalOpen(false);
        fetchJobs();
      } else {
        const created = await trainingApi.create(payload);
        // The job now exists independently of auto-start. Close the editor and
        // refresh it even when KFP rejects the start request.
        setModalOpen(false);
        try {
          await trainingApi.start(created.data.id);
          message.success(t('pages.training.created'));
        } finally {
          fetchJobs();
        }
      }
    } catch (error) {
      const detail = (error as any)?.response?.data?.detail;
      message.error(typeof detail === 'string' && detail ? detail : t('common.error'));
      console.error('Save failed:', error);
    }
  };

  return (
    <div>
      <PageHeader
        title={t('pages.training.title')}
        onCreate={handleCreate}
        onRefresh={fetchJobs}
      actions={[<HelpButton module="training" />]}
      />
      <Card>
        <FilterSearchBar
          filterPlaceholder={t('pages.training.filterStatus')}
          filterValue={statusFilter}
          onFilterChange={setStatusFilter}
          filterOptions={statusOptions}
          searchPlaceholder={t('pages.training.searchPlaceholder')}
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
        title={editingJob ? t('common.edit') : t('common.create')}
        open={modalOpen}
        onOk={() => form.validateFields().then(() => {
          // Collapsed panels may never have mounted their Form.Items. Validation
          // covers mounted fields; submission must still use the complete store.
          return handleSubmit(form.getFieldsValue(true));
        })}
        onCancel={() => setModalOpen(false)}

        width={800}
      >
        <Form form={form} layout="vertical">
          <Form.Item name="base_model_id" hidden><Input /></Form.Item>
          <Form.Item
            name="task_type"
            label={t('pages.training.taskType')}
            initialValue="detection"
          >
            <Select
              options={[
                { value: 'detection', label: t('pages.training.ttDetection') },
                { value: 'segmentation', label: t('pages.training.ttSegmentation') },
                { value: 'classification', label: t('pages.training.ttClassification') },
                { value: 'llm', label: t('pages.training.ttLlm') },
                { value: 'vlm', label: t('pages.training.ttVlm') },
              ]}
              onChange={(tt) => {
                // Reset hyperparameters/base model for the new task family so
                // LLM jobs never inherit YOLO imgsz/augmentation and vice versa.
                applyTaskTypeDefaults(form, tt);
                // The picked source model id belongs to the old task family.
                form.setFieldValue('base_model_id', undefined);
                // Drop a dataset selection that is incompatible with the new
                // task type (CV corpus vs SFT, and det/seg/cls narrowing).
                const curDsId = form.getFieldValue('dataset_id');
                if (curDsId) {
                  const curDs = datasets.find((d) => d.id === curDsId);
                  if (!curDs || !datasetMatchesTaskType(curDs, tt)) {
                    form.setFieldValue('dataset_id', undefined);
                    form.setFieldValue('dataset_version', undefined);
                  }
                }
              }}
            />
          </Form.Item>
          <Form.Item
            name="name"
            label={t('pages.training.jobName')}
            rules={[{ required: true, message: t('common.required') }]}
          >
            <Input placeholder={t('pages.training.jobName')} />
          </Form.Item>
          <Form.Item
            name="description"
            label={t('pages.datasets.description')}
          >
            <Input.TextArea rows={2} placeholder={t('pages.datasets.description')} />
          </Form.Item>
          <Form.Item noStyle shouldUpdate={(prev, cur) => prev.task_type !== cur.task_type}>
            {() => {
              return (
                <Form.Item
                  name="model_id"
                  label={t('pages.training.model')}
                  extra={t('pages.training.modelAssociateHint')}
                >
                  <Select
                    allowClear
                    placeholder={t('pages.training.model')}
                    options={models.map(m => ({ label: m.name, value: m.id }))}
                  />
                </Form.Item>
              );
            }}
          </Form.Item>
          <Form.Item
            name="dataset_id"
            label={t('pages.training.dataset')}
            rules={[{ required: true, message: t('common.required') }]}
          >
            <Select
              placeholder={t('pages.training.dataset')}
              notFoundContent={isLlmFormTask ? t('pages.training.noSftDatasets') : t('pages.training.noDatasetsForTaskType')}
              options={visibleDatasets.map(d => ({ label: `${d.name} (${d.format || 'coco'})`, value: d.id }))}
              onChange={() => form.setFieldValue('dataset_version', undefined)}
            />
          </Form.Item>
          <Form.Item
            name="dataset_version"
            label={t('pages.training.annotatedVersion')}
            rules={[{ required: true, message: t('common.required') }]}
            extra={t('pages.training.annotatedVersionHint')}
          >
            <Select
              placeholder={t('pages.training.annotatedVersion')}
              options={versions.map(v => ({ value: v.version, label: `v${v.version}${v.description ? ` · ${v.description}` : ''}` }))}
            />
          </Form.Item>
          <Form.Item noStyle shouldUpdate={(prev, cur) => prev.task_type !== cur.task_type || prev.accelerator !== cur.accelerator}>
            {() => {
              const tt = form.getFieldValue('task_type') || 'detection';
              const isLlmTask = tt === 'llm' || tt === 'vlm';
              const gpuVendors = [
                { value: 'nvidia', label: t('pages.training.accNvidia') },
                { value: 'amd', label: t('pages.training.accAmd') },
                { value: 'intel', label: t('pages.training.accIntel') },
              ];
              const selectedVendor = form.getFieldValue('accelerator') || (isLlmTask ? 'nvidia' : 'cpu');
              const isGpuVendor = selectedVendor !== 'cpu';
              // Bound by real per-node device-plugin capacity (incl. shared
              // vGPUs); undefined leaves the field unbounded when node reads
              // are unavailable.
              const maxGpuCount = gpuCapacity[selectedVendor]?.max_per_node;
              const capacityHint = maxGpuCount
                ? t('pages.training.gpuCountCapacity', { max: maxGpuCount })
                : t('pages.training.gpuCountHint');
              return (
                <>
                  <Form.Item
                    name="accelerator"
                    label={t('pages.training.accelerator')}
                    initialValue={isLlmTask ? 'nvidia' : 'cpu'}
                    extra={t('pages.training.acceleratorHint')}
                  >
                    <Select
                      options={isLlmTask ? gpuVendors : [
                        { value: 'cpu', label: t('pages.training.accCpu') },
                        ...gpuVendors,
                      ]}
                      onChange={(vendor) => {
                        const currentTaskType = form.getFieldValue('task_type') || 'detection';
                        const currentGpuCount = Number(form.getFieldValue('gpu_count') || 0);
                        form.setFieldsValue({
                          custom_image: compatibleTrainerImage(
                            form.getFieldValue('custom_image'), currentTaskType, vendor,
                          ),
                          gpu_count: vendor === 'cpu'
                            ? undefined
                            : vendor === 'intel'
                              ? 1
                              : currentGpuCount >= 1 ? currentGpuCount : 1,
                        });
                      }}
                    />
                  </Form.Item>
                  {isGpuVendor && (
                    <Form.Item
                      name="gpu_count"
                      label={t('pages.training.gpuCount')}
                      initialValue={1}
                      rules={[{ required: true, message: t('common.required') }]}
                      extra={capacityHint}
                    >
                      <InputNumber min={1} max={maxGpuCount} style={{ width: '100%' }} />
                    </Form.Item>
                  )}
                </>
              );
            }}
          </Form.Item>
          <Form.Item noStyle shouldUpdate={(prev, cur) =>
            prev.task_type !== cur.task_type
            || prev.custom_image !== cur.custom_image
            || prev.initialization !== cur.initialization
          }>
            {() => {
              const tt = form.getFieldValue('task_type') || 'detection';
              const isLlmTask = tt === 'llm' || tt === 'vlm';
              const isClsTask = tt === 'classification';
              const initialization: InitializationMode = form.getFieldValue('initialization') || 'pretrained';
              // Architectures announced by the selected (or any matching)
              // trainer image; locked entries stay visible but disabled.
              const selectedImage: string = form.getFieldValue('custom_image') || '';
              const announced = announcedArchOptions(capImages, tt, selectedImage);
              const architectureOptions = announced.map((a) => ({
                value: a.id,
                label: `${a.label || a.id} · ${a.id}${
                  a.locked ? ` · 🔒 ${t('pages.training.extendedTag')}` : ''
                }`,
                disabled: a.locked,
                title: a.locked
                  ? t('pages.training.extendedLockedHint', {
                      entitlements: (a.missing || []).join(', '),
                    })
                  : undefined,
              }));
              const taskContinueOptions = tt === 'classification'
                ? continueOptions.classification
                : tt === 'segmentation'
                  ? continueOptions.segmentation
                  : continueOptions.detection;
              return (
                <Collapse
                  ghost
                  style={{ marginLeft: -12 }}
                  items={[{
                    key: 'architecture-initialization',
                    label: t('pages.training.advancedSettings'),
                    children: isLlmTask ? (
                      <Form.Item
                        name="base_model"
                        label={t('pages.training.baseModelLlm')}
                        rules={[{ required: true, message: t('common.required') }]}
                      >
                        <AutoComplete
                          placeholder={t('pages.training.baseModelLlmPlaceholder')}
                          options={[
                            ...baseModels.map((m) => ({ value: m.uri, label: m.name })),
                            ...continueOptions.llm,
                          ]}
                          onChange={syncStartingWeights}
                          filterOption={(input, option) =>
                            String(option?.value ?? '').toLowerCase().includes(input.toLowerCase())
                          }
                          notFoundContent={t('pages.training.noBaseModels')}
                        />
                      </Form.Item>
                    ) : (
                      <>
                        <Form.Item
                          name="architecture"
                          label={t('pages.training.architecture')}
                          extra={t(initialization === 'checkpoint'
                            ? 'pages.training.architectureCheckpointHint'
                            : 'pages.training.architectureHint')}
                          rules={[{
                            required: initialization !== 'checkpoint',
                            message: t('common.required'),
                          }]}
                        >
                          {isClsTask ? (
                            <AutoComplete
                              allowClear
                              placeholder="resnet50"
                              options={announced.length
                                ? architectureOptions
                                : CLASSIFICATION_ARCHS.map((m) => ({ value: m }))}
                              filterOption={(input, option) =>
                                String(option?.value ?? '').toLowerCase().includes(input.toLowerCase())
                              }
                            />
                          ) : (
                            <Select
                              allowClear
                              showSearch
                              optionFilterProp="label"
                              options={architectureOptions}
                            />
                          )}
                        </Form.Item>
                        <Form.Item
                          name="initialization"
                          label={t('pages.training.initialization')}
                          initialValue="pretrained"
                        >
                          <Radio.Group
                            optionType="button"
                            buttonStyle="solid"
                            onChange={(event) => {
                              const mode = event.target.value as InitializationMode;
                              // Initialization does not change the explicitly
                              // chosen output model; only the source lineage is reset.
                              form.setFieldValue('base_model_id', '');
                              form.setFieldValue('starting_weights', undefined);
                              if (mode === 'checkpoint') {
                                form.setFieldValue('architecture', undefined);
                              } else if (!form.getFieldValue('architecture')) {
                                form.setFieldValue('architecture', defaultArchFor(tt));
                              }
                            }}
                            options={[
                              {
                                value: 'pretrained',
                                label: t('pages.training.initPretrained'),
                              },
                              {
                                value: 'scratch',
                                label: t('pages.training.initScratch'),
                              },
                              {
                                value: 'checkpoint',
                                label: t('pages.training.initCheckpoint'),
                              },
                            ]}
                          />
                        </Form.Item>
                        {initialization === 'checkpoint' && (
                          <Form.Item
                            name="starting_weights"
                            label={t('pages.training.startingWeights')}
                            extra={t('pages.training.startingWeightsHint')}
                            rules={[{ required: true, message: t('common.required') }]}
                          >
                            <Select
                              showSearch
                              optionFilterProp="label"
                              onChange={syncStartingWeights}
                              options={[
                                {
                                  label: t('pages.training.augGroupPool'),
                                  options: cvPoolEntries.map((e) => ({
                                    value: e.uri,
                                    label: e.name.replace(/\.(?:pt|pth)$/i, ''),
                                  })),
                                },
                                {
                                  label: t('pages.training.augGroupContinue'),
                                  options: taskContinueOptions,
                                },
                              ]}
                              notFoundContent={t('pages.training.noPoolEntries')}
                            />
                          </Form.Item>
                        )}
                      </>
                    ),
                  }]}
                />
              );
            }}
          </Form.Item>
          <Form.Item noStyle shouldUpdate={(prev, cur) => prev.task_type !== cur.task_type}>
            {() => {
              const tt = form.getFieldValue('task_type') || 'detection';
              const isLlmTask = tt === 'llm' || tt === 'vlm';
              const recommendedEpochs = tt === 'vlm' ? 5 : tt === 'llm' ? 3 : tt === 'classification' ? 50 : 100;
              const recommendedBatch = tt === 'vlm' ? 2 : tt === 'llm' ? 4 : tt === 'classification' ? 32 : 16;
              const recommendedImgsz = tt === 'classification' ? 224 : 640;
              const showImageFields = tt !== 'llm' && tt !== 'vlm';
              return (
                <>
                  <Row gutter={12}>
                    <Col span={12}>
                      <Form.Item
                        name="epochs"
                        label={`${t('pages.training.epochs')} (${t('pages.training.recommended')}: ${recommendedEpochs})`}
                      >
                        <InputNumber min={1} max={1000} style={{ width: '100%' }} />
                      </Form.Item>
                    </Col>
                    <Col span={12}>
                      <Form.Item
                        name="batch"
                        label={`${t('pages.training.batch')} (${t('pages.training.recommended')}: ${recommendedBatch})`}
                      >
                        <InputNumber min={1} max={256} style={{ width: '100%' }} />
                      </Form.Item>
                    </Col>
                  </Row>
                  {showImageFields && (
                    <Row gutter={12}>
                      <Col span={8}>
                        <Form.Item
                          name="imgsz"
                          label={`${t('pages.training.imgsz')} (${t('pages.training.recommended')}: ${recommendedImgsz})`}
                        >
                          <InputNumber min={64} max={1920} step={32} style={{ width: '100%' }} />
                        </Form.Item>
                      </Col>
                      <Col span={8}>
                        <Form.Item
                          name="split_ratio"
                          label={`${t('pages.training.splitRatio')} (${t('pages.training.recommended')}: 0.8)`}
                          extra={tt === 'classification' ? undefined : t('pages.training.splitRatioHint')}
                        >
                          <InputNumber min={0.5} max={0.95} step={0.05} style={{ width: '100%' }} />
                        </Form.Item>
                      </Col>
                      <Col span={8}>
                        <Form.Item
                          name="workers"
                          label={t('pages.training.workers')}
                          extra={t('pages.training.workersHint')}
                        >
                          <InputNumber min={0} max={64} step={1} precision={0} style={{ width: '100%' }} />
                        </Form.Item>
                      </Col>
                    </Row>
                  )}
                  {isLlmTask && (
                    <LlmHyperFields />
                  )}
                </>
              );
            }}
          </Form.Item>
          <Form.Item noStyle shouldUpdate={(prev, cur) => prev.task_type !== cur.task_type}>
            {() => {
              const tt = form.getFieldValue('task_type') || 'detection';
              if (tt !== 'detection' && tt !== 'segmentation') return null;
              return <AugmentationPanel form={form} />;
            }}
          </Form.Item>
          <Form.Item
            noStyle
            shouldUpdate={(prev, cur) =>
              prev.task_type !== cur.task_type || prev.accelerator !== cur.accelerator
            }
          >
            {() => {
              const tt = form.getFieldValue('task_type') || 'detection';
              const acc = form.getFieldValue('accelerator') || 'cpu';
              // Narrow the catalog to images that ANNOUNCE both this task and
              // the selected device (ADR-015); unannounced rows fall back to
              // their task_types list and the repo-name vendor suffix; an
              // empty catalog falls back to the built-in static list.
              const catalog: CapabilityImage[] = capImages.length
                ? capImages
                : trainImages.map((im) => ({
                    id: '', name: im.name, repository: '', tag: '',
                    full_ref: im.full_ref, is_builtin: false, devices: [],
                    announced: false, tasks: [], capabilities: [],
                    task_types: im.task_types,
                  } as unknown as CapabilityImage));
              const matching = announcedMatching(catalog, tt, acc);
              const lockHint = (missing: string[]) =>
                t('pages.training.extendedLockedHint', { entitlements: missing.join(', ') });
              return (
                <Form.Item
                  name="custom_image"
                  label={t('pages.training.trainerImage')}
                  extra={t('pages.training.trainerImageHint')}
                  initialValue=""
                >
                  <Select
                    allowClear
                    showSearch
                    placeholder={t('pages.training.trainerImagePlaceholder')}
                    onChange={(value: string) => {
                      // Prefer the image's own default architecture for the
                      // announced default over the static fallback table.
                      const announced = announcedDefaultArch(capImages, tt, value || '');
                      const initialization = form.getFieldValue('initialization') || 'pretrained';
                      if (announced && initialization !== 'checkpoint') {
                        form.setFieldValue('architecture', announced);
                      }
                    }}
                    options={[
                      { value: '', label: t('pages.training.trainerAuto') },
                      ...(matching.length
                        ? matching.map((im) => {
                            const lock = taskLock(im, tt);
                            return {
                              value: im.full_ref,
                              label: `${im.name} (${im.full_ref})${
                                lock.locked ? ` · 🔒 ${t('pages.training.extendedTag')}` : ''
                              }`,
                              disabled: lock.locked,
                              title: lock.locked ? lockHint(lock.missing) : undefined,
                            };
                          })
                        : TRAINER_IMAGE_FALLBACKS),
                    ]}
                    filterOption={(input, option) =>
                      String(option?.label ?? '').toLowerCase().includes(input.toLowerCase()) ||
                      String(option?.value ?? '').toLowerCase().includes(input.toLowerCase())
                    }
                    notFoundContent={null}
                  />
                </Form.Item>
              );
            }}
          </Form.Item>
          <Form.Item
            noStyle
            shouldUpdate={(prev, cur) => prev.custom_image !== cur.custom_image}
          >
            {({ getFieldValue }) =>
              getFieldValue('custom_image') === '' || !getFieldValue('custom_image') ? null : (
                <Form.Item
                  name="custom_image_manual"
                  label={t('pages.training.customImage')}
                  extra={t('pages.training.customImageHint')}
                >
                  <Input placeholder={t('pages.training.customImagePlaceholder')} onBlur={(e) => {
                    const v = e.target.value.trim();
                    if (v) form.setFieldValue('custom_image', v);
                  }} />
                </Form.Item>
              )
            }
          </Form.Item>
          <Form.Item
            noStyle
            shouldUpdate={(prev, cur) =>
              prev.custom_image !== cur.custom_image || prev.task_type !== cur.task_type
            }
          >
            {({ getFieldValue }) => {
              const ref: string = getFieldValue('custom_image') || '';
              if (!ref || !capImages.length) return null;
              const image = capImages.find((im) => im.full_ref === ref);
              const lock = taskLock(image, getFieldValue('task_type') || 'detection');
              if (!lock.locked) return null;
              return (
                <Alert
                  type="warning"
                  showIcon
                  style={{ marginBottom: 12 }}
                  message={t('pages.training.extendedLockedTitle')}
                  description={t('pages.training.extendedLockedHint', {
                    entitlements: lock.missing.join(', '),
                  })}
                />
              );
            }}
          </Form.Item>
          <Form.Item
            name="custom_script_uri"
            label={t('pages.training.customScript')}
            extra={t('pages.training.customScriptHint')}
            initialValue=""
          >
            <Input placeholder={t('pages.training.customScriptPlaceholder')} allowClear />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

// LLM / VLM LoRA finetuning hyperparameters. Flat form fields (llm_*) are
// collapsed into config.llm by the submit handler; defaults are applied on
// task-type switch and restored from the source job on retrain/clone.
const LlmHyperFields: React.FC = () => {
  const { t } = useTranslation();
  return (
    <>
      <Row gutter={12}>
        <Col span={12}>
          <Form.Item name="llm_learning_rate" label={t('pages.training.llmLearningRate')}>
            <InputNumber min={0} step={0.00001} style={{ width: '100%' }} />
          </Form.Item>
        </Col>
        <Col span={12}>
          <Form.Item name="llm_grad_accum" label={t('pages.training.llmGradAccum')}>
            <InputNumber min={1} max={128} style={{ width: '100%' }} />
          </Form.Item>
        </Col>
      </Row>
      <Row gutter={12}>
        <Col span={12}>
          <Form.Item name="llm_cutoff" label={t('pages.training.llmCutoff')}>
            <InputNumber min={64} max={32768} step={64} style={{ width: '100%' }} />
          </Form.Item>
        </Col>
        <Col span={12}>
          <Form.Item name="llm_template" label={t('pages.training.llmTemplate')}>
            <Select
              showSearch
              options={['qwen', 'llama3', 'chatml', 'deepseek', 'default'].map((v) => ({ value: v, label: v }))}
            />
          </Form.Item>
        </Col>
      </Row>
      <Row gutter={12}>
        <Col span={8}>
          <Form.Item name="llm_rank" label={t('pages.training.llmRank')}>
            <InputNumber min={1} max={256} style={{ width: '100%' }} />
          </Form.Item>
        </Col>
        <Col span={8}>
          <Form.Item name="llm_alpha" label={t('pages.training.llmAlpha')}>
            <InputNumber min={1} max={512} style={{ width: '100%' }} />
          </Form.Item>
        </Col>
        <Col span={8}>
          <Form.Item name="llm_dropout" label={t('pages.training.llmDropout')}>
            <InputNumber min={0} max={0.9} step={0.01} style={{ width: '100%' }} />
          </Form.Item>
        </Col>
      </Row>
      <Form.Item name="llm_quant" label={t('pages.training.llmQuantization')}>
        <Select
          options={[
            { value: 0, label: t('pages.training.llmQuantNone') },
            { value: 4, label: '4-bit QLoRA (bitsandbytes)' },
            { value: 8, label: '8-bit LoRA (bitsandbytes)' },
          ]}
        />
      </Form.Item>
    </>
  );
};

// Augmentation form panel (YOLO detection/segmentation only). Flat fields are
// named aug_<key>; the submit handler collapses them into config.augmentation.
// Compact numeric label for slider endpoint marks (drops trailing zeros).
const formatSliderNum = (n: number): string =>
  Number.isInteger(n) ? String(n) : String(parseFloat(n.toFixed(4)));

// Small, unobtrusive endpoint labels rendered below augmentation sliders.
const sliderMarkStyle = { fontSize: 11, color: '#8c8c8c' } as const;

const AugmentationPanel: React.FC<{ form: FormInstance }> = ({ form }) => {
  const { t } = useTranslation();
  const [showAdvanced, setShowAdvanced] = useState(false);

  const applyPreset = (preset: 'off' | 'light' | 'medium' | 'strong') => {
    const vals = AUG_PRESETS[preset];
    const patch: Record<string, unknown> = { aug_preset: preset };
    AUG_FIELDS.forEach((f) => { patch[`aug_${f.key}`] = vals[f.key]; });
    form.setFieldsValue(patch);
    // Reveal the augmentation sliders so the parameter change is visible.
    setShowAdvanced(true);
  };

  const fieldLabel = (key: AugKey) => t(`pages.training.aug_${key}`);

  const renderControl = (f: AugFieldDef) => {
    const label = <span style={{ fontSize: 13 }}>{fieldLabel(f.key)}</span>;
    if (f.widget === 'switch') {
      return (
        <Form.Item name={`aug_${f.key}`} key={f.key} label={label} valuePropName="checked">
          <Switch size="small" />
        </Form.Item>
      );
    }
    if (f.widget === 'select') {
      return (
        <Form.Item name={`aug_${f.key}`} key={f.key} label={label}>
          <Select size="small" options={(f.options || []).map((o) => ({ value: o, label: o }))} />
        </Form.Item>
      );
    }
    if (f.widget === 'number') {
      return (
        <Form.Item name={`aug_${f.key}`} key={f.key} label={label}>
          <InputNumber size="small" min={f.min} max={f.max} step={f.step} style={{ width: '100%' }} />
        </Form.Item>
      );
    }
    const sMin = f.min ?? 0;
    const sMax = f.max ?? 1;
    return (
      <Form.Item name={`aug_${f.key}`} key={f.key} label={label}>
        <Slider
          min={sMin}
          max={sMax}
          step={f.step}
          tooltip={{ open: undefined }}
          marks={{
            [sMin]: { label: formatSliderNum(sMin), style: sliderMarkStyle },
            [sMax]: { label: formatSliderNum(sMax), style: sliderMarkStyle },
          }}
        />
      </Form.Item>
    );
  };

  return (
    <Collapse
      ghost
      items={[{
        key: 'aug',
        label: t('pages.training.augTitle'),
        children: (
          <>
            <Form.Item
              name="aug_preset"
              label={t('pages.training.augPreset')}
              extra={t('pages.training.augPresetHint')}
            >
              <Radio.Group onChange={(e) => applyPreset(e.target.value)}>
                <Radio.Button value="off">{t('pages.training.augOff')}</Radio.Button>
                <Radio.Button value="light">{t('pages.training.augLight')}</Radio.Button>
                <Radio.Button value="medium">{t('pages.training.augMedium')}</Radio.Button>
                <Radio.Button value="strong">{t('pages.training.augStrong')}</Radio.Button>
              </Radio.Group>
            </Form.Item>
            {/* Production controls are always shown: early stop, optimizer/LR,
                multi-scale training, mixed precision and reproducibility. */}
            <Divider style={{ margin: '8px 0' }} orientation="left" orientationMargin={0}>
              <span style={{ fontSize: 13 }}>{t('pages.training.productionTitle')}</span>
            </Divider>
            <Row gutter={16}>
              {PRODUCTION_FIELDS.map((f) => (
                <Col span={8} key={f.key}>{renderControl(f)}</Col>
              ))}
            </Row>
            <Button type="link" size="small" style={{ padding: 0 }} onClick={() => setShowAdvanced((s) => !s)}>
              {showAdvanced ? t('pages.training.augHideAdvanced') : t('pages.training.augShowAdvanced')}
            </Button>
            {showAdvanced && (
              <Row gutter={16}>
                {SLIDER_FIELDS.map((f) => (
                  <Col span={12} key={f.key}>
                    {renderControl(f)}
                  </Col>
                ))}
              </Row>
            )}
          </>
        ),
      }]}
    />
  );
};

// Training Detail Page
export const TrainingDetail: React.FC = () => {
  const { t } = useTranslation();
  const { id: jobId } = useParams<{ id: string }>();
  const [job, setJob] = useState<TrainingJob | null>(null);
  const [loading, setLoading] = useState(true);
  const [metrics, setMetrics] = useState<TrainingMetrics | null>(null);
  const [logs, setLogs] = useState('');
  // Cluster-side objects (TrainJob CR / JobSet pods) mapped from this NF job.
  const [runtime, setRuntime] = useState<TrainingRuntime | null>(null);
  const [activeTab, setActiveTab] = useState(() => new URLSearchParams(window.location.search).get('tab') || 'overview');
  const logsRef = React.useRef<HTMLPreElement>(null);
  // URL loaded into the embedded Netron viewer modal (null = closed).
  const [netronUrl, setNetronUrl] = useState<string | null>(null);
  // Resolved model/dataset names for the enriched overview (best-effort).
  const [model, setModel] = useState<Model | null>(null);
  const [dataset, setDataset] = useState<Dataset | null>(null);

  // Format a duration in seconds as a compact "Xm Ys" string.
  const fmtDuration = (s: unknown): string => {
    const n = Number(s);
    if (!isFinite(n) || n <= 0) return '-';
    const m = Math.floor(n / 60);
    const sec = Math.round(n % 60);
    return m > 0 ? `${m}m ${sec}s` : `${sec}s`;
  };

  const handleDeployArtifact = useCallback(async (onnxUri: string) => {
    // Prefer the model this run actually produced; fall back to the source
    // model id for older jobs that predate auto-registration.
    const targetId = job?.output_model_id || job?.model_id;
    if (!job || !targetId) return;
    try {
      // Generate a version identifier for this deployment
      const version = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
      
      // Deploy to KServe with the specific ONNX URI
      await modelApi.deploy(targetId, version, 'kserve', {
        runtime_type: 'kserve-tritonserver',
        gpu: 'cpu',
        model_uri: onnxUri,  // Pass the ONNX URI directly
      });
      message.success(t('pages.training.deployStarted'));
    } catch (e) {
      console.error(e);
      message.error(t('pages.training.deployFailed'));
    }
  }, [job, t]);

  const fetchJob = useCallback(async () => {
    try {
      if (!jobId) return;
      const [jobRes, metricsRes, logsRes, runtimeRes] = await Promise.all([
        trainingApi.get(jobId),
        trainingApi.getMetrics(jobId),
        trainingApi.getLogs(jobId, { tail: 2000 }).catch(() => ({ data: { logs: '' } })),
        trainingApi.getRuntime(jobId).catch(() => ({ data: null })),
      ]);
      setJob(jobRes.data);
      setMetrics(metricsRes.data ?? null);
      setLogs(logsRes.data?.logs || '');
      setRuntime(runtimeRes.data ?? null);
      // Resolve model and dataset names for the overview (best-effort,
      // non-blocking — the IDs are shown until the names arrive).
      const jm = jobRes.data;
      // Prefer the produced model for the overview; fall back to the source.
      const modelRef = jm.output_model_id || jm.model_id;
      if (modelRef) modelApi.get(modelRef).then((r) => setModel(r.data)).catch(() => {});
      if (jm.dataset_id) datasetApi.get(jm.dataset_id).then((r) => setDataset(r.data)).catch(() => {});
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

  // Tail the logs like a console: keep pinned to the bottom on new lines
  // unless the user scrolled up to inspect earlier output.
  useEffect(() => {
    const el = logsRef.current;
    if (!el || activeTab !== 'logs') return;
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
    if (nearBottom) el.scrollTop = el.scrollHeight;
  }, [logs, activeTab]);

  const handleTabChange = (key: string) => {
    setActiveTab(key);
    const url = new URL(window.location.href);
    if (key === 'overview') url.searchParams.delete('tab');
    else url.searchParams.set('tab', key);
    window.history.replaceState(null, '', url.toString());
    if (key === 'logs') {
      // Jump to the newest lines immediately when opening the console.
      setTimeout(() => {
        const el = logsRef.current;
        if (el) el.scrollTop = el.scrollHeight;
      }, 50);
    }
  };

  // Split training metrics into downloadable model artifacts (s3:// URIs)
  // and the remaining numeric/structured metric fields.
  const { artifacts, otherMetrics } = useMemo(() => {
    if (!metrics) return { artifacts: [] as [string, string][], otherMetrics: [] as [string, unknown][] };
    const arts: [string, string][] = [];
    const rest: [string, unknown][] = [];
    for (const [k, v] of Object.entries(metrics)) {
      // Only treat s3:// URIs pointing at a file (no trailing slash) as
      // downloadable artifacts. Directory URIs (e.g. tensorboard_uri) fall
      // through to the generic metric rendering below.
      if (typeof v === 'string' && v.startsWith('s3://') && !v.endsWith('/')) {
        arts.push([k, v]);
      } else {
        rest.push([k, v]);
      }
    }
    return { artifacts: arts, otherMetrics: rest };
  }, [metrics]);

  if (loading) return <Spin size="large" style={{ display: 'block', margin: '100px auto' }} />;
  if (!job) return <Alert message={t('common.notFound')} type="error" />;

  const detailTaskType = job.config?.task_type || 'detection';
  const isLlmDetail = detailTaskType === 'llm' || detailTaskType === 'vlm';
  const detailSelection = trainingSelectionFromConfig(
    (job.config || {}) as unknown as Record<string, unknown>,
    detailTaskType,
  );
  const initializationLabels: Record<InitializationMode, string> = {
    pretrained: t('pages.training.initPretrained'),
    scratch: t('pages.training.initScratch'),
    checkpoint: t('pages.training.initCheckpoint'),
  };

  return (
    <div>
      <PageHeader
        title={job.name}
        subtitle={`${t('pages.training.status')}: ${t(statusLabelKey(job.status))} • ${t('common.progress')}: ${job.progress}%`}
        onRefresh={fetchJob}
      />
      
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col span={6}>
          <Statistic title={t('pages.training.status')} value={t(statusLabelKey(job.status))} />
        </Col>
        <Col span={6}>
          <Statistic title={t('common.progress')} value={`${job.progress}%`} />
        </Col>
        <Col span={6}>
          <Statistic title={t('pages.training.currentEpoch')} value={`${job.current_epoch}/${job.total_epochs}`} />
        </Col>
        <Col span={6}>
          {(job.status === 'running' || job.status === 'pending') && typeof (metrics as Record<string, unknown> | null)?.loss === 'number' ? (
            <Statistic
              title={t('pages.training.curveTrainLoss')}
              value={Number((metrics as Record<string, unknown>).loss).toFixed(4)}
              valueStyle={{ color: '#1890ff' }}
            />
          ) : (
            <Statistic title={t('common.createdAt')} value={formatDateTime(job.created_at)} />
          )}
        </Col>
      </Row>

      <Tabs activeKey={activeTab} onChange={handleTabChange} items={[
        { key: 'overview', label: t('common.overview') },
        { key: 'curves', label: t('pages.training.curves') },
        { key: 'metrics', label: t('pages.training.metrics') },
        { key: 'artifacts', label: t('pages.training.artifacts') },
        { key: 'logs', label: t('common.logs') },
        { key: 'config', label: t('pages.training.config') },
      ]} />

      <Card style={{ marginTop: 16 }}>
        {activeTab === 'overview' && (
          <div>
            {/* Key evaluation metrics (completed jobs with metrics only). */}
            {metrics && (() => {
              const m = metrics as Record<string, unknown>;
              const kms: { label: string; value: number; color: string; isPercent?: boolean }[] = [];
              const mAP95 = m['mAP50-95'];
              if (typeof mAP95 === 'number') kms.push({ label: 'mAP50-95', value: mAP95 * 100, color: '#52c41a' });
              const mAP50 = m.mAP50;
              if (typeof mAP50 === 'number') kms.push({ label: 'mAP50', value: mAP50 * 100, color: '#1890ff' });
              const prec = m.precision;
              if (typeof prec === 'number') kms.push({ label: t('pages.training.curvePrecision'), value: prec * 100, color: '#722ed1' });
              const rec = m.recall;
              if (typeof rec === 'number') kms.push({ label: t('pages.training.curveRecall'), value: rec * 100, color: '#eb2f96' });
              // VLM/LLM jobs report a single training-loss value rather
              // than mAP (progress/epochs live in the Progress column).
              const fl = m.final_loss;
              if (typeof fl === 'number') kms.push({ label: 'Final Loss', value: fl, color: '#1677ff', isPercent: false });
              if (!kms.length) return null;
              return (
                <>
                  <h4>{t('pages.training.keyMetrics')}</h4>
                  <Row gutter={[12, 12]} style={{ marginBottom: 16 }}>
                    {kms.map((k) => (
                      <Col span={6} key={k.label}>
                        <Card size="small">
                          <Statistic
                            title={k.label}
                            value={k.value}
                            precision={'isPercent' in k && !k.isPercent ? 1 : 2}
                            suffix={k.isPercent !== false ? '%' : undefined}
                            valueStyle={{ color: k.color }}
                          />
                        </Card>
                      </Col>
                    ))}
                  </Row>
                </>
              );
            })()}

            {/* Training configuration summary. */}
            <Card size="small" title={t('pages.training.config')} style={{ marginBottom: 16 }}>
              <Descriptions column={2} size="small" bordered>
<Descriptions.Item label={t('pages.training.model')}>
  {/* Produced model (with version) when present; otherwise the source model. */}
  {job.output_model_id
    ? (model?.name || job.output_model_name || job.output_model_id.slice(0, 12)) +
      (job.output_model_version ? ` (v${job.output_model_version})` : '')
    : model?.name || job.model_id?.slice(0, 12) || '-'}
</Descriptions.Item>
                <Descriptions.Item label={t('pages.training.dataset')}>
                  {dataset ? `${dataset.name} (v${job.dataset_version})` : job.dataset_id?.slice(0, 12) || '-'}
                </Descriptions.Item>
                <Descriptions.Item label={t(isLlmDetail
                  ? 'pages.training.baseModelLlm'
                  : 'pages.training.architecture')}
                >
                  {isLlmDetail
                    ? job.config?.base_model || '-'
                    : detailSelection.architecture || t('pages.training.architectureAuto')}
                </Descriptions.Item>
                {!isLlmDetail && (
                  <Descriptions.Item label={t('pages.training.initialization')}>
                    {initializationLabels[detailSelection.initialization]}
                  </Descriptions.Item>
                )}
                {!isLlmDetail && detailSelection.initialization === 'checkpoint' && (
                  <Descriptions.Item label={t('pages.training.startingWeights')}>
                    {detailSelection.startingWeights || '-'}
                  </Descriptions.Item>
                )}
                <Descriptions.Item label={t('pages.training.accelerator')}>
                  {job.config?.accelerator || '-'}
                </Descriptions.Item>
                <Descriptions.Item label={t('pages.training.gpuCount')}>
                  {job.config?.gpu_count ?? 0}
                </Descriptions.Item>
                <Descriptions.Item label={t('pages.training.trainerImage')}>
                  {(() => {
                    const img = job.config?.trainer_image || job.trainer_image || '';
                    return img ? <Tooltip title={img}><span>{img.split('/').pop() || img}</span></Tooltip> : '-';
                  })()}
                </Descriptions.Item>
                <Descriptions.Item label={t('pages.training.epochs')}>
                  {job.total_epochs}
                </Descriptions.Item>
                {job.config?.llm ? (
                  <Descriptions.Item label={t('pages.training.llmHyper')}>
                    {(() => {
                      const l = job.config!.llm!;
                      const parts = [
                        `lr=${l.learning_rate ?? '-'}`,
                        `bs=${l.per_device_train_batch_size ?? job.config?.batch ?? '-'}`,
                        `accum=${l.gradient_accumulation_steps ?? '-'}`,
                        `cutoff=${l.cutoff_len ?? '-'}`,
                        `r=${l.lora_rank ?? '-'}/${l.lora_alpha ?? '-'}`,
                        `p=${l.lora_dropout ?? '-'}`,
                      ];
                      if (l.quantization_bit) parts.push(`${l.quantization_bit}bit`);
                      return <span style={{ fontSize: 12 }}>{parts.join(' · ')}</span>;
                    })()}
                  </Descriptions.Item>
                ) : (
                  <>
                    <Descriptions.Item label={t('pages.training.imgsz')}>
                      {job.config?.imgsz || '-'}
                    </Descriptions.Item>
                    <Descriptions.Item label={t('pages.training.splitRatio')}>
                      {job.config?.split_ratio || '-'}
                    </Descriptions.Item>
                  </>
                )}
                <Descriptions.Item label={t('pages.training.batch')}>
                  {job.config?.batch || '-'}
                </Descriptions.Item>
                {/* Vision training parameters (preset + production controls).
                    Legacy jobs predate the block and show "defaults". */}
                {!job.config?.llm && (
                  <Descriptions.Item label={t('pages.training.parameters')} span={2}>
                    {(() => {
                      const aug = (job.config!.augmentation
                        || {}) as Record<string, AugValue>;
                      const parts: Array<[string, AugValue]> = [
                        ['patience', aug.patience],
                        ['optimizer', aug.optimizer],
                        ['lr0', aug.lr0],
                        ['lrf', aug.lrf],
                        ['cos_lr', aug.cos_lr],
                        ['warmup', aug.warmup_epochs],
                        ['multi_scale', aug.multi_scale],
                        ['dropout', aug.dropout],
                        ['amp', aug.amp],
                        ['deterministic', aug.deterministic],
                        ['seed', aug.seed],
                        ['save_period', aug.save_period],
                      ];
                      const tags = parts
                        .filter(([, v]) => v !== undefined && v !== null && v !== '')
                        .map(([k, v]) => (
                          <Tag key={k}>
                            {k === 'warmup' ? 'warmup_ep' : k}=
                            {typeof v === 'boolean' ? (v ? 'on' : 'off') : String(v)}
                          </Tag>
                        ));
                      return (
                        <Space size={[4, 8]} wrap>
                          {typeof aug.preset === 'string' ? (
                            <Tag color="blue">{aug.preset}</Tag>
                          ) : tags.length === 0 ? (
                            <Tag>{t('pages.training.parametersDefaults')}</Tag>
                          ) : null}
                          {tags}
                        </Space>
                      );
                    })()}
                  </Descriptions.Item>
                )}
                <Descriptions.Item label={t('pages.training.trainingTime')}>
                  {fmtDuration(trainingDurationSeconds(
                    metrics as Record<string, unknown> | null,
                    runtime,
                  ))}
                </Descriptions.Item>
                <Descriptions.Item label={t('pages.training.startedAt')}>
                  {job.started_at ? formatDateTime(job.started_at) : '-'}
                </Descriptions.Item>
                <Descriptions.Item label={t('common.createdAt')}>
                  {formatDateTime(job.created_at)}
                </Descriptions.Item>
                <Descriptions.Item label={t('pages.training.kfpRunId')}>
                  {job.kfp_run_id ? <code style={{ fontSize: 12 }}>{job.kfp_run_id.slice(0, 8)}</code> : '-'}
                </Descriptions.Item>
              </Descriptions>
            </Card>

            {/* Cluster workload mapping: KFP run -> TrainJob CR -> JobSet pods.
                Lets users correlate an NF job with the exact object names seen
                in kubectl; these are the workloads removed on delete/stop. */}
            <Card size="small" style={{ marginTop: 16 }} title={t('pages.training.clusterResources')}>
              {!runtime || (runtime.trainjobs.length === 0 && runtime.pods.length === 0) ? (
                <Alert type="info" showIcon message={t('pages.training.noClusterResources')} />
              ) : (
                <Descriptions column={1} size="small" bordered>
                  {runtime.kfp_run_id && (
                    <Descriptions.Item label={t('pages.training.kfpRunId')}>
                      <code style={{ fontSize: 12 }}>{runtime.kfp_run_id}</code>
                    </Descriptions.Item>
                  )}
                  {runtime.trainjobs.length > 0 && (
                    <Descriptions.Item label={t('pages.training.trainJobCr')}>
                      <Space size={[4, 8]} wrap>
                        {runtime.trainjobs.map((tj) => (
                          <Tag key={tj.name}>
                            <code style={{ fontSize: 12 }}>{tj.name}</code>
                            {tj.state ? ` · ${tj.state}` : ''}
                          </Tag>
                        ))}
                      </Space>
                    </Descriptions.Item>
                  )}
                  {runtime.pods.length > 0 && (
                    <Descriptions.Item label={t('pages.training.trainerPods')}>
                      <Space direction="vertical" size={4}>
                        {runtime.pods.map((p) => (
                          <Space key={p.name} size={8}>
                            <code style={{ fontSize: 12 }}>{p.name}</code>
                            <Tag color={p.phase === 'Running' ? 'green' : p.phase === 'Failed' ? 'red' : p.phase === 'Pending' ? 'orange' : 'default'}>
                              {p.phase}
                            </Tag>
                            {p.node && <Tag>{p.node}</Tag>}
                          </Space>
                        ))}
                      </Space>
                    </Descriptions.Item>
                  )}
                </Descriptions>
              )}
            </Card>

            {/* Error message (if any). */}
            {job.error_message && (
              <Alert message={t('common.error')} description={job.error_message} type="error" />
            )}
          </div>
        )}
        {activeTab === 'curves' && (
          <TrainingCurves jobId={job.id} />
        )}
        {activeTab === 'metrics' && metrics && (
          <div>
            <h4>{t('pages.training.metrics')}</h4>
            {otherMetrics.length > 0 ? (
              <Descriptions column={2} size="small" bordered>
                {otherMetrics.map(([k, v]) => (
                  <Descriptions.Item key={k} label={k}>
                    {typeof v === 'number' ? v.toFixed(4) : typeof v === 'object' ? JSON.stringify(v) : String(v)}
                  </Descriptions.Item>
                ))}
              </Descriptions>
            ) : (
              <Alert message={t('pages.training.noArtifacts')} type="info" />
            )}
          </div>
        )}
        {activeTab === 'artifacts' && (
          <div>
            <h4>{t('pages.training.artifacts')}</h4>
            {artifacts.length > 0 ? (
              <Row gutter={[12, 12]}>
                {artifacts.map(([k, uri]) => {
                  // Build a same-origin HTTP proxy URL for the s3:// object so the
                  // browser can download/visualize it without hitting SeaweedFS directly.
                  const proxyUrl = `/api/v1/s3-object?uri=${encodeURIComponent(uri)}`;
                  const filename = uri.split('/').pop() || 'model.bin';
                  // Netron supports these formats for structure visualization.
                  const netronSupported = /\.(onnx|pt|pth|tflite|pb|keras|h5|gguf|caffemodel|mlmodel)$/.test(filename.toLowerCase());
                  const isOnnx = filename.toLowerCase().endsWith('.onnx');
                  return (
                    <Col span={12} key={k}>
                      <Card
                        size="small"
                        title={k}
                        extra={
                          <Space>
                            <a href={proxyUrl} download={filename}>
                              <Button size="small" type="primary" ghost icon={<DownloadOutlined />}>
                                {t('common.download')}
                              </Button>
                            </a>
                            {netronSupported && (
                              <Tooltip title={t('pages.training.netronHint')}>
                                <Button
                                  size="small"
                                  icon={<EyeOutlined />}
                                  onClick={() => setNetronUrl(`/netron/index.html?url=${encodeURIComponent(proxyUrl)}`)}
                                >
                                  {t('pages.training.visualize')}
                                </Button>
                              </Tooltip>
                            )}
                            {isOnnx && job.model_id && (
                              <Popconfirm
                                title={t('pages.training.deployConfirm', { filename })}
                                onConfirm={() => handleDeployArtifact(uri)}
                                okText={t('common.confirm')}
                                cancelText={t('common.cancel')}
                              >
                                <Button size="small" icon={<RocketOutlined />}>
                                  {t('pages.training.deploy')}
                                </Button>
                              </Popconfirm>
                            )}
                          </Space>
                        }
                      >
                        <div style={{ fontFamily: 'monospace', fontSize: 12, wordBreak: 'break-all', color: '#888' }}>
                          {filename}
                        </div>
                      </Card>
                    </Col>
                  );
                })}
              </Row>
            ) : (
              <Alert message={t('pages.training.noArtifacts')} type="info" />
            )}
          </div>
        )}
        {activeTab === 'logs' && (
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
              <h4 style={{ margin: 0 }}>{t('common.logs')}</h4>
              <Space>
                {job.status === 'running' || job.status === 'pending' ? (
                  <span style={{ color: '#52c41a', fontSize: 12 }}>● {t('pages.training.logsLive')}</span>
                ) : null}
                <Button
                  size="small"
                  icon={<DownloadOutlined />}
                  disabled={!logs}
                  onClick={() => {
                    const blob = new Blob([logs || ''], { type: 'text/plain' });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `${job.name || jobId}-logs.txt`;
                    a.click();
                    URL.revokeObjectURL(url);
                  }}
                >
                  {t('pages.training.downloadLogs')}
                </Button>
              </Space>
            </div>
            <pre
              ref={logsRef}
              style={{ background: '#1e1e1e', color: '#d4d4d4', padding: 16, borderRadius: 4, maxHeight: 480, overflow: 'auto', fontSize: 12, lineHeight: 1.5, whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}
            >
              {logs || t('pages.training.noLogs')}
            </pre>
          </div>
        )}
        {activeTab === 'config' && (
          <div>
            <h4>{t('pages.training.config')}</h4>
            <pre style={{ background: '#f5f5f5', padding: 16, borderRadius: 4, maxHeight: 300, overflow: 'auto', fontSize: 12 }}>
              {JSON.stringify(job.config, null, 2)}
            </pre>
            {job.params && (
              <>
                <h4>{t('pages.training.params')}</h4>
                <pre style={{ background: '#f5f5f5', padding: 16, borderRadius: 4, maxHeight: 300, overflow: 'auto', fontSize: 12 }}>
                  {JSON.stringify(job.params, null, 2)}
                </pre>
              </>
            )}
          </div>
        )}
      </Card>

      {/* Embedded Netron model viewer (self-hosted under /netron/). */}
      <Modal
        title={t('pages.training.netronViewer')}
        open={!!netronUrl}
        onCancel={() => setNetronUrl(null)}
        footer={null}
        destroyOnClose
        width="90%"
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
    </div>
  );
};

export default Training;
