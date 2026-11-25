// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

import React, { useEffect, useRef, useState, useCallback } from 'react';
import {
  Card, Button, Space, Spin, Alert, Row, Col, List, Tag, Modal, Input, Empty,
  Tooltip, Badge, message, Select, Checkbox, Progress, InputNumber,
} from 'antd';
import {
  SaveOutlined, ExportOutlined, RocketOutlined, ThunderboltOutlined,
  LeftOutlined, RightOutlined, RobotOutlined, LinkOutlined, CheckCircleOutlined,
  VideoCameraOutlined, ReloadOutlined, AimOutlined, TagsOutlined,
  PauseOutlined, CaretRightOutlined, StopOutlined, CloseOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import { useNavigate, useParams } from 'react-router-dom';
import { annotationApi, samApi, modelApi } from '@api/endpoints';
import { PageHeader } from '@components/PageHeader';
import { statusLabelKey } from '@components/StatusBadge';
import { CVATAnnotator } from '@components/CVATAnnotator';
import type { AnnotationTask, LSTaskView, CVATTask, Model, ModelVersion } from '@nftypes/api';

declare global {
  interface Window {
    LabelStudio?: any;
    Htx?: any;
    APP_SETTINGS?: any;
  }
}

const LS_JS = '/label-studio/ls-main.js';
const LS_CSS = '/label-studio/ls-main.css';

// Load LS SDK (UMD build exposes window.LabelStudio)
let sdkLoadingPromise: Promise<void> | null = null;
function loadLS() {
  if (window.LabelStudio) return Promise.resolve();
  if (sdkLoadingPromise) return sdkLoadingPromise;

  sdkLoadingPromise = new Promise((resolve, reject) => {
    if (!document.querySelector(`link[href="${LS_CSS}"]`)) {
      const link = document.createElement('link');
      link.rel = 'stylesheet';
      link.href = LS_CSS;
      document.head.appendChild(link);
    }
    if (document.querySelector(`script[src="${LS_JS}"]`)) {
      resolve();
      return;
    }
    const script = document.createElement('script');
    script.src = LS_JS;
    script.async = true;
    script.onload = () => resolve();
    script.onerror = () => reject(new Error('Failed to load Label Studio SDK'));
    document.head.appendChild(script);
  });
  return sdkLoadingPromise;
}

// Build label config XML from project schema
function buildConfigFromSchema(schema?: Record<string, unknown>): string | null {
  const labels = Array.isArray(schema?.labels) ? (schema.labels as unknown[]) : null;
  const rectType = typeof schema?.type === 'string' ? schema.type : 'rectanglelabels';
  if (!labels || labels.length === 0) return null;
  const tag = (l: unknown) => `<Label value="${String(l)}"/>`;
  const labelTags = labels.map(tag).join('');
  if (rectType === 'polygonlabels') {
    return `<View><Image name="image" value="$image"/><PolygonLabels name="label" toName="image">${labelTags}</PolygonLabels></View>`;
  }
  return `<View><Image name="image" value="$image"/><RectangleLabels name="label" toName="image">${labelTags}</RectangleLabels></View>`;
}

const DEFAULT_CONFIG = `<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="person"/><Label value="bicycle"/><Label value="car"/><Label value="motorcycle"/><Label value="airplane"/><Label value="bus"/><Label value="train"/><Label value="truck"/>
  </RectangleLabels>
</View>`;

export const AnnotationStudio: React.FC = () => {
  const { t } = useTranslation();
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();

  const [task, setTask] = useState<AnnotationTask | null>(null);
  const [lsTasks, setLSTasks] = useState<LSTaskView[]>([]);
  const [cvatTasks, setCVATTasks] = useState<CVATTask[]>([]);
  const [selectedCVATTasks, setSelectedCVATTasks] = useState<Set<number>>(new Set());
  const [totalTasks, setTotalTasks] = useState(0);
  const [page, setPage] = useState(1);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [loading, setLoading] = useState(true);
  const [sdkReady, setSdkReady] = useState(false);
  const [sdkError, setSdkError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  // LS-image SAM assist: overlay div captures clicks over the LS canvas
  const [lsSamMode, setLsSamMode] = useState(false);
  const [lsSamBusy, setLsSamBusy] = useState(false);
  const lsSamHostRef = useRef<HTMLDivElement | null>(null);
  const [exportOpen, setExportOpen] = useState(false);
  const [exportName, setExportName] = useState('');
  const [exportFormat, setExportFormat] = useState('coco');
  const [exporting, setExporting] = useState(false);
  const [preannotating, setPreannotating] = useState(false);
  const [mlBackends, setMLBackends] = useState<any[]>([]);
  const [selectedMLBackend, setSelectedMLBackend] = useState<string>('');
  // Pre-annotation model (CNN) chosen for this project. The shared ML backend
  // is model-agnostic; the user picks a registry model + version, remembered
  // per project (preannotate_model) and injected as s3 weights at inference.
  const [preModels, setPreModels] = useState<Model[]>([]);
  const [preVersions, setPreVersions] = useState<ModelVersion[]>([]);
  const [preModelName, setPreModelName] = useState<string | undefined>(undefined);
  const [preVersion, setPreVersion] = useState<string | undefined>(undefined);
  const [frameStride, setFrameStride] = useState(5);
  const [scoreThreshold, setScoreThreshold] = useState(0.25);
  const [labelConfig, setLabelConfig] = useState(DEFAULT_CONFIG);
  const [selectedIds, setSelectedIds] = useState<number[]>([]);
  // Inline (non-modal) async-task progress bar + pause/resume/stop controls.
  const [asyncTaskId, setAsyncTaskId] = useState<string | null>(null);
  const [asyncTitle, setAsyncTitle] = useState('');
  const [asyncProgress, setAsyncProgress] = useState(0);
  const [asyncStatus, setAsyncStatus] = useState(''); // pending|processing|paused|completed|failed|cancelled
  const [asyncStartedAt, setAsyncStartedAt] = useState<string>('');
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  // True while an annotation task is queued/running/paused — the semi-auto and
  // batch-accept buttons must be disabled then (one running task per project).
  const taskRunning = ['pending', 'processing', 'paused'].includes(asyncStatus);

  // CVAT routing follows ADR-010 D1: video-tracking projects live in CVAT.
  // Both fields are checked so legacy rows (annotation_tool mis-recorded as
  // label-studio) still route correctly.
  const isCVAT = task?.annotation_tool === 'cvat' || task?.task_type === 'video-tracking';
  const currentTask = isCVAT ? null : lsTasks[currentIndex];
  // Bumped by the CVAT header refresh button to force CVATAnnotator refetch.
  const [annotatorReloadNonce, setAnnotatorReloadNonce] = useState(0);

  const sdkContainerRef = useRef<HTMLDivElement>(null);
  const lsInstanceRef = useRef<any>(null);

  // Rewrite image URLs: backend-generated //host/api/... or http://host/api/... or s3://...
  // Rewrite to current page origin + path (browser-reachable, avoids backend host like localhost:8080)
  const resolveImageSrc = useCallback((url: string) => {
    if (!url) return '';
    // Convert s3:// URI to NF backend proxy (backend reads S3 and returns HTTP, browser-reachable)
    if (url.startsWith('s3://')) {
      return `/api/v1/s3-object?uri=${encodeURIComponent(url)}`;
    }
    let u = url;
    if (u.startsWith('//')) u = window.location.protocol + u;
    // If /api/v1/... relative path, return as-is; otherwise replace absolute host with current origin
    try {
      const parsed = new URL(u);
      if (parsed.pathname.startsWith('/api/')) {
        return window.location.origin + parsed.pathname + parsed.search;
      }
    } catch { /* not a URL */ }
    return u;
  }, []);

  const fetchCVATTasks = useCallback(async (pg: number) => {
    if (!id) return;
    const res = await annotationApi.getCVATTasks(id, { page: pg, page_size: 50 });
    setCVATTasks(res.data.items);
    setTotalTasks(res.data.total);
    // auto-select the first task when nothing is selected yet
    setSelectedCVATTasks((prev) => {
      if (prev.size === 1 && res.data.items.some((t2) => t2.id === [...prev][0])) return prev;
      return res.data.items.length ? new Set([res.data.items[0].id]) : new Set<number>();
    });
  }, [id]);

  const fetchLSTasks = useCallback(async (pg: number) => {
    if (!id) return;
    const res = await annotationApi.getLSTasks(id, { page: pg, page_size: 20 });
    setLSTasks(res.data.items);
    setTotalTasks(res.data.total);
    setCurrentIndex(0);
  }, [id]);

  const fetchTask = useCallback(async () => {
    if (!id) return;
    const res = await annotationApi.get(id);
    setTask(res.data);
    const config = buildConfigFromSchema(res.data.schema);
    if (config) setLabelConfig(config);
    if (res.data.annotation_tool === 'cvat' || res.data.task_type === 'video-tracking') {
      await fetchCVATTasks(1);
    } else {
      // Fetch LS tasks
      await fetchLSTasks(1);
    }
  }, [id, fetchCVATTasks, fetchLSTasks]);

  // CVAT header refresh: reload project + task list and force the embedded
  // CVATAnnotator to refetch its annotations/frames (props alone won't).
  const refreshCVAT = useCallback(() => {
    fetchTask();
    setAnnotatorReloadNonce((n) => n + 1);
  }, [fetchTask]);

  // Add-label modal: lets the annotator extend the label set mid-task; the
  // backend persists schema.labels and syncs to LS label_config / CVAT.
  const [labelsOpen, setLabelsOpen] = useState(false);
  const [labelsInput, setLabelsInput] = useState('');
  const [savingLabels, setSavingLabels] = useState(false);

  const submitLabels = async () => {
    if (!id) return;
    const names = labelsInput.split(',').map((s) => s.trim()).filter(Boolean);
    if (!names.length) { setLabelsOpen(false); return; }
    setSavingLabels(true);
    try {
      const res = await annotationApi.addLabels(id, names);
      const updated = res.data;
      setTask(updated);
      message.success(t('pages.annotations.studio.labelsAdded'));
      setLabelsOpen(false);
      setLabelsInput('');
      if (isCVAT) {
        // CVATAnnotator reads labels on mount/reload; force a refetch.
        setAnnotatorReloadNonce((n) => n + 1);
      } else {
        // The LS SDK effect rebuilds from labelConfig; regenerate from schema.
        const cfg = buildConfigFromSchema(updated.schema);
        if (cfg) setLabelConfig(cfg);
        await fetchLSTasks(page);
      }
    } catch (e) {
      const detail = (e as any)?.response?.data?.detail || (e as any)?.response?.data?.message;
      message.error(detail || t('pages.annotations.studio.labelsAddFailed'));
    } finally {
      setSavingLabels(false);
    }
  };


  // CVAT job and marks the NF project 100%/completed; for LS projects it just
  // finalizes the NF status. Returns to the annotation list on success.
  const [completing, setCompleting] = useState(false);
  const handleComplete = () => {
    if (!id) return;
    Modal.confirm({
      title: t('pages.annotations.studio.completeConfirmTitle'),
      content: t('pages.annotations.studio.completeConfirmContent'),
      okText: t('pages.annotations.studio.complete'),
      okButtonProps: { danger: false, type: 'primary' },
      cancelText: t('common.cancel'),
      onOk: async () => {
        setCompleting(true);
        try {
          await annotationApi.complete(id);
          message.success(t('pages.annotations.studio.completeSuccess'));
          navigate('/annotations');
        } catch (e) {
          const detail = (e as any)?.response?.data?.detail || (e as any)?.response?.data?.message;
          message.error(detail || t('pages.annotations.studio.completeFailed'));
        } finally {
          setCompleting(false);
        }
      },
    });
  };

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  // Apply a fetched task object to the inline progress-bar state.
  const applyTaskState = useCallback((tk: any, title: string) => {
    setAsyncTaskId(tk.id);
    setAsyncTitle(title);
    setAsyncProgress(tk.progress ?? 0);
    setAsyncStatus(tk.status);
    setAsyncStartedAt(tk.created_at || '');
  }, []);

  // Poll an async task's progress into the inline bar (no modal); refresh the
  // task list once it reaches a terminal state so new boxes/annotations show.
  const startAsyncPolling = useCallback((taskId: string, title: string, onDone?: () => void) => {
    setAsyncTaskId(taskId);
    setAsyncTitle(title);
    setAsyncProgress(0);
    setAsyncStatus('processing');
    setSelectedIds([]);
    stopPolling();
    pollRef.current = setInterval(async () => {
      if (!id) return;
      try {
        const res = await annotationApi.getAsyncTask(id, taskId);
        const tk = res.data;
        setAsyncProgress(tk.progress);
        setAsyncStatus(tk.status);
        if (tk.status === 'completed') {
          stopPolling();
          if (isCVAT) {
            await fetchCVATTasks(page);
          } else {
            await fetchLSTasks(page);
          }
          onDone?.();
        } else if (tk.status === 'failed') {
          stopPolling();
          message.error(tk.error_message || String(tk.status));
        } else if (tk.status === 'cancelled') {
          stopPolling();
          if (isCVAT) {
            await fetchCVATTasks(page);
          } else {
            await fetchLSTasks(page);
          }
        }
      } catch (e) {
        console.error('Async task poll failed:', e);
      }
    }, 1500);
  }, [id, isCVAT, fetchLSTasks, fetchCVATTasks, page, stopPolling]);

  useEffect(() => () => stopPolling(), [stopPolling]);

  // Pause / resume / stop controls for the running async task.
  const handleAsyncControl = useCallback(async (action: 'pause' | 'resume' | 'cancel') => {
    if (!id || !asyncTaskId) return;
    try {
      const res = await annotationApi.controlAsyncTask(id, asyncTaskId, action);
      const tk = res.data;
      setAsyncStatus(tk.status);
      if (action === 'cancel') {
        stopPolling();
        setAsyncStatus('cancelled');
        message.info(t('pages.annotations.studio.taskStopped'));
        if (isCVAT) {
          await fetchCVATTasks(page);
        } else {
          await fetchLSTasks(page);
        }
      } else if (action === 'pause') {
        message.info(t('pages.annotations.studio.taskPaused'));
      } else if (action === 'resume') {
        message.info(t('pages.annotations.studio.taskResumed'));
      }
    } catch (e) {
      console.error('Async task control failed:', e);
      message.error(String((e as any)?.response?.data?.detail || e));
    }
  }, [id, asyncTaskId, isCVAT, fetchLSTasks, fetchCVATTasks, page, stopPolling, t]);

  // Restore the inline bar after a page refresh if a task is still running.
  useEffect(() => {
    if (!id) return;
    let cancelled = false;
    (async () => {
      try {
        const res = await annotationApi.getActiveAsyncTask(id);
        const tk = res.data?.task;
        if (!cancelled && tk && ['pending', 'processing', 'paused'].includes(tk.status)) {
          const title = tk.type === 'batch_accept'
            ? t('pages.annotations.studio.acceptPrediction')
            : tk.type === 'ls_import'
            ? t('pages.annotations.studio.importTasks')
            : t('pages.annotations.studio.semiAuto');
          applyTaskState(tk, title);
          startAsyncPolling(tk.id, title);
        }
      } catch { /* no active task */ }
    })();
    return () => { cancelled = true; };
  }, [id, applyTaskState, startAsyncPolling, t]);

  // Semi-auto annotation (async batch): selected tasks or all unlabeled → background inference, inline progress bar
  const handlePreannotate = async () => {
    if (!id || taskRunning) return;
    setPreannotating(true);
    try {
      const ids = !isCVAT && selectedIds.length ? selectedIds.join(',') : undefined;
      const res = await annotationApi.preannotate(id, {
        limit: 0,
        ml_backend_id: selectedMLBackend || undefined,
        task_ids: ids,
        frame_stride: frameStride,
        score_threshold: scoreThreshold,
        model_name: preModelName || undefined,
        model_version: preVersion || undefined,
      });
      startAsyncPolling(res.data.task_id, t('pages.annotations.studio.semiAuto'), () => {
        message.success(t('pages.annotations.studio.preannotateSuccess'));
      });
    } catch (e) {
      console.error('Preannotate failed:', e);
      const detail = (e as any)?.response?.data?.detail;
      message.error(detail || t('pages.annotations.studio.preannotateFailed'));
    } finally {
      setPreannotating(false);
    }
  };

  // Batch accept annotations (async): selected tasks or all unlabeled → convert predictions to annotations, write empty annotations if no predictions
  const handleBatchAccept = async () => {
    if (!id || taskRunning) return;
    setPreannotating(true);
    try {
      const ids = selectedIds.length ? { task_ids: selectedIds } : { task_ids: lsTasks.filter((x) => !x.is_labeled).map((x) => x.id) };
      const res = await annotationApi.batchAccept(id, ids);
      startAsyncPolling(res.data.task_id, t('pages.annotations.studio.acceptPrediction'), () => {
        message.success(t('pages.annotations.studio.acceptPredictionSuccess'));
      });
    } catch (e) {
      console.error('Batch accept failed:', e);
      const detail = (e as any)?.response?.data?.detail;
      message.error(detail || t('pages.annotations.studio.acceptPredictionFailed'));
    } finally {
      setPreannotating(false);
    }
  };

  // Accept all predictions: convert predictions on all unlabeled tasks directly to annotations (skip empty annotations)
  const handleBatchAcceptAll = async () => {
    if (!id || taskRunning) return;
    setPreannotating(true);
    try {
      const res = await annotationApi.batchAccept(id, { task_ids: [] });
      startAsyncPolling(res.data.task_id, t('pages.annotations.studio.acceptAllPredictions'), () => {
        message.success(t('pages.annotations.studio.acceptPredictionSuccess'));
      });
    } catch (e) {
      console.error('Batch accept all failed:', e);
      const detail = (e as any)?.response?.data?.detail;
      message.error(detail || t('pages.annotations.studio.acceptPredictionFailed'));
    } finally {
      setPreannotating(false);
    }
  };

  // Auto-predict single image: run inference on an unlabeled image with no predictions when opened
  const autoPredictCurrentTask = useCallback(async () => {
    if (!id || !currentTask || currentTask.is_labeled || !currentTask.predictions?.length) return;
    if (!preModelName || !preVersion) return;
    try {
      await annotationApi.preannotate(id, {
        limit: 1,
        ml_backend_id: selectedMLBackend || undefined,
        task_ids: String(currentTask.id),
        model_name: preModelName,
        model_version: preVersion,
      });
      await fetchLSTasks(page);
      message.success(t('pages.annotations.studio.preannotateSuccess'));
    } catch (e) {
      console.error('Auto predict failed:', e);
    }
  }, [id, currentTask, selectedMLBackend, preModelName, preVersion, fetchLSTasks, page, t]);

  useEffect(() => {
    if (!id) return;
    setLoading(true);
    fetchTask()
      .catch((e) => setSdkError(String(e?.message || e)))
      .finally(() => setLoading(false));
  }, [id, fetchTask]);

  // Load the LS SDK only for Label Studio projects (CVAT pages never use it).
  useEffect(() => {
    if (!task || isCVAT) return;
    loadLS()
      .then(() => setSdkReady(true))
      .catch((e) => setSdkError(String(e?.message || e)));
  }, [task, isCVAT]);

  // Load available ML backends (selectable models for semi-auto annotation)
  useEffect(() => {
    import('@api/endpoints').then(({ mlBackendApi }) => {
      mlBackendApi.list({ page_size: 50 }).then((res) => {
        setMLBackends(res.data.items ?? []);
      }).catch(() => {});
    });
  }, []);

  // Load CNN models usable for box/segmentation pre-annotation (SAM is an
  // interactive tool picked separately inside the annotator, so excluded).
  useEffect(() => {
    Promise.all([
      modelApi.list({ page_size: 100, task_type: 'detection' }),
      modelApi.list({ page_size: 100, task_type: 'segmentation' }),
    ])
      .then(([det, seg]) => setPreModels([...(det.data.items || []), ...(seg.data.items || [])]))
      .catch(() => setPreModels([]));
  }, []);

  // Load versions whenever the chosen model changes.
  useEffect(() => {
    if (!preModelName) { setPreVersions([]); return; }
    const m = preModels.find((mm) => mm.name === preModelName);
    if (!m) { setPreVersions([]); return; }
    let cancelled = false;
    modelApi.getVersions(m.id, { page_size: 100 })
      .then((r) => { if (!cancelled) setPreVersions(r.data.items || []); })
      .catch(() => { if (!cancelled) setPreVersions([]); });
    return () => { cancelled = true; };
  }, [preModelName, preModels]);

  // A version is usable for CNN inference only when it carries .pt/.onnx
  // weights (training artifact weights_uri or an uploaded checkpoint).
  const versionHasWeights = (v: ModelVersion): boolean =>
    !!v.metadata?.weights_uri || /\.(pt|onnx)$/i.test(v.file_path || '');
  const usableVersions = preVersions.filter(versionHasWeights);

  // Initialise the picker exactly once after the project preference and model
  // catalogue are available: remembered project choice wins, else first model.
  const prePickInitRef = useRef(false);
  useEffect(() => {
    if (prePickInitRef.current || !task || preModels.length === 0) return;
    const pref = task.preannotate_model || {};
    if (pref.model_name && preModels.some((m) => m.name === pref.model_name)) {
      setPreModelName(pref.model_name);
      setPreVersion(pref.model_version || undefined);
      prePickInitRef.current = true;
    } else {
      setPreModelName(preModels[0].name);
    }
  }, [task, preModels]);

  // Persist a pre-annotation model choice to the project (per-project memory).
  const persistPreModel = useCallback((modelName: string | undefined, version: string | undefined) => {
    if (!id || !modelName || !version) return;
    annotationApi.update(id, {
      preannotate_model: { model_name: modelName, model_version: version },
    } as Partial<AnnotationTask>).catch(() => {});
  }, [id]);

  // Picking a model clears the version; the version-load effect below then
  // auto-selects the first usable version and persists it for the project.
  const onPickPreModel = (name: string | undefined) => {
    setPreModelName(name);
    setPreVersion(undefined);
  };

  const onPickPreVersion = (v: string | undefined) => {
    setPreVersion(v);
    persistPreModel(preModelName, v);
  };

  // Auto-select + persist the first usable version once versions load, unless
  // the project already remembered a specific version. Re-runs per model.
  const preVersionInitForRef = useRef<string | undefined>(undefined);
  useEffect(() => {
    if (!preModelName || usableVersions.length === 0) return;
    if (preVersionInitForRef.current === preModelName) return;
    const pref = task?.preannotate_model || {};
    const prefUsable = pref.model_name === preModelName
      && pref.model_version
      && usableVersions.some((v) => v.version === pref.model_version);
    if (prefUsable) {
      setPreVersion(pref.model_version);
    } else {
      const v = usableVersions[0].version;
      setPreVersion(v);
      persistPreModel(preModelName, v);
    }
    preVersionInitForRef.current = preModelName;
  }, [usableVersions, preModelName, task, persistPreModel]);

  // CNN pre-annotation is only runnable once a model + weighted version exist.
  const preModelReady = !!preModelName && !!preVersion;

  // Persist the active annotation to LS via the NF backend. `ann` is the
  // completion object handed to the SDK onSubmit/onUpdate callbacks; fall back
  // to the store selection for the NF "Save annotation" button.
  const persistCurrentAnnotation = useCallback(async (ann?: any) => {
    if (!id || !currentTask) return;
    setSaving(true);
    try {
      let result: unknown[] = [];
      try {
        const source = ann?.serializeAnnotation ? ann : window.Htx?.annotationStore?.selected;
        if (source?.serializeAnnotation) {
          result = source.serializeAnnotation();
        }
      } catch (e) {
        console.warn('Serialize annotation failed, saving empty:', e);
      }
      await annotationApi.saveLSAnnotation(id, currentTask.id, { result });
      // Update current task as labeled locally (with serialized result) so SDK can immediately continue to next image
      setLSTasks((prev) => prev.map((t) =>
        t.id === currentTask.id
          ? { ...t, is_labeled: true, annotations: [{ id: -1, result }] }
          : t
      ));
      message.success(t('pages.annotations.studio.saveSuccess'));
    } catch (e) {
      console.error('Save failed:', e);
      message.error(String((e as any)?.response?.data?.error || e));
    } finally {
      setSaving(false);
    }
  }, [id, currentTask, t]);

  const handleSave = () => persistCurrentAnnotation();

  const destroyLS = useCallback(() => {
    try {
      if (lsInstanceRef.current?.destroy) {
        lsInstanceRef.current.destroy();
      }
      if (window.LabelStudio?.destroyAll) {
        window.LabelStudio.destroyAll();
      }
    } catch {
      /* noop */
    }
    lsInstanceRef.current = null;
  }, []);

  const renderTask = useCallback(() => {
    if (!currentTask || !sdkContainerRef.current || !window.LabelStudio) return;

    destroyLS();

    const taskData: any = {
      id: currentTask.id,
      data: currentTask.data
        ? { ...currentTask.data, image: resolveImageSrc(String(currentTask.data.image || '')) }
        : {},
      predictions: currentTask.predictions || [],
      // LSF 1.8 adds predictions first and then annotations, selecting the
      // LAST annotation on init — so an unconditional empty draft here gets
      // selected and hides every prediction region (boxes invisible). Only
      // seed the draft for a truly empty task; when predictions exist, open
      // in prediction-view (boxes visible; Accept copies them into an
      // annotation), matching the native Label Studio UX.
      annotations: (currentTask.annotations?.length ?? 0) > 0
        ? currentTask.annotations
        : (currentTask.predictions?.length ?? 0) > 0
          ? []
          : [{ id: -1, result: [] }],
    };

    try {
      // Clean up any leftover global APP_SETTINGS to prevent SDK from reading stale user data
      if (window.APP_SETTINGS) {
        delete (window as any).APP_SETTINGS;
      }
      const instance = new window.LabelStudio(sdkContainerRef.current, {
        config: labelConfig,
        task: taskData,
        interfaces: ['topbar', 'panel', 'update', 'submit', 'controls', 'side-column', 'annotations:menu'],
        user: { id: 1, pk: 1, firstName: 'Annotator', lastName: '' },
        onLabelStudioLoad: (LS: any) => {
          lsInstanceRef.current = LS;
        },
        // The in-canvas Submit/Update buttons (interfaces 'submit'/'update')
        // would otherwise only mutate local MST state and never hit the server;
        // persist through the same path as the NF "Save annotation" button.
        onSubmitAnnotation: (_LS: any, annotation: any) => {
          void persistCurrentAnnotation(annotation);
        },
        onUpdateAnnotation: (_LS: any, annotation: any) => {
          void persistCurrentAnnotation(annotation);
        },
      });
      lsInstanceRef.current = instance;
    } catch (e) {
      console.error('LS init error:', e);
    }
  }, [currentTask, labelConfig, destroyLS, resolveImageSrc, persistCurrentAnnotation]);

  useEffect(() => {
    if (sdkReady && currentTask) {
      const timer = setTimeout(renderTask, 50);
      return () => {
        clearTimeout(timer);
        destroyLS();
      };
    }
  }, [sdkReady, currentTask, renderTask, destroyLS]);

  // Auto-run prediction when opening an image with no predictions
  useEffect(() => {
    if (sdkReady && currentTask && !currentTask.is_labeled && currentTask.predictions?.length === 0) {
      const timer = setTimeout(autoPredictCurrentTask, 400);
      return () => clearTimeout(timer);
    }
  }, [sdkReady, currentTask, autoPredictCurrentTask]);

  // LS task list paging (LS projects only; wait for task so isCVAT is known)
  useEffect(() => {
    if (!id || !task || isCVAT) return;
    let cancelled = false;
    const load = async () => {
      try {
        const res = await annotationApi.getLSTasks(id, { page, page_size: 20 });
        if (cancelled) return;
        setLSTasks(res.data.items);
        setTotalTasks(res.data.total);
        setCurrentIndex(0);
      } catch (e) {
        if (!cancelled) console.error('Failed to load task page:', e);
      }
    };
    load();
    return () => { cancelled = true; };
  }, [id, task, page, isCVAT]);


  // LS-image SAM assist: click on the image -> polygon -> LS PolygonLabels
  // region injected into the active annotation and saved.
  const runLsSam = async (evt: React.MouseEvent<HTMLDivElement>) => {
    if (!lsSamMode || !currentTask || !id) return;
    // locate the LS image element for coordinates
    const imgEl = lsSamHostRef.current?.querySelector('img') as HTMLImageElement | null;
    if (!imgEl) return;
    const rect = imgEl.getBoundingClientRect();
    const sx = imgEl.naturalWidth / rect.width;
    const sy = imgEl.naturalHeight / rect.height;
    const px = (evt.clientX - rect.left) * sx;
    const py = (evt.clientY - rect.top) * sy;
    setLsSamBusy(true);
    try {
      // frame/blob reuse is CVAT-only; for LS images fetch via s3 proxy to b64
      const src = resolveImageSrc(String(currentTask.data?.image || ''));
      const blob = await fetch(src).then((r) => r.blob());
      const b64 = await new Promise<string>((resolve) => {
        const fr = new FileReader();
        fr.onload = () => resolve(String(fr.result).split(',')[1] || '');
        fr.readAsDataURL(blob);
      });
      const res = await samApi.segment({
        image_base64: b64,
        points: [[px, py]],
        ml_backend_id: selectedMLBackend || undefined,
      });
      const poly = res.data.polygon;
      if (!poly || poly.length < 3) {
        message.warning(res.data.message || t('pages.annotations.studio.samNoMask'));
        return;
      }
      // inject a polygonlabels region into the LS annotation store
      const Htx = window.Htx;
      const ann = Htx?.annotationStore?.selected;
      if (!ann) {
        message.warning(t('pages.annotations.studio.samNeedAnnotation'));
        return;
      }
      const xs = poly.map((pt) => pt[0]);
      const ys = poly.map((pt) => pt[1]);
      const w = imgEl.naturalWidth || 1;
      const h = imgEl.naturalHeight || 1;
      const labelName = String((task?.schema?.labels as string[] | undefined)?.[0] || 'object');
      // polygon projects get the true contour; detection (rectangle) configs
      // fall back to the polygon's bounding box as a rectanglelabels region
      const schemaType = String((task?.schema as Record<string, unknown> | undefined)?.type || 'rectanglelabels');
      const isPolyCfg = schemaType === 'polygonlabels';
      const x = +(Math.min(...xs) / w * 100).toFixed(2);
      const y = +(Math.min(...ys) / h * 100).toFixed(2);
      const width = +((Math.max(...xs) - Math.min(...xs)) / w * 100).toFixed(2);
      const height = +((Math.max(...ys) - Math.min(...ys)) / h * 100).toFixed(2);
      const region = ann.createRegion(
        isPolyCfg
          ? {
              type: 'polygonlabels',
              from_name: 'label',
              to_name: 'image',
              value: {
                polygonlabels: [labelName],
                points: poly.map((pt) => [+(pt[0] / w * 100).toFixed(2), +(pt[1] / h * 100).toFixed(2)]),
                x, y, width, height,
              },
            }
          : {
              type: 'rectanglelabels',
              from_name: 'label',
              to_name: 'image',
              value: {
                rectanglelabels: [labelName],
                x, y, width, height,
              },
            },
      );
      if (region) message.success(t('pages.annotations.studio.samCommitted'));
    } catch (e) {
      console.error('LS SAM failed:', e);
      message.error(String((e as any)?.response?.data?.detail || e));
    } finally {
      setLsSamBusy(false);
    }
  };

  // Accept prediction: convert current task's predictions to formal annotations, set is_labeled -> true
  const handleAcceptPrediction = async () => {
    if (!currentTask || !currentTask.predictions?.length) return;
    setSaving(true);
    try {
      const result = currentTask.predictions[0].result ?? [];
      await annotationApi.saveLSAnnotation(id!, currentTask.id, { result });
      // Update current task state locally to avoid slow full-page refresh
      setLSTasks((prev) => prev.map((t) =>
        t.id === currentTask.id
          ? { ...t, is_labeled: true, annotations: [{ id: -1, result }] }
          : t
      ));
      message.success(t('pages.annotations.studio.acceptPredictionSuccess'));
    } catch (e) {
      console.error('Accept prediction failed:', e);
      message.error(t('pages.annotations.studio.acceptPredictionFailed'));
    } finally {
      setSaving(false);
    }
  };

  const goNext = useCallback(() => {
    if (currentIndex + 1 < lsTasks.length) {
      setCurrentIndex(currentIndex + 1);
    } else if (page * 20 < totalTasks) {
      setPage(page + 1);
    }
  }, [currentIndex, lsTasks.length, page, totalTasks]);

  const goPrev = useCallback(() => {
    if (currentIndex > 0) {
      setCurrentIndex(currentIndex - 1);
    } else if (page > 1) {
      setPage(page - 1);
      setCurrentIndex(19);
    }
  }, [currentIndex, page]);

  const handleExport = async () => {
    setExporting(true);
    try {
      const res = await annotationApi.exportDataset(id!, {
        name: exportName || `${task?.name || 'annotated'}-dataset`,
        format: exportFormat,
      });
      setExportOpen(false);
      Modal.success({
        title: t('pages.annotations.studio.exportSuccess'),
        content: (
          <div>
            <p>Dataset: {res.data.name} (v{res.data.dataset_version})</p>
            <Space>
              <Button type="primary" icon={<RocketOutlined />} onClick={() => navigate('/training')}>
                {t('pages.annotations.studio.goTraining')}
              </Button>
              <Button onClick={() => navigate('/datasets')}>Datasets</Button>
            </Space>
          </div>
        ),
      });
    } catch (e) {
      console.error('Export failed:', e);
      message.error(String((e as any)?.response?.data?.error || e));
    } finally {
      setExporting(false);
    }
  };

  if (loading) {
    return <Spin size="large" style={{ display: 'block', margin: '100px auto' }} />;
  }

  if (!task) {
    return <Alert message="Task not found" type="error" />;
  }

  // CNN pre-annotation model + version pickers (shared by CVAT and LS views).
  const preModelSelect = (
    <Space.Compact>
      <Select
        showSearch
        optionFilterProp="label"
        placeholder={t('pages.annotations.studio.selectPreModel')}
        style={{ minWidth: 150 }}
        value={preModelName || undefined}
        onChange={onPickPreModel}
        options={preModels.map((m) => ({ value: m.name, label: m.name }))}
      />
      <Select
        style={{ minWidth: 96 }}
        placeholder={t('pages.annotations.studio.selectPreVersion')}
        value={preVersion || undefined}
        onChange={onPickPreVersion}
        options={usableVersions.map((v) => ({ value: v.version, label: `v${v.version}` }))}
      />
    </Space.Compact>
  );
  const semiAutoDisabled = taskRunning || !preModelReady;

  // Shared "+ label" control (extends the label set mid-annotation) and its
  // input modal; used by both the CVAT and Label Studio headers.
  const addLabelButton = (
    <Tooltip title={t('pages.annotations.studio.addLabelHint')}>
      <Button icon={<TagsOutlined />} onClick={() => setLabelsOpen(true)}>
        {t('pages.annotations.studio.addLabel')}
      </Button>
    </Tooltip>
  );
  const addLabelModal = (
    <Modal
      title={t('pages.annotations.studio.addLabel')}
      open={labelsOpen}
      onOk={submitLabels}
      confirmLoading={savingLabels}
      onCancel={() => setLabelsOpen(false)}
      okText={t('common.save')}
      cancelText={t('common.cancel')}
    >
      <Input
        value={labelsInput}
        onChange={(e) => setLabelsInput(e.target.value)}
        placeholder={t('pages.annotations.studio.labelsPlaceholder')}
        onPressEnter={submitLabels}
      />
    </Modal>
  );

  // -- CVAT view: NF-native editor (task list left, frame annotator right).

  const selectedCVATTaskId = selectedCVATTasks.size === 1 ? [...selectedCVATTasks][0] : null;

  // Full-width inline progress bar (replaces the old modal). It stays visible
  // after the task finishes (completed / failed / stopped) so the user can see
  // the result, and is only removed via its own close (×) button or when a new
  // task is started.
  // NOTE: this runs after the early returns above, so it must NOT be a hook
  // (would change the hook count between renders). It only closes over stable
  // setters and the memoized stopPolling, so a plain function is fine.
  const dismissAsyncTask = () => {
    stopPolling();
    setAsyncTaskId(null);
    setAsyncTitle('');
    setAsyncProgress(0);
    setAsyncStatus('');
    setAsyncStartedAt('');
  };

  const startedLabel = asyncStartedAt
    ? new Date(asyncStartedAt).toLocaleString()
    : '';
  const isTerminal = ['completed', 'failed', 'cancelled'].includes(asyncStatus);
  const terminalTag = asyncStatus === 'completed'
    ? <Tag color="green" style={{ margin: 0 }}>{t('pages.annotations.studio.taskDone')}</Tag>
    : asyncStatus === 'failed'
      ? <Tag color="red" style={{ margin: 0 }}>{t('pages.annotations.studio.taskFailed')}</Tag>
      : <Tag color="default" style={{ margin: 0 }}>{t('pages.annotations.studio.taskStopped')}</Tag>;
  const inlineTaskBar = asyncTaskId ? (
    <Card size="small" style={{ marginBottom: 12 }} styles={{ body: { padding: '10px 16px' } }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
        <ThunderboltOutlined style={{ color: isTerminal ? '#52c41a' : '#1677ff' }} />
        <span style={{ fontWeight: 600, whiteSpace: 'nowrap' }}>{asyncTitle}</span>
        {asyncStatus === 'paused' && (
          <Tag color="orange" style={{ margin: 0 }}>{t('pages.annotations.studio.taskPaused')}</Tag>
        )}
        {isTerminal && terminalTag}
        <Progress
          percent={asyncProgress}
          size="small"
          status={isTerminal
            ? (asyncStatus === 'failed' ? 'exception' : 'success')
            : (asyncStatus === 'paused' ? 'normal' : 'active')}
          style={{ flex: 1, minWidth: 160, margin: 0 }}
        />
        <span style={{ whiteSpace: 'nowrap', color: '#888', fontSize: 12 }}>{asyncProgress}%</span>
        {taskRunning && (
          <>
            {asyncStatus === 'paused' ? (
              <Button size="small" type="primary" ghost icon={<CaretRightOutlined />} onClick={() => handleAsyncControl('resume')}>
                {t('pages.annotations.studio.taskResume')}
              </Button>
            ) : (
              <Button size="small" icon={<PauseOutlined />} disabled={asyncStatus === 'pending'} onClick={() => handleAsyncControl('pause')}>
                {t('pages.annotations.studio.taskPause')}
              </Button>
            )}
            <Button size="small" danger icon={<StopOutlined />} onClick={() => handleAsyncControl('cancel')}>
              {t('pages.annotations.studio.taskStop')}
            </Button>
          </>
        )}
        <Tooltip title={startedLabel}>
          <span style={{ whiteSpace: 'nowrap', color: '#aaa', fontSize: 12 }}>
            {startedLabel ? new Date(asyncStartedAt).toLocaleTimeString() : ''}
          </span>
        </Tooltip>
        {isTerminal && (
          <Tooltip title={t('pages.annotations.studio.taskDismiss')}>
            <Button size="small" type="text" icon={<CloseOutlined />} onClick={dismissAsyncTask} />
          </Tooltip>
        )}
      </div>
    </Card>
  ) : null;

  if (isCVAT) {
    return (
      <div>
        <PageHeader
          title={task.name}
          subtitle={`CVAT • ${t(statusLabelKey(task.status))} • ${task.annotated_items}/${task.total_items}`}
          breadcrumbs={[{ title: t('pages.annotations.title'), href: '/annotations' }]}
          onRefresh={() => { refreshCVAT(); }}
          extra={
            <Space wrap>
              {task.external_url && (
                <Button icon={<VideoCameraOutlined />} onClick={() => window.open(task.external_url, '_blank')}>
                  {t('pages.annotations.studio.cvatOpenProject')}
                </Button>
              )}
              {preModelSelect}
              <Select
                placeholder={t('pages.annotations.studio.selectMLBackend')}
                allowClear
                style={{ minWidth: 150 }}
                value={selectedMLBackend || undefined}
                onChange={setSelectedMLBackend}
                options={mlBackends.map((b) => ({ value: b.id, label: b.name || b.id }))}
              />
              <Tooltip title={preModelReady ? t('pages.annotations.studio.cvatSemiAutoHint') : t('pages.annotations.studio.needPreModel')}>
                <Space.Compact>
                  <Button
                    icon={<ThunderboltOutlined />}
                    loading={preannotating}
                    disabled={semiAutoDisabled}
                    onClick={handlePreannotate}
                  >
                    {t('pages.annotations.studio.semiAuto')}
                  </Button>
                  <InputNumber
                    min={1}
                    max={60}
                    value={frameStride}
                    onChange={(v) => setFrameStride(v || 5)}
                    style={{ width: 64 }}
                    title={t('pages.annotations.studio.cvatFrameStride')}
                  />
                  <InputNumber
                    min={0.05}
                    max={1}
                    step={0.05}
                    value={scoreThreshold}
                    onChange={(v) => setScoreThreshold(v || 0.25)}
                    style={{ width: 70 }}
                    title={t('pages.annotations.studio.cvatScoreThreshold')}
                  />
                </Space.Compact>
              </Tooltip>
              {addLabelButton}
              <Button icon={<ExportOutlined />} onClick={() => setExportOpen(true)}>
                {t('pages.annotations.studio.exportDataset')}
              </Button>
              <Tooltip title={task.status === 'completed' ? t('pages.annotations.studio.completedTip') : t('pages.annotations.studio.completeHint')}>
                <Button
                  type="primary"
                  ghost
                  icon={<CheckCircleOutlined />}
                  loading={completing}
                  disabled={task.status === 'completed' || taskRunning}
                  onClick={handleComplete}
                >
                  {t('pages.annotations.studio.complete')}
                </Button>
              </Tooltip>
            </Space>
          }
        />

        {inlineTaskBar}

        <Card style={{ marginBottom: 12 }}>
          <Alert type="info" showIcon message={t('pages.annotations.studio.cvatEditorHint')} />
        </Card>

        {cvatTasks.length === 0 ? (
          <Card size="small">
            <Empty
              image={<VideoCameraOutlined style={{ fontSize: 48, color: '#bbb' }} />}
              description={t('pages.annotations.studio.cvatNoTasks')}
            >
              <Button icon={<ReloadOutlined />} onClick={() => fetchTask()}>
                {t('pages.annotations.studio.loading')}
              </Button>
            </Empty>
          </Card>
        ) : (
          <Row gutter={16}>
            <Col span={5}>
              <Card size="small" title={t('pages.annotations.studio.taskNavigation')} style={{ height: 'calc(100vh - 300px)', overflow: 'auto' }}>
                <List
                  size="small"
                  dataSource={cvatTasks}
                  renderItem={(item) => (
                    <List.Item
                      onClick={() => setSelectedCVATTasks(new Set([item.id]))}
                      style={{
                        cursor: 'pointer',
                        padding: '8px 12px',
                        background: item.id === selectedCVATTaskId ? '#e6f4ff' : undefined,
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8, width: '100%' }}>
                        <VideoCameraOutlined style={{ color: '#888' }} />
                        <div style={{ flex: 1, minWidth: 0 }}>
                          <div style={{ fontSize: 12, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                            {item.name || `Task #${item.id}`}
                          </div>
                          <div style={{ fontSize: 11, color: '#888' }}>
                            {item.size || (item.stop_frame ?? 0) + 1} {t('pages.annotations.studio.cvatFramesCol')}
                          </div>
                        </div>
                        <Badge
                          status={item.annotations?.length ? 'success' : 'default'}
                          text={item.annotations?.length || 0}
                          style={{ fontSize: 11 }}
                        />
                      </div>
                    </List.Item>
                  )}
                />
              </Card>
            </Col>
            <Col span={19}>
              <Card size="small" style={{ height: 'calc(100vh - 300px)', overflow: 'auto' }}>
                {selectedCVATTaskId == null ? (
                  <Empty description={t('pages.annotations.studio.cvatSelectTask')} style={{ margin: '120px 0' }} />
                ) : (
                  <CVATAnnotator
                    projectId={task.id}
                    taskId={selectedCVATTaskId}
                    onSaved={() => fetchCVATTasks(1)}
                    selectedMLBackendId={selectedMLBackend}
                    reloadNonce={annotatorReloadNonce}
                  />
                )}
              </Card>
            </Col>
          </Row>
        )}

        <Modal
          title={t('pages.annotations.studio.exportDataset')}
          open={exportOpen}
          onOk={handleExport}
          onCancel={() => setExportOpen(false)}
          confirmLoading={exporting}
          okText={t('pages.annotations.studio.exportDataset')}
        >
          <p style={{ color: '#888', fontSize: 13 }}>{t('pages.annotations.studio.exportHint')}</p>
          <Space direction="vertical" style={{ width: '100%' }} size="middle">
            <Input
              placeholder={t('pages.annotations.studio.exportNamePlaceholder')}
              value={exportName}
              onChange={(e) => setExportName(e.target.value)}
            />
            <Select
              placeholder={t('pages.annotations.studio.exportFormat')}
              value={exportFormat}
              onChange={setExportFormat}
              style={{ width: '100%' }}
              options={[
                { value: 'coco', label: 'COCO JSON' },
                { value: 'yolo', label: 'YOLO txt' },
              ]}
            />
          </Space>
        </Modal>
        {addLabelModal}
      </div>
    );
  }

  // -- Label Studio view (images / text).

  return (
    <div>
      <PageHeader
        title={task.name}
        subtitle={`${task.annotation_tool || 'label-studio'} • ${t(statusLabelKey(task.status))} • ${task.annotated_items}/${task.total_items}`}
        breadcrumbs={[{ title: t('pages.annotations.title'), href: '/annotations' }]}
        onRefresh={() => { fetchTask(); fetchLSTasks(page); }}
        extra={
          <Space wrap>
            {task.external_url && (
              <Button icon={<LinkOutlined />} onClick={() => window.open(task.external_url, '_blank')}>
                {t('pages.annotations.studio.openLabelStudio')}
              </Button>
            )}
            {preModelSelect}
            <Tooltip title={t('pages.annotations.studio.selectMLBackend')}>
              <Select
                placeholder={t('pages.annotations.studio.selectMLBackend')}
                allowClear
                style={{ minWidth: 150 }}
                value={selectedMLBackend || undefined}
                onChange={setSelectedMLBackend}
                options={mlBackends.map((b) => ({ value: b.id, label: b.name || b.id }))}
              />
            </Tooltip>
            <Tooltip title={preModelReady ? t('pages.annotations.studio.semiAuto') : t('pages.annotations.studio.needPreModel')}>
              <Button
                icon={<ThunderboltOutlined />}
                loading={preannotating}
                disabled={semiAutoDisabled}
                onClick={handlePreannotate}
              >
                {t('pages.annotations.studio.semiAuto')}
              </Button>
            </Tooltip>
            <Tooltip title={t('pages.annotations.studio.samModeHint')}>
              <Button
                type={lsSamMode ? 'primary' : 'default'}
                icon={<AimOutlined />}
                loading={lsSamBusy}
                onClick={() => setLsSamMode((v) => !v)}
              >
                {t('pages.annotations.studio.samMode')}
              </Button>
            </Tooltip>
            <Tooltip title={t('pages.annotations.studio.acceptPredictionHint')}>
              <Button
                icon={<RobotOutlined />}
                disabled={taskRunning}
                onClick={handleBatchAccept}
              >
                {t('pages.annotations.studio.acceptPrediction')}
              </Button>
            </Tooltip>
            <Tooltip title={t('pages.annotations.studio.acceptAllPredictionsHint')}>
              <Button
                icon={<CheckCircleOutlined />}
                disabled={taskRunning}
                onClick={handleBatchAcceptAll}
              >
                {t('pages.annotations.studio.acceptAllPredictions')}
              </Button>
            </Tooltip>
            {addLabelButton}
            <Button icon={<ExportOutlined />} onClick={() => setExportOpen(true)}>
              {t('pages.annotations.studio.exportDataset')}
            </Button>
          </Space>
        }
      />

      {inlineTaskBar}

      {sdkError && <Alert message={sdkError} type="error" style={{ marginBottom: 16 }} />}

      {lsTasks.length === 0 ? (
        <Card>
          {taskRunning
            ? <Empty image={<Spin />} description={t('pages.annotations.studio.importingTasks')} />
            : <Empty description={t('pages.annotations.studio.noTasks')} />}
        </Card>
      ) : (
        <Row gutter={16}>
          <Col span={5}>
            <Card size="small" title={t('pages.annotations.studio.taskNavigation')} style={{ height: 'calc(100vh - 220px)', overflow: 'auto' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8, marginBottom: 8 }}>
                <Checkbox
                  checked={lsTasks.length > 0 && selectedIds.length === lsTasks.length}
                  indeterminate={selectedIds.length > 0 && selectedIds.length < lsTasks.length}
                  onChange={(e) => {
                    if (e.target.checked) setSelectedIds(lsTasks.map((t) => t.id));
                    else setSelectedIds([]);
                  }}
                />
                <span style={{ fontSize: 12, color: '#888' }}>
                  {selectedIds.length > 0 ? `${t('pages.annotations.studio.selected')} ${selectedIds.length}` : t('pages.annotations.studio.selectAll')}
                </span>
              </div>
              <List
                size="small"
                dataSource={lsTasks}
                renderItem={(item, idx) => (
                  <List.Item
                    onClick={() => setCurrentIndex(idx)}
                    style={{ cursor: 'pointer', padding: '8px 12px', background: idx === currentIndex ? '#e6f4ff' : undefined }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, width: '100%' }}>
                      <Checkbox
                        checked={selectedIds.includes(item.id)}
                        onClick={(e) => e.stopPropagation()}
                        onChange={(e) => {
                          setSelectedIds((prev) => (e.target.checked ? [...prev, item.id] : prev.filter((x) => x !== item.id)));
                        }}
                      />
                      {item.image_url && (
                        <img
                          src={resolveImageSrc(item.image_url)}
                          alt={String(item.inner_id)}
                          style={{ width: 40, height: 40, objectFit: 'cover', borderRadius: 4 }}
                          onError={(e) => { (e.target as HTMLImageElement).style.visibility = 'hidden'; }}
                        />
                      )}
                      <div style={{ flex: 1 }}>
                        <div style={{ fontSize: 12 }}>#{item.inner_id}</div>
                        <Badge
                          status={item.is_labeled ? 'success' : item.predictions?.length ? 'processing' : 'default'}
                          text={item.is_labeled ? 'labeled' : item.predictions?.length ? 'pred' : 'pending'}
                          style={{ fontSize: 11 }}
                        />
                      </div>
                    </div>
                  </List.Item>
                )}
              />
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 8 }}>
                <Button size="small" disabled={page <= 1} onClick={() => setPage(page - 1)}>
                  {t('pages.annotations.studio.prevPage')}
                </Button>
                <span style={{ fontSize: 12, color: '#888' }}>
                  {page} / {Math.max(1, Math.ceil(totalTasks / 20))}
                </span>
                <Button size="small" disabled={page * 20 >= totalTasks} onClick={() => setPage(page + 1)}>
                  {t('pages.annotations.studio.nextPage')}
                </Button>
              </div>
            </Card>
          </Col>
          <Col span={19}>
            <Card size="small" style={{ height: 'calc(100vh - 220px)', overflow: 'hidden' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                <Space>
                  <Button size="small" icon={<LeftOutlined />} onClick={goPrev} />
                  <Button size="small" icon={<RightOutlined />} onClick={goNext} />
                  <span style={{ fontSize: 12, color: '#888' }}>
                    {currentIndex + 1} / {totalTasks} ({t('pages.annotations.studio.progressLabel')}: {totalTasks > 0 ? Math.round(((page - 1) * 20 + currentIndex + 1) / totalTasks * 100) : 0}%)
                  </span>
                </Space>
                <Space size="small">
                  {currentTask?.predictions && currentTask.predictions.length > 0 && (
                    <Tag color="purple" icon={<RobotOutlined />}>
                      {t('pages.annotations.studio.semiAuto')}: {currentTask.predictions.length}
                    </Tag>
                  )}
                  <Tag color={currentTask?.is_labeled ? 'green' : 'default'}>
                    {currentTask?.is_labeled ? 'Labeled' : 'Pending'}
                  </Tag>
                  <Button type="primary" size="small" icon={<SaveOutlined />} loading={saving} onClick={handleSave}>
                    {t('pages.annotations.studio.saveAnnotation')}
                  </Button>
                  <Tooltip title={t('pages.annotations.studio.acceptPredictionHint')}>
                    <Button
                      size="small"
                      icon={<RobotOutlined />}
                      disabled={!currentTask?.predictions?.length || currentTask?.is_labeled}
                      onClick={handleAcceptPrediction}
                    >
                      {t('pages.annotations.studio.acceptPrediction')}
                    </Button>
                  </Tooltip>
                </Space>
              </div>
              <div
                key={`sdk-${currentTask?.id ?? 'none'}`}
                ref={(node: HTMLDivElement | null): void => { (sdkContainerRef as React.MutableRefObject<HTMLDivElement | null>).current = node; lsSamHostRef.current = node; }}
                onClick={lsSamMode ? runLsSam : undefined}
                style={{
                  height: 'calc(100vh - 300px)',
                  overflow: 'auto',
                  cursor: lsSamMode ? 'crosshair' : undefined,
                  position: 'relative',
                }}
              />
            </Card>
          </Col>
        </Row>
      )}

      <Modal
        title={t('pages.annotations.studio.exportDataset')}
        open={exportOpen}
        onOk={handleExport}
        onCancel={() => setExportOpen(false)}
        confirmLoading={exporting}
        okText={t('pages.annotations.studio.exportDataset')}
      >
        <p style={{ color: '#888', fontSize: 13 }}>{t('pages.annotations.studio.exportHint')}</p>
        <Space direction="vertical" style={{ width: '100%' }} size="middle">
          <Input
            placeholder={t('pages.annotations.studio.exportNamePlaceholder')}
            value={exportName}
            onChange={(e) => setExportName(e.target.value)}
          />
          <Select
            placeholder={t('pages.annotations.studio.exportFormat')}
            value={exportFormat}
            onChange={setExportFormat}
            style={{ width: '100%' }}
            options={isCVAT ? [
              // CVAT export: bbox/track → COCO/YOLO only
              { value: 'coco', label: 'COCO JSON' },
              { value: 'yolo', label: 'YOLO txt' },
            ] : [
              { value: 'coco', label: 'COCO JSON' },
              { value: 'yolo', label: 'YOLO txt' },
              { value: 'labelstudio', label: 'Label Studio JSON' },
              { value: 'csv', label: 'CSV' },
            ]}
          />
        </Space>
      </Modal>
      {addLabelModal}
    </div>
  );
};

export default AnnotationStudio;
