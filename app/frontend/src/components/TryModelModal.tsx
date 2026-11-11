// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  Alert, Input, InputNumber, Modal, Upload, Button, Slider, Switch, Space, Spin,
  message, Empty, Tag, Radio, Select, Tooltip,
} from 'antd';
import { UploadOutlined } from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import type { UploadFile } from 'antd';
import { modelApi, datasetApi } from '../api/endpoints';
import type {
  Dataset, DatasetVersion, Model, ModelVersion, TrialDecoderConfig, TrialResultKind,
  TrialSchema, TrialTask, TryResult, VersionFile,
} from '../types/api';
import {
  classificationTopK, createCustomDecoderConfig, drawTrialOverlay, patchTrialOutput,
  trialDecoderError, trialErrorMessage, trialResultKind,
} from '../utils/modelTrial';

interface Props {
  model: Model | null;
  version: ModelVersion | null;
  // A trial launched from an artifact row always targets that exact sibling.
  filePath?: string | null;
  onClose: () => void;
}

// Images already on the platform are referenced via the read-only S3 proxy.
const s3ObjectUrl = (uri: string) =>
  `/api/v1/s3-object?uri=${encodeURIComponent(uri)}`;

/** Field keeps dense mapping controls aligned without introducing a second form state. */
const Field: React.FC<{ label: string; children: React.ReactNode }> = ({ label, children }) => (
  <label style={{ display: 'inline-flex', flexDirection: 'column', gap: 4, fontSize: 12 }}>
    <span style={{ color: '#666' }}>{label}</span>
    {children}
  </label>
);

/** TryModelModal configures and runs one artifact, then renders its normalized task result. */
const TryModelModal: React.FC<Props> = ({ model, version, filePath, onClose }) => {
  const { t } = useTranslation();
  const targetKey = `${model?.name || ''}/${version?.version || ''}/${filePath || ''}`;
  const [source, setSource] = useState<'dataset' | 'local'>('dataset');
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [dsVersions, setDsVersions] = useState<DatasetVersion[]>([]);
  const [dsVersionsFor, setDsVersionsFor] = useState('');
  const [files, setFiles] = useState<VersionFile[]>([]);
  const [dsId, setDsId] = useState<string>('');
  const [dsVersion, setDsVersion] = useState<number | undefined>();
  const [dsPath, setDsPath] = useState<string>('');
  const [loadingFiles, setLoadingFiles] = useState(false);
  const [autoPick, setAutoPick] = useState(true);

  const [fileList, setFileList] = useState<UploadFile[]>([]);
  const [localUrl, setLocalUrl] = useState<string>('');
  const [conf, setConf] = useState(0.25);
  const [topK, setTopK] = useState(5);
  const [cam, setCam] = useState(false);
  const [view, setView] = useState<'overlay' | 'heatmap'>('overlay');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<TryResult | null>(null);
  const [schema, setSchema] = useState<TrialSchema | null>(null);
  const [schemaLoading, setSchemaLoading] = useState(false);
  const [schemaError, setSchemaError] = useState('');
  const [decoderConfig, setDecoderConfig] = useState<TrialDecoderConfig>({ mode: 'auto' });
  const [maskImage, setMaskImage] = useState<HTMLImageElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const requestGeneration = useRef(0);
  const trialGeneration = useRef(0);
  const datasetVersionsGeneration = useRef(0);
  const datasetFilesGeneration = useRef(0);

  const hasPt = filePath ? /\.pt$/i.test(filePath) : (
    !!version?.metadata?.weights_uri || /\.pt$/i.test(version?.file_path || '')
  );

  /** taskLabel localizes known task names while preserving future registry labels. */
  const taskLabel = (task?: TrialResultKind) => {
    if (task === 'detection') return t('pages.models.tryTaskDetection');
    if (task === 'instance_segmentation') return t('pages.models.tryTaskInstanceSegmentation');
    if (task === 'semantic_segmentation') return t('pages.models.tryTaskSemanticSegmentation');
    if (task === 'classification') return t('pages.models.tryTaskClassification');
    return task || t('pages.models.tryTaskUnknown');
  };

  // Target changes invalidate both image state and late responses from the previous artifact.
  useEffect(() => {
    const generation = ++requestGeneration.current;
    trialGeneration.current += 1;
    setResult(null);
    setMaskImage(null);
    setLoading(false);
    setSchemaLoading(false);
    setView('overlay');
    setFileList([]);
    setLocalUrl('');
    setDsPath('');
    setFiles([]);
    setDsVersion(undefined);
    setAutoPick(true);
    setSchema(null);
    setSchemaError('');
    setDecoderConfig({ mode: 'auto' });
    const srcDataset = version?.metadata?.dataset_id as string | undefined;
    setDsId(srcDataset || '');
    if (!model || !version) return;
    setSchemaLoading(true);
    modelApi.getTrySchema(model.name, version.version, filePath || undefined)
      .then((response) => {
        if (generation !== requestGeneration.current) return;
        setSchema(response.data);
        setDecoderConfig(response.data.saved_decoder || { mode: 'auto' });
      })
      .catch((error) => {
        if (generation !== requestGeneration.current) return;
        setSchemaError(trialErrorMessage(error, t('pages.models.trySchemaFailed')));
      })
      .finally(() => {
        if (generation === requestGeneration.current) setSchemaLoading(false);
      });
  }, [targetKey, model, version, filePath, t]);

  useEffect(() => {
    if (!hasPt) setCam(false);
  }, [hasPt]);

  // Revoke every replaced local object URL, including the final URL on close.
  useEffect(() => () => {
    if (localUrl) URL.revokeObjectURL(localUrl);
  }, [localUrl]);

  useEffect(() => {
    datasetApi.list({ page_size: 100 })
      .then((r) => setDatasets(r.data.items || []))
      .catch(() => setDatasets([]));
  }, []);

  useEffect(() => {
    const generation = ++datasetVersionsGeneration.current;
    setDsVersions([]);
    setDsVersionsFor('');
    if (!dsId) return;
    datasetApi.getVersions(dsId, { page_size: 100 })
      .then((response) => {
        if (generation === datasetVersionsGeneration.current) {
          setDsVersions(response.data.items || []);
          setDsVersionsFor(dsId);
        }
      })
      .catch(() => {
        if (generation === datasetVersionsGeneration.current) setDsVersions([]);
      });
  }, [dsId]);

  useEffect(() => {
    if (!autoPick || dsVersionsFor !== dsId || !dsVersions.length || dsVersion !== undefined) return;
    const wanted = Number(version?.metadata?.dataset_version || 0);
    let pick = dsVersions.find((v) => v.version === wanted);
    if (!pick) {
      const annotated = dsVersions.filter((v) => v.is_annotated);
      pick = annotated.length === 1
        ? annotated[0]
        : [...dsVersions].sort((a, b) => b.version - a.version)[0];
    }
    if (pick) setDsVersion(pick.version);
  }, [autoPick, dsId, dsVersions, dsVersionsFor, dsVersion, version]);

  useEffect(() => {
    const generation = ++datasetFilesGeneration.current;
    setFiles([]); setDsPath('');
    if (!dsId || dsVersion === undefined) {
      setLoadingFiles(false);
      return;
    }
    setLoadingFiles(true);
    datasetApi.getVersionFiles(dsId, dsVersion)
      .then((response) => {
        if (generation === datasetFilesGeneration.current) {
          setFiles((response.data.files || []).filter((item) => item.is_media));
        }
      })
      .catch(() => {
        if (generation === datasetFilesGeneration.current) setFiles([]);
      })
      .finally(() => {
        if (generation === datasetFilesGeneration.current) setLoadingFiles(false);
      });
  }, [dsId, dsVersion]);

  const selectedVersion = dsVersionsFor === dsId
    ? dsVersions.find((v) => v.version === dsVersion)
    : undefined;
  const datasetImageUri = useMemo(() => {
    if (!selectedVersion?.uri || !dsPath) return '';
    return `${selectedVersion.uri.replace(/\/$/, '')}/${dsPath.replace(/^\//, '')}`;
  }, [selectedVersion, dsPath]);

  const resetResult = () => {
    trialGeneration.current += 1;
    setResult(null);
  };
  const imageReady = source === 'dataset' ? !!datasetImageUri : fileList.length > 0;
  const imageUrl = source === 'dataset'
    ? (datasetImageUri ? s3ObjectUrl(datasetImageUri) : '')
    : localUrl;

  const selectedTask: TrialResultKind | undefined = decoderConfig.mode === 'custom'
    ? decoderConfig.task
    : decoderConfig.mode === 'builtin'
      ? schema?.decoders.find((item) => item.id === decoderConfig.decoder_id)?.task
      : schema?.auto_decoder?.task;
  const decoderError = schema ? trialDecoderError(decoderConfig, schema) : 'schema required';
  const decoderErrorText = decoderError === 'decoder mapping required'
    ? t('pages.models.tryMappingRequired')
    : decoderError === 'output mapping required'
      ? t('pages.models.tryOutputMappingRequired')
      : decoderError === 'input mapping required'
        ? t('pages.models.tryInputMappingRequired')
        : decoderError === 'builtin decoder required'
          ? t('pages.models.tryBuiltinRequired')
          : schemaError;

  /** changeDecoderConfig invalidates pixels produced under the previous contract. */
  const changeDecoderConfig = (config: TrialDecoderConfig) => {
    resetResult();
    setDecoderConfig(config);
  };

  /** changeDecoderMode enters custom mode with editable defaults, not inferred output semantics. */
  const changeDecoderMode = (mode: TrialDecoderConfig['mode']) => {
    if (!schema) return;
    if (mode === 'auto') changeDecoderConfig({ mode });
    else if (mode === 'builtin') {
      changeDecoderConfig({
        mode,
        decoder_id: schema.auto_decoder?.id || schema.decoders[0]?.id || '',
      });
    } else {
      const candidate = schema.auto_decoder?.task;
      const supportedTasks: TrialResultKind[] = [
        'detection', 'classification', 'instance_segmentation', 'semantic_segmentation',
      ];
      const task = supportedTasks.includes(candidate || '') ? candidate as TrialTask : 'detection';
      changeDecoderConfig(createCustomDecoderConfig(schema, task));
    }
  };

  const input = decoderConfig.input;
  const output = (decoderConfig.output || {}) as Record<string, any>;
  const tensorOptions = (schema?.outputs || []).map((item) => ({
    value: item.name,
    label: `${item.name} [${item.shape.map((dim) => String(dim ?? '?')).join(',')}]`,
  }));

  /** updateInput merges one preprocessing field while retaining the selected artifact contract. */
  const updateInput = (patch: Record<string, unknown>) => {
    if (!input) return;
    changeDecoderConfig({ ...decoderConfig, input: { ...input, ...patch } });
  };

  /** updateOutput merges fields and omits optional axes explicitly cleared by the user. */
  const updateOutput = (patch: Record<string, unknown>) => {
    changeDecoderConfig({
      ...decoderConfig,
      output: patchTrialOutput(output, patch) as unknown as TrialDecoderConfig['output'],
    });
  };

  /** numberList parses the mean/std inputs without coercing an empty item to zero. */
  const numberList = (value: string) => value.split(',').map((item) => item.trim())
    .filter(Boolean).map(Number);

  const run = async () => {
    if (!model || !version || !imageReady) {
      message.warning(t('pages.models.tryPickImage'));
      return;
    }
    if (decoderError) {
      message.warning(decoderErrorText || t('pages.models.tryMappingRequired'));
      return;
    }
    const generation = ++trialGeneration.current;
    const artifactGeneration = requestGeneration.current;
    setLoading(true);
    try {
      let imageFile: File;
      if (source === 'local') {
        imageFile = fileList[0].originFileObj as File;
      } else {
        const response = await fetch(s3ObjectUrl(datasetImageUri));
        if (!response.ok) throw new Error(t('pages.models.tryImageLoadFailed'));
        const blob = await response.blob();
        const ext = dsPath.split('.').pop() || 'jpg';
        imageFile = new File([blob], dsPath.split('/').pop() || `image.${ext}`, { type: blob.type });
      }
      const response = await modelApi.tryVersion(model.name, version.version, {
        image: imageFile, conf, cam, artifactPath: filePath || undefined,
        decoderConfig, topK,
      });
      if (generation !== trialGeneration.current) return;
      setResult(response.data);
      setView('overlay');
      if (schema) setSchema({ ...schema, saved_decoder: decoderConfig });
    } catch (error: any) {
      if (generation === trialGeneration.current) {
        message.error(trialErrorMessage(error, t('pages.models.tryFailed')));
      }
    } finally {
      if (artifactGeneration === requestGeneration.current) setLoading(false);
    }
  };

  const heatUrl = useMemo(
    () => (result?.heatmap ? `data:image/jpeg;base64,${result.heatmap}` : ''),
    [result],
  );
  const srcUrl = view === 'heatmap' && heatUrl ? heatUrl : imageUrl;
  const resultKind = result ? trialResultKind(result) : undefined;
  const topKResults = result ? classificationTopK(result) : [];

  // Semantic PNG transport keeps large class maps out of JSON while preserving an RGBA overlay.
  useEffect(() => {
    if (!result?.mask_overlay) {
      setMaskImage(null);
      return;
    }
    let alive = true;
    const image = new Image();
    image.onload = () => { if (alive) setMaskImage(image); };
    image.onerror = () => { if (alive) setMaskImage(null); };
    image.src = `data:image/png;base64,${result.mask_overlay}`;
    return () => { alive = false; };
  }, [result?.mask_overlay]);

  /** redrawOverlay uses original-image coordinates and lets CSS perform display scaling. */
  const redrawOverlay = () => {
    if (canvasRef.current && result && view === 'overlay') {
      drawTrialOverlay(canvasRef.current, result, result.mask_overlay ? maskImage : null);
    }
  };

  useEffect(redrawOverlay, [result, maskImage, view, imageUrl]);

  /** tensorSelect renders one required output-role selector for custom mappings. */
  const tensorSelect = (field: string) => (
    <Select
      showSearch
      style={{ width: 230 }}
      value={output[field] || undefined}
      options={tensorOptions}
      onChange={(value) => updateOutput({ [field]: value })}
    />
  );

  return (
    <Modal
      title={
        <Space size={8} wrap>
          <span>{t('pages.models.tryTitle', { model: model?.name || '', version: version?.version || '' })}</span>
          {filePath && <Tag color="blue">{filePath.split('/').pop()}</Tag>}
        </Space>
      }
      open={!!model && !!version}
      onCancel={onClose}
      width={960}
      zIndex={1200}
      footer={[
        <Button key="close" onClick={onClose}>{t('common.close')}</Button>,
        <Button
          key="run"
          type="primary"
          loading={loading}
          disabled={!imageReady || !!decoderError || schemaLoading}
          onClick={run}
        >
          {t('pages.models.tryRun')}
        </Button>,
      ]}
    >
      <Space direction="vertical" style={{ width: '100%' }} size="middle">
        <Radio.Group
          value={source}
          optionType="button"
          buttonStyle="solid"
          size="small"
          onChange={(event) => { setSource(event.target.value); resetResult(); }}
          options={[
            { label: t('pages.models.tryFromDataset'), value: 'dataset' },
            { label: t('pages.models.tryFromLocal'), value: 'local' },
          ]}
        />

        {source === 'dataset' ? (
          <Space wrap style={{ width: '100%' }}>
            <Select
              showSearch
              style={{ width: 220 }}
              placeholder={t('pages.models.trySelectDataset')}
              optionFilterProp="label"
              value={dsId || undefined}
              onChange={(value) => {
                setDsId(value); setDsVersion(undefined); setDsPath(''); setFiles([]); setAutoPick(false); resetResult();
              }}
              options={datasets.map((item) => ({ value: item.id, label: item.name }))}
            />
            <Select
              style={{ width: 150 }}
              placeholder={t('pages.models.trySelectVersion')}
              value={dsVersion}
              onChange={(value) => { setDsVersion(value); setDsPath(''); setFiles([]); resetResult(); }}
              options={dsVersions.map((item) => ({ value: item.version, label: `v${item.version}` }))}
            />
            <Select
              showSearch
              loading={loadingFiles}
              style={{ width: 300 }}
              placeholder={t('pages.models.trySelectImage')}
              optionFilterProp="label"
              value={dsPath || undefined}
              onChange={(value) => { setDsPath(value); resetResult(); }}
              options={files.map((item) => ({ value: item.path, label: item.path }))}
            />
          </Space>
        ) : (
          <Upload
            accept="image/*"
            maxCount={1}
            showUploadList={false}
            beforeUpload={(file) => {
              setLocalUrl(URL.createObjectURL(file));
              resetResult();
              return false;
            }}
            fileList={fileList}
            onChange={({ fileList: nextFiles }) => setFileList(nextFiles)}
          >
            <Button icon={<UploadOutlined />}>{t('pages.models.tryPickImage')}</Button>
          </Upload>
        )}

        <div style={{ border: '1px solid #f0f0f0', borderRadius: 6, padding: 12 }}>
          <Space direction="vertical" style={{ width: '100%' }} size={12}>
            <Space wrap align="end">
              <Field label={t('pages.models.tryDecoder')}>
                <Select
                  style={{ width: 180 }}
                  loading={schemaLoading}
                  value={decoderConfig.mode}
                  onChange={changeDecoderMode}
                  options={[
                    { value: 'auto', label: t('pages.models.tryDecoderAuto') },
                    { value: 'builtin', label: t('pages.models.tryDecoderBuiltin') },
                    { value: 'custom', label: t('pages.models.tryDecoderCustom') },
                  ]}
                />
              </Field>
              {decoderConfig.mode === 'builtin' && (
                <Field label={t('pages.models.tryBuiltinDecoder')}>
                  <Select
                    showSearch
                    style={{ width: 330 }}
                    value={decoderConfig.decoder_id || undefined}
                    onChange={(value) => changeDecoderConfig({ mode: 'builtin', decoder_id: value })}
                    options={(schema?.decoders || []).map((item) => ({
                      value: item.id,
                      label: `${item.label} · ${taskLabel(item.task)}${item.matched ? ' ✓' : ''}`,
                    }))}
                  />
                </Field>
              )}
              {decoderConfig.mode === 'custom' && (
                <Field label={t('pages.models.tryTask')}>
                  <Select
                    style={{ width: 220 }}
                    value={decoderConfig.task}
                    onChange={(task: TrialTask) => schema && changeDecoderConfig(createCustomDecoderConfig(schema, task))}
                    options={(['detection', 'instance_segmentation', 'semantic_segmentation', 'classification'] as TrialTask[])
                      .map((task) => ({ value: task, label: taskLabel(task) }))}
                  />
                </Field>
              )}
              {selectedTask === 'classification' && (
                <Field label={t('pages.models.tryTopK')}>
                  <InputNumber
                    min={1}
                    max={100}
                    value={topK}
                    onChange={(value) => { setTopK(value || 5); resetResult(); }}
                  />
                </Field>
              )}
            </Space>

            {decoderConfig.mode === 'auto' && schema?.auto_decoder && (
              <Alert
                type="success"
                showIcon
                message={`${schema.auto_decoder.id} · ${taskLabel(schema.auto_decoder.task)}`}
              />
            )}
            {decoderError === 'decoder mapping required' && decoderConfig.mode === 'auto' && (
              <Alert
                type="warning"
                showIcon
                message={t('pages.models.tryMappingRequired')}
                description={t('pages.models.tryMappingRequiredHint')}
              />
            )}
            {schemaError && <Alert type="error" showIcon message={schemaError} />}
            {decoderConfig.mode !== 'auto' && !!decoderError && decoderError !== 'schema required' && (
              <Alert type="warning" showIcon message={decoderErrorText} />
            )}

            {decoderConfig.mode === 'custom' && input && (
              <>
                <Alert type="info" showIcon message={t('pages.models.tryCustomHint')} />
                <div>
                  <div style={{ fontWeight: 600, marginBottom: 8 }}>{t('pages.models.tryInputPreprocessing')}</div>
                  <Space wrap align="end">
                    <Field label={t('pages.models.tryInputTensor')}>
                      <Select
                        style={{ width: 230 }}
                        value={input.tensor || undefined}
                        options={(schema?.inputs || []).map((item) => ({ value: item.name, label: item.name }))}
                        onChange={(value) => updateInput({ tensor: value })}
                      />
                    </Field>
                    <Field label={t('pages.models.tryWidth')}>
                      <InputNumber min={1} value={input.width} onChange={(value) => updateInput({ width: value || 0 })} />
                    </Field>
                    <Field label={t('pages.models.tryHeight')}>
                      <InputNumber min={1} value={input.height} onChange={(value) => updateInput({ height: value || 0 })} />
                    </Field>
                    <Field label={t('pages.models.tryLayout')}>
                      <Select
                        value={input.layout}
                        options={['NCHW', 'NHWC'].map((value) => ({ value }))}
                        onChange={(value) => updateInput({ layout: value })}
                      />
                    </Field>
                    <Field label={t('pages.models.tryColor')}>
                      <Select
                        value={input.color}
                        options={['RGB', 'BGR'].map((value) => ({ value }))}
                        onChange={(value) => updateInput({ color: value })}
                      />
                    </Field>
                    <Field label={t('pages.models.tryResize')}>
                      <Select
                        style={{ width: 120 }}
                        value={input.resize}
                        options={['letterbox', 'stretch'].map((value) => ({ value }))}
                        onChange={(value) => updateInput({ resize: value })}
                      />
                    </Field>
                    <Field label={t('pages.models.tryScale')}>
                      <InputNumber
                        style={{ width: 130 }}
                        step={0.001}
                        value={input.scale}
                        onChange={(value) => updateInput({ scale: value ?? 1 })}
                      />
                    </Field>
                    <Field label={t('pages.models.tryMean')}>
                      <Input
                        style={{ width: 150 }}
                        value={(input.mean || []).join(',')}
                        onChange={(event) => updateInput({ mean: numberList(event.target.value) })}
                      />
                    </Field>
                    <Field label={t('pages.models.tryStd')}>
                      <Input
                        style={{ width: 150 }}
                        value={(input.std || []).join(',')}
                        onChange={(event) => updateInput({ std: numberList(event.target.value) })}
                      />
                    </Field>
                  </Space>
                </div>

                <div>
                  <div style={{ fontWeight: 600, marginBottom: 8 }}>{t('pages.models.tryOutputMapping')}</div>
                  {decoderConfig.task === 'detection' && (
                    <Space wrap align="end">
                      <Field label={t('pages.models.tryPredictionTensor')}>{tensorSelect('prediction')}</Field>
                      <Field label={t('pages.models.tryMatrixLayout')}>
                        <Select
                          value={output.layout}
                          options={[{ value: 'rows', label: 'N × D' }, { value: 'columns', label: 'D × N' }]}
                          onChange={(value) => updateOutput({ layout: value })}
                        />
                      </Field>
                      <Field label={t('pages.models.tryBBoxStart')}>
                        <InputNumber min={0} value={output.bbox_start} onChange={(value) => updateOutput({ bbox_start: value || 0 })} />
                      </Field>
                      <Field label={t('pages.models.tryBBoxEncoding')}>
                        <Select
                          value={output.bbox_encoding}
                          options={['xyxy', 'cxcywh'].map((value) => ({ value }))}
                          onChange={(value) => updateOutput({ bbox_encoding: value })}
                        />
                      </Field>
                      <Field label={t('pages.models.tryScoreIndex')}>
                        <InputNumber min={0} value={output.score_index} onChange={(value) => updateOutput({ score_index: value || 0 })} />
                      </Field>
                      <Field label={t('pages.models.tryClassIndex')}>
                        <InputNumber min={0} value={output.class_index} onChange={(value) => updateOutput({ class_index: value || 0 })} />
                      </Field>
                      <Field label={t('pages.models.tryCoordinateSpace')}>
                        <Select
                          style={{ width: 130 }}
                          value={output.coordinate_space}
                          options={['pixels', 'normalized'].map((value) => ({ value }))}
                          onChange={(value) => updateOutput({ coordinate_space: value })}
                        />
                      </Field>
                      <Field label={t('pages.models.tryScoreActivation')}>
                        <Select
                          value={output.score_activation}
                          options={['none', 'sigmoid'].map((value) => ({ value }))}
                          onChange={(value) => updateOutput({ score_activation: value })}
                        />
                      </Field>
                    </Space>
                  )}
                  {decoderConfig.task === 'instance_segmentation' && (
                    <Space wrap align="end">
                      {['boxes', 'scores', 'labels', 'masks'].map((field) => (
                        <Field key={field} label={t(`pages.models.tryTensor_${field}`)}>{tensorSelect(field)}</Field>
                      ))}
                      <Field label={t('pages.models.tryBBoxEncoding')}>
                        <Select
                          value={output.bbox_encoding}
                          options={['xyxy', 'cxcywh'].map((value) => ({ value }))}
                          onChange={(value) => updateOutput({ bbox_encoding: value })}
                        />
                      </Field>
                      <Field label={t('pages.models.tryCoordinateSpace')}>
                        <Select
                          value={output.coordinate_space}
                          options={['pixels', 'normalized'].map((value) => ({ value }))}
                          onChange={(value) => updateOutput({ coordinate_space: value })}
                        />
                      </Field>
                      <Field label={t('pages.models.tryMaskActivation')}>
                        <Select
                          value={output.mask_activation}
                          options={['sigmoid', 'none'].map((value) => ({ value }))}
                          onChange={(value) => updateOutput({ mask_activation: value })}
                        />
                      </Field>
                      <Field label={t('pages.models.tryMaskThreshold')}>
                        <InputNumber
                          min={0} max={1} step={0.05} value={output.mask_threshold}
                          onChange={(value) => updateOutput({ mask_threshold: value ?? 0.5 })}
                        />
                      </Field>
                    </Space>
                  )}
                  {(decoderConfig.task === 'semantic_segmentation' || decoderConfig.task === 'classification') && (
                    <Space wrap align="end">
                      <Field label={t('pages.models.tryOutputTensor')}>{tensorSelect('output')}</Field>
                      {decoderConfig.task === 'semantic_segmentation' && (
                        <Field label={t('pages.models.tryOutputFormat')}>
                          <Select
                            value={output.format}
                            options={['logits', 'class_ids'].map((value) => ({ value }))}
                            onChange={(value) => updateOutput({
                              format: value,
                              class_axis: value === 'class_ids' ? undefined : (output.class_axis ?? 1),
                            })}
                          />
                        </Field>
                      )}
                      <Field label={t('pages.models.tryClassAxis')}>
                        <InputNumber
                          value={output.class_axis}
                          placeholder={decoderConfig.task === 'semantic_segmentation' && output.format === 'class_ids' ? 'NHW' : undefined}
                          onChange={(value) => updateOutput({
                            class_axis: value ?? (
                              decoderConfig.task === 'semantic_segmentation' && output.format === 'class_ids'
                                ? undefined
                                : 0
                            ),
                          })}
                        />
                      </Field>
                      {decoderConfig.task === 'semantic_segmentation' ? (
                        <Field label={t('pages.models.tryIgnoreIndex')}>
                          <InputNumber value={output.ignore_index} onChange={(value) => updateOutput({ ignore_index: value ?? -1 })} />
                        </Field>
                      ) : (
                        <Field label={t('pages.models.tryActivation')}>
                          <Select
                            value={output.activation}
                            options={['softmax', 'sigmoid', 'none'].map((value) => ({ value }))}
                            onChange={(value) => updateOutput({ activation: value })}
                          />
                        </Field>
                      )}
                    </Space>
                  )}
                  <div style={{ marginTop: 10 }}>
                    <Field label={t('pages.models.tryLabels')}>
                      <Input
                        style={{ width: '100%' }}
                        placeholder="person,car,bicycle"
                        value={(decoderConfig.labels || []).join(',')}
                        onChange={(event) => changeDecoderConfig({
                          ...decoderConfig,
                          labels: event.target.value.split(',').map((item) => item.trim()).filter(Boolean),
                        })}
                      />
                    </Field>
                  </div>
                </div>
              </>
            )}
          </Space>
        </div>

        <Space wrap align="center">
          <span style={{ minWidth: 230 }}>
            <span style={{ marginRight: 8 }}>{t('pages.models.tryConf')}: {conf.toFixed(2)}</span>
            <Slider
              min={0.05} max={0.95} step={0.05} value={conf}
              onChange={(value) => { setConf(value); resetResult(); }}
              style={{ width: 150, display: 'inline-flex', verticalAlign: 'middle' }}
            />
          </span>
          <span>
            <span style={{ marginRight: 8 }}>Grad-CAM</span>
            <Tooltip title={hasPt ? undefined : t('pages.models.tryNoPt')}>
              <Switch
                checked={cam}
                onChange={(value) => { setCam(value); resetResult(); }}
                disabled={!hasPt}
              />
            </Tooltip>
          </span>
        </Space>
        {!hasPt && <Tag color="orange">{t('pages.models.tryNoPt')}</Tag>}

        <Spin spinning={loading || schemaLoading} tip={loading ? t('pages.models.tryRunning') : undefined}>
          {!srcUrl ? (
            <Empty description={t('pages.models.tryEmpty')} style={{ padding: 32 }} />
          ) : (
            <div style={{ position: 'relative', display: 'inline-block', maxWidth: '100%', lineHeight: 0 }}>
              <img
                src={srcUrl}
                alt={t('pages.models.tryPreviewAlt')}
                crossOrigin="anonymous"
                onLoad={redrawOverlay}
                style={{ maxWidth: '100%', display: 'block', borderRadius: 4 }}
              />
              {view === 'overlay' && result && (
                <canvas
                  ref={canvasRef}
                  style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }}
                />
              )}
            </div>
          )}
        </Spin>

        {result && (
          <Space direction="vertical" style={{ width: '100%' }} size={8}>
            <Space wrap>
              {heatUrl && (
                <Radio.Group
                  value={view}
                  onChange={(event) => setView(event.target.value)}
                  size="small"
                  optionType="button"
                  buttonStyle="solid"
                  options={[
                    { label: t('pages.models.tryViewOverlay'), value: 'overlay' },
                    { label: t('pages.models.tryViewHeatmap'), value: 'heatmap' },
                  ]}
                />
              )}
              <Tag>{t('pages.models.tryLatency')}: {result.time_ms} ms</Tag>
              {result.decoder && <Tag color="blue">{result.decoder.id} · {taskLabel(resultKind)}</Tag>}
              {result.cam_target && <Tag color="volcano">CAM: {result.cam_target.label}</Tag>}
              {result.served_artifact === 'onnx_fallback' && (
                <Tooltip title={t('pages.models.tryOnnxFallbackTip')}>
                  <Tag color="gold">{t('pages.models.tryOnnxFallback')}</Tag>
                </Tooltip>
              )}
            </Space>

            {resultKind === 'classification' && (
              <div>
                <div style={{ fontWeight: 600, marginBottom: 6 }}>{t('pages.models.tryTopK')}</div>
                <Space wrap>
                  {topKResults.map((item, index) => (
                    <Tag key={`${item.class_index}-${index}`} color={index === 0 ? 'blue' : 'default'}>
                      {item.label} {item.score.toFixed(3)}
                    </Tag>
                  ))}
                </Space>
              </div>
            )}
            {resultKind === 'detection' && (
              <span style={{ fontSize: 12, color: '#888' }}>
                {t('pages.models.tryDetections')}: {result.boxes?.length || 0}
              </span>
            )}
            {resultKind === 'instance_segmentation' && (
              <span style={{ fontSize: 12, color: '#888' }}>
                {t('pages.models.tryInstances')}: {result.polygons?.length || 0}
              </span>
            )}
            {resultKind === 'semantic_segmentation' && (
              <Space wrap>
                {(result.mask_legend || []).map((item) => (
                  <Tag key={item.cls} color={`rgb(${item.color.slice(0, 3).join(',')})`}>{item.label}</Tag>
                ))}
              </Space>
            )}
            {!['classification', 'detection', 'instance_segmentation', 'semantic_segmentation'].includes(resultKind || '') && (
              <Alert type="info" message={t('pages.models.tryUnsupportedResult', { kind: resultKind })} />
            )}
            {result.mask_note && <Alert type="warning" message={result.mask_note} />}
          </Space>
        )}
      </Space>
    </Modal>
  );
};

export default TryModelModal;
