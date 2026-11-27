// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

import http from 'node:http';
import { readFileSync } from 'node:fs';

const PORT = 8000;
const now = (days = 0) => new Date(Date.now() - days * 86400000).toISOString();

const user = {
  id: 'dev-user',
  username: 'dev-admin',
  email: 'admin@nereidai.local',
  full_name: 'Dev Admin',
  roles: ['admin', 'user'],
  tenant_id: 'default',
  is_active: true,
  created_at: now(30),
  updated_at: now(1),
};

const datasets = [
  {
    id: 'ds-001',
    name: 'coco128-val',
    description: 'COCO128 validation subset for detection',
    tenant_id: 'default',
    version: 3,
    size_bytes: 2147483648,
    file_count: 128,
    status: 'active',
    created_by: user.username,
    created_at: now(14),
    updated_at: now(2),
  },
  {
    id: 'ds-002',
    name: 'mnist-train',
    description: 'Handwritten digits for classification',
    tenant_id: 'default',
    version: 1,
    size_bytes: 12058624,
    file_count: 60000,
    status: 'active',
    created_by: user.username,
    created_at: now(9),
    updated_at: now(3),
  },
  {
    id: 'ds-003',
    name: 'custom-pose-data',
    description: 'Pose estimation keypoint dataset',
    tenant_id: 'default',
    version: 2,
    size_bytes: 536870912,
    file_count: 5120,
    status: 'draft',
    created_by: user.username,
    created_at: now(5),
    updated_at: now(1),
  },
];

const versions = (datasetId) => [
  {
    id: `${datasetId}-v3`,
    dataset_id: datasetId,
    version: 3,
    description: 'Added new classes',
    size_bytes: 2147483648,
    file_count: 128,
    commit_hash: 'a1b2c3d4e5f6',
    created_by: user.username,
    created_at: now(2),
  },
  {
    id: `${datasetId}-v2`,
    dataset_id: datasetId,
    version: 2,
    description: 'Cleaned annotations',
    size_bytes: 2048000000,
    file_count: 126,
    commit_hash: 'f6e5d4c3b2a1',
    created_by: user.username,
    created_at: now(8),
  },
  {
    id: `${datasetId}-v1`,
    dataset_id: datasetId,
    version: 1,
    description: 'Initial import',
    size_bytes: 1900000000,
    file_count: 120,
    commit_hash: '0000aaaa1111',
    created_by: user.username,
    created_at: now(14),
  },
];

const annotations = [
  {
    id: 'at-001',
    name: 'coco128-bbox-labeling',
    description: 'Label bounding boxes for detection',
    dataset_id: 'ds-001',
    dataset_version: 3,
    status: 'in_progress',
    progress: 46,
    total_items: 128,
    annotated_items: 59,
    assignee_id: 'annotator-a',
    created_by: user.username,
    created_at: now(6),
    updated_at: now(0),
  },
  {
    id: 'at-002',
    name: 'mnist-classify-review',
    description: 'Review auto-generated labels',
    dataset_id: 'ds-002',
    dataset_version: 1,
    status: 'completed',
    progress: 100,
    total_items: 60000,
    annotated_items: 60000,
    assignee_id: 'annotator-b',
    created_by: user.username,
    created_at: now(10),
    updated_at: now(4),
  },
  {
    id: 'at-003',
    name: 'pose-keypoint-check',
    description: 'Verify keypoint annotations',
    dataset_id: 'ds-003',
    dataset_version: 2,
    status: 'pending',
    progress: 0,
    total_items: 5120,
    annotated_items: 0,
    assignee_id: null,
    created_by: user.username,
    created_at: now(1),
    updated_at: now(1),
  },
];

const trainingJobs = [
  {
    id: 'tr-001',
    name: 'yolov8s-coco-finetune',
    description: 'Fine-tune YOLOv8s on custom data',
    model_id: 'yolov8s',
    dataset_id: 'ds-001',
    dataset_version: 3,
    config: {
      epochs: 100,
      batch_size: 32,
      learning_rate: 0.001,
      optimizer: 'adamw',
      loss_function: 'ciou',
      device: 'cuda',
      mixed_precision: true,
      distributed: false,
      num_workers: 4,
      checkpoint_interval: 10,
      early_stopping_patience: 15,
    },
    status: 'running',
    progress: 46,
    current_epoch: 46,
    total_epochs: 100,
    metrics: { train_loss: [0.9, 0.7, 0.5, 0.4], val_loss: [1.0, 0.8, 0.6, 0.5], learning_rate: [0.001, 0.001, 0.001, 0.001], epoch_times: [120, 118, 121, 119] },
    created_by: user.username,
    created_at: now(2),
    updated_at: now(0),
    started_at: now(1),
  },
  {
    id: 'tr-002',
    name: 'mnist-cnn-baseline',
    description: 'Baseline CNN for digit classification',
    model_id: 'lenet5',
    dataset_id: 'ds-002',
    dataset_version: 1,
    config: {
      epochs: 50,
      batch_size: 64,
      learning_rate: 0.01,
      optimizer: 'sgd',
      loss_function: 'cross_entropy',
      device: 'cuda',
      mixed_precision: false,
      distributed: false,
      num_workers: 2,
      checkpoint_interval: 5,
    },
    status: 'completed',
    progress: 100,
    current_epoch: 50,
    total_epochs: 50,
    metrics: { train_loss: [0.5, 0.2, 0.1], val_loss: [0.4, 0.15, 0.08], train_accuracy: [0.9, 0.97, 0.99], val_accuracy: [0.88, 0.96, 0.985], learning_rate: [0.01, 0.01, 0.01], epoch_times: [30, 28, 29] },
    created_by: user.username,
    created_at: now(8),
    updated_at: now(3),
    started_at: now(8),
    completed_at: now(3),
  },
  {
    id: 'tr-003',
    name: 'pose-lite-training',
    description: 'Lightweight pose model on edge data',
    model_id: 'pose-lite',
    dataset_id: 'ds-003',
    dataset_version: 2,
    config: {
      epochs: 80,
      batch_size: 16,
      learning_rate: 0.0005,
      optimizer: 'adam',
      loss_function: 'mse',
      device: 'cpu',
      mixed_precision: false,
      distributed: false,
      num_workers: 2,
      checkpoint_interval: 20,
    },
    status: 'failed',
    progress: 12,
    current_epoch: 10,
    total_epochs: 80,
    error_message: 'CUDA out of memory during epoch 11',
    created_by: user.username,
    created_at: now(1),
    updated_at: now(0),
    started_at: now(1),
  },
];

const optimizationJobs = [
  {
    id: 'op-001',
    name: 'yolov8s-trt-int8',
    description: 'TensorRT INT8 quantization',
    source_model_id: 'tr-001',
    target_device: 'jetson-orin-nano',
    quantization: { method: 'static', dtype: 'int8', calibration_dataset_id: 'ds-001', calibration_samples: 512 },
    status: 'running',
    progress: 55,
    original_size_bytes: 22464000,
    optimized_size_bytes: 5620000,
    speedup_ratio: 2.4,
    accuracy_drop: 0.8,
    created_by: user.username,
    created_at: now(2),
    updated_at: now(0),
    started_at: now(1),
  },
  {
    id: 'op-002',
    name: 'lenet5-onnx-fp16',
    description: 'FP16 conversion for edge deployment',
    source_model_id: 'tr-002',
    target_device: 'rk3588',
    quantization: { method: 'dynamic', dtype: 'float16' },
    status: 'completed',
    progress: 100,
    original_size_bytes: 2140000,
    optimized_size_bytes: 1070000,
    speedup_ratio: 1.8,
    accuracy_drop: 0.1,
    created_by: user.username,
    created_at: now(6),
    updated_at: now(4),
    started_at: now(6),
    completed_at: now(4),
  },
  {
    id: 'op-003',
    name: 'pose-lite-openvino',
    description: 'OpenVINO IR conversion',
    source_model_id: 'tr-003',
    target_device: 'intel-arc',
    quantization: { method: 'dynamic', dtype: 'float16' },
    status: 'failed',
    progress: 30,
    original_size_bytes: 5120000,
    error_message: 'Unsupported op: InstanceNormalization',
    created_by: user.username,
    created_at: now(1),
    updated_at: now(0),
    started_at: now(1),
  },
];

const evaluationJobs = [
  {
    id: 'ev-001',
    name: 'yolov8s-coco-val-eval',
    description: 'Evaluate fine-tuned model on val set',
    model_id: 'tr-001',
    dataset_id: 'ds-001',
    dataset_version: 3,
    metrics: ['mAP', 'precision', 'recall'],
    status: 'running',
    progress: 40,
    created_by: user.username,
    created_at: now(1),
    updated_at: now(0),
    started_at: now(1),
  },
  {
    id: 'ev-002',
    name: 'lenet5-mnist-eval',
    description: 'Baseline accuracy evaluation',
    model_id: 'tr-002',
    dataset_id: 'ds-002',
    dataset_version: 1,
    metrics: ['accuracy', 'precision', 'recall', 'f1'],
    status: 'completed',
    progress: 100,
    results: {
      accuracy: 0.985,
      precision: 0.984,
      recall: 0.983,
      f1_score: 0.983,
      per_class_metrics: {
        '0': { precision: 0.99, recall: 0.99, f1_score: 0.99, support: 980 },
        '1': { precision: 0.98, recall: 0.99, f1_score: 0.985, support: 1135 },
        '2': { precision: 0.985, recall: 0.98, f1_score: 0.982, support: 1032 },
      },
      inference_time_ms: 0.8,
      throughput_fps: 1250,
    },
    created_by: user.username,
    created_at: now(6),
    updated_at: now(3),
    started_at: now(6),
    completed_at: now(3),
  },
];

const models = [
  {
    id: 'md-001',
    name: 'yolov8s',
    task_type: 'detection',
    framework: 'pytorch',
    latest_version: 'v1.2.0',
    stage: 'deployed',
    created_by: user.username,
    created_at: now(20),
    updated_at: now(2),
  },
  {
    id: 'md-002',
    name: 'lenet5',
    task_type: 'classification',
    framework: 'onnx',
    latest_version: 'v2.0.0',
    stage: 'ready',
    created_by: user.username,
    created_at: now(15),
    updated_at: now(4),
  },
  {
    id: 'md-003',
    name: 'pose-lite',
    task_type: 'pose_estimation',
    framework: 'openvino',
    latest_version: 'v0.9.0',
    stage: 'draft',
    created_by: user.username,
    created_at: now(7),
    updated_at: now(1),
  },
];

const modelVersions = [
  { version: 'v1.2.0', stage: 'deployed', artifacts_count: 3, created_at: now(2) },
  { version: 'v1.1.0', stage: 'archived', artifacts_count: 2, created_at: now(6) },
  { version: 'v1.0.0', stage: 'archived', artifacts_count: 2, created_at: now(12) },
];

const mlBackends = [
  {
    id: 'mb-001',
    name: 'gpu-cluster-a',
    url: 'http://ml-backend-a:8000',
    description: 'Primary CUDA inference backend',
    status: 'healthy',
    health_check_interval: 30,
    last_health_check: now(0),
    capabilities: ['cuda', 'tensorrt', 'batch'],
    created_by: user.username,
    created_at: now(30),
    updated_at: now(0),
  },
  {
    id: 'mb-002',
    name: 'jetson-edge-nano',
    url: 'http://jetson-edge:8000',
    description: 'Edge device for on-device inference',
    status: 'unhealthy',
    health_check_interval: 60,
    last_health_check: now(0),
    capabilities: ['tensorrt'],
    created_by: user.username,
    created_at: now(28),
    updated_at: now(0),
  },
  {
    id: 'mb-003',
    name: 'rk3588-batch',
    url: 'http://rk3588-cluster:8000',
    description: 'Rockchip batch inference pool',
    status: 'maintenance',
    health_check_interval: 120,
    last_health_check: now(0),
    capabilities: ['rknn', 'openvino'],
    created_by: user.username,
    created_at: now(25),
    updated_at: now(0),
  },
];

const stats = {
  datasets_count: datasets.length,
  models_count: models.length,
  training_jobs_count: trainingJobs.length,
  annotation_tasks_count: annotations.length,
  recent_jobs: [
    { id: 'tr-001', name: 'yolov8s-coco-finetune', type: 'training', status: 'running', created_at: now(2) },
    { id: 'op-001', name: 'yolov8s-trt-int8', type: 'optimization', status: 'running', created_at: now(2) },
    { id: 'ev-002', name: 'lenet5-mnist-eval', type: 'evaluation', status: 'completed', created_at: now(6) },
    { id: 'tr-002', name: 'mnist-cnn-baseline', type: 'training', status: 'completed', created_at: now(8) },
  ],
};

const oidcConfig = {
  issuer: 'https://identity.example.com',
  client_id: 'nereidforge-dev',
  redirect_uri: 'http://localhost:5173/callback',
  scope: 'openid profile email roles',
  pkce_method: 'S256',
};

const paginate = (items, params) => {
  const page = Number(params.get('page') || 1);
  const page_size = Number(params.get('page_size') || 20);
  const start = (page - 1) * page_size;
  return {
    items: items.slice(start, start + page_size),
    total: items.length,
    page,
    page_size,
    total_pages: Math.max(1, Math.ceil(items.length / page_size)),
  };
};

const json = (res, status, body) => {
  res.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8' });
  res.end(JSON.stringify(body));
};

const server = http.createServer((req, res) => {
  const url = new URL(req.url, `http://${req.headers.host}`);
  const { pathname, searchParams } = url;
  const method = req.method;

  if (method === 'GET') {
    if (pathname === '/dashboard/stats') return json(res, 200, stats);
    if (pathname === '/auth/me') return json(res, 200, user);
    if (pathname === '/auth/oidc/config') return json(res, 200, oidcConfig);
    if (pathname === '/datasets') return json(res, 200, paginate(datasets, searchParams));
    if (/^\/datasets\/[^/]+\/versions$/.test(pathname)) return json(res, 200, paginate(versions('ds-001'), searchParams));
    if (pathname === '/annotations') return json(res, 200, paginate(annotations, searchParams));
    if (pathname === '/training') return json(res, 200, paginate(trainingJobs, searchParams));
    if (pathname === '/optimization') return json(res, 200, paginate(optimizationJobs, searchParams));
    if (pathname === '/evaluation') return json(res, 200, paginate(evaluationJobs, searchParams));
    if (pathname === '/models') return json(res, 200, paginate(models, searchParams));
    if (/^\/models\/[^/]+\/versions$/.test(pathname)) return json(res, 200, paginate(modelVersions, searchParams));
    if (pathname === '/ml-backends') return json(res, 200, paginate(mlBackends, searchParams));
    if (/^\/training\/[^/]+$/.test(pathname)) return json(res, 200, trainingJobs[0]);
    if (/^\/training\/[^/]+\/metrics$/.test(pathname)) return json(res, 200, trainingJobs[0].metrics);
    if (/^\/training\/[^/]+\/logs$/.test(pathname)) return json(res, 200, 'Epoch 46/100\n  train_loss: 0.41\n  val_loss: 0.48\nEpoch 47/100 (running)');
    if (/^\/training\/[^/]+\/checkpoints$/.test(pathname)) return json(res, 200, [
      { epoch: 40, path: 'gs://models/tr-001/epoch-40.pt', metrics: { val_loss: 0.51 } },
      { epoch: 30, path: 'gs://models/tr-001/epoch-30.pt', metrics: { val_loss: 0.55 } },
    ]);
    if (/^\/optimization\/[^/]+$/.test(pathname)) return json(res, 200, optimizationJobs[0]);
    if (/^\/optimization\/[^/]+\/logs$/.test(pathname)) return json(res, 200, 'Quantizing layer conv_1: int8\nCalibration batch 200/512');
    if (/^\/evaluation\/[^/]+$/.test(pathname)) return json(res, 200, evaluationJobs[0]);
    if (/^\/evaluation\/[^/]+\/logs$/.test(pathname)) return json(res, 200, 'Evaluating on val set: 128 images');
    if (/^\/evaluation\/[^/]+\/results$/.test(pathname)) return json(res, 200, evaluationJobs[1].results);
    if (/^\/ml-backends\/[^/]+\/health$/.test(pathname)) return json(res, 200, { status: 'healthy', latency_ms: 12 });
    if (/^\/ml-backends\/[^/]+\/capabilities$/.test(pathname)) return json(res, 200, ['cuda', 'tensorrt']);
    return json(res, 200, paginate([], searchParams));
  }

  if (method === 'POST' || method === 'PUT' || method === 'PATCH' || method === 'DELETE') {
    let body = '';
    req.on('data', (chunk) => (body += chunk));
    req.on('end', () => {
      let parsed = {};
      try {
        parsed = body ? JSON.parse(body) : {};
      } catch {
        parsed = { _raw: body.slice(0, 200) };
      }
      return json(res, method === 'POST' ? 201 : 200, { ...parsed, id: `mock-${Date.now()}`, created_at: now(0), updated_at: now(0) });
    });
    return;
  }

  return json(res, 405, { message: 'method not allowed' });
});

server.listen(PORT, () => {
  console.log(`[mock-api] listening on http://localhost:${PORT}`);
});
