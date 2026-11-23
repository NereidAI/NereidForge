// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

export interface User {
  id: string;
  username: string;
  email: string;
  full_name: string;
  roles: string[];
  tenant_id: string;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

export interface Tenant {
  id: string;
  name: string;
  display_name: string;
  description?: string;
  is_active: boolean;
  admin_username?: string;
  namespace?: string;
  created_at: string;
  updated_at: string;
}

export interface Dataset {
  id: string;
  name: string;
  description?: string;
  tenant_id: string;
  format?: string;
  // Training task types this corpus can be used for (inferred server-side
  // from its format and the annotation projects its versions came from).
  task_types?: string[];
  version: number;
  size_bytes: number;
  file_count: number;
  status: DatasetStatus;
  created_by: string;
  created_at: string;
  updated_at: string;
  uri?: string;
}

export enum DatasetStatus {
  DRAFT = 'draft',
  ACTIVE = 'active',
  ARCHIVED = 'archived',
}

export interface DatasetVersion {
  id: string;
  dataset_id: string;
  version: number;
  description?: string;
  uri?: string;
  format?: string;
  size_bytes: number;
  file_count: number;
  commit_hash: string;
  created_by: string;
  created_at: string;
  parent_version_id?: string;
  source_project_id?: string;
  is_annotated?: boolean;
  is_draft?: boolean;
  has_videos?: boolean;
}

export interface VersionFile {
  path: string;
  name: string;
  size: number;
  ext: string;
  is_text: boolean;
  is_media: boolean;
}

export interface AnnotationTask {
  id: string;
  name: string;
  description?: string;
  annotation_tool?: string;
  task_type?: string;
  schema?: Record<string, unknown>;
  dataset_id: string;
  dataset_version: number;
  status: AnnotationTaskStatus;
  progress: number;
  total_items: number;
  annotated_items: number;
  assignee_id?: string;
  assignee_name?: string;
  external_id?: string;
  external_url?: string;
  import_uri?: string;
  // Per-project pre-annotation model preference {model_name, model_version}.
  preannotate_model?: { model_name?: string; model_version?: string } | null;
  created_by: string;
  created_at: string;
  updated_at: string;
}

export enum AnnotationTaskStatus {
  PENDING = 'pending',
  IN_PROGRESS = 'in_progress',
  COMPLETED = 'completed',
  FAILED = 'failed',
  CANCELLED = 'cancelled',
}

export interface AnnotationItem {
  id: string;
  item_id: string;
  item_index: number;
  status: 'pending' | 'annotated' | 'skipped';
  annotator_id?: string;
  annotation: unknown;
  annotations: unknown;
  annotated_at?: string;
  updated_at?: string;
}

export interface LSTaskView {
  id: number;
  inner_id: number;
  data: Record<string, unknown>;
  image_url: string;
  is_labeled: boolean;
  annotations: Array<{
    id: number;
    result: unknown[];
    created_at?: string;
  }>;
  predictions: Array<{
    id: number;
    result: unknown[];
    score?: number;
    model_version?: string;
  }>;
  created_at: string;
  updated_at: string;
}

export interface CVATTask {
  id: number;
  name: string;
  size: number;
  mode?: string;
  status: string;
  progress?: number;
  start_frame?: number;
  stop_frame?: number;
  width: number;
  height: number;
  updated_date?: string;
  data: Record<string, unknown>;
  annotations: Array<{
    id: number;
    type: string;
    label_id: number;
    frame: number;
    points: number[];
    occluded?: boolean;
    outside?: boolean;
    track_id?: number;
  }>;
}

export interface CVATTaskDetail extends CVATTask {}

// -- CVAT native editor types ----------------------------------------------

export interface CVATTrackKf {
  frame: number;
  points: number[]; // [x1, y1, x2, y2] absolute pixels (rectangle)
  outside: boolean;
  occluded: boolean;
  // polygon keyframes: shape='polygon' with flat vertex list [x1,y1,x2,y2,...]
  shape?: 'rectangle' | 'polygon';
  poly?: number[];
}

export interface CVATEditorTrack {
  id: number;
  label_id: number;
  frame: number;
  keyframes: CVATTrackKf[];
}

export interface CVATAnnotationDetail {
  task: {
    id: number;
    name: string;
    mode: string;
    width: number;
    height: number;
    frame_count: number;
    size: number;
  };
  labels: Array<{ id: number; name: string }>;
  tracks: CVATEditorTrack[];
  shapes: Array<{
    id: number;
    label_id: number;
    frame: number;
    points: number[];
    type: string;
  }>;
}

export interface CVATTrackInput {
  id?: number;
  label_id: number;
  keyframes: CVATTrackKf[];
}

export interface AsyncTask {
  id: string;
  tenant_id: string;
  type: string;
  status: string;
  progress: number;
  params: Record<string, unknown>;
  error_message?: string;
  created_by: string;
  created_at: string;
  updated_at?: string;
}

// Cluster-side runtime objects for one training job (GET /training/jobs/{id}/runtime).
export interface TrainingRuntime {
  kfp_run_id: string;
  trainjobs: Array<{ name: string; state: string; created?: string }>;
  pods: Array<{ name: string; phase: string; node?: string; job?: string; started?: string; finished?: string }>;
}

export interface TrainingJob {
  id: string;
  name: string;
  description?: string;
  model_id: string;
  dataset_id: string;
  dataset_version: number;
  config: TrainingConfig;
  status: JobStatus;
  progress: number;
  current_epoch: number;
  total_epochs: number;
  metrics?: TrainingMetrics;
  error_message?: string;
  created_by: string;
  created_at: string;
  updated_at: string;
  started_at?: string;
  completed_at?: string;
  // Fields returned by the API but not previously typed.
  kfp_run_id?: string;
  trainer_image?: string;
  tenant_id?: string;
  params?: Record<string, unknown>;
  results?: { message?: string; metrics?: Record<string, unknown> };
  // Trained artifact auto-registered on completion (populated by the worker
  // via results.registered_model). model_id is the requested target model;
  // legacy jobs may only carry the source model in this field.
  output_model_id?: string;
  output_model_name?: string;
  output_model_version?: string;
}

export interface TrainingConfig {
  epochs: number;
  batch_size: number;
  learning_rate: number;
  optimizer: string;
  loss_function: string;
  device: string;
  mixed_precision: boolean;
  distributed: boolean;
  num_workers: number;
  checkpoint_interval: number;
  early_stopping_patience?: number;
  base_model?: string;
  // Vision architecture and initialization are independent wire fields. The
  // legacy base_model value remains for old jobs and external API consumers.
  architecture?: string;
  pretrained?: boolean;
  base_weights_uri?: string;
  // Explicit source model id when the user picked a registered version to
  // continue from; the worker versions output under that exact model row.
  base_model_id?: string;
  split_ratio?: number;
  imgsz?: number;
  batch?: number;
  // Optional DataLoader parallelism; omission preserves each trainer's default.
  workers?: number;
  trainer_image?: string;
  node?: string;
  // GPU vendor selector (cpu | nvidia | amd | intel); drives the per-vendor
  // trainer image and scheduler extended resource, see backend accelerators.py.
  accelerator?: string;
  gpu_count?: number;
  num_proc_per_node?: number;
  task_type?: string;
  // YOLO detection augmentation / regularization (ultralytics hyperparams).
  // preset is one of off | light | medium | strong; the remaining optional
  // fields override the preset for advanced/custom tuning.
  augmentation?: AugmentationConfig;
  // LLM/VLM LoRA finetuning hyperparameters (mapped to the backend _llm_args
  // block / LLaMA-Factory and the native transformers XPU entrypoints).
  llm?: LlmTrainConfig;
}

export interface LlmTrainConfig {
  finetuning_type?: string;
  stage?: string;
  base_model_uri?: string;
  template?: string;
  cutoff_len?: number;
  num_train_epochs?: number;
  learning_rate?: number;
  per_device_train_batch_size?: number;
  gradient_accumulation_steps?: number;
  lora_rank?: number;
  lora_alpha?: number;
  lora_dropout?: number;
  lora_target?: string;
  quantization_bit?: number;
}

export interface AugmentationConfig {
  preset?: 'off' | 'light' | 'medium' | 'strong';
  fliplr?: number;
  flipud?: number;
  degrees?: number;
  shear?: number;
  translate?: number;
  scale?: number;
  perspective?: number;
  hsv_h?: number;
  hsv_s?: number;
  hsv_v?: number;
  mosaic?: number;
  mixup?: number;
  copy_paste?: number;
  close_mosaic?: number;
  patience?: number;
  freeze?: number;
  weight_decay?: number;
  // Production controls (validated against train.py's AUG_SPECS server-side).
  optimizer?: string;
  lr0?: number;
  lrf?: number;
  cos_lr?: boolean;
  warmup_epochs?: number;
  warmup_momentum?: number;
  warmup_bias_lr?: number;
  multi_scale?: boolean;
  dropout?: number;
  seed?: number;
  deterministic?: boolean;
  amp?: boolean;
  save_period?: number;
}

export interface TrainingMetrics {
  train_loss: number[];
  val_loss: number[];
  train_accuracy?: number[];
  val_accuracy?: number[];
  learning_rate: number[];
  epoch_times: number[];
  // Artifact URIs written by the training pipeline (s3:// paths).
  // Present once training finishes and the model files are uploaded.
  weights_uri?: string;
  onnx_uri?: string;
  [key: string]: unknown;
}

export interface OptimizationJob {
  id: string;
  name: string;
  description?: string;
  batch_id?: string;
  status: JobStatus;
  progress: number;
  params?: Record<string, any>;
  config?: Record<string, any>;
  metrics?: Record<string, any>;
  results?: {
    message?: string;
    metrics?: Record<string, any>;
    report?: GateReport;
    gate_result?: string;
    gate_override?: { by: string; reason: string; at: string; registered_version: string };
  };
  error_message?: string;
  kfp_run_id?: string;
  created_by: string;
  created_at: string;
  updated_at: string;
}

export interface GateReport {
  job_type?: string;
  job_id?: string;
  task_type?: string;
  model_uri?: string;
  baseline_model_uri?: string;
  dataset_uri?: string;
  metrics?: Record<string, any>;
  baseline_metrics?: Record<string, any>;
  delta?: Record<string, number>;
  gates?: Record<string, { hard: boolean; threshold: number; value: number; passed: boolean; detail?: string }>;
  gate_result?: 'PASS' | 'YELLOW' | 'FAIL';
  hard_gates_passed?: boolean;
  benchmark?: Record<string, number>;
  environment?: Record<string, any>;
  timestamp?: string;
}

export interface EvaluationJob {
  id: string;
  name: string;
  description?: string;
  status: JobStatus;
  progress: number;
  params?: Record<string, any>;
  metrics?: Record<string, any>;
  results?: {
    message?: string;
    metrics?: Record<string, any>;
    report?: GateReport;
    gate_result?: string;
  };
  error_message?: string;
  kfp_run_id?: string;
  created_by: string;
  created_at: string;
  updated_at: string;
}

export interface EvaluationResults {
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1_score?: number;
  mAP?: number;
  confusion_matrix?: number[][];
  per_class_metrics?: Record<string, ClassMetrics>;
  inference_time_ms?: number;
  throughput_fps?: number;
}

export interface ClassMetrics {
  precision: number;
  recall: number;
  f1_score: number;
  support: number;
}

export interface Model {
  id: string;
  name: string;
  description?: string;
  framework: ModelFramework;
  task_type: ModelTaskType;
  tenant_id: string;
  created_by: string;
  created_at: string;
  updated_at: string;
  versions: ModelVersion[];
}

export interface ModelSummary {
  model_name: string;
  task: string;
  framework: string;
  latest_version: string;
  stage: string;
}

export interface Artifact {
  artifact_id: string;
  model_name: string;
  version: string;
  hardware_target: string;
  format: string;
  precision: string;
  status: string;
  metrics: Record<string, unknown>;
  benchmark: Record<string, unknown>;
  created_at: string;
}

// Entry in the user-managed S3 base-model pool (base-models/cv|llm).
export interface BaseModelPoolEntry {
  id: string;
  kind: 'cv' | 'llm';
  name: string;
  uri: string;
  size_bytes: number;
  framework: string; // pytorch | tensorflow | onnx
  task: string; // detection | segmentation | classification | llm | vlm
  status: string; // original | develop | production
  description: string;
  source: string; // manual | s3-reconcile | seed
  uploaded_by: string;
  created_at: string;
}

// TrialDecoderConfig is the per-artifact execution contract selected in the
// image-trial modal. Custom mappings are declarative data, never executable code.
export type TrialTask = 'detection' | 'instance_segmentation' | 'semantic_segmentation' | 'classification';
export type TrialResultKind = TrialTask | (string & {});

export interface TrialInputMapping {
  tensor: string;
  width: number;
  height: number;
  layout: 'NCHW' | 'NHWC';
  color: 'RGB' | 'BGR';
  resize: 'letterbox' | 'stretch';
  scale: number;
  mean?: number[];
  std?: number[];
}

export interface TrialDetectionMapping {
  prediction: string;
  layout: 'rows' | 'columns';
  bbox_start: number;
  bbox_encoding: 'xyxy' | 'cxcywh';
  score_index: number;
  class_index: number;
  coordinate_space: 'pixels' | 'normalized';
  score_activation: 'none' | 'sigmoid';
}

export interface TrialInstanceSegmentationMapping {
  boxes: string;
  scores: string;
  labels: string;
  masks: string;
  bbox_encoding: 'xyxy' | 'cxcywh';
  coordinate_space: 'pixels' | 'normalized';
  mask_activation: 'sigmoid' | 'none';
  mask_threshold: number;
}

export interface TrialSemanticSegmentationMapping {
  output: string;
  format: 'logits' | 'class_ids';
  // Required for logits and rank-4 N1HW maps; omitted for batch-1 NHW class IDs.
  class_axis?: number;
  ignore_index?: number;
}

export interface TrialClassificationMapping {
  output: string;
  class_axis: number;
  activation: 'softmax' | 'sigmoid' | 'none';
}

export type TrialOutputMapping = TrialDetectionMapping | TrialInstanceSegmentationMapping |
  TrialSemanticSegmentationMapping | TrialClassificationMapping;

export interface TrialDecoderConfig {
  mode: 'auto' | 'builtin' | 'custom';
  decoder_id?: string;
  task?: TrialTask;
  input?: TrialInputMapping;
  output?: TrialOutputMapping;
  labels?: string[];
}

export interface TrialTensor {
  name: string;
  shape: Array<number | string | null>;
  dtype: string;
}

export interface TrialDecoderDescriptor {
  id: string;
  label: string;
  task: TrialResultKind;
  matched: boolean;
}

export interface TrialSchema {
  artifact: string;
  inputs: TrialTensor[];
  outputs: TrialTensor[];
  decoders: TrialDecoderDescriptor[];
  auto_decoder: { id: string; task: TrialResultKind } | null;
  mapping_required: boolean;
  names: Record<string, string>;
  saved_decoder: TrialDecoderConfig | null;
  input_shape?: number[] | null;
}

export interface TryResult {
  task: string;
  kind?: TrialResultKind;
  width: number;
  height: number;
  names: Record<string, string>;
  boxes?: Array<{ bbox: [number, number, number, number]; score: number; cls: number; label: string }>;
  polygons?: Array<{ score: number; cls: number; label: string; points: Array<[number, number]> }>;
  mask_overlay?: string;
  mask_legend?: Array<{ cls: number; label: string; color: [number, number, number, number] }>;
  mask_note?: string;
  probs?: {
    class_index: number;
    label: string;
    score: number;
    top5: Array<[string, number]>;
    topk?: Array<{ class_index: number; label: string; score: number }>;
  } | null;
  decoder?: { mode: string; id: string; task: TrialResultKind };
  heatmap?: string;
  cam_target?: { cls: number; label: string };
  time_ms: number;
  served_artifact?: string;
}

// ArtifactFormatDescriptor is supplied by the backend registry; the UI must
// not infer deployability from a filename extension.
export interface ArtifactFormatDescriptor {
  id: string;
  label: string;
  extensions: string[];
  primary_extensions: string[];
  capabilities: string[];
  compiled: boolean;
}

export interface ArtifactFormatCatalog {
  formats: ArtifactFormatDescriptor[];
  archive_suffixes: string[];
}

export interface ModelVersion {
  id: string;
  model_id: string;
  version: string;
  description?: string;
  file_path: string;
  size_bytes: number;
  metadata: ModelMetadata;
  status: ModelVersionStatus;
  artifacts?: Artifact[];
  artifacts_count?: number;
  created_by: string;
  created_at: string;
}

// One file inside a model version's S3 artifact bundle (GET .../versions/v/files)
export interface ModelVersionFile {
  name: string;
  size: number;
  uri: string;
  modified_at?: string;
}

// EdgePackageSummary is one precision row from GET .../versions/v/packages.
// Direct ONNX artifacts ship FP32 after structural validation; optimized IR
// retains its FP16/INT8 numerical and quality-gate requirements.
export interface EdgePackageSummary {
  precision: 'FP32' | 'FP16' | 'INT8';
  status: string;
  revision?: string;
  package_uri?: string;
  package_sha256?: string;
  package_bytes?: number;
  detection_count?: number | null;
  devices?: Array<{ device: string; status: string }>;
  quality_gate_passed?: boolean | null;
  map50_baseline?: number | null;
  map50_candidate?: number | null;
}

// EdgePackageRecord is the metadata.packages[] row persisted per precision by
// the backend (build-state view shown while a package is pending/running).
export interface EdgePackageRecord {
  precision: 'fp32' | 'fp16' | 'int8';
  status: string;
  revision?: string;
  task_id?: string;
  package_uri?: string;
  package_sha256?: string;
  package_bytes?: number;
  detection_count?: number | null;
  devices?: Array<{ device: string; status: string }>;
  quality_gate_passed?: boolean | null;
  error?: string;
}

export interface ModelMetadata {
  input_shape?: number[];
  output_shape?: number[];
  trial_decoders?: Record<string, TrialDecoderConfig>;
  num_parameters?: number;
  flops?: number;
  classes?: string[];
  preprocessing?: PreprocessingConfig;
  onnx_uri?: string;
  weights_uri?: string;
  target_device?: string;
  speedup_ratio?: number;
  size_reduction_pct?: number;
  optimizer?: string;
  source_version?: string;
  train_metrics?: Record<string, unknown>;
  opt_metrics?: Record<string, unknown>;
  uploaded_by?: string;
  job_id?: string;
  job_name?: string;
  artifact_format?: string;
  artifact_layout?: 'single-file' | 'multi-file' | 'zip' | 'tar.gz' | 'tgz';
  artifact_capabilities?: string[];
  bundle_uri?: string;
  primary_file?: string;
  runtime_requirements?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface PreprocessingConfig {
  mean: number[];
  std: number[];
  resize?: [number, number];
  normalization: 'imagenet' | 'inception' | 'custom';
}

export interface ModelDeployment {
  name: string;
  namespace: string;
  ready: boolean;
  url: string;
  runtime: string;
  image: string;
  model_uri: string;
  node: string;
  age: string;
  error: string;
  created_at: string;
  // Enhanced fields from backend
  model_name?: string;
  model_version?: string;
  model_type?: string;
  model_framework?: string;
  model_task_type?: string;
  // Backend returns deployed boolean and status object
  deployed?: boolean;
  status?: any;
}

export enum ModelFramework {
  PYTORCH = 'pytorch',
  TENSORFLOW = 'tensorflow',
  ONNX = 'onnx',
  TENSORRT = 'tensorrt',
  OPENVINO = 'openvino',
}

export enum ModelTaskType {
  CLASSIFICATION = 'classification',
  DETECTION = 'detection',
  SEGMENTATION = 'segmentation',
  // Interactive Segment Anything (click-to-segment) chosen in the video
  // annotator; distinct from instance-segmentation (YOLOv8-seg) weights.
  SAM = 'sam',
  POSE_ESTIMATION = 'pose_estimation',
  OCR = 'ocr',
  CUSTOM = 'custom',
}

export enum ModelVersionStatus {
  DRAFT = 'draft',
  READY = 'ready',
  DEPLOYING = 'deploying',
  DEPLOYED = 'deployed',
  DEPLOY_FAILED = 'deploy_failed',
  ARCHIVED = 'archived',
}

export interface MLBackend {
  id: string;
  name: string;
  url: string;
  description?: string;
  status: MLBackendStatus;
  health_check_interval: number;
  last_health_check?: string;
  capabilities: string[];
  model_name?: string;
  model_version?: string;
  model_type?: string;
  confidence_threshold?: number;
  annotations_schema?: Record<string, unknown>;
  created_by: string;
  created_at: string;
  updated_at: string;
}

export enum MLBackendStatus {
  HEALTHY = 'healthy',
  UNHEALTHY = 'unhealthy',
  UNKNOWN = 'unknown',
  MAINTENANCE = 'maintenance',
}

export enum JobStatus {
  PENDING = 'pending',
  QUEUED = 'queued',
  RUNNING = 'running',
  COMPLETED = 'completed',
  FAILED = 'failed',
  CANCELLED = 'cancelled',
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

export interface ApiError {
  code: string;
  message: string;
  details?: Record<string, unknown>;
}

export interface WebSocketMessage<T = unknown> {
  type: string;
  job_id: string;
  payload: T;
  timestamp: string;
}

export interface JobProgressPayload {
  job_id: string;
  status: JobStatus;
  progress: number;
  message?: string;
  metrics?: Record<string, unknown>;
}

export interface AuthTokens {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
}

export interface LoginResponse extends AuthTokens {
  user: User;
}

export type RefreshResponse = Omit<AuthTokens, 'refresh_token'>;

export interface LoginRequest {
  username: string;
  password: string;
  namespace?: string;
  captcha_id?: string;
  captcha_answer?: number;
}

export interface RefreshTokenRequest {
  refresh_token: string;
}

export type SsoMode = 'local' | 'ns' | 'oidc';

export interface SsoStatus {
  mode: SsoMode;
  enabled: boolean;
  login_path: string;
  logout_url: string;
  last_error: string;
}

export interface SsoCallbackRequest {
  code: string;
  state: string;
}

export interface SsoCallbackResponse extends LoginResponse {
  // Validated same-origin deep link to land on after login.
  to: string;
  // Present (instead of tokens) when the exchange failed.
  error?: string;
}
