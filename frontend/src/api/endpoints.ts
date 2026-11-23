// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

import apiClient from './client';
import type {
  User,
  Tenant,
  Dataset,
  DatasetVersion,
  VersionFile,
  EdgePackageSummary,
  AnnotationTask,
  AnnotationItem,
  LSTaskView,
  CVATTask,
  CVATTaskDetail,
  CVATAnnotationDetail,
  CVATTrackInput,
  AsyncTask,
  TrainingJob,
  TrainingRuntime,
  TrainingConfig,
  OptimizationJob,
  EvaluationJob,
  Model,
  ModelVersion,
  ModelVersionFile,
  ArtifactFormatCatalog,
  TrialDecoderConfig,
  TrialSchema,
  TryResult,
  BaseModelPoolEntry,
  MLBackend,
  PaginatedResponse,
  JobStatus,
  LoginRequest,
  LoginResponse,
  RefreshResponse,
  SsoStatus,
  SsoCallbackResponse,
} from '@nftypes/api';
import type { AxiosRequestConfig } from 'axios';

const withParams = (params?: object): AxiosRequestConfig => ({ params: params as any });

// Auth endpoints
export const authApi = {
  login: (data: LoginRequest) => apiClient.post<LoginResponse>('/auth/login', data),
  captcha: () => apiClient.get<{ captcha_id: string; expression: string }>('/auth/captcha'),
  logout: () => apiClient.post('/auth/logout'),
  refresh: (refreshToken: string) => apiClient.post<RefreshResponse>('/auth/refresh', { refresh_token: refreshToken }),
  getCurrentUser: () => apiClient.get<User>('/auth/me'),
  updateProfile: (data: Partial<User>) => apiClient.patch<User>('/auth/me', data),
  // Suite SSO (NereidIdentity standard v2): status drives the login-page entry,
  // the exchange posts the authorization code back for a normal token pair.
  ssoStatus: () => apiClient.get<SsoStatus>('/auth/sso/status'),
  ssoCallback: (code: string, state: string) =>
    apiClient.post<SsoCallbackResponse>('/auth/sso/callback', { code, state }),
};

// Tenant endpoints
export const tenantApi = {
  list: (params?: { page?: number; page_size?: number; search?: string }) =>
    apiClient.get<PaginatedResponse<Tenant>>('/tenants', withParams(params)),
  get: (id: string) => apiClient.get<Tenant>(`/tenants/${id}`),
  create: (data: Partial<Tenant>) => apiClient.post<Tenant>('/tenants', data),
  update: (id: string, data: Partial<Tenant>) => apiClient.patch<Tenant>(`/tenants/${id}`, data),
  delete: (id: string) => apiClient.delete(`/tenants/${id}`),
  provision: (id: string) => apiClient.post<{ tenant_id: string; steps: string[]; status: string }>(`/tenants/${id}/provision`),
};

// Dataset endpoints
export const datasetApi = {
  list: (params?: { page?: number; page_size?: number; search?: string; status?: string }) =>
    apiClient.get<PaginatedResponse<Dataset>>('/datasets', withParams(params)),
  get: (id: string) => apiClient.get<Dataset>(`/datasets/${id}`),
  getReferences: (id: string) => apiClient.get<{ references: string[] }>(`/datasets/${id}/references`),
  create: (data: Partial<Dataset>) => apiClient.post<Dataset>('/datasets', data),
  update: (id: string, data: Partial<Dataset>) => apiClient.patch<Dataset>(`/datasets/${id}`, data),
  delete: (id: string) => apiClient.delete(`/datasets/${id}`),
  getVersions: (id: string, params?: { page?: number; page_size?: number }) =>
    apiClient.get<PaginatedResponse<DatasetVersion>>(`/datasets/${id}/versions`, withParams(params)),
  getVersion: (id: string, version: number) => apiClient.get<DatasetVersion>(`/datasets/${id}/versions/${version}`),
  createVersion: (id: string, data: { description?: string }) =>
    apiClient.post<DatasetVersion>(`/datasets/${id}/versions`, data),
  updateVersion: (id: string, version: number, data: { description?: string }) =>
    apiClient.patch<DatasetVersion>(`/datasets/${id}/versions/${version}`, data),
  deleteVersion: (id: string, version: number) =>
    apiClient.delete(`/datasets/${id}/versions/${version}`),
  createDraft: (id: string, data: { description?: string }) =>
    apiClient.post<DatasetVersion>(`/datasets/${id}/versions/draft`, data),
  finalizeDraft: (id: string, draftId: string) =>
    apiClient.post<DatasetVersion>(`/datasets/${id}/versions/draft/${draftId}/finalize`),
  deleteDraft: (id: string, draftId: string) =>
    apiClient.delete(`/datasets/${id}/versions/draft/${draftId}`),
  copyVersionToDraft: (id: string, versionId: string) =>
    apiClient.post<DatasetVersion>(`/datasets/${id}/versions/${versionId}/copy-to-draft`),
  // Register an existing S3 prefix (no copy) as a finalized trainable version.
  registerVersion: (id: string, data: { uri: string; format?: string; description?: string }) =>
    apiClient.post<DatasetVersion>(`/datasets/${id}/versions/register`, data),
  getVersionImages: (id: string, version: number, params?: { page?: number; page_size?: number }) =>
    apiClient.get<{ dataset_id: string; version: number; images: string[]; total: number; page: number; page_size: number }>(`/datasets/${id}/versions/${version}/images`, withParams(params)),
  getVersionFiles: (id: string, version: number) =>
    apiClient.get<{ dataset_id: string; version: number; files: VersionFile[]; total: number }>(`/datasets/${id}/versions/${version}/files`),
  getVersionCard: (id: string, version: number) =>
    apiClient.get<{ dataset_id: string; version: number; raw: string; readme: string; frontmatter: Record<string, any> }>(`/datasets/${id}/versions/${version}/card`),
  saveVersionCard: (id: string, version: number, content: string) =>
    apiClient.put<{ dataset_id: string; version: number; raw: string; readme: string; frontmatter: Record<string, any> }>(`/datasets/${id}/versions/${version}/card`, { content }),
  getVersionFile: (id: string, version: number, path: string) =>
    apiClient.get<{ dataset_id: string; version: number; path: string; content: string }>(`/datasets/${id}/versions/${version}/file`, withParams({ path })),
  deleteVersionImage: (id: string, version: number, imageName: string) =>
    apiClient.delete(`/datasets/${id}/versions/${version}/images/${encodeURIComponent(imageName)}`),
  uploadFiles: (id: string, files: File[], onProgress?: (percent: number) => void) => {
    const formData = new FormData();
    files.forEach(f => formData.append('files', f));
    return apiClient.post(`/datasets/${id}/upload`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      // Override the global 30s axios timeout (which otherwise cancels large
      // dataset/zip uploads mid-transfer) — same ceiling as model uploads.
      timeout: 1_800_000,
      onUploadProgress: onProgress
        ? (e) => { if (e.total) onProgress(Math.round((e.loaded / e.total) * 100)); }
        : undefined,
    });
  },
  download: (id: string, version?: number) => {
    const url = version ? `/datasets/${id}/download?version=${version}` : `/datasets/${id}/download`;
    return apiClient.get<{ presigned_url: string }>(url);
  },
};

// Annotation endpoints
export const annotationApi = {
  list: (params?: { page?: number; page_size?: number; dataset_id?: string; status?: string; assignee_id?: string; search?: string }) =>
    apiClient.get<PaginatedResponse<AnnotationTask>>('/projects', withParams(params)),
  get: (id: string) => apiClient.get<AnnotationTask>(`/projects/${id}`),
  create: (data: Partial<AnnotationTask> & {
    annotation_tool?: string;
    task_type?: string;
    schema?: Record<string, unknown>;
    import_uri?: string;
  }) => apiClient.post<AnnotationTask>('/projects', data),
  update: (id: string, data: Partial<AnnotationTask>) => apiClient.patch<AnnotationTask>(`/projects/${id}`, data),
  delete: (id: string) => apiClient.delete(`/projects/${id}`),
  start: (id: string) => apiClient.post(`/projects/${id}/start`),
  complete: (id: string) => apiClient.post(`/projects/${id}/complete`),
  cancel: (id: string) => apiClient.post(`/projects/${id}/cancel`),
  addLabels: (id: string, labels: string[]) => apiClient.post<AnnotationTask>(`/projects/${id}/labels`, { labels }),
  getItems: (id: string, params?: { page?: number; page_size?: number }) =>
    apiClient.get<PaginatedResponse<AnnotationItem>>(`/projects/${id}/items`, withParams(params)),
  submitAnnotation: (taskId: string, itemId: string, data: unknown) =>
    apiClient.post(`/projects/${taskId}/items/${itemId}`, data),
  getLSTasks: (id: string, params?: { page?: number; page_size?: number }) =>
    apiClient.get<PaginatedResponse<LSTaskView> & { ls_project_id?: number }>(`/projects/${id}/ls-tasks`, withParams(params)),
  saveLSAnnotation: (id: string, taskId: number, data: { result: unknown[]; draft_id?: number }) =>
    apiClient.post(`/projects/${id}/ls-tasks/${taskId}/annotations`, data),
  getCVATTasks: (id: string, params?: { page?: number; page_size?: number }) =>
    apiClient.get<{ items: CVATTask[]; total: number; page: number; page_size: number; cvat_project_id: number }>(`/projects/${id}/cvat-tasks`, withParams(params)),
  getCVATTask: (id: string, taskId: number) =>
    apiClient.get<CVATTaskDetail>(`/projects/${id}/cvat-tasks/${taskId}`),
  getCVATTaskAnnotation: (id: string, taskId: number) =>
    apiClient.get<CVATAnnotationDetail>(`/projects/${id}/cvat-tasks/${taskId}/annotation`),
  replaceCVATTracks: (id: string, taskId: number, data: { tracks: CVATTrackInput[] }) =>
    apiClient.put<{ saved: boolean; tracks: number }>(`/projects/${id}/cvat-tasks/${taskId}/annotations`, data),
  getCVATFrame: (id: string, taskId: number, frame: number, quality: string = 'compressed', signal?: AbortSignal) =>
    apiClient.get<ArrayBuffer>(`/projects/${id}/cvat-tasks/${taskId}/frames/${frame}`, {
      params: { quality },
      responseType: 'arraybuffer',
      signal,
    }),
  saveCVATAnnotation: (id: string, taskId: number, data: { result: unknown[] }) =>
    apiClient.post(`/projects/${id}/cvat-tasks/${taskId}/annotations`, data),
  exportDataset: (id: string, data: { name?: string; description?: string; format?: string }) =>
    apiClient.post<{ dataset_id: string; dataset_version: number; name: string; format: string }>(`/projects/${id}/export-dataset`, data),
  preannotate: (id: string, data?: { limit?: number; ml_backend_id?: string; task_ids?: string; frame_stride?: number; score_threshold?: number; model_name?: string; model_version?: string }) =>
    apiClient.post<{ task_id: string; status: string }>(`/projects/${id}/preannotate`, data),
  batchAccept: (id: string, data: { task_ids: number[] }) =>
    apiClient.post<{ task_id: string; status: string }>(`/projects/${id}/batch-accept`, data),
  getAsyncTask: (id: string, asyncTaskId: string) =>
    apiClient.get<AsyncTask>(`/projects/${id}/async-tasks/${asyncTaskId}`),
  getActiveAsyncTask: (id: string) =>
    apiClient.get<{ task: AsyncTask | null }>(`/projects/${id}/async-tasks/active`),
  controlAsyncTask: (id: string, asyncTaskId: string, action: 'pause' | 'resume' | 'cancel') =>
    apiClient.post<AsyncTask>(`/projects/${id}/async-tasks/${asyncTaskId}/control`, { action }),
  reimportTasks: (id: string) => apiClient.post<{ imported: number }>(`/projects/${id}/import-tasks`),
  imageURL: (id: string, path: string) => `/api/v1/projects/${id}/image?path=${encodeURIComponent(path)}`,
};

// Training endpoints
export const baseModelPoolApi = {
  list: (params: { kind?: string; search?: string; page?: number; page_size?: number } = {}) =>
    apiClient.get<PaginatedResponse<BaseModelPoolEntry>>('/base-model-pool', withParams(params)),
  upload: (
    data: {
      kind: 'cv' | 'llm'; name?: string; files: File[];
      framework?: string; task?: string; status?: string; description?: string;
    },
    onProgress?: (percent: number) => void,
  ) => {
    const fd = new FormData();
    fd.append('kind', data.kind);
    if (data.name) fd.append('name', data.name);
    if (data.framework) fd.append('framework', data.framework);
    if (data.task) fd.append('task', data.task);
    fd.append('status', data.status || 'develop');
    if (data.description) fd.append('description', data.description);
    data.files.forEach((f) => fd.append('files', f));
    return apiClient.post<{ items: BaseModelPoolEntry[]; total: number }>('/base-model-pool', fd, {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress: (e) => {
        if (onProgress && e.total) onProgress(Math.round((e.loaded / e.total) * 100));
      },
      timeout: 0,
    });
  },
  delete: (id: string) => apiClient.delete<{ deleted: boolean }>(`/base-model-pool/${id}`),
};

export const trainingApi = {
  list: (params?: { page?: number; page_size?: number; status?: string; model_id?: string; search?: string }) =>
    apiClient.get<PaginatedResponse<TrainingJob>>('/training/jobs', withParams(params)),
  get: (id: string) => apiClient.get<TrainingJob>(`/training/jobs/${id}`),
  create: (data: { name: string; description?: string; model_id?: string; dataset_id: string; dataset_version: number; config: TrainingConfig; custom_image?: string; custom_script_uri?: string }) =>
    apiClient.post<TrainingJob>('/training/jobs', data),
  update: (id: string, data: Partial<TrainingJob>) => apiClient.patch<TrainingJob>(`/training/jobs/${id}`, data),
  delete: (id: string) => apiClient.delete(`/training/jobs/${id}`),
  start: (id: string) => apiClient.post(`/training/jobs/${id}/start`),
  stop: (id: string) => apiClient.post(`/training/jobs/${id}/stop`),
  restart: (id: string) => apiClient.post(`/training/jobs/${id}/restart`),
  getLogs: (id: string, params?: { tail?: number }) =>
    apiClient.get<{ logs: string }>(`/training/jobs/${id}/logs`, withParams(params)),
  getRuntime: (id: string) =>
    apiClient.get<TrainingRuntime>(`/training/jobs/${id}/runtime`),
  getMetrics: (id: string) => apiClient.get<TrainingJob['metrics']>(`/training/jobs/${id}/metrics`),
  getCurves: (id: string) => apiClient.get<{
    curves: Array<Record<string, unknown>>;
    history?: Array<Record<string, unknown>>;
  }>(`/training/jobs/${id}/curves`),
  getCheckpoints: (id: string) => apiClient.get<Array<{ epoch: number; path: string; metrics: Record<string, number> }>>(`/training/jobs/${id}/checkpoints`),
};

// Optimization endpoints
export const optimizationApi = {
  list: (params?: { page?: number; page_size?: number; status?: string; search?: string }) =>
    apiClient.get<PaginatedResponse<OptimizationJob>>('/optimization/jobs', withParams(params)),
  get: (id: string) => apiClient.get<OptimizationJob>(`/optimization/jobs/${id}`),
  create: (data: {
    name: string;
    description?: string;
    source_model_id: string;
    source_version?: string;
    dataset_id?: string;
    dataset_version?: number;
    targets: Array<{ target_device: string; quant_method: string; dtype?: string; calib_n?: number }>;
    params?: Record<string, unknown>;
  }) => apiClient.post<{ items: OptimizationJob[]; batch_id: string }>('/optimization/jobs', data),
  update: (id: string, data: Partial<OptimizationJob>) => apiClient.patch<OptimizationJob>(`/optimization/jobs/${id}`, data),
  delete: (id: string) => apiClient.delete(`/optimization/jobs/${id}`),
  start: (id: string) => apiClient.post(`/optimization/jobs/${id}/start`),
  stop: (id: string) => apiClient.post(`/optimization/jobs/${id}/stop`),
  getLogs: (id: string, params?: { tail?: number }) =>
    apiClient.get<{ logs: string }>(`/optimization/jobs/${id}/logs`, withParams(params)),
  overrideGate: (id: string, reason: string) =>
    apiClient.post<OptimizationJob>(`/optimization/jobs/${id}/override`, { reason }),
  // Streaming download (the API proxies seaweedfs bytes; no presigned URLs)
  downloadUrl: (id: string) => `/api/v1/optimization/jobs/${id}/download`,
};

// Evaluation endpoints
export const evaluationApi = {
  list: (params?: { page?: number; page_size?: number; status?: string; model_id?: string; search?: string }) =>
    apiClient.get<PaginatedResponse<EvaluationJob>>('/evaluation/jobs', withParams(params)),
  get: (id: string) => apiClient.get<EvaluationJob>(`/evaluation/jobs/${id}`),
  create: (data: {
    name: string;
    description?: string;
    model_id: string;
    model_version?: string;
    baseline_model_id?: string;
    baseline_model_version?: string;
    dataset_id?: string;
    dataset_version?: number;
    params?: Record<string, unknown>;
  }) => apiClient.post<EvaluationJob>('/evaluation/jobs', data),
  update: (id: string, data: Partial<EvaluationJob>) => apiClient.patch<EvaluationJob>(`/evaluation/jobs/${id}`, data),
  delete: (id: string) => apiClient.delete(`/evaluation/jobs/${id}`),
  start: (id: string) => apiClient.post(`/evaluation/jobs/${id}/start`),
  stop: (id: string) => apiClient.post(`/evaluation/jobs/${id}/stop`),
  getLogs: (id: string, params?: { tail?: number }) =>
    apiClient.get<{ logs: string }>(`/evaluation/jobs/${id}/logs`, withParams(params)),
  getResults: (id: string) => apiClient.get<EvaluationJob['results']>(`/evaluation/jobs/${id}/results`),
  // Streaming download of the report JSON
  downloadUrl: (id: string) => `/api/v1/evaluation/jobs/${id}/download`,
};

// Model endpoints
export const modelApi = {
  list: (params?: { page?: number; page_size?: number; framework?: string; task_type?: string; search?: string }) =>
    apiClient.get<PaginatedResponse<Model>>('/models', withParams(params)),
  get: (id: string) => apiClient.get<Model>(`/models/${id}`),
  create: (data: Partial<Model>) => apiClient.post<Model>('/models', data),
  update: (id: string, data: Partial<Model>) => apiClient.patch<Model>(`/models/${id}`, data),
  delete: (id: string) => apiClient.delete<{ deleted: boolean }>(`/models/${id}`),
  deleteVersion: (id: string, version: string) =>
    apiClient.delete<{ deleted: boolean }>(`/models/${id}/versions/${version}`),
  getVersions: (id: string, params?: { page?: number; page_size?: number }) =>
    apiClient.get<PaginatedResponse<ModelVersion>>(`/models/${id}/versions`, withParams(params)),
  getVersion: (modelId: string, version: string) => apiClient.get<ModelVersion>(`/models/${modelId}/versions/${version}`),
  getArtifactFormats: () => apiClient.get<ArtifactFormatCatalog>('/models/artifact-formats'),
  // List the complete recursive bundle; names remain relative to bundle_uri.
  getVersionFiles: (modelId: string, version: string) =>
    apiClient.get<{ model_name: string; version: string; files: ModelVersionFile[]; total: number }>(
      `/models/${modelId}/versions/${version}/files`,
    ),
  // Inspect the selected artifact through the ML Backend's decoder registry.
  // The saved per-artifact selection is returned alongside graph tensor metadata.
  getTrySchema: (modelId: string, version: string, artifactPath?: string) => {
    const qs = artifactPath ? `?path=${encodeURIComponent(artifactPath)}` : '';
    return apiClient.get<TrialSchema>(`/models/${modelId}/versions/${version}/try/schema${qs}`, {
      timeout: 200_000,
    });
  },
  // Ad-hoc single-image trial without a KServe deployment. Decoder choices are
  // persisted by the backend only after the selected contract runs successfully.
  tryVersion: (
    modelId: string,
    version: string,
    opts: {
      image: File;
      conf?: number;
      iou?: number;
      cam?: boolean;
      classIndex?: number;
      artifactPath?: string;
      decoderConfig?: TrialDecoderConfig;
      topK?: number;
    },
  ) => {
    const fd = new FormData();
    fd.append('image', opts.image);
    fd.append('conf', String(opts.conf ?? 0.25));
    fd.append('iou', String(opts.iou ?? 0.7));
    fd.append('cam', String(opts.cam ?? false));
    fd.append('top_k', String(opts.topK ?? 5));
    if (opts.classIndex !== undefined) fd.append('class_index', String(opts.classIndex));
    if (opts.decoderConfig) fd.append('decoder_config', JSON.stringify(opts.decoderConfig));
    // Optional exact relative path within the recursive version bundle.
    const qs = opts.artifactPath ? `?path=${encodeURIComponent(opts.artifactPath)}` : '';
    return apiClient.post<TryResult>(`/models/${modelId}/versions/${version}/try${qs}`, fd, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 200_000,
    });
  },
  createVersion: (modelId: string, data: { version: string; description?: string; files: File[]; metadata: ModelVersion['metadata'] }, onProgress?: (percent: number) => void) => {
    const formData = new FormData();
    formData.append('version', data.version);
    if (data.description) formData.append('description', data.description);
    // Repeated fields preserve IR companions and other multi-file artifacts;
    // archives remain one File and are validated/extracted by the backend.
    data.files.forEach(file => formData.append('file', file));
    formData.append('metadata', JSON.stringify(data.metadata));
    return apiClient.post<ModelVersion>(`/models/${modelId}/versions`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      // Model checkpoints can be hundreds of MB and take minutes to upload;
      // override the global 30s axios timeout (which otherwise cancels mid-upload).
      timeout: 1_800_000,
      onUploadProgress: onProgress ? (e) => {
        if (e.total) onProgress(Math.round((e.loaded / e.total) * 100));
      } : undefined,
    });
  },
  // deploy is fully async: it enqueues a model_deploy worker task (ONNX
  // conversion can take 5-10 minutes) and returns in milliseconds with
  // state:"deploying". The UI polls servingStatus until ready/failed.
  // undeploy is also async: it only submits the KFP run and returns in ~1s.
  deploy: (modelId: string, version: string, target: string, opts?: { runtime_type?: string; runtime_image?: string; gpu?: string; domain?: string; model_uri?: string }) =>
    apiClient.post(`/models/${modelId}/versions/${version}/deploy`, { target, runtime_type: opts?.runtime_type, runtime_image: opts?.runtime_image, gpu: opts?.gpu, domain: opts?.domain }),
  undeploy: (modelId: string, version: string) =>
    apiClient.post(`/models/${modelId}/versions/${version}/undeploy`),
  undeployByName: (name: string, namespace?: string) =>
    apiClient.post(`/models/deployments/${name}/undeploy`, undefined, withParams({ namespace })),
  servingStatus: (modelId: string, version: string) =>
    apiClient.get<{ deployed: boolean; ready?: boolean; state?: string; error?: string; url?: string; service?: string }>(`/models/${modelId}/versions/${version}/serving`),
  getServingRuntimes: () => apiClient.get<{ items: Array<{ name: string; model_type: string; image: string }> }>('/models/serving-runtimes'),
  // This read-only API lists structurally validated FP32 ONNX packages and
  // numerically verified FP16/INT8 IR packages; GET /package streams the tar.
  listPackages: (modelId: string, version: string) =>
    apiClient.get<{ model_id: string; version: string; packages: EdgePackageSummary[] }>(
      `/models/${modelId}/versions/${version}/packages`),
  downloadPackage: (modelId: string, version: string, precision: 'fp32' | 'fp16' | 'int8') =>
    apiClient.get(`/models/${modelId}/versions/${version}/package`, {
      params: { precision }, responseType: 'blob',
    }),
};

// Generic async-task polling (dataset imports, edge packaging, ...).
export const asyncTaskApi = {
  get: (taskId: string) =>
    apiClient.get<{ id: string; type: string; status: string; progress: number; error_message?: string }>(`/async-tasks/${taskId}`),
};

// ML Backend endpoints
export const mlBackendApi = {
  list: (params?: { page?: number; page_size?: number; status?: string; search?: string }) =>
    apiClient.get<PaginatedResponse<MLBackend>>('/ml-backends', withParams(params)),
  get: (id: string) => apiClient.get<MLBackend>(`/ml-backends/${id}`),
  create: (data: Partial<MLBackend>) => apiClient.post<MLBackend>('/ml-backends', data),
  update: (id: string, data: Partial<MLBackend>) => apiClient.patch<MLBackend>(`/ml-backends/${id}`, data),
  delete: (id: string) => apiClient.delete(`/ml-backends/${id}`),
  healthCheck: (id: string) => apiClient.post<{ status: string; latency_ms: number }>(`/ml-backends/${id}/health`),
  getCapabilities: (id: string) => apiClient.get<string[]>(`/ml-backends/${id}/capabilities`),
  createCustom: (data: { model_name: string; model_version: string; s3_path: string; backend_type: string; framework: string }) =>
    apiClient.post<MLBackend>('/ml-backends/custom', data),
  createCustomFromRegistry: (data: { model_name: string; model_version: string }) =>
    apiClient.post<MLBackend>('/ml-backends/custom/from-registry', data),
  getStatus: (id: string) => apiClient.get<{ service_name: string; url: string; status: string; ready_replicas: number; total_replicas: number; pods: Array<{ name: string; status: string; ready: boolean }> }>(`/ml-backends/${id}/status`),
};

// Dashboard endpoints
export const dashboardApi = {
  getStats: () => apiClient.get<{
    datasets_count: number;
    models_count: number;
    training_jobs_count: number;
    annotation_tasks_count: number;
    recent_jobs: Array<{ id: string; name: string; type: string; status: JobStatus; created_at: string }>;
  }>('/dashboard/stats'),
};

/** GpuVendorCapacity reports the schedulable devices of one accelerator vendor. */
export interface GpuVendorCapacity {
  resource: string;
  total: number;
  max_per_node: number;
  nodes: Array<{ name: string; devices: number }>;
}

// Settings endpoints (admin only)
export const settingsApi = {
  getUsers: (params?: { page?: number; page_size?: number; search?: string }) =>
    apiClient.get<PaginatedResponse<User>>('/settings/users', withParams(params)),
  createUser: (data: { username: string; email: string; full_name?: string; password: string; roles: string[]; tenant_id: string }) =>
    apiClient.post<User>('/settings/users', data),
  updateUser: (id: string, data: Partial<User>) => apiClient.patch<User>(`/settings/users/${id}`, data),
  deleteUser: (id: string) => apiClient.delete(`/settings/users/${id}`),
  getRoles: () => apiClient.get<Array<{ name: string; description: string; permissions: string[] }>>('/settings/roles'),
  createRole: (data: { name: string; description: string; permissions: string[] }) =>
    apiClient.post('/settings/roles', data),
  updateRole: (name: string, data: { description: string; permissions: string[] }) =>
    apiClient.patch(`/settings/roles/${name}`, data),
  deleteRole: (name: string) => apiClient.delete(`/settings/roles/${name}`),
  getSystemConfig: () => apiClient.get<Record<string, unknown>>('/system/config'),
  testRegistry: (data: { host?: string; project?: string; username?: string; password?: string; insecure?: boolean }) =>
    apiClient.post<{ ok: boolean; message?: string; host?: string; project?: string; repo_count?: number; reachable?: boolean; project_exists?: boolean }>('/system/registry/test', data),
  updateSystemConfig: (data: Record<string, unknown>) => apiClient.put('/system/config', { data }),
  preflight: () =>
    apiClient.get<{
      summary: 'ok' | 'warn' | 'error';
      ok: boolean;
      error_count: number;
      warning_count: number;
      checks: Array<{ name: string; status: 'ok' | 'warn' | 'error'; message: string }>;
    }>('/system/preflight'),
  getComputeResources: () => apiClient.get<{ resources: Array<{ device: string; name: string; count: number; unit: string; trainable: boolean; gpu_model?: string }> }>('/system/compute-resources'),
  getHardwareCapabilities: () => apiClient.get<Record<string, { available: boolean; detail?: string }>>('/hardware-capabilities'),
  getGpuCapacity: () => apiClient.get<{ vendors: Record<string, GpuVendorCapacity> }>('/gpu-capacity'),
  getNodes: () => apiClient.get<{ items: Array<{ name: string; status: string }> }>('/nodes'),
  getDeployments: () => apiClient.get<{ services: Array<{ name: string; namespace: string; ready: boolean; url: string; domain: string }> }>('/models/deployments'),
  getAuditLogs: (params?: { page?: number; page_size?: number; action?: string; search?: string }) =>
    apiClient.get<PaginatedResponse<{ id: string; tenant_id: string; user_id: string; action: string; resource_type: string; resource_id: string; details: Record<string, unknown>; created_at: string }>>('/system/audit-logs', withParams(params)),
};
// Container image catalog (train / prediction)
export interface AnnouncedArchitecture {
  id: string;
  label?: string;
  size?: number;
  tier?: 'core' | 'extended' | string;
  license?: string;
  requires?: string[];
  locked?: boolean;
  missing?: string[];
}

export interface AnnouncedTask {
  id: string;
  label?: string;
  tier?: 'core' | 'extended' | string;
  requires?: string[];
  default_arch?: string;
  locked?: boolean;
  missing?: string[];
  architectures: AnnouncedArchitecture[];
}

export interface AnnouncedCapability {
  id: string;
  label?: string;
  tier?: string;
  requires?: string[];
  locked?: boolean;
  missing?: string[];
}

export interface CapabilityImage {
  id: string;
  name: string;
  repository: string;
  tag: string;
  full_ref: string;
  is_builtin: boolean;
  // Legacy task filter retained for catalog images without an announcement.
  task_types: string[];
  devices: string[];
  announced: boolean;
  profile_error?: string;
  tasks: AnnouncedTask[];
  capabilities: AnnouncedCapability[];
}

export interface CapabilitiesResponse {
  entitlements: string[];
  images: CapabilityImage[];
}

export interface ContainerImageEntry {
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
  profile?: Record<string, unknown>;
  profile_at?: string | null;
  profile_error?: string;
  default_archs?: Record<string, string>;
  extended?: Array<{ id: string; kind: string; label?: string; task?: string; requires: string[] }>;
  is_builtin: boolean;
  created_by: string;
  created_at: string;
}

export const imageApi = {
  list: (params?: { page?: number; page_size?: number; category?: string; task_type?: string; search?: string }) =>
    apiClient.get<PaginatedResponse<ContainerImageEntry>>('/images', withParams(params)),
  capabilities: () =>
    apiClient.get<CapabilitiesResponse>('/capabilities'),
  create: (data: Partial<ContainerImageEntry>) =>
    apiClient.post<ContainerImageEntry>('/images', data),
  update: (id: string, data: Partial<ContainerImageEntry>) =>
    apiClient.patch<ContainerImageEntry>(`/images/${id}`, data),
  setProfile: (id: string, profile: Record<string, unknown> | string | null) =>
    apiClient.post<ContainerImageEntry>(`/images/${id}/profile`, { profile }),
  refreshProfile: (id: string) =>
    apiClient.post<ContainerImageEntry>(`/images/${id}/profile/refresh`),
  delete: (id: string) => apiClient.delete(`/images/${id}`),
};

// Interactive SAM segmentation (click-to-segment)
export const samApi = {
  segment: (data: {
    image_url?: string;
    image_base64?: string;
    points?: number[][];
    negative_points?: number[][];
    box?: number[];
    ml_backend_id?: string;
    // User-selected SAM model from the registry (resolved to s3 weights server-side).
    model_name?: string;
    model_version?: string;
  }) =>
    apiClient.post<{ polygon: number[][] | null; bbox: number[] | null; message?: string; error?: string }>('/sam/segment', data),
};
