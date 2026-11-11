// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE_URL: string;
  readonly VITE_APP_VERSION: string;
  readonly VITE_IMAGE_TAG: string;
  readonly VITE_DATASET_UPLOAD_MAX_FILES: string;
  readonly VITE_DATASET_UPLOAD_MAX_BYTES: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
