// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import svgr from 'vite-plugin-svgr';
import path from 'path';

// Vite 8 is built on Rolldown, which no longer accepts the object form of
// output.manualChunks; each package group below therefore maps to an explicit
// membership test inside this function while keeping the same chunk names.
const MANUAL_CHUNKS: Record<string, string[]> = {
  vendor: ['react', 'react-dom', 'react-router-dom'],
  antd: ['antd', '@ant-design/icons'],
  utils: ['axios', 'zustand', 'lodash-es', 'dayjs'],
};

/**
 * resolveManualChunk mirrors the former object-form manualChunks grouping.
 * @param id absolute module identifier emitted by the bundler.
 * @returns the destination chunk name, or undefined for default placement.
 */
function resolveManualChunk(id: string): string | undefined {
  if (!id.includes('node_modules')) {
    return undefined;
  }
  for (const [chunkName, packages] of Object.entries(MANUAL_CHUNKS)) {
    if (packages.some((pkg) => id.includes(`node_modules/${pkg}/`))) {
      return chunkName;
    }
  }
  return undefined;
}

export default defineConfig({
  plugins: [
    react(),
    svgr({
      svgrOptions: {
        exportType: 'default',
      },
    }),
  ],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
      '@api': path.resolve(__dirname, './src/api'),
      '@components': path.resolve(__dirname, './src/components'),
      '@pages': path.resolve(__dirname, './src/pages'),
      '@hooks': path.resolve(__dirname, './src/hooks'),
      '@store': path.resolve(__dirname, './src/store'),
      '@nftypes': path.resolve(__dirname, './src/types'),
      '@styles': path.resolve(__dirname, './src/styles'),
    },
  },
  server: {
    port: 5173,
    host: true,
    proxy: {
      '/api': {
        target: 'http://localhost:8080',
        changeOrigin: true,
      },
      // SSO mode-ns public callback is a product route served by the backend
      // (also routed via ingress in prod); keep dev behaviour identical.
      '/auth/ns-callback': {
        target: 'http://localhost:8080',
        changeOrigin: true,
      },
      '/notebook-servers': {
        target: 'http://localhost:8080',
        changeOrigin: false,
        ws: true,
      },
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    rollupOptions: {
      output: {
        manualChunks: resolveManualChunk,
      },
    },
  },
});
