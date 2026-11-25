// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

import React, { Suspense, lazy } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { useAuthStore } from '@store/authStore';
import { Layout } from '@components/Layout';
import { ErrorBoundary } from '@components/ErrorBoundary';
import { Spin } from 'antd';

// Lazy load pages
const Dashboard = lazy(() => import('@pages/Dashboard'));
const Datasets = lazy(() => import('@pages/Datasets'));
const DatasetVersions = lazy(() => import('@pages/Datasets').then(m => ({ default: m.DatasetVersions || m.default })));
const Annotations = lazy(() => import('@pages/Annotations'));
const AnnotationStudio = lazy(() => import('@pages/AnnotationStudio'));
const Training = lazy(() => import('@pages/Training'));
const Notebooks = lazy(() => import('@pages/Notebooks'));
const TrainingDetail = lazy(() => import('@pages/Training').then(m => ({ default: m.TrainingDetail || m.default })));
const Optimization = lazy(() => import('@pages/Optimization'));
const OptimizationDetail = lazy(() => import('@pages/Optimization').then(m => ({ default: m.OptimizationDetail || m.default })));
const Evaluation = lazy(() => import('@pages/Evaluation'));
const EvaluationDetail = lazy(() => import('@pages/Evaluation').then(m => ({ default: m.EvaluationDetail || m.default })));
const Models = lazy(() => import('@pages/Models'));
const ModelVersions = lazy(() => import('@pages/Models').then(m => ({ default: m.ModelVersions || m.default })));
const MLBackend = lazy(() => import('@pages/MLBackend'));
const Images = lazy(() => import('@pages/Images'));
const Deployments = lazy(() => import('@pages/Deployments'));
const Settings = lazy(() => import('@pages/Settings'));
const HelpCenter = lazy(() => import('@pages/HelpCenter'));
const Profile = lazy(() => import('@pages/Profile'));
const BaseModels = lazy(() => import('@pages/admin/BaseModels'));
const Login = lazy(() => import('@pages/Login'));
const SSOCallback = lazy(() => import('@pages/SSOCallback'));
const NotFound = lazy(() => import('@pages/NotFound'));

// Loading fallback
const PageLoader = () => (
  <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '400px' }}>
    <Spin size="large" />
  </div>
);

// Wrapper with ErrorBoundary and Suspense
const WithErrorBoundary = ({ children }: { children: React.ReactNode }) => (
  <ErrorBoundary>
    <Suspense fallback={<PageLoader />}>{children}</Suspense>
  </ErrorBoundary>
);

// Protected route wrapper
const ProtectedRoute = ({ children, adminOnly = false }: { children: React.ReactNode; adminOnly?: boolean }) => {
  const { isAuthenticated, isLoading, isAdmin } = useAuthStore();

  if (isLoading) {
    return <PageLoader />;
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" replace state={{ from: window.location.pathname }} />;
  }

  if (adminOnly && !isAdmin()) {
    return <Navigate to="/dashboard" replace />;
  }

  return <WithErrorBoundary>{children}</WithErrorBoundary>;
};

// Public route (for login, callback)
const PublicRoute = ({ children }: { children: React.ReactNode }) => {
  const { isAuthenticated, isLoading } = useAuthStore();

  if (isLoading) {
    return <PageLoader />;
  }

  if (isAuthenticated) {
    return <Navigate to="/dashboard" replace />;
  }

  return <WithErrorBoundary>{children}</WithErrorBoundary>;
};

const Router: React.FC = () => {
  return (
    <Routes>
      {/* Public routes */}
      <Route path="/" element={<Navigate to="/dashboard" replace />} />
      <Route path="/login" element={<PublicRoute><Login /></PublicRoute>} />
      {/* SSO callback: the backend/idP lands here with ?code=&state= (oidc) or
          #session= (ns); it owns finishing the login, so it is public. */}
      <Route path="/auth/callback" element={<WithErrorBoundary><SSOCallback /></WithErrorBoundary>} />

      {/* Protected routes with layout */}
      <Route element={<Layout />}>
        <Route
          path="/dashboard"
          element={
            <ProtectedRoute>
              <Dashboard />
            </ProtectedRoute>
          }
        />
        <Route
          path="/datasets"
          element={
            <ProtectedRoute>
              <Datasets />
            </ProtectedRoute>
          }
        />
        <Route
          path="/datasets/:id/versions"
          element={
            <ProtectedRoute>
              <DatasetVersions />
            </ProtectedRoute>
          }
        />
        <Route
          path="/annotations"
          element={
            <ProtectedRoute>
              <Annotations />
            </ProtectedRoute>
          }
        />
        <Route
          path="/annotations/:id"
          element={
            <ProtectedRoute>
              <AnnotationStudio />
            </ProtectedRoute>
          }
        />
        <Route
          path="/training"
          element={
            <ProtectedRoute>
              <Training />
            </ProtectedRoute>
          }
        />
        <Route
          path="/notebooks"
          element={
            <ProtectedRoute>
              <Notebooks />
            </ProtectedRoute>
          }
        />
        <Route
          path="/training/:id"
          element={
            <ProtectedRoute>
              <TrainingDetail />
            </ProtectedRoute>
          }
        />
        <Route
          path="/optimization"
          element={
            <ProtectedRoute>
              <Optimization />
            </ProtectedRoute>
          }
        />
        <Route
          path="/optimization/:id"
          element={
            <ProtectedRoute>
              <OptimizationDetail />
            </ProtectedRoute>
          }
        />
        <Route
          path="/evaluation"
          element={
            <ProtectedRoute>
              <Evaluation />
            </ProtectedRoute>
          }
        />
        <Route
          path="/evaluation/:id"
          element={
            <ProtectedRoute>
              <EvaluationDetail />
            </ProtectedRoute>
          }
        />
        <Route
          path="/models"
          element={
            <ProtectedRoute>
              <Models />
            </ProtectedRoute>
          }
        />
        <Route
          path="/models/:name/versions/:version"
          element={
            <ProtectedRoute>
              <ModelVersions />
            </ProtectedRoute>
          }
        />
        <Route
          path="/ml-backend"
          element={
            <ProtectedRoute>
              <MLBackend />
            </ProtectedRoute>
          }
        />
        <Route
          path="/images"
          element={
            <ProtectedRoute adminOnly>
              <Images />
            </ProtectedRoute>
          }
        />
        <Route
          path="/base-models"
          element={
            <ProtectedRoute>
              <BaseModels />
            </ProtectedRoute>
          }
        />
        <Route
          path="/deployments"
          element={
            <ProtectedRoute>
              <Deployments />
            </ProtectedRoute>
          }
        />
        <Route
          path="/settings"
          element={
            <ProtectedRoute adminOnly>
              <Settings />
            </ProtectedRoute>
          }
        />
        <Route
          path="/help"
          element={
            <ProtectedRoute>
              <HelpCenter />
            </ProtectedRoute>
          }
        />
        <Route
          path="/profile"
          element={
            <ProtectedRoute>
              <Profile />
            </ProtectedRoute>
          }
        />
      </Route>

      {/* 404 */}
      <Route path="*" element={<NotFound />} />
    </Routes>
  );
};

export default Router;
