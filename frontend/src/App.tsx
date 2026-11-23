// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

import React from 'react';
import { App as AntApp } from 'antd';
import Router from './router';
import { useAuthStore } from '@store/authStore';

const App: React.FC = () => {
  const { isLoading } = useAuthStore();

  if (isLoading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '100vh' }}>
        <div style={{ textAlign: 'center' }}>
          <div className="ant-spin-dot ant-spin-dot-spin" style={{ width: 32, height: 32, margin: '0 auto 16px' }}>
            <i className="ant-spin-dot-item" />
            <i className="ant-spin-dot-item" />
            <i className="ant-spin-dot-item" />
            <i className="ant-spin-dot-item" />
          </div>
          <div>Initializing...</div>
        </div>
      </div>
    );
  }

  return (
    <AntApp>
      <Router />
    </AntApp>
  );
};

export default App;
