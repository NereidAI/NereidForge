// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

import { Component, ErrorInfo, ReactNode } from 'react';
import { Result, Button } from 'antd';
import { ReloadOutlined } from '@ant-design/icons';
import i18n from '../i18n';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
    errorInfo: null,
  };

  public static getDerivedStateFromError(error: Error): Partial<State> {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    this.setState({ error, errorInfo });
    console.error('ErrorBoundary caught an error:', error, errorInfo);
    
    if (this.props.onError) {
      this.props.onError(error, errorInfo);
    }
  }

  private handleRetry = () => {
    this.setState({ hasError: false, error: null, errorInfo: null });
  };

  public render() {
    const t = i18n.t.bind(i18n);
    
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <div style={{ padding: 48, textAlign: 'center' }}>
          <Result
            status="500"
            title={t('common.error')}
            subTitle={this.state.error?.message || t('common.error')}
            extra={
              <Button type="primary" icon={<ReloadOutlined />} onClick={this.handleRetry}>
                {t('common.refresh') || 'Retry'}
              </Button>
            }
          />
          {import.meta.env.MODE === 'development' && this.state.error && (
            <details style={{ textAlign: 'left', marginTop: 24, padding: 16, background: '#f5f5f5', borderRadius: 4 }}>
              <summary style={{ cursor: 'pointer', fontWeight: 500 }}>Error Details</summary>
              <pre style={{ marginTop: 16, overflow: 'auto', maxHeight: 300 }}>
                {this.state.error.stack}
              </pre>
            </details>
          )}
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
