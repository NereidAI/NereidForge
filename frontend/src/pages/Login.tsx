// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Card, Button, Alert, Form, Input, Row, Col, Checkbox, message, Divider } from 'antd';
import { UserOutlined, LockOutlined, SafetyOutlined, LoginOutlined } from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import { useSearchParams } from 'react-router-dom';
import { useAuth } from '../hooks/useAuth';
import { authApi } from '@api/endpoints';
import { ssoLoginUrl, isRelativePath } from '@/utils/sso';
import type { SsoStatus } from '@nftypes/api';

interface LoginFormValues {
  username: string;
  password: string;
  namespace?: string;
  captcha_answer?: number;
  remember?: boolean;
}

const REMEMBER_KEY = 'nereidforge.remembered_username';

const Login: React.FC = () => {
  const { t } = useTranslation();
  const { login, isLoading: authLoading } = useAuth();
  const [searchParams] = useSearchParams();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [sso, setSso] = useState<SsoStatus | null>(null);
  const [captcha, setCaptcha] = useState<{ captcha_id: string; expression: string } | null>(null);
  const [form] = Form.useForm<LoginFormValues>();
  const captchaLoaded = useRef(false);

  const refreshCaptcha = useCallback(async () => {
    try {
      const res = await authApi.captcha();
      setCaptcha(res.data);
      form.setFieldValue('captcha_answer', undefined);
    } catch (e) {
      console.error('captcha fetch failed', e);
    }
  }, [form]);

  // Preload the arithmetic captcha + remembered username on mount.
  useEffect(() => {
    if (!captchaLoaded.current) {
      captchaLoaded.current = true;
      refreshCaptcha();
    }
    const remembered = window.localStorage.getItem(REMEMBER_KEY);
    if (remembered) {
      form.setFieldsValue({ username: remembered, remember: true });
    }
    // SSO status decides whether the "Sign in with SSO" entry is shown.
    authApi
      .ssoStatus()
      .then((res) => setSso(res.data))
      .catch(() => setSso(null));
  }, [form, refreshCaptcha]);

  // Surface errors the SSO callback redirects back with (?oidc_error=code).
  useEffect(() => {
    const code = searchParams.get('oidc_error');
    if (code) {
      setError(t(`auth.ssoError.${code}`, { defaultValue: t('auth.ssoFailed') }));
    }
  }, [searchParams, t]);

  const handleSsoLogin = () => {
    const target = isRelativePath(searchParams.get('redirect')) ? searchParams.get('redirect')! : '/dashboard';
    window.location.assign(ssoLoginUrl(sso?.login_path || '/api/v1/auth/sso/login', target));
  };

  const handleLogin = async (values: LoginFormValues) => {
    setLoading(true);
    setError('');
    try {
      if (values.remember) {
        window.localStorage.setItem(REMEMBER_KEY, values.username);
      } else {
        window.localStorage.removeItem(REMEMBER_KEY);
      }
      await login(
        values.username,
        values.password,
        values.namespace || undefined,
        captcha?.captcha_id,
        values.captcha_answer,
      );
    } catch (e: any) {
      const msg =
        e?.response?.data?.detail ||
        e?.response?.data?.message ||
        e?.message ||
        'Login failed';
      setError(msg);
      message.error(msg);
      refreshCaptcha();
    } finally {
      setLoading(false);
    }
  };

  if (authLoading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '400px' }}>
        <span>Loading...</span>
      </div>
    );
  }

  return (
    <Card style={{ maxWidth: 400, margin: '100px auto' }}>
      <div style={{ textAlign: 'center', marginBottom: 24 }}>
        <h1>NereidForge</h1>
        <p style={{ color: '#999' }}>Model Factory for Edge AI</p>
      </div>
      {error && <Alert message={t('common.loginFailed')} description={error} type="error" style={{ marginBottom: 16 }} />}
      <Form<LoginFormValues> form={form} onFinish={handleLogin} layout="vertical">
        <Form.Item name="username" rules={[{ required: true, message: t('common.required') }]}>
          <Input prefix={<UserOutlined />} placeholder={t('auth.username')} size="large" autoComplete="username" />
        </Form.Item>
        <Form.Item name="password" rules={[{ required: true, message: t('common.required') }]}>
          <Input.Password prefix={<LockOutlined />} placeholder={t('auth.password')} size="large" autoComplete="current-password" />
        </Form.Item>
        <Form.Item
          name="namespace"
          extra={t('auth.namespaceHint')}
          style={{ marginBottom: 12 }}
        >
          <Input
            prefix={<SafetyOutlined />}
            placeholder={t('auth.namespace')}
            size="large"
            autoComplete="off"
          />
        </Form.Item>
        <Form.Item name="captcha_answer" rules={[{ required: true, message: t('common.required') }]} style={{ marginBottom: 12 }}>
          <Row gutter={8} align="middle">
            <Col span={16}>
              <Input
                prefix={<SafetyOutlined />}
                placeholder={t('auth.captcha')}
                size="large"
                inputMode="numeric"
                autoComplete="off"
                addonBefore={<span title={t('auth.captchaHint')}>{captcha?.expression ?? '...'}</span>}
              />
            </Col>
            <Col span={8}>
              <Button size="large" block onClick={refreshCaptcha}>
                {t('auth.refreshCaptcha')}
              </Button>
            </Col>
          </Row>
        </Form.Item>
        <Form.Item name="remember" valuePropName="checked" style={{ marginBottom: 12 }}>
          <Checkbox>{t('auth.rememberMe')}</Checkbox>
        </Form.Item>
        <Button type="primary" htmlType="submit" size="large" block loading={loading}>
          {t('auth.signIn')}
        </Button>
        {sso?.enabled && (
          <>
            <Divider plain style={{ margin: '16px 0' }}>{t('auth.ssoOr')}</Divider>
            <Button size="large" block icon={<LoginOutlined />} onClick={handleSsoLogin}>
              {t('auth.ssoSignIn')}
            </Button>
          </>
        )}
      </Form>
    </Card>
  );
};

export default Login;
