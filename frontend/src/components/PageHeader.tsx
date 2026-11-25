// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

import React from 'react';
import { Typography, Breadcrumb, Button, Space, Tooltip, Divider } from 'antd';
import { PlusOutlined, ReloadOutlined } from '@ant-design/icons';

const { Title } = Typography;

interface PageHeaderProps {
  title: string;
  subtitle?: string;
  breadcrumbs?: Array<{ title: string; href?: string }>;
  actions?: React.ReactNode;
  extra?: React.ReactNode;
  onRefresh?: () => void;
  onCreate?: () => void;
  loading?: boolean;
}

export const PageHeader: React.FC<PageHeaderProps> = ({
  title,
  subtitle,
  breadcrumbs,
  actions,
  extra,
  onRefresh,
  onCreate,
  loading = false,
}) => {
  const breadcrumbItems = (breadcrumbs || []).map((item) => ({
    title: item.href ? <a href={item.href}>{item.title}</a> : item.title,
  }));
  return (
    <div style={{ marginBottom: 24, background: '#fff', padding: '16px 24px', borderRadius: 8, boxShadow: '0 1px 2px rgba(0,0,0,0.03)' }}>
      {breadcrumbItems.length > 0 && (
        <Breadcrumb
          style={{ marginBottom: 12 }}
          items={breadcrumbItems}
        />
      )}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 16 }}>
        <div>
          <Title level={3} style={{ marginBottom: 4, fontWeight: 600 }}>
            {title}
          </Title>
          {subtitle && (
            <p style={{ margin: 0, color: '#8c8c8c', fontSize: 14 }}>{subtitle}</p>
          )}
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
          {extra}
          <Space>
            {actions && React.Children.map(actions, (child, i) =>
              child ? React.cloneElement(child as React.ReactElement, { key: i }) : null
            )}
            {onCreate && (
              <Tooltip title="Create">
                <Button type="primary" icon={<PlusOutlined />} onClick={onCreate}>
                  Create
                </Button>
              </Tooltip>
            )}
            {onRefresh && (
              <Tooltip title="Refresh">
                <Button icon={<ReloadOutlined />} onClick={onRefresh} loading={loading} />
              </Tooltip>
            )}
          </Space>
        </div>
      </div>
      {extra && <Divider style={{ marginTop: 16 }} />}
    </div>
  );
};

export default PageHeader;
