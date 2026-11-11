// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

import React from 'react';
import { Card, Descriptions, Tag, Avatar } from 'antd';
import { useTranslation } from 'react-i18next';
import { useAuthStore } from '@store/authStore';
import { PageHeader } from '@components/PageHeader';
import { formatDateTime } from '../utils/format';

const Profile: React.FC = () => {
  const { t } = useTranslation();
  const { user } = useAuthStore();

  if (!user) {
    return (
      <Card>
        <PageHeader title={t('pages.profile.title')} />
        <p>{t('common.noData')}</p>
      </Card>
    );
  }

  const roleColors: Record<string, string> = {
    admin: 'gold',
    user: 'blue',
  };

  return (
    <Card>
      <PageHeader title={t('pages.profile.title')} subtitle={t('pages.profile.subtitle')} />
      <div style={{ display: 'flex', alignItems: 'center', gap: 24, marginBottom: 24 }}>
        <Avatar size={96} src={user.email ? `https://www.gravatar.com/avatar/${user.email}` : undefined}>
          {user.username?.[0]?.toUpperCase() || 'U'}
        </Avatar>
        <div>
          <h2 style={{ margin: 0 }}>{user.username}</h2>
          <p style={{ margin: '4px 0 0', color: '#8c8c8c' }}>{user.full_name}</p>
        </div>
      </div>
      <Descriptions bordered column={1} size="middle">
        <Descriptions.Item label={t('pages.profile.fullName')}>{user.full_name}</Descriptions.Item>
        <Descriptions.Item label={t('pages.profile.email')}>{user.email}</Descriptions.Item>
        <Descriptions.Item label={t('pages.profile.tenant')}>{user.tenant_id}</Descriptions.Item>
        <Descriptions.Item label={t('pages.profile.roles')}>
          {(user.roles || []).map((role) => (
            <Tag key={role} color={roleColors[role] || 'default'}>
              {role}
            </Tag>
          ))}
        </Descriptions.Item>
        <Descriptions.Item label={t('pages.profile.status')}>
          <Tag color={user.is_active ? 'green' : 'red'}>
            {user.is_active ? t('pages.profile.active') : t('pages.profile.inactive')}
          </Tag>
        </Descriptions.Item>
        <Descriptions.Item label={t('pages.profile.createdAt')}>
          {formatDateTime(user.created_at)}
        </Descriptions.Item>
        <Descriptions.Item label={t('pages.profile.updatedAt')}>
          {formatDateTime(user.updated_at)}
        </Descriptions.Item>
      </Descriptions>
    </Card>
  );
};

export default Profile;
