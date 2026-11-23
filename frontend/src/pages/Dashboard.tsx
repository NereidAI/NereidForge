// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

import React, { useEffect, useState } from 'react';
import { Card, Row, Col, Statistic, Spin, Alert } from 'antd';
import { DashboardOutlined, DatabaseOutlined, RocketOutlined, EditOutlined } from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import { useNavigate } from 'react-router-dom';
import { dashboardApi } from '@api/endpoints';
import { PageHeader } from '@components/PageHeader';
import { formatDateTime } from '../utils/format';

interface DashboardStats {
  datasets_count: number;
  models_count: number;
  training_jobs_count: number;
  annotation_tasks_count: number;
  recent_jobs: Array<{
    id: string;
    name: string;
    type: string;
    status: string;
    created_at: string;
  }>;
}

export const Dashboard: React.FC = () => {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        setLoading(true);
        const response = await dashboardApi.getStats();
        setStats(response.data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load dashboard');
      } finally {
        setLoading(false);
      }
    };

    fetchStats();
  }, []);

  const statCards = [
    {
      title: t('pages.dashboard.stats.datasets'),
      value: stats?.datasets_count || 0,
      icon: <DatabaseOutlined style={{ color: '#1677ff' }} />,
      color: '#1677ff',
      path: '/datasets',
    },
    {
      title: t('pages.dashboard.stats.models'),
      value: stats?.models_count || 0,
      icon: <RocketOutlined style={{ color: '#52c41a' }} />,
      color: '#52c41a',
      path: '/models',
    },
    {
      title: t('pages.dashboard.stats.trainingJobs'),
      value: stats?.training_jobs_count || 0,
      icon: <DashboardOutlined style={{ color: '#fa8c16' }} />,
      color: '#fa8c16',
      path: '/training',
    },
    {
      title: t('pages.dashboard.stats.annotations'),
      value: stats?.annotation_tasks_count || 0,
      icon: <EditOutlined style={{ color: '#722ed1' }} />,
      color: '#722ed1',
      path: '/projects',
    },
  ];

  if (loading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '400px' }}>
        <Spin size="large" />
      </div>
    );
  }

  if (error) {
    return (
      <Alert
        message={t('common.error')}
        description={error}
        type="error"
        showIcon
        style={{ marginBottom: 24 }}
      />
    );
  }

  return (
    <div>
      <PageHeader
        title={t('pages.dashboard.title')}
        subtitle={t('pages.dashboard.welcome')}
      />
      
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        {statCards.map((item, index) => (
          <Col key={index} xs={24} sm={12} lg={6}>
            <Card
              style={{ cursor: 'pointer', transition: 'box-shadow 0.2s', boxShadow: '0 1px 2px rgba(0,0,0,0.03)' }}
              onClick={() => item.path && navigate(item.path)}
            >
              <Statistic
                title={item.title}
                value={item.value}
                prefix={item.icon}
                valueStyle={{ color: item.color }}
              />
            </Card>
          </Col>
        ))}
      </Row>

      <Row gutter={[16, 16]}>
        <Col span={24}>
          <Card title={t('common.recentActivity') || 'Recent Activity'} variant="borderless">
            {stats?.recent_jobs.length === 0 ? (
              <div style={{ textAlign: 'center', padding: 48, color: '#999' }}>
                {t('common.noData') || 'No recent activity'}
              </div>
            ) : (
              <div style={{ maxHeight: 400, overflow: 'auto' }}>
                {stats?.recent_jobs.map((job) => (
                  <div
                    key={job.id}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      padding: '12px 0',
                      borderBottom: '1px solid #f0f0f0',
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                      <div
                        style={{
                          width: 8,
                          height: 8,
                          borderRadius: '50%',
                          backgroundColor: 
                            job.status === 'completed' ? '#52c41a' :
                            job.status === 'running' ? '#1677ff' :
                            job.status === 'failed' ? '#ff4d4f' : '#fa8c16',
                        }}
                      />
                      <div>
                        <div style={{ fontWeight: 500 }}>{job.name}</div>
                        <div style={{ fontSize: 12, color: '#999' }}>
                          {job.type} • {formatDateTime(job.created_at)}
                        </div>
                      </div>
                    </div>
                    <span style={{ textTransform: 'capitalize' }}>{job.status}</span>
                  </div>
                ))}
              </div>
            )}
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default Dashboard;
