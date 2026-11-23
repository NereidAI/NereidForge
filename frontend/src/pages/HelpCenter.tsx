// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

import React from 'react';
import { Card, List, Typography, Alert } from 'antd';
import { BookOutlined, DatabaseOutlined, TagsOutlined, ExperimentOutlined, ThunderboltOutlined, DeploymentUnitOutlined, RobotOutlined } from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import PageHeader from '@components/PageHeader';
import { HelpDrawer } from '@components/HelpDrawer';

const { Title, Paragraph } = Typography;

const HelpCenter: React.FC = () => {
  const { t } = useTranslation();
  const [openModule, setOpenModule] = React.useState<string | null>(null);

  const modules = [
    { key: 'datasets', icon: <DatabaseOutlined />, title: t('pages.datasets.title') },
    { key: 'annotations', icon: <TagsOutlined />, title: t('pages.annotations.title') },
    { key: 'training', icon: <ExperimentOutlined />, title: t('pages.training.title') },
    { key: 'optimization', icon: <ThunderboltOutlined />, title: t('pages.optimization.title') },
    { key: 'evaluation', icon: <DeploymentUnitOutlined />, title: t('pages.evaluation.title') },
    { key: 'models', icon: <BookOutlined />, title: t('pages.models.title') },
    { key: 'mlbackend', icon: <RobotOutlined />, title: t('pages.mlBackend.title') },
  ];

  return (
    <div>
      <PageHeader title={t('pages.help.title')} subtitle={t('pages.help.subtitle')} />
      <Alert
        type="info"
        showIcon
        message={t('pages.help.overview')}
        description={t('pages.help.overviewDesc')}
        style={{ marginBottom: 16 }}
      />
      <Card>
        <List
          grid={{ gutter: 16, column: 2 }}
          dataSource={modules}
          renderItem={(m) => (
            <List.Item>
              <Card hoverable onClick={() => setOpenModule(m.key)} style={{ cursor: 'pointer' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                  <span style={{ fontSize: 22, color: '#1677ff' }}>{m.icon}</span>
                  <div>
                    <Title level={5} style={{ margin: 0 }}>{m.title}</Title>
                    <Paragraph type="secondary" style={{ margin: 0, fontSize: 12 }}>{t('pages.help.clickToView')}</Paragraph>
                  </div>
                </div>
              </Card>
            </List.Item>
          )}
        />
      </Card>
      <HelpDrawer module={openModule || ''} open={!!openModule} onClose={() => setOpenModule(null)} title={modules.find(m => m.key === openModule)?.title} />
    </div>
  );
};

export default HelpCenter;
