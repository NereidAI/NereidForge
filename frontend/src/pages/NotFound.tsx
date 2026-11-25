// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

import React from 'react';
import { Result, Button } from 'antd';
import { useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';

const NotFound: React.FC = () => {
  const { t } = useTranslation();
  const navigate = useNavigate();

  return (
    <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', minHeight: '60vh', textAlign: 'center', padding: 24 }}>
      <Result
        status="404"
        title={t('common.notFound')}
        subTitle={t('common.notFoundDesc')}
        extra={[
          <Button type="primary" key="back" onClick={() => navigate(-1)}>{t('common.back')}</Button>,
          <Button key="home" onClick={() => navigate('/dashboard')}>{t('common.home')}</Button>,
        ]}
      />
    </div>
  );
};

export default NotFound;
