// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

import React from 'react';
import { Tag, Tooltip } from 'antd';
import { useTranslation } from 'react-i18next';

type StatusColor = 'blue' | 'green' | 'red' | 'orange' | 'purple' | 'cyan' | 'gray' | 'gold' | 'volcano' | 'magenta' | 'lime' | 'geekblue';

interface StatusConfig {
  color: StatusColor;
  label: string;
}

const statusConfigMap: Record<string, StatusConfig> = {
  // JobStatus
  pending: { color: 'gray', label: 'components.statusBadge.pending' },
  queued: { color: 'blue', label: 'components.statusBadge.queued' },
  running: { color: 'blue', label: 'components.statusBadge.running' },
  completed: { color: 'green', label: 'components.statusBadge.completed' },
  failed: { color: 'red', label: 'components.statusBadge.failed' },
  cancelled: { color: 'volcano', label: 'components.statusBadge.cancelled' },
  
  // DatasetStatus
  draft: { color: 'gray', label: 'components.statusBadge.draft' },
  active: { color: 'green', label: 'components.statusBadge.active' },
  archived: { color: 'volcano', label: 'components.statusBadge.archived' },
  
  // AnnotationTaskStatus
  in_progress: { color: 'blue', label: 'components.statusBadge.inProgress' },
  
  // MLBackendStatus
  healthy: { color: 'green', label: 'components.statusBadge.healthy' },
  unhealthy: { color: 'red', label: 'components.statusBadge.unhealthy' },
  unknown: { color: 'gray', label: 'components.statusBadge.unknown' },
  maintenance: { color: 'orange', label: 'components.statusBadge.maintenance' },
  
  // ModelVersionStatus
  ready: { color: 'green', label: 'components.statusBadge.ready' },
  deployed: { color: 'cyan', label: 'components.statusBadge.deployed' },
};

interface StatusBadgeProps {
  status: string;
  showTooltip?: boolean;
}

/** Return the i18n key for a status string (e.g. "in_progress" -> "components.statusBadge.inProgress").
 *  Use this in non-JSX contexts (subtitles, Statistic values) where the <StatusBadge> tag can't be used. */
export const statusLabelKey = (status: string): string => {
  const config = statusConfigMap[status];
  return config ? config.label : status;
};

// Statuses offered by the filter dropdown shared by every jobs list page.
export const JOB_STATUS_FILTER_VALUES = [
  'pending', 'queued', 'running', 'completed', 'failed', 'cancelled',
] as const;

/** Build translated filter dropdown options for the standard job statuses. */
export const useJobStatusOptions = () => {
  const { t } = useTranslation();
  return JOB_STATUS_FILTER_VALUES.map((value) => ({
    value, label: t(statusConfigMap[value]?.label || value),
  }));
};

export const StatusBadge: React.FC<StatusBadgeProps> = ({
  status,
  showTooltip = true,
}) => {
  const { t } = useTranslation();
  const config = statusConfigMap[status] || { color: 'gray', label: status };

  const badge = (
    <Tag color={config.color}>
      {t(config.label)}
    </Tag>
  );

  if (showTooltip) {
    return <Tooltip title={t(config.label)}>{badge}</Tooltip>;
  }

  return badge;
};

export default StatusBadge;
