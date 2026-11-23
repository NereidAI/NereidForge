// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

/**
 * Edge package view for one model version.
 *
 * There is no manual build button. Direct ONNX versions expose their FP32
 * structural package; optimized IR versions retain FP16/INT8 records, quality
 * gates, and device-verification summaries. The modal streams the selected
 * tar.gz through the byte-passthrough endpoint (NH/NL publish it unchanged).
 */
import { Button, Modal, Space, Table, Tag, Tooltip, message } from 'antd';
import { DownloadOutlined } from '@ant-design/icons';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { modelApi } from '../api/endpoints';
import type { ModelVersion, EdgePackageRecord } from '../types/api';

type Precision = 'fp32' | 'fp16' | 'int8';

interface Props {
  open: boolean;
  modelName: string;
  version: ModelVersion | null;
  onClose: () => void;
}

// IR_PRECISIONS preserves the existing optimization-package row order.
const IR_PRECISIONS: Precision[] = ['fp16', 'int8'];

// deviceStatus renders one vendor-prefixed verification matrix entry.
function deviceStatus(devices: Array<{ device: string; status: string }> | undefined): string {
  if (!devices || devices.length === 0) return '—';
  return devices.map((d) => `${d.device}:${d.status}`).join('  ');
}

// EdgePackageModal is read-only; builds happen automatically.
export default function EdgePackageModal({ open, modelName, version, onClose }: Props) {
  const { t } = useTranslation();
  const [downloading, setDownloading] = useState<Precision | null>(null);

  // Records arrive through version metadata (polled by the models page).
  const records = new Map<Precision, EdgePackageRecord>();
  for (const r of (version?.metadata?.packages as EdgePackageRecord[] | undefined) || []) {
    if (r && (r.precision === 'fp32' || r.precision === 'fp16' || r.precision === 'int8')) {
      records.set(r.precision, r);
    }
  }
  const precisions = version?.metadata?.artifact_format === 'onnx' || records.has('fp32')
    ? ['fp32'] as Precision[]
    : IR_PRECISIONS;

  // download streams the exact tar.gz bytes NH/NL must pass through unchanged.
  const download = async (precision: Precision) => {
    setDownloading(precision);
    try {
      const { data } = await modelApi.downloadPackage(modelName, version!.version, precision);
      const url = URL.createObjectURL(data as Blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `nereid-model-${precision}.tar.gz`;
      a.click();
      URL.revokeObjectURL(url);
    } catch {
      message.error(t('pages.models.packageFailed'));
    } finally {
      setDownloading(null);
    }
  };

  // statusTag renders pending/ready/failed states for one precision record.
  const statusTag = (r?: EdgePackageRecord) => {
    if (!r) return <Tag>{t('pages.models.packageNone')}</Tag>;
    if (r.status === 'ready') return <Tag color="green">{t('pages.models.packageReady')}</Tag>;
    if (r.status === 'failed') {
      return (
        <Tooltip title={r.error || ''}>
          <Tag color="red">{t('pages.models.packageFailed')}</Tag>
        </Tooltip>
      );
    }
    return <Tag color="orange">{t('pages.models.packageBuilding')}</Tag>;
  };

  // gateTag renders the map50 quality gate; INT8 ships only when passed.
  const gateTag = (r?: EdgePackageRecord) => {
    if (!r || r.quality_gate_passed === undefined || r.quality_gate_passed === null) return '—';
    return r.quality_gate_passed
      ? <Tag color="green">{t('pages.models.packageQualityGate')} {t('pages.models.packageGatePassed')}</Tag>
      : <Tag color="red">{t('pages.models.packageQualityGate')} {t('pages.models.packageGateFailed')}</Tag>;
  };

  return (
    <Modal
      title={`${t('pages.models.edgePackage')} - ${version?.version || ''}`}
      open={open}
      onCancel={onClose}
      footer={<Button onClick={onClose}>{t('common.close')}</Button>}
      width={1000}
      style={{ maxWidth: 'calc(100vw - 32px)' }}
      destroyOnHidden
    >
      <Table
        rowKey="precision"
        size="small"
        pagination={false}
        scroll={{ x: 'max-content' }}
        dataSource={precisions.map((precision) => ({ precision, rec: records.get(precision) }))}
        columns={[
          {
            title: t('pages.models.packageTarget'),
            dataIndex: 'precision',
            render: (precision: Precision) => {
              if (precision !== 'int8') return precision.toUpperCase();
              return (
                <Space size={4}>
                  INT8
                  <Tag color="geekblue" style={{ marginInlineStart: 0 }}>
                    {t('pages.models.packageGpuVerified')}
                  </Tag>
                </Space>
              );
            },
          },
          { title: t('pages.models.deployStatus'), key: 'status', render: (_, row) => statusTag(row.rec) },
          { title: 'mAP50', key: 'gate', render: (_, row) => gateTag(row.rec) },
          {
            title: t('pages.models.packageDevices'),
            key: 'devices',
            render: (_, row) => (
              <Tooltip title={deviceStatus(row.rec?.devices)}>
                <span style={{ fontFamily: 'monospace', fontSize: 12 }}>
                  {deviceStatus(row.rec?.devices)}
                </span>
              </Tooltip>
            ),
          },
          {
            title: t('common.actions'),
            key: 'actions',
            render: (_, row) => {
              const r = row.rec;
              return (
                r?.status === 'ready' ? (
                  <Tooltip title={r.package_sha256 ? `sha256:${r.package_sha256.slice(0, 16)}…` : ''}>
                    <Button
                      size="small"
                      icon={<DownloadOutlined />}
                      loading={downloading === row.precision}
                      onClick={() => download(row.precision)}
                    >
                      {t('common.download')}
                    </Button>
                  </Tooltip>
                ) : null
              );
            },
          },
        ]}
      />
    </Modal>
  );
}
