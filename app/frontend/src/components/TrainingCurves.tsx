// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

import React, { useEffect, useState, useMemo } from 'react';
import { Card, Empty, Spin } from 'antd';
import { useTranslation } from 'react-i18next';
import { trainingApi } from '@api/endpoints';

// A single results.csv row (Ultralytics format): column name -> string value.
type CurveRow = Record<string, string>;

// A single history.jsonl row produced by the HF/timm trainers.
interface HistoryPoint {
  step?: number;
  epoch?: number;
  loss?: number | null;
  eval_loss?: number | null;
  eval_accuracy?: number | null;
  learning_rate?: number | null;
}

// Distinct colors cycled across curves within a group.
const PALETTE = [
  '#1890ff', '#fa8c16', '#52c41a', '#722ed1', '#eb2f96',
  '#13c2c2', '#fa541c', '#2f54eb', '#a0d911', '#f5222d',
  '#fadb14', '#08979c',
];

// Convert a CSV cell to a number; empty/invalid -> NaN (kept for x-axis alignment).
const toNum = (v: unknown): number => {
  if (v === undefined || v === null || v === '') return NaN;
  const n = Number(v);
  return isNaN(n) ? NaN : n;
};

// Format a numeric value for axis labels, tolerating tiny learning-rate values.
const fmt = (v: number): string => {
  if (!isFinite(v)) return '0';
  if (v === 0) return '0';
  if (Math.abs(v) < 0.001) return v.toExponential(1);
  if (Math.abs(v) >= 1000) return v.toFixed(0);
  return v.toFixed(4);
};

// Whether a CSV column should be plotted (excludes epoch/time/meta columns).
const isPlottable = (key: string): boolean => {
  const k = key.trim().toLowerCase();
  if (!k || k === 'epoch' || k === 'time' || k === ' ') return false;
  return true;
};

// Classify a column into a group key based on Ultralytics naming conventions.
// Handles both prefixed (train/box_loss) and flat (train_loss) column names,
// so detection/segmentation/classification CSVs all render correctly.
const groupOf = (key: string): string => {
  const k = key.toLowerCase();
  if (k.startsWith('train/') || k.startsWith('train_')) return 'train';
  if (k.startsWith('val/') || k.startsWith('val_')) return 'val';
  if (k.startsWith('metrics/')) return 'metrics';
  if (k.startsWith('lr/') || k.startsWith('lr_')) return 'lr';
  // Flat metric names without a prefix.
  if (/(precision|recall|map|accuracy|f1)/.test(k)) return 'metrics';
  if (k.includes('loss')) return 'train';
  return 'other';
};

// Turn a raw column key into a human-readable curve label.
//   train/box_loss      -> "Box Loss"
//   metrics/mAP50(B)    -> "mAP50 (B)"
//   lr/pg0              -> "Param Group 0"
const curveLabel = (key: string): string => {
  // Strip a single category prefix (train/, val/, metrics/, lr/).
  const slashIdx = key.indexOf('/');
  let name = slashIdx >= 0 ? key.slice(slashIdx + 1) : key;
  // Also strip flat train_/val_ prefixes.
  name = name.replace(/^(train|val|metrics|lr)_/i, '');
  // Special-case learning-rate param groups.
  const pg = name.match(/^pg(\d+)$/i);
  if (pg) return `Param Group ${pg[1]}`;
  // Separate a trailing parenthetical like (B)/(M) and insert a space.
  const m = name.match(/^([^(]*)(\([^)]*\))?$/);
  if (!m) return name;
  // Capitalize the first letter of all-lowercase words (preserves mAP50, DFL).
  const base = m[1]
    .replace(/_/g, ' ')
    .trim()
    .split(' ')
    .map(w => (/^[a-z]+$/.test(w) ? w.charAt(0).toUpperCase() + w.slice(1) : w))
    .join(' ');
  const suffix = m[2] ? ` ${m[2]}` : '';
  return (base + suffix).trim() || name;
};

interface CurveDef {
  key: string;
  label: string;
  color: string;
  data: number[];
}

interface CurveGroup {
  key: string;
  title: string;
  curves: CurveDef[];
}

// Simple responsive SVG line chart (no external chart dependency).
const LineChart: React.FC<{ data: number[]; label: string; color: string; xLabel?: string }> = ({ data, label, color, xLabel }) => {
  const { t } = useTranslation();
  // NaN points (empty CSV cells) are ignored for min/max and point-count
  // checks but keep their epoch index so the x-axis stays aligned.
  const valid = data.filter(v => !isNaN(v));
  if (valid.length < 2) {
    return <div style={{ color: '#999', fontSize: 12, padding: 8 }}>{label}: {t('pages.training.curveInsufficient')}</div>;
  }
  const W = 600;
  const H = 180;
  const pad = 34;
  const w = W - pad * 2;
  const h = H - pad * 2;
  const min = Math.min(...valid);
  const max = Math.max(...valid);
  const range = max - min || 1;
  const denom = data.length - 1 || 1;
  // Split the curve into contiguous segments at NaN gaps.
  const segments: string[][] = [];
  let current: string[] = [];
  data.forEach((v, i) => {
    if (isNaN(v)) {
      if (current.length) {
        segments.push(current);
        current = [];
      }
      return;
    }
    const x = pad + (i / denom) * w;
    const y = pad + h - ((v - min) / range) * h;
    current.push(`${x.toFixed(1)},${y.toFixed(1)}`);
  });
  if (current.length) segments.push(current);
  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', marginBottom: 4 }}>
        <span style={{ width: 10, height: 10, background: color, display: 'inline-block', marginRight: 6, borderRadius: 2 }} />
        <span style={{ fontSize: 13, color: '#555' }}>{label}</span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto' }}>
        <line x1={pad} y1={pad} x2={pad} y2={pad + h} stroke="#e0e0e0" strokeWidth={1} />
        <line x1={pad} y1={pad + h} x2={pad + w} y2={pad + h} stroke="#e0e0e0" strokeWidth={1} />
        <text x={pad} y={pad - 8} fontSize={10} fill="#999">{fmt(max)}</text>
        <text x={pad} y={pad + h + 14} fontSize={10} fill="#999">{fmt(min)}</text>
        <text x={pad + w} y={pad + h + 14} fontSize={10} fill="#999" textAnchor="end">{data.length}{xLabel || 'ep'}</text>
        {segments.map((pts, si) => (
          <polyline key={si} points={pts.join(' ')} fill="none" stroke={color} strokeWidth={2} />
        ))}
        {data.map((v, i) => !isNaN(v) && (
          <circle
            key={i}
            cx={pad + (i / denom) * w}
            cy={pad + h - ((v - min) / range) * h}
            r={2}
            fill={color}
          />
        ))}
      </svg>
    </div>
  );
};

// Training curves panel: reads /training/jobs/{id}/curves and dynamically
// renders every numeric column from the Ultralytics results.csv, grouped by
// category (train / val / metrics / learning rate).
export const TrainingCurves: React.FC<{ jobId: string }> = ({ jobId }) => {
  const { t } = useTranslation();
  const [curves, setCurves] = useState<CurveRow[]>([]);
  const [history, setHistory] = useState<HistoryPoint[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const load = async () => {
      try {
        const res = await trainingApi.getCurves(jobId);
        setCurves((res.data.curves as CurveRow[]) || []);
        setHistory((res.data.history as HistoryPoint[]) || []);
      } catch {
        setCurves([]);
        setHistory([]);
      } finally {
        setLoading(false);
      }
    };
    load();
    const timer = setInterval(load, 5000);
    return () => clearInterval(timer);
  }, [jobId]);

  // Live per-step curves from history.jsonl (LLM/VLM/timm trainers). Only
  // series with >=2 numeric points render; dense arrays keep the x-axis
  // aligned to optimizer steps rather than epochs.
  const historyCharts = useMemo(() => {
    if (!history.length) return [] as Array<{ key: string; label: string; color: string; data: number[] }>;
    const series = [
      {
        key: 'loss',
        label: t('pages.training.curveTrainLoss'),
        color: PALETTE[0],
        data: history.map(h => toNum(h.loss)),
      },
      {
        key: 'eval_loss',
        label: t('pages.training.curveEvalLoss'),
        color: PALETTE[1],
        data: history.map(h => toNum(h.eval_loss)),
      },
      {
        key: 'eval_accuracy',
        label: t('pages.training.curveEvalAccuracy'),
        color: PALETTE[2],
        data: history.map(h => toNum(h.eval_accuracy)),
      },
      {
        key: 'learning_rate',
        label: t('pages.training.curveLearningRate'),
        color: PALETTE[3],
        data: history.map(h => toNum(h.learning_rate)),
      },
    ];
    return series.filter(s => s.data.filter(v => !isNaN(v)).length >= 2);
  }, [history, t]);

  // Dynamically build grouped curve definitions from whatever columns the
  // results.csv actually contains (works for detection / segmentation /
  // classification / pose models without hardcoded column names).
  const groups = useMemo<CurveGroup[]>(() => {
    if (!curves.length) return [];
    const cols = Object.keys(curves[0]).filter(isPlottable);
    const order: { key: string; title: string }[] = [
      { key: 'train', title: t('pages.training.curveGroupTrain') },
      { key: 'val', title: t('pages.training.curveGroupVal') },
      { key: 'metrics', title: t('pages.training.curveGroupMetrics') },
      { key: 'lr', title: t('pages.training.curveGroupLr') },
      { key: 'other', title: t('pages.training.curveGroupOther') },
    ];
    return order
      .map(g => {
        const cs = cols
          .filter(c => groupOf(c) === g.key)
          .map((c, i) => ({
            key: c,
            label: curveLabel(c),
            color: PALETTE[i % PALETTE.length],
            // Keep all rows (NaN included) so the x-axis stays aligned to
            // epochs; the chart skips NaN points below.
            data: curves.map(row => toNum(row[c])),
          }));
        return { ...g, curves: cs };
      })
      .filter(g => g.curves.length > 0);
  }, [curves, t]);

  if (loading) {
    return <div style={{ textAlign: 'center', padding: 40 }}><Spin /></div>;
  }
  if ((!curves || curves.length === 0) && historyCharts.length === 0) {
    return <Empty description={t('pages.training.curveEmpty')} style={{ padding: 40 }} />;
  }

  // HF/timm path: no Ultralytics results.csv, render the per-step history.
  if ((!curves || curves.length === 0) && historyCharts.length > 0) {
    return (
      <Card size="small" title={`${t('pages.training.curvesTitle')} · ${t('pages.training.curveSteps', { count: history.length })}`}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 16 }}>
          {historyCharts.map(c => (
            <LineChart key={c.key} data={c.data} label={c.label} color={c.color} xLabel="step" />
          ))}
        </div>
      </Card>
    );
  }

  return (
    <Card size="small" title={`${t('pages.training.curvesTitle')} · ${t('pages.training.curveEpochs', { count: curves.length })}`}>
      {groups.map(g => (
        <div key={g.key} style={{ marginBottom: 20 }}>
          <div style={{ fontWeight: 600, marginBottom: 8, color: '#555', fontSize: 14 }}>{g.title}</div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 16 }}>
            {g.curves.map(c => (
              <LineChart key={c.key} data={c.data} label={c.label} color={c.color} />
            ))}
          </div>
        </div>
      ))}
    </Card>
  );
};

export default TrainingCurves;
