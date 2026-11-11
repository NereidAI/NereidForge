// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

import React, { useEffect, useState } from 'react';
import { Drawer, Spin, Typography, Empty, Button } from 'antd';
import { QuestionCircleOutlined } from '@ant-design/icons';
import { useTranslation } from 'react-i18next';

const { Paragraph } = Typography;

// Minimal Markdown renderer for help documents (headings, lists, paragraphs, and emphasis).
const renderMD = (text: string): React.ReactNode[] => {
  const lines = text.split('\n');
  const out: React.ReactNode[] = [];
  let list: string[] = [];
  const flushList = (key: number) => {
    if (list.length) {
      out.push(
        <ul key={key}>
          {list.map((li, i) => <li key={i}>{li}</li>)}
        </ul>
      );
      list = [];
    }
  };
  let k = 0;
  for (const raw of lines) {
    const line = raw.trimEnd();
    if (line.startsWith('# ')) {
      flushList(k++);
      out.push(<h2 key={k++} style={{ borderBottom: '1px solid #f0f0f0', paddingBottom: 8 }}>{line.slice(2)}</h2>);
    } else if (line.startsWith('## ')) {
      flushList(k++);
      out.push(<h4 key={k++} style={{ marginTop: 16 }}>{line.slice(3)}</h4>);
    } else if (line.startsWith('- ')) {
      list.push(line.slice(2));
    } else if (line.trim() === '') {
      flushList(k++);
    } else {
      flushList(k++);
      out.push(<Paragraph key={k++} style={{ marginBottom: 4 }}>{line}</Paragraph>);
    }
  }
  flushList(k++);
  return out;
};

export const HelpDrawer: React.FC<{ module: string; open: boolean; onClose: () => void; title?: string }> = ({ module, open, onClose, title }) => {
  const { t, i18n } = useTranslation();
  const [content, setContent] = useState('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!open || !module) return;
    setLoading(true);
    const lang = i18n.language?.startsWith('zh') ? 'zh' : 'en';
    const path = `${import.meta.env.BASE_URL || '/'}help/${module}.md`;
    // Help documents are bilingual at paragraph level; load them directly.
    fetch(path)
      .then((res) => (res.ok ? res.text() : Promise.reject()))
      .then(setContent)
      .catch(() => setContent(`# ${t('common.help')}\n\n${t('common.helpUnavailable')}`))
      .finally(() => setLoading(false));
    void lang;
  }, [open, module, i18n.language, t]);

  return (
    <Drawer
      title={title || t('common.help')}
      open={open}
      onClose={onClose}
      width={480}
      destroyOnHidden
    >
      {loading ? <div style={{ textAlign: 'center', padding: 48 }}><Spin /></div> : content ? renderMD(content) : <Empty />}
    </Drawer>
  );
};

// Page help button for PageHeader actions.
export const HelpButton: React.FC<{ module: string; title?: string }> = ({ module, title }) => {
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);
  return (
    <>
      <Button icon={<QuestionCircleOutlined />} onClick={() => setOpen(true)}>
        {t('common.help')}
      </Button>
      <HelpDrawer module={module} open={open} onClose={() => setOpen(false)} title={title || t('common.help')} />
    </>
  );
};

export default HelpDrawer;
