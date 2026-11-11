// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

import i18n from '../i18n';

// Format dates and times according to the active UI language:
// Chinese: YYYY/MM/DD HH:mm.
// English and other languages: MM/DD/YYYY, h:mm AM/PM.
export const formatDateTime = (d: string | number | Date): string => {
  const date = new Date(d);
  if (isNaN(date.getTime())) return '-';
  const lang = i18n.language || 'en';
  if (lang.startsWith('zh')) {
    const pad = (n: number) => String(n).padStart(2, '0');
    return `${date.getFullYear()}/${pad(date.getMonth() + 1)}/${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}`;
  }
  // Use the US locale for English and other languages.
  return date.toLocaleString('en-US', {
    month: 'numeric', day: 'numeric', year: 'numeric',
    hour: 'numeric', minute: '2-digit', hour12: true,
  });
};

export const formatDate = (d: string | number | Date): string => {
  const date = new Date(d);
  if (isNaN(date.getTime())) return '-';
  const lang = i18n.language || 'en';
  if (lang.startsWith('zh')) {
    const pad = (n: number) => String(n).padStart(2, '0');
    return `${date.getFullYear()}/${pad(date.getMonth() + 1)}/${pad(date.getDate())}`;
  }
  return date.toLocaleDateString('en-US', { month: 'numeric', day: 'numeric', year: 'numeric' });
};

export default formatDateTime;
