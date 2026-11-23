// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import LanguageDetector from 'i18next-browser-languagedetector';

import en from './locales/en';
import zh from './locales/zh';

// How to add a new language:
//   1. Create src/locales/<lang>.ts with `const <lang>: Translation`
//      (import the `Translation` type from en.ts so structure is enforced)
//   2. Add it to `resources` and `supportedLngs` below
//   3. Add its antd locale mapping in src/App.tsx
//   4. (Optional) Add it to the language switcher in src/components/Layout.tsx
const RESOURCES = {
  en: { translation: en },
  zh: { translation: zh },
} as const;

const SUPPORTED_LNGS = Object.keys(RESOURCES);

// Migrate the legacy locale codes ('zh-CN' / 'en-US') that may have been
// cached in localStorage before the switch to 'zh' / 'en'
const convertDetectedLanguage = (lng: string): string => {
  const normalized = lng.toLowerCase();
  if (normalized === 'zh-cn' || normalized === 'zh') return 'zh';
  if (normalized === 'en-us' || normalized === 'en') return 'en';
  return normalized;
};

i18n
  .use(LanguageDetector)
  .use(initReactI18next)
  .init({
    resources: RESOURCES,
    fallbackLng: 'en',
    supportedLngs: SUPPORTED_LNGS,
    nonExplicitSupportedLngs: true,
    detection: {
      order: ['localStorage', 'navigator', 'htmlTag'],
      caches: ['localStorage'],
      convertDetectedLanguage,
    },
    interpolation: {
      escapeValue: false,
    },
    react: {
      useSuspense: false,
    },
  });

export default i18n;
