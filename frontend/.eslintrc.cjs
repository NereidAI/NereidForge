// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

// ESLint checks core JavaScript correctness and React hook boundaries.
// Strict TypeScript semantics remain the TypeScript compiler's responsibility;
// generated assets and installed dependencies are outside this source gate.
module.exports = {
  root: true,
  env: { browser: true, es2020: true },
  extends: [
    "eslint:recommended",
    "plugin:@typescript-eslint/eslint-recommended",
    "plugin:react-hooks/recommended",
  ],
  ignorePatterns: ["dist", "node_modules"],
  parser: "@typescript-eslint/parser",
  plugins: ["@typescript-eslint"],
  rules: {
    // The base rule misreads TypeScript interfaces, enums, and declarations.
    "no-unused-vars": "off",
    "@typescript-eslint/no-unused-vars": ["error", { argsIgnorePattern: "^_" }],
  },
};
