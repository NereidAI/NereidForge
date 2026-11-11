// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

import React from 'react';
import { Input, Select, Space } from 'antd';
import { SearchOutlined } from '@ant-design/icons';

// One option in the filter dropdown.
export interface FilterOption {
  value: string;
  label: string;
}

interface FilterSearchBarProps {
  // Filter dropdown (optional: pages with no filter render search only).
  filterPlaceholder?: string;
  filterOptions?: FilterOption[];
  filterValue?: string;
  onFilterChange?: (value: string) => void;
  // Name search box (optional: pages with no search render filter only).
  searchPlaceholder?: string;
  searchValue?: string;
  onSearchChange?: (value: string) => void;
}

// Standard right-aligned list toolbar: filter dropdown followed by a search
// input (magnifier icon). Every list page uses this so layout and behavior
// stay identical.
const FilterSearchBar: React.FC<FilterSearchBarProps> = ({
  filterPlaceholder, filterOptions, filterValue, onFilterChange,
  searchPlaceholder, searchValue, onSearchChange,
}) => (
  <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: 12 }}>
    <Space>
      {filterOptions && onFilterChange && (
        <Select
          allowClear
          style={{ width: 170 }}
          placeholder={filterPlaceholder}
          value={filterValue || undefined}
          onChange={(v) => onFilterChange(v || '')}
          options={filterOptions}
        />
      )}
      {onSearchChange && (
        <Input
          allowClear
          prefix={<SearchOutlined />}
          style={{ width: 240 }}
          placeholder={searchPlaceholder}
          value={searchValue}
          onChange={(e) => onSearchChange(e.target.value)}
        />
      )}
    </Space>
  </div>
);

export default FilterSearchBar;
