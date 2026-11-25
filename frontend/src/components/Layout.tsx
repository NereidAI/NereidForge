// Copyright (C) 2026 YIQISOFT
// SPDX-License-Identifier: Apache-2.0

import React, { useState } from 'react';
import { Outlet, NavLink, useNavigate, useLocation } from 'react-router-dom';
import {
  Layout as AntLayout,
  Menu,
  Breadcrumb,
  Avatar,
  Dropdown,
  Button,
  Tooltip,
  Badge,
  Select,
  Typography,
} from 'antd';
import type { MenuProps } from 'antd';
import {
  MenuUnfoldOutlined,
  MenuFoldOutlined,
  UserOutlined,
  LogoutOutlined,
  BellOutlined,
  DashboardOutlined,
  DatabaseOutlined,
  EditOutlined,
  RocketOutlined,
  ThunderboltOutlined,
  BarChartOutlined,
  AppstoreOutlined,
  CloudServerOutlined,
  SafetyOutlined,
  QuestionCircleOutlined,
  DeploymentUnitOutlined,
  ContainerOutlined,
  FileTextOutlined,
  InboxOutlined,
  FolderOutlined,
  ExperimentOutlined,
  CloudOutlined,
  SettingOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import i18n from '../i18n';
import { useAuthStore } from '@store/authStore';

const { Header, Sider, Content } = AntLayout;

interface LayoutProps {
  children?: React.ReactNode;
}

// Sidebar menu tree. Leaf keys are route paths (clicking navigates); group
// keys are grp-* placeholders that only expand/collapse the submenu.
interface MenuLeaf {
  key: string;
  icon: React.ReactNode;
  label: string;
  path: string;
  adminOnly?: boolean;
}
interface MenuGroup {
  key: string;
  icon: React.ReactNode;
  label: string;
  adminOnly?: boolean;
  children: MenuEntry[];
}
type MenuEntry = MenuLeaf | MenuGroup;

const menuItems: MenuEntry[] = [
  {
    key: 'dashboard',
    icon: <DashboardOutlined />,
    label: 'nav.dashboard',
    path: '/dashboard',
  },
  {
    key: 'grp-data',
    icon: <FolderOutlined />,
    label: 'nav.groupData',
    children: [
      {
        key: 'datasets',
        icon: <DatabaseOutlined />,
        label: 'nav.datasets',
        path: '/datasets',
      },
      {
        key: 'annotations',
        icon: <EditOutlined />,
        label: 'nav.annotations',
        path: '/annotations',
      },
    ],
  },
  {
    key: 'grp-training',
    icon: <RocketOutlined />,
    label: 'nav.groupTraining',
    children: [
      {
        key: 'training',
        icon: <ExperimentOutlined />,
        label: 'nav.training',
        path: '/training',
      },
      {
        key: 'notebooks',
        icon: <FileTextOutlined />,
        label: 'nav.notebooks',
        path: '/notebooks',
      },
      {
        key: 'optimization',
        icon: <ThunderboltOutlined />,
        label: 'nav.optimization',
        path: '/optimization',
      },
      {
        key: 'evaluation',
        icon: <BarChartOutlined />,
        label: 'nav.evaluation',
        path: '/evaluation',
      },
    ],
  },
  {
    key: 'grp-serving',
    icon: <CloudOutlined />,
    label: 'nav.groupServing',
    children: [
      {
        key: 'models',
        icon: <AppstoreOutlined />,
        label: 'nav.models',
        path: '/models',
      },
      {
        key: 'deployments',
        icon: <DeploymentUnitOutlined />,
        label: 'nav.deployments',
        path: '/deployments',
      },
      {
        key: 'mlBackend',
        icon: <CloudServerOutlined />,
        label: 'nav.mlBackend',
        path: '/ml-backend',
      },
    ],
  },
  {
    key: 'grp-system',
    icon: <SettingOutlined />,
    label: 'nav.groupSystem',
    adminOnly: true,
    children: [
      {
        key: 'baseModels',
        icon: <InboxOutlined />,
        label: 'nav.baseModels',
        path: '/base-models',
        adminOnly: true,
      },
      {
        key: 'images',
        icon: <ContainerOutlined />,
        label: 'nav.images',
        path: '/images',
        adminOnly: true,
      },
      {
        key: 'settings',
        icon: <SafetyOutlined />,
        label: 'nav.settings',
        path: '/settings',
        adminOnly: true,
      },
    ],
  },
  {
    key: 'help',
    icon: <QuestionCircleOutlined />,
    label: 'nav.help',
    path: '/help',
  },
];

export const Layout: React.FC<LayoutProps> = ({ children }) => {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const location = useLocation();
  const { user, isAdmin, logout } = useAuthStore();
  const [collapsed, setCollapsed] = useState(false);
  // Expanded submenu groups (controlled while the sider is expanded; in the
  // collapsed state antd renders submenus as flyouts without openKeys).
  const [openKeys, setOpenKeys] = useState<string[]>([]);

  // Filter admin-only leaves, then drop groups left without children.
  const filterTree = (items: MenuEntry[]): MenuEntry[] =>
    items.flatMap<MenuEntry>((item) => {
      if ('children' in item) {
        const children = filterTree(item.children);
        return children.length ? [{ ...item, children }] : [];
      }
      return item.adminOnly && !isAdmin() ? [] : [item];
    });
  const filteredMenuItems = filterTree(menuItems);

  const handleMenuClick = (key: string) => {
    // Group keys are placeholders; only leaf paths navigate.
    if (key.startsWith('/')) navigate(key);
  };

  // Find the leaf (longest matching route prefix) to highlight, plus the
  // parent group whose submenu should stay open.
  const findActive = (
    pathname: string, items: MenuEntry[], parent = '',
  ): { leaf: string; group: string } => {
    let best = { leaf: '', group: '' };
    for (const item of items) {
      if ('children' in item) {
        const found = findActive(pathname, item.children, item.key);
        if (found.leaf.length > best.leaf.length) best = found;
      } else if (
        (pathname === item.path || pathname.startsWith(item.path + '/'))
        && item.path.length > best.leaf.length
      ) {
        best = { leaf: item.path, group: parent };
      }
    }
    return best;
  };
  const active = findActive(location.pathname, filteredMenuItems);

  // Navigating into a group opens that group automatically.
  React.useEffect(() => {
    if (active.group) {
      setOpenKeys((prev) => (prev.includes(active.group) ? prev : [...prev, active.group]));
    }
  }, [active.group]);

  // Map the tree to antd Menu item props (children -> submenu).
  const renderMenu = (items: MenuEntry[]): NonNullable<MenuProps['items']> =>
    items.map((item) => {
      if ('children' in item) {
        return {
          key: item.key,
          icon: item.icon,
          label: t(item.label),
          children: renderMenu(item.children),
        };
      }
      return { key: item.path, icon: item.icon, label: t(item.label) };
    });

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const userMenuItems = [
    {
      label: t('components.layout.profile'),
      key: 'profile',
      icon: <UserOutlined />,
      onClick: () => navigate('/profile'),
    },
    {
      type: 'divider' as const,
    },
    {
      label: t('components.layout.logout'),
      key: 'logout',
      icon: <LogoutOutlined />,
      onClick: handleLogout,
      danger: true,
    },
  ];

  const languageItems = [
    { value: 'en', label: 'English' },
    { value: 'zh', label: '中文' },
  ];

  return (
    <AntLayout style={{ minHeight: '100vh' }}>
      <Sider
        trigger={null}
        collapsible
        collapsed={collapsed}
        width={256}
        collapsedWidth={80}
        theme="light"
        style={{ borderRight: '1px solid #f0f0f0', height: '100vh', position: 'fixed', left: 0, top: 0, zIndex: 100 }}
      >
        <div
          style={{
            height: 64,
            display: 'flex',
            alignItems: 'center',
            justifyContent: collapsed ? 'center' : 'flex-start',
            padding: '0 16px',
            borderBottom: '1px solid #f0f0f0',
          }}
        >
          {!collapsed && (
            <span style={{ fontSize: 18, fontWeight: 600, color: '#1677ff' }}>
              NereidForge
            </span>
          )}
          {collapsed && (
            <span style={{ fontSize: 18, fontWeight: 600, color: '#1677ff' }}>
              NF
            </span>
          )}
        </div>
        <Menu
          mode="inline"
          theme="light"
          selectedKeys={active.leaf ? [active.leaf] : []}
          onClick={({ key }) => handleMenuClick(key)}
          {...(!collapsed ? { openKeys, onOpenChange: setOpenKeys } : {})}
          style={{ borderRight: 'none', marginTop: 8 }}
          items={renderMenu(filteredMenuItems)}
        />
        <div
          style={{
            position: 'absolute',
            bottom: 0,
            left: 0,
            right: 0,
            padding: '12px 16px',
            borderTop: '1px solid #f0f0f0',
            background: '#fff',
            fontSize: 12,
            color: '#999',
            textAlign: collapsed ? 'center' : 'left',
          }}
        >
          {!collapsed && <span>Version: </span>}
          <Typography.Text ellipsis>{import.meta.env.VITE_APP_VERSION || 'dev'}</Typography.Text>
        </div>
      </Sider>
      <AntLayout style={{ marginLeft: collapsed ? 80 : 256, transition: 'margin-left 0.2s' }}>
        <Header
          style={{
            padding: '0 24px',
            height: 64,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            background: '#fff',
            borderBottom: '1px solid #f0f0f0',
            position: 'sticky',
            top: 0,
            zIndex: 50,
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
            <Button
              type="text"
              icon={collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
              onClick={() => setCollapsed(!collapsed)}
              aria-label={collapsed ? t('components.layout.sidebarExpand') : t('components.layout.sidebarCollapse')}
              style={{ fontSize: 16 }}
            />
            <Breadcrumb
              style={{ fontSize: 14 }}
              items={location.pathname.split('/').filter(Boolean).map((segment, index, arr) => {
                const path = '/' + arr.slice(0, index + 1).join('/');
                const isLast = index === arr.length - 1;
                // Use i18n titles for known route segments and type names for dynamic segments such as IDs.
                let label: string;
                let clickable = false;
                const tKey = segment === 'base-models'
                  ? 'pages.baseModels.title'
                  : `pages.${segment}.title`;
                const translated = t(tKey);
                if (translated !== tKey) {
                  label = translated;
                  clickable = !isLast;
                } else if (/^ds-/.test(segment)) {
                  // Dataset ID: prefix the shortened ID with the localized dataset title.
                  label = `${t('pages.datasets.title')} · ${segment.slice(0, 10)}`;
                } else if (/^prj-/.test(segment)) {
                  label = `${t('pages.annotations.title')} · ${segment.slice(0, 10)}`;
                } else if (/^job-/.test(segment)) {
                  label = `${t('pages.training.title')} · ${segment.slice(0, 10)}`;
                } else if (/^m-/.test(segment)) {
                  label = `${t('pages.models.title')} · ${segment.slice(0, 10)}`;
                } else if (/^versions?$/.test(segment)) {
                  label = t('pages.datasets.newVersion');
                } else {
                  label = segment.length > 16 ? segment.slice(0, 16) + '…' : segment;
                }
                // Only intermediate static routes are clickable; dynamic links such as
                // /datasets/ds-xxx do not exist and cause a suspend error.
                return {
                  title: clickable && !isLast ? <NavLink to={path}>{label}</NavLink> : label,
                };
              })}
            />
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
            <Select
              value={i18n.language.startsWith('zh') ? 'zh' : 'en'}
              onChange={(value) => i18n.changeLanguage(value)}
              options={languageItems}
              size="small"
              style={{ width: 100 }}
              aria-label="Language"
            />
            <Tooltip title={t('components.layout.notifications')}>
              <Badge count={0} showZero={false}>
                <Button type="text" icon={<BellOutlined />} />
              </Badge>
            </Tooltip>
            <Dropdown
              menu={{ items: userMenuItems }}
              trigger={['click']}
              placement="bottomRight"
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer', padding: '4px 8px', borderRadius: 6 }}>
                <Avatar size={32} src={user?.email ? `https://www.gravatar.com/avatar/${user.email}` : undefined}>
                  {user?.username?.[0]?.toUpperCase() || 'U'}
                </Avatar>
                {!collapsed && (
                  <span style={{ fontSize: 14, fontWeight: 500 }}>
                    {user?.username || 'User'}
                  </span>
                )}
              </div>
            </Dropdown>
          </div>
        </Header>
        <Content style={{ padding: 24, background: '#f5f5f5', minHeight: 'calc(100vh - 64px)' }}>
          {children || <Outlet />}
        </Content>
      </AntLayout>
    </AntLayout>
  );
};

export default Layout;
