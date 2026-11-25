# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Provide explicit non-production credentials before application modules import.

The application validates required secrets while loading configuration, which is
before test modules import clients and services. These process-local dummy values
keep unit tests independent of developer or deployment credential files; tests
that exercise missing-secret failures clear or replace them with monkeypatch.
"""

import os


_TEST_CREDENTIALS = {
    "NEREIDFORGE_DATABASE_PASSWORD": "pytest-database-password",
    "NEREIDFORGE_STORAGE_S3_ACCESS_KEY": "pytest-storage-access-key",
    "NEREIDFORGE_STORAGE_S3_SECRET_KEY": "pytest-storage-secret-key",
    "NEREIDFORGE_KFP_OBJECT_STORE_ACCESS_KEY": "pytest-kfp-access-key",
    "NEREIDFORGE_KFP_OBJECT_STORE_SECRET_KEY": "pytest-kfp-secret-key",
}

for _name, _value in _TEST_CREDENTIALS.items():
    os.environ[_name] = _value
