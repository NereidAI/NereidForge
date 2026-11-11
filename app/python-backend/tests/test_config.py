# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Lock in config loading behaviour that regressed once already.

Regression history: NEREIDFORGE_* env vars silently stopped applying (load_config
only read the rendered /etc/nereidforge/config.yaml), so run-dev.sh's
NEREIDFORGE_AUTH_JWT_SECRET was ignored and login in dev 500'd with
"HMAC key must not be empty". These tests pin the contract:
  1. env overrides file values, 2. rendered yaml path works with complete data.
"""

from pathlib import Path

import pytest

from app.core.config import load_config


@pytest.fixture
def rendered(tmp_path: Path) -> Path:
    p = tmp_path / "config.yaml"
    # Minimal rendered config with explicit keys that environment variables can override.
    p.write_text(
        "auth:\n"
        "  jwt_secret: ''\n"
        "  enabled: false\n"
        "database:\n"
        "  password: pytest-database-password\n"
        "storage:\n"
        "  s3_access_key: pytest-storage-access-key\n"
        "  s3_secret_key: pytest-storage-secret-key\n"
        "  upload_max_files: 200\n"
        "  upload_max_bytes: 1073741824\n"
    )
    return p


def test_env_override_wins_over_rendered_file(rendered: Path, monkeypatch):
    """env NEREIDFORGE_* must override the rendered config values."""
    monkeypatch.setenv("NEREIDFORGE_AUTH_ENABLED", "true")
    monkeypatch.setenv("NEREIDFORGE_AUTH_JWT_SECRET", "pytest-only-jwt-signing-key-0123456789abcdef0123456789abcdef")
    monkeypatch.setenv("NEREIDFORGE_ENV", "dev")

    cfg = load_config(rendered)

    assert cfg.auth.enabled is True
    assert cfg.auth.jwt_secret == "pytest-only-jwt-signing-key-0123456789abcdef0123456789abcdef"


def test_upload_limits_env_override(rendered: Path, monkeypatch):
    """Dataset upload limits must be tunable without rebuilding the backend."""
    monkeypatch.setenv("NEREIDFORGE_STORAGE_UPLOAD_MAX_FILES", "350")
    monkeypatch.setenv("NEREIDFORGE_STORAGE_UPLOAD_MAX_BYTES", "2147483648")

    cfg = load_config(rendered)

    assert cfg.storage.upload_max_files == 350
    assert cfg.storage.upload_max_bytes == 2147483648
