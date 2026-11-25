# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Driver translation tests and opt-in PostgreSQL integration coverage.

Set NEREIDFORGE_TEST_POSTGRESQL_DSN to a dedicated, disposable PostgreSQL
database to run the integration test. The test creates and alters the NF schema,
so this variable must never point at a development or production database.
"""

from __future__ import annotations

import os
import ssl
import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session

from app.db.session import Base, _engine_parameters, engine
from app.models.orm import AnnotationProject, Job


def test_disable_ssl_uses_pg8000_without_libpq_query() -> None:
    """The tracked disable configuration must never attempt an SSL upgrade."""
    url, connect_args = _engine_parameters(
        "postgresql://nf:secret@db.example:5432/nereidforge?sslmode=disable"
    )

    assert url.drivername == "postgresql+pg8000"
    assert url.username == "nf"
    assert url.password == "secret"
    assert "sslmode" not in url.query
    assert connect_args["ssl_context"] is False


def test_prefer_ssl_preserves_pg8000_fallback() -> None:
    """A None context retains pg8000's SSL-then-plaintext negotiation."""
    _, connect_args = _engine_parameters("postgresql://nf@db/nf?sslmode=prefer")

    assert connect_args["ssl_context"] is None


def test_require_ssl_forces_encryption_without_verification() -> None:
    """A True context preserves libpq require mode's encryption-only policy."""
    _, connect_args = _engine_parameters("postgresql://nf@db/nf?sslmode=require")

    assert connect_args["ssl_context"] is True


def test_verify_ca_disables_only_hostname_verification() -> None:
    """CA verification still validates the chain without checking the host name."""
    _, connect_args = _engine_parameters("postgresql://nf@db/nf?sslmode=verify-ca")
    context = connect_args["ssl_context"]

    assert isinstance(context, ssl.SSLContext)
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname is False


def test_verify_full_enables_certificate_and_hostname_verification() -> None:
    """The strictest libpq mode maps to Python's default verified context."""
    _, connect_args = _engine_parameters("postgresql://nf@db/nf?sslmode=verify-full")
    context = connect_args["ssl_context"]

    assert isinstance(context, ssl.SSLContext)
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname is True


@pytest.mark.parametrize("ssl_mode", ["allow", "invalid"])
def test_unsupported_ssl_modes_fail_before_connecting(ssl_mode: str) -> None:
    """Unsupported policies must fail explicitly instead of changing security."""
    with pytest.raises(ValueError, match="Unsupported database sslmode"):
        _engine_parameters(f"postgresql://nf@db/nf?sslmode={ssl_mode}")


def test_application_engine_uses_pg8000() -> None:
    """The production engine itself must select the replacement dialect."""
    assert engine.url.drivername == "postgresql+pg8000"
    assert engine.dialect.name == "postgresql"
    assert engine.dialect.driver == "pg8000"


def test_postgresql_schema_types_transactions_and_pool() -> None:
    """Exercise the real driver against an explicitly selected disposable database."""
    dsn = os.environ.get("NEREIDFORGE_TEST_POSTGRESQL_DSN")
    if not dsn:
        pytest.skip("NEREIDFORGE_TEST_POSTGRESQL_DSN selects the disposable test database")

    from app.main import (
        _ensure_columns,
        _relax_dataset_version_constraints,
        _relax_job_timestamp_constraints,
    )

    url, connect_args = _engine_parameters(dsn)
    pg_engine = create_engine(
        url,
        connect_args=connect_args,
        pool_size=2,
        max_overflow=1,
        pool_pre_ping=True,
    )
    suffix = uuid.uuid4().hex
    job_id = f"pg8000-job-{suffix}"
    project_id = f"pg8000-project-{suffix}"
    rollback_id = f"pg8000-rollback-{suffix}"
    instant = datetime(2026, 11, 11, 12, 30, 45, 123456, tzinfo=timezone.utc)

    try:
        Base.metadata.create_all(pg_engine)
        with pg_engine.begin() as connection:
            # Clean up after an interrupted earlier run before restoring the
            # legacy NOT NULL constraints on this dedicated test database.
            connection.execute(
                text("DELETE FROM annotation_projects WHERE tenant_id = :tenant"),
                {"tenant": "pg8000-test"},
            )
            connection.execute(
                text("DELETE FROM jobs WHERE tenant_id = :tenant"),
                {"tenant": "pg8000-test"},
            )
            # Reproduce the old constraints and missing column that startup
            # migrations must repair before the ORM writes through pg8000.
            connection.execute(text("ALTER TABLE jobs ALTER COLUMN started_at SET NOT NULL"))
            connection.execute(text("ALTER TABLE jobs ALTER COLUMN completed_at SET NOT NULL"))
            connection.execute(
                text("ALTER TABLE dataset_versions ALTER COLUMN source_project_id SET NOT NULL")
            )
            connection.execute(text("ALTER TABLE annotation_projects DROP COLUMN preannotate_model"))

        _ensure_columns(pg_engine)
        _relax_job_timestamp_constraints(pg_engine)
        _relax_dataset_version_constraints(pg_engine)
        # A second startup must be a no-op rather than repeating destructive DDL.
        _ensure_columns(pg_engine)
        _relax_job_timestamp_constraints(pg_engine)
        _relax_dataset_version_constraints(pg_engine)

        inspector = inspect(pg_engine)
        job_columns = {column["name"]: column for column in inspector.get_columns("jobs")}
        project_columns = {column["name"] for column in inspector.get_columns("annotation_projects")}
        version_columns = {
            column["name"]: column for column in inspector.get_columns("dataset_versions")
        }
        assert job_columns["started_at"]["nullable"] is True
        assert job_columns["completed_at"]["nullable"] is True
        assert version_columns["source_project_id"]["nullable"] is True
        assert "preannotate_model" in project_columns

        with Session(pg_engine, expire_on_commit=False) as session:
            job = Job(
                id=job_id,
                tenant_id="pg8000-test",
                type="training",
                name="pg8000 integration",
                params={"nested": {"items": [1, None, {"enabled": True}]}},
                config={},
                results={},
                metrics={},
                started_at=instant,
            )
            project = AnnotationProject(
                id=project_id,
                tenant_id="pg8000-test",
                name="pg8000 integration",
                schema_={"labels": ["object"], "nullable": None},
                preannotate_model=None,
            )
            session.add_all([job, project])
            session.commit()
            session.refresh(job)
            session.refresh(project)

            assert job.params == {"nested": {"items": [1, None, {"enabled": True}]}}
            assert job.started_at == instant
            assert job.started_at is not None and job.started_at.tzinfo is not None
            assert job.created_at.tzinfo is not None
            assert project.schema_ == {"labels": ["object"], "nullable": None}
            assert project.preannotate_model is None

            row = session.execute(
                text("SELECT :value AS value, CAST(:seen_at AS timestamptz) AS seen_at"),
                {"value": "bound-parameter", "seen_at": instant},
            ).one()
            assert row.value == "bound-parameter"
            assert row.seen_at == instant

            session.add(
                Job(
                    id=rollback_id,
                    tenant_id="pg8000-test",
                    type="training",
                    name="rolled back",
                )
            )
            session.flush()
            session.rollback()
            assert session.get(Job, rollback_id) is None

            session.delete(job)
            session.delete(project)
            session.commit()

        # Returning and reusing a pooled connection exercises pre_ping on the
        # second checkout, in addition to normal driver query execution.
        for _ in range(2):
            with pg_engine.connect() as connection:
                assert connection.exec_driver_sql("SELECT 1").scalar_one() == 1
    finally:
        with pg_engine.begin() as connection:
            connection.execute(
                text("DELETE FROM annotation_projects WHERE tenant_id = :tenant"),
                {"tenant": "pg8000-test"},
            )
            connection.execute(
                text("DELETE FROM jobs WHERE tenant_id = :tenant"),
                {"tenant": "pg8000-test"},
            )
        pg_engine.dispose()
