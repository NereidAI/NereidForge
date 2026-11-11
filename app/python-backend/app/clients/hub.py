# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Kubeflow Hub (Model Registry) client using the official Python SDK.

Wraps the model_registry package (pip install model-registry==0.3.14).
The hub is the lifecycle source of truth for models; the NF backend writes to
it via this client while PostgreSQL keeps operational metadata for the UI.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from app.core.config import config

log = logging.getLogger(__name__)


class HubClient:
    """Thin wrapper around the ModelRegistry SDK with lazy init."""

    def __init__(self) -> None:
        self._client: Any = None
        self._init_error: Optional[Exception] = None

    def _ensure(self) -> Any:
        """Lazily initialize the ModelRegistry client."""
        if self._client is not None:
            return self._client
        if self._init_error is not None:
            return None
        host = config.hub.host
        if not host:
            self._init_error = ValueError("hub host not configured")
            return None
        try:
            from model_registry import ModelRegistry
            from urllib.parse import urlparse

            # SDK expects a full URL; parse out the port if present.
            parsed = urlparse(host)
            port = None
            if parsed.port:
                port = parsed.port
            self._client = ModelRegistry(
                server_address=host,
                port=port,
                author="nereidforge",
                is_secure=parsed.scheme == "https",
            )
        except Exception as e:  # noqa: BLE001
            if "Can't patch loop" in str(e):
                # model-registry SDK 0.3.14 calls nest_asyncio2.apply() on
                # init, which cannot patch a uvloop event loop (uvicorn's
                # default). Uvicorn must therefore use --loop asyncio.
                self._init_error = RuntimeError(
                    f"{e} — run uvicorn with '--loop asyncio' because uvloop "
                    "cannot be patched by model-registry SDK initialization"
                )
            else:
                self._init_error = e
            return None
        return self._client

    def enabled(self) -> bool:
        """Return True when the hub client is usable."""
        return self._ensure() is not None

    def register_model(
        self, name: str, uri: str, *, model_format_name: str, model_format_version: str,
        version: str, metadata: Optional[dict] = None, owner: str = "",
    ) -> Any:
        """Register a model + version to the hub."""
        client = self._ensure()
        if client is None:
            return None
        return client.register_model(
            name=name,
            uri=uri,
            model_format_name=model_format_name,
            model_format_version=model_format_version,
            version=version,
            metadata=metadata or {},
            owner=owner or None,
            description="registered by NereidForge",
        )

    def list_models(self) -> list[dict]:
        """List registered models from the hub."""
        client = self._ensure()
        if client is None:
            return []
        try:
            pager = client.get_registered_models()
            out = []
            for m in pager:
                out.append({
                    "name": m.name,
                    "description": m.description or "",
                    "external_id": getattr(m, "external_id", None) or "",
                    "state": getattr(m, "state", None) or "",
                })
            return out
        except Exception:
            return []

    def list_models_detailed(self) -> list[dict]:
        """List models from the hub with their versions (name/version/uri/state).

        The hub is the source of truth for model lifecycle metadata; this returns
        everything the NF model pages need without consulting PostgreSQL.
        Version dicts use the 'version' key to match the NF frontend shape.
        """
        client = self._ensure()
        if client is None:
            return []
        try:
            pager = client.get_registered_models()
            out = []
            for m in pager:
                state = str(getattr(m, "state", "") or "")
                if "ARCHIVED" in state.upper():
                    continue
                versions = []
                try:
                    vs = client.get_model_versions(m.name)
                    for v in vs:
                        state = str(getattr(v, "state", "") or "")
                        # Archived versions are the registry's soft-deleted
                        # records; returning them recreates a deleted row after
                        # the local PostgreSQL record has been removed.
                        if "ARCHIVED" in state.upper():
                            continue
                        versions.append({
                            "version": v.name,
                            "name": v.name,
                            "state": state.replace("ModelVersionState.", ""),
                            "author": getattr(v, "author", None) or "",
                            "external_id": getattr(v, "external_id", None) or "",
                            "create_time": getattr(v, "create_time_since_epoch", None),
                            "registered_model_id": getattr(v, "registered_model_id", None),
                            "id": getattr(v, "id", None),
                            "uri": "",
                            "metadata": {},
                            "status": "ready",
                            "file_path": "",
                            "size_bytes": 0,
                        })
                except Exception as e:  # noqa: BLE001 - keep model visible w/o versions
                    # A per-model version fetch failure must not be silent:
                    # the UI's Version dropdown then renders empty for an
                    # otherwise healthy model (deploy modal). The frontend
                    # re-fetches versions on demand; this log makes the
                    # transient registry error diagnosable.
                    log.warning("hub: get_model_versions failed for %s: %s", m.name, e)
                    versions = []
                out.append({
                    "name": m.name,
                    "description": m.description or "",
                    "owner": getattr(m, "owner", None) or "",
                    "external_id": getattr(m, "external_id", None) or "",
                    "state": state.replace("RegisteredModelState.", ""),
                    "create_time": getattr(m, "create_time_since_epoch", None),
                    "versions": versions,
                })
            return out
        except Exception:
            return []

    def get_model(self, name: str) -> Optional[dict]:
        """Get a single registered model by name."""
        client = self._ensure()
        if client is None:
            return None
        try:
            m = client.get_registered_model(name)
            return {
                "name": m.name,
                "description": m.description or "",
                "external_id": getattr(m, "external_id", None) or "",
                "state": getattr(m, "state", None) or "",
            }
        except Exception:
            return None

    def archive_model(self, name: str) -> None:
        """Archive a registered model, propagating failures to prevent false deletion."""
        client = self._ensure()
        if client is None:
            return
        from model_registry.types import RegisteredModelState

        m = client.get_registered_model(name)
        if m is None:
            return
        m.state = RegisteredModelState.ARCHIVED
        client.update(m)

    def archive_model_version(self, model_name: str, version: str) -> None:
        """Archive a model version, propagating failures to the deletion route."""
        client = self._ensure()
        if client is None:
            return
        from model_registry.types import ModelVersionState

        vs = client.get_model_versions(model_name)
        for v in vs:
            if v.name == version:
                v.state = ModelVersionState.ARCHIVED
                client.update(v)
                return


# Module-level singleton.
hub_client: HubClient = HubClient()
