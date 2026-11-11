# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""
Label Studio Client Wrapper

Provides convenient methods for interacting with the Label Studio API
(projects, tasks, annotation export).
"""

import json
import logging
from typing import Any, Dict, List, Optional

import requests
from label_studio_sdk import Client as LSClient

logger = logging.getLogger(__name__)


def _safe_get(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name)
    except Exception:
        return default


def _project_to_dict(project: Any) -> Dict[str, Any]:
    """Normalize an SDK Project object into a plain dict."""
    params = _safe_get(project, "params")
    if isinstance(params, dict) and params:
        return dict(params)
    return {
        "id": _safe_get(project, "id"),
        "title": _safe_get(project, "title"),
        "description": _safe_get(project, "description"),
        "created_at": _safe_get(project, "created_at"),
    }


class LabelStudioClient:
    """Wrapper around the Label Studio SDK with additional utilities."""

    def __init__(
        self,
        url: str,
        api_token: str,
        verify_ssl: bool = True,
        timeout: int = 60,
    ):
        """
        Initialize Label Studio client.

        Args:
            url: Label Studio base URL (e.g., https://labelstudio.example.com)
            api_token: Label Studio user API token
            verify_ssl: Whether to verify SSL certificates
            timeout: Request timeout in seconds
        """
        self.url = url.rstrip("/")
        self.api_token = api_token
        self.timeout = timeout
        session = requests.Session()
        session.verify = verify_ssl
        self._client = LSClient(
            url=self.url,
            api_key=self.api_token,
            session=session,
        )

    def check_connection(self) -> bool:
        """Verify the API token is valid."""
        try:
            return bool(self._client.check_connection())
        except Exception as e:
            logger.error("Label Studio connection check failed: %s", e)
            return False

    def get_versions(self) -> Dict[str, Any]:
        """Get server version info."""
        return self._client.get_versions()

    def list_projects(self) -> List[Dict[str, Any]]:
        """List all annotation projects."""
        projects = self._client.list_projects()
        return [_project_to_dict(p) for p in projects]

    def get_project(self, project_id: int) -> Dict[str, Any]:
        """Get a single project by ID."""
        return _project_to_dict(self._client.get_project(project_id))

    def create_project(
        self,
        title: str,
        label_config: str,
        description: str = "",
    ) -> Dict[str, Any]:
        """Create a new annotation project."""
        project = self._client.create_project(
            title=title,
            label_config=label_config,
            description=description,
        )
        return _project_to_dict(project)

    def delete_project(self, project_id: int) -> bool:
        """Delete a project by ID."""
        return bool(self._client.delete_project(project_id))

    def _get(self, path: str, **kwargs) -> Any:
        resp = requests.get(
            f"{self.url}{path}",
            headers={"Authorization": f"Token {self.api_token}"},
            timeout=self.timeout,
            **kwargs,
        )
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, json: Any, **kwargs) -> Any:
        resp = requests.post(
            f"{self.url}{path}",
            headers={"Authorization": f"Token {self.api_token}"},
            json=json,
            timeout=self.timeout,
            **kwargs,
        )
        resp.raise_for_status()
        return resp.json()

    def list_tasks(
        self,
        project_id: int,
        page_size: int = 100,
        page: int = 1,
    ) -> List[Dict[str, Any]]:
        """List tasks of a project."""
        return self._get(
            f"/api/projects/{project_id}/tasks",
            params={"page": page, "page_size": page_size},
        )

    def create_tasks(self, project_id: int, tasks: List[Dict[str, Any]]) -> Any:
        """Create tasks in a project (bulk import; returns import result)."""
        if len(tasks) == 1:
            return self._post(f"/api/projects/{project_id}/tasks", json=tasks[0])
        files = {"file": ("tasks.json", json.dumps(tasks), "application/json")}
        resp = requests.post(
            f"{self.url}/api/projects/{project_id}/import",
            headers={"Authorization": f"Token {self.api_token}"},
            files=files,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def export_tasks(self, project_id: int, export_type: str = "JSON") -> List[Dict[str, Any]]:
        """Export annotations from a project."""
        return self._get(
            f"/api/projects/{project_id}/export",
            params={"exportType": export_type},
        )
