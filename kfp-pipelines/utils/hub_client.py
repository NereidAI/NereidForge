# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""
Hub Client

Client for interacting with NereidForge Model Hub API.
"""

import json
import logging
from typing import Any, Dict, List, Optional

import requests
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class ModelInfo(BaseModel):
    name: str
    version: str
    description: str = ""
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ArtifactInfo(BaseModel):
    name: str
    type: str  # model, dataset, checkpoint, etc.
    format: str  # onnx, pytorch, gguf, etc.
    uri: str
    size_bytes: int
    checksum: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class HubClient:
    """Client for NereidForge Model Hub API."""

    def __init__(
        self,
        base_url: str = "http://hub.nereidforge.svc.cluster.local:8080",
        token: str = "",
        timeout: int = 30,
    ):
        """
        Initialize Hub client.

        Args:
            base_url: Hub API base URL
            token: Authentication token
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self.session = requests.Session()
        if token:
            self.session.headers.update({"Authorization": f"Bearer {token}"})
        self.session.headers.update({"Content-Type": "application/json"})

    def _request(self, method: str, path: str, **kwargs) -> requests.Response:
        """Make HTTP request with retry logic."""
        url = f"{self.base_url}{path}"
        kwargs.setdefault("timeout", self.timeout)

        @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
        def _do_request():
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response

        return _do_request()

    # Model operations
    def register_model(self, model: ModelInfo, artifact_uri: str) -> Dict[str, Any]:
        """Register a new model version."""
        payload = {
            "name": model.name,
            "version": model.version,
            "description": model.description,
            "tags": model.tags,
            "metadata": model.metadata,
            "artifact_uri": artifact_uri,
        }
        response = self._request("POST", "/api/v1/models", json=payload)
        return response.json()

    def get_model(self, name: str, version: str) -> Dict[str, Any]:
        """Get model information."""
        response = self._request("GET", f"/api/v1/models/{name}/versions/{version}")
        return response.json()

    def list_models(self, name: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """List models."""
        params = {"limit": limit}
        if name:
            params["name"] = name
        response = self._request("GET", "/api/v1/models", params=params)
        return response.json().get("models", [])

    def delete_model(self, name: str, version: str) -> None:
        """Delete a model version."""
        self._request("DELETE", f"/api/v1/models/{name}/versions/{version}")

    # Artifact operations
    def register_artifact(self, artifact: ArtifactInfo, model_name: str, model_version: str) -> Dict[str, Any]:
        """Register an artifact for a model."""
        payload = artifact.model_dump()
        payload["model_name"] = model_name
        payload["model_version"] = model_version
        response = self._request("POST", "/api/v1/artifacts", json=payload)
        return response.json()

    def get_artifact(self, artifact_id: str) -> Dict[str, Any]:
        """Get artifact information."""
        response = self._request("GET", f"/api/v1/artifacts/{artifact_id}")
        return response.json()

    def list_artifacts(
        self,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
        artifact_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List artifacts."""
        params = {"limit": limit}
        if model_name:
            params["model_name"] = model_name
        if model_version:
            params["model_version"] = model_version
        if artifact_type:
            params["type"] = artifact_type
        response = self._request("GET", "/api/v1/artifacts", params=params)
        return response.json().get("artifacts", [])

    # Search
    def search_models(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search models."""
        payload = {"query": query}
        if filters:
            payload["filters"] = filters
        response = self._request("POST", "/api/v1/search/models", json=payload)
        return response.json().get("results", [])

    # Health check
    def health_check(self) -> Dict[str, Any]:
        """Check Hub health."""
        response = self._request("GET", "/health")
        return response.json()


def create_hub_client_from_env() -> HubClient:
    """Create Hub client from environment variables."""
    import os

    base_url = os.getenv("HUB_API_URL", "http://hub.nereidforge.svc.cluster.local:8080")
    token = os.getenv("HUB_TOKEN", "")
    return HubClient(base_url=base_url, token=token)
