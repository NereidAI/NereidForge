# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""
KFP Client Wrapper

Provides convenient methods for interacting with KFP API.
"""

import logging
from typing import Any, Dict, List, Optional

import kfp
from kfp import dsl
from kfp.client import Client
from kfp_server_api import ApiException

logger = logging.getLogger(__name__)


class KFPClient:
    """Wrapper around KFP SDK Client with additional utilities."""

    def __init__(
        self,
        host: Optional[str] = None,
        namespace: str = "kubeflow",
        cookies: Optional[str] = None,
        verify_ssl: bool = True,
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize KFP client.

        Args:
            host: KFP API host (e.g., https://kubeflow.example.com)
            namespace: Kubernetes namespace for KFP
            cookies: Authentication cookies for managed KFP
            verify_ssl: Whether to verify SSL certificates
            headers: Extra HTTP headers, e.g. {"kubeflow-userid": "user@example.com"}
        """
        self.host = host
        self.namespace = namespace
        self.headers = headers or {}
        self.client = Client(
            host=host,
            namespace=namespace,
            cookies=cookies,
            verify_ssl=verify_ssl,
        )
        self._apply_headers()

    def _apply_headers(self) -> None:
        """Inject custom headers into the shared generated API client."""
        if not self.headers:
            return
        api_client = self.client._experiment_api.api_client
        api_client.default_headers.update(self.headers)

    def create_experiment(self, name: str, description: str = "") -> str:
        """Create or get experiment."""
        try:
            exp = self.client.create_experiment(name=name, description=description)
            return exp.experiment_id
        except ApiException as e:
            if e.status == 409:
                # Experiment exists, get it
                exps = self.client.list_experiments(filter=f"name={name}")
                if exps.experiments:
                    return exps.experiments[0].experiment_id
            raise

    def run_pipeline(
        self,
        experiment_id: str,
        pipeline_package_path: str,
        run_name: str,
        parameters: Dict[str, Any],
        enable_caching: bool = False,
    ) -> str:
        """Submit a pipeline run."""
        run = self.client.run_pipeline(
            experiment_id=experiment_id,
            job_name=run_name,
            pipeline_package_path=pipeline_package_path,
            params=parameters,
            enable_caching=enable_caching,
        )
        return run.run_id

    def wait_for_run(
        self,
        run_id: str,
        timeout: int = 3600,
        poll_interval: int = 30,
    ) -> Dict[str, Any]:
        """Wait for run completion and return final status."""
        import time

        start_time = time.time()
        while time.time() - start_time < timeout:
            run = self.client.get_run(run_id)
            state = (run.state or "").upper()
            if state in ["SUCCEEDED", "FAILED", "ERROR"]:
                return {
                    "run_id": run_id,
                    "status": state,
                    "finished_at": run.finished_at,
                    "error": run.error if state in ("FAILED", "ERROR") else None,
                }
            time.sleep(poll_interval)

        raise TimeoutError(f"Run {run_id} did not complete within {timeout} seconds")

    def get_run_logs(self, run_id: str) -> List[Dict[str, Any]]:
        """Get logs for a pipeline run."""
        run_detail = self.client.get_run(run_id)
        # Logs would be fetched from the run detail
        return []

    def list_pipelines(self, page_size: int = 100) -> List[Dict[str, Any]]:
        """List all pipelines."""
        result = self.client.list_pipelines(page_size=page_size)
        return [{"id": p.pipeline_id, "name": p.name} for p in result.pipelines]

    def upload_pipeline(
        self,
        pipeline_package_path: str,
        pipeline_name: str,
        description: str = "",
    ) -> str:
        """Upload a compiled pipeline."""
        pipeline = self.client.upload_pipeline(
            pipeline_package_path=pipeline_package_path,
            pipeline_name=pipeline_name,
            description=description,
        )
        return pipeline.pipeline_id

    def create_recurring_run(
        self,
        experiment_id: str,
        pipeline_package_path: str,
        run_name: str,
        parameters: Dict[str, Any],
        cron_expression: str,
        enabled: bool = True,
    ) -> str:
        """Create a recurring run (scheduled pipeline)."""
        recurring_run = self.client.create_recurring_run(
            experiment_id=experiment_id,
            pipeline_package_path=pipeline_package_path,
            job_name=run_name,
            params=parameters,
            cron_expression=cron_expression,
            enabled=enabled,
        )
        return recurring_run.recurring_run_id

    def delete_run(self, run_id: str) -> None:
        """Delete a pipeline run."""
        self.client.delete_run(run_id)

    def archive_run(self, run_id: str) -> None:
        """Archive a pipeline run."""
        self.client.archive_run(run_id)

    def unarchive_run(self, run_id: str) -> None:
        """Unarchive a pipeline run."""
        self.client.unarchive_run(run_id)


def compile_pipeline(
    pipeline_func: dsl.PipelineFunc,
    package_path: str,
    type_check: bool = True,
) -> None:
    """
    Compile a KFP pipeline function to YAML.

    Args:
        pipeline_func: The pipeline function decorated with @dsl.pipeline
        package_path: Output path for compiled YAML
        type_check: Whether to perform type checking
    """
    from kfp import compiler

    compiler.Compiler().compile(
        pipeline_func=pipeline_func,
        package_path=package_path,
        type_check=type_check,
    )
    logger.info(f"Pipeline compiled to {package_path}")


def create_pipeline_from_yaml(yaml_path: str) -> dsl.PipelineFunc:
    """Load a pipeline from compiled YAML (for inspection)."""
    import yaml
    with open(yaml_path) as f:
        return yaml.safe_load(f)
