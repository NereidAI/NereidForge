# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Training runtime serialization tests for completed trainer containers."""

from datetime import datetime, timezone
from types import SimpleNamespace

from app.api.routers.jobs import JobMetricsResponse
from app.clients.k8s import K8sTrainLogClient


class _CustomApi:
    """Return one TrainJob used to discover its trainer Pod."""

    def list_namespaced_custom_object(self, *_args, **_kwargs):
        return {"items": [{"metadata": {"name": "train-job-1"}}]}


class _CoreApi:
    """Return a terminated trainer container with exact lifecycle timestamps."""

    def list_namespaced_pod(self, *_args, **_kwargs):
        started = datetime(2026, 11, 2, 9, 16, 34, tzinfo=timezone.utc)
        finished = datetime(2026, 11, 2, 10, 14, 19, tzinfo=timezone.utc)
        pod = SimpleNamespace(
            metadata=SimpleNamespace(name="trainer-pod-1"),
            spec=SimpleNamespace(
                containers=[SimpleNamespace(name="node")], node_name="xpu-node-a",
            ),
            status=SimpleNamespace(
                phase="Succeeded",
                start_time=started,
                container_statuses=[SimpleNamespace(
                    name="node",
                    state=SimpleNamespace(
                        terminated=SimpleNamespace(started_at=started, finished_at=finished),
                        running=None,
                    ),
                )],
            ),
        )
        return SimpleNamespace(items=[pod])


def test_runtime_exposes_trainer_container_finish_time():
    """Completed BYOT jobs need real workload bounds when metrics omit duration."""
    client = K8sTrainLogClient.__new__(K8sTrainLogClient)
    client.namespace = "nereidforge"
    client._apis = lambda: (_CustomApi(), _CoreApi())

    pods = client.find_trainer_pods("job-1")

    assert len(pods) == 1
    assert pods[0]["started"] == "2026-11-02T09:16:34+00:00"
    assert pods[0]["finished"] == "2026-11-02T10:14:19+00:00"


def test_metrics_response_preserves_the_versioned_trainer_schema():
    """The API exposes the same schema field used by external trainer artifacts."""
    response = JobMetricsResponse.model_validate({
        "schema": "nf.training-metrics/1", "train_time_s": 3465.0, "mAP50": 0.8,
    })

    assert response.model_dump(by_alias=True, exclude_none=True) == {
        "schema": "nf.training-metrics/1", "train_time_s": 3465.0, "mAP50": 0.8,
    }
