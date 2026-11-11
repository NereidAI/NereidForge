# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Tests for training-job delete/stop cascading to the cluster.

A UI delete must remove the Kubeflow TrainJob CR (and via owner GC its
JobSet/pods), not just the DB row; a UI stop of a running job must terminate
the KFP run and delete the TrainJob so trainer pods actually stop.
"""

import types

from app.services.jobs import JobService


class _FakeDB:
    def __init__(self):
        self.deleted = []
        self.commits = 0

    def delete(self, row):
        self.deleted.append(row)

    def commit(self):
        self.commits += 1


class _FakeCluster:
    def __init__(self, available=True, fail=False):
        self._available = available
        self.fail = fail
        self.deleted = []

    @property
    def available(self):
        return self._available

    def delete_trainjobs(self, artifact_id):
        if self.fail:
            raise RuntimeError("api error")
        self.deleted.append(artifact_id)
        return [f"train-{artifact_id}"]


class _FakeKFP:
    def __init__(self):
        self.cancelled = []

    def cancel_run(self, run_id):
        self.cancelled.append(run_id)


def _svc(job, cluster, kfp=None):
    svc = JobService.__new__(JobService)
    svc.db = _FakeDB()
    svc.kfp = kfp or _FakeKFP()
    svc.get_job = lambda jt, tenant, jid: job if jid == job.id else None
    svc._train_log_client = lambda: cluster
    return svc


def test_delete_training_job_cascades_to_trainjob_and_terminates_run():
    job = types.SimpleNamespace(
        id="job-1", type="training", status="running", kfp_run_id="run-1",
    )
    cluster = _FakeCluster()
    kfp = _FakeKFP()
    svc = _svc(job, cluster, kfp)

    assert svc.delete_job("training", "default", "job-1") is True

    assert cluster.deleted == ["job-1"]
    assert kfp.cancelled == ["run-1"]
    assert svc.db.deleted == [job]
    assert svc.db.commits == 1


def test_delete_keeps_db_row_when_cluster_delete_fails():
    job = types.SimpleNamespace(
        id="job-2", type="training", status="completed", kfp_run_id="run-2",
    )
    cluster = _FakeCluster(fail=True)
    svc = _svc(job, cluster)

    try:
        svc.delete_job("training", "default", "job-2")
    except RuntimeError:
        pass
    else:  # pragma: no cover - the failure must propagate
        raise AssertionError("cluster failure must propagate")

    # The row stays so the UI action can be retried (no orphaned CR).
    assert svc.db.deleted == []
    assert svc.db.commits == 0


def test_delete_completed_job_skips_terminate_but_still_deletes_cr():
    job = types.SimpleNamespace(
        id="job-3", type="training", status="completed", kfp_run_id="run-3",
    )
    cluster = _FakeCluster()
    kfp = _FakeKFP()
    svc = _svc(job, cluster, kfp)

    svc.delete_job("training", "default", "job-3")

    assert cluster.deleted == ["job-3"]
    assert kfp.cancelled == []
    assert svc.db.deleted == [job]


def test_delete_non_training_job_does_not_touch_cluster():
    job = types.SimpleNamespace(
        id="job-4", type="evaluation", status="completed", kfp_run_id="run-4",
    )
    cluster = _FakeCluster()
    svc = _svc(job, cluster)

    assert svc.delete_job("evaluation", "default", "job-4") is True
    assert cluster.deleted == []
    assert svc.db.deleted == [job]


def test_delete_unavailable_cluster_still_deletes_db_row():
    # Dev box without kube config: observability/cleanup degrades gracefully.
    job = types.SimpleNamespace(
        id="job-5", type="training", status="running", kfp_run_id="run-5",
    )
    cluster = _FakeCluster(available=False)
    svc = _svc(job, cluster)

    assert svc.delete_job("training", "default", "job-5") is True
    assert svc.db.deleted == [job]
