# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Lock in semi-auto pre-annotation re-run behaviour.

Regression history: re-clicking "pre-annotate" processed 0 tasks because
run_preannotate_task skipped every task with is_labeled=True (set by an earlier
empty annotation), and each run POSTed a brand-new prediction so stale boxes
stacked up. The contract pinned here:
  1. a task's auto prediction is replaced (old predictions deleted, one fresh
     saved), never duplicated, on every run;
  2. the predictions list endpoint is parsed whether LS returns a bare list or
     a paginated {"results": [...]} envelope.
"""

from app.clients.labelstudio import LabelStudioClient


def _client() -> LabelStudioClient:
    # Bypass __init__ (it reads global config); the methods under test only use
    # the HTTP helpers, which are stubbed per test.
    return LabelStudioClient.__new__(LabelStudioClient)


def test_replace_prediction_deletes_then_saves():
    """replace_prediction must drop prior predictions and save exactly one new one."""
    c = _client()
    calls: list[tuple] = []

    def fake_get(path, params=None):
        calls.append(("get", path, params))
        # LS returns a bare list for the task-filtered predictions endpoint.
        return [{"id": 7}, {"id": 9}]

    def fake_delete(path):
        calls.append(("delete", path))
        return None

    def fake_post(path, body):
        calls.append(("post", path, body))
        return {"id": 100}

    c._get = fake_get          # type: ignore[assignment]
    c._delete = fake_delete    # type: ignore[assignment]
    c._post = fake_post        # type: ignore[assignment]

    c.replace_prediction(task_id=3, result=[{"x": 1}], score=0.9, model_version="yolo")

    # Both stale predictions are deleted by their integer pk ...
    assert ("delete", "/api/predictions/7/") in calls
    assert ("delete", "/api/predictions/9/") in calls
    # ... and exactly one fresh prediction is written for the task.
    posts = [c for c in calls if c[0] == "post"]
    assert len(posts) == 1
    _, path, body = posts[0]
    assert path == "/api/predictions/"
    assert body["task"] == 3
    assert body["result"] == [{"x": 1}]
    assert body["model_version"] == "yolo"


def test_replace_prediction_failed_delete_still_saves():
    """A stale prediction that fails to delete must not block the fresh save."""
    c = _client()
    c._get = lambda path, params=None: [{"id": 5}]          # type: ignore[assignment]

    def boom(path):
        raise RuntimeError("ls delete failed")

    saved: dict = {}

    def fake_post(path, body):
        saved.update(body)
        return {"id": 1}

    c._delete = boom                                          # type: ignore[assignment]
    c._post = fake_post                                       # type: ignore[assignment]

    c.replace_prediction(task_id=8, result=[], score=0.0, model_version="yolo")

    assert saved.get("task") == 8


def test_list_predictions_accepts_envelope():
    """list_predictions must unwrap a paginated {"results": [...]} envelope."""
    c = _client()
    c._get = lambda path, params=None: {"results": [{"id": 1}]}  # type: ignore[assignment]
    assert [p["id"] for p in c.list_predictions(2)] == [1]

    c2 = _client()
    c2._get = lambda path, params=None: [{"id": 4}, {"id": 5}]   # type: ignore[assignment]
    assert [p["id"] for p in c2.list_predictions(2)] == [4, 5]
