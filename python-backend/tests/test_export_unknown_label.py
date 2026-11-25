# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Regression: annotated export must keep labels absent from the schema.

A project whose schema labels drifted from the source dataset class names
(e.g. schema "unmask" vs imported data.yaml class "no-mask") used to silently
drop every box whose label was not in the schema category map. The export
became single-class, negatives vanished, and the trained model hallucinated
the surviving class.
"""

import json
import types

from app.services.project import ProjectService


class _FakeS3:
    def __init__(self, tasks):
        self.files: dict[str, bytes] = {}
        self._tasks = tasks

    def s3_uri_to_filer_path(self, uri):
        return "/buckets/" + uri[len("s3://"):]

    def read_object(self, path):
        # Unreadable image bytes -> default 640x480 dims in the exporter.
        return b""

    def list_entries(self, prefix):
        return []

    def put_object(self, path, data):
        self.files[path] = data


def _region(label, x, y, w=10.0, h=10.0):
    return {
        "type": "rectanglelabels",
        "value": {"x": x, "y": y, "width": w, "height": h,
                  "rectanglelabels": [label]},
    }


def _service(project, tasks):
    svc = ProjectService.__new__(ProjectService)

    class _LS:
        def enabled(self):
            return True

        def export_annotations(self, ls_id):
            return tasks

    svc.ls = _LS()
    svc.cvat = None
    svc.s3 = _FakeS3(tasks)

    src = types.SimpleNamespace(
        id="dsv-1", uri="s3://nereidforge/datasets/ds-1/v1/")

    class _DatasetSvc:
        def get_dataset_version(self, tenant, ds_id, ver):
            return src

        def get_dataset(self, tenant, ds_id):
            return {"version": 1}

    svc.dataset = _DatasetSvc()

    class _DB:
        def __init__(self, project):
            self.added = []
            self._project = project
            self._scalar_calls = 0

        def scalar(self, stmt):
            # export_as_dataset: 1st scalar loads the project (called by the
            # method itself); later scalar is the Dataset existence check.
            self._scalar_calls += 1
            return self._project if self._scalar_calls == 1 else None

        def add(self, row):
            self.added.append(row)

        def commit(self):
            pass

    svc.db = _DB(project)
    return svc


def test_export_keeps_labels_missing_from_schema():
    project = types.SimpleNamespace(
        id="prj-1",
        name="mask-wearing",
        external_id="7",
        dataset_id="ds-1",
        dataset_version=1,
        task_type="detection",
        annotation_tool="labelstudio",
        schema_={"type": "rectanglelabels", "labels": ["mask", "unmask"]},
        tenant_id="default",
    )
    tasks = [
        {"data": {"image": "s3://nereidforge/datasets/ds-1/v1/images/a.jpg"},
         "annotations": [{"result": [_region("mask", 10, 10)]}]},
        {"data": {"image": "s3://nereidforge/datasets/ds-1/v1/images/b.jpg"},
         "annotations": [{"result": [_region("no-mask", 20, 20)]}]},
    ]
    svc = _service(project, tasks)
    svc.export_as_dataset("default", "prj-1", "exp", "", "coco", "tester")

    coco_path = next(k for k in svc.s3.files if k.endswith("annotations.json"))
    coco = json.loads(svc.s3.files[coco_path])
    cats = {c["name"]: c["id"] for c in coco["categories"]}
    assert cats == {"mask": 0, "unmask": 1, "no-mask": 2}
    # Both boxes survived; the off-schema label kept its own category id.
    assert len(coco["annotations"]) == 2
    by_img = {a["image_id"]: a["category_id"] for a in coco["annotations"]}
    assert by_img == {1: 0, 2: 2}
