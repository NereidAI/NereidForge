# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Tests for BaseModelService (S3 base-model pool).

Uses an in-memory SQLite session (JSONB compiled as JSON) for real ORM
queries plus an in-memory S3 double capturing uploaded objects.
"""

import io
import types
import zipfile

import pytest
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import Session

from app.db.base import Base
from app.services.base_models import (
    BaseModelError,
    BaseModelService,
    infer_stock_arch,
)
from app.models.orm import BaseModel as BaseModelORM


# -- stock architecture inference -------------------------------------------

def test_infer_stock_arch_recognises_ultralytics_families():
    """Stock YOLO/RT-DETR file names encode the architecture in the stem."""
    assert infer_stock_arch("yolov8n.pt") == "yolov8n"
    assert infer_stock_arch("yolov8n-seg.pt") == "yolov8n-seg"
    assert infer_stock_arch("yolo11x-cls.pt") == "yolo11x-cls"
    assert infer_stock_arch("yolov5nu.pt") == "yolov5nu"
    assert infer_stock_arch("yolov9e.pth") == "yolov9e"
    assert infer_stock_arch("yolov10b.safetensors") == "yolov10b"
    assert infer_stock_arch("yolo12n-pose.pt") == "yolo12n-pose"
    assert infer_stock_arch("rtdetr-l.pt") == "rtdetr-l"
    assert infer_stock_arch("rtdetr-v2-x.pt") == "rtdetr-v2-x"


def test_infer_stock_arch_strips_s3_path_and_suffix():
    """The basename of a pool URI resolves the same as the bare file name."""
    assert infer_stock_arch(
        "s3://nereidforge/base-models/cv/yolov8n.pt"
    ) == "yolov8n"
    assert infer_stock_arch("/weights/YOLO11N-SEG.PT") == "yolo11n-seg"


def test_infer_stock_arch_rejects_arbitrary_names():
    """User checkpoint names and non-weight files carry no stock arch."""
    assert infer_stock_arch("my-finetuned-2026.pt") == ""
    assert infer_stock_arch("fasterrcnn_resnet50_fpn.pth") == ""
    assert infer_stock_arch("yolov8n.onnx") == ""
    assert infer_stock_arch("yolov8n") == ""
    assert infer_stock_arch("") == ""
    # A valid stem followed by a directory separator is not a file name.
    assert infer_stock_arch("cv/yolov8n.pt/extra") == ""


@compiles(JSONB, "sqlite")
def _sqlite_jsonb(element, compiler, **kw):
    """Compile Postgres JSONB as SQLite JSON for the test session."""
    return "JSON"


class _MemS3:
    """In-memory S3 double: stores filer path -> bytes."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def upload_fileobj(self, filer_path, fileobj):
        fileobj.seek(0)
        self.objects[filer_path] = fileobj.read()

    def put_object(self, filer_path, data):
        self.objects[filer_path] = bytes(data)

    def read_object(self, filer_path):
        return self.objects[filer_path]

    def delete_object(self, filer_path):
        self.objects.pop(filer_path, None)

    def object_size(self, uri_or_path):
        path = uri_or_path
        if path.startswith("s3://"):
            path = self.s3_uri_to_filer_path(path)
        return len(self.objects.get(path, b""))

    @staticmethod
    def s3_uri_to_filer_path(uri):
        assert uri.startswith("s3://")
        rest = uri[len("s3://"):]
        bucket, _, key = rest.partition("/")
        return f"/buckets/{bucket}/{key}"

    def walk(self, filer_path):
        return [p for p in self.objects if p.startswith(filer_path.rstrip("/") + "/")]

    def list_files_with_size(self, filer_path):
        base = filer_path.rstrip("/") + "/"
        return [
            (p[len(base):], len(data))
            for p, data in self.objects.items()
            if p.startswith(base) and "/" not in p[len(base):]
        ]

    def list_entries(self, filer_path):
        base = filer_path.rstrip("/") + "/"
        out = []
        seen = set()
        for p in self.objects:
            if not p.startswith(base):
                continue
            rel = p[len(base):]
            if "/" in rel:
                name = rel.split("/")[0]
                if name not in seen:
                    seen.add(name)
                    out.append(types.SimpleNamespace(name=name, is_dir=True))
            else:
                out.append(types.SimpleNamespace(name=rel, is_dir=False))
        return out


@pytest.fixture
def ctx():
    """Build (service, mem s3) on a fresh in-memory database."""
    from datetime import datetime

    from sqlalchemy import event

    engine = create_engine("sqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def _register_now(dbapi_conn, _record):
        # ORM columns use server_default=text("now()"), which SQLite lacks.
        dbapi_conn.create_function("now", 0, lambda: datetime.now().isoformat())

    Base.metadata.create_all(engine)
    db = Session(engine)
    mem = _MemS3()
    svc = BaseModelService(db, mem, bucket="nereidforge")
    yield svc, mem
    db.close()


def _bytes(n=10):
    """Return a small binary payload."""
    return b"x" * n


# -- cv upload -------------------------------------------------------------

def test_cv_upload_creates_row(ctx):
    svc, mem = ctx
    row = svc.upload_cv("default", "yolov8n.pt", io.BytesIO(_bytes(42)), "alice")
    assert row["kind"] == "cv"
    assert row["name"] == "yolov8n.pt"
    assert row["uri"] == "s3://nereidforge/base-models/cv/yolov8n.pt"
    assert row["size_bytes"] == 42
    assert mem.objects["/buckets/nereidforge/base-models/cv/yolov8n.pt"] == _bytes(42)


def test_upload_is_committed_for_a_fresh_session(ctx):
    # Regression: uploads must commit, not only flush, or rows vanish when
    # the request session closes.
    svc, mem = ctx
    svc.upload_cv("default", "yolov8n.pt", io.BytesIO(b"x"), "alice")
    fresh = Session(svc.db.bind)
    try:
        assert fresh.query(BaseModelORM).count() == 1
    finally:
        fresh.close()


def test_cv_upload_rejects_bad_suffix(ctx):
    svc, _ = ctx
    with pytest.raises(BaseModelError):
        svc.upload_cv("default", "model.txt", io.BytesIO(b"x"), "alice")


def test_cv_repeated_upload_upserts_same_row(ctx):
    svc, _ = ctx
    first = svc.upload_cv("default", "yolov8n.pt", io.BytesIO(b"a"), "alice")
    second = svc.upload_cv("default", "yolov8n.pt", io.BytesIO(b"bb"), "bob")
    assert first["id"] == second["id"]
    assert second["size_bytes"] == 2
    items, total = svc.list_paged("default")
    assert total == 1
    assert items[0]["uploaded_by"] == "bob"


# -- metadata: framework / task / status / source --------------------------

def test_cv_upload_infers_framework_and_task(ctx):
    svc, _ = ctx
    det = svc.upload_cv("default", "yolo11n.pt", io.BytesIO(b"x"), "alice")
    assert det["framework"] == "pytorch"
    assert det["task"] == "detection"
    assert det["status"] == "develop"
    assert det["source"] == "manual"


def test_cv_upload_infers_seg_and_cls_tasks(ctx):
    svc, _ = ctx
    seg = svc.upload_cv("default", "yolo11s-seg.pt", io.BytesIO(b"x"), "alice")
    cls = svc.upload_cv("default", "yolo11n-cls.pt", io.BytesIO(b"x"), "alice")
    assert seg["task"] == "segmentation"
    assert cls["task"] == "classification"


def test_cv_upload_records_explicit_metadata(ctx):
    svc, _ = ctx
    row = svc.upload_cv(
        "default", "custom.onnx", io.BytesIO(b"x"), "alice",
        framework="onnx", task="detection", status="production",
        description="exported detector",
    )
    assert row["framework"] == "onnx"
    assert row["status"] == "production"
    assert row["description"] == "exported detector"
    assert row["source"] == "manual"


def test_llm_upload_defaults_task_llm(ctx):
    svc, _ = ctx
    files = [("config.json", io.BytesIO(b"{}"))]
    row = svc.upload_llm_files(
        "default", "qwen3", files, "alice", task="vlm", status="original",
    )
    assert row["task"] == "vlm"
    assert row["status"] == "original"
    assert row["source"] == "manual"


# -- llm zip upload --------------------------------------------------------

def _zip_bytes(files):
    """Build a zip in memory from name -> bytes."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf


def test_llm_zip_upload_extracts_files(ctx):
    svc, mem = ctx
    zf = _zip_bytes({"config.json": b"{}", "model.safetensors": _bytes(20)})
    row = svc.upload_llm_zip("default", "qwen3", zf, "alice")
    assert row["kind"] == "llm"
    assert row["name"] == "qwen3"
    assert row["uri"] == "s3://nereidforge/base-models/llm/qwen3/"
    assert row["size_bytes"] == 22
    assert "/buckets/nereidforge/base-models/llm/qwen3/config.json" in mem.objects
    assert "/buckets/nereidforge/base-models/llm/qwen3/model.safetensors" in mem.objects


def test_llm_zip_rejects_missing_config(ctx):
    svc, _ = ctx
    zf = _zip_bytes({"model.safetensors": b"xx"})
    with pytest.raises(BaseModelError):
        svc.upload_llm_zip("default", "qwen3", zf, "alice")


def test_llm_zip_rejects_zip_slip(ctx):
    svc, _ = ctx
    zf = _zip_bytes({"../evil.pt": b"x", "config.json": b"{}"})
    with pytest.raises(BaseModelError):
        svc.upload_llm_zip("default", "qwen3", zf, "alice")


def test_llm_bad_zip_rejected(ctx):
    svc, _ = ctx
    with pytest.raises(BaseModelError):
        svc.upload_llm_zip("default", "qwen3", io.BytesIO(b"not a zip"), "alice")


# -- llm multi-file upload -------------------------------------------------

def test_llm_multi_file_upload(ctx):
    svc, mem = ctx
    files = [
        ("config.json", io.BytesIO(b"{}")),
        ("model.safetensors", io.BytesIO(_bytes(5))),
    ]
    row = svc.upload_llm_files("default", "qwen3", files, "alice")
    assert row["name"] == "qwen3"
    assert row["size_bytes"] == 7


def test_llm_multi_file_requires_name(ctx):
    # The router enforces this; the service validates path safety.
    svc, _ = ctx
    with pytest.raises(BaseModelError):
        svc.upload_llm_files("default", "../bad", [("config.json", io.BytesIO(b"{}"))], "alice")


# -- listing ---------------------------------------------------------------

def test_list_filtered_by_kind(ctx):
    svc, _ = ctx
    svc.upload_cv("default", "yolov8n.pt", io.BytesIO(b"x"), "alice")
    svc.upload_llm_files("default", "qwen3", [("config.json", io.BytesIO(b"{}"))], "alice")
    cv_items, cv_total = svc.list_paged("default", kind="cv")
    llm_items, llm_total = svc.list_paged("default", kind="llm")
    assert cv_total == 1 and cv_items[0]["kind"] == "cv"
    assert llm_total == 1 and llm_items[0]["kind"] == "llm"
    assert len(svc.list_for_training("default", "cv")) == 1


def test_list_search_by_name(ctx):
    svc, _ = ctx
    svc.upload_cv("default", "yolov8n.pt", io.BytesIO(b"x"), "alice")
    svc.upload_cv("default", "yolo11n.pt", io.BytesIO(b"y"), "alice")
    items, total = svc.list_paged("default", search="11n")
    assert total == 1 and items[0]["name"] == "yolo11n.pt"
    # Search combines with the kind filter and is case-insensitive.
    assert svc.list_paged("default", kind="llm", search="YOLO")[1] == 0


# -- delete ----------------------------------------------------------------

def test_delete_removes_row_and_cv_file(ctx):
    svc, mem = ctx
    row = svc.upload_cv("default", "yolov8n.pt", io.BytesIO(b"x"), "alice")
    assert svc.delete("default", row["id"]) is True
    assert "/buckets/nereidforge/base-models/cv/yolov8n.pt" not in mem.objects
    assert svc.get("default", row["id"]) is None
    assert svc.delete("default", row["id"]) is False


def test_delete_llm_removes_prefix(ctx):
    svc, mem = ctx
    zf = _zip_bytes({"config.json": b"{}", "weights/part1": b"abc"})
    row = svc.upload_llm_zip("default", "qwen3", zf, "alice")
    assert svc.delete("default", row["id"]) is True
    assert mem.objects == {}


# -- reconcile metadata ----------------------------------------------------

def test_reconcile_backfills_metadata(ctx):
    svc, mem = ctx
    # Simulate a pre-existing stock checkpoint in S3 with no pool row.
    mem.objects["/buckets/nereidforge/base-models/cv/yolo11x-seg.pt"] = b"abc"
    n = svc.reconcile_with_s3("default")
    assert n == 1
    items, total = svc.list_paged("default", kind="cv")
    assert total == 1
    assert items[0]["source"] == "s3-reconcile"
    assert items[0]["status"] == "original"
    assert items[0]["framework"] == "pytorch"
    assert items[0]["task"] == "segmentation"
