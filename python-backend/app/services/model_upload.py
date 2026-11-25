# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Validate model-version uploads and safely materialize artifact bundles.

Single checkpoints keep their spooled, streaming multipart files. Archives are
extracted into a private temporary directory under the configured file/count
limits, validated completely, and exposed through one uniform streaming API.
No model code is imported or executed during upload.
"""
from __future__ import annotations

import io
import stat
import tarfile
import tempfile
import zipfile
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterator

from starlette.datastructures import UploadFile

from app.core.config import config
from app.services.model_format import (
    ARCHIVE_SUFFIXES,
    FORMAT_OPENVINO_IR,
    FORMAT_PADDLE,
    artifact_format_definition,
    artifact_format_for_primary,
    normalize_artifact_relative_path,
)


class ArtifactUploadError(ValueError):
    """ArtifactUploadError reports a user-correctable artifact or archive contract failure."""


@dataclass
class PreparedArtifactFile:
    """PreparedArtifactFile exposes one validated file without assuming its storage backend."""

    relative_path: str
    size: int
    _stream: BinaryIO | None = None
    _path: Path | None = None

    @contextmanager
    def open(self) -> Iterator[BinaryIO]:
        """open yields a binary stream positioned at zero and restores multipart streams."""
        if self._path is not None:
            with self._path.open("rb") as stream:
                yield stream
            return
        if self._stream is None:
            raise ArtifactUploadError(f"upload source is unavailable: {self.relative_path}")
        position = self._stream.tell()
        try:
            self._stream.seek(0)
            yield self._stream
        finally:
            # Storage implementations may take ownership and close the source.
            # That terminal state needs no rewind; seeking it would also replace
            # an in-flight storage exception with a secondary cleanup failure.
            if not self._stream.closed:
                self._stream.seek(position)


@dataclass
class PreparedArtifactUpload:
    """PreparedArtifactUpload describes a fully validated version before any S3 write."""

    files: list[PreparedArtifactFile]
    primary_path: str
    format_id: str
    capabilities: list[str]
    layout: str
    runtime_requirements: dict[str, Any]

    @property
    def size_bytes(self) -> int:
        """size_bytes sums the complete bundle rather than hiding companion sizes."""
        return sum(item.size for item in self.files)


def _file_size(upload: UploadFile) -> int:
    """_file_size obtains a multipart length without consuming its stream."""
    if upload.size is not None:
        return int(upload.size)
    stream = upload.file
    position = stream.tell()
    stream.seek(0, io.SEEK_END)
    size = stream.tell()
    stream.seek(position)
    return int(size)


def _safe_member_path(name: str) -> str:
    """_safe_member_path applies archive-specific error context to path validation."""
    try:
        return normalize_artifact_relative_path(name)
    except ValueError as exc:
        raise ArtifactUploadError(f"unsafe archive path: {name}") from exc


def _check_file_budget(count: int, size: int, max_files: int, max_bytes: int) -> None:
    """_check_file_budget stops archive bombs before their content reaches disk."""
    if count > max_files:
        raise ArtifactUploadError(f"archive contains more than {max_files} files")
    if size > max_bytes:
        raise ArtifactUploadError(f"archive expanded size exceeds {max_bytes} bytes")


def _write_member(
    source: BinaryIO,
    destination: Path,
    expected_size: int,
    total_size: int,
    max_bytes: int,
) -> int:
    """_write_member enforces actual bytes as well as untrusted archive metadata."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with destination.open("wb") as output:
        while chunk := source.read(1024 * 1024):
            written += len(chunk)
            _check_file_budget(1, total_size + written, max_files=1, max_bytes=max_bytes)
            output.write(chunk)
    if written != expected_size:
        raise ArtifactUploadError("archive member size does not match its declared size")
    return written


def _extract_zip(
    upload: UploadFile, root: Path, max_files: int, max_bytes: int
) -> list[PreparedArtifactFile]:
    """_extract_zip safely extracts regular, unencrypted ZIP members."""
    files: list[PreparedArtifactFile] = []
    total_size = 0
    upload.file.seek(0)
    try:
        with zipfile.ZipFile(upload.file) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                mode = info.external_attr >> 16
                # DOS-created ZIPs often carry permissions but no Unix file
                # type. Only a present, non-regular type is unsafe.
                if stat.S_IFMT(mode) not in (0, stat.S_IFREG):
                    raise ArtifactUploadError(f"unsupported archive member: {info.filename}")
                if info.flag_bits & 0x1:
                    raise ArtifactUploadError("encrypted ZIP archives are not supported")
                relative = _safe_member_path(info.filename)
                files.append(PreparedArtifactFile(relative, 0))
                _check_file_budget(len(files), total_size, max_files, max_bytes)
                total_size += int(info.file_size)
                _check_file_budget(len(files), total_size, max_files, max_bytes)
                destination = root / relative
                with archive.open(info, "r") as source:
                    _write_member(source, destination, int(info.file_size), total_size - int(info.file_size), max_bytes)
                files[-1].size = int(info.file_size)
                files[-1]._path = destination
    except (zipfile.BadZipFile, NotImplementedError, OSError) as exc:
        raise ArtifactUploadError("invalid or unsupported ZIP archive") from exc
    return files


def _extract_tar(
    upload: UploadFile, root: Path, max_files: int, max_bytes: int
) -> list[PreparedArtifactFile]:
    """_extract_tar safely extracts regular gzip-tar members without following links."""
    files: list[PreparedArtifactFile] = []
    total_size = 0
    upload.file.seek(0)
    try:
        with tarfile.open(fileobj=upload.file, mode="r:gz") as archive:
            for info in archive:
                if info.isdir():
                    continue
                if not info.isreg():
                    raise ArtifactUploadError(f"unsupported archive member: {info.name}")
                relative = _safe_member_path(info.name)
                files.append(PreparedArtifactFile(relative, 0))
                _check_file_budget(len(files), total_size, max_files, max_bytes)
                total_size += int(info.size)
                _check_file_budget(len(files), total_size, max_files, max_bytes)
                source = archive.extractfile(info)
                if source is None:
                    raise ArtifactUploadError(f"archive member is unreadable: {info.name}")
                destination = root / relative
                with source:
                    _write_member(source, destination, int(info.size), total_size - int(info.size), max_bytes)
                files[-1].size = int(info.size)
                files[-1]._path = destination
    except (tarfile.TarError, OSError) as exc:
        raise ArtifactUploadError("invalid or unsupported tar.gz archive") from exc
    return files


def _strip_one_wrapper(files: list[PreparedArtifactFile]) -> list[PreparedArtifactFile]:
    """_strip_one_wrapper removes one archive-wide packaging directory, not model layout."""
    if not files:
        return files
    parts = [item.relative_path.split("/") for item in files]
    first = parts[0][0]
    if any(len(path) < 2 or path[0] != first for path in parts):
        return files
    for item in files:
        item.relative_path = item.relative_path.split("/", 1)[1]
    return files


def _deduplicate(files: list[PreparedArtifactFile]) -> None:
    """_deduplicate rejects normalized and case-folded key collisions."""
    seen: set[str] = set()
    for item in files:
        key = item.relative_path.casefold()
        if key in seen:
            raise ArtifactUploadError(f"duplicate archive path: {item.relative_path}")
        seen.add(key)


def _select_primary(
    files: list[PreparedArtifactFile], metadata: dict[str, Any]
) -> tuple[PreparedArtifactFile, dict]:
    """_select_primary resolves one explicit or unambiguous executable artifact."""
    by_path = {item.relative_path: item for item in files}
    declared_primary = str(metadata.get("primary_file") or "").strip()
    declared_format = str(metadata.get("artifact_format") or "").strip()
    definition = None
    if declared_format and declared_format != "auto":
        definition = artifact_format_definition(declared_format)
        if definition is None:
            raise ArtifactUploadError(f"unsupported model artifact format: {declared_format}")

    candidates = [
        (item, artifact_format_for_primary(item.relative_path))
        for item in files
        if artifact_format_for_primary(item.relative_path) is not None
    ]
    if definition is not None:
        candidates = [item for item in candidates if item[1]["id"] == definition["id"]]

    selected: PreparedArtifactFile | None = None
    selected_definition = definition
    if declared_primary:
        try:
            normalized = normalize_artifact_relative_path(declared_primary)
        except ValueError as exc:
            raise ArtifactUploadError("unsafe primary_file path") from exc
        selected = by_path.get(normalized)
        if selected is None:
            raise ArtifactUploadError(f"primary_file is absent from the upload: {normalized}")
        detected = artifact_format_for_primary(selected.relative_path)
        if detected is None or (definition is not None and detected["id"] != definition["id"]):
            raise ArtifactUploadError("primary_file does not match the declared artifact_format")
        selected_definition = detected
    elif len(candidates) == 1:
        selected, selected_definition = candidates[0]
    elif not candidates:
        raise ArtifactUploadError("unsupported model artifact: upload contains no supported primary artifact")
    else:
        raise ArtifactUploadError("multiple primary artifacts found; set primary_file explicitly")

    if selected is None or selected_definition is None:
        raise ArtifactUploadError("unable to resolve a primary model artifact")

    companion_suffix = None
    if selected_definition["id"] == FORMAT_OPENVINO_IR:
        companion_suffix = ".bin"
    elif selected_definition["id"] == FORMAT_PADDLE:
        companion_suffix = ".pdiparams"
    if companion_suffix is not None:
        base = selected.relative_path.rsplit(".", 1)[0]
        if base + companion_suffix not in by_path:
            raise ArtifactUploadError(
                f"{selected_definition['label']} requires a same-basename {companion_suffix} companion"
            )
    return selected, selected_definition


@contextmanager
def prepare_artifact_upload(
    uploads: list[UploadFile],
    metadata: dict[str, Any] | None = None,
    *,
    max_files: int | None = None,
    max_bytes: int | None = None,
) -> Iterator[PreparedArtifactUpload]:
    """prepare_artifact_upload validates files and archives before yielding an S3-ready bundle.

    Archive limits reuse storage.upload_max_files/upload_max_bytes. Direct files
    retain the historical streaming behavior and are not copied to temporary disk.
    The temporary extraction tree is deleted on every success and error path.
    """
    if not uploads:
        raise ArtifactUploadError("at least one model file is required")
    metadata = metadata or {}
    runtime_requirements = metadata.get("runtime_requirements") or {}
    if not isinstance(runtime_requirements, dict):
        raise ArtifactUploadError("runtime_requirements must be a JSON object")
    max_files = max_files or int(config.storage.upload_max_files)
    max_bytes = max_bytes or int(config.storage.upload_max_bytes)
    if len(uploads) > max_files:
        raise ArtifactUploadError(f"upload contains more than {max_files} files")

    archive_upload = next(
        (
            upload for upload in uploads
            if (upload.filename or "").lower().endswith(ARCHIVE_SUFFIXES)
        ),
        None,
    )
    if archive_upload is not None and len(uploads) != 1:
        raise ArtifactUploadError("an archive cannot be combined with loose files")

    if archive_upload is None:
        files: list[PreparedArtifactFile] = []
        for upload in uploads:
            try:
                relative = normalize_artifact_relative_path(upload.filename or "")
            except ValueError as exc:
                raise ArtifactUploadError(f"unsafe upload filename: {upload.filename}") from exc
            files.append(PreparedArtifactFile(relative, _file_size(upload), _stream=upload.file))
        _deduplicate(files)
        primary, definition = _select_primary(files, metadata)
        yield PreparedArtifactUpload(
            files=sorted(files, key=lambda item: item.relative_path),
            primary_path=primary.relative_path,
            format_id=definition["id"],
            capabilities=list(definition["capabilities"]),
            layout="single-file" if len(files) == 1 else "multi-file",
            runtime_requirements=deepcopy(runtime_requirements),
        )
        return

    filename = (archive_upload.filename or "").lower()
    layout = "zip" if filename.endswith(".zip") else ("tgz" if filename.endswith(".tgz") else "tar.gz")
    with tempfile.TemporaryDirectory(prefix="nf-model-upload-") as temporary:
        root = Path(temporary)
        if layout == "zip":
            files = _extract_zip(archive_upload, root, max_files, max_bytes)
        else:
            files = _extract_tar(archive_upload, root, max_files, max_bytes)
        if not files:
            raise ArtifactUploadError("archive contains no regular files")
        files = _strip_one_wrapper(files)
        _deduplicate(files)
        primary, definition = _select_primary(files, metadata)
        yield PreparedArtifactUpload(
            files=sorted(files, key=lambda item: item.relative_path),
            primary_path=primary.relative_path,
            format_id=definition["id"],
            capabilities=list(definition["capabilities"]),
            layout=layout,
            runtime_requirements=deepcopy(runtime_requirements),
        )
