# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""SeaweedFS S3 client (boto3), mirroring the Go SeaweedFSClient semantics.

All backend data access goes through S3 (bucket "nereidforge"). Filer paths of
the form /buckets/{bucket}/{key...} are converted to S3 bucket/key pairs,
preserving the filer-path signatures used across the Go codebase.
"""
from __future__ import annotations

import zipfile
from dataclasses import dataclass
from typing import BinaryIO, Optional

import boto3
from botocore.client import Config as BotoConfig

from app.core.config import config


@dataclass
class Entry:
    """A directory listing entry."""
    name: str
    is_dir: bool


class SeaweedFSClient:
    """S3 protocol client for SeaweedFS (minio-go equivalent using boto3)."""

    def __init__(self) -> None:
        endpoint = config.storage.s3_endpoint or config.storage.endpoint
        # Normalize filer 8888 -> S3 gateway 8333 for legacy configs.
        if endpoint and endpoint.split(":")[-1].startswith("8888"):
            host = endpoint.rsplit(":", 1)[0]
            endpoint = f"{host}:8333"
        self.endpoint = (endpoint or "http://localhost:8333").rstrip("/")
        self.bucket = config.storage.bucket or "nereidforge"
        self.client = boto3.client(
            "s3",
            endpoint_url=self.endpoint,
            aws_access_key_id=config.storage.s3_access_key or "",
            aws_secret_access_key=config.storage.s3_secret_key or "",
            region_name="us-east-1",
            config=BotoConfig(signature_version="s3v4"),
        )

    def enabled(self) -> bool:
        """Return True when the S3 client is usable."""
        return True

    def s3_endpoint(self) -> str:
        """Return the S3 gateway endpoint URL."""
        return self.endpoint

    def s3_bucket(self) -> str:
        """Return the default bucket name."""
        return self.bucket

    def presigned_cluster_get_url(self, bucket: str, key: str, expires: int = 3600) -> str:
        """Return a presigned GET URL signed against the cluster S3 endpoint.

        In-cluster consumers (the LS ML backend, CVAT, isvc) fetch images by plain
        HTTP GET and cannot sign S3 requests; SeaweedFS denies anonymous reads
        (403). We sign a short-lived URL with the *cluster* endpoint so the SigV4
        Host header matches what the consumer will hit. The worker signs offline
        (no network needed), so this works in dev too where the worker uses a
        localhost port-forward but the consumer runs in the cluster.
        """
        ep = (config.storage.s3_cluster_endpoint or self.endpoint).rstrip("/")
        cluster_client = boto3.client(
            "s3",
            endpoint_url=ep,
            aws_access_key_id=config.storage.s3_access_key or "",
            aws_secret_access_key=config.storage.s3_secret_key or "",
            region_name="us-east-1",
            config=BotoConfig(signature_version="s3v4"),
        )
        return cluster_client.generate_presigned_url(
            "get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=expires
        )

    # -- path helpers -------------------------------------------------------

    @staticmethod
    def s3_uri_to_filer_path(uri: str) -> str:
        """Convert s3://bucket/prefix/ to /buckets/bucket/prefix/ (compat layer)."""
        if not uri.startswith("s3://"):
            return ""
        return "/buckets/" + uri[len("s3://"):].rstrip("/")

    @staticmethod
    def tenant_uri(uri: str, tenant_id: str) -> str:
        """Inject the tenant prefix into an s3:// URI (default tenant unchanged)."""
        if not tenant_id or tenant_id == "default":
            return uri
        rest = uri[len("s3://"):] if uri.startswith("s3://") else ""
        if rest == uri:
            return uri
        idx = rest.find("/")
        if idx < 0:
            return f"s3://{rest}/{tenant_id}"
        bucket = rest[:idx]
        tail = rest[idx + 1:]
        if tail.startswith(tenant_id + "/"):
            return uri
        return f"s3://{bucket}/{tenant_id}/{tail}"

    @staticmethod
    def tenant_filer_path(path: str, tenant_id: str) -> str:
        """Inject the tenant prefix into a /buckets/... filer path."""
        if not tenant_id or tenant_id == "default":
            return path
        rest = path[len("/buckets/"):] if path.startswith("/buckets/") else ""
        if rest == path:
            return path
        idx = rest.find("/")
        if idx < 0:
            return path
        bucket = rest[:idx]
        tail = rest[idx + 1:]
        if tail.startswith(tenant_id + "/"):
            return path
        return f"/buckets/{bucket}/{tenant_id}/{tail}"

    @staticmethod
    def _split_filer_path(filer_path: str) -> tuple[str, str]:
        """Split /buckets/{bucket}/{key...} into (bucket, key)."""
        p = filer_path[len("/buckets/"):] if filer_path.startswith("/buckets/") else filer_path
        idx = p.find("/")
        if idx < 0:
            return p, ""
        return p[:idx], p[idx + 1:].strip("/")

    # -- object operations --------------------------------------------------

    def list_entries(self, filer_path: str) -> list[Entry]:
        """List immediate entries under a filer path (name + is_dir)."""
        bucket, prefix = self._split_filer_path(filer_path)
        prefix = prefix.strip("/")
        if prefix and not prefix.endswith("/"):
            prefix += "/"
        entries: list[Entry] = []
        kwargs = {"Bucket": bucket, "Prefix": prefix, "Delimiter": "/"}
        while True:
            resp = self.client.list_objects_v2(**kwargs)
            for o in resp.get("Contents", []) or []:
                key = o["Key"]
                rel = key[len(prefix):] if prefix else key
                if rel and not rel.endswith("/"):
                    entries.append(Entry(name=rel.rstrip("/"), is_dir=False))
            for p in resp.get("CommonPrefixes", []) or []:
                rel = p["Prefix"][len(prefix):].rstrip("/") if prefix else p["Prefix"].rstrip("/")
                if rel:
                    entries.append(Entry(name=rel, is_dir=True))
            if resp.get("IsTruncated") and resp.get("NextContinuationToken"):
                kwargs["ContinuationToken"] = resp["NextContinuationToken"]
            else:
                break
        return entries

    def list_files_with_size(self, filer_path: str) -> list[tuple[str, int]]:
        """List all files (recursively) under a filer path as (rel_path, size).

        rel_path is relative to filer_path with no leading slash; size is the
        object byte length from the LIST response (no per-object HEAD needed).
        Used by the dataset file browser so a version's full tree is fetched in
        a single paginated LIST instead of N HEAD requests.
        """
        bucket, prefix = self._split_filer_path(filer_path)
        prefix = prefix.strip("/")
        if prefix and not prefix.endswith("/"):
            prefix += "/"
        out: list[tuple[str, int]] = []
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        while True:
            resp = self.client.list_objects_v2(**kwargs)
            for o in resp.get("Contents", []) or []:
                key = o["Key"]
                if key.endswith("/"):
                    continue
                rel = key[len(prefix):] if prefix else key
                if rel:
                    out.append((rel, int(o.get("Size", 0) or 0)))
            if resp.get("IsTruncated") and resp.get("NextContinuationToken"):
                kwargs["ContinuationToken"] = resp["NextContinuationToken"]
            else:
                break
        out.sort(key=lambda x: x[0])
        return out

    def count_files_in_dir(self, file_uri: str) -> int:
        """Count every file below a model version's S3 bundle prefix.

        A trailing-slash URI is already the bundle root; otherwise the primary
        file's parent is the root. Recursive listing keeps nested archive
        layouts consistent with the file browser and artifacts_count.
        """
        if not file_uri.startswith("s3://"):
            return 0
        rest = file_uri[len("s3://"):]
        slash = rest.find("/")
        if slash < 0:
            return 0
        bucket = rest[:slash]
        key = rest[slash + 1:]
        dir_slash = key.rfind("/")
        prefix = key[:dir_slash + 1] if dir_slash >= 0 else ""
        count = 0
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        while True:
            resp = self.client.list_objects_v2(**kwargs)
            for o in resp.get("Contents", []) or []:
                if not o["Key"].endswith("/"):
                    count += 1
            if resp.get("IsTruncated") and resp.get("NextContinuationToken"):
                kwargs["ContinuationToken"] = resp["NextContinuationToken"]
            else:
                break
        return count

    def list_files_in_dir(self, file_uri: str) -> list[dict]:
        """List all files below the primary file's parent or explicit bundle root.

        Names stay relative to that root, including nested directories. The
        recursive semantics mirror :meth:`count_files_in_dir`, so the UI list
        and artifact count describe the same complete bundle.
        """
        if not file_uri.startswith("s3://"):
            return []
        rest = file_uri[len("s3://"):]
        slash = rest.find("/")
        if slash < 0:
            return []
        bucket = rest[:slash]
        key = rest[slash + 1:]
        dir_slash = key.rfind("/")
        prefix = key[:dir_slash + 1] if dir_slash >= 0 else ""
        out: list[dict] = []
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        while True:
            resp = self.client.list_objects_v2(**kwargs)
            for o in resp.get("Contents", []) or []:
                k = o["Key"]
                if k.endswith("/"):
                    continue
                name = k[len(prefix):] if prefix else k
                out.append({
                    "name": name,
                    "size": int(o.get("Size", 0) or 0),
                    "modified_at": str(o.get("LastModified") or "")[:19],
                    "uri": f"s3://{bucket}/{k}",
                })
            if resp.get("IsTruncated") and resp.get("NextContinuationToken"):
                kwargs["ContinuationToken"] = resp["NextContinuationToken"]
            else:
                break
        out.sort(key=lambda x: x["name"])
        return out

    def walk(self, filer_path: str, max_depth: Optional[int] = None) -> list[str]:
        """Recursively collect all file paths under a filer path (full paths)."""
        bucket, prefix = self._split_filer_path(filer_path)
        prefix = prefix.strip("/")
        if prefix and not prefix.endswith("/"):
            prefix += "/"
        files: list[str] = []
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        while True:
            resp = self.client.list_objects_v2(**kwargs)
            for o in resp.get("Contents", []) or []:
                if o["Key"].endswith("/"):
                    continue
                files.append(f"/buckets/{bucket}/{o['Key']}")
            if resp.get("IsTruncated") and resp.get("NextContinuationToken"):
                kwargs["ContinuationToken"] = resp["NextContinuationToken"]
            else:
                break
        return files

    def read_object(self, filer_path: str) -> bytes:
        """Read an object fully and return its bytes."""
        bucket, key = self._split_filer_path(filer_path)
        resp = self.client.get_object(Bucket=bucket, Key=key)
        return resp["Body"].read()

    def object_size(self, uri_or_path: str) -> int:
        """Return an object's byte size via HEAD (no body download).

        Accepts either an s3:// URI or a /buckets/{bucket}/{key} filer path.
        """
        path = uri_or_path
        if path.startswith("s3://"):
            path = self.s3_uri_to_filer_path(path)
        bucket, key = self._split_filer_path(path)
        if not bucket or not key:
            return 0
        try:
            resp = self.client.head_object(Bucket=bucket, Key=key)
            return int(resp.get("ContentLength", 0) or 0)
        except Exception:
            return 0

    def put_object(self, filer_path: str, data: bytes) -> None:
        """Write bytes to an object (S3 needs no directory creation)."""
        bucket, key = self._split_filer_path(filer_path)
        self.client.put_object(Bucket=bucket, Key=key, Body=data)

    def upload_fileobj(self, filer_path: str, fileobj: BinaryIO) -> None:
        """Stream a binary file-like object to S3 without buffering it in RAM.

        boto3 multipart-uploads large checkpoints in chunks, so a multi-hundred-
        MB model never has to be read fully into the API process memory.
        """
        bucket, key = self._split_filer_path(filer_path)
        pos = fileobj.tell()
        try:
            fileobj.seek(0)
            extra = {"ContentType": "application/octet-stream"}
            self.client.upload_fileobj(fileobj, bucket, key, ExtraArgs=extra)
        finally:
            try:
                fileobj.seek(pos)
            except Exception:
                pass

    def delete_object(self, filer_path: str) -> None:
        """Delete an object; a directory path deletes all objects under it."""
        bucket, key = self._split_filer_path(filer_path)
        if not key:
            objects = [
                {"Key": o["Key"]}
                for o in self._list_all(bucket, "")
                if not o["Key"].endswith("/")
            ]
            if objects:
                self.client.delete_objects(Bucket=bucket, Delete={"Objects": objects})
            return
        self.client.delete_object(Bucket=bucket, Key=key)

    def count_files(self, filer_path: str) -> int:
        """Count all files (non-directory objects) under a filer path."""
        bucket, prefix = self._split_filer_path(filer_path)
        prefix = prefix.strip("/")
        if prefix and not prefix.endswith("/"):
            prefix += "/"
        return sum(1 for o in self._list_all(bucket, prefix) if not o["Key"].endswith("/"))

    def stream_file(self, filer_path: str, out: BinaryIO) -> int:
        """Stream an object's content to out, returning bytes written."""
        data = self.read_object(filer_path)
        out.write(data)
        return len(data)

    def download_to_zip(self, filer_path: str, out: BinaryIO) -> None:
        """Pack all files under a filer path into a zip written to out."""
        files = self.walk(filer_path)
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in files:
                rel = f[len(filer_path):].lstrip("/")
                if not rel:
                    continue
                data = self.read_object(f)
                zf.writestr(rel, data)

    # -- helpers ------------------------------------------------------------

    def _list_all(self, bucket: str, prefix: str) -> list[dict]:
        """List all objects under bucket/prefix (used by delete/count)."""
        objs: list[dict] = []
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        while True:
            resp = self.client.list_objects_v2(**kwargs)
            objs.extend(resp.get("Contents", []) or [])
            if resp.get("IsTruncated") and resp.get("NextContinuationToken"):
                kwargs["ContinuationToken"] = resp["NextContinuationToken"]
            else:
                break
        return objs

    def resolve_object_url(self, uri: str, object_path: str) -> str:
        """Return a public HTTP URL for an object (S3 endpoint + filer path)."""
        filer_path = self.s3_uri_to_filer_path(uri)
        return f"{self.endpoint}{filer_path}/{object_path}".replace("//", "/").replace("http:/", "http://")


# Module-level singleton matching the Go dependency-injection pattern.
s3_client: SeaweedFSClient = SeaweedFSClient()
