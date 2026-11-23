#!/usr/bin/env python3
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Read and initialize the operator-managed SeaweedFS credential Secret.

Credential values are accepted only on stdin (an existing Kubernetes resource)
or through the process environment (first-time initialization). The manifest
command writes a Secret directly to stdout so callers can pipe it to kubectl
without creating a values file or exposing credentials in command arguments.
"""

import argparse
import base64
import binascii
import hashlib
import json
import os
import shlex
import sys


SECRET_NAME = "seaweedfs-s3-credentials"
CONFIG_KEY = "s3-config.json"
ANONYMOUS_READ_PREFIXES = (
    "Read:nereidforge/models/*",
    "Read:nereidforge/optimized/*",
)


class CredentialError(ValueError):
    """The existing object-store identity is absent or malformed."""


def _validate_credentials(access_key, secret_key):
    if not isinstance(access_key, str) or not access_key:
        raise CredentialError("SeaweedFS accessKey is missing or empty")
    if not isinstance(secret_key, str) or not secret_key:
        raise CredentialError("SeaweedFS secretKey is missing or empty")
    if "\x00" in access_key or "\x00" in secret_key:
        raise CredentialError("SeaweedFS credentials cannot contain NUL bytes")
    return access_key, secret_key


def _resource_value(resource, key):
    data = resource.get("data", {})
    if isinstance(data, dict) and key in data:
        if resource.get("kind") == "ConfigMap":
            return data[key]
        try:
            return base64.b64decode(data[key], validate=True).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError) as error:
            raise CredentialError(f"SeaweedFS Secret key {key!r} is not valid UTF-8 base64") from error
    string_data = resource.get("stringData", {})
    if isinstance(string_data, dict) and key in string_data:
        return string_data[key]
    raise CredentialError(f"SeaweedFS Secret is missing {key}")


def credentials_from_secret(resource):
    """Return the shared identity without changing it."""
    return _validate_credentials(
        _resource_value(resource, "accessKey"),
        _resource_value(resource, "secretKey"),
    )


def credentials_from_config(resource):
    """Migrate the identity from a legacy SeaweedFS config resource."""
    try:
        config = json.loads(_resource_value(resource, CONFIG_KEY))
        identities = config["identities"]
        admin = next(identity for identity in identities if identity.get("name") == "admin")
        credential = admin["credentials"][0]
    except (KeyError, IndexError, StopIteration, TypeError, json.JSONDecodeError) as error:
        raise CredentialError("Legacy SeaweedFS config has no admin identity") from error
    return _validate_credentials(credential.get("accessKey"), credential.get("secretKey"))


def s3_config(access_key, secret_key):
    """Build SeaweedFS's admin identity and narrowly scoped anonymous reads."""
    _validate_credentials(access_key, secret_key)
    return {
        "identities": [
            {
                "name": "admin",
                "credentials": [{"accessKey": access_key, "secretKey": secret_key}],
                "actions": ["Admin", "Read", "List", "Tagging", "Write"],
            },
            {
                "name": "anonymous",
                "credentials": [],
                "actions": list(ANONYMOUS_READ_PREFIXES),
            },
        ]
    }


def secret_manifest(access_key, secret_key, namespace="nereidforge", name=SECRET_NAME):
    """Return a retain-on-helm-upgrade Secret containing identity and config."""
    _validate_credentials(access_key, secret_key)
    config = json.dumps(s3_config(access_key, secret_key), ensure_ascii=False, indent=2) + "\n"
    return {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "labels": {
                "app.kubernetes.io/name": "seaweedfs",
                "app.kubernetes.io/part-of": "nereidforge",
                "component": "s3",
            },
            # Older development releases rendered this Secret from Helm. Keep
            # it when the new external-reference chart omits that resource.
            "annotations": {"helm.sh/resource-policy": "keep"},
        },
        "type": "Opaque",
        "stringData": {
            "accessKey": access_key,
            "secretKey": secret_key,
            CONFIG_KEY: config,
        },
    }


def shell_exports(access_key, secret_key):
    """Quote arbitrary environment-safe credential bytes for a Bash caller."""
    _validate_credentials(access_key, secret_key)
    return (
        f"export SEAWEEDFS_S3_ACCESS_KEY={shlex.quote(access_key)}\n"
        f"export SEAWEEDFS_S3_SECRET_KEY={shlex.quote(secret_key)}\n"
    )


def secret_checksum(resource):
    """Digest only Secret data; never emit the credentials themselves."""
    credentials_from_secret(resource)
    data = resource.get("data", {})
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _read_resource():
    try:
        resource = json.load(sys.stdin)
    except json.JSONDecodeError as error:
        raise CredentialError("Could not parse the Kubernetes credential resource") from error
    if not isinstance(resource, dict):
        raise CredentialError("Kubernetes credential resource must be an object")
    return resource


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    exports = subparsers.add_parser("exports", help="read a resource on stdin and print Bash exports")
    exports.add_argument("source", choices=("secret", "config"))

    checksum = subparsers.add_parser("checksum", help="digest a Secret read on stdin")
    checksum.add_argument("source", choices=("secret",), nargs="?", default="secret")

    manifest = subparsers.add_parser("manifest", help="build a Secret from first-use environment input")
    manifest.add_argument("--namespace", default="nereidforge")
    manifest.add_argument("--name", default=SECRET_NAME)

    args = parser.parse_args(argv)
    try:
        if args.command == "exports":
            resource = _read_resource()
            extractor = credentials_from_secret if args.source == "secret" else credentials_from_config
            sys.stdout.write(shell_exports(*extractor(resource)))
        elif args.command == "checksum":
            print(secret_checksum(_read_resource()))
        else:
            access_key = os.environ.get("SEAWEEDFS_S3_ACCESS_KEY", "")
            secret_key = os.environ.get("SEAWEEDFS_S3_SECRET_KEY", "")
            json.dump(
                secret_manifest(access_key, secret_key, args.namespace, args.name),
                sys.stdout,
                ensure_ascii=False,
            )
            sys.stdout.write("\n")
    except CredentialError as error:
        parser.exit(1, f"S3 credential error: {error}\n")


if __name__ == "__main__":
    main()
