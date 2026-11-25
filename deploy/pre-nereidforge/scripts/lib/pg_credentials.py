#!/usr/bin/env python3
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Create and read the canonical pre-NereidForge PostgreSQL basic-auth Secret."""

import base64
import binascii
import json
import os
import shlex
import sys


class CredentialError(ValueError):
    """The retained Secret does not satisfy the basic-auth contract."""


def credentials_from_secret(resource):
    data = resource.get("data")
    if not isinstance(data, dict):
        raise CredentialError("PostgreSQL credentials Secret has no data object")
    values = []
    for key in ("username", "password"):
        encoded = data.get(key)
        if not isinstance(encoded, str) or not encoded:
            raise CredentialError(f"PostgreSQL credentials Secret is missing {key}")
        try:
            value = base64.b64decode(encoded, validate=True).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError) as error:
            raise CredentialError(f"PostgreSQL credentials Secret has invalid {key}") from error
        if not value:
            raise CredentialError(f"PostgreSQL credentials Secret has empty {key}")
        if "\x00" in value:
            raise CredentialError(f"PostgreSQL credentials Secret has NUL in {key}")
        values.append(value)
    return tuple(values)


def shell_exports(resource):
    username, password = credentials_from_secret(resource)
    return (
        f"export PG_SECRET_USERNAME={shlex.quote(username)}\n"
        f"export PG_SECRET_PASSWORD={shlex.quote(password)}\n"
    )


def secret_manifest(name, namespace, username, password):
    if not all((name, namespace, username, password)):
        raise CredentialError("PostgreSQL Secret name, namespace, username, and password are required")
    if any("\x00" in value for value in (username, password)):
        raise CredentialError("PostgreSQL username and password must not contain NUL")
    return {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "annotations": {"helm.sh/resource-policy": "keep"},
        },
        "type": "kubernetes.io/basic-auth",
        "stringData": {"username": username, "password": password},
    }


def main(argv):
    if len(argv) == 2 and argv[1] == "exports":
        resource = json.load(sys.stdin)
        sys.stdout.write(shell_exports(resource))
        return 0
    if len(argv) == 4 and argv[1] == "manifest":
        manifest = secret_manifest(
            argv[2],
            argv[3],
            os.environ.get("PG_DB_USER", ""),
            os.environ.get("PG_ADMIN_PASSWORD", ""),
        )
        json.dump(manifest, sys.stdout, separators=(",", ":"))
        sys.stdout.write("\n")
        return 0
    raise CredentialError("Usage: pg_credentials.py exports | manifest NAME NAMESPACE")


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv))
    except (CredentialError, json.JSONDecodeError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
