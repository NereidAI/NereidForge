#!/usr/bin/env python3
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Generate and validate the non-secret pre-NereidForge to NF Helm handoff.

Only explicit non-secret CLI arguments are consumed. In particular, this program
never reads token/password environment variables or Kubernetes Secrets. JSON
string scalars are valid YAML and keep every emitted value unambiguous without a
third-party YAML dependency.
"""

import argparse
import json
import os
from pathlib import Path
import re
import sys
import tempfile


MARKER = "# pre-nereidforge-handoff/v1; non-secret NereidForge Helm values"
SENSITIVE_KEY = re.compile(
    r"(?:^[ \t]*|[,{][ \t]*)[\"']?(?:secrets|stringData|password|jwt_secret|client_secret|"
    r"lsToken|token|accessKey|secretKey|NEREIDFORGE_[A-Z0-9_]*(?:PASSWORD|SECRET|TOKEN))"
    r"[\"']?\s*:",
    re.MULTILINE,
)


def scalar(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value), ensure_ascii=True)


def yaml_lines(value, indent=0):
    """Serialize the small dict/list/scalar schema with deterministic key order."""
    prefix = " " * indent
    if isinstance(value, dict):
        lines = []
        for key in sorted(value):
            item = value[key]
            if isinstance(item, (dict, list)):
                if not item:
                    lines.append(f"{prefix}{key}: {'{}' if isinstance(item, dict) else '[]'}")
                else:
                    lines.append(f"{prefix}{key}:")
                    lines.extend(yaml_lines(item, indent + 2))
            else:
                lines.append(f"{prefix}{key}: {scalar(item)}")
        return lines
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, dict):
                keys = sorted(item)
                first = keys[0]
                first_value = item[first]
                if isinstance(first_value, (dict, list)):
                    lines.append(f"{prefix}- {first}:")
                    lines.extend(yaml_lines(first_value, indent + 4))
                else:
                    lines.append(f"{prefix}- {first}: {scalar(first_value)}")
                remainder = {key: item[key] for key in keys[1:]}
                lines.extend(yaml_lines(remainder, indent + 2))
            elif isinstance(item, list):
                raise TypeError("nested lists are not part of the handoff schema")
            else:
                lines.append(f"{prefix}- {scalar(item)}")
        return lines
    raise TypeError("a top-level scalar is not a values mapping")


def frontend_paths():
    return [
        {"path": path, "pathType": "Prefix", "serviceName": service, "servicePort": port}
        for path, service, port in (
            ("/", "nereidforge-frontend", 80),
            ("/api", "nereidforge-api", 8080),
            ("/auth/ns-callback", "nereidforge-api", 8080),
            ("/v1/integration/ns", "nereidforge-api", 8080),
            ("/notebook-servers", "nereidforge-api", 8080),
        )
    ]


def build_values(args):
    frontend_host = args.frontend_host or f"nereidforge.{args.base_domain}"
    sso = {
        "mode": args.sso_mode,
        "jit_tenant": "default",
        "ns": {
            "base_url": args.sso_ns_base_url or f"https://nereidsphere.{args.base_domain}",
            "product_id": "forge",
            "callback_path": "/auth/ns-callback",
            "reconcile_memberships": True,
        },
        "oidc": {
            "issuer": args.sso_oidc_issuer,
            "client_id": args.sso_oidc_client_id,
            "redirect_url": f"https://{frontend_host}/auth/callback",
            "frontend_base_url": f"https://{frontend_host}",
        },
    }
    return {
        "handoff": {
            "schemaVersion": 1,
            "preVersion": args.pre_version,
            "sourceCommit": args.source_commit,
            "namespace": "nereidforge",
            "runtimeSecret": "nereidforge-service-credentials",
            "adminSecret": "nereidforge-admin-credentials",
            "caBundleSecret": "nereid-internal-ca",
        },
        "namespace": {"create": False},
        "global": {
            "domain": args.base_domain,
            "storageClass": args.storage_class,
            "tls": {"enabled": True, "clusterIssuer": "internal-ca"},
        },
        "caBundleSecret": "nereid-internal-ca",
        "config": {
            "database": {
                "host": f"{args.pg_cluster}-rw.nereidforge.svc.cluster.local",
                "port": 5432,
                "name": args.pg_database,
                "user": args.pg_user,
                "ssl_mode": "disable",
            },
            "sso": sso,
            "kfp": {
                "host": "http://ml-pipeline.kubeflow:8888",
                "namespace": "nereidforge",
                "object_store_endpoint": "http://seaweedfs-s3.nereidforge.svc.cluster.local:8333",
                "object_store_bucket": "mlpipeline",
                "object_store_namespace": "nereidforge",
            },
            "labelstudio": {
                "host": "http://labelstudio.nereidforge.svc.cluster.local:8080",
                "public_host": f"https://labelstudio.{args.base_domain}",
                "ml_backend_url": "http://ml-backend.nereidforge.svc.cluster.local:9090",
                "ml_backend_cluster_url": "http://ml-backend.nereidforge.svc.cluster.local:9090",
            },
            "cvat": {
                "host": "http://cvat-backend-service.nereidforge.svc.cluster.local:8080",
                "public_host": f"https://cvat.{args.base_domain}",
            },
            "storage": {
                "backend": "seaweedfs",
                "endpoint": "http://seaweedfs-filer.nereidforge.svc.cluster.local:8888",
                "s3_endpoint": "http://seaweedfs-s3.nereidforge.svc.cluster.local:8333",
                "bucket": "nereidforge",
            },
        },
        "s3": {
            "endpoint": "http://seaweedfs-s3.nereidforge.svc.cluster.local:8333",
            "existingSecret": "seaweedfs-s3-credentials",
            "accessKeyKey": "accessKey",
            "secretKeyKey": "secretKey",
        },
        "postgresql": {
            "createSecret": False,
            "existingSecret": f"pg-{args.pg_cluster}-credentials",
            "existingSecretKey": "password",
        },
        "runtimeSecret": {"existingSecret": "nereidforge-service-credentials"},
        "postInstall": {
            "admin": {
                "enabled": True,
                "existingSecret": "nereidforge-admin-credentials",
                "passwordKey": "password",
                "username": "admin",
                "email": args.dex_email,
                "rotatePassword": False,
            },
            "pipelines": {
                "enabled": True,
                "host": "http://ml-pipeline.kubeflow:8888",
                "namespace": "nereidforge",
                "userID": args.dex_email,
            },
        },
        "ingress": {
            "enabled": True,
            "className": "traefik",
            "host": frontend_host,
            "paths": frontend_paths(),
            "tls": [{"secretName": "nereidforge-frontend-tls", "hosts": [frontend_host]}],
        },
        "env": [
            {"name": "NODE_ENV", "value": "production"},
            {"name": "NEXT_PUBLIC_API_URL", "value": f"https://{frontend_host}/api"},
            {"name": "NEXT_PUBLIC_WS_URL", "value": f"wss://{frontend_host}"},
        ],
        "secret": {
            "name": "ml-backend-ls-token",
            "existingSecret": "nereidforge-service-credentials",
            "lsTokenKey": "NEREIDFORGE_LABELSTUDIO_TOKEN",
        },
        "backend": {
            "labelStudioHost": "http://labelstudio.nereidforge.svc.cluster.local:8080",
        },
    }


def validate_text(text, *, generated):
    if generated and (not text.startswith(MARKER + "\n") or "  schemaVersion: 1\n" not in text):
        raise ValueError("generated handoff marker or schemaVersion is missing")
    match = SENSITIVE_KEY.search(text)
    if match:
        line = text.count("\n", 0, match.start()) + 1
        raise ValueError(f"literal credential key is forbidden in non-secret values (line {line})")


def generate(args):
    values = build_values(args)
    text = MARKER + "\n" + "\n".join(yaml_lines(values)) + "\n"
    validate_text(text, generated=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".values.generated.", dir=output.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.chmod(temporary, 0o644)
        os.replace(temporary, output)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    print(f"Generated non-secret NF Helm values: {output}")


def validate(args):
    for filename, generated in ((args.values, True), (args.operator_values, False)):
        path = Path(filename)
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"required values file is missing or empty: {path}")
        validate_text(path.read_text(encoding="utf-8"), generated=generated)
    print("Non-secret handoff values validation passed")


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    generate_parser = commands.add_parser("generate")
    generate_parser.add_argument("--output", required=True)
    generate_parser.add_argument("--pre-version", required=True)
    generate_parser.add_argument("--source-commit", default="")
    generate_parser.add_argument("--base-domain", required=True)
    generate_parser.add_argument("--storage-class", required=True)
    generate_parser.add_argument("--pg-cluster", required=True)
    generate_parser.add_argument("--pg-database", required=True)
    generate_parser.add_argument("--pg-user", required=True)
    generate_parser.add_argument("--dex-email", required=True)
    generate_parser.add_argument("--frontend-host", default="")
    generate_parser.add_argument("--sso-mode", choices=("ns", "oidc", "local"), default="ns")
    generate_parser.add_argument("--sso-ns-base-url", default="")
    generate_parser.add_argument("--sso-oidc-issuer", default="")
    generate_parser.add_argument("--sso-oidc-client-id", default="nereidforge")
    generate_parser.set_defaults(function=generate)
    validate_parser = commands.add_parser("validate")
    validate_parser.add_argument("--values", required=True)
    validate_parser.add_argument("--operator-values", required=True)
    validate_parser.set_defaults(function=validate)
    return result


def main():
    args = parser().parse_args()
    if args.command == "generate" and args.sso_mode == "oidc" and not args.sso_oidc_issuer:
        raise ValueError("--sso-oidc-issuer is required in oidc mode")
    args.function(args)


if __name__ == "__main__":
    try:
        main()
    except (OSError, TypeError, ValueError) as error:
        print(f"handoff error: {error}", file=sys.stderr)
        raise SystemExit(1)
