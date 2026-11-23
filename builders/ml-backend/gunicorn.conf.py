# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Gunicorn runtime config for the NereidForge ML backend.

The image serves both the Label Studio ML API (/predict, /health) and the
custom NF routes (/try/predict, /sam/segment). Access logging is enabled to
stdout so every inbound call is visible in `kubectl logs`; readiness/liveness
probes (/health) are filtered out to keep that signal readable.
"""

# Emit access and error logs to stdout (captured by the container runtime).
accesslog = "-"
errorlog = "-"
loglevel = "info"
capture_output = True

# remote - "METHOD PATH PROTO" status bytes duration(s) referer user-agent
access_log_format = '%(h)s - "%(r)s" %(s)s %(b)s %(L)s s "%(f)s" "%(a)s"'


class _SkipHealthProbes:
    """Drop access-log records produced by readiness/liveness probes."""

    def filter(self, record) -> bool:  # noqa: ANN001
        try:
            msg = record.getMessage()
        except Exception:
            return True
        return "GET /health" not in msg and "GET /ready" not in msg


def post_worker_init(worker):  # noqa: ANN001
    """Attach the health-filter after gunicorn configures its loggers."""
    import logging

    logging.getLogger("gunicorn.access").addFilter(_SkipHealthProbes())
