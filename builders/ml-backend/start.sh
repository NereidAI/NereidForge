#!/bin/bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# Execute gunicorn. Access/error logging is configured in gunicorn.conf.py
# (logs every inbound call to stdout, filtering out /health probes).
exec gunicorn --config /app/gunicorn.conf.py \
  --bind :${PORT:-9090} \
  --workers ${WORKERS:-1} \
  --threads ${THREADS:-4} \
  --timeout 0 \
  _wsgi:app
