#!/bin/bash
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

# ML-Backend Custom startup script
# Downloads model from S3 and starts the server

set -e

echo "Starting ML-Backend Custom..."
echo "MODEL_PATH: ${MODEL_PATH:-not set}"
echo "BACKEND_TYPE: ${BACKEND_TYPE:-not set}"
echo "FRAMEWORK: ${FRAMEWORK:-not set}"

# Create model directory if not exists
mkdir -p ${MODEL_ROOT:-/app/models}

# Start the server
exec gunicorn --bind :${PORT:-9090} --workers ${WORKERS:-1} --threads ${THREADS:-4} --timeout 300 _wsgi:app
