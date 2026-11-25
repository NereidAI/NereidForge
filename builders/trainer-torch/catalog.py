# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Print the trainer capability catalog as JSON.

The backend queries `python catalog.py` once per image (cached), so the UI never
hardcodes which architectures a trainer supports:

    docker run --rm <trainer-image> python catalog.py

Output: {"framework": "torch", "tasks": [{"id", "architectures", "default"}], ...}
"""

from __future__ import annotations

import json

from models import catalog

if __name__ == "__main__":
    print(json.dumps(catalog(), indent=2))
