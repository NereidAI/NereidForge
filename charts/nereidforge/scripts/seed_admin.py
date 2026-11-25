# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Idempotently seed the NF administrator without implicit password rotation."""

import os

from sqlalchemy import select

from app.db.session import SessionLocal
from app.models.orm import User
from app.services.admin import AdminService


def main():
    username = os.environ.get("NF_ADMIN_USERNAME", "admin")
    password = os.environ.get("NF_ADMIN_PASSWORD", "")
    if not password:
        raise RuntimeError("NF_ADMIN_PASSWORD is required from the retained admin Secret")
    rotate = os.environ.get("NF_ADMIN_ROTATE", "false").lower() == "true"

    db = SessionLocal()
    try:
        service = AdminService(db)
        existing = db.scalar(select(User).where(User.username == username))
        if existing and not rotate:
            print("admin exists; password unchanged")
        elif existing:
            service.update_user(existing.id, {"password": password})
            print("admin password rotated by explicit request")
        else:
            user = service.create_user(
                username,
                os.environ.get("NF_ADMIN_EMAIL", ""),
                "Platform Admin",
                password,
                "default",
                ["admin", "user"],
            )
            print(f"admin seeded: {user['username']} / {user['tenant_id']}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
