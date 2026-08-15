"""
On-disk session allow-list store (JSON).
Implements the domain SessionStore protocol.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Set

from src.config.settings import SethSettings, get_settings

logger = logging.getLogger(__name__)


class JsonSessionStore:
    """Manages the allow-list of authorized sessions stored in a JSON file."""

    def __init__(self, settings: SethSettings | None = None) -> None:
        self.settings = settings or get_settings()
        self.filepath = self.settings.allowed_users_file
        self._ensure_storage_exists()
        self.allowed_users: Set[str] = self._load_users()

    def _ensure_storage_exists(self) -> None:
        os.makedirs(self.filepath.parent, exist_ok=True)
        if not self.filepath.exists():
            try:
                with open(self.filepath, "w", encoding="utf-8") as f:
                    json.dump({"users": []}, f, indent=4)
                logger.info("📁 [SECURITY] Created clean session database at %s", self.filepath)
            except Exception as e:
                logger.error("❌ Error creating allowed_api_users.json: %s", e)

    def _load_users(self) -> Set[str]:
        users: Set[str] = set()
        if self.filepath.exists():
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    users.update(data.get("users", []))
            except Exception as e:
                logger.error("❌ Error reading allowed_api_users.json: %s", e)

        # Merge environment overrides if any
        if self.settings.allowed_api_user_ids:
            env_users = [u.strip() for u in self.settings.allowed_api_user_ids.split(",") if u.strip()]
            users.update(env_users)

        return users

    def is_allowed(self, user_id: str) -> bool:
        """Verifies if user_id is in the authorized set."""
        return user_id in self.allowed_users

    def register(self, user_id: str, token: str) -> bool:
        """Authorizes a new user session if the registration token matches."""
        if token.strip() == self.settings.registration_token:
            self.allowed_users.add(user_id)
            try:
                with open(self.filepath, "w", encoding="utf-8") as f:
                    json.dump({"users": sorted(list(self.allowed_users))}, f, indent=4)
                logger.info("🔒 [SECURITY] New session registered dynamically: %s", user_id)
                return True
            except Exception as e:
                logger.error("❌ Error saving new user to JSON: %s", e)
        return False
