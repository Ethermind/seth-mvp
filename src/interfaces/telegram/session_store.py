"""
On-disk mapping of Telegram user_id -> SETH API session user_id (JSON).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, Optional

from src.config.settings import SethSettings, get_settings

logger = logging.getLogger(__name__)


class TelegramSessionStore:
    """Stores and retrieves mapping from Telegram user ID to SETH API session ID."""

    def __init__(self, settings: SethSettings | None = None) -> None:
        self.settings = settings or get_settings()
        self.filepath = self.settings.telegram_sessions_file
        self._ensure_storage_exists()
        self.sessions: Dict[str, str] = self._load_sessions()

    def _ensure_storage_exists(self) -> None:
        os.makedirs(self.filepath.parent, exist_ok=True)
        if not self.filepath.exists():
            try:
                with open(self.filepath, "w", encoding="utf-8") as f:
                    json.dump({}, f, indent=4)
                logger.info("📁 [TELEGRAM SESSIONS] Created database file at %s", self.filepath)
            except Exception as e:
                logger.error("❌ Error creating telegram_sessions.json: %s", e)

    def _load_sessions(self) -> Dict[str, str]:
        if self.filepath.exists():
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error("❌ Error reading telegram_sessions.json: %s", e)
        return {}

    def is_allowed(self, telegram_user_id: int | str) -> bool:
        return str(telegram_user_id) in self.sessions

    def get_api_user_id(self, telegram_user_id: int | str) -> Optional[str]:
        return self.sessions.get(str(telegram_user_id))

    def store(self, telegram_user_id: int | str, api_user_id: str) -> None:
        self.sessions[str(telegram_user_id)] = api_user_id
        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(self.sessions, f, indent=4)
            logger.info("🔒 [TELEGRAM AUTH] Mapped ID %s -> %s...", telegram_user_id, api_user_id[:8])
        except Exception as e:
            logger.error("❌ Error saving telegram session mapping: %s", e)
