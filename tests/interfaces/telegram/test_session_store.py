"""
Unit tests for TelegramSessionStore (src.interfaces.telegram.session_store).
"""

from __future__ import annotations

from pathlib import Path
from src.config.settings import SethSettings
from src.interfaces.telegram.session_store import TelegramSessionStore


def test_telegram_session_store_save_and_load(tmp_path: Path):
    settings = SethSettings(project_root=tmp_path)
    store = TelegramSessionStore(settings)

    # Initial state empty
    assert store.is_allowed(12345) is False
    assert store.get_api_user_id(12345) is None

    # Store session
    store.store(12345, "api_uuid_999")
    assert store.is_allowed(12345) is True
    assert store.get_api_user_id(12345) == "api_uuid_999"

    # Reload from disk
    store_reloaded = TelegramSessionStore(settings)
    assert store_reloaded.is_allowed(12345) is True
    assert store_reloaded.get_api_user_id(12345) == "api_uuid_999"
