"""
Unit tests for JsonSessionStore (src.infrastructure.security.json_sessions).
"""

from __future__ import annotations

from pathlib import Path
from src.config.settings import SethSettings
from src.infrastructure.security.json_sessions import JsonSessionStore


def test_json_session_store_registration_and_persistence(tmp_path: Path):
    settings = SethSettings(project_root=tmp_path, registration_token="top_secret_token")
    store = JsonSessionStore(settings)

    # Invalid token rejected
    assert store.register("user_1", "wrong_token") is False
    assert store.is_allowed("user_1") is False

    # Valid token accepted
    assert store.register("user_1", "top_secret_token") is True
    assert store.is_allowed("user_1") is True

    # Reload from disk
    store_reloaded = JsonSessionStore(settings)
    assert store_reloaded.is_allowed("user_1") is True
