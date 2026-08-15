"""
Unit tests for Terminal TUI App (src.interfaces.tui.app).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch
import pytest

from src.config.settings import SethSettings
from src.interfaces.tui.app import SethTerminalApp


@pytest.mark.anyio
async def test_tui_app_initialize_session_success(tmp_path):
    settings = SethSettings(project_root=tmp_path, registration_token="secret_tui_token")
    app = SethTerminalApp(api_base_url="http://mock-api", settings=settings)

    with patch.object(app.client, "get_status", new=AsyncMock(return_value={"vllm": "online", "whisper": "online"})):
        with patch.object(app.client, "register", new=AsyncMock(return_value="session_tui_uuid")):
            ok = await app.initialize_session()

    assert ok is True
    assert app.user_id == "session_tui_uuid"


@pytest.mark.anyio
async def test_tui_app_initialize_session_unreachable(tmp_path):
    settings = SethSettings(project_root=tmp_path)
    app = SethTerminalApp(api_base_url="http://mock-api", settings=settings)

    with patch.object(app.client, "get_status", side_effect=ConnectionError("Offline")):
        ok = await app.initialize_session()

    assert ok is False
    assert app.user_id is None
