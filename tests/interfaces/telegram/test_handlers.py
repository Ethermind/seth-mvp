"""
Unit tests for TelegramBridgeHandlers (src.interfaces.telegram.handlers).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
import pytest

from src.config.settings import SethSettings
from src.interfaces.client import SethClient
from src.interfaces.telegram.handlers import TelegramBridgeHandlers
from src.interfaces.telegram.session_store import TelegramSessionStore


@pytest.mark.anyio
async def test_telegram_handlers_start(tmp_path):
    settings = SethSettings(project_root=tmp_path)
    store = TelegramSessionStore(settings)
    client = SethClient(base_url="http://mock-api")
    handlers = TelegramBridgeHandlers(settings=settings, api_client=client, session_store=store)

    update = MagicMock()
    update.effective_chat.id = 123456
    update.effective_user.first_name = "Luis"
    update.message.reply_text = AsyncMock()

    context = MagicMock()

    await handlers.start_cmd(update, context)
    update.message.reply_text.assert_called_once()
    call_args = update.message.reply_text.call_args[0][0]
    assert "SETH-IN-A-BOX" in call_args


@pytest.mark.anyio
async def test_telegram_send_long_message(tmp_path):
    settings = SethSettings(project_root=tmp_path)
    store = TelegramSessionStore(settings)
    client = SethClient(base_url="http://mock-api")
    handlers = TelegramBridgeHandlers(settings=settings, api_client=client, session_store=store)

    update = MagicMock()
    update.message.reply_text = AsyncMock()

    # Short message
    await handlers._send_long_message(update, "Short message", max_length=4000)
    update.message.reply_text.assert_called_once_with("Short message")

    # Long message
    update.message.reply_text.reset_mock()
    long_text = "Line text\n" * 500
    await handlers._send_long_message(update, long_text, max_length=500)
    assert update.message.reply_text.call_count > 1
