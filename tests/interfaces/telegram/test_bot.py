"""
Unit tests for TelegramBotRunner (src.interfaces.telegram.bot).
"""

from __future__ import annotations

from src.config.settings import SethSettings
from src.interfaces.telegram.bot import TelegramBotRunner


def test_telegram_bot_runner_init(tmp_path):
    settings = SethSettings(
        project_root=tmp_path,
        registration_token="reg_token",
        telegram_token="123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11",
    )

    runner = TelegramBotRunner(settings=settings)
    assert runner.settings.telegram_token == "123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"
    assert runner.api_client is not None
