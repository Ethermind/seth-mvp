"""
Unit tests for CLI entrypoint (src.main).
"""

from __future__ import annotations

import sys
from unittest.mock import patch
import pytest

from src.main import main, run_api, run_telegram, run_tui


def test_main_help_when_no_mode(capsys):
    with patch.object(sys, "argv", ["main.py"]):
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1


def test_main_api_mode():
    with patch.object(sys, "argv", ["main.py", "api", "--host", "0.0.0.0", "--port", "9000"]):
        with patch("src.main.run_api") as mock_run_api:
            main()
            mock_run_api.assert_called_once_with(host="0.0.0.0", port=9000)


def test_main_telegram_mode():
    with patch.object(sys, "argv", ["main.py", "telegram"]):
        with patch("src.main.run_telegram") as mock_run_tg:
            main()
            mock_run_tg.assert_called_once()


def test_main_tui_mode():
    with patch.object(sys, "argv", ["main.py", "tui"]):
        with patch("src.main.run_tui") as mock_run_tui:
            main()
            mock_run_tui.assert_called_once()


def test_main_web_mode():
    with patch.object(sys, "argv", ["main.py", "web", "--port", "6000", "--no-browser"]):
        with patch("src.main.run_web") as mock_run_web:
            main()
            mock_run_web.assert_called_once_with(
                host="127.0.0.1", port=6000, api_url="http://127.0.0.1:8080", open_browser=False
            )
