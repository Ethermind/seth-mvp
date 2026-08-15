"""
Unit tests for Web TUI standalone server (src.interfaces.web.app).
"""

from __future__ import annotations

from unittest.mock import patch
from src.interfaces.web.app import check_api_status


def test_check_api_status_online():
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_resp = mock_urlopen.return_value.__enter__.return_value
        mock_resp.status = 200
        assert check_api_status("http://127.0.0.1:8080") is True


def test_check_api_status_offline():
    with patch("urllib.request.urlopen", side_effect=Exception("Connection refused")):
        assert check_api_status("http://127.0.0.1:8080") is False
