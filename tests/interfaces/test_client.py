"""
Unit tests for SethClient SDK (src.interfaces.client).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from src.interfaces.client import SethClient


@pytest.mark.anyio
async def test_seth_client_register_success():
    client = SethClient(base_url="http://127.0.0.1:8080")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"status": "ok", "user_id": "test-uuid-123"}

    with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_resp)):
        user_id = await client.register(token="secret_token")

    assert user_id == "test-uuid-123"


@pytest.mark.anyio
async def test_seth_client_register_failure():
    client = SethClient(base_url="http://127.0.0.1:8080")

    mock_resp = MagicMock()
    mock_resp.status_code = 400
    mock_resp.json.return_value = {"status": "error"}

    with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_resp)):
        user_id = await client.register(token="bad_token")

    assert user_id is None


@pytest.mark.anyio
async def test_seth_client_get_status():
    client = SethClient(base_url="http://127.0.0.1:8080")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"vllm": "online", "whisper": "online"}

    with patch("httpx.AsyncClient.get", new=AsyncMock(return_value=mock_resp)):
        status = await client.get_status()

    assert status["vllm"] == "online"
    assert status["whisper"] == "online"
