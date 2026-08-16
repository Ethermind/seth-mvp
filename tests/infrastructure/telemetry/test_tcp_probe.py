"""
Unit tests for TCP Probe (src.infrastructure.telemetry.tcp_probe).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from src.infrastructure.telemetry.tcp_probe import probe_tcp_port


@pytest.mark.anyio
async def test_probe_tcp_port_online():
    mock_reader = AsyncMock()
    mock_writer = MagicMock()
    mock_writer.wait_closed = AsyncMock()

    with patch("asyncio.open_connection", return_value=(mock_reader, mock_writer)):
        status = await probe_tcp_port("http://localhost:8000")
        assert status == "online"


@pytest.mark.anyio
async def test_probe_tcp_port_offline():
    with patch("asyncio.open_connection", side_effect=ConnectionRefusedError("Connection refused")):
        status = await probe_tcp_port("http://localhost:9999")
        assert "offline" in status
