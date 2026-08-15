"""
Unit tests for CollectTelemetryUseCase (src.application.use_cases.telemetry).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch
import pytest

from src.application.use_cases.telemetry import CollectTelemetryUseCase
from src.config.settings import SethSettings
from src.domain.models import GpuTelemetry, SystemStatus


@pytest.mark.anyio
async def test_collect_telemetry_all_online(tmp_path):
    settings = SethSettings(project_root=tmp_path)
    mock_openai = AsyncMock()
    mock_openai.models.list = AsyncMock(return_value=None)

    mock_graphiti = AsyncMock()
    mock_graphiti._graphiti_instance = object()

    use_case = CollectTelemetryUseCase(
        settings=settings,
        openai_client=mock_openai,
        graphiti_adapter=mock_graphiti,
    )

    with patch("src.application.use_cases.telemetry.probe_tcp_port", new=AsyncMock(return_value="online")):
        with patch(
            "src.application.use_cases.telemetry.NvidiaSmiProbe.probe_vram",
            new=AsyncMock(return_value=[GpuTelemetry(index=0, name="RTX 4090", used_gb=8.0, total_gb=24.0)]),
        ):
            with patch.object(use_case, "_probe_qdrant", new=AsyncMock(return_value="online")):
                status = await use_case.execute()

    assert isinstance(status, SystemStatus)
    assert status.vllm == "online"
    assert status.whisper == "online"
    assert status.qdrant == "online"
    assert status.neo4j_graphiti == "online"
    assert len(status.vram) == 1
    assert status.vram[0].name == "RTX 4090"
