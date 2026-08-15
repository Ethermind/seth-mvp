"""
Unit tests for Status route GET /api/status (src.interfaces.api.routes.status).
"""

from __future__ import annotations

from unittest.mock import AsyncMock
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
import pytest

from src.domain.models import GpuTelemetry, SystemStatus
from src.interfaces.api.dependencies import get_telemetry_use_case
from src.interfaces.api.routes.status import router as status_router


@pytest.mark.anyio
async def test_status_route():
    app = FastAPI()
    app.include_router(status_router)

    mock_use_case = AsyncMock()
    mock_use_case.execute.return_value = SystemStatus(
        vllm="online",
        whisper="online",
        qdrant="online",
        neo4j_graphiti="online",
        vram=[GpuTelemetry(index=0, name="RTX 4090", used_gb=10.0, total_gb=24.0)],
    )

    app.dependency_overrides[get_telemetry_use_case] = lambda: mock_use_case

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["vllm"] == "online"
        assert data["whisper"] == "online"
        assert len(data["vram"]) == 1
