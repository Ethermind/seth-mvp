"""
Telemetry and health status endpoint (GET /api/status).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from src.application.use_cases.telemetry import CollectTelemetryUseCase
from src.interfaces.api.dependencies import get_telemetry_use_case

router = APIRouter(prefix="/api", tags=["Telemetry"])


@router.get("/status")
async def status_endpoint(
    telemetry_use_case: CollectTelemetryUseCase = Depends(get_telemetry_use_case),
):
    """Returns live telemetry snapshot (VRAM, reachability of vLLM/Qdrant/Whisper/Neo4j)."""
    status = await telemetry_use_case.execute()
    return status.to_dict()
