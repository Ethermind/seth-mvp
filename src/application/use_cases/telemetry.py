"""
Use Case: Collect Telemetry and Microservice Reachability Snapshot.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx
from openai import AsyncOpenAI
from src.config.settings import SethSettings, get_settings
from src.domain.models import SystemStatus
from src.infrastructure.telemetry.nvidia_smi import NvidiaSmiProbe
from src.infrastructure.telemetry.tcp_probe import probe_tcp_port

logger = logging.getLogger(__name__)


class CollectTelemetryUseCase:
    """Consolidates system health checks, reachability probes, and GPU telemetry."""

    def __init__(
        self,
        settings: SethSettings | None = None,
        openai_client: AsyncOpenAI | None = None,
        graphiti_adapter: Any | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.openai_client = openai_client or AsyncOpenAI(
            base_url=self.settings.vllm_url, api_key=self.settings.api_key
        )
        self.graphiti_adapter = graphiti_adapter

    async def execute(self) -> SystemStatus:
        """Collects telemetry from hardware probes and services concurrently."""
        vllm_task = self._probe_vllm()
        whisper_task = probe_tcp_port(self.settings.whisper_url)
        qdrant_task = self._probe_qdrant()
        graphiti_task = self._probe_graphiti()
        vram_task = NvidiaSmiProbe.probe_vram()

        vllm_res, whisper_res, qdrant_res, graphiti_res, vram_res = await asyncio.gather(
            vllm_task, whisper_task, qdrant_task, graphiti_task, vram_task
        )

        return SystemStatus(
            vllm=vllm_res,
            whisper=whisper_res,
            qdrant=qdrant_res,
            neo4j_graphiti=graphiti_res,
            vram=vram_res,
        )

    async def _probe_vllm(self) -> str:
        try:
            await asyncio.wait_for(self.openai_client.models.list(), timeout=3.0)
            return "online"
        except Exception as e:
            return f"offline ({e})"

    async def _probe_qdrant(self) -> str:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"http://{self.settings.qdrant_host}:{self.settings.qdrant_port}/collections")
                return "online" if resp.status_code == 200 else f"http_{resp.status_code}"
        except Exception as e:
            return f"offline ({e})"

    async def _probe_graphiti(self) -> str:
        if self.graphiti_adapter and getattr(self.graphiti_adapter, "_graphiti_instance", None) is not None:
            return "online"
        return "not_initialized"
