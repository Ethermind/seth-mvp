"""
Unit tests for Chat route POST /api/chat (src.interfaces.api.routes.chat).
"""

from __future__ import annotations

from unittest.mock import AsyncMock
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
import pytest

from src.application.orchestrator import ConversationOrchestrator
from src.config.settings import SethSettings
from src.domain.models import RegulatorPresets, StreamChunk, StreamEventType
from src.infrastructure.tools.registry import ToolRegistry
from src.interfaces.api.dependencies import (
    get_current_settings,
    get_orchestrator,
    get_whisper_client,
    require_authorized_user,
)
from src.interfaces.api.routes.chat import router as chat_router
from tests.conftest import (
    MockConversationHistory,
    MockGraphMemory,
    MockLLMProvider,
    MockSemanticMemory,
)


class DummyRegulator:
    async def adjust_for_query(self, user_id, query):
        return RegulatorPresets.default()


@pytest.mark.anyio
async def test_chat_route_streaming_sse(tmp_path):
    settings = SethSettings(project_root=tmp_path)
    app = FastAPI()
    app.include_router(chat_router)

    chunks = [
        StreamChunk(event_type=StreamEventType.REASONING, text="Reasoning delta"),
        StreamChunk(event_type=StreamEventType.CONTENT, text="Content delta"),
    ]
    llm = MockLLMProvider(stream_batches=[chunks])

    orchestrator = ConversationOrchestrator(
        llm=llm,
        tools=ToolRegistry(),
        semantic_memory=MockSemanticMemory(),
        graph_memory=MockGraphMemory(),
        history_repo=MockConversationHistory(),
        regulator=DummyRegulator(),  # type: ignore[arg-type]
        system_prompt="Test",
        settings=settings,
    )

    app.dependency_overrides[require_authorized_user] = lambda: "authorized_user_1"
    app.dependency_overrides[get_orchestrator] = lambda: orchestrator
    app.dependency_overrides[get_current_settings] = lambda: settings
    app.dependency_overrides[get_whisper_client] = lambda: AsyncMock()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/api/chat",
            data={"message": "Hello SETH"},
            headers={"X-Seth-User": "authorized_user_1"},
        )
        assert resp.status_code == 200
        text = resp.text
        assert "data:" in text
        assert "[DONE]" in text
