"""
Unit tests for VllmClient (src.infrastructure.llm.vllm_client).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
import pytest

from src.config.settings import SethSettings
from src.domain.models import Message, Role, StreamEventType
from src.infrastructure.llm.vllm_client import VllmClient


@pytest.mark.anyio
async def test_vllm_client_generate(tmp_path):
    settings = SethSettings(project_root=tmp_path)
    mock_openai = AsyncMock()

    mock_choice = MagicMock()
    mock_choice.message.content = "Response from mock vLLM"
    mock_choice.message.tool_calls = None
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)

    client = VllmClient(settings=settings, client=mock_openai)
    msg = await client.generate(messages=[Message(role=Role.USER, content="Hello")])

    assert msg.role == Role.ASSISTANT
    assert msg.content == "Response from mock vLLM"
    assert len(msg.tool_calls) == 0


@pytest.mark.anyio
async def test_vllm_client_generate_stream(tmp_path):
    settings = SethSettings(project_root=tmp_path)
    mock_openai = AsyncMock()

    # Create mock stream chunks
    chunk1 = MagicMock()
    chunk1.choices = [MagicMock(delta=MagicMock(reasoning="Thinking...", content=None, tool_calls=None), finish_reason=None)]

    chunk2 = MagicMock()
    chunk2.choices = [MagicMock(delta=MagicMock(reasoning=None, content="Answer text.", tool_calls=None), finish_reason="stop")]

    async def mock_stream_iter():
        yield chunk1
        yield chunk2

    mock_openai.chat.completions.create = AsyncMock(return_value=mock_stream_iter())

    client = VllmClient(settings=settings, client=mock_openai)
    chunks = []
    async for c in client.generate_stream(messages=[Message(role=Role.USER, content="Hello")]):
        chunks.append(c)

    assert len(chunks) == 2
    assert chunks[0].event_type == StreamEventType.REASONING
    assert chunks[0].text == "Thinking..."
    assert chunks[1].event_type == StreamEventType.CONTENT
    assert chunks[1].text == "Answer text."
