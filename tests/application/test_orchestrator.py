"""
Unit tests for ConversationOrchestrator (src.application.orchestrator).
"""

from __future__ import annotations

import pytest

from src.application.orchestrator import ConversationOrchestrator
from src.domain.models import RegulatorPresets, RegulatorState, StreamChunk, StreamEventType, ToolCall
from src.infrastructure.tools.registry import ToolRegistry, tool
from tests.conftest import (
    MockConversationHistory,
    MockGraphMemory,
    MockLLMProvider,
    MockSemanticMemory,
)


class MockRegulatorUseCase:
    def __init__(self, target_state: RegulatorState | None = None):
        self.target_state = target_state or RegulatorPresets.default()

    async def adjust_for_query(self, user_id: str, query: str) -> RegulatorState:
        return self.target_state


class CalculatorTool:
    @tool
    def multiply(self, x: int, y: int) -> int:
        """Multiplies x by y."""
        return x * y


@pytest.mark.anyio
async def test_orchestrator_single_turn_streaming(mock_settings):
    chunks = [
        StreamChunk(event_type=StreamEventType.REASONING, text="Thinking..."),
        StreamChunk(event_type=StreamEventType.CONTENT, text="The result is 42."),
    ]
    llm = MockLLMProvider(stream_batches=[chunks])
    tools = ToolRegistry()
    semantic_mem = MockSemanticMemory()
    graph_mem = MockGraphMemory()
    history_repo = MockConversationHistory()
    regulator = MockRegulatorUseCase()

    orch = ConversationOrchestrator(
        llm=llm,
        tools=tools,
        semantic_memory=semantic_mem,
        graph_memory=graph_mem,
        history_repo=history_repo,
        regulator=regulator,  # type: ignore[arg-type]
        system_prompt="You are SETH.",
        settings=mock_settings,
    )

    emitted = []
    async for chunk in orch.execute_stream(user_id="user_1", user_text="What is the answer?"):
        emitted.append(chunk)

    types = [c.event_type for c in emitted]
    assert types == [StreamEventType.REASONING, StreamEventType.CONTENT, StreamEventType.DONE]

    done = emitted[-1]
    assert done.text == "The result is 42."

    # History saved
    history = await history_repo.get_history("user_1")
    assert len(history) == 2
    assert history[0].content == "What is the answer?"
    assert history[1].content == "The result is 42."


@pytest.mark.anyio
async def test_orchestrator_multi_hop_tool_dispatch(mock_settings):
    tool_service = CalculatorTool()
    tools = ToolRegistry()
    tools.register_instance(tool_service)

    tc = ToolCall(call_id="call_mult_1", name="multiply", arguments='{"x": 6, "y": 7}')

    hop1 = [
        StreamChunk(
            event_type=StreamEventType.TOOL_START,
            tool_name="multiply",
            metadata={"tool_call": tc, "assistant_message_dict": {"role": "assistant", "content": None}},
        )
    ]
    hop2 = [
        StreamChunk(event_type=StreamEventType.CONTENT, text="6 * 7 = 42."),
    ]

    llm = MockLLMProvider(stream_batches=[hop1, hop2])
    semantic_mem = MockSemanticMemory()
    graph_mem = MockGraphMemory()
    history_repo = MockConversationHistory()
    regulator = MockRegulatorUseCase()

    orch = ConversationOrchestrator(
        llm=llm,
        tools=tools,
        semantic_memory=semantic_mem,
        graph_memory=graph_mem,
        history_repo=history_repo,
        regulator=regulator,  # type: ignore[arg-type]
        system_prompt="You are SETH.",
        settings=mock_settings,
    )

    emitted = []
    async for chunk in orch.execute_stream(user_id="user_2", user_text="Calculate 6 times 7"):
        emitted.append(chunk)

    types = [c.event_type for c in emitted]
    assert StreamEventType.TOOL_START in types
    assert StreamEventType.TOOL_END in types
    assert StreamEventType.CONTENT in types
    assert StreamEventType.DONE in types

    done = emitted[-1]
    assert "multiply" in done.metadata["tool_calls_used"]
    assert done.metadata["hop_count"] == 2


@pytest.mark.anyio
async def test_orchestrator_error_handling(mock_settings):
    llm = MockLLMProvider(stream_batches=[[StreamChunk(event_type=StreamEventType.ERROR, text="Inference Timeout", ok=False)]])
    tools = ToolRegistry()
    semantic_mem = MockSemanticMemory()
    graph_mem = MockGraphMemory()
    history_repo = MockConversationHistory()
    regulator = MockRegulatorUseCase()

    orch = ConversationOrchestrator(
        llm=llm,
        tools=tools,
        semantic_memory=semantic_mem,
        graph_memory=graph_mem,
        history_repo=history_repo,
        regulator=regulator,  # type: ignore[arg-type]
        system_prompt="You are SETH.",
        settings=mock_settings,
    )

    emitted = []
    async for chunk in orch.execute_stream(user_id="user_err", user_text="Hello"):
        emitted.append(chunk)

    assert len(emitted) == 1
    assert emitted[0].event_type == StreamEventType.ERROR
    assert emitted[0].text == "Inference Timeout"
