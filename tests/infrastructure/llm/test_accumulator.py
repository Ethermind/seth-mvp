"""
Unit tests for StreamAccumulator (src.infrastructure.llm.accumulator).
"""

from __future__ import annotations

from unittest.mock import MagicMock
from src.infrastructure.llm.accumulator import StreamAccumulator


def test_accumulator_text_accumulation():
    acc = StreamAccumulator()
    acc.reasoning += "Thinking step 1. "
    acc.reasoning += "Thinking step 2."
    acc.content += "Result content"

    assert acc.reasoning == "Thinking step 1. Thinking step 2."
    assert acc.content == "Result content"


def test_accumulator_tool_call_deltas():
    acc = StreamAccumulator()

    # Simulated OpenAI tool call delta 1
    d1 = MagicMock()
    d1.index = 0
    d1.id = "call_abc"
    d1.function.name = "web_search"
    d1.function.arguments = '{"query": "'

    # Simulated OpenAI tool call delta 2
    d2 = MagicMock()
    d2.index = 0
    d2.id = None
    d2.function.name = None
    d2.function.arguments = 'fastapi"}'

    acc.ingest_tool_call_deltas([d1, d2])

    tool_calls = acc.to_domain_tool_calls()
    assert len(tool_calls) == 1
    assert tool_calls[0].call_id == "call_abc"
    assert tool_calls[0].name == "web_search"
    assert tool_calls[0].arguments == '{"query": "fastapi"}'

    msg = acc.to_message_dict()
    assert msg["role"] == "assistant"
    assert len(msg["tool_calls"]) == 1
