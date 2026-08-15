"""
Unit tests for domain models and dataclasses (src.domain.models).
"""

from __future__ import annotations

import pytest
from src.domain.models import (
    GpuTelemetry,
    MediaAttachment,
    MediaType,
    Message,
    RegulatorPresets,
    RegulatorState,
    Role,
    Session,
    StreamEventType,
    SystemStatus,
    ToolCall,
)


def test_enums():
    assert Role.USER == "user"
    assert Role.ASSISTANT == "assistant"
    assert Role.SYSTEM == "system"
    assert Role.TOOL == "tool"

    assert StreamEventType.REASONING == "reasoning"
    assert StreamEventType.CONTENT == "content"
    assert StreamEventType.TOOL_START == "tool_start"
    assert StreamEventType.TOOL_END == "tool_end"
    assert StreamEventType.DONE == "done"
    assert StreamEventType.ERROR == "error"

    assert MediaType.IMAGE == "image"
    assert MediaType.AUDIO == "audio"


def test_message_serialization():
    # Plain text message
    msg = Message(role=Role.USER, content="Hello")
    assert msg.to_dict() == {"role": "user", "content": "Hello"}

    # Assistant message with tool calls
    tc = ToolCall(call_id="call_1", name="search", arguments='{"q": "ai"}')
    msg_with_tc = Message(role=Role.ASSISTANT, content="Searching...", tool_calls=[tc])
    d = msg_with_tc.to_dict()
    assert d["role"] == "assistant"
    assert d["tool_calls"][0]["id"] == "call_1"
    assert d["tool_calls"][0]["function"]["name"] == "search"
    assert d["tool_calls"][0]["function"]["arguments"] == '{"q": "ai"}'

    # Tool output message
    msg_tool = Message(role=Role.TOOL, content="Found 2 items", tool_call_id="call_1")
    d_tool = msg_tool.to_dict()
    assert d_tool["role"] == "tool"
    assert d_tool["tool_call_id"] == "call_1"


def test_regulator_state_interpolation():
    state = RegulatorState(temperature=0.2, top_p=0.8, presence_penalty=0.1)
    target = RegulatorState(temperature=1.0, top_p=1.0, presence_penalty=0.9)
    state.interpolate(target, alpha=0.5)

    assert pytest.approx(state.temperature, 0.001) == 0.6
    assert pytest.approx(state.top_p, 0.001) == 0.9
    assert pytest.approx(state.presence_penalty, 0.001) == 0.5


def test_regulator_state_dict_roundtrip():
    state = RegulatorState(temperature=0.7, top_p=0.9, presence_penalty=0.4)
    d = state.to_dict()
    restored = RegulatorState.from_dict(d)
    assert restored.temperature == 0.7
    assert restored.top_p == 0.9
    assert restored.presence_penalty == 0.4


def test_regulator_presets():
    assert RegulatorPresets.default().temperature == 0.25
    assert RegulatorPresets.rigorous().temperature == 0.1
    assert RegulatorPresets.chaotic().temperature == 1.3
    assert RegulatorPresets.verbose().temperature == 0.85


def test_media_attachment_and_session():
    att = MediaAttachment(media_type=MediaType.IMAGE, local_path="/tmp/img.png", url="/images/img.png")
    assert att.media_type == MediaType.IMAGE
    assert att.url == "/images/img.png"

    sess = Session(user_id="user_123", is_allowed=True, metadata={"role": "admin"})
    assert sess.user_id == "user_123"
    assert sess.is_allowed is True
    assert sess.metadata["role"] == "admin"


def test_system_status_to_dict():
    gpu = GpuTelemetry(index=0, name="RTX 4090", used_gb=12.0, total_gb=24.0)
    status = SystemStatus(
        vllm="online", whisper="online", qdrant="online", neo4j_graphiti="online", vram=[gpu]
    )
    d = status.to_dict()
    assert d["vllm"] == "online"
    assert len(d["vram"]) == 1
    assert d["vram"][0]["name"] == "RTX 4090"
