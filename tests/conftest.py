"""
Global test fixtures and protocol mock implementations for SETH-IN-A-BOX test suite.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional
import pytest

from src.config.settings import SethSettings
from src.domain.models import (
    Message,
    RegulatorPresets,
    RegulatorState,
    Role,
    StreamChunk,
    StreamEventType,
    ToolCall,
)
from src.domain.protocols import ConversationHistory, GraphMemory, LLMProvider, SemanticMemory, SessionStore


@pytest.fixture
def mock_settings(tmp_path: Path) -> SethSettings:
    """Provides an isolated SethSettings instance pointing to temporary directories."""
    src_dir = tmp_path / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir = src_dir / "prompt"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    (prompt_dir / "seth.md").write_text("You are SETH Test Mock Prompt.", encoding="utf-8")

    return SethSettings(
        src_dir=src_dir,
        project_root=tmp_path,
        registration_token="test-secret-token",
        allowed_api_user_ids="",
        vllm_url="http://mock-vllm:8000/v1",
        whisper_url="http://mock-whisper:8010/v1",
    )


class MockLLMProvider:
    """Mock implementation of LLMProvider protocol for deterministic testing."""

    def __init__(self, stream_batches: Optional[List[List[StreamChunk]]] = None) -> None:
        self.stream_batches = stream_batches or []
        self.recorded_calls: List[Dict[str, Any]] = []

    async def generate(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        config: Optional[RegulatorState] = None,
    ) -> Message:
        self.recorded_calls.append({"messages": messages, "tools": tools, "config": config})
        return Message(role=Role.ASSISTANT, content="Mock non-streaming response")

    async def generate_stream(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        config: Optional[RegulatorState] = None,
    ) -> AsyncIterator[StreamChunk]:
        self.recorded_calls.append({"messages": messages, "tools": tools, "config": config})
        if self.stream_batches:
            batch = self.stream_batches.pop(0)
            for chunk in batch:
                yield chunk
        else:
            yield StreamChunk(event_type=StreamEventType.CONTENT, text="Hello from mock LLM!")


class MockSemanticMemory:
    """In-memory dictionary mock implementing SemanticMemory protocol."""

    def __init__(self) -> None:
        self.storage: Dict[str, List[str]] = {}

    async def search(self, user_id: str, query: str, limit: int = 10) -> List[str]:
        return self.storage.get(user_id, [])

    async def save(self, user_id: str, fact: str, response: str) -> bool:
        self.storage.setdefault(user_id, []).append(fact)
        return True


class MockGraphMemory:
    """In-memory mock implementing GraphMemory protocol."""

    def __init__(self) -> None:
        self.episodes: List[Dict[str, str]] = []

    async def add_episode(self, user_id: str, user_text: str, assistant_text: str) -> None:
        self.episodes.append({"user_id": user_id, "user": user_text, "assistant": assistant_text})

    async def query_relations(self, user_id: str, query: str) -> List[str]:
        return [f"Relation for {query}"]


class MockConversationHistory:
    """In-memory mock implementing ConversationHistory protocol."""

    def __init__(self) -> None:
        self.histories: Dict[str, List[Message]] = {}

    async def get_history(self, user_id: str) -> List[Message]:
        return list(self.histories.get(user_id, []))

    async def append_turn(self, user_id: str, user_text: str, assistant_text: str) -> None:
        user_list = self.histories.setdefault(user_id, [])
        user_list.append(Message(role=Role.USER, content=user_text))
        user_list.append(Message(role=Role.ASSISTANT, content=assistant_text))


class MockSessionStore:
    """In-memory mock implementing SessionStore protocol."""

    def __init__(self, valid_token: str = "test-secret-token") -> None:
        self.valid_token = valid_token
        self.allowed_users: set[str] = set()

    def is_allowed(self, user_id: str) -> bool:
        return user_id in self.allowed_users

    def register(self, user_id: str, token: str) -> bool:
        if token == self.valid_token:
            self.allowed_users.add(user_id)
            return True
        return False
