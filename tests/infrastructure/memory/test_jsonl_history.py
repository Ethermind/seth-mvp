"""
Unit tests for JsonlConversationHistory (src.infrastructure.memory.jsonl_history).
"""

from __future__ import annotations

from pathlib import Path
import pytest

from src.config.settings import SethSettings
from src.domain.models import Role
from src.infrastructure.memory.jsonl_history import JsonlConversationHistory


@pytest.mark.anyio
async def test_jsonl_history_append_and_recover(tmp_path: Path):
    settings = SethSettings(project_root=tmp_path)
    history_repo = JsonlConversationHistory(settings=settings, max_history=5)

    # Empty initially
    msgs = await history_repo.get_history("user_abc")
    assert msgs == []

    # Append turns
    await history_repo.append_turn("user_abc", "Hello", "Hi Luis")
    await history_repo.append_turn("user_abc", "How are you?", "Operational.")

    # Read back from same instance
    msgs = await history_repo.get_history("user_abc")
    assert len(msgs) == 4
    assert msgs[0].role == Role.USER
    assert msgs[0].content == "Hello"
    assert msgs[1].role == Role.ASSISTANT
    assert msgs[1].content == "Hi Luis"

    # Create new instance to test cold restoration from JSONL on disk
    history_repo_cold = JsonlConversationHistory(settings=settings, max_history=5)
    restored_msgs = await history_repo_cold.get_history("user_abc")
    assert len(restored_msgs) == 4
    assert restored_msgs[2].content == "How are you?"
    assert restored_msgs[3].content == "Operational."
