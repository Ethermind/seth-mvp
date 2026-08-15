"""
Unit tests for ContextWindowGuard (src.infrastructure.llm.context_guard).
"""

from __future__ import annotations

from unittest.mock import MagicMock
import pytest

from src.config.settings import SethSettings
from src.domain.exceptions import ContextLimitExceededError
from src.infrastructure.llm.context_guard import ContextWindowGuard


def test_context_guard_safe_output_calculation(tmp_path):
    settings = SethSettings(project_root=tmp_path, max_tokens=8192)
    guard = ContextWindowGuard(settings)

    # Mock apply_chat_template to simulate token estimation
    guard.tokenizer.apply_chat_template = MagicMock(return_value=list(range(200)))

    short_messages = [{"role": "user", "content": "Short query"}]
    safe_out = guard.calculate_safe_output_tokens(raw_messages=short_messages, requested_output_tokens=4096)
    assert safe_out == 4096

    # Simulate heavy prompt approaching context limit (e.g. 7000 tokens)
    guard.tokenizer.apply_chat_template = MagicMock(return_value=list(range(7000)))
    clamped_out = guard.calculate_safe_output_tokens(raw_messages=short_messages, requested_output_tokens=4096)
    assert clamped_out < 4096
    assert clamped_out >= 256

    # Simulate overflow
    guard.tokenizer.apply_chat_template = MagicMock(return_value=list(range(8100)))
    with pytest.raises(ContextLimitExceededError):
        guard.calculate_safe_output_tokens(raw_messages=short_messages, requested_output_tokens=4096)
