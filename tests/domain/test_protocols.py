"""
Unit tests verifying Protocol definitions and structural compliance (src.domain.protocols).
"""

from __future__ import annotations

import inspect
from src.domain.protocols import (
    AudioTranscriber,
    ConversationHistory,
    GraphMemory,
    HardwareProbe,
    ImageGenerator,
    LLMProvider,
    SemanticMemory,
    SessionStore,
    SpeechSynthesizer,
    WebSearcher,
)
from tests.conftest import (
    MockConversationHistory,
    MockGraphMemory,
    MockLLMProvider,
    MockSemanticMemory,
    MockSessionStore,
)


def test_protocol_mock_compliance():
    # Verify our test mocks implement the expected protocol methods
    llm = MockLLMProvider()
    assert hasattr(llm, "generate") and hasattr(llm, "generate_stream")

    sem = MockSemanticMemory()
    assert hasattr(sem, "search") and hasattr(sem, "save")

    graph = MockGraphMemory()
    assert hasattr(graph, "add_episode") and hasattr(graph, "query_relations")

    hist = MockConversationHistory()
    assert hasattr(hist, "get_history") and hasattr(hist, "append_turn")

    sess = MockSessionStore()
    assert hasattr(sess, "is_allowed") and hasattr(sess, "register")


def test_protocol_methods_exist():
    # Check method signatures on protocols
    assert "generate" in LLMProvider.__dict__
    assert "generate_stream" in LLMProvider.__dict__
    assert "search" in SemanticMemory.__dict__
    assert "save" in SemanticMemory.__dict__
    assert "add_episode" in GraphMemory.__dict__
    assert "query_relations" in GraphMemory.__dict__
    assert "get_history" in ConversationHistory.__dict__
    assert "append_turn" in ConversationHistory.__dict__
    assert "transcribe" in AudioTranscriber.__dict__
    assert "generate_image" in ImageGenerator.__dict__
    assert "synthesize" in SpeechSynthesizer.__dict__
    assert "search_and_crawl" in WebSearcher.__dict__
    assert "is_allowed" in SessionStore.__dict__
    assert "register" in SessionStore.__dict__
    assert "probe_vram" in HardwareProbe.__dict__
    assert "collect_status" in HardwareProbe.__dict__
