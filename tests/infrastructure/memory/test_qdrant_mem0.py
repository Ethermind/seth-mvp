"""
Unit tests for Mem0SemanticMemory (src.infrastructure.memory.qdrant_mem0).
"""

from __future__ import annotations

from unittest.mock import MagicMock
import pytest

from src.config.settings import SethSettings
from src.infrastructure.memory.qdrant_mem0 import Mem0SemanticMemory


@pytest.mark.anyio
async def test_mem0_semantic_memory_search_and_save(tmp_path):
    settings = SethSettings(project_root=tmp_path)
    mem_adapter = Mem0SemanticMemory(settings=settings)

    mock_mem0 = MagicMock()
    mock_mem0.search.return_value = [{"memory": "User is a senior software architect."}]
    mock_mem0.add.return_value = {"status": "ok"}
    mem_adapter._memory_instance = mock_mem0

    # Search
    results = await mem_adapter.search("user_abc", "What is my role?")
    assert len(results) == 1
    assert "User is a senior software architect." in results[0]

    # Save
    success = await mem_adapter.save("user_abc", "I am building SETH-IN-A-BOX", "Acknowledged")
    assert success is True
    mock_mem0.add.assert_called_once()
