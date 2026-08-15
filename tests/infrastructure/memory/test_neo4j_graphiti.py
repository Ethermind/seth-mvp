"""
Unit tests for GraphitiRelationalMemory (src.infrastructure.memory.neo4j_graphiti).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
import pytest

from src.config.settings import SethSettings
from src.infrastructure.memory.neo4j_graphiti import GraphitiRelationalMemory


@pytest.mark.anyio
async def test_graphiti_memory_mocked(tmp_path):
    settings = SethSettings(project_root=tmp_path)
    mem = GraphitiRelationalMemory(settings=settings)

    # Mock internal graphiti client
    mock_client = AsyncMock()
    mock_edge = MagicMock()
    mock_edge.fact = "User likes Clean Architecture"
    mock_client.search = AsyncMock(return_value=[mock_edge])
    mock_client.add_episode = AsyncMock(return_value=None)

    mem._graphiti_instance = mock_client

    # Test query
    results = await mem.query_relations(user_id="user_123", query="architecture")
    assert len(results) == 1
    assert "User likes Clean Architecture" in results[0]

    # Test add episode
    await mem.add_episode(user_id="user_123", user_text="I prefer hexagonal", assistant_text="Noted")
    mock_client.add_episode.assert_called_once()
