"""
Unit tests for Crawl4AiSearcher (src.infrastructure.tools.web_search).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch
import pytest

from src.infrastructure.tools.web_search import Crawl4AiSearcher


@pytest.mark.anyio
async def test_crawl4ai_searcher_mocked():
    searcher = Crawl4AiSearcher()

    with patch.object(searcher, "search_and_crawl", new=AsyncMock(return_value="## Search Results\nPage content")):
        res = await searcher.web_search("Python 3.12 release notes")

    assert "Search Results" in res
