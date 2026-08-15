"""
Live web search and concurrent crawling adapter using DuckDuckGo and Crawl4AI.
Implements WebSearcher protocol and registers @tool methods.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Annotated, Any, Dict, List

from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from ddgs import DDGS
from src.infrastructure.tools.registry import tool

logger = logging.getLogger(__name__)


class Crawl4AiSearcher:
    """Performs live web search via DuckDuckGo and concurrent content extraction with Crawl4AI."""

    def __init__(self) -> None:
        self.browser_config = BrowserConfig(headless=True, verbose=False)
        self._crawl_semaphore = asyncio.Semaphore(3)

    @tool
    async def web_search(
        self,
        query: Annotated[str, (
            "The precise, sanitized search query. Use targeted keywords (e.g., 'fastapi lifespan syntax'). "
            "Do NOT include conversational filler, punctuation, or commands like 'search' or 'find'."
        )],
        max_results: int = 5,
    ) -> str:
        """
        EXECUTION RULES FOR LIVE WEB SEARCH: Executes a live internet search to fetch real-time data,
        current events, market conditions, or breaking updates.

        USE CASES: Use this ONLY for public knowledge that requires real-time accuracy, validation of
        recent news, or up-to-date documentation of external frameworks/libraries.

        CRITICAL RESTRICTION: Do NOT use this tool if user asks about local files or internal code.
        """
        return await self.search_and_crawl(query, max_results)

    async def search_and_crawl(self, query: str, max_results: int = 5) -> str:
        """Performs search and concurrent crawling."""
        try:
            results = await asyncio.to_thread(self._ddgs_with_retries, query, max_results)
            if not results:
                return "<WEB_SEARCH_RESULTS>No results found.</WEB_SEARCH_RESULTS>"

            context_str = "\n<WEB_SEARCH_RESULTS>\n"
            crawled_count = 0
            max_crawls = 3

            async with AsyncWebCrawler(config=self.browser_config) as crawler:
                run_config = CrawlerRunConfig(
                    cache_mode=CacheMode.BYPASS,
                    word_count_threshold=120,
                    page_timeout=7000,
                    wait_for_images=False,
                    process_iframes=False,
                )

                tasks = []
                for res in results[:max_results]:
                    url = res.get("href")
                    if url:
                        tasks.append(self._crawl_one(crawler, url, run_config, res))

                crawled_results = await asyncio.gather(*tasks, return_exceptions=True)

                for item in crawled_results:
                    if isinstance(item, Exception) or not item:
                        continue

                    url, res, crawl = item
                    if crawl and getattr(crawl, "success", False) and getattr(crawl, "markdown", None):
                        content = re.sub(r"\s+", " ", crawl.markdown.strip())[:2200]
                        context_str += f"Source: {url}\nTitle: {res.get('title', '')}\nContent: {content}\n\n"
                        crawled_count += 1
                        if crawled_count >= max_crawls:
                            break

            return context_str + "</WEB_SEARCH_RESULTS>"

        except Exception as e:
            logger.error("🌐 WEB_SEARCH ERROR: %s", e)
            return f"<WEB_SEARCH_RESULT_ERROR>{str(e)}</WEB_SEARCH_RESULT_ERROR>"

    async def _crawl_one(self, crawler: AsyncWebCrawler, url: str, run_config: CrawlerRunConfig, res: Dict[str, Any]) -> Any:
        try:
            async with self._crawl_semaphore:
                crawl = await crawler.arun(url=url, config=run_config)
                return (url, res, crawl)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug("_crawl_one failed for %s: %s", url, e)
            return None

    def _ddgs_with_retries(self, query: str, max_results: int, retries: int = 3, backoff: float = 1.0) -> List[Dict[str, Any]]:
        for attempt in range(1, retries + 1):
            try:
                with DDGS() as ddgs:
                    return list(ddgs.text(query, max_results=max_results, timelimit="m"))
            except Exception as e:
                logger.warning("DDGS attempt %d failed: %s", attempt, e)
                if attempt == retries:
                    raise
                time.sleep(backoff * attempt)
        return []
