"""
Semantic long-term memory adapter using Mem0 and Qdrant.
Implements the domain SemanticMemory protocol.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
import logging
from typing import Any, Dict, List

from mem0 import Memory
from src.config.settings import SethSettings, get_settings

logger = logging.getLogger(__name__)


class Mem0SemanticMemory:
    """Manages semantic long-term user memories indexed into Qdrant via Mem0."""

    def __init__(self, settings: SethSettings | None = None, collection_name: str = "SETH_CORE_SPACE") -> None:
        self.settings = settings or get_settings()
        self.collection_name = collection_name
        self._memory_instance: Memory | None = None
        self._lock = asyncio.Lock()

    def _get_or_create_mem0(self) -> Memory:
        if self._memory_instance is None:
            config = {
                "llm": {
                    "provider": "openai",
                    "config": {
                        "model": self.settings.llm_model,
                        "openai_base_url": self.settings.vllm_url,
                        "api_key": self.settings.api_key,
                    },
                },
                "embedder": {
                    "provider": "huggingface",
                    "config": {"model": self.settings.embedding_model},
                },
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "host": self.settings.qdrant_host,
                        "port": self.settings.qdrant_port,
                        "collection_name": self.collection_name,
                        "embedding_model_dims": self.settings.embedding_dims,
                    },
                },
            }
            self._memory_instance = Memory.from_config(config)
        return self._memory_instance

    @property
    def raw_mem0(self) -> Memory:
        """Exposes raw Mem0 instance for embedder sharing."""
        return self._get_or_create_mem0()

    async def search(self, user_id: str, query: str, limit: int = 10) -> List[str]:
        """Searches long-term memories for a specific user asynchronously."""
        if not query.strip():
            return []

        def _sync_search() -> List[str]:
            mem = self._get_or_create_mem0()
            raw = mem.search(query, filters={"user_id": user_id}, limit=limit)
            results = raw if isinstance(raw, list) else raw.get("results", [])
            records = sorted(results, key=lambda x: x.get("score", 0), reverse=True)
            return [r["memory"].strip() for r in records if r.get("memory")]

        try:
            return await asyncio.to_thread(_sync_search)
        except Exception as e:
            logger.warning("Error retrieving memories for user=%s: %s", user_id, e)
            return []

    async def save(self, user_id: str, fact: str, response: str) -> bool:
        """Saves a factual statement to user's long-term memory."""
        expiration = (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d")

        def _sync_save() -> Any:
            mem = self._get_or_create_mem0()
            return mem.add(
                [
                    {"role": "user", "content": fact},
                    {"role": "assistant", "content": response},
                ],
                user_id=user_id,
                agent_id="SETH",
                metadata={"memory_bucket": "constraints", "expires_on": expiration},
            )

        try:
            await asyncio.to_thread(_sync_save)
            return True
        except Exception as e:
            logger.error("Failed to save memory for user=%s: %s", user_id, e)
            return False
