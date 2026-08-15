"""
Temporal and relational knowledge graph memory adapter using Graphiti and Neo4j.
Implements the domain GraphMemory protocol.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
import logging
import time
from typing import Any, List, Optional

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.llm_client import OpenAIClient, LLMConfig
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.cross_encoder.bge_reranker_client import BGERerankerClient
from graphiti_core.llm_client.gliner2_client import GLiNER2Client

from src.config.settings import SethSettings, get_settings

logger = logging.getLogger(__name__)


class _LocalBgeEmbedder(EmbedderClient):
    """Bridge for Graphiti to use the local sentence-transformers model."""

    def __init__(self, sentence_transformer_model: Any) -> None:
        self._model = sentence_transformer_model

    async def create(self, input_data: Any) -> List[float]:
        texts = input_data if isinstance(input_data, list) else [input_data]
        vectors = await asyncio.to_thread(self._model.encode, texts)
        return vectors[0].tolist()

    async def create_batch(self, input_data_list: List[str]) -> List[List[float]]:
        vectors = await asyncio.to_thread(self._model.encode, input_data_list)
        return [v.tolist() for v in vectors]


class GraphitiRelationalMemory:
    """Manages knowledge graph temporal episodes and relationship queries via Graphiti + Neo4j."""

    def __init__(self, settings: SethSettings | None = None, embedding_model: Any | None = None) -> None:
        self.settings = settings or get_settings()
        self.embedding_model = embedding_model
        self._graphiti_instance: Optional[Graphiti] = None
        self._lock = asyncio.Lock()

    async def get_client(self) -> Graphiti:
        """Initializes and returns the Graphiti instance with warmed-up indices."""
        if self._graphiti_instance is not None:
            return self._graphiti_instance

        async with self._lock:
            if self._graphiti_instance is None:
                vllm_backend = OpenAIClient(
                    config=LLMConfig(
                        api_key=self.settings.api_key,
                        model=self.settings.llm_model,
                        small_model=self.settings.llm_model,
                        base_url=self.settings.vllm_url,
                    )
                )

                llm_client = await asyncio.to_thread(GLiNER2Client, llm_client=vllm_backend)

                embedder_model = self.embedding_model
                if embedder_model is None:
                    # Fallback to loading sentence transformers if not injected
                    from sentence_transformers import SentenceTransformer
                    embedder_model = SentenceTransformer(self.settings.embedding_model, device="cuda" if self.settings.neo4j_uri else "cpu")

                graphiti = Graphiti(
                    uri=self.settings.neo4j_uri,
                    user=self.settings.neo4j_user,
                    password=self.settings.neo4j_password,
                    llm_client=llm_client,
                    embedder=_LocalBgeEmbedder(embedder_model),
                    cross_encoder=BGERerankerClient(),
                )

                logger.info("🕸️ [GRAPHITI] Building indices and constraints in Neo4j...")
                try:
                    await graphiti.build_indices_and_constraints()
                except Exception as e:
                    logger.warning("⚠️ [GRAPHITI] Indices warm-up warning (non-fatal): %s", e)

                self._graphiti_instance = graphiti

        return self._graphiti_instance

    async def add_episode(self, user_id: str, user_text: str, assistant_text: str) -> None:
        """Appends a new conversation turn to Graphiti in background."""
        try:
            client = await self.get_client()
            await client.add_episode(
                name=f"turn_{user_id}_{int(time.time())}",
                episode_body=f"User: {user_text}\nAssistant: {assistant_text}",
                source=EpisodeType.message,
                source_description="SETH conversation turn",
                reference_time=datetime.now(),
                group_id=user_id,
            )
            logger.info("🕸️ [GRAPHITI] Episode saved for user=%s.", user_id)
        except Exception as e:
            logger.warning("⚠️ [GRAPHITI] Error saving episode (non-critical): %s", e)

    async def query_relations(self, user_id: str, query: str) -> List[str]:
        """Queries relational and temporal connections in the knowledge graph."""
        try:
            client = await self.get_client()
            results = await client.search(query=query, group_ids=[user_id], num_results=16)
            if not results:
                return []
            return [f"- {r.fact[:1024]}" for r in results if getattr(r, "fact", None)]
        except Exception as e:
            logger.warning("⚠️ [GRAPHITI] Error in query_relations: %s", e)
            return []
