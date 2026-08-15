"""
FastAPI Application Factory for SETH-IN-A-BOX Backend Service.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI

from src.application.orchestrator import ConversationOrchestrator
from src.application.use_cases.regulate import RegulateInferenceUseCase
from src.application.use_cases.telemetry import CollectTelemetryUseCase
from src.config.settings import SethSettings, get_settings
from src.infrastructure.llm.vllm_client import VllmClient
from src.infrastructure.logging.setup import setup_logging
from src.infrastructure.memory.jsonl_history import JsonlConversationHistory
from src.infrastructure.memory.neo4j_graphiti import GraphitiRelationalMemory
from src.infrastructure.memory.qdrant_mem0 import Mem0SemanticMemory
from src.infrastructure.security.json_sessions import JsonSessionStore
from src.infrastructure.tools.code_inspector import AstCodeInspector
from src.infrastructure.tools.image_diffusion import StableDiffusionGenerator
from src.infrastructure.tools.registry import ToolRegistry, tool
from src.infrastructure.tools.speech_kokoro import KokoroSynthesizer
from src.infrastructure.tools.web_search import Crawl4AiSearcher
from src.interfaces.api.middleware.log_filter import SuppressNoisyAccessLogFilter
from src.interfaces.api.middleware.pna import PrivateNetworkAccessMiddleware
from src.interfaces.api.routes.auth import router as auth_router
from src.interfaces.api.routes.chat import router as chat_router
from src.interfaces.api.routes.status import router as status_router

logger = logging.getLogger(__name__)


def create_app(settings: Optional[SethSettings] = None) -> FastAPI:
    """Creates and configures the FastAPI application with all dependencies."""
    settings = settings or get_settings()
    setup_logging(settings)
    settings.validate_for_api()

    # 1. Initialize Infrastructures & Memories
    session_store = JsonSessionStore(settings)
    semantic_memory = Mem0SemanticMemory(settings)
    history_repo = JsonlConversationHistory(settings)

    # Preload embedding model from Mem0 for sharing with Graphiti and Regulator
    raw_mem = semantic_memory.raw_mem0
    embedding_model = raw_mem.embedding_model.model

    graph_memory = GraphitiRelationalMemory(settings, embedding_model=embedding_model)

    # 2. Tool Adapters & Registration
    search_tool = Crawl4AiSearcher()
    image_tool = StableDiffusionGenerator(settings)
    speech_tool = KokoroSynthesizer(settings)
    inspector_tool = AstCodeInspector()

    class _DomainMemoryTools:
        """Exposes Mem0 and Graphiti methods to the LLM ToolRegistry."""

        def __init__(self, sem_mem: Mem0SemanticMemory, gr_mem: GraphitiRelationalMemory) -> None:
            self.sem_mem = sem_mem
            self.gr_mem = gr_mem

        @tool
        async def save_long_term_memory(self, user_input: str, response: str, user_id: str = "anonymous") -> str:
            """Persists facts, preferences, decisions, or constraints into long-term memory."""
            ok = await self.sem_mem.save(user_id=user_id, fact=user_input, response=response)
            return '{"status": "ok"}' if ok else '{"status": "error"}'

        @tool
        async def query_relationship_graph(self, query: str, user_id: str = "anonymous") -> str:
            """Queries the temporal knowledge graph for relationships between entities over time."""
            facts = await self.gr_mem.query_relations(user_id=user_id, query=query)
            return "\n".join(facts) if facts else "No relevant relationships found in the graph."

    memory_tools = _DomainMemoryTools(semantic_memory, graph_memory)

    tool_registry = ToolRegistry()
    tool_registry.register_instances([search_tool, image_tool, speech_tool, inspector_tool, memory_tools])

    # 3. LLM & Application Use Cases
    vllm_client = VllmClient(settings)
    whisper_client = AsyncOpenAI(base_url=settings.whisper_url, api_key=settings.api_key)
    regulator_use_case = RegulateInferenceUseCase(settings, embedding_engine=embedding_model)
    telemetry_use_case = CollectTelemetryUseCase(settings, openai_client=vllm_client.client, graphiti_adapter=graph_memory)

    # Read System Prompt
    system_prompt = "You are SETH."
    if settings.system_prompt_path.exists():
        try:
            with open(settings.system_prompt_path, "r", encoding="utf-8") as f:
                system_prompt = f.read()
        except Exception as e:
            logger.error("Error reading system prompt: %s", e)

    orchestrator = ConversationOrchestrator(
        llm=vllm_client,
        tools=tool_registry,
        semantic_memory=semantic_memory,
        graph_memory=graph_memory,
        history_repo=history_repo,
        regulator=regulator_use_case,
        system_prompt=system_prompt,
        settings=settings,
    )

    # 4. Lifespan for indices warm-up
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("🚀 [SETH API STARTUP] Warming up Graphiti indices...")
        try:
            await graph_memory.get_client()
        except Exception as e:
            logger.warning("Graphiti startup notice: %s", e)
        yield
        logger.info("🛑 [SETH API SHUTDOWN] Cleanup complete.")

    app = FastAPI(title="SETH-IN-A-BOX API", version="2.0.0", lifespan=lifespan)

    # State Injection
    app.state.settings = settings
    app.state.session_store = session_store
    app.state.orchestrator = orchestrator
    app.state.telemetry_use_case = telemetry_use_case
    app.state.whisper_client = whisper_client

    # 5. Middlewares
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.parsed_cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(PrivateNetworkAccessMiddleware)

    # 6. Static Mounts for Generated Media
    os.makedirs(settings.storage_images_dir, exist_ok=True)
    os.makedirs(settings.storage_audio_dir, exist_ok=True)
    app.mount("/storage/images", StaticFiles(directory=str(settings.storage_images_dir)), name="images")
    app.mount("/storage/audio", StaticFiles(directory=str(settings.storage_audio_dir)), name="audio")

    @app.get("/", include_in_schema=False)
    async def api_root():
        """Root API endpoint returning service identification."""
        return {
            "status": "online",
            "service": "SETH-IN-A-BOX Backend Engine",
            "version": "2.0.0",
            "endpoints": {
                "status": "/api/status",
                "register": "/api/register",
                "chat": "/api/chat",
                "docs": "/docs",
            },
            "web_tui": "Run 'python -m src.main web' to open the Web TUI interface (default: http://127.0.0.1:5500/)",
        }

    # 7. Include Routers
    app.include_router(auth_router)
    app.include_router(chat_router)
    app.include_router(status_router)

    # 8. Unhandled Exception Handler
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.exception("❌ Unhandled API exception on %s: %s", request.url.path, exc)
        return JSONResponse(status_code=500, content={"status": "error", "error": str(exc)})

    # Suppress noisy polling in uvicorn access log
    logging.getLogger("uvicorn.access").addFilter(SuppressNoisyAccessLogFilter())

    return app
