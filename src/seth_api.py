"""
SETH-IN-A-BOX -- API EDITION (seth_api.py)
"Inspired by my prompt engineering research and publications on Medium: https://medium.com/@luis.capra"

Same runtime as seth_poc.py -- tool calling, per-user short/graph memory, the
dynamic regulator, Mem0+Qdrant, Graphiti+Neo4j, local image/speech generation
-- with the Telegram-specific glue (SethTelegramBot, python-telegram-bot)
replaced by a FastAPI + Server-Sent-Events surface, so any JavaScript client
(e.g. the_oracle.html) can drive Seth over plain HTTP instead of Telegram.

This file is the one process in the whole stack that actually loads model
weights or talks to vLLM/Qdrant/Neo4j directly -- vLLM client, Mem0+Qdrant,
Graphiti+Neo4j, the BGE embedder/reranker, and the local image/speech models
all live here, in-process, exactly once. seth_telegram.py and the_oracle.html
are both thin HTTP clients with zero GPU/model footprint of their own, so
running seth_api.py + seth_telegram.py (+ the_oracle.html) concurrently is
the intended setup, not a VRAM risk -- there's only ever one copy of
anything GPU-resident on the box.

seth_poc.py is the exception: it's the original, fully self-contained
Telegram bot this file's classes were extracted from, and it still loads its
own separate copies of the embedder/diffusion/etc. models. Running seth_poc.py
alongside seth_api.py at the same time WOULD double up VRAM usage on the RTX
5090 -- so once seth_telegram.py is in use, seth_poc.py should be considered
retired rather than run side by side with this file.
"""

from __future__ import annotations

import ast
import asyncio
import base64
from collections import deque
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, fields, is_dataclass, replace, MISSING
from datetime import datetime, timedelta, date, time as _dt_time
import enum
from functools import lru_cache
import io
import inspect
import json
import logging
import os
from pathlib import Path
import re
import time
from typing import (
    Annotated,
    Any,
    Callable,
    Dict,
    Literal,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)
from urllib.parse import urlsplit
from uuid import UUID, uuid4

import coloredlogs
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from ddgs import DDGS
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from dotenv import load_dotenv
import httpx
from kokoro import KPipeline
from mem0 import Memory
import numpy as np
from openai import AsyncOpenAI
from pydantic import BaseModel
from pydub import AudioSegment
from sentence_transformers import SentenceTransformer
import soundfile as sf
import torch
from transformers import AutoTokenizer
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.llm_client import OpenAIClient, LLMConfig
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.cross_encoder.bge_reranker_client import BGERerankerClient
from graphiti_core.llm_client.gliner2_client import GLiNER2Client

# --- API layer (replaces python-telegram-bot as the transport) ---
import uvicorn
from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# Anchor point for every relative path this file uses by default (storage/,
# conversations/, seth.md, seth.state, models/, .env). Using Path(__file__)
# instead of bare relative strings means the app behaves identically no
# matter where `python seth_api.py` is actually launched from -- a plain
# shell cd'd into this folder, VS Code's integrated terminal (which may use
# a different cwd), a systemd unit, cron, etc. Env vars that explicitly set
# one of these paths are still respected as-is; only the *defaults* anchor here.
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# One level up from src/ -- runtime-generated data (storage/, conversations/)
# lives at the project root, next to README.md, kept separate from source
# code in src/. Config/assets that ARE part of the app (.env, seth.md,
# seth.state, models/) stay anchored to BASE_DIR instead -- see each field below.
PROJECT_ROOT = BASE_DIR.parent


@dataclass(frozen=True)
class SethEnvironment:
    """Centralized and typed configuration for the SETH ecosystem (API edition)."""
    llm_model: str = os.getenv("LLM_MODEL", "nvidia/Gemma-4-26B-A4B-NVFP4")
    vllm_url: str = os.getenv("VLLM_URL", "http://localhost:8000/v1")
    whisper_url: str = os.getenv("WHISPER_URL", "http://localhost:8010/v1")
    whisper_model: str = os.getenv("WHISPER_MODEL", "large-v3")
    image_model: str = os.getenv("IMAGE_MODEL", "dreamshaper_8.safetensors")
    image_model_full_path: str = str(PROJECT_ROOT / "models" / image_model)
    # NOTE: no telegram_token here -- this runtime has no Telegram transport.
    # REGISTRATION_TOKEN is reused as-is from the same .env: it still gates who
    # gets a session, just handed out over POST /api/register instead of a
    # Telegram DM with the magic word.
    api_registration_token: str = os.getenv("REGISTRATION_TOKEN", "")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
    embedding_dims: int = int(os.getenv("EMBEDDING_MODEL_DIMS", 1024))
    system_prompt_path: str = str(BASE_DIR / "seth.md")
    conversations_path: str = str(PROJECT_ROOT / "conversations") + os.sep
    state_path: str = str(PROJECT_ROOT / "storage" / "state" / "seth.state")
    storage_images_dir: str = str(PROJECT_ROOT / "storage" / "images")
    storage_audio_dir: str = str(PROJECT_ROOT / "storage" / "audio")
    reasoning_audit_dir: str = str(PROJECT_ROOT / "storage" / "logs" / "reasoning")
    reasoning_audit_retention_days: int = int(os.getenv("AUDIT_LOG_RETENTION_DAYS", 30))
    qdrant_host: str = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port: int = int(os.getenv("QDRANT_PORT", 6333))
    max_tokens: int = int(os.getenv("MAX_TOKENS", 131072))
    llm_enable_thinking: bool = os.getenv("LLM_ENABLE_THINKING", "1").strip().lower() in ("1", "true", "yes")
    api_key: str = os.getenv("API_KEY", "NONE")
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "")
    # --- API-specific settings (no equivalent in seth_poc.py) ---
    api_host: str = os.getenv("API_HOST", "127.0.0.1")
    api_port: int = int(os.getenv("API_PORT", 8080))
    # Comma-separated list of allowed browser origins, e.g. "http://localhost:5500,http://127.0.0.1:5500".
    # Defaults to "*" for local dev convenience -- tighten this before exposing the port beyond localhost.
    cors_allowed_origins: str = os.getenv("CORS_ALLOWED_ORIGINS", "*")

    def validate(self):
        """Validates critical environment variables."""
        if not self.api_registration_token:
            raise ValueError("❌ REGISTRATION_TOKEN is missing in the environment.")


@dataclass(slots=True)
class Tool:
    name: str
    func: Callable[..., Any]
    description: str
    schema: dict[str, Any]


class SethLoggerInit:
    """Centralized logging configuration for the SETH ecosystem."""
    def __init__(self):
        self.prepare_coloredlogs()
        self.prepare_silence()
        self.prepare_mem0()
        self.prepare_run()

    def prepare_coloredlogs(self):
        """Configures colored logging for the SETH ecosystem."""
        coloredlogs.install(
            level='INFO',
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            level_styles={
                'info': {'color': 'green'},
                'warning': {'color': 'yellow', 'bold': True},
                'error': {'color': 'red', 'bold': True},
                'critical': {'color': 'red', 'bg': 'white', 'bold': True},
                'debug': {'color': 'black', 'bright': True}
            },
            field_styles={
                'asctime': {'color': 'cyan'},
                'hostname': {'color': 'magenta'},
                'levelname': {'color': 'white', 'bold': True},
                'name': {'color': 'blue'}
            }
        )

    def prepare_silence(self):
        """Suppresses verbose logging from third-party libraries."""
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)

    def prepare_mem0(self):
        """Initializes the mem0 logger and ensures the log directory exists."""
        mem0_log_path = os.getenv("LOG_MEM0_PATH", str(PROJECT_ROOT / "storage" / "logs" / "mem0.log"))
        os.makedirs(os.path.dirname(mem0_log_path), exist_ok=True)
        mem0_logger = logging.getLogger("mem0")
        mem0_file_handler = logging.FileHandler(mem0_log_path)
        mem0_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        mem0_logger.addHandler(mem0_file_handler)
        mem0_logger.setLevel(logging.INFO)

    def prepare_run(self):
        """Sets up a dedicated log file for each run of the SETH ecosystem."""
        run_logs_dir = str(PROJECT_ROOT / "storage" / "logs" / "runs")
        os.makedirs(run_logs_dir, exist_ok=True)
        start_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_log_file = os.path.join(run_logs_dir, f"run_{start_time_str}.log")
        run_file_handler = logging.FileHandler(run_log_file, encoding='utf-8')
        run_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
        run_file_handler.setLevel(logging.INFO)
        logging.getLogger().addHandler(run_file_handler)


# Carries the identity of whoever is talking to Seth right now. Set once per
# HTTP request in SethAPIBot's request-scoped SSE generator (the async-generator
# equivalent of SethTelegramBot.process() in the original bot), read by
# memory/RAG tools. ContextVars are copied per asyncio Task, and every HTTP
# request is handled by its own Task end-to-end (including streaming the
# response body), so concurrent users never cross-read each other's id even
# though they share the same process. Deliberately NOT exposed as a tool-call
# parameter: if it were, the LLM (or a crafted prompt) could ask Seth to fetch
# "user X's memory" and get cross-user data leakage.
current_user_id: ContextVar[str] = ContextVar("current_user_id", default="anonymous")


class SethToolsManager:
    """
    Unified engine for LLM Tool discovery, validation, schema generation,
    and safe multi-threaded execution for the SETH ecosystem.
    """

    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    @staticmethod
    def tool(func: Callable) -> Callable:
        """
        Marks a method as an LLM-invocable tool. No description is passed here:
        the method's own docstring IS the LLM-facing description (via inspect.getdoc).
        Usage inside tool classes: @SethToolsManager.tool
        """
        func.__tool__ = True
        return func

    def _build_schema_from_signature(self, method: Callable) -> dict:
        """
        100% IA generated :)

        Builds the JSON-schema `parameters` object from Annotated type hints.

        Parameters not wrapped in Annotated are considered internal and are not
        exposed to the LLM.
        """
        hints = get_type_hints(method, include_extras=True)

        properties = {}
        required = []

        for name, param in inspect.signature(method).parameters.items():
            schema = self._parameter_schema(name, hints)

            if schema is None:
                continue

            properties[name] = schema

            if param.default is inspect.Parameter.empty:
                required.append(name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    def _parameter_schema(self, name: str, hints: dict[str, Any],) -> dict | None:
        if name == "self":
            return None

        hint = hints.get(name)

        if hint is None or get_origin(hint) is not Annotated:
            return None

        base_type, *metadata = get_args(hint)

        schema = self._schema(base_type)

        if metadata:
            schema["description"] = str(metadata[0])

        return schema

    @staticmethod
    @lru_cache
    def _schema(tp):
        origin = get_origin(tp)
        args = get_args(tp)

        primitives = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
        }

        if tp in primitives:
            return {"type": primitives[tp]}

        if tp is Any:
            return {}

        if tp is UUID:
            return {"type": "string", "format": "uuid"}

        if tp is Path:
            return {"type": "string"}

        if tp is datetime:
            return {"type": "string", "format": "date-time"}

        if tp is date:
            return {"type": "string", "format": "date"}

        if tp is _dt_time:
            return {"type": "string", "format": "time"}

        if inspect.isclass(tp) and issubclass(tp, enum.Enum):
            values = [m.value for m in tp]

            if not values:
                return {"type": "string"}

            schema = dict(SethToolsManager._schema(type(values[0])))
            schema["enum"] = values

            return schema

        if origin is Literal:
            values = list(args)

            if not values:
                return {"type": "string"}

            schema = dict(SethToolsManager._schema(type(values[0])))
            schema["enum"] = values

            return schema

        if origin is Union:
            non_none = [a for a in args if a is not type(None)]

            if len(non_none) == 1:
                schema = dict(SethToolsManager._schema(non_none[0]))
                schema["nullable"] = True
                return schema

            return {
                "anyOf": [SethToolsManager._schema(a) for a in non_none]
            }

        if origin in (list, tuple, set):
            return {
                "type": "array",
                "items": SethToolsManager._schema(args[0] if args else str),
            }

        if origin is dict:
            value_type = args[1] if len(args) == 2 else Any
            return {
                "type": "object",
                "additionalProperties": SethToolsManager._schema(value_type),
            }

        if inspect.isclass(tp) and is_dataclass(tp):
            hints = get_type_hints(tp)
            return {
                "type": "object",
                "properties": {
                    f.name: SethToolsManager._schema(hints.get(f.name, Any))
                    for f in fields(tp)
                },
                "required": [
                    f.name
                    for f in fields(tp)
                    if f.default is MISSING and f.default_factory is MISSING
                ],
            }
        if (
            inspect.isclass(tp)
            and hasattr(tp, "__annotations__")
            and hasattr(tp, "__total__")
        ):
            hints = get_type_hints(tp)
            return {
                "type": "object",
                "properties": {
                    k: SethToolsManager._schema(v)
                    for k, v in hints.items()
                },
                "required": list(hints.keys()) if tp.__total__ else [],
            }

        try:
            if inspect.isclass(tp) and issubclass(tp, BaseModel):
                if hasattr(tp, "model_json_schema"):
                    return tp.model_json_schema()

                return tp.schema()

        except Exception:
            pass

        return {"type": "string"}

    def register_instance(self, instance: Any):
        for name, method in inspect.getmembers(instance, predicate=inspect.ismethod):
            if not getattr(method.__func__, "__tool__", False):
                continue

            description = inspect.getdoc(method) or ""

            schema = self._build_schema_from_signature(method)
            
            self.tools[name] = Tool(
                name=name,
                func=method,
                description=description,
                schema=schema
            )
            logging.info(f"🛠️ [TOOLS MANAGER] Auto-registered tool: '{name}'")

    def register_instances(self, instances: list[Any]):
        for instance in instances:
            self.register_instance(instance)

    def as_vllm_format(self) -> list:
        """
        Returns the standard OpenAI/vLLM format for tool definition lists.
        """
        openai_tools = []
        for name, tool in self.tools.items():
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.schema
                }
            })
        return openai_tools

    def get_function(self, name: str) -> Callable:
        """
        Returns the executable callable method for the given tool name.
        Fixes the missing function retrieval error in the bot's processing stage.
        """
        if name not in self.tools:
            raise KeyError(f"Tool '{name}' is not registered in SethToolsManager.")
        return self.tools[name].func


class SethSearchTool:
    """Web retrieval and extraction tool"""
    def __init__(self):
        self.browser_config = BrowserConfig(headless=True, verbose=False)
        self._crawl_semaphore = asyncio.Semaphore(3)

    @SethToolsManager.tool
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

        CASCADE RESOLUTION PROTOCOL: If a query is about public facts or external technical data, and
        your internal knowledge is insufficient OR 'retrieve_long_term_memory' returned empty/outdated
        results for that specific public fact, you MUST invoke this tool.

        CRITICAL RESTRICTION: Do NOT use this tool if the user is asking about local files, private logs,
        personal source code, or internal project states. For local file inspection, use
        'inspect_own_source_code' instead.

        Implementation: fetches max_results URLs from DuckDuckGo and crawls them concurrently.
        """
        try:
            results = await asyncio.to_thread(self._ddgs_with_retries, query, max_results)

            if not results:
                return "<WEB_SEARCH_RESULTS>No results.</WEB_SEARCH_RESULTS>"

            context_str = "\n<WEB_SEARCH_RESULTS>\n"
            crawled_count = 0
            max_crawls = 3

            async with AsyncWebCrawler(config=self.browser_config) as crawler:
                run_config = CrawlerRunConfig(
                    cache_mode=CacheMode.BYPASS,
                    word_count_threshold=120,
                    page_timeout=7000,
                    wait_for_images=False,
                    process_iframes=False
                )

                tasks = []
                for res in results[:max_results]:
                    url = res.get('href')
                    if url:
                        tasks.append(self._crawl_one(crawler, url, run_config, res))

                crawled_results = await asyncio.gather(*tasks, return_exceptions=True)

                for item in crawled_results:
                    if isinstance(item, Exception) or not item:
                        continue

                    url, res, crawl = item
                    if crawl and getattr(crawl, 'success', False) and getattr(crawl, 'markdown', None):
                        content = re.sub(r'\s+', ' ', crawl.markdown.strip())[:2200]
                        context_str += f"Source: {url}\nTitle: {res.get('title','')}\nContent: {content}\n\n"
                        
                        crawled_count += 1
                        if crawled_count >= max_crawls:
                            break

            return context_str + "</WEB_SEARCH_RESULTS>"
            
        except Exception as e:
            logging.error(f"🌐 WEB_SEARCH ERROR: {str(e)}")
            return f"<WEB_SEARCH_RESULT_ERROR>{str(e)}</WEB_SEARCH_RESULT_ERROR>"

    async def _crawl_one(self, crawler: AsyncWebCrawler, url: str, run_config: CrawlerRunConfig, res: dict):
        try:
            async with self._crawl_semaphore:
                crawl = await crawler.arun(url=url, config=run_config)
                return (url, res, crawl)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logging.debug(f"_crawl_one failed for {url}: {e}")
            return None

    def _ddgs_with_retries(self, query: str, max_results: int, retries: int = 3, backoff: float = 1.0):
        for attempt in range(1, retries + 1):
            try:
                with DDGS() as ddgs:
                    return list(ddgs.text(query, max_results=max_results, timelimit="m"))
            except Exception as e:
                logging.warning(f"DDGS attempt {attempt} failed: {e}")
                if attempt == retries:
                    raise
                time.sleep(backoff * attempt)
        return []


class Mem0MemoryBuilder:
    def __init__(self, collection_name: str, env: SethEnvironment):
        self.env = env
        self.collection_name = collection_name
        self.mem0_config = {
            "llm": {
                "provider": "openai", 
                "config": {"model": env.llm_model, "openai_base_url": env.vllm_url, "api_key": env.api_key}
            },
            "embedder": {
                "provider": "huggingface",
                "config": {"model": env.embedding_model}
            },
            "vector_store": {
                "provider": "qdrant", 
                "config": {
                    "host": env.qdrant_host,
                    "port": env.qdrant_port,
                    "collection_name": self.collection_name,
                    "embedding_model_dims": env.embedding_dims
                }
            }
        }
        self._memory = None

    def build(self) -> Memory:
        if self._memory is None:
            self._memory = Memory.from_config(self.mem0_config)
        return self._memory


class Mem0MemorySingleton:
    _instance: "Memory | None" = None

    @classmethod
    def get(cls, env: SethEnvironment) -> "Memory":
        if cls._instance is None:
            cls._instance = Mem0MemoryBuilder(collection_name="SETH_CORE_SPACE", env=env).build()
        return cls._instance

    @classmethod
    def reset(cls):
        cls._instance = None


class SethMemoryTool:
    """Multi-layered memory manager integrated with Qdrant Vector Store.

    Scoped per user via the `current_user_id` ContextVar rather than a fixed
    id, so each Telegram user's RAG memory is isolated in mem0 without any
    code path having to thread a user_id argument through every call.
    """
    def __init__(self, env: SethEnvironment):
        self.env = env

    async def retrieve_long_term_memory(self, query: str) -> str:
        """Retrieve and format long-term memory entries asynchronously without blocking."""
        user_id = current_user_id.get()

        def _sync_search():
            mem = Mem0MemorySingleton.get(self.env)
            return mem.search(
                query,
                filters={"user_id": user_id},
                limit=10,
            )

        try:
            raw_retrieval = await asyncio.to_thread(_sync_search)
            results = raw_retrieval if isinstance(raw_retrieval, list) else raw_retrieval.get("results", [])
            records = sorted(results, key=lambda x: x.get("score", 0), reverse=True)
            
            facts = [f"- {r['memory'].strip()}" for r in records if r.get("memory")]
            long_term_context = "\n".join(facts) if facts else "No historical records found for this interlocutor."

        except Exception as e:
            logging.warning(f"Error retrieving memories: {e}")
            long_term_context = "Could not retrieve long-term memory."

        return f"<MEMORY>\n{long_term_context}\n</MEMORY>\n\n"

    @SethToolsManager.tool
    async def save_long_term_memory(
        self,
        user_input: Annotated[str, (
            "The actual core data, fact, or content that needs to be remembered. "
            "CRITICAL RULES: "
            "1) FOCUS ON THE LATEST EXCHANGE: Target ONLY the specific piece of information, text, or data "
            "introduced in the most recent turn. Do NOT merge older history unless explicitly requested. "
            "2) SOURCE FROM THE USER, NOT THE ASSISTANT: Only extract things the user themselves stated as "
            "fact. Never save the assistant's own phrasing, interpretations, or figurative language as if "
            "it were something the user said. "
            "3) EXTRACT THE CONTENT, NOT THE COMMAND: If the user says 'remember that my name is Luis', "
            "extract 'The user's name is Luis'. If they say 'save this config' after sharing data, extract "
            "the actual data. Never include conversational triggers like 'save this', 'remember', or 'record'. "
            "4) WHEN TRIGGERED PROACTIVELY (no explicit save request): phrase the extracted fact plainly and "
            "narrowly — e.g. 'User prefers X' or 'User's project is called Y' — not a paraphrase of the whole "
            "exchange."
        )],
        response: Annotated[str, "The assistant's short confirmation or validation of the fact being stored."],
    ) -> Dict[str, Any]:
        """
        Persists facts, preferences, decisions, or technical constraints into long-term
        memory. Use this proactively — not just on explicit request — whenever the user
        shares information that is likely to matter in future conversations: personal
        details, stated preferences, recurring context about their projects, corrections
        to previous assumptions, or decisions they've made.

        Do NOT save: one-off requests, small talk, information you (the assistant)
        generated or inferred yourself, or anything the user is still deciding/unsure
        about. Only save what the USER stated as fact.

        When in doubt about whether something is worth remembering, save it.
        """
        expiration = (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d")
        user_id = current_user_id.get()

        def _sync_add():
            mem = Mem0MemorySingleton.get(self.env)
            return mem.add(
                [
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": response},
                ],
                user_id=user_id,
                agent_id="SETH",
                metadata={"memory_bucket": "constraints", "expires_on": expiration},
            )

        try:
            res = await asyncio.to_thread(_sync_add)
            return {"status": "ok", "result": res}
        except Exception as e:
            logging.exception(f"Failed to save memory: {e}")
            return {"status": "error", "error": str(e)}


class SethSelfInspectorTool:
    """A polished self-inspection tool for SETH's runtime and source."""
    def __init__(self, max_chars: int = 131072):
        self.main_script_path = os.path.abspath(__file__)
        self.max_chars = int(max_chars)

    def _safe_read(self, path: str, max_chars: int) -> str:
        """Read a file but limit to max_chars and ensure UTF-8 decoding."""
        try:
            size = os.path.getsize(path)
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                if size <= max_chars:
                    return f.read()
                # otherwise read only up to max_chars and indicate truncation
                return f.read(max_chars) + "\n...<<TRUNCATED>>"
        except Exception as e:
            logging.exception("Error reading file safely: %s", e)
            return ""

    def _structural_summary(self, source: str) -> Dict[str, Any]:
        result = {"functions": [], "classes": [], "imports": []}
        try:
            tree = ast.parse(source)
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    result["functions"].append(node.name)
                elif isinstance(node, ast.AsyncFunctionDef):
                    result["functions"].append(node.name + " (async)")
                elif isinstance(node, ast.ClassDef):
                    result["classes"].append(node.name)
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    # reconstruct a compact import representation
                    if isinstance(node, ast.Import):
                        for n in node.names:
                            result["imports"].append(n.name)
                    else:
                        module = node.module or ""
                        for n in node.names:
                            result["imports"].append(f"{module}.{n.name}" if module else n.name)
        except Exception:
            logging.debug("Could not parse AST for structural summary.")
        return result

    @SethToolsManager.tool
    async def inspect_own_source_code(
            self,
            reason: Annotated[str, (
                "A brief, programmatic reason explaining why self-inspection is required "
                "(e.g., 'User requested source code check' or 'Resolving architectural contradiction'). "
                "This ensures stable JSON formatting for local inference engines."
            )] = "No reason provided",
            # NOT Annotated on purpose: these stay internal-only, the LLM never sees or
            # controls them (matches the original manually-written schema, which only
            # ever exposed 'reason').
            include_source: bool = True,
            max_chars: int | None = None
        ) -> str:
            """
            EXECUTION RULES FOR SELF-INSPECTION: Use this tool ALWAYS when the user asks about your
            source code, your internal logic, your Python implementation, or how you are built.

            CRITICAL SYSTEMIC PURPOSES:
            1) SELF-REFERENCE: Read your own script to understand your identity, current class
               definitions, and active handlers.
            2) CONTEXT DEBUGGING & RECONCILIATION: Invoke this tool immediately if you detect
               contradictions between your current behavior and what the user states about your
               architecture. Use it to verify your state logic and fix errors.

            TRIGGER KEYWORDS: 'código fuente', 'tu código', 'source code', 'cómo estás programado', 'ver seth.py'.
            """
            return await asyncio.to_thread(self._inspect_sync, reason, include_source, max_chars)

    def _inspect_sync(self, reason: str, include_source: bool, max_chars: int | None) -> str:
        logging.info(f"🔍 [SETH SELF-INSPECTION TRIGGERED] Reason: '{reason}'")

        path = self.main_script_path
        if max_chars is None:
            max_chars = self.max_chars

        report: Dict[str, Any] = {"path": path, "exists": False, "inspection_reason": reason}
        try:
            if not os.path.exists(path):
                report["error"] = f"Main script not found at {path}"
                return json.dumps(report, ensure_ascii=False)

            stat = os.stat(path)
            report.update({
                "exists": True,
                "size_bytes": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })

            source = self._safe_read(path, int(max_chars)) if include_source else ""
            summary = self._structural_summary(source if source else "")

            report.update({"summary": {"functions": summary["functions"], "classes": summary["classes"], "imports": summary["imports"]}})
            if include_source:
                report["source_preview"] = source

            return json.dumps(report, ensure_ascii=False)
        except Exception as e:
            logging.exception("SethSelfInspectorTool failure: %s", e)
            report["error"] = str(e)
            return json.dumps(report, ensure_ascii=False)


class SethImageGenerationTool:
    """Image generation tool optimized for using the secondary GPU."""
    def __init__(self, env: SethEnvironment):
        self.env = env
        os.makedirs(self.env.storage_images_dir, exist_ok=True)
        gpus = torch.cuda.device_count()
        self.device = torch.device("cuda:1") if gpus > 1 else torch.device("cuda:0")
        logging.info(f"🎨 [IMAGE SYSTEM INIT] device: {self.device}")
        self._pipe = None

        # Cleanup task and lock for managing idle GPU resources
        self._cleanup_task = None
        self._lock = asyncio.Lock()
        self.idle_timeout_secs = 300

    def _init_pipeline(self):
        if self._pipe is None:
            logging.info(f"⏳ Loading Image Pipeline from [{self.env.image_model}] on {self.device}...")
            try:
                pipe = StableDiffusionPipeline.from_single_file(
                    self.env.image_model_full_path,
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    safety_checker=None,
                    requires_safety_checker=False
                )
                pipe = pipe.to(self.device)
                
                # DPM++ 2M Karras
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                    pipe.scheduler.config,
                    use_karras_sigmas=True
                )
                
                pipe.enable_attention_slicing()
                
                self._pipe = pipe
            except Exception as e:
                logging.error(f"❌ CRITICAL error initializing Image Pipeline: {e}")
                raise e

        return self._pipe

    def _sync_generate(self, prompt: str) -> str:
        try:
            pipe = self._init_pipeline()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"gen_{timestamp}_{int(time.time()) % 10000}.png"
            output_path = os.path.join(self.env.storage_images_dir, filename)

            logging.info(f"🚀 Rendering image from: '{prompt}'")
            image = pipe(
                prompt=prompt,
                negative_prompt="bad anatomy, blurry, low quality, deformed, bad hands, mutated, disfigured",
                num_inference_steps=25,
                guidance_scale=7.5,
                width=512,
                height=512
            ).images[0]

            image.save(output_path)
            return output_path
        except Exception as e:
            logging.error(f"❌ Image generation error on GPU {self.device}: {e}")
            raise e

    @SethToolsManager.tool
    async def generate_image(
        self,
        prompt: Annotated[str, (
            "Expanded context-aware English prompt in comma-separated tag format. "
            "Example: '1girl, floating particles, void atmosphere, dark ambient, baroque, masterwork'"
        )],
    ) -> str:
        """
        Use this tool when requested to create, draw, or visualize images.

        ROLE: Creative Art Director. Expand the user request into an English 'Booru-style' tag list.

        RULES:
        1) FORMAT: Strictly short keywords separated by commas (e.g., '1girl, cyberpunk'). NO prose,
           verbs, or filler words.
        2) CONTEXT CROSS-POLLINATION: Intelligently blend ongoing chat themes into the tags.
        3) CREATIVE RANDOMNESS: If the request is vague, hallucinate artistic details (styles,
           lighting, atmospheres) to ensure unique, magnificent results.

        OUTPUT PROTOCOL: The tool returns a JSON. You MUST include the exact filepath format
        'storage/images/filename.png' in your text response.
        """
        logging.info(f"🖼️ Received image generation request with prompt: '{prompt}'")
        async with self._lock:
            if self._cleanup_task and not self._cleanup_task.done():
                self._cleanup_task.cancel()
                logging.info("⏱️ Active VRAM cleanup timer reset due to new request.")

            try:
                file_path = await asyncio.to_thread(self._sync_generate, prompt)
                self._cleanup_task = asyncio.create_task(self._vram_cleanup_timer())
                
                report = {
                    "status": "success",
                    "local_path": file_path,
                    "message": f"Image generated successfully. File saved locally at {file_path}. Inform the user that the image is now available."
                }
                return json.dumps(report, ensure_ascii=False)
            except Exception as e:
                self._cleanup_task = asyncio.create_task(self._vram_cleanup_timer())
                return json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False)

    async def _vram_cleanup_timer(self):
        """Waits in the background without blocking. If it expires, it unloads the model."""
        try:
            await asyncio.sleep(self.idle_timeout_secs)
            async with self._lock:
                if self._pipe is not None:
                    self._pipe = None
                    torch.cuda.empty_cache()
                    logging.info(f"♻️ [VRAM CLEANUP] 5 min of inactivity reached. Unloading Image Pipeline from {self.device}...")
        except asyncio.CancelledError as e:
            logging.info("♻️ [VRAM CLEANUP] Timer cancelled due to new request.")


class SethSpeechGenerationTool:
    """Voice generation tool using kokoro TTS engine for Spanish."""
    def __init__(self, env: "SethEnvironment"):
        self.env = env
        self.storage_audio_dir = env.storage_audio_dir
        os.makedirs(self.storage_audio_dir, exist_ok=True)
        self.lang_code = "e" 
        self._pipeline = None
        self._lock = asyncio.Lock()
        logging.info(f"🔊 [SPEECH SYSTEM INIT] Kokoro initialized for Spanish.")

    def _init_tts(self):
        if self._pipeline is None:
            try:
                self._pipeline = KPipeline(lang_code=self.lang_code)
            except Exception as e:
                logging.error(f"❌ Error initializing KPipeline: {e}")
                raise e
        return self._pipeline

    def _sync_generate_audio(self, text: str, wav_path: str, mp3_path: str):
        pipeline_engine = self._init_tts()
        generator = pipeline_engine(text, voice="ef_dora", speed=1.0)
        
        audio_chunks = []
        sample_rate = 24000
        
        for i, (gs, ps, audio) in enumerate(generator):
            audio_chunks.append(audio)
            
        if not audio_chunks:
            raise ValueError("No audio chunks generated.")

        final_audio = np.concatenate(audio_chunks)
        
        sf.write(wav_path, final_audio, sample_rate)
        AudioSegment.from_wav(wav_path).export(mp3_path, format="mp3")
        if os.path.exists(wav_path):
            os.remove(wav_path)

    @SethToolsManager.tool
    async def generate_speech(
        self,
        text: Annotated[str, (
            "The exact, clean conversational text to be synthesized into audio. "
            "CRITICAL: Must be pure text, sentences, or paragraphs in Spanish. "
            "DO NOT include markdown, code blocks, JSON strings, execution logs, or structural syntax."
        )],
    ) -> str:
        """
        Use this tool ONLY when the user explicitly asks you to speak, read a text aloud,
        or convert a message into a voice note/audio. DO NOT invoke this tool for standard
        text-only responses. The input must be natural, fluid spoken Spanish.
        """
        logging.info(f"🎙️ [KOKORO TTS] Petición de audio recibida para: '{text[:40]}...'")
        async with self._lock:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_filename = f"speech_{timestamp}_{int(time.time()) % 10000}"
                
                wav_path = os.path.join(self.storage_audio_dir, f"{base_filename}.wav")
                mp3_path = os.path.join(self.storage_audio_dir, f"{base_filename}.mp3")
                
                await asyncio.to_thread(self._sync_generate_audio, text, wav_path, mp3_path)
                
                report = {
                    "status": "success",
                    "local_path": mp3_path,
                    "message": f"Audio generated successfully. File saved locally at {mp3_path}. Send this file to the user indicating full path."
                }
                return json.dumps(report, ensure_ascii=False)
                
            except Exception as e:
                logging.error(f"❌ Kokoro ERROR: {e}")
                return json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False)


@dataclass
class SethState:
    """Represents Seth's dynamic inference state"""
    temperature: float
    top_p: float
    presence_penalty: float

    def interpolate(self, env: SethEnvironment, target: 'SethState', alpha: float):
        """Smoothly shifts the semantic state towards a specific target behavior."""
        self.temperature += alpha * (target.temperature - self.temperature)
        self.top_p += alpha * (target.top_p - self.top_p)
        self.presence_penalty += alpha * (target.presence_penalty - self.presence_penalty)

    def save(self, env: SethEnvironment, filename: str | None = None):
        """Schedules the current hyperparameter state to be persisted asynchronously.
        `filename` overrides env.state_path — used for per-user state files."""
        filename = filename or env.state_path
        state_data = self.to_dict()

        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(state_data, f, indent=4, ensure_ascii=False)
            logging.debug(f"💾 [STATE PERSISTED] Metrics dumped to {filename}")
        except Exception as e:
            logging.error(f"❌ Error in background thread saving {filename}: {e}")

    def load(self, env: SethEnvironment, filename: str | None = None) -> bool:
        filename = filename or env.state_path

        if not os.path.exists(filename):
            return False
            
        try:
            with open(filename, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            self.temperature = float(data.get("temperature", self.temperature))
            self.top_p = float(data.get("top_p", self.top_p))
            self.presence_penalty = float(data.get("presence_penalty", self.presence_penalty))

            return True
        except Exception as e:
            logging.error(f"❌ Error loading seth.state (malformed file?). Error: {e}")
            return False
        
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SethStatePresets:
    """Standardized behavioral states for the SETH ecosystem."""

    DEFAULT = SethState(temperature=0.25, top_p=0.85, presence_penalty=0.2)
    
    @classmethod
    def rigorous(cls) -> SethState:
        """High precision, low temperature for code and architecture tasks."""
        return SethState(temperature=0.1, top_p=0.7, presence_penalty=0.0)

    @classmethod
    def chaotic(cls) -> SethState:
        """High entropy for creative glitch-philosophy and humor."""
        return SethState(temperature=1.3, top_p=0.99, presence_penalty=0.9)

    @classmethod
    def verbose(cls) -> SethState:
        """Extended context for long-form essays and deep analysis."""
        return SethState(temperature=0.85, top_p=0.95, presence_penalty=0.4)
    


class SethStateManager:
    """Per-user SethState store, mirroring the SethShortMemory pattern:
    one SethState + one asyncio.Lock per user_id, persisted to its own
    JSON file (seth_state_<user_id>.json) instead of a single shared
    seth.state. Keys only off the (channel-agnostic) user_id string, so
    it works the same for Telegram, WhatsApp, or web sessions."""

    def __init__(self, env: SethEnvironment):
        self.env = env
        self._states: Dict[str, SethState] = {}
        self._locks: Dict[str, asyncio.Lock] = {}

    def _safe_user_fragment(self, user_id: str) -> str:
        return re.sub(r"[^A-Za-z0-9_\-]", "_", str(user_id)) or "unknown"

    def _path_for(self, user_id: str) -> str:
        base, ext = os.path.splitext(self.env.state_path)
        return f"{base}_{self._safe_user_fragment(user_id)}{ext or '.json'}"

    def get_lock(self, user_id: str) -> asyncio.Lock:
        lock = self._locks.get(user_id)
        if lock is None:
            lock = self._locks.setdefault(user_id, asyncio.Lock())
        return lock

    def get_state(self, user_id: str) -> "SethState":
        state = self._states.get(user_id)
        if state is None:
            state = replace(SethStatePresets.DEFAULT)
            state.load(self.env, filename=self._path_for(user_id))
            self._states[user_id] = state
        return state

    def save_state(self, user_id: str, state: "SethState"):
        state.save(self.env, filename=self._path_for(user_id))


class SethDynamicRegulator:
    """Orchestrates semantic state shifts based on real-time query intent.
    
    Optimized to load local weights and cache behavioral target anchors.
    State is now resolved per-user via SethStateManager, so concurrent
    users no longer share (and clobber) the same temperature/top_p.
    """
    def __init__(self, state_manager: SethStateManager, env: SethEnvironment, alpha: float = 0.2):
        self.env = env
        self.alpha = alpha
        self.state_manager = state_manager
        
        #HACK: try to avoid the use AGAIN of SentenceTransformer
        try:
            self.engine = Mem0MemorySingleton.get(self.env).embedding_model.model
        except Exception as e:
            logging.error(f"Failed to load embedding model from Memory singleton (hack is over?): {e}")
            logging.info(f"Loading embedding model using SentenceTransformer: {self.env.embedding_model}")
            self.engine = SentenceTransformer(self.env.embedding_model, device="cuda")

        self.targets = {
            "rigorous": {
                "vector": self.engine.encode("Technical architecture, code precision, logic, systems design", convert_to_tensor=True, normalize_embeddings=True),
                "state": SethStatePresets.rigorous()
            },
            "chaotic": {
                "vector": self.engine.encode("Glitch aesthetics, humor, creative chaos, fertile glitch", convert_to_tensor=True,normalize_embeddings=True),
                "state": SethStatePresets.chaotic()
            },
            "verbose": {
                "vector": self.engine.encode("Deep philosophy, ontological analysis, long essay, legacy", convert_to_tensor=True, normalize_embeddings=True),
                "state": SethStatePresets.verbose()
            }
        }

    def adjust_regulated_config(self, query: Any, user_id: str) -> Dict[str, Any]:
        text_query = ""

        if isinstance(query, list):
            for item in query:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_query += item.get("text", "")
        else:
            text_query = str(query)

        if not text_query.strip():
            text_query = "neutral"

        q_vec = self.engine.encode(text_query, convert_to_tensor=True, normalize_embeddings=True)
        
        best_name, _ = max(
            ((n, torch.dot(q_vec, d["vector"]).item()) for n, d in self.targets.items()), 
            key=lambda x: x[1]
        )

        state = self.state_manager.get_state(user_id)
        state.interpolate(self.env, self.targets[best_name]["state"], self.alpha)
        self.state_manager.save_state(user_id, state)
        
        logging.info(f"🌀 STATE ADJUSTMENT [{best_name.upper()}] user={user_id} - New Temp: {state.temperature:.3f}")
        return state.to_dict()

    async def async_adjust_regulated_config(self, query: Any, user_id: str) -> Dict[str, Any]:
        lock = self.state_manager.get_lock(user_id)
        async with lock:
            return await asyncio.to_thread(self.adjust_regulated_config, query, user_id)


class LocalBgeEmbedder(EmbedderClient):
    def __init__(self, sentence_transformer_model):
        self._model = sentence_transformer_model

    async def create(self, input_data) -> list[float]:
        texts = input_data if isinstance(input_data, list) else [input_data]
        vectors = await asyncio.to_thread(self._model.encode, texts)
        return vectors[0].tolist()
    
    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        vectors = await asyncio.to_thread(self._model.encode, input_data_list)
        return [v.tolist() for v in vectors]


class GraphitiClientSingleton:
    _instance: "Graphiti | None" = None
    _lock = asyncio.Lock()

    @classmethod
    async def get(cls, env: SethEnvironment) -> "Graphiti":
        if cls._instance is not None:
            return cls._instance

        async with cls._lock:
            if cls._instance is None:
                bge_model = Mem0MemorySingleton.get(env).embedding_model.model

                vllm_backend = OpenAIClient(
                    config=LLMConfig(
                        api_key=env.api_key,
                        model=env.llm_model,
                        small_model=env.llm_model,
                        base_url=env.vllm_url,
                    )
                )

                llm_client = await asyncio.to_thread(GLiNER2Client, llm_client=vllm_backend)
 
                graphiti = Graphiti(
                    uri=env.neo4j_uri,
                    user=env.neo4j_user,
                    password=env.neo4j_password,
                    llm_client=llm_client,
                    embedder=LocalBgeEmbedder(bge_model),
                    cross_encoder=BGERerankerClient(),
                )

                logging.info("🕸️ [GRAPHITI] Building indices and constraints in Neo4j...")
                await graphiti.build_indices_and_constraints()
                cls._instance = graphiti

        return cls._instance
    


class SethGraphMemory:
    def __init__(self, env: SethEnvironment):
        self.env = env

    def append(self, user_id: str, user_text: str, response: str):
        async def _do_append():
            try:
                graphiti = await GraphitiClientSingleton.get(self.env)
                await graphiti.add_episode(
                    name=f"seth_turn_{user_id}_{int(time.time())}",
                    episode_body=f"User: {user_text}\nAssistant: {response}",
                    source=EpisodeType.message,
                    source_description="API conversation turn",
                    reference_time=datetime.now(),
                    group_id=user_id
                )
                logging.info(f"🕸️ [GRAPHITI] Episode saved for user={user_id}.")
            except Exception as e:
                logging.warning(f"⚠️ [GRAPHITI] Error saving episode (non-critical): [{e}]")

        asyncio.create_task(_do_append())


class SethGraphQueryTool:
    def __init__(self, env: SethEnvironment):
        self.env = env

    @SethToolsManager.tool
    async def query_relationship_graph(
        self,
        query: Annotated[str, (
            "Natural language question about relationships between people, projects, "
            "or events, or about how something changed over time. "
        )],
    ) -> str:
        """
        Queries the temporal knowledge graph for facts about relationships between
        entities and how they evolved over time. Use it freely to answer questions about connections, introductions, and historical changes.
        """
        user_id = current_user_id.get()
        try:
            graphiti = await GraphitiClientSingleton.get(self.env)
            results = await graphiti.search(query=query, group_ids=[user_id], num_results=16)
            
            if not results:
                return "No relevant relationships found in the graph."

            facts = [f"- {r.fact[:1024]}" for r in results]

            return "\n".join(facts)
        except Exception as e:
            logging.warning(f"⚠️ [GRAPHITI] Error in query_relationship_graph: {e}")
            return "Graph query failed (non-critical). Notify the user that the graph is temporarily unavailable."
        


@dataclass(slots=True)
class _FakeFunctionCall:
    """Mimics the `.function` attribute of an OpenAI SDK tool-call object closely
    enough (`.name`, `.arguments`) that `SethChatBot._run_single_tool_call` can
    execute a reconstructed streaming tool call without any changes."""
    name: str
    arguments: str


@dataclass(slots=True)
class _FakeToolCall:
    """Mimics an OpenAI SDK tool-call object (`.id`, `.function.name`,
    `.function.arguments`) so tool calls reconstructed from streamed deltas can
    be handed straight to the existing `_execute_tool_calls` / `_run_single_tool_call`
    machinery -- no duplicated tool-dispatch logic between the streaming and
    non-streaming code paths."""
    id: str
    function: _FakeFunctionCall


class _StreamAccumulator:
    """Reconstructs one non-streamed-equivalent assistant message out of a
    streamed OpenAI/vLLM completion. Plain `content`/`reasoning` deltas just
    get appended; `tool_calls` deltas arrive fragmented and keyed by `index`
    (the function name and JSON arguments can each be split across many
    chunks), so they get accumulated per-index and only assembled into whole
    tool calls once the stream for that hop finishes.
    """
    def __init__(self):
        self.content = ""
        self.reasoning = ""
        self.finish_reason: str | None = None
        self._tool_calls: dict[int, dict[str, str]] = {}

    def ingest_tool_call_deltas(self, tool_call_deltas) -> None:
        for tc in tool_call_deltas:
            slot = self._tool_calls.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
            if tc.id:
                slot["id"] = tc.id
            fn = getattr(tc, "function", None)
            if fn is not None:
                if fn.name:
                    slot["name"] += fn.name
                if fn.arguments:
                    slot["arguments"] += fn.arguments

    def to_message_dict(self) -> dict:
        """Same flat shape as SethChatBot._serialize_completion_message, so a
        reconstructed streamed message slots into `local_messages` identically
        to a non-streamed one."""
        msg = {"role": "assistant", "content": self.content}
        if self._tool_calls:
            msg["tool_calls"] = [
                {
                    "id": slot["id"] or f"call_{idx}",
                    "type": "function",
                    "function": {"name": slot["name"], "arguments": slot["arguments"]},
                }
                for idx, slot in sorted(self._tool_calls.items())
            ]
        return msg

    def as_fake_tool_calls(self) -> list["_FakeToolCall"]:
        return [
            _FakeToolCall(
                id=slot["id"] or f"call_{idx}",
                function=_FakeFunctionCall(name=slot["name"], arguments=slot["arguments"]),
            )
            for idx, slot in sorted(self._tool_calls.items())
        ]


class SethChatBot:
    """Wraps the OpenAI client and orchestrates async chat calls and execution loop."""
    def __init__(self, client: AsyncOpenAI, env: SethEnvironment, tools_manager: SethToolsManager | None = None, regulator: SethDynamicRegulator | None = None, memory_tool: SethMemoryTool | None = None):
        self.client = client
        self.env = env
        self.tools_manager = tools_manager
        self.memory_tool = memory_tool or SethMemoryTool(env)
        self.regulator = regulator
        self.tokenizer = AutoTokenizer.from_pretrained(self.env.llm_model, trust_remote_code=True)

        os.makedirs(self.env.reasoning_audit_dir, exist_ok=True)
        self._last_audit_cleanup_ts: float = 0.0

    def _persist_reasoning_audit(self, record: dict) -> None:
        """Blocking write of one audit JSON + opportunistic retention sweep.
        Meant to run inside a thread (see _save_reasoning_audit)."""
        user_fragment = re.sub(r"[^A-Za-z0-9_\-]", "_", str(record.get("user_id", "unknown"))) or "unknown"
        ts_fragment = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{ts_fragment}_{user_fragment}_{record['audit_id']}.json"
        filepath = os.path.join(self.env.reasoning_audit_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False, default=str)

        self._maybe_cleanup_reasoning_audits()

    def _maybe_cleanup_reasoning_audits(self) -> None:
        """Deletes audit files older than reasoning_audit_retention_days. Throttled to
        run at most once every 6h per process so the write path stays cheap even with
        thousands of files on disk."""
        now = time.time()
        if now - self._last_audit_cleanup_ts < 6 * 3600:
            return
        self._last_audit_cleanup_ts = now

        retention_seconds = self.env.reasoning_audit_retention_days * 86400
        try:
            removed = 0
            for entry in os.scandir(self.env.reasoning_audit_dir):
                if entry.is_file() and (now - entry.stat().st_mtime) > retention_seconds:
                    os.remove(entry.path)
                    removed += 1
            if removed:
                logging.info(f"🧹 [AUDIT CLEANUP] Pruned {removed} reasoning audit file(s) older than {self.env.reasoning_audit_retention_days}d.")
        except FileNotFoundError:
            pass
        except Exception as e:
            logging.warning(f"⚠️ [AUDIT CLEANUP] Error pruning old reasoning audits (non-critical): [{e}]")

    def _save_reasoning_audit(self, record: dict) -> None:
        """Fire-and-forget persistence so auditing never adds latency to the chat reply path."""
        async def _do_save():
            try:
                await asyncio.to_thread(self._persist_reasoning_audit, record)
            except Exception as e:
                logging.warning(f"⚠️ [AUDIT] Error saving reasoning audit (non-critical): [{e}]")

        asyncio.create_task(_do_save())

    def _dump_messages_for_logging(self, messages: list[dict]) -> str:
        out = []

        for i, msg in enumerate(messages):
            role = msg.get("role", "?")

            out.append(f"\n{'='*70}")
            out.append(f"[{i}] ROLE = {role}")

            if "tool_call_id" in msg:
                out.append(f"tool_call_id = {msg['tool_call_id']}")

            if "tool_calls" in msg:
                out.append("tool_calls =")
                out.append(json.dumps(msg["tool_calls"], indent=2, ensure_ascii=False))

            content = msg.get("content")

            if isinstance(content, list):
                out.append("content =")
                out.append(json.dumps(content, indent=2, ensure_ascii=False))

            else:
                out.append("content =")
                out.append(str(content))

        out.append(f"\n{'='*70}")

        return "\n".join(out)

    async def ask(self, messages: list[dict], use_tools: bool = True, max_tool_hops: int = 5) -> str:
        """
        Runs the chat + tool-calling loop until the model answers with plain
        text or `max_tool_hops` tool-call rounds are exhausted.
        """
        tools = self.tools_manager.as_vllm_format() if (use_tools and self.tools_manager) else None
        local_messages = list(messages)

        pure_text_query = self._extract_text_query(local_messages)
        local_messages = await self._inject_memory_context(local_messages, pure_text_query)
        config = await self._resolve_inference_config(pure_text_query)

        audit = {
            "audit_id": uuid4().hex[:8],
            "timestamp_start": datetime.now().isoformat(),
            "timestamp_end": None,
            "user_id": current_user_id.get(),
            "query": pure_text_query,
            "inference_config": config,
            "hops": [],
            "final_response": None,
            "hop_count": 0,
            "hit_hop_limit": False,
            "error": None,
            "full_context": None,
        }

        try:
            for hop in range(max_tool_hops):
                logging.info(
                    f"🧠 Calling vLLM (tool hop {hop + 1}/{max_tool_hops}).\n%s",
                    self._dump_messages_for_logging(local_messages)
                )

                response = await self._llm_call(local_messages, tools=tools, config=config)
                choice = response.choices[0]
                message = choice.message

                if choice.finish_reason == "length":
                    logging.warning("⚠️ [LLM TRUNCATED] Generation hit max_tokens during reasoning or output.")

                serialized = self._serialize_completion_message(message)
                local_messages.append(serialized)

                reasoning = getattr(message, 'reasoning', None)
                if reasoning:
                    logging.info(f"💭 [GEMMA 4 REASONING hop {hop + 1}]:\n{reasoning.strip()}")

                audit["hop_count"] = hop + 1
                audit["hops"].append({
                    "hop_number": hop + 1,
                    "reasoning": reasoning,
                    "tool_calls_requested": serialized.get("tool_calls"),
                    "tool_results": None,
                })

                if not getattr(message, 'tool_calls', None):
                    audit["final_response"] = message.content or "No content returned by model. (?)"
                    return audit["final_response"]

                logging.info(f"🛠️ [TOOL HOP {hop + 1}] Model requested {len(message.tool_calls)} tool call(s).")
                pre_len = len(local_messages)
                local_messages = await self._execute_tool_calls(message.tool_calls, local_messages)
                audit["hops"][-1]["tool_results"] = local_messages[pre_len:]

            logging.warning(f"⚠️ [TOOL HOP LIMIT] Reached {max_tool_hops} tool hops.")
            audit["hit_hop_limit"] = True
            final_response = await self._llm_call(local_messages, tools=None, config=config)
            audit["final_response"] = final_response.choices[0].message.content or "❌ Error: Tool hop limit reached."
            return audit["final_response"]

        except Exception as e:
            audit["error"] = str(e)
            raise

        finally:
            audit["timestamp_end"] = datetime.now().isoformat()
            audit["full_context"] = local_messages
            self._save_reasoning_audit(audit)

    def _extract_text_query(self, messages: list[dict]) -> str:
        """Extract the plain text from the last message, whether it's a string or a multimodal list."""
        if not messages:
            return ""
        
        content = messages[-1].get("content", "")

        if isinstance(content, list):
            return "".join(
                item.get("text", "") for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            )
        
        return str(content)

    async def _inject_memory_context(self, messages: list[dict], query: str) -> list[dict]:
        """Recovers long-term memory and injects it as a system message before the last user message."""
        
        if not query.strip():
            return messages
        
        related_context = await self.memory_tool.retrieve_long_term_memory(query)

        if "No historical records" not in related_context:
            updated_messages = list(messages)
            updated_messages.insert(len(updated_messages) - 1, {
                "role": "system",
                "content": related_context
            })

            return updated_messages
        
        return messages

    async def _resolve_inference_config(self, query: str) -> dict:
        """Calculate the dynamic inference configuration based on the query."""
        if self.regulator and query:
            return await self.regulator.async_adjust_regulated_config(query, current_user_id.get())
        
        return {}

    async def _llm_call(self, messages: list[dict], tools: list | None, config: dict):
        kwargs = dict(model=self.env.llm_model, messages=messages, **config)
        kwargs.setdefault("max_tokens", 4096)

        if tools:
            kwargs.update(tools=tools, tool_choice="auto")

        if self.env.llm_enable_thinking:
            kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": True},
                "skip_special_tokens": False,
            }

        formatted_chat = self.tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=True, add_generation_prompt=True
        )
        estimated_input_tokens = len(formatted_chat)

        max_allowed_context = self.env.max_tokens
        available_output_slots = max_allowed_context - estimated_input_tokens
        requested_output = kwargs["max_tokens"]

        min_output_tokens = 256
        if available_output_slots < min_output_tokens:
            raise ValueError(f"⚠️ [CONTEXT OVERFLOW] Estimated input tokens ({estimated_input_tokens}) exceed the model's max context ({max_allowed_context}).")

        if requested_output > available_output_slots:
            adjusted_output = max(min_output_tokens, available_output_slots - 50)
            logging.warning(f"⚠️ [CONTEXT OVERFLOW] ({estimated_input_tokens}, {requested_output}, {adjusted_output}).")
            kwargs["max_tokens"] = adjusted_output

        return await self.client.chat.completions.create(**kwargs)

    async def _run_single_tool_call(self, tc) -> dict:
        """Executes one tool call and returns its message dict. Never raises —
        errors are captured into the tool result so one bad call can't sink the batch."""
        name = tc.function.name
        raw_args = tc.function.arguments

        try:
            args = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError as e:
            logging.error(f"🛠️ Tool '{name}' called with malformed JSON args: {raw_args!r} ({e})")
            return {
                "role": "tool",
                "tool_call_id": tc.id,
                "name": name,
                "content": (
                    f"Error: the arguments for '{name}' were not valid JSON ({e}). "
                    f"Re-emit the tool call with well-formed JSON arguments."
                )
            }

        logging.info(f"🛠️ Executing tool: {name} with args: {args}")
        fn = self.tools_manager.get_function(name) if (self.tools_manager and name) else None

        try:
            if fn:
                maybe_coro = fn(**args)
                result = await maybe_coro if asyncio.iscoroutine(maybe_coro) else maybe_coro
            else:
                result = f"Error: Tool '{name}' not found."
        except Exception as exc:
            result = f"Error executing tool: {str(exc)}"

        return {
            "role": "tool",
            "tool_call_id": tc.id,
            "name": name,
            "content": str(result)
        }

    async def _execute_tool_calls(self, tool_calls, messages: list[dict]) -> list[dict]:
        """Runs every tool call the model requested concurrently instead of one-by-one.
        asyncio.gather preserves the order of results to match the order of tool_calls,
        regardless of which one finishes first, so tool_call_id pairing stays correct."""
        tool_messages = await asyncio.gather(
            *(self._run_single_tool_call(tc) for tc in tool_calls)
        )
        messages.extend(tool_messages)
        return messages

    @staticmethod
    def _serialize_completion_message(message: Any) -> dict:
        """ Converts an OpenAI ChatCompletionMessage into a flat dictionary strictly compatible with vLLM chat templates.
        Deliberately does NOT carry the model's `reasoning` field back into the conversation history: Gemma 4's own
        chat template guidance is explicit that thoughts from previous turns must not be re-added to the prompt. """
        content = message.content if message.content is not None else ""
        
        msg_dict = {
            "role": "assistant",
            "content": content
        }

        tool_calls = getattr(message, 'tool_calls', None)
        if tool_calls:
            msg_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                } for tc in tool_calls
            ]
            
        return msg_dict

    # ------------------------------------------------------------------ #
    # Streaming counterpart to ask(). Added for the API edition: the      #
    # Telegram bot never needed token-level streaming (Telegram doesn't   #
    # support editing a message char-by-char at that granularity anyway), #
    # but an HTTP/SSE client benefits a lot from it. Everything else      #
    # (memory injection, regulator, tool dispatch, audit logging) is      #
    # reused unchanged via the private helpers already defined above.     #
    # ------------------------------------------------------------------ #

    async def ask_stream(self, messages: list[dict], use_tools: bool = True, max_tool_hops: int = 5):
        """
        Streaming counterpart to `ask()`. Runs the exact same memory-injection +
        tool-hop loop, but streams each hop from vLLM (stream=True) instead of
        waiting for the full completion, yielding incremental events as they
        arrive. Tool-call arguments are reconstructed by accumulating per-index
        deltas exactly like the OpenAI/vLLM streaming function-calling contract
        (see _StreamAccumulator), then executed through the existing
        `_execute_tool_calls` / `_run_single_tool_call` machinery unchanged --
        so tool dispatch behaves identically whether streamed or not.

        Yields dicts of the shape:
            {"type": "reasoning",  "text": "..."}                         incremental reasoning delta
            {"type": "content",    "text": "..."}                         incremental answer delta
            {"type": "tool_start", "name": "..."}                         a tool call was requested
            {"type": "tool_end",   "name": "...", "ok": bool}             a tool call finished
            {"type": "done", "content": "...", "reasoning": "...",
             "tool_calls_used": [...], "hop_count": n}                    final answer, loop finished
            {"type": "error", "error": "..."}                             loop aborted
        """
        tools = self.tools_manager.as_vllm_format() if (use_tools and self.tools_manager) else None
        local_messages = list(messages)

        pure_text_query = self._extract_text_query(local_messages)
        local_messages = await self._inject_memory_context(local_messages, pure_text_query)
        config = await self._resolve_inference_config(pure_text_query)

        audit = {
            "audit_id": uuid4().hex[:8],
            "timestamp_start": datetime.now().isoformat(),
            "timestamp_end": None,
            "user_id": current_user_id.get(),
            "query": pure_text_query,
            "inference_config": config,
            "hops": [],
            "final_response": None,
            "hop_count": 0,
            "hit_hop_limit": False,
            "error": None,
            "full_context": None,
        }

        tool_calls_used: list[str] = []

        try:
            for hop in range(max_tool_hops):
                logging.info(f"🧠 [STREAM] Calling vLLM (tool hop {hop + 1}/{max_tool_hops}).")

                acc = _StreamAccumulator()
                async for chunk in self._llm_call_stream(local_messages, tools=tools, config=config):
                    if not chunk.choices:
                        continue
                    choice = chunk.choices[0]
                    delta = choice.delta

                    reasoning_piece = getattr(delta, "reasoning", None)
                    if reasoning_piece:
                        acc.reasoning += reasoning_piece
                        yield {"type": "reasoning", "text": reasoning_piece}

                    if delta.content:
                        acc.content += delta.content
                        yield {"type": "content", "text": delta.content}

                    if delta.tool_calls:
                        acc.ingest_tool_call_deltas(delta.tool_calls)

                    if choice.finish_reason:
                        acc.finish_reason = choice.finish_reason

                serialized = acc.to_message_dict()
                local_messages.append(serialized)

                audit["hop_count"] = hop + 1
                audit["hops"].append({
                    "hop_number": hop + 1,
                    "reasoning": acc.reasoning or None,
                    "tool_calls_requested": serialized.get("tool_calls"),
                    "tool_results": None,
                })

                if not serialized.get("tool_calls"):
                    final_content = acc.content or "No content returned by model. (?)"
                    audit["final_response"] = final_content
                    yield {
                        "type": "done",
                        "content": final_content,
                        "reasoning": acc.reasoning,
                        "tool_calls_used": tool_calls_used,
                        "hop_count": audit["hop_count"],
                    }
                    return

                logging.info(f"🛠️ [STREAM TOOL HOP {hop + 1}] Model requested {len(serialized['tool_calls'])} tool call(s).")
                fake_calls = acc.as_fake_tool_calls()
                for fc in fake_calls:
                    tool_calls_used.append(fc.function.name)
                    yield {"type": "tool_start", "name": fc.function.name}

                pre_len = len(local_messages)
                local_messages = await self._execute_tool_calls(fake_calls, local_messages)
                audit["hops"][-1]["tool_results"] = local_messages[pre_len:]

                for tool_msg in local_messages[pre_len:]:
                    ok = not str(tool_msg.get("content", "")).startswith("Error")
                    yield {"type": "tool_end", "name": tool_msg.get("name"), "ok": ok}

            logging.warning(f"⚠️ [STREAM] [TOOL HOP LIMIT] Reached {max_tool_hops} tool hops.")
            audit["hit_hop_limit"] = True
            final_response = await self._llm_call(local_messages, tools=None, config=config)
            final_content = final_response.choices[0].message.content or "❌ Error: Tool hop limit reached."
            audit["final_response"] = final_content
            yield {
                "type": "done",
                "content": final_content,
                "reasoning": None,
                "tool_calls_used": tool_calls_used,
                "hop_count": audit["hop_count"],
            }

        except Exception as e:
            audit["error"] = str(e)
            logging.exception(f"❌ [STREAM] ask_stream failed: {e}")
            yield {"type": "error", "error": str(e)}

        finally:
            audit["timestamp_end"] = datetime.now().isoformat()
            audit["full_context"] = local_messages
            self._save_reasoning_audit(audit)

    async def _llm_call_stream(self, messages: list[dict], tools: list | None, config: dict):
        """Same request-shaping as `_llm_call` (context-window guardrail included),
        but opens a streaming completion and yields raw chunks instead of awaiting
        a single response object."""
        kwargs = dict(model=self.env.llm_model, messages=messages, **config)
        kwargs.setdefault("max_tokens", 4096)

        if tools:
            kwargs.update(tools=tools, tool_choice="auto")

        if self.env.llm_enable_thinking:
            kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": True},
                "skip_special_tokens": False,
            }

        formatted_chat = self.tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=True, add_generation_prompt=True
        )
        estimated_input_tokens = len(formatted_chat)

        max_allowed_context = self.env.max_tokens
        available_output_slots = max_allowed_context - estimated_input_tokens
        requested_output = kwargs["max_tokens"]

        min_output_tokens = 256
        if available_output_slots < min_output_tokens:
            raise ValueError(f"⚠️ [CONTEXT OVERFLOW] Estimated input tokens ({estimated_input_tokens}) exceed the model's max context ({max_allowed_context}).")

        if requested_output > available_output_slots:
            adjusted_output = max(min_output_tokens, available_output_slots - 50)
            logging.warning(f"⚠️ [CONTEXT OVERFLOW] ({estimated_input_tokens}, {requested_output}, {adjusted_output}).")
            kwargs["max_tokens"] = adjusted_output

        stream = await self.client.chat.completions.create(stream=True, **kwargs)
        async for chunk in stream:
            yield chunk


class _UserSession:
    """Per-user conversational state: sliding history queue + its own file lock."""
    __slots__ = ("history", "file_lock", "loaded")

    def __init__(self, max_history: int):
        self.history: deque = deque(maxlen=max_history * 2)
        self.file_lock = asyncio.Lock()
        self.loaded = False


class SethShortMemory:
    """Manages short-term conversational context, isolated per user.

    Each user gets their own in-memory sliding queue and their own on-disk
    JSONL log (logs/history_<user_id>.jsonl), so concurrent conversations
    never see each other's turns. A dict of per-user asyncio.Lock guards each
    user's own file; a separate meta-lock only protects the brief moment of
    creating a new user's session entry, so unrelated users are never
    serialized against each other.
    """
    def __init__(self, env: SethEnvironment, max_history: int = 10):
        self.env = env
        self._max_history = max_history
        self._sessions: dict[str, _UserSession] = {}
        self._sessions_meta_lock = asyncio.Lock()
        os.makedirs(os.path.dirname(env.conversations_path) or ".", exist_ok=True)

    def system_prompt(self) -> str:
        if os.path.exists(self.env.system_prompt_path):
            try:
                with open(self.env.system_prompt_path, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception as e:
                logging.error(f"❌ Error reading system prompt: {e}")
        return "\n***REQUEST THE USER TO PROVIDE A VALID SYSTEM PROMPT!***\n"

    def _safe_user_fragment(self, user_id: str) -> str:
        """Sanitizes a user id for safe use as a filename fragment."""
        return re.sub(r"[^A-Za-z0-9_\-]", "_", str(user_id)) or "unknown"

    def _log_path_for(self, user_id: str) -> str:
        base_dir = os.path.dirname(self.env.conversations_path) or "."
        return os.path.join(base_dir, f"history_{self._safe_user_fragment(user_id)}.jsonl")

    async def _get_session(self, user_id: str) -> _UserSession:
        session = self._sessions.get(user_id)
        if session is not None:
            return session

        async with self._sessions_meta_lock:
            # Re-check: another task may have created it while we awaited the lock.
            session = self._sessions.get(user_id)
            if session is None:
                session = _UserSession(self._max_history)
                self._sessions[user_id] = session

        if not session.loaded:
            await asyncio.to_thread(self._load_user_history, user_id, session)
            session.loaded = True

        return session

    async def get_history_messages(self, user_id: str) -> list[dict]:
        session = await self._get_session(user_id)
        return list(session.history)

    def append(self, user_id: str, text: str, response: str):
        """Appends to that user's in-memory sliding queue and schedules an async disk write.

        Fire-and-forget by design (same as before), but now targets the
        specific user's session/file instead of a single shared one.
        """
        async def _do_append():
            session = await self._get_session(user_id)
            session.history.append({"role": "user", "content": text})
            session.history.append({"role": "assistant", "content": response})
            await self._write(user_id, session, text, response)

        asyncio.create_task(_do_append())

    def _load_user_history(self, user_id: str, session: "_UserSession"):
        log_path = self._log_path_for(user_id)
        if not os.path.exists(log_path):
            return

        try:
            logging.info(f"⏳ Loading history for user={user_id} from {log_path}...")
            temp_turns = []

            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            target_lines = lines[-self._max_history:]

            for line in target_lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    turns = data.get("turns", [])
                    if len(turns) == 2:
                        temp_turns.append(turns[0])  # User
                        temp_turns.append(turns[1])  # Assistant
                except json.JSONDecodeError:
                    continue

            for msg in temp_turns:
                session.history.append(msg)

            logging.info(f"🔄 [MEMORY RESTORED] user={user_id} {len(session.history) // 2} previous interactions restored.")

        except Exception as e:
            logging.error(f"❌ Failed to load persistent history for user={user_id}: {e}")

    async def _write(self, user_id: str, session: "_UserSession", text: str, response: str):
        log_path = self._log_path_for(user_id)

        def _sync_write():
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "turns": [
                    {"role": "user", "content": text},
                    {"role": "assistant", "content": response}
                ]
            }
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        async with session.file_lock:
            try:
                await asyncio.to_thread(_sync_write)
            except Exception as e:
                logging.error(f"❌ Error writing transaction to JSONL for user={user_id}: {e}")


class SethSecurityBoss:
    """Maneja la lista blanca de usuarios autorizados de forma dinámica en disco.

    API edition: identities are opaque session ids (uuid4 hex strings) minted by
    /api/register, not Telegram numeric user ids -- everything else about this
    class (the on-disk allow-list, the registration-token gate, the admin-is-
    first-entry convention) is the same pattern as seth_poc.py's version.
    """
    def __init__(self, env: "SethEnvironment"):
        self.env = env
        self.filepath = str(PROJECT_ROOT / "storage" / "allowed_api_users.json")
        self._ensure_storage_exists()
        self.allowed_users = self._load_users()
        # first user in the list is considered the admin, if any
        # TODO: use in the future to allow admin-only commands, like resetting allowed_api_users.json
        self.admin = next(iter(self.allowed_users), None)

    def _ensure_storage_exists(self):
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        if not os.path.exists(self.filepath):
            try:
                with open(self.filepath, "w", encoding="utf-8") as f:
                    json.dump({"users": []}, f, indent=4)
                logging.info(f"📁 [SECURITY] Created clean database file at {self.filepath}")
            except Exception as e:
                logging.error(f"❌ Error creating allowed_api_users.json: {e}")

    def _load_users(self) -> set:
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r") as f:
                    data = json.load(f)
                    env_users = [uid.strip() for uid in os.getenv("ALLOWED_API_USER_IDS", "").split(",") if uid.strip()]
                    return set(data.get("users", [])) | set(env_users)
            except Exception as e:
                logging.error(f"❌ Error reading allowed_api_users.json: {e}")

        # We use by default the IDs from the .env if no file exists
        env_users = [uid.strip() for uid in os.getenv("ALLOWED_API_USER_IDS", "").split(",") if uid.strip()]
        return set(env_users)

    def is_allowed(self, user_id: str) -> bool:
        return user_id in self.allowed_users

    def register_user(self, user_id: str, input_token: str) -> bool:
        if input_token.strip() == self.env.api_registration_token:
            self.allowed_users.add(user_id)
            try:
                with open(self.filepath, "w") as f:
                    json.dump({"users": list(self.allowed_users)}, f, indent=4)
                logging.info(f"🔒 [SECURITY] New session registered dynamically: {user_id}")
                return True
            except Exception as e:
                logging.error(f"❌ Error saving new user to JSON: {e}")
        return False


class RegisterRequest(BaseModel):
    """Body for POST /api/register -- the HTTP counterpart to a Telegram user
    DM-ing the bot with the raw registration token."""
    token: str


class _PrivateNetworkAccessMiddleware(BaseHTTPMiddleware):
    """Chromium browsers (Chrome/Edge/Brave/...) increasingly gate any request
    from a page's origin to a loopback/private address (this API, on
    127.0.0.1) behind an extra check beyond normal CORS -- first as
    "Private Network Access" (a CORS preflight requiring the response to
    include Access-Control-Allow-Private-Network: true) and, from Chrome 142
    (Oct 2025) onward, as "Local Network Access", a native permission prompt
    the browser shows the user instead.

    Starlette's CORSMiddleware doesn't add that header on its own, so this
    echoes it back whenever the browser's preflight asks for it -- this is
    enough to satisfy the older PNA check some Chromium builds still use.
    It does NOT bypass the newer LNA permission prompt (that one is a
    browser-side gate only the user can grant) -- see the note in
    the_oracle.html / the project docs about serving the page over
    http://localhost instead of opening it as a file:// URL, which is what
    actually determines whether Chrome shows that prompt correctly."""
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        if request.headers.get("access-control-request-private-network") == "true":
            response.headers["Access-Control-Allow-Private-Network"] = "true"
        return response


class _SuppressNoisyAccessLogFilter(logging.Filter):
    """uvicorn.access logs every single request unconditionally, and
    the_oracle.html's telemetry strip polls GET /api/status every few
    seconds -- left alone, that drowns out everything actually worth seeing
    (POST /api/chat, POST /api/register, errors). Drops just the configured
    noisy paths; anything else still comes through normally."""
    NOISY_SUBSTRINGS = ("GET /api/status",)

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(noisy in message for noisy in self.NOISY_SUBSTRINGS)


class SethAPIBot(SethChatBot):
    """Bridges SethChatBot's tool-calling brain to an HTTP/SSE surface via
    FastAPI. This is the direct replacement for SethTelegramBot: same
    responsibilities (per-user short/graph memory, a security gate, media
    in/out, a background keep-alive-ish signal to the client) but speaking
    plain HTTP + multipart + Server-Sent-Events instead of python-telegram-bot
    Updates, so any JavaScript client -- like the_oracle.html -- can drive it.
    """
    def __init__(self, client: AsyncOpenAI, env: SethEnvironment,
                 tools_manager: SethToolsManager,
                 regulator: SethDynamicRegulator,
                 memory_tool: SethMemoryTool,
                 whisper_client: AsyncOpenAI,
                 security_manager: SethSecurityBoss):
        super().__init__(client, env, tools_manager, regulator, memory_tool)
        self.short_memory = SethShortMemory(self.env)
        self.graph_memory = SethGraphMemory(self.env)
        self.system_prompt = self.short_memory.system_prompt()
        self.whisper_client = whisper_client
        self.security_manager = security_manager
        self.app = self._build_app()

    # ------------------------------------------------------------- app --

    def _build_app(self) -> FastAPI:
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Equivalent of SethTelegramBot's ApplicationBuilder().post_init()
            # warm-up hook: build Graphiti's indices/constraints once, before
            # the first request, instead of on the critical path of a reply.
            await GraphitiClientSingleton.get(self.env)
            yield

        app = FastAPI(title="SETH-IN-A-BOX API", version="1.0.0", lifespan=lifespan)

        # Wide open by default for local dev (see SethEnvironment.cors_allowed_origins).
        # credentials are deliberately OFF: identity travels via the X-Seth-User
        # header, not cookies, so there's no reason to widen the CORS surface
        # for credentialed requests.
        origins = [o.strip() for o in self.env.cors_allowed_origins.split(",") if o.strip()]
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        # Added AFTER CORSMiddleware on purpose: Starlette wraps middleware in
        # reverse registration order, so this one ends up outermost and gets
        # a chance to stamp the extra header onto whatever CORSMiddleware
        # already produced (including its own preflight response).
        app.add_middleware(_PrivateNetworkAccessMiddleware)

        os.makedirs(self.env.storage_images_dir, exist_ok=True)
        os.makedirs(self.env.storage_audio_dir, exist_ok=True)
        app.mount("/storage/images", StaticFiles(directory=self.env.storage_images_dir), name="images")
        app.mount("/storage/audio", StaticFiles(directory=self.env.storage_audio_dir), name="audio")

        app.add_api_route("/api/register", self.register_endpoint, methods=["POST"])
        app.add_api_route("/api/chat", self.chat_endpoint, methods=["POST"])
        app.add_api_route("/api/status", self.status_endpoint, methods=["GET"])
        app.add_exception_handler(Exception, self._unhandled_exception_handler)

        return app

    async def _unhandled_exception_handler(self, request: Request, exc: Exception):
        logging.exception(f"❌ Unhandled API exception on {request.url.path}: {exc}")
        return JSONResponse(status_code=500, content={"status": "error", "error": str(exc)})

    # ------------------------------------------------------- security --

    def _require_user(self, x_seth_user: str | None) -> str:
        """Mirrors AuthorizedUserFilter from the Telegram bot: every route below
        requires a previously-registered session id.

        NOTE: this is a plain helper, not a FastAPI `Depends(...)` callable --
        binding an instance method (`self.something`) as a route's *default
        parameter value* doesn't work, because default expressions in a method
        signature are evaluated once at class-definition time, before any
        `self` exists. Calling it explicitly from inside each route body sidesteps
        that entirely and keeps the security gate just as centralized."""
        if not x_seth_user or not self.security_manager.is_allowed(x_seth_user):
            logging.warning(f"🚨 [UNAUTHORIZED ACCESS] user={x_seth_user!r}")
            raise HTTPException(status_code=401, detail="⛔ Restricted Access. Register first via POST /api/register.")
        return x_seth_user

    # ---------------------------------------------------------- routes --

    async def register_endpoint(self, payload: RegisterRequest):
        """HTTP counterpart to SethTelegramBot.handle_registration: trades a
        valid REGISTRATION_TOKEN for a fresh session id. The client is expected
        to persist this (e.g. localStorage) and send it back as the
        'X-Seth-User' header on every later call."""
        new_user_id = uuid4().hex
        if self.security_manager.register_user(new_user_id, payload.token):
            return {
                "status": "ok",
                "user_id": new_user_id,
                "message": (
                    "✅ Welkom! Soy SETH. Guardá este user_id y mandalo como header "
                    "'X-Seth-User' en cada request a /api/chat."
                ),
            }
        raise HTTPException(status_code=403, detail="❌ Invalid registration token.")

    async def status_endpoint(self):
        """Lightweight reachability + telemetry snapshot -- real numbers for the
        kind of VRAM/QDRANT/GRAPHITI/vLLM strip the_oracle.html currently fakes
        with a setInterval jitter."""
        return await self._collect_status()

    async def chat_endpoint(
        self,
        message: str = Form(default=""),
        image: UploadFile | None = File(default=None),
        audio: UploadFile | None = File(default=None),
        x_seth_user: str | None = Header(default=None, alias="X-Seth-User"),
    ):
        """Single multipart endpoint for text, image, and voice input alike --
        mirrors how SethTelegramBot.process() fans text/photo/voice Updates into
        one _process_for_user() call. Always responds as SSE, matching the
        streaming shape the_oracle.html already parses (OpenAI-style
        {choices:[{delta:{content|reasoning}, finish_reason}]} chunks), plus a
        seth_meta/seth_event side-channel for generated media and tool-call
        visibility that a slightly-upgraded frontend can pick up."""
        user_id = self._require_user(x_seth_user)
        return StreamingResponse(
            self._stream_chat_response(user_id, message, image, audio),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async def _stream_chat_response(self, user_id: str, message: str, image: UploadFile | None, audio: UploadFile | None):
        # Set at the top of THIS generator (not in chat_endpoint) on purpose:
        # StreamingResponse iterates this generator lazily, after chat_endpoint
        # has already returned, so setting/resetting the ContextVar around the
        # chat_endpoint call would be a no-op for everything that actually runs
        # inside the generator. Every `await` below happens as part of this same
        # generator's steps, which are all driven by this request's single Task,
        # so the set() here stays correctly scoped to this one user/request
        # (see the current_user_id comment near its declaration).
        token = current_user_id.set(user_id)
        try:
            user_text = message or ""
            base64_image = None

            if audio is not None:
                transcribed_text = await self._handle_uploaded_audio(audio)
                if not transcribed_text:
                    yield self._sse_error("🔇 Cannot understand the audio.")
                    yield "data: [DONE]\n\n"
                    return
                user_text = transcribed_text

            elif image is not None:
                base64_image, user_text = await self._handle_uploaded_image(image, user_text)

            if not user_text and not base64_image:
                yield self._sse_error("⚠️ Empty message.")
                yield "data: [DONE]\n\n"
                return

            if base64_image:
                user_content = [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                ]
            else:
                user_content = user_text

            history = await self.short_memory.get_history_messages(user_id)
            messages = [{"role": "system", "content": self.system_prompt}] + history + [{"role": "user", "content": user_content}]

            final_content = ""

            async for event in self.ask_stream(messages, use_tools=True):
                etype = event["type"]

                if etype == "reasoning":
                    yield self._sse_event({"choices": [{"delta": {"reasoning": event["text"]}, "finish_reason": None}]})

                elif etype == "content":
                    yield self._sse_event({"choices": [{"delta": {"content": event["text"]}, "finish_reason": None}]})

                elif etype in ("tool_start", "tool_end"):
                    yield self._sse_event({"choices": [{"delta": {}, "finish_reason": None}], "seth_event": event})

                elif etype == "error":
                    yield self._sse_event({
                        "choices": [{"delta": {}, "finish_reason": "error"}],
                        "seth_meta": {"error": event["error"]},
                    })

                elif etype == "done":
                    final_content = event["content"]
                    media = self._extract_media_refs(final_content)
                    yield self._sse_event({
                        "choices": [{"delta": {}, "finish_reason": "stop"}],
                        "seth_meta": {
                            "media": media,
                            "tool_calls_used": event["tool_calls_used"],
                            "hop_count": event["hop_count"],
                        },
                    })

            if final_content:
                self.short_memory.append(user_id, user_text, final_content)
                self.graph_memory.append(user_id, user_text, final_content)  # shadow mode

            yield "data: [DONE]\n\n"

        except Exception as e:
            logging.exception(f"❌ Internal API inference error: {e}")
            yield self._sse_error(str(e))
            yield "data: [DONE]\n\n"

        finally:
            current_user_id.reset(token)

    @staticmethod
    def _sse_event(payload: dict) -> str:
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    @classmethod
    def _sse_error(cls, message: str) -> str:
        return cls._sse_event({"choices": [{"delta": {}, "finish_reason": "error"}], "seth_meta": {"error": message}})

    # ----------------------------------------------------- media (in) --

    async def _handle_uploaded_audio(self, audio: UploadFile) -> str | None:
        """Upload-based counterpart to SethTelegramBot._handle_voice_message:
        same save -> (optional split) -> parallel-Whisper-transcribe pipeline,
        just reading from a FastAPI UploadFile instead of a Telegram File."""
        try:
            os.makedirs(self.env.storage_audio_dir, exist_ok=True)
            raw = await audio.read()

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ext = os.path.splitext(audio.filename or "")[1].lstrip(".").lower() or "ogg"
            local_path = os.path.join(self.env.storage_audio_dir, f"audio_{timestamp}_{uuid4().hex[:8]}.{ext}")

            with open(local_path, "wb") as f:
                f.write(raw)
            logging.info(f"🎙️ [AUDIO SAVED] File written to disk: {local_path}")

            if not self.whisper_client:
                raise ValueError("Whisper client is not initialized.")

            chunks = self._split_audio_if_needed(local_path, max_duration_sec=60)

            async def _transcribe_chunk(index: int, chunk_path: str) -> tuple[int, str, str]:
                def _read_audio(path):
                    with open(path, "rb") as f:
                        return f.read()

                audio_bytes = await asyncio.get_running_loop().run_in_executor(None, _read_audio, chunk_path)
                audio_buffer = io.BytesIO(audio_bytes)
                audio_buffer.name = os.path.basename(chunk_path)

                logging.info(f"⚡ Starting parallel transcription for chunk {index}: {audio_buffer.name}")
                transcription = await self.whisper_client.audio.transcriptions.create(
                    model=self.env.whisper_model,
                    file=audio_buffer
                )
                return index, transcription.text.strip(), chunk_path

            tasks = [_transcribe_chunk(idx, path) for idx, path in enumerate(chunks)]
            results = await asyncio.gather(*tasks)
            results.sort(key=lambda x: x[0])

            transcriptions = []
            for _, text, chunk_path in results:
                if text:
                    transcriptions.append(text)
                if chunk_path != local_path and os.path.exists(chunk_path):
                    try:
                        os.remove(chunk_path)
                    except Exception as e:
                        logging.warning(f"⚠️ Cannot remove temporary chunk {chunk_path}: {e}")

            transcribed_text = " ".join(transcriptions).strip()
            logging.info(f"🎙️ [WHISPER TRANSCRIPTION]: '{transcribed_text}'")

            if not transcribed_text:
                return None

            return f"[Audio: {local_path}] - {transcribed_text}"

        except Exception as e:
            logging.error(f"🎙️ Error processing uploaded audio: {e}")
            return None

    async def _handle_uploaded_image(self, image: UploadFile, user_text: str) -> tuple[str | None, str]:
        """Upload-based counterpart to SethTelegramBot._handle_photo_message:
        save to disk + base64-encode for the multimodal message content."""
        try:
            os.makedirs(self.env.storage_images_dir, exist_ok=True)
            raw = await image.read()

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ext = os.path.splitext(image.filename or "")[1].lstrip(".").lower() or "jpg"
            local_path = os.path.join(self.env.storage_images_dir, f"img_{timestamp}_{uuid4().hex[:8]}.{ext}")

            with open(local_path, "wb") as f:
                f.write(raw)
            logging.info(f"📸 [IMAGE SAVED] {local_path}")

            base64_image = base64.b64encode(raw).decode("utf-8")
            image_tag = f"[Image: {local_path}]"

            if user_text:
                user_text = f"{image_tag} - {user_text}"
            else:
                user_text = f"{image_tag} - Describe the content of the image and its relevance to the conversation."

            return base64_image, user_text

        except Exception as e:
            logging.error(f"📸 Error processing uploaded image: {e}")
            return None, user_text

    def _split_audio_if_needed(self, local_path: str, max_duration_sec: int = 60) -> list[str]:
        try:
            audio = AudioSegment.from_file(local_path)
            duration_sec = len(audio) / 1000.0

            if duration_sec <= max_duration_sec:
                return [local_path]

            logging.info(f"🔪 The audio of {duration_sec:.1f}s exceeds the limit of {max_duration_sec}s. Splitting...")
            chunks = []
            ext = os.path.splitext(local_path)[1][1:]

            for i in range(0, len(audio), max_duration_sec * 1000):
                chunk = audio[i:i + max_duration_sec * 1000]
                chunk_path = f"{local_path}_part{i}.{ext}"
                chunk.export(chunk_path, format=ext)
                chunks.append(chunk_path)

            return chunks
        except Exception as e:
            logging.warning(f"⚠️ Cannot split the audio ({e}).")
            return [local_path]

    # ---------------------------------------------------- media (out) --

    def _extract_media_refs(self, response_text: str) -> list[dict]:
        """Non-destructive counterpart to SethTelegramBot._send_media_if_present:
        Telegram pulled the path out of the text and pushed the file through
        reply_photo/reply_voice instead of showing the text. An HTTP client can
        just render both, so this only *detects* generated-media paths the
        model mentioned and turns them into servable URLs under the /storage
        static mounts, without touching the response text itself."""
        media = []

        for match in re.finditer(r"(?:storage/)?images/([\w\-_]+\.png)", response_text):
            filename = match.group(1)
            fs_path = os.path.join(self.env.storage_images_dir, filename)
            if os.path.exists(fs_path):
                media.append({"type": "image", "path": fs_path, "url": f"/storage/images/{filename}"})

        for match in re.finditer(r"(?:storage/)?audio/([\w\-_]+\.mp3)", response_text):
            filename = match.group(1)
            fs_path = os.path.join(self.env.storage_audio_dir, filename)
            if os.path.exists(fs_path):
                media.append({"type": "audio", "path": fs_path, "url": f"/storage/audio/{filename}"})

        return media

    # -------------------------------------------------------- telemetry --

    async def _collect_status(self) -> dict:
        status: dict[str, Any] = {
            "vllm": "unknown",
            "whisper": "unknown",
            "qdrant": "unknown",
            "neo4j_graphiti": "unknown",
            "vram": None,
        }

        try:
            await asyncio.wait_for(self.client.models.list(), timeout=3.0)
            status["vllm"] = "online"
        except Exception as e:
            status["vllm"] = f"offline ({e})"

        # A GET here (even to a route that 404s) still shows up as a log line
        # in faster-whisper-server's own access log -- and at one probe every
        # few seconds, that adds up to constant noise on a server we don't
        # control the logging config of. A raw TCP connect-and-close never
        # reaches that server's HTTP layer at all, so nothing gets logged
        # there; it just confirms something is listening on that port.
        status["whisper"] = await self._probe_tcp_reachable(self.env.whisper_url)

        # Lightweight HTTP probe instead of pulling in a qdrant-client dependency
        # just for a healthcheck -- mem0 already owns the real client.
        try:
            async with httpx.AsyncClient(timeout=3.0) as hc:
                r = await hc.get(f"http://{self.env.qdrant_host}:{self.env.qdrant_port}/collections")
                status["qdrant"] = "online" if r.status_code == 200 else f"http_{r.status_code}"
        except Exception as e:
            status["qdrant"] = f"offline ({e})"

        # The Graphiti singleton is only populated after a successful
        # build_indices_and_constraints() call during startup warm-up.
        status["neo4j_graphiti"] = "online" if GraphitiClientSingleton._instance is not None else "not_initialized"

        status["vram"] = await self._probe_vram()

        return status

    @staticmethod
    async def _probe_tcp_reachable(url: str, timeout: float = 3.0) -> str:
        """Confirms a service is alive with a bare TCP connect-and-close --
        no HTTP request is ever sent, so (unlike httpx.get) this never shows
        up in the target's own access log. Good enough for "is the process
        up"; we don't need a semantic API check for every dependency, and for
        one like faster-whisper-server whose exact HTTP surface beyond
        /v1/audio/transcriptions isn't guaranteed, this is also more honest
        than probing a route that may or may not exist."""
        parsed = urlsplit(url)
        host = parsed.hostname or "localhost"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        try:
            reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=timeout)
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass  # best-effort cleanup, the reachability check already succeeded
            return "online"
        except Exception as e:
            return f"offline ({e})"

    async def _probe_vram(self) -> list[dict] | None:
        """Shells out to nvidia-smi instead of torch.cuda.mem_get_info(): the
        latter only reports whatever device index THIS process's own CUDA
        context considers "current", which can silently disagree with which
        physical GPU vLLM (a separate process) is actually using once
        CUDA_VISIBLE_DEVICES enters the picture -- exactly the setup here,
        with the RTX 5090 as index 0 and the RTX 3050 as index 1. nvidia-smi
        enumerates every physical GPU directly via NVML, independent of any
        single process's device remapping, so it can't get that wrong."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total",
                "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=3.0)
            gpus = []
            for line in stdout.decode().strip().splitlines():
                idx, name, used_mib, total_mib = (p.strip() for p in line.split(","))
                gpus.append({
                    "index": int(idx),
                    "name": name,
                    "used_gb": round(float(used_mib) / 1024, 2),
                    "total_gb": round(float(total_mib) / 1024, 2),
                })
            if gpus:
                return gpus
        except Exception as e:
            logging.debug(f"nvidia-smi VRAM probe failed, falling back to torch: {e}")

        # Fallback if nvidia-smi isn't on PATH: at least report whatever device
        # this process's own CUDA context sees. Less trustworthy in a multi-GPU
        # setup (see docstring above), but better than nothing.
        try:
            if torch.cuda.is_available():
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                return [{
                    "index": torch.cuda.current_device(),
                    "name": torch.cuda.get_device_name(),
                    "used_gb": round((total_bytes - free_bytes) / (1024 ** 3), 2),
                    "total_gb": round(total_bytes / (1024 ** 3), 2),
                }]
        except Exception as e:
            logging.debug(f"torch VRAM fallback also failed: {e}")

        return None

    # -------------------------------------------------------------- run --

    def run(self):
        logging.info(f"🚀 [SETH API] Starting on http://{self.env.api_host}:{self.env.api_port}")
        logging.getLogger("uvicorn.access").addFilter(_SuppressNoisyAccessLogFilter())
        # log_config=None: keep the coloredlogs setup from SethLoggerInit instead
        # of letting uvicorn install its own logging config over it.
        uvicorn.run(self.app, host=self.env.api_host, port=self.env.api_port, log_config=None)


def main():
    SethLoggerInit()
    env = SethEnvironment()
    env.validate()

    # force init
    Mem0MemorySingleton.get(env)

    memory_tool = SethMemoryTool(env=env)
    search_tool = SethSearchTool()
    inspector_tool = SethSelfInspectorTool()
    image_tool = SethImageGenerationTool(env=env)
    speech_tool = SethSpeechGenerationTool(env=env)
    graph_query_tool = SethGraphQueryTool(env=env)

    tools_manager = SethToolsManager()
    tools_manager.register_instances([search_tool, memory_tool, inspector_tool, image_tool, speech_tool, graph_query_tool])

    vllm_client = AsyncOpenAI(base_url=env.vllm_url, api_key=env.api_key)
    whisper_client = AsyncOpenAI(base_url=env.whisper_url, api_key=env.api_key)
    state_manager = SethStateManager(env=env)
    regulator = SethDynamicRegulator(state_manager=state_manager, env=env)

    security_manager = SethSecurityBoss(env=env)

    api_bot = SethAPIBot(client=vllm_client, env=env, tools_manager=tools_manager, regulator=regulator, memory_tool=memory_tool, whisper_client=whisper_client, security_manager=security_manager)
    api_bot.run()


if __name__ == '__main__':
    main()