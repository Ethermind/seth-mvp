"""
Centralized, typed configuration for SETH-IN-A-BOX using Pydantic Settings v2.
"""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base anchors
SRC_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SRC_DIR.parent


class SethSettings(BaseSettings):
    """
    Typed and validated settings for the entire SETH ecosystem.
    Values can be overridden by environment variables or .env files.
    """

    model_config = SettingsConfigDict(
        env_file=(str(SRC_DIR / ".env"), str(PROJECT_ROOT / ".env")),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # --- Project & Directory Paths ---
    src_dir: Path = SRC_DIR
    project_root: Path = PROJECT_ROOT

    # --- LLM Ingestion & Inference (vLLM / SGLang / OpenAI-compatible) ---
    llm_model: str = Field(
        default="nvidia/Gemma-4-26B-A4B-NVFP4",
        alias="LLM_MODEL",
    )
    vllm_url: str = Field(
        default="http://localhost:8000/v1",
        alias="VLLM_URL",
    )
    api_key: str = Field(
        default="NONE",
        alias="API_KEY",
    )
    max_tokens: int = Field(
        default=131072,
        alias="MAX_TOKENS",
    )
    llm_enable_thinking: bool = Field(
        default=True,
        alias="LLM_ENABLE_THINKING",
    )

    # --- Audio Transcription (Whisper) ---
    whisper_url: str = Field(
        default="http://localhost:8010/v1",
        alias="WHISPER_URL",
    )
    whisper_model: str = Field(
        default="large-v3",
        alias="WHISPER_MODEL",
    )

    # --- Image Generation (Stable Diffusion) ---
    image_model: str = Field(
        default="dreamshaper_8.safetensors",
        alias="IMAGE_MODEL",
    )

    # --- Embeddings & Vector Memory (Mem0 / Qdrant) ---
    embedding_model: str = Field(
        default="BAAI/bge-large-en-v1.5",
        alias="EMBEDDING_MODEL",
    )
    embedding_dims: int = Field(
        default=1024,
        alias="EMBEDDING_MODEL_DIMS",
    )
    qdrant_host: str = Field(
        default="localhost",
        alias="QDRANT_HOST",
    )
    qdrant_port: int = Field(
        default=6333,
        alias="QDRANT_PORT",
    )
    log_mem0_path: str = Field(
        default="",
        alias="LOG_MEM0_PATH",
    )

    # --- Relational & Temporal Graph (Graphiti / Neo4j) ---
    neo4j_uri: str = Field(
        default="bolt://localhost:7687",
        alias="NEO4J_URI",
    )
    neo4j_user: str = Field(
        default="neo4j",
        alias="NEO4J_USER",
    )
    neo4j_password: str = Field(
        default="",
        alias="NEO4J_PASSWORD",
    )

    # --- Authentication & Session Security ---
    registration_token: str = Field(
        default="",
        alias="REGISTRATION_TOKEN",
    )
    allowed_api_user_ids: str = Field(
        default="",
        alias="ALLOWED_API_USER_IDS",
    )

    # --- API Service Configuration ---
    api_host: str = Field(
        default="127.0.0.1",
        alias="API_HOST",
    )
    api_port: int = Field(
        default=8080,
        alias="API_PORT",
    )
    cors_allowed_origins: str = Field(
        default="*",
        alias="CORS_ALLOWED_ORIGINS",
    )

    # --- Client Connections (Telegram & TUI) ---
    telegram_token: str = Field(
        default="",
        alias="TELEGRAM_TOKEN",
    )
    seth_api_base_url: str = Field(
        default="http://127.0.0.1:8080",
        alias="SETH_API_BASE_URL",
    )

    # --- Auditing & Storage Paths ---
    reasoning_audit_retention_days: int = Field(
        default=30,
        alias="AUDIT_LOG_RETENTION_DAYS",
    )

    @property
    def image_model_full_path(self) -> Path:
        return self.project_root / "models" / self.image_model

    @property
    def system_prompt_path(self) -> Path:
        # Check prompt/seth.md or fallback to src/seth.md
        prompt_in_pkg = self.src_dir / "prompt" / "seth.md"
        if prompt_in_pkg.exists():
            return prompt_in_pkg
        return self.src_dir / "seth.md"

    @property
    def conversations_path(self) -> Path:
        return self.project_root / "conversations"

    @property
    def state_path(self) -> Path:
        return self.project_root / "storage" / "state" / "seth.state"

    @property
    def storage_images_dir(self) -> Path:
        return self.project_root / "storage" / "images"

    @property
    def storage_audio_dir(self) -> Path:
        return self.project_root / "storage" / "audio"

    @property
    def reasoning_audit_dir(self) -> Path:
        return self.project_root / "storage" / "logs" / "reasoning"

    @property
    def allowed_users_file(self) -> Path:
        return self.project_root / "storage" / "allowed_api_users.json"

    @property
    def telegram_sessions_file(self) -> Path:
        return self.project_root / "storage" / "telegram_sessions.json"

    @property
    def parsed_cors_origins(self) -> List[str]:
        return [o.strip() for o in self.cors_allowed_origins.split(",") if o.strip()]

    def validate_for_api(self) -> None:
        """Validates critical environment variables required to run the API backend."""
        if not self.registration_token:
            raise ValueError("❌ REGISTRATION_TOKEN is missing in the environment.")

    def validate_for_telegram(self) -> None:
        """Validates critical environment variables required to run the Telegram bot."""
        if not self.telegram_token:
            raise ValueError("❌ TELEGRAM_TOKEN is missing in the environment.")
        if not self.registration_token:
            raise ValueError("❌ REGISTRATION_TOKEN is missing in the environment.")


@lru_cache(maxsize=1)
def get_settings() -> SethSettings:
    """Returns a cached singleton instance of SethSettings."""
    return SethSettings()
