"""
FastAPI dependency injectors.
"""

from __future__ import annotations

import logging
from fastapi import Depends, Header, HTTPException, Request
from src.application.orchestrator import ConversationOrchestrator
from src.application.use_cases.authenticate import AuthenticateSessionUseCase
from src.application.use_cases.telemetry import CollectTelemetryUseCase
from src.config.settings import SethSettings, get_settings
from src.domain.protocols import SessionStore

logger = logging.getLogger(__name__)


def get_current_settings() -> SethSettings:
    return get_settings()


def get_session_store(request: Request) -> SessionStore:
    return request.app.state.session_store


def get_auth_use_case(request: Request) -> AuthenticateSessionUseCase:
    return AuthenticateSessionUseCase(request.app.state.session_store)


def get_orchestrator(request: Request) -> ConversationOrchestrator:
    return request.app.state.orchestrator


def get_telemetry_use_case(request: Request) -> CollectTelemetryUseCase:
    return request.app.state.telemetry_use_case


def get_whisper_client(request: Request):
    return request.app.state.whisper_client


def require_authorized_user(
    x_seth_user: str | None = Header(default=None, alias="X-Seth-User"),
    session_store: SessionStore = Depends(get_session_store),
) -> str:
    """Verifies that the caller has provided an authorized session ID."""
    if not x_seth_user or not session_store.is_allowed(x_seth_user):
        logger.warning("🚨 [UNAUTHORIZED ACCESS] Attempt with user=%r", x_seth_user)
        raise HTTPException(
            status_code=401,
            detail="⛔ Restricted Access. Register first via POST /api/register.",
        )
    return x_seth_user
