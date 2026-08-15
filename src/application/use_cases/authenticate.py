"""
Use Case: Authenticate and Register User Sessions.
"""

from __future__ import annotations

import logging
from uuid import uuid4

from src.application.dto import RegisterRequestDTO, RegisterResponseDTO
from src.domain.protocols import SessionStore

logger = logging.getLogger(__name__)


class AuthenticateSessionUseCase:
    """Handles exchanging a registration token for an opaque session user_id."""

    def __init__(self, session_store: SessionStore) -> None:
        self.session_store = session_store

    def execute(self, request: RegisterRequestDTO) -> RegisterResponseDTO:
        new_user_id = uuid4().hex
        if self.session_store.register(new_user_id, request.token):
            logger.info("✅ Successfully authorized and created session: %s", new_user_id)
            return RegisterResponseDTO(
                status="ok",
                user_id=new_user_id,
                message="✅ Welcome! I am SETH. Save this user_id and pass it as the 'X-Seth-User' header on every request.",
            )

        logger.warning("🔒 Rejected invalid registration token attempt.")
        return RegisterResponseDTO(
            status="error",
            user_id=None,
            message="❌ Invalid registration token.",
        )
