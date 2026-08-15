"""
Per-user short-term conversation history persisted as JSONL files.
Implements the domain ConversationHistory protocol.
"""

from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import re
from typing import Dict, List

from src.config.settings import SethSettings, get_settings
from src.domain.models import Message, Role

logger = logging.getLogger(__name__)


class _UserSessionState:
    """Per-user memory state: sliding queue and file lock."""
    __slots__ = ("history", "file_lock", "loaded")

    def __init__(self, max_history: int) -> None:
        self.history: deque[Message] = deque(maxlen=max_history * 2)
        self.file_lock = asyncio.Lock()
        self.loaded = False


class JsonlConversationHistory:
    """Manages isolated short-term conversational context per user."""

    def __init__(self, settings: SethSettings | None = None, max_history: int = 10) -> None:
        self.settings = settings or get_settings()
        self._max_history = max_history
        self._sessions: Dict[str, _UserSessionState] = {}
        self._meta_lock = asyncio.Lock()
        os.makedirs(self.settings.conversations_path, exist_ok=True)

    def _safe_user_fragment(self, user_id: str) -> str:
        return re.sub(r"[^A-Za-z0-9_\-]", "_", str(user_id)) or "unknown"

    def _log_path_for(self, user_id: str) -> Path:
        return self.settings.conversations_path / f"history_{self._safe_user_fragment(user_id)}.jsonl"

    async def _get_session(self, user_id: str) -> _UserSessionState:
        session = self._sessions.get(user_id)
        if session is not None:
            return session

        async with self._meta_lock:
            session = self._sessions.get(user_id)
            if session is None:
                session = _UserSessionState(self._max_history)
                self._sessions[user_id] = session

        if not session.loaded:
            await asyncio.to_thread(self._load_user_history, user_id, session)
            session.loaded = True

        return session

    def _load_user_history(self, user_id: str, session: _UserSessionState) -> None:
        log_path = self._log_path_for(user_id)
        if not log_path.exists():
            return

        try:
            temp_messages: List[Message] = []
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
                        temp_messages.append(Message(role=Role(turns[0].get("role", "user")), content=turns[0].get("content", "")))
                        temp_messages.append(Message(role=Role(turns[1].get("role", "assistant")), content=turns[1].get("content", "")))
                except (json.JSONDecodeError, ValueError):
                    continue

            for msg in temp_messages:
                session.history.append(msg)

            logger.info("🔄 [MEMORY RESTORED] user=%s restored %d turns.", user_id, len(session.history) // 2)

        except Exception as e:
            logger.error("❌ Failed to load persistent history for user=%s: %s", user_id, e)

    async def get_history(self, user_id: str) -> List[Message]:
        """Retrieves recent message history for a user."""
        session = await self._get_session(user_id)
        return list(session.history)

    async def append_turn(self, user_id: str, user_text: str, assistant_text: str) -> None:
        """Appends user message and assistant reply to history and writes to disk."""
        session = await self._get_session(user_id)
        session.history.append(Message(role=Role.USER, content=user_text))
        session.history.append(Message(role=Role.ASSISTANT, content=assistant_text))
        await self._write_to_disk(user_id, session, user_text, assistant_text)

    async def _write_to_disk(self, user_id: str, session: _UserSessionState, text: str, response: str) -> None:
        log_path = self._log_path_for(user_id)

        def _sync_write() -> None:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "turns": [
                    {"role": "user", "content": text},
                    {"role": "assistant", "content": response},
                ],
            }
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        async with session.file_lock:
            try:
                await asyncio.to_thread(_sync_write)
            except Exception as e:
                logger.error("❌ Error writing transaction to JSONL for user=%s: %s", user_id, e)
