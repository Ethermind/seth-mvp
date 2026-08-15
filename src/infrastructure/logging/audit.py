"""
Reasoning trace and multi-hop audit persistence with automated retention cleanup.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import re
import time
from typing import Any, Dict

from src.config.settings import SethSettings, get_settings

logger = logging.getLogger(__name__)


class ReasoningAuditLogger:
    """Persists full reasoning traces and tool invocations per conversation turn."""

    def __init__(self, settings: SethSettings | None = None) -> None:
        self.settings = settings or get_settings()
        self.audit_dir = self.settings.reasoning_audit_dir
        os.makedirs(self.audit_dir, exist_ok=True)
        self.retention_days = self.settings.reasoning_audit_retention_days
        self._last_cleanup_ts: float = 0.0

    def log_turn(self, record: Dict[str, Any]) -> None:
        """Schedules fire-and-forget asynchronous audit persistence."""
        asyncio.create_task(self._async_persist(record))

    async def _async_persist(self, record: Dict[str, Any]) -> None:
        try:
            await asyncio.to_thread(self._persist_sync, record)
        except Exception as e:
            logger.warning("⚠️ [AUDIT] Error persisting reasoning audit (non-critical): %s", e)

    def _persist_sync(self, record: Dict[str, Any]) -> None:
        user_fragment = re.sub(r"[^A-Za-z0-9_\-]", "_", str(record.get("user_id", "unknown"))) or "unknown"
        ts_fragment = datetime.now().strftime("%Y%m%d_%H%M%S")
        audit_id = record.get("audit_id", "turn")
        filename = f"{ts_fragment}_{user_fragment}_{audit_id}.json"
        filepath = self.audit_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False, default=str)

        self._maybe_cleanup_old_audits()

    def _maybe_cleanup_old_audits(self) -> None:
        now = time.time()
        # Throttled to run at most once every 6h
        if now - self._last_cleanup_ts < 6 * 3600:
            return
        self._last_cleanup_ts = now

        retention_seconds = self.retention_days * 86400
        try:
            removed = 0
            for entry in os.scandir(self.audit_dir):
                if entry.is_file() and (now - entry.stat().st_mtime) > retention_seconds:
                    os.remove(entry.path)
                    removed += 1
            if removed:
                logger.info("🧹 [AUDIT CLEANUP] Pruned %d audit files older than %dd.", removed, self.retention_days)
        except Exception as e:
            logger.warning("⚠️ [AUDIT CLEANUP] Error pruning old audits: %s", e)
