"""
Unit tests for ReasoningAuditLogger (src.infrastructure.logging.audit).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import pytest

from src.config.settings import SethSettings
from src.infrastructure.logging.audit import ReasoningAuditLogger


@pytest.mark.anyio
async def test_audit_logger_write_and_retention(tmp_path: Path):
    settings = SethSettings(project_root=tmp_path, reasoning_audit_retention_days=30)
    logger = ReasoningAuditLogger(settings)

    record = {
        "audit_id": "test_123",
        "user_id": "user_abc",
        "query": "Hello",
        "final_response": "Hi there",
    }
    logger.log_turn(record)
    await asyncio.sleep(0.1)

    audit_files = list(settings.reasoning_audit_dir.glob("*.json"))
    assert len(audit_files) == 1

    with open(audit_files[0], "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["audit_id"] == "test_123"
    assert data["user_id"] == "user_abc"
