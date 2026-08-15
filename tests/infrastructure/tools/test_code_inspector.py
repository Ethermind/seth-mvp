"""
Unit tests for AstCodeInspector (src.infrastructure.tools.code_inspector).
"""

from __future__ import annotations

import json
import pytest

from src.infrastructure.tools.code_inspector import AstCodeInspector


@pytest.mark.anyio
async def test_code_inspector_inspect_own_source_code():
    inspector = AstCodeInspector(target_script_path="src/domain/models.py")

    inspected_json = await inspector.inspect_own_source_code(
        reason="Testing AST inspector",
        file_path="src/domain/models.py",
        include_source=True,
    )
    inspected = json.loads(inspected_json)

    assert inspected["exists"] is True
    assert "summary" in inspected
    assert "Message" in inspected["summary"]["classes"]
    assert "Role" in inspected["summary"]["classes"]
