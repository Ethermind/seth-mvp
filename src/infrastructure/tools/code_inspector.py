"""
Self-inspection tool for runtime AST parsing and structural codebase exploration.
"""

from __future__ import annotations

import ast
import asyncio
from datetime import datetime
import json
import logging
import os
from typing import Annotated, Any, Dict, Optional

from src.infrastructure.tools.registry import tool

logger = logging.getLogger(__name__)


class AstCodeInspector:
    """Introspects local Python source files and extracts structural summaries."""

    def __init__(self, target_script_path: Optional[str] = None, max_chars: int = 131072) -> None:
        self.target_script_path = target_script_path or os.path.abspath(__file__)
        self.max_chars = max_chars

    @tool
    async def inspect_own_source_code(
        self,
        reason: Annotated[str, (
            "A brief, programmatic reason explaining why self-inspection is required "
            "(e.g., 'User requested source code check' or 'Resolving architectural contradiction')."
        )] = "No reason provided",
        file_path: Optional[str] = None,
        include_source: bool = True,
    ) -> str:
        """
        EXECUTION RULES FOR SELF-INSPECTION: Use this tool ALWAYS when user asks about your
        source code, internal logic, Python implementation, or how you are built.
        """
        return await asyncio.to_thread(self._inspect_sync, reason, file_path, include_source)

    def _inspect_sync(self, reason: str, file_path: Optional[str], include_source: bool) -> str:
        path = file_path or self.target_script_path
        logger.info("🔍 [SELF-INSPECTION] Target: %s | Reason: '%s'", path, reason)

        report: Dict[str, Any] = {"path": path, "exists": False, "reason": reason}
        try:
            if not os.path.exists(path):
                report["error"] = f"Script not found at {path}"
                return json.dumps(report, ensure_ascii=False)

            stat = os.stat(path)
            report.update({
                "exists": True,
                "size_bytes": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })

            source = self._safe_read(path, self.max_chars) if include_source else ""
            summary = self._structural_summary(source)
            report["summary"] = summary
            if include_source:
                report["source_preview"] = source

            return json.dumps(report, ensure_ascii=False)
        except Exception as e:
            logger.exception("Error in AstCodeInspector: %s", e)
            report["error"] = str(e)
            return json.dumps(report, ensure_ascii=False)

    def _safe_read(self, path: str, max_chars: int) -> str:
        try:
            size = os.path.getsize(path)
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                if size <= max_chars:
                    return f.read()
                return f.read(max_chars) + "\n...<<TRUNCATED>>"
        except Exception as e:
            logger.error("Error reading file safely: %s", e)
            return ""

    def _structural_summary(self, source: str) -> Dict[str, Any]:
        result: Dict[str, Any] = {"functions": [], "classes": [], "imports": []}
        if not source:
            return result
        try:
            tree = ast.parse(source)
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    result["functions"].append(node.name)
                elif isinstance(node, ast.AsyncFunctionDef):
                    result["functions"].append(f"{node.name} (async)")
                elif isinstance(node, ast.ClassDef):
                    result["classes"].append(node.name)
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        for n in node.names:
                            result["imports"].append(n.name)
                    else:
                        module = node.module or ""
                        for n in node.names:
                            result["imports"].append(f"{module}.{n.name}" if module else n.name)
        except Exception as e:
            logger.debug("AST parsing notice: %s", e)
        return result
