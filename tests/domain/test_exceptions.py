"""
Unit tests for domain exceptions (src.domain.exceptions).
"""

from __future__ import annotations

from src.domain.exceptions import (
    ConfigurationError,
    ContextLimitExceededError,
    InferenceEngineError,
    SethError,
    ToolExecutionError,
    UnauthorizedSessionError,
)


def test_exception_inheritance_hierarchy():
    base = SethError("Generic error")
    assert isinstance(base, Exception)

    cfg = ConfigurationError("Missing token")
    assert isinstance(cfg, SethError)

    auth = UnauthorizedSessionError("user_123")
    assert isinstance(auth, SethError)
    assert "user_123" in str(auth)

    ctx = ContextLimitExceededError(estimated_tokens=5000, max_tokens=4096)
    assert isinstance(ctx, SethError)
    assert ctx.estimated_tokens == 5000
    assert ctx.max_tokens == 4096

    tool = ToolExecutionError("web_search", "connection reset")
    assert isinstance(tool, SethError)
    assert tool.tool_name == "web_search"
    assert tool.reason == "connection reset"

    inf = InferenceEngineError("vLLM connection refused")
    assert isinstance(inf, SethError)
