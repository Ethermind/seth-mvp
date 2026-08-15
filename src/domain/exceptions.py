"""
Domain exceptions hierarchy for SETH-IN-A-BOX.
"""

from __future__ import annotations


class SethError(Exception):
    """Base exception for all domain and runtime errors in SETH."""
    def __init__(self, message: str = "An unexpected error occurred in SETH."):
        super().__init__(message)
        self.message = message


class ConfigurationError(SethError):
    """Raised when environment variables or settings are missing/invalid."""
    pass


class UnauthorizedSessionError(SethError):
    """Raised when a request is made with an unauthorized or missing session ID."""
    def __init__(self, user_id: str | None = None):
        msg = f"⛔ Restricted Access. Unauthorized session ID: {user_id!r}" if user_id else "⛔ Restricted Access."
        super().__init__(msg)
        self.user_id = user_id


class ContextLimitExceededError(SethError):
    """Raised when estimated prompt tokens exceed the model context window limit."""
    def __init__(self, estimated_tokens: int, max_tokens: int):
        super().__init__(
            f"⚠️ [CONTEXT OVERFLOW] Estimated tokens ({estimated_tokens}) exceed model context ({max_tokens})."
        )
        self.estimated_tokens = estimated_tokens
        self.max_tokens = max_tokens


class ToolExecutionError(SethError):
    """Raised when execution of a tool callable fails."""
    def __init__(self, tool_name: str, reason: str):
        super().__init__(f"❌ Error executing tool '{tool_name}': {reason}")
        self.tool_name = tool_name
        self.reason = reason


class InferenceEngineError(SethError):
    """Raised when communication with vLLM / LLM provider fails."""
    pass
