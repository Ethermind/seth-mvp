"""API routes package."""

from src.interfaces.api.routes.auth import router as auth_router
from src.interfaces.api.routes.chat import router as chat_router
from src.interfaces.api.routes.status import router as status_router

__all__ = ["auth_router", "chat_router", "status_router"]
