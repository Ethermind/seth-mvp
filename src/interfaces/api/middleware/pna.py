"""
Chromium Private Network Access / Local Network Access middleware.
"""

from __future__ import annotations

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware


class PrivateNetworkAccessMiddleware(BaseHTTPMiddleware):
    """
    Ensures preflight requests from browser pages (e.g. localhost) receive the
    Access-Control-Allow-Private-Network: true header to satisfy Chromium PNA checks.
    """

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        response = await call_next(request)
        if request.headers.get("access-control-request-private-network") == "true":
            response.headers["Access-Control-Allow-Private-Network"] = "true"
        return response
