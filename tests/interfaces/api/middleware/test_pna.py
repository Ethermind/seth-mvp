"""
Unit tests for PrivateNetworkAccessMiddleware (src.interfaces.api.middleware.pna).
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from httpx import ASGITransport, AsyncClient
import pytest

from src.interfaces.api.middleware.pna import PrivateNetworkAccessMiddleware


@pytest.mark.anyio
async def test_pna_middleware_headers():
    app = FastAPI()
    app.add_middleware(PrivateNetworkAccessMiddleware)

    @app.get("/ping")
    async def ping():
        return PlainTextResponse("pong")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Standard GET with PNA header
        resp_get = await client.get("/ping", headers={"Access-Control-Request-Private-Network": "true"})
        assert resp_get.status_code == 200
        assert resp_get.headers.get("Access-Control-Allow-Private-Network") == "true"
