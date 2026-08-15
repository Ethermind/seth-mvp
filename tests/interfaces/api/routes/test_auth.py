"""
Unit tests for Auth route POST /api/register (src.interfaces.api.routes.auth).
"""

from __future__ import annotations

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
import pytest

from src.application.use_cases.authenticate import AuthenticateSessionUseCase
from src.interfaces.api.dependencies import get_auth_use_case
from src.interfaces.api.routes.auth import router as auth_router
from tests.conftest import MockSessionStore


@pytest.mark.anyio
async def test_auth_route_success_and_failure():
    app = FastAPI()
    app.include_router(auth_router)

    store = MockSessionStore(valid_token="secret_key")
    use_case = AuthenticateSessionUseCase(session_store=store)
    app.dependency_overrides[get_auth_use_case] = lambda: use_case

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Success
        resp_ok = await client.post("/api/register", json={"token": "secret_key"})
        assert resp_ok.status_code == 200
        data_ok = resp_ok.json()
        assert data_ok["status"] == "ok"
        assert data_ok["user_id"] is not None

        # Failure returns 403 Forbidden
        resp_err = await client.post("/api/register", json={"token": "bad_token"})
        assert resp_err.status_code == 403
        assert "Invalid registration token" in resp_err.json()["detail"]
