"""
Unit tests for AuthenticateSessionUseCase (src.application.use_cases.authenticate).
"""

from __future__ import annotations

from src.application.dto import RegisterRequestDTO
from src.application.use_cases.authenticate import AuthenticateSessionUseCase
from tests.conftest import MockSessionStore


def test_authenticate_success():
    store = MockSessionStore(valid_token="secret_pass_123")
    use_case = AuthenticateSessionUseCase(session_store=store)

    dto = RegisterRequestDTO(token="secret_pass_123")
    res = use_case.execute(dto)

    assert res.status == "ok"
    assert res.user_id is not None
    assert len(res.user_id) == 32
    assert store.is_allowed(res.user_id) is True


def test_authenticate_failure():
    store = MockSessionStore(valid_token="secret_pass_123")
    use_case = AuthenticateSessionUseCase(session_store=store)

    dto = RegisterRequestDTO(token="invalid_pass")
    res = use_case.execute(dto)

    assert res.status == "error"
    assert res.user_id is None
    assert "Invalid registration token" in res.message
