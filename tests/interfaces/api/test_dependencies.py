"""
Unit tests for FastAPI dependencies (src.interfaces.api.dependencies).
"""

from __future__ import annotations

from fastapi import HTTPException
import pytest

from src.interfaces.api.dependencies import require_authorized_user
from tests.conftest import MockSessionStore


def test_require_authorized_user_dependency():
    store = MockSessionStore()
    store.allowed_users.add("authorized_uid")

    # Missing header
    with pytest.raises(HTTPException) as exc1:
        require_authorized_user(x_seth_user=None, session_store=store)
    assert exc1.value.status_code == 401

    # Unauthorized user
    with pytest.raises(HTTPException) as exc2:
        require_authorized_user(x_seth_user="unknown_uid", session_store=store)
    assert exc2.value.status_code == 401

    # Authorized user
    uid = require_authorized_user(x_seth_user="authorized_uid", session_store=store)
    assert uid == "authorized_uid"
