"""
Authentication endpoint (POST /api/register).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from src.application.dto import RegisterRequestDTO
from src.application.use_cases.authenticate import AuthenticateSessionUseCase
from src.interfaces.api.dependencies import get_auth_use_case

router = APIRouter(prefix="/api", tags=["Authentication"])


class RegisterRequestBody(BaseModel):
    token: str


@router.post("/register")
async def register_endpoint(
    payload: RegisterRequestBody,
    auth_use_case: AuthenticateSessionUseCase = Depends(get_auth_use_case),
):
    """Exchanges a registration token for a session user_id."""
    result = auth_use_case.execute(RegisterRequestDTO(token=payload.token))
    if result.status == "ok":
        return {
            "status": "ok",
            "user_id": result.user_id,
            "message": result.message,
        }
    raise HTTPException(status_code=403, detail="❌ Invalid registration token.")
