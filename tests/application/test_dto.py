"""
Unit tests for Application DTOs (src.application.dto).
"""

from __future__ import annotations

from src.application.dto import (
    ChatRequestDTO,
    RegisterRequestDTO,
    RegisterResponseDTO,
    TelemetryResponseDTO,
)


def test_chat_request_dto_defaults_and_custom():
    dto = ChatRequestDTO(user_id="u1")
    assert dto.user_id == "u1"
    assert dto.message == ""
    assert dto.image_bytes is None
    assert dto.image_filename == "image.jpg"
    assert dto.audio_bytes is None
    assert dto.audio_filename == "audio.ogg"

    dto_custom = ChatRequestDTO(
        user_id="u2",
        message="Look at this",
        image_bytes=b"fake_img",
        image_filename="pic.png",
        audio_bytes=b"fake_audio",
        audio_filename="voice.ogg",
    )
    assert dto_custom.message == "Look at this"
    assert dto_custom.image_bytes == b"fake_img"


def test_register_dtos():
    req = RegisterRequestDTO(token="secret123")
    assert req.token == "secret123"

    res = RegisterResponseDTO(status="ok", user_id="abc-123", message="Welcome")
    assert res.status == "ok"
    assert res.user_id == "abc-123"
    assert res.message == "Welcome"


def test_telemetry_dto():
    dto = TelemetryResponseDTO(
        vllm="online",
        whisper="offline (connection refused)",
        qdrant="online",
        neo4j_graphiti="online",
        vram=[{"index": 0, "used_gb": 10.0, "total_gb": 24.0}],
    )
    assert dto.vllm == "online"
    assert "offline" in dto.whisper
    assert len(dto.vram) == 1
