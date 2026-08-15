"""
Data Transfer Objects (DTOs) for application layer boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class ChatRequestDTO:
    user_id: str
    message: str = ""
    image_bytes: Optional[bytes] = None
    image_filename: str = "image.jpg"
    audio_bytes: Optional[bytes] = None
    audio_filename: str = "audio.ogg"


@dataclass(slots=True)
class RegisterRequestDTO:
    token: str


@dataclass(slots=True)
class RegisterResponseDTO:
    status: str
    user_id: Optional[str] = None
    message: str = ""


@dataclass(slots=True)
class TelemetryResponseDTO:
    vllm: str
    whisper: str
    qdrant: str
    neo4j_graphiti: str
    vram: Optional[List[Dict[str, Any]]] = None
