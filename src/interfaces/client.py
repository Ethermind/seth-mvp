"""
Unified Asynchronous HTTP Client SDK for SETH API.
Used by Telegram bot, TUI, and external Python clients.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, Dict, Optional

import httpx

logger = logging.getLogger(__name__)


class SethClient:
    """Async HTTP Client for interacting with the SETH FastAPI backend."""

    def __init__(self, base_url: str = "http://127.0.0.1:8080", timeout_secs: float = 300.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._timeout = httpx.Timeout(connect=15.0, read=timeout_secs, write=120.0, pool=10.0)

    async def register(self, token: str) -> Optional[str]:
        """Exchanges a registration token for an authorized session user_id."""
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(f"{self.base_url}/api/register", json={"token": token})
                if resp.status_code == 200:
                    return resp.json().get("user_id")
                logger.warning("Registration rejected: HTTP %d", resp.status_code)
                return None
        except Exception as e:
            logger.error("Could not reach SETH API for registration: %s", e)
            return None

    async def chat_stream(
        self,
        user_id: str,
        message: str = "",
        image_bytes: Optional[bytes] = None,
        image_filename: str = "image.jpg",
        image_content_type: str = "image/jpeg",
        audio_bytes: Optional[bytes] = None,
        audio_filename: str = "audio.ogg",
        audio_content_type: str = "audio/ogg",
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Sends multipart chat request and yields parsed SSE payload events:
        - {"type": "reasoning"|"content", "text": "..."}
        - {"type": "tool_start"|"tool_end", "name": "...", "ok": bool}
        - {"type": "done", "media": [...], "tool_calls_used": [...]}
        - {"type": "error", "error": "..."}
        """
        files = {}
        if image_bytes is not None:
            files["image"] = (image_filename, image_bytes, image_content_type)
        if audio_bytes is not None:
            files["audio"] = (audio_filename, audio_bytes, audio_content_type)

        headers = {"X-Seth-User": user_id}

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                data={"message": message},
                files=files or None,
                headers=headers,
            ) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    raise RuntimeError(f"HTTP {resp.status_code} from SETH API: {body.decode(errors='replace')[:300]}")

                buffer = ""
                async for chunk in resp.aiter_text():
                    buffer += chunk
                    lines = buffer.split("\n")
                    buffer = lines.pop()

                    for line in lines:
                        line = line.strip()
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            continue
                        try:
                            payload = json.loads(data)
                        except json.JSONDecodeError:
                            continue

                        if payload.get("seth_event"):
                            yield payload["seth_event"]
                            continue

                        choice = (payload.get("choices") or [{}])[0]
                        delta = choice.get("delta") or {}
                        finish_reason = choice.get("finish_reason")

                        if delta.get("reasoning"):
                            yield {"type": "reasoning", "text": delta["reasoning"]}
                        if delta.get("content"):
                            yield {"type": "content", "text": delta["content"]}

                        if finish_reason == "error":
                            meta = payload.get("seth_meta") or {}
                            yield {"type": "error", "error": meta.get("error", "Unknown SETH API error.")}
                        elif finish_reason == "stop":
                            meta = payload.get("seth_meta") or {}
                            yield {
                                "type": "done",
                                "media": meta.get("media", []),
                                "tool_calls_used": meta.get("tool_calls_used", []),
                            }

    async def get_status(self) -> Dict[str, Any]:
        """Queries telemetry status from the backend."""
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(f"{self.base_url}/api/status")
            resp.raise_for_status()
            return resp.json()

    async def download_media(self, relative_url: str) -> bytes:
        """Fetches raw bytes of a generated image or audio file."""
        url = f"{self.base_url}{relative_url}" if relative_url.startswith("/") else f"{self.base_url}/{relative_url}"
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.content
