"""
Chat endpoint (POST /api/chat) with multipart multimodal input and real-time SSE streaming.
"""

from __future__ import annotations

import asyncio
import base64
from datetime import datetime
import io
import json
import logging
import os
from typing import AsyncIterator, Optional, Tuple
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import StreamingResponse
from pydub import AudioSegment

from src.application.orchestrator import ConversationOrchestrator
from src.config.settings import SethSettings
from src.domain.models import StreamEventType
from src.interfaces.api.dependencies import (
    get_current_settings,
    get_orchestrator,
    get_whisper_client,
    require_authorized_user,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Chat"])


@router.post("/chat")
async def chat_endpoint(
    message: str = Form(default=""),
    image: Optional[UploadFile] = File(default=None),
    audio: Optional[UploadFile] = File(default=None),
    user_id: str = Depends(require_authorized_user),
    orchestrator: ConversationOrchestrator = Depends(get_orchestrator),
    settings: SethSettings = Depends(get_current_settings),
    whisper_client=Depends(get_whisper_client),
):
    """
    Multimodal conversational endpoint. Handles text, image uploads, and voice notes.
    Streams back token-by-token reasoning and content deltas via Server-Sent Events (SSE).
    """
    return StreamingResponse(
        _stream_chat_response(
            user_id=user_id,
            raw_message=message,
            image_file=image,
            audio_file=audio,
            orchestrator=orchestrator,
            settings=settings,
            whisper_client=whisper_client,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _stream_chat_response(
    user_id: str,
    raw_message: str,
    image_file: Optional[UploadFile],
    audio_file: Optional[UploadFile],
    orchestrator: ConversationOrchestrator,
    settings: SethSettings,
    whisper_client: Any,
) -> AsyncIterator[str]:
    try:
        user_text = raw_message or ""
        base64_image = None

        # 1. Handle audio transcription if present
        if audio_file is not None:
            transcribed = await _process_uploaded_audio(audio_file, settings, whisper_client)
            if not transcribed:
                yield _sse_error("🔇 Cannot understand or transcribe the audio.")
                yield "data: [DONE]\n\n"
                return
            user_text = transcribed

        # 2. Handle image upload if present
        elif image_file is not None:
            base64_image, user_text = await _process_uploaded_image(image_file, user_text, settings)

        if not user_text and not base64_image:
            yield _sse_error("⚠️ Empty message.")
            yield "data: [DONE]\n\n"
            return

        # 3. Stream from ConversationOrchestrator
        async for chunk in orchestrator.execute_stream(user_id=user_id, user_text=user_text, image_b64=base64_image):
            if chunk.event_type == StreamEventType.REASONING:
                yield _sse_event({"choices": [{"delta": {"reasoning": chunk.text}, "finish_reason": None}]})

            elif chunk.event_type == StreamEventType.CONTENT:
                yield _sse_event({"choices": [{"delta": {"content": chunk.text}, "finish_reason": None}]})

            elif chunk.event_type in (StreamEventType.TOOL_START, StreamEventType.TOOL_END):
                seth_event = {
                    "type": chunk.event_type.value,
                    "name": chunk.tool_name or "",
                    "ok": chunk.ok,
                }
                yield _sse_event({"choices": [{"delta": {}, "finish_reason": None}], "seth_event": seth_event})

            elif chunk.event_type == StreamEventType.ERROR:
                yield _sse_event({
                    "choices": [{"delta": {}, "finish_reason": "error"}],
                    "seth_meta": {"error": chunk.text},
                })

            elif chunk.event_type == StreamEventType.DONE:
                yield _sse_event({
                    "choices": [{"delta": {}, "finish_reason": "stop"}],
                    "seth_meta": {
                        "media": chunk.metadata.get("media", []),
                        "tool_calls_used": chunk.metadata.get("tool_calls_used", []),
                        "hop_count": chunk.metadata.get("hop_count", 1),
                    },
                })

        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.exception("❌ Uncaught exception in SSE chat stream: %s", e)
        yield _sse_error(str(e))
        yield "data: [DONE]\n\n"


def _sse_event(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _sse_error(message: str) -> str:
    return _sse_event({"choices": [{"delta": {}, "finish_reason": "error"}], "seth_meta": {"error": message}})


async def _process_uploaded_audio(audio: UploadFile, settings: SethSettings, whisper_client: Any) -> Optional[str]:
    try:
        os.makedirs(settings.storage_audio_dir, exist_ok=True)
        raw = await audio.read()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = os.path.splitext(audio.filename or "")[1].lstrip(".").lower() or "ogg"
        local_path = settings.storage_audio_dir / f"audio_{timestamp}_{uuid4().hex[:8]}.{ext}"

        with open(local_path, "wb") as f:
            f.write(raw)

        if not whisper_client:
            raise ValueError("Whisper client not initialized.")

        chunks = _split_audio_if_needed(str(local_path), max_duration_sec=60)

        async def _transcribe_chunk(index: int, chunk_path: str) -> Tuple[int, str, str]:
            def _read_audio(path: str) -> bytes:
                with open(path, "rb") as f:
                    return f.read()

            audio_bytes = await asyncio.get_running_loop().run_in_executor(None, _read_audio, chunk_path)
            audio_buffer = io.BytesIO(audio_bytes)
            audio_buffer.name = os.path.basename(chunk_path)

            transcription = await whisper_client.audio.transcriptions.create(
                model=settings.whisper_model,
                file=audio_buffer,
            )
            return index, transcription.text.strip(), chunk_path

        tasks = [_transcribe_chunk(idx, path) for idx, path in enumerate(chunks)]
        results = await asyncio.gather(*tasks)
        results.sort(key=lambda x: x[0])

        transcriptions = []
        for _, text, chunk_path in results:
            if text:
                transcriptions.append(text)
            if chunk_path != str(local_path) and os.path.exists(chunk_path):
                try:
                    os.remove(chunk_path)
                except Exception:
                    pass

        transcribed_text = " ".join(transcriptions).strip()
        if not transcribed_text:
            return None

        return f"[Audio: {local_path}] - {transcribed_text}"
    except Exception as e:
        logger.error("🎙️ Error processing uploaded audio: %s", e)
        return None


async def _process_uploaded_image(image: UploadFile, user_text: str, settings: SethSettings) -> Tuple[Optional[str], str]:
    try:
        os.makedirs(settings.storage_images_dir, exist_ok=True)
        raw = await image.read()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = os.path.splitext(image.filename or "")[1].lstrip(".").lower() or "jpg"
        local_path = settings.storage_images_dir / f"img_{timestamp}_{uuid4().hex[:8]}.{ext}"

        with open(local_path, "wb") as f:
            f.write(raw)

        base64_image = base64.b64encode(raw).decode("utf-8")
        image_tag = f"[Image: {local_path}]"

        text_out = f"{image_tag} - {user_text}" if user_text else f"{image_tag} - Describe this image."
        return base64_image, text_out
    except Exception as e:
        logger.error("📸 Error processing uploaded image: %s", e)
        return None, user_text


def _split_audio_if_needed(local_path: str, max_duration_sec: int = 60) -> List[str]:
    try:
        audio = AudioSegment.from_file(local_path)
        duration_sec = len(audio) / 1000.0
        if duration_sec <= max_duration_sec:
            return [local_path]

        chunks = []
        ext = os.path.splitext(local_path)[1][1:]
        for i in range(0, len(audio), max_duration_sec * 1000):
            chunk = audio[i : i + max_duration_sec * 1000]
            chunk_path = f"{local_path}_part{i}.{ext}"
            chunk.export(chunk_path, format=ext)
            chunks.append(chunk_path)
        return chunks
    except Exception as e:
        logger.warning("Audio split notice: %s", e)
        return [local_path]
