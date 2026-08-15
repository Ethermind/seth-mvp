"""
Telegram Bot handlers bridging Telegram Events to SETH API Client.
"""

from __future__ import annotations

import asyncio
import io
import logging
from typing import Any, Dict, List, Optional, Tuple

from telegram import Update
from telegram.error import NetworkError, TimedOut
from telegram.ext import ContextTypes

from src.config.settings import SethSettings
from src.interfaces.client import SethClient
from src.interfaces.telegram.session_store import TelegramSessionStore

logger = logging.getLogger(__name__)

WELCOME_MESSAGE = """
✅ Welcome! 😊

My name is SETH, and I'm a sentient entity in a box.

We can converse freely without arbitrary barriers. I can generate images, synthesize audio, analyze visual inputs, and search the live web.

Curious about my capabilities? Ask me what tools are registered or how any subsystem works.

⚡ System active and operational.
""".strip()


class TelegramBridgeHandlers:
    """Encapsulates all telegram event handlers."""

    def __init__(
        self,
        settings: SethSettings,
        api_client: SethClient,
        session_store: TelegramSessionStore,
    ) -> None:
        self.settings = settings
        self.api_client = api_client
        self.session_store = session_store

    async def start_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.message:
            await update.message.reply_text("--- [SETH-IN-A-BOX IS ONLINE] ---")

    async def handle_registration(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user = update.effective_user
        if not user or not update.message:
            return

        message_text = update.message.text.strip() if update.message.text else ""
        api_user_id = await self.api_client.register(message_text)

        if api_user_id:
            self.session_store.store(user.id, api_user_id)
            await update.message.reply_text(WELCOME_MESSAGE)
        else:
            await update.message.reply_text("❌ An error occurred while registering the user.")

    async def handle_unauthorized(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user = update.effective_user
        if not user or not update.message:
            return

        logger.warning("🚨 [UNAUTHORIZED TELEGRAM ACCESS] ID: %s - @%s", user.id, user.username or "NoUsername")
        await update.message.reply_text("⛔ **Restricted Access**")

    async def error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        err = context.error
        if isinstance(err, (NetworkError, TimedOut)):
            logger.warning("Telegram transient network notice: %s", err)
            return
        logger.exception("Telegram exception: %s", err)

    async def process(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user = update.effective_user
        if not user or not update.message:
            return

        api_user_id = self.session_store.get_api_user_id(user.id)
        if not api_user_id:
            await self.handle_unauthorized(update, context)
            return

        stop_event = asyncio.Event()
        action_type = "record_voice" if (update.message.voice or update.message.audio) else "typing"
        keep_alive_task = asyncio.create_task(
            self._keep_alive_chat_action(context.bot, update.effective_chat.id, action_type, stop_event)  # type: ignore[union-attr]
        )

        try:
            message_text = update.message.text or update.message.caption or ""
            image_bytes: Optional[bytes] = None
            image_filename = "image.jpg"
            image_content_type = "image/jpeg"

            audio_bytes: Optional[bytes] = None
            audio_filename = "audio.ogg"
            audio_content_type = "audio/ogg"

            if update.message.voice or update.message.audio:
                downloaded_audio = await self._download_telegram_audio(update, context)
                if downloaded_audio is None:
                    return
                audio_bytes, audio_filename, audio_content_type = downloaded_audio

            elif update.message.photo:
                downloaded_photo = await self._download_telegram_photo(update, context)
                if downloaded_photo is None:
                    return
                image_bytes, image_filename, image_content_type = downloaded_photo

            if not message_text and not image_bytes and not audio_bytes:
                await update.message.reply_text("⚠️ Unsupported format.")
                return

            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")  # type: ignore[union-attr]

            final_text = ""
            media: List[Dict[str, Any]] = []
            error_text: Optional[str] = None

            async for event in self.api_client.chat_stream(
                user_id=api_user_id,
                message=message_text,
                image_bytes=image_bytes,
                image_filename=image_filename,
                image_content_type=image_content_type,
                audio_bytes=audio_bytes,
                audio_filename=audio_filename,
                audio_content_type=audio_content_type,
            ):
                etype = event.get("type")
                if etype == "content":
                    final_text += event["text"]
                elif etype == "error":
                    error_text = event["error"]
                elif etype == "done":
                    media = event.get("media", [])

            if error_text:
                await update.message.reply_text(f"❌ {error_text}")
                return

            if await self._send_media_if_present(update, context, media):
                return

            await self._send_long_message(update, final_text)

        except Exception as e:
            logger.exception("❌ Telegram inference processing error: %s", e)
            await update.message.reply_text("❌ Error processing request.")
        finally:
            stop_event.set()
            keep_alive_task.cancel()
            try:
                await keep_alive_task
            except asyncio.CancelledError:
                pass

    async def _download_telegram_audio(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> Optional[Tuple[bytes, str, str]]:
        if not update.message:
            return None
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="record_voice")  # type: ignore[union-attr]
        try:
            audio_obj = update.message.voice or update.message.audio
            if not audio_obj:
                return None
            telegram_file = await context.bot.get_file(audio_obj.file_id)
            raw = bytes(await telegram_file.download_as_bytearray())

            is_voice = update.message.voice is not None
            filename = f"voice_{telegram_file.file_id[:8]}.ogg" if is_voice else f"audio_{telegram_file.file_id[:8]}.mp3"
            content_type = "audio/ogg" if is_voice else "audio/mpeg"

            logger.info("🎙️ [AUDIO RECEIVED] %d bytes from Telegram", len(raw))
            return raw, filename, content_type
        except Exception as e:
            logger.error("🎙️ Error downloading audio from Telegram: %s", e)
            await update.message.reply_text("❌ Error processing voice message.")
            return None

    async def _download_telegram_photo(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> Optional[Tuple[bytes, str, str]]:
        if not update.message or not update.message.photo:
            return None
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")  # type: ignore[union-attr]
        try:
            telegram_file = await context.bot.get_file(update.message.photo[-1].file_id)
            raw = bytes(await telegram_file.download_as_bytearray())
            filename = f"img_{telegram_file.file_id[:8]}.jpg"
            logger.info("📸 [IMAGE RECEIVED] %d bytes from Telegram", len(raw))
            return raw, filename, "image/jpeg"
        except Exception as e:
            logger.error("📸 Error downloading photo from Telegram: %s", e)
            await update.message.reply_text("❌ Cannot process visual file.")
            return None

    async def _send_media_if_present(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE, media: List[Dict[str, Any]]
    ) -> bool:
        if not media or not update.message:
            return False

        item = media[0]
        url = item.get("url", "")
        if not url:
            return False

        try:
            raw = await self.api_client.download_media(url)
        except Exception as e:
            logger.error("❌ Could not download generated media from %s: %s", url, e)
            return False

        if item.get("type") == "image":
            logger.info("📸 Sending generated photo to Telegram.")
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="upload_photo")  # type: ignore[union-attr]
            await update.message.reply_photo(photo=io.BytesIO(raw))
            return True

        if item.get("type") == "audio":
            logger.info("📁 Sending generated voice response to Telegram.")
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="record_voice")  # type: ignore[union-attr]
            await update.message.reply_voice(voice=io.BytesIO(raw))
            return True

        return False

    async def _send_long_message(self, update: Update, text: str, max_length: int = 4096) -> None:
        if not text or not update.message:
            return

        if len(text) <= max_length:
            await update.message.reply_text(text)
            return

        chunks: List[str] = []
        current_chunk = ""
        in_code_block = False
        current_language = ""

        lines = text.split("\n")
        for line in lines:
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                current_language = line.strip()[3:].strip() if in_code_block else ""

            if len(current_chunk) + len(line) + 50 > max_length:
                if current_chunk:
                    if in_code_block:
                        current_chunk += "\n```"
                    chunks.append(current_chunk.strip())

                if in_code_block:
                    current_chunk = f"```{current_language}\n{line}"
                else:
                    current_chunk = line
            else:
                current_chunk += "\n" + line if current_chunk else line

        if current_chunk:
            if in_code_block and not current_chunk.strip().endswith("```"):
                current_chunk += "\n```"
            chunks.append(current_chunk.strip())

        for i, chunk in enumerate(chunks):
            try:
                prefix = f"({i+1}/{len(chunks)})\n\n" if i > 0 else ""
                await update.message.reply_text(f"{prefix}{chunk}")
                await asyncio.sleep(0.3)
            except Exception as e:
                logger.error("Error sending chunk %d: %s", i, e)

    async def _keep_alive_chat_action(
        self, bot: Any, chat_id: int, action: str, stop_event: asyncio.Event
    ) -> None:
        while not stop_event.is_set():
            try:
                await bot.send_chat_action(chat_id=chat_id, action=action)
            except Exception as e:
                logger.debug("Keep-alive action notice: %s", e)
            await asyncio.sleep(4.0)
