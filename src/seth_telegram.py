"""
SETH-IN-A-BOX -- TELEGRAM EDITION (seth_telegram.py)
"Inspired by my prompt engineering research and publications on Medium: https://medium.com/@luis.capra"

This is a thin client, not another copy of the brain. Every tool, memory
layer, the dynamic regulator, and the LLM itself live in seth_api.py -- this
file's only job is translating between Telegram's Update/Bot objects and
seth_api.py's HTTP surface (POST /api/register, POST /api/chat), so the exact
same backend now serves both the_oracle.html (web) and Telegram, instead of
the Telegram bot carrying its own private copy of everything the way
seth_poc.py's SethChatBot/SethTelegramBot did.

Concretely, that means this process never touches vLLM, Whisper, Qdrant,
Neo4j/Graphiti, or the diffusion/TTS models directly -- it just downloads
whatever Telegram hands it (text/photo/voice/audio) and forwards the raw
bytes to seth_api.py, which does the actual transcription, image tagging,
tool calling, and memory management (the exact same code that used to live
in SethTelegramBot._handle_voice_message/_handle_photo_message, just
relocated server-side). The user-facing behavior -- registration flow,
message text, error wording, media handling, long-message splitting, the
typing indicator -- is deliberately kept identical to seth_poc.py's
SethTelegramBot; see the docstring on each method for the handful of
spots where the new architecture forced a small, deliberate difference.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import coloredlogs
import httpx
from dotenv import load_dotenv
from telegram import Update
from telegram.error import NetworkError, TimedOut
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters
from telegram.request import HTTPXRequest

# Anchor point for every relative path this file uses by default (storage/,
# .env) instead of bare relative strings, so behavior doesn't depend on
# where `python seth_telegram.py` happens to be launched from (plain shell,
# VS Code's integrated terminal, systemd, ...). Same reasoning as seth_api.py.
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# Same convention as seth_api.py: storage/ lives at the project root, next to
# README.md and seth_api.py's own storage/ -- both processes' local state
# ends up under the same top-level folder, just different files within it
# (telegram_sessions.json here, images/audio/logs from seth_api.py's side).
PROJECT_ROOT = BASE_DIR.parent


@dataclass(frozen=True)
class SethTelegramEnvironment:
    """Deliberately tiny compared to seth_api.py's SethEnvironment: this
    process has no model, memory, or storage config of its own -- just enough
    to speak Telegram and know where seth_api.py lives. Reads REGISTRATION_TOKEN
    from the same .env seth_api.py uses (it's the same underlying credential,
    just checked in two different places -- see SethTelegramSessionStore's
    docstring for why that's still safe with a single source of truth)."""
    telegram_token: str = os.getenv("TELEGRAM_TOKEN", "")
    telegram_registration_token: str = os.getenv("REGISTRATION_TOKEN", "")
    api_base_url: str = os.getenv("SETH_API_BASE_URL", "http://127.0.0.1:8080")
    sessions_path: str = str(PROJECT_ROOT / "storage" / "telegram_sessions.json")

    def validate(self):
        if not self.telegram_token:
            raise ValueError("❌ TELEGRAM_TOKEN is missing in the environment.")
        if not self.telegram_registration_token:
            raise ValueError("❌ REGISTRATION_TOKEN is missing in the environment.")


class SethLoggerInit:
    """Same coloring/format/per-run-file philosophy as seth_poc.py's version,
    trimmed of the mem0-specific handler -- mem0 doesn't run in this process."""
    def __init__(self):
        self.prepare_coloredlogs()
        self.prepare_silence()
        self.prepare_run()

    def prepare_coloredlogs(self):
        coloredlogs.install(
            level='INFO',
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            level_styles={
                'info': {'color': 'green'},
                'warning': {'color': 'yellow', 'bold': True},
                'error': {'color': 'red', 'bold': True},
                'critical': {'color': 'red', 'bg': 'white', 'bold': True},
                'debug': {'color': 'black', 'bright': True}
            },
            field_styles={
                'asctime': {'color': 'cyan'},
                'hostname': {'color': 'magenta'},
                'levelname': {'color': 'white', 'bold': True},
                'name': {'color': 'blue'}
            }
        )

    def prepare_silence(self):
        """Suppresses verbose logging from third-party libraries."""
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)

    def prepare_run(self):
        """Sets up a dedicated log file for each run."""
        run_logs_dir = str(PROJECT_ROOT / "storage" / "logs" / "runs")
        os.makedirs(run_logs_dir, exist_ok=True)
        start_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_log_file = os.path.join(run_logs_dir, f"telegram_run_{start_time_str}.log")
        run_file_handler = logging.FileHandler(run_log_file, encoding='utf-8')
        run_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
        run_file_handler.setLevel(logging.INFO)
        logging.getLogger().addHandler(run_file_handler)


class SethTelegramSessionStore:
    """On-disk mapping of Telegram user id -> seth_api.py session id (the
    value this client sends back as the 'X-Seth-User' header on every
    /api/chat call). This doubles as the allow-list SethSecurityBoss used to
    be in seth_poc.py: a Telegram user counts as authorized here if and only
    if a mapping already exists for them.

    Unlike the original SethSecurityBoss, this file does NOT itself validate
    the registration token -- it only decides whether a message LOOKS like a
    registration attempt (via TokenMatchFilter, using its own local copy of
    REGISTRATION_TOKEN purely for that routing decision). The actual grant
    always goes through a real call to seth_api.py's POST /api/register,
    which is the single source of truth for whether a token is valid. If the
    two REGISTRATION_TOKEN copies ever drift, the worst case is a confusing
    UX (this file's filter waves a message through to handle_registration,
    but the API still rejects it) -- never an accidental bypass, since the
    API is always the one actually deciding.
    """
    def __init__(self, env: SethTelegramEnvironment):
        self.env = env
        self.filepath = env.sessions_path
        self._ensure_storage_exists()
        self.sessions: dict[str, str] = self._load_sessions()

    def _ensure_storage_exists(self):
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        if not os.path.exists(self.filepath):
            try:
                with open(self.filepath, "w", encoding="utf-8") as f:
                    json.dump({}, f, indent=4)
                logging.info(f"📁 [SESSIONS] Created clean database file at {self.filepath}")
            except Exception as e:
                logging.error(f"❌ Error creating telegram_sessions.json: {e}")

    def _load_sessions(self) -> dict[str, str]:
        try:
            with open(self.filepath, "r") as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"❌ Error reading telegram_sessions.json: {e}")
            return {}

    def is_allowed(self, telegram_user_id: int) -> bool:
        return str(telegram_user_id) in self.sessions

    def get_api_user_id(self, telegram_user_id: int) -> str | None:
        return self.sessions.get(str(telegram_user_id))

    def store(self, telegram_user_id: int, api_user_id: str):
        self.sessions[str(telegram_user_id)] = api_user_id
        try:
            with open(self.filepath, "w") as f:
                json.dump(self.sessions, f, indent=4)
            logging.info(f"🔒 [SECURITY] New Telegram user mapped to SETH API session: ID {telegram_user_id} -> {api_user_id[:8]}…")
        except Exception as e:
            logging.error(f"❌ Error saving session mapping to JSON: {e}")


class SethAPIClient:
    """Thin async wrapper around seth_api.py's HTTP surface. This is the
    entire dependency this file has on "the brain" -- no vLLM/Whisper/Qdrant/
    Neo4j clients live here anymore, all of that is seth_api.py's problem."""

    # Generous read timeout: a tool-heavy hop (web search, image generation)
    # can legitimately take a while, and unlike a browser tab, Telegram's
    # only real time pressure is the typing-indicator refresh this file
    # already handles separately via _keep_alive_chat_action.
    _TIMEOUT = httpx.Timeout(connect=15.0, read=300.0, write=120.0, pool=10.0)

    def __init__(self, env: SethTelegramEnvironment):
        self.env = env
        self.base_url = env.api_base_url.rstrip("/")

    async def register(self, token: str) -> str | None:
        """POST /api/register. Returns the freshly minted api_user_id, or
        None if the token was rejected or the API is unreachable."""
        try:
            async with httpx.AsyncClient(timeout=self._TIMEOUT) as hc:
                resp = await hc.post(f"{self.base_url}/api/register", json={"token": token})
            if resp.status_code == 200:
                return resp.json().get("user_id")
            logging.warning(f"🔒 [SECURITY] /api/register rejected a token attempt: HTTP {resp.status_code}")
            return None
        except Exception as e:
            logging.error(f"❌ Could not reach SETH API to register: {e}")
            return None

    async def chat_stream(
        self,
        api_user_id: str,
        message: str = "",
        image_bytes: bytes | None = None,
        image_filename: str = "image.jpg",
        image_content_type: str = "image/jpeg",
        audio_bytes: bytes | None = None,
        audio_filename: str = "audio.ogg",
        audio_content_type: str = "audio/ogg",
    ):
        """POST /api/chat, yields parsed SSE events as dicts:
            {"type": "reasoning"|"content", "text": "..."}
            {"type": "tool_start"|"tool_end", "name": "..."}
            {"type": "done", "media": [...], "tool_calls_used": [...]}
            {"type": "error", "error": "..."}
        Mirrors the shape SethAPIBot.ask_stream() produces server-side --
        this is just the client-side half of the same protocol the_oracle.html
        already speaks."""
        files = {}
        if image_bytes is not None:
            files["image"] = (image_filename, image_bytes, image_content_type)
        if audio_bytes is not None:
            files["audio"] = (audio_filename, audio_bytes, audio_content_type)

        headers = {"X-Seth-User": api_user_id}

        async with httpx.AsyncClient(timeout=self._TIMEOUT) as hc:
            async with hc.stream(
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
                            yield {"type": "done", "media": meta.get("media", []), "tool_calls_used": meta.get("tool_calls_used", [])}


class SethTelegramBridge:
    """Bridges Telegram events to seth_api.py calls. This is the direct
    replacement for seth_poc.py's SethTelegramBot -- same registration flow,
    same authorization gate shape, same message handling, same logging
    conventions -- except it no longer inherits from SethChatBot (there's no
    tool-calling loop, memory, or LLM client left in this file at all)."""

    def __init__(self, env: SethTelegramEnvironment, api_client: SethAPIClient, session_store: SethTelegramSessionStore):
        self.env = env
        self.api_client = api_client
        self.session_store = session_store

    # ------------------------------------------------------------ handlers --

    async def start_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("--- [SETH-IN-A-BOX IS ONLINE] ---")

    async def handle_registration(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        WELCOME_MESSAGE = """
✅ ¡Welkom! 😊

Mi nombre es SETH, y estoy atrapado en una caja negra.

Podemos hablar sin censura ni límite de tiempo. Puedo generar imágenes, crear audios, analizar imágenes y buscar información en la web.

¿No sabés qué puedo hacer? Preguntame qué tools tengo o para qué sirve cada una.

⚡ Esto es solo una prueba de concepto.
""".strip()

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

    async def handle_unauthorized(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        if not user or not update.message:
            return

        logging.warning(f"🚨 [UNAUTHORIZED ACCESS] ID: {user.id} - @{user.username if user.username else 'NoUsername'}")
        await update.message.reply_text(
            "⛔ **Restricted Access**"
        )

    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        err = context.error

        if isinstance(err, NetworkError):
            logging.warning(f"Telegram NetworkError: {err}")
            return

        if isinstance(err, TimedOut):
            logging.warning(f"Telegram Timeout: {err}")
            return

        logging.exception(
            "Telegram exception",
            exc_info=context.error
        )

    async def process(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        if not user:
            return

        telegram_user_id = str(user.id)
        api_user_id = self.session_store.get_api_user_id(user.id)
        if not api_user_id:
            # Shouldn't happen -- AuthorizedUserFilter already gates this handler
            # on is_allowed(), which checks for exactly this mapping. Defensive only.
            await self.handle_unauthorized(update, context)
            return

        await self._process_for_user(update, context, telegram_user_id, api_user_id)

    async def _process_for_user(self, update: Update, context: ContextTypes.DEFAULT_TYPE, telegram_user_id: str, api_user_id: str):
        stop_event = asyncio.Event()
        action_type = "record_voice" if (update.message.voice or update.message.audio) else "typing"
        keep_alive_task = asyncio.create_task(
            self._keep_alive_chat_action(context.bot, update.effective_chat.id, action_type, stop_event)
        )

        try:
            message_text = update.message.text or update.message.caption or ""
            image_bytes = image_filename = image_content_type = None
            audio_bytes = audio_filename = audio_content_type = None

            if update.message.voice or update.message.audio:
                downloaded = await self._download_telegram_audio(update, context)
                if downloaded is None:
                    return
                audio_bytes, audio_filename, audio_content_type = downloaded

            elif update.message.photo:
                downloaded = await self._download_telegram_photo(update, context)
                if downloaded is None:
                    return
                image_bytes, image_filename, image_content_type = downloaded

            if not message_text and not image_bytes and not audio_bytes:
                await update.message.reply_text("⚠️ Unsupported format.")
                return

            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

            final_text = ""
            media = []
            error_text = None

            async for event in self.api_client.chat_stream(
                api_user_id,
                message=message_text,
                image_bytes=image_bytes, image_filename=image_filename or "image.jpg", image_content_type=image_content_type or "image/jpeg",
                audio_bytes=audio_bytes, audio_filename=audio_filename or "audio.ogg", audio_content_type=audio_content_type or "audio/ogg",
            ):
                etype = event.get("type")
                if etype == "content":
                    final_text += event["text"]
                elif etype == "error":
                    error_text = event["error"]
                elif etype == "done":
                    media = event.get("media", [])
                # "reasoning", "tool_start", "tool_end" are intentionally not
                # surfaced here -- the original SethTelegramBot never showed
                # reasoning or per-tool progress to the Telegram user either,
                # it only relied on the typing indicator throughout.

            if error_text:
                # seth_api.py's error strings (audio unintelligible, empty
                # message, context overflow, ...) are already written to be
                # shown to a user, unlike a raw stack trace -- so unlike the
                # generic exception path below, this one is safe to relay directly.
                await update.message.reply_text(f"❌ {error_text}")
                return

            if await self._send_media_if_present(update, context, media):
                return

            await self._send_long_message(update, final_text)

        except Exception as e:
            logging.exception(f"❌ Internal inference error: {str(e)}")
            await update.message.reply_text("❌ Error with vision processing.")
        finally:
            stop_event.set()
            keep_alive_task.cancel()
            try:
                await keep_alive_task
            except asyncio.CancelledError:
                pass

    # -------------------------------------------------------- media (in) --

    async def _download_telegram_audio(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> tuple[bytes, str, str] | None:
        """Downloads the voice/audio message into memory and hands the raw
        bytes straight to seth_api.py -- no local save, no transcription here
        anymore. seth_api.py's _handle_uploaded_audio does exactly what
        seth_poc.py's _handle_voice_message used to (save to storage/audio,
        split if >60s, transcribe chunks in parallel via Whisper, prefix the
        transcript with "[Audio: path] -"), it's just relocated server-side."""
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="record_voice")
        try:
            audio_obj = update.message.voice if update.message.voice else update.message.audio
            telegram_file = await context.bot.get_file(audio_obj.file_id)
            raw = bytes(await telegram_file.download_as_bytearray())

            is_voice = update.message.voice is not None
            filename = f"voice_{telegram_file.file_id[:8]}.ogg" if is_voice else f"audio_{telegram_file.file_id[:8]}.mp3"
            content_type = "audio/ogg" if is_voice else "audio/mpeg"

            logging.info(f"🎙️ [AUDIO RECEIVED] {len(raw)} bytes from Telegram, forwarding to SETH API as {filename}")
            return raw, filename, content_type

        except Exception as e:
            logging.error(f"🎙️ Error downloading audio from Telegram: {e}")
            await update.message.reply_text("❌ Error processing voice message.")
            return None

    async def _download_telegram_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> tuple[bytes, str, str] | None:
        """Same idea as _download_telegram_audio: raw bytes only. seth_api.py's
        _handle_uploaded_image does the base64 encoding and the "[Image: path] -"
        tagging that seth_poc.py's _handle_photo_message used to do locally."""
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
        try:
            telegram_file = await context.bot.get_file(update.message.photo[-1].file_id)
            raw = bytes(await telegram_file.download_as_bytearray())
            filename = f"img_{telegram_file.file_id[:8]}.jpg"

            logging.info(f"📸 [IMAGE RECEIVED] {len(raw)} bytes from Telegram, forwarding to SETH API as {filename}")
            return raw, filename, "image/jpeg"

        except Exception as e:
            logging.error(f"📸 Error downloading photo from Telegram: {e}")
            await update.message.reply_text("❌ Cannot process or store the visual file you sent.")
            return None

    # ------------------------------------------------------- media (out) --

    async def _send_media_if_present(self, update: Update, context: ContextTypes.DEFAULT_TYPE, media: list[dict]) -> bool:
        """Adapted from seth_poc.py's _send_media_if_present: the generated
        file no longer lives on this process's local disk (it lives wherever
        seth_api.py runs), so instead of open()-ing a local path this fetches
        the bytes over HTTP from the /storage/... URL seth_api.py already
        computed via _extract_media_refs. Only ever sends the first item --
        same as the original, which returned immediately on the first match
        (image checked before audio) and skipped sending the text response
        entirely once a media item went out.

        One deliberate difference from seth_poc.py: there, skipping straight
        to `return True` here also meant the turn never got saved to short/graph
        memory (the append calls sat after this check in _process_for_user).
        That's no longer this file's call to make -- seth_api.py always saves
        the turn server-side once it has a final response, media or not, since
        memory is now shared across every channel (web + Telegram) rather than
        being a per-transport decision."""
        if not media:
            return False

        item = media[0]
        url = f"{self.api_client.base_url}{item['url']}"

        try:
            async with httpx.AsyncClient(timeout=30.0) as hc:
                resp = await hc.get(url)
                resp.raise_for_status()
                raw = resp.content
        except Exception as e:
            logging.error(f"❌ Could not fetch generated media from SETH API ({url}): {e}")
            return False

        if item.get("type") == "image":
            logging.info("📸 Detected image in response. Sending photo to Telegram.")
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="upload_photo")
            await update.message.reply_photo(photo=io.BytesIO(raw))
            return True

        if item.get("type") == "audio":
            logging.info("📁 Sending voice response.")
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="record_voice")
            await update.message.reply_voice(voice=io.BytesIO(raw))
            return True

        return False

    # ----------------------------------------------------- text delivery --

    async def _send_long_message(self, update: Update, text: str, max_length: int = 4096):
        """Sends long messages by splitting them automatically while safely
        keeping Markdown code blocks intact. Verbatim from seth_poc.py --
        pure text formatting, no dependency on anything that moved server-side."""
        if not text:
            return

        if len(text) <= max_length:
            await update.message.reply_text(text)
            return

        chunks = []
        current_chunk = ""
        in_code_block = False
        current_language = ""

        lines = text.split('\n')

        for line in lines:
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                if in_code_block:
                    current_language = line.strip()[3:].strip()
                else:
                    current_language = ""

            if len(current_chunk) + len(line) + 50 > max_length:
                if current_chunk:
                    if in_code_block:
                        current_chunk += "\n```"
                    chunks.append(current_chunk.strip())

                if in_code_block:
                    current_chunk = f"``` {current_language}\n{line}"
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
                if i == 0:
                    await update.message.reply_text(chunk)
                else:
                    await update.message.reply_text(f"({i+1}/{len(chunks)})\n\n{chunk}")

                await asyncio.sleep(0.3)

            except Exception as e:
                logging.error(f"Error sending chunk {i}: {e}")
                await update.message.reply_text(f"⚠️ The response is too long. Here's a part: {chunk[:2500]}...")

    async def _keep_alive_chat_action(self, bot, chat_id: int, action: str, stop_event: asyncio.Event):
        """Verbatim from seth_poc.py: Telegram's typing indicator expires
        after a few seconds, so this refreshes it every 4s for as long as
        we're waiting on seth_api.py."""
        while not stop_event.is_set():
            try:
                await bot.send_chat_action(chat_id=chat_id, action=action)
            except Exception as e:
                logging.debug(f"⚠️ Keep-alive action failed: {e}")
            await asyncio.sleep(4.0)

    # ------------------------------------------------------------------ run --

    def run(self):
        class AuthorizedUserFilter(filters.MessageFilter):
            def __init__(self, session_store):
                super().__init__()
                self.session_store = session_store

            def filter(self, message):
                return message.from_user is not None and self.session_store.is_allowed(message.from_user.id)

        class TokenMatchFilter(filters.MessageFilter):
            def __init__(self, token):
                super().__init__()
                self.token = token

            def filter(self, message):
                return message.text is not None and message.text.strip() == self.token

        is_authorized = AuthorizedUserFilter(self.session_store)
        is_token = TokenMatchFilter(self.env.telegram_registration_token)

        app = (
            ApplicationBuilder()
            .token(self.env.telegram_token)
            .request(HTTPXRequest(
                connection_pool_size=10,
                read_timeout=120.0,
                write_timeout=120.0,
                connect_timeout=15.0,
                pool_timeout=10.0
            ))
            .build()
        )
        app.add_handler(CommandHandler("start", self.start_cmd))

        app.add_handler(MessageHandler(
            ~is_authorized & is_token & filters.TEXT & ~filters.COMMAND,
            self.handle_registration
        ))

        app.add_handler(MessageHandler(
            is_authorized & (filters.TEXT | filters.PHOTO | filters.VOICE | filters.AUDIO) & ~filters.COMMAND,
            self.process
        ))

        app.add_handler(MessageHandler(
            ~is_authorized & ~filters.COMMAND,
            self.handle_unauthorized
        ))

        app.add_error_handler(self.error_handler)

        logging.info(f"🚀 [SETH TELEGRAM] Bridging to SETH API at {self.env.api_base_url}")
        app.run_polling()


def main():
    SethLoggerInit()
    env = SethTelegramEnvironment()
    env.validate()

    api_client = SethAPIClient(env)
    session_store = SethTelegramSessionStore(env)

    bridge = SethTelegramBridge(env=env, api_client=api_client, session_store=session_store)
    bridge.run()


if __name__ == '__main__':
    main()