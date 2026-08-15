"""
Telegram Bot Application Runner for SETH-IN-A-BOX.
"""

from __future__ import annotations

import logging
from typing import Optional

from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters
from telegram.request import HTTPXRequest

from src.config.settings import SethSettings, get_settings
from src.infrastructure.logging.setup import setup_logging
from src.interfaces.client import SethClient
from src.interfaces.telegram.handlers import TelegramBridgeHandlers
from src.interfaces.telegram.session_store import TelegramSessionStore

logger = logging.getLogger(__name__)


class TelegramBotRunner:
    """Configures and runs the Telegram Bot polling application."""

    def __init__(self, settings: Optional[SethSettings] = None) -> None:
        self.settings = settings or get_settings()
        self.settings.validate_for_telegram()
        self.api_client = SethClient(base_url=self.settings.seth_api_base_url)
        self.session_store = TelegramSessionStore(self.settings)
        self.handlers = TelegramBridgeHandlers(
            settings=self.settings,
            api_client=self.api_client,
            session_store=self.session_store,
        )

    def run(self) -> None:
        """Starts the telegram polling loop."""
        setup_logging(self.settings)

        class AuthorizedUserFilter(filters.MessageFilter):
            def __init__(self, store: TelegramSessionStore) -> None:
                super().__init__()
                self.store = store

            def filter(self, message):  # type: ignore[no-untyped-def]
                return message.from_user is not None and self.store.is_allowed(message.from_user.id)

        class TokenMatchFilter(filters.MessageFilter):
            def __init__(self, token: str) -> None:
                super().__init__()
                self.token = token

            def filter(self, message):  # type: ignore[no-untyped-def]
                return message.text is not None and message.text.strip() == self.token

        is_authorized = AuthorizedUserFilter(self.session_store)
        is_token = TokenMatchFilter(self.settings.registration_token)

        app = (
            ApplicationBuilder()
            .token(self.settings.telegram_token)
            .request(
                HTTPXRequest(
                    connection_pool_size=10,
                    read_timeout=120.0,
                    write_timeout=120.0,
                    connect_timeout=15.0,
                    pool_timeout=10.0,
                )
            )
            .build()
        )

        app.add_handler(CommandHandler("start", self.handlers.start_cmd))
        app.add_handler(
            MessageHandler(
                ~is_authorized & is_token & filters.TEXT & ~filters.COMMAND,
                self.handlers.handle_registration,
            )
        )
        app.add_handler(
            MessageHandler(
                is_authorized & (filters.TEXT | filters.PHOTO | filters.VOICE | filters.AUDIO) & ~filters.COMMAND,
                self.handlers.process,
            )
        )
        app.add_handler(
            MessageHandler(~is_authorized & ~filters.COMMAND, self.handlers.handle_unauthorized)
        )
        app.add_error_handler(self.handlers.error_handler)

        logger.info("🚀 [SETH TELEGRAM] Bridging to API at %s", self.settings.seth_api_base_url)
        app.run_polling()


def main() -> None:
    runner = TelegramBotRunner()
    runner.run()


if __name__ == "__main__":
    main()
