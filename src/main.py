"""
Unified Entrypoint CLI for SETH-IN-A-BOX.
Run with:
    python -m src.main api       (Starts FastAPI + SSE backend)
    python -m src.main web       (Starts and opens Retro CRT Web TUI)
    python -m src.main tui       (Starts interactive Terminal console)
    python -m src.main telegram  (Starts Telegram Bot bridge)
"""

from __future__ import annotations

import argparse
import sys
import uvicorn

from src.config.settings import get_settings
from src.interfaces.api.app import create_app
from src.interfaces.telegram.bot import TelegramBotRunner
from src.interfaces.tui.app import main as tui_main
from src.interfaces.web.app import run_web


def run_api(host: str | None = None, port: int | None = None) -> None:
    """Launches the FastAPI backend service with uvicorn."""
    settings = get_settings()
    api_host = host or settings.api_host
    api_port = port or settings.api_port

    app = create_app(settings)
    print(f"🚀 [SETH API] Starting on http://{api_host}:{api_port}")
    print(f"🌐 [SETH Web TUI] Available at http://{api_host}:{api_port}/")
    uvicorn.run(app, host=api_host, port=api_port, log_config=None)


def run_telegram() -> None:
    """Launches the Telegram Bot client."""
    runner = TelegramBotRunner()
    runner.run()


def run_tui() -> None:
    """Launches the Terminal User Interface console."""
    tui_main()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SETH-IN-A-BOX // Multi-Interface AI Agentic Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="mode", help="Execution mode")

    # API mode
    api_parser = subparsers.add_parser("api", help="Launch FastAPI REST & SSE Service")
    api_parser.add_argument("--host", type=str, default=None, help="Host to bind (default: 127.0.0.1)")
    api_parser.add_argument("--port", type=int, default=None, help="Port to bind (default: 8080)")

    # Web TUI mode
    web_parser = subparsers.add_parser("web", help="Launch and serve Web TUI CRT Console")
    web_parser.add_argument("--host", type=str, default=None, help="Host to bind (default: 127.0.0.1)")
    web_parser.add_argument("--port", type=int, default=None, help="Port to bind (default: 8080)")
    web_parser.add_argument("--no-browser", action="store_true", help="Do not automatically open browser")

    # Telegram mode
    subparsers.add_parser("telegram", help="Launch Telegram Bot bridge")

    # TUI mode
    subparsers.add_parser("tui", help="Launch Terminal User Interface CRT console")

    args = parser.parse_args()

    if args.mode == "api":
        run_api(host=args.host, port=args.port)
    elif args.mode == "web":
        run_web(host=args.host, port=args.port, open_browser=not args.no_browser)
    elif args.mode == "telegram":
        run_telegram()
    elif args.mode == "tui":
        run_tui()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
