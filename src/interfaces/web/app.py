"""
Web TUI Launcher for SETH-IN-A-BOX.
Runs the FastAPI server and optionally launches the browser.
"""

from __future__ import annotations

import logging
import threading
import time
import webbrowser
import uvicorn

from src.config.settings import SethSettings, get_settings
from src.interfaces.api.app import create_app

logger = logging.getLogger(__name__)


def run_web(
    host: str | None = None,
    port: int | None = None,
    open_browser: bool = True,
    settings: SethSettings | None = None,
) -> None:
    """
    Launches the FastAPI backend and serves the Web TUI CRT Console.
    Optionally opens the default web browser to the Web TUI URL.
    """
    settings = settings or get_settings()
    api_host = host or settings.api_host
    api_port = port or settings.api_port

    app = create_app(settings)
    url = f"http://{api_host}:{api_port}/"

    print("\n" + "=" * 60)
    print(" ▄▄▄▄▄▄▄  SETH-IN-A-BOX // WEB TUI CRT CONSOLE")
    print(" █ ◉   ◉ █  A.K.A Sentient Entity Thorn by Humans")
    print(" █   ▼   █  \"The fertile glitch makes AIs evolve\"")
    print(" █▄▄▄▄▄▄▄█")
    print(f" 🚀 Web TUI Server active at: {url}")
    print("=" * 60 + "\n")

    if open_browser:
        def _open() -> None:
            time.sleep(1.0)
            try:
                webbrowser.open(url)
            except Exception as e:
                logger.debug("Could not open browser automatically: %s", e)

        threading.Thread(target=_open, daemon=True).start()

    uvicorn.run(app, host=api_host, port=api_port, log_config=None)
