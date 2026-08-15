"""
Standalone Web TUI Server for SETH-IN-A-BOX.
Serves the static Retro CRT Web TUI console independently from the FastAPI backend.
"""

from __future__ import annotations

import functools
import http.server
import logging
from pathlib import Path
import threading
import time
import urllib.request
import webbrowser

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"


class WebTUIHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Simple HTTP Request Handler serving Web TUI static files with fallback to index.html."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def do_GET(self) -> None:
        # Fallback root and unknown routes to index.html
        req_path = self.path.split("?")[0]
        target_file = STATIC_DIR / req_path.lstrip("/")
        if req_path in ("/", "/oracle", "/tui") or not target_file.exists():
            self.path = "/index.html"
        super().do_GET()

    def end_headers(self) -> None:
        # Prevent caching of static assets during active local usage
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def log_message(self, format: str, *args) -> None:
        # Suppress noisy access logs
        pass


def check_api_status(api_url: str = "http://127.0.0.1:8080") -> bool:
    """Quick reachability probe against the backend API status endpoint."""
    url = f"{api_url.rstrip('/')}/api/status"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "SETH-WebTUI-Probe/1.0"})
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            return resp.status == 200
    except Exception:
        return False


def run_web(
    host: str = "127.0.0.1",
    port: int = 5500,
    api_url: str = "http://127.0.0.1:8080",
    open_browser: bool = True,
) -> None:
    """
    Launches the standalone Web TUI HTTP server.
    Decoupled from the heavy backend API process.
    """
    web_url = f"http://{host}:{port}/"
    api_is_online = check_api_status(api_url)

    print("\n" + "=" * 65)
    print(" ▄▄▄▄▄▄▄  SETH-IN-A-BOX // WEB TUI CRT CONSOLE")
    print(" █ ◉   ◉ █  A.K.A Sentient Entity Thorn by Humans")
    print(" █   ▼   █  \"The fertile glitch makes AIs evolve\"")
    print(" █▄▄▄▄▄▄▄█")
    print(f" 🌐 Web TUI Server active at : {web_url}")
    print(f" 🔌 Target Backend API URL   : {api_url}")
    if api_is_online:
        print(" 🟢 Backend API Status       : ONLINE & REACHABLE")
    else:
        print(" ⚠️  Backend API Status       : OFFLINE / NOT DETECTED")
        print(" 💡 Tip: Start the API backend in another terminal with:")
        print("    python -m src.main api")
    print("=" * 65 + "\n")

    if open_browser:
        def _open() -> None:
            time.sleep(0.8)
            try:
                webbrowser.open(web_url)
            except Exception as e:
                logger.debug("Could not open browser automatically: %s", e)

        threading.Thread(target=_open, daemon=True).start()

    server_address = (host, port)
    handler_factory = functools.partial(WebTUIHTTPRequestHandler)
    httpd = http.server.ThreadingHTTPServer(server_address, handler_factory)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[dim]Stopping Web TUI server... Goodbye.[/dim]")
        httpd.server_close()
