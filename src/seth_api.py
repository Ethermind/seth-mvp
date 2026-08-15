"""
SETH-IN-A-BOX -- API EDITION (seth_api.py)
Backward-compatibility entrypoint delegating to modular src.interfaces.api.
"""

from __future__ import annotations

import uvicorn
from src.config.settings import get_settings
from src.interfaces.api.app import create_app


def main():
    settings = get_settings()
    app = create_app(settings)
    print(f"🚀 [SETH API] Starting on http://{settings.api_host}:{settings.api_port}")
    uvicorn.run(app, host=settings.api_host, port=settings.api_port, log_config=None)


if __name__ == "__main__":
    main()