"""
SETH-IN-A-BOX -- TELEGRAM EDITION (seth_telegram.py)
Backward-compatibility entrypoint delegating to modular src.interfaces.telegram.
"""

from __future__ import annotations

from src.interfaces.telegram.bot import main


if __name__ == "__main__":
    main()