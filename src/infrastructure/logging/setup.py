"""
Centralized logging initialization for SETH.
"""

from __future__ import annotations

from datetime import datetime
import logging
import os

import warnings

import coloredlogs
from src.config.settings import SethSettings, get_settings


def setup_logging(settings: SethSettings | None = None) -> None:
    """Configures colored logs, file handlers and suppresses third-party noise."""
    settings = settings or get_settings()

    # Suppress benign library deprecation/compatibility warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message=".*Qdrant client version.*")
    warnings.filterwarnings("ignore", message=".*Support for class-based `config` is deprecated.*")

    # 1. Colored terminal logs
    coloredlogs.install(
        level="INFO",
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level_styles={
            "info": {"color": "green"},
            "warning": {"color": "yellow", "bold": True},
            "error": {"color": "red", "bold": True},
            "critical": {"color": "red", "bg": "white", "bold": True},
            "debug": {"color": "black", "bright": True},
        },
        field_styles={
            "asctime": {"color": "cyan"},
            "hostname": {"color": "magenta"},
            "levelname": {"color": "white", "bold": True},
            "name": {"color": "blue"},
        },
    )

    # 2. Silence third-party noise
    for noisy in (
        "httpx",
        "httpcore",
        "openai",
        "urllib3",
        "asyncio",
        "watchfiles",
        "posthog",
        "neo4j.notifications",
    ):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    # 3. Mem0 dedicated file log
    mem0_log_path = settings.log_mem0_path or str(settings.project_root / "storage" / "logs" / "mem0.log")
    os.makedirs(os.path.dirname(mem0_log_path), exist_ok=True)
    mem0_logger = logging.getLogger("mem0")
    mem0_handler = logging.FileHandler(mem0_log_path, encoding="utf-8")
    mem0_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    mem0_logger.addHandler(mem0_handler)
    mem0_logger.setLevel(logging.INFO)

    # 4. Per-run log file
    run_logs_dir = settings.project_root / "storage" / "logs" / "runs"
    os.makedirs(run_logs_dir, exist_ok=True)
    start_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_file = run_logs_dir / f"run_{start_time_str}.log"
    run_handler = logging.FileHandler(run_log_file, encoding="utf-8")
    run_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    run_handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(run_handler)
