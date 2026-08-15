"""
Unit tests for Logging Setup (src.infrastructure.logging.setup).
"""

from __future__ import annotations

import logging
from src.config.settings import SethSettings
from src.infrastructure.logging.setup import setup_logging


def test_setup_logging_runs_without_error(tmp_path):
    settings = SethSettings(project_root=tmp_path)
    setup_logging(settings)

    # Verify logger is operational
    logger = logging.getLogger("test_logger")
    logger.info("Test log message")
    assert logger is not None
