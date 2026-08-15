"""
Log filter to suppress high-frequency telemetry polling in uvicorn access logs.
"""

from __future__ import annotations

import logging


class SuppressNoisyAccessLogFilter(logging.Filter):
    """Filters out /api/status telemetry polls from uvicorn access log."""

    NOISY_SUBSTRINGS = ("GET /api/status",)

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(noisy in msg for noisy in self.NOISY_SUBSTRINGS)
