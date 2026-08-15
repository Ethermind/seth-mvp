"""
Unit tests for SuppressNoisyAccessLogFilter (src.interfaces.api.middleware.log_filter).
"""

from __future__ import annotations

import logging
from src.interfaces.api.middleware.log_filter import SuppressNoisyAccessLogFilter


def test_log_filter_suppression():
    filter_instance = SuppressNoisyAccessLogFilter()

    # Suppressed routes
    rec_status = logging.LogRecord("uvicorn.access", logging.INFO, "", 0, 'GET /api/status HTTP/1.1 200', (), None)
    assert filter_instance.filter(rec_status) is False

    # Allowed routes
    rec_chat = logging.LogRecord("uvicorn.access", logging.INFO, "", 0, 'POST /api/chat HTTP/1.1 200', (), None)
    assert filter_instance.filter(rec_chat) is True
