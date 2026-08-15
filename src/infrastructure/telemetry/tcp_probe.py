"""
Non-invasive TCP reachability probing.
"""

from __future__ import annotations

import asyncio
import logging
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)


async def probe_tcp_port(url_or_host: str, default_port: int = 80, timeout: float = 3.0) -> str:
    """
    Confirms a service is listening via a bare TCP connect-and-close.
    Does not send HTTP requests, avoiding noisy access logs in target servers.
    """
    if "://" in url_or_host:
        parsed = urlsplit(url_or_host)
        host = parsed.hostname or "localhost"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    else:
        parts = url_or_host.split(":")
        host = parts[0]
        port = int(parts[1]) if len(parts) > 1 else default_port

    try:
        reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=timeout)
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
        return "online"
    except Exception as e:
        return f"offline ({e})"
