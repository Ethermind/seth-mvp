"""
Hardware probe using nvidia-smi with PyTorch fallback.
"""

from __future__ import annotations

import asyncio
import logging
from typing import List, Optional

import torch
from src.domain.models import GpuTelemetry

logger = logging.getLogger(__name__)


class NvidiaSmiProbe:
    """Queries per-GPU physical memory via nvidia-smi NVML query."""

    @staticmethod
    async def probe_vram() -> Optional[List[GpuTelemetry]]:
        """Enumerates physical GPUs directly via nvidia-smi."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total",
                "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=3.0)
            gpus: List[GpuTelemetry] = []
            for line in stdout.decode().strip().splitlines():
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    idx, name, used_mib, total_mib = parts[0], parts[1], parts[2], parts[3]
                    gpus.append(
                        GpuTelemetry(
                            index=int(idx),
                            name=name,
                            used_gb=round(float(used_mib) / 1024, 2),
                            total_gb=round(float(total_mib) / 1024, 2),
                        )
                    )
            if gpus:
                return gpus
        except Exception as e:
            logger.debug("nvidia-smi probe failed, falling back to torch.cuda: %s", e)

        # Fallback to torch.cuda if available
        try:
            if torch.cuda.is_available():
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                return [
                    GpuTelemetry(
                        index=torch.cuda.current_device(),
                        name=torch.cuda.get_device_name(),
                        used_gb=round((total_bytes - free_bytes) / (1024**3), 2),
                        total_gb=round(total_bytes / (1024**3), 2),
                    )
                ]
        except Exception as e:
            logger.debug("torch.cuda VRAM fallback also failed: %s", e)

        return None
