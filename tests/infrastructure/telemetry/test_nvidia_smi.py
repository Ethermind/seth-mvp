"""
Unit tests for NvidiaSmiProbe (src.infrastructure.telemetry.nvidia_smi).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch
import pytest

from src.infrastructure.telemetry.nvidia_smi import NvidiaSmiProbe


@pytest.mark.anyio
async def test_nvidia_smi_probe_parsing():
    sample_csv = "0, NVIDIA GeForce RTX 4090, 8192, 24576\n1, NVIDIA RTX A6000, 4096, 49152\n"

    with patch("asyncio.create_subprocess_exec") as mock_exec:
        proc = AsyncMock()
        proc.communicate.return_value = (sample_csv.encode("utf-8"), b"")
        proc.returncode = 0
        mock_exec.return_value = proc

        gpus = await NvidiaSmiProbe.probe_vram()

    assert gpus is not None
    assert len(gpus) == 2
    assert gpus[0].name == "NVIDIA GeForce RTX 4090"
    assert gpus[0].used_gb == 8.0
    assert gpus[0].total_gb == 24.0
    assert gpus[1].name == "NVIDIA RTX A6000"


@pytest.mark.anyio
async def test_nvidia_smi_probe_failure():
    with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError):
        with patch("torch.cuda.is_available", return_value=False):
            gpus = await NvidiaSmiProbe.probe_vram()
    assert gpus is None
