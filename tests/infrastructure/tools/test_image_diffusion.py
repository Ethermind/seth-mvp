"""
Unit tests for StableDiffusionGenerator (src.infrastructure.tools.image_diffusion).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from src.config.settings import SethSettings
from src.infrastructure.tools.image_diffusion import StableDiffusionGenerator


@pytest.mark.anyio
async def test_image_diffusion_generation_and_cleanup_timer(tmp_path: Path):
    settings = SethSettings(project_root=tmp_path)
    gen = StableDiffusionGenerator(settings=settings, idle_timeout_secs=10)

    mock_img_path = str(settings.storage_images_dir / "test_gen.png")
    Path(mock_img_path).touch()

    with patch.object(gen, "_sync_generate", return_value=mock_img_path):
        res_path = await gen.generate_image("a cyberpunk hacker terminal")

    assert res_path == mock_img_path
    assert gen._cleanup_task is not None
    assert not gen._cleanup_task.done()

    # Cancel task cleanly
    gen._cleanup_task.cancel()
