"""
Unit tests for KokoroSynthesizer (src.infrastructure.tools.speech_kokoro).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch
import pytest

from src.config.settings import SethSettings
from src.infrastructure.tools.speech_kokoro import KokoroSynthesizer


@pytest.mark.anyio
async def test_kokoro_synthesizer_generate_speech(tmp_path: Path):
    settings = SethSettings(project_root=tmp_path)
    synth = KokoroSynthesizer(settings=settings)

    mock_mp3_path = str(settings.storage_audio_dir / "speech_test.mp3")
    Path(mock_mp3_path).touch()

    with patch.object(synth, "synthesize", return_value=mock_mp3_path):
        res_json = await synth.generate_speech("Hola, soy SETH.")

    data = json.loads(res_json)
    assert data["status"] == "success"
    assert data["local_path"] == mock_mp3_path
