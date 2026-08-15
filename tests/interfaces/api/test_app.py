"""
Unit tests for FastAPI App Factory (src.interfaces.api.app).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from fastapi import FastAPI

from src.config.settings import SethSettings
from src.interfaces.api.app import create_app


def test_create_app_factory(tmp_path):
    settings = SethSettings(project_root=tmp_path, registration_token="secret_token")

    # Mock heavy ML preloads during startup
    mock_mem = MagicMock()
    mock_mem.raw_mem0.embedding_model.model = MagicMock()

    with patch("src.interfaces.api.app.Mem0SemanticMemory", return_value=mock_mem):
        with patch("src.interfaces.api.app.GraphitiRelationalMemory"):
            with patch("src.interfaces.api.app.StableDiffusionGenerator"):
                with patch("src.interfaces.api.app.KokoroSynthesizer"):
                    app = create_app(settings=settings)

    assert isinstance(app, FastAPI)
    assert app.title == "SETH-IN-A-BOX API"
