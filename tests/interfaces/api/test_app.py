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


def test_create_app_memory_tool_schemas(tmp_path):
    """Verifies that memory tools exposed to LLM have valid non-empty schemas with required parameters."""
    settings = SethSettings(project_root=tmp_path, registration_token="secret_token")
    mock_mem = MagicMock()
    mock_mem.raw_mem0.embedding_model.model = MagicMock()

    with patch("src.interfaces.api.app.Mem0SemanticMemory", return_value=mock_mem):
        with patch("src.interfaces.api.app.GraphitiRelationalMemory"):
            with patch("src.interfaces.api.app.StableDiffusionGenerator"):
                with patch("src.interfaces.api.app.KokoroSynthesizer"):
                    with patch("src.interfaces.api.app.ToolRegistry") as MockToolRegistry:
                        # Capture instances passed to register_instances
                        real_registry_instance = None
                        from src.infrastructure.tools.registry import ToolRegistry
                        real_registry = ToolRegistry()
                        MockToolRegistry.return_value = real_registry

                        create_app(settings=settings)

                        schemas = {s["function"]["name"]: s["function"]["parameters"] for s in real_registry.get_openai_schemas()}

                        # query_relationship_graph
                        assert "query_relationship_graph" in schemas
                        assert "query" in schemas["query_relationship_graph"]["properties"]
                        assert "query" in schemas["query_relationship_graph"]["required"]
                        assert "user_id" not in schemas["query_relationship_graph"]["properties"]

                        # save_long_term_memory
                        assert "save_long_term_memory" in schemas
                        assert "user_input" in schemas["save_long_term_memory"]["properties"]
                        assert "response" in schemas["save_long_term_memory"]["properties"]
                        assert "user_input" in schemas["save_long_term_memory"]["required"]
                        assert "response" in schemas["save_long_term_memory"]["required"]
                        assert "user_id" not in schemas["save_long_term_memory"]["properties"]
