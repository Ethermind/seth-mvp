"""
Unit tests for SethSettings (src.config.settings).
"""

from __future__ import annotations

from pathlib import Path
import pytest

from src.config.settings import SethSettings, get_settings


def test_settings_default_paths(tmp_path: Path):
    settings = SethSettings(
        src_dir=tmp_path / "src",
        project_root=tmp_path,
        registration_token="test_token_123",
        telegram_token="tg_token_123",
    )

    assert settings.image_model_full_path == tmp_path / "models" / "dreamshaper_8.safetensors"
    assert settings.conversations_path == tmp_path / "conversations"
    assert settings.storage_images_dir == tmp_path / "storage" / "images"
    assert settings.storage_audio_dir == tmp_path / "storage" / "audio"
    assert settings.reasoning_audit_dir == tmp_path / "storage" / "logs" / "reasoning"
    assert settings.allowed_users_file == tmp_path / "storage" / "allowed_api_users.json"
    assert settings.telegram_sessions_file == tmp_path / "storage" / "telegram_sessions.json"


def test_cors_origins_parsing(tmp_path: Path):
    settings = SethSettings(
        project_root=tmp_path,
        cors_allowed_origins="http://localhost:3000, http://127.0.0.1:5500, https://oracle.seth.ai",
    )
    origins = settings.parsed_cors_origins
    assert len(origins) == 3
    assert "http://localhost:3000" in origins
    assert "http://127.0.0.1:5500" in origins
    assert "https://oracle.seth.ai" in origins


def test_system_prompt_path_fallback(tmp_path: Path):
    src_dir = tmp_path / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = src_dir / "seth.md"
    prompt_file.write_text("Fallback prompt", encoding="utf-8")

    settings = SethSettings(src_dir=src_dir, project_root=tmp_path)
    assert settings.system_prompt_path == prompt_file


def test_system_prompt_path_pkg(tmp_path: Path):
    src_dir = tmp_path / "src"
    prompt_dir = src_dir / "prompt"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = prompt_dir / "seth.md"
    prompt_file.write_text("Pkg prompt", encoding="utf-8")

    settings = SethSettings(src_dir=src_dir, project_root=tmp_path)
    assert settings.system_prompt_path == prompt_file


def test_validate_for_api(tmp_path: Path):
    settings = SethSettings(project_root=tmp_path, registration_token="")
    with pytest.raises(ValueError, match="REGISTRATION_TOKEN is missing"):
        settings.validate_for_api()

    valid_settings = SethSettings(project_root=tmp_path, registration_token="secret")
    valid_settings.validate_for_api()


def test_validate_for_telegram(tmp_path: Path):
    settings = SethSettings(project_root=tmp_path, registration_token="", telegram_token="")
    with pytest.raises(ValueError, match="TELEGRAM_TOKEN is missing"):
        settings.validate_for_telegram()

    settings_with_tg = SethSettings(project_root=tmp_path, registration_token="", telegram_token="tok")
    with pytest.raises(ValueError, match="REGISTRATION_TOKEN is missing"):
        settings_with_tg.validate_for_telegram()

    valid = SethSettings(project_root=tmp_path, registration_token="tok", telegram_token="tok")
    valid.validate_for_telegram()


def test_get_settings_singleton():
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
