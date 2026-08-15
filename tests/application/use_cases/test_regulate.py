"""
Unit tests for RegulateInferenceUseCase and UserStateManager (src.application.use_cases.regulate).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock
import pytest
import torch

from src.application.use_cases.regulate import RegulateInferenceUseCase, UserStateManager
from src.config.settings import SethSettings
from src.domain.models import RegulatorState


def test_user_state_manager_save_and_load(tmp_path: Path):
    settings = SethSettings(project_root=tmp_path)
    mgr = UserStateManager(settings)

    # Initial state is default
    st = mgr.get_state("user_test_1")
    assert st.temperature == 0.25

    # Modify and save
    st.temperature = 0.85
    st.top_p = 0.95
    mgr.save_state("user_test_1", st)

    # Recreate manager and load from disk
    mgr_fresh = UserStateManager(settings)
    loaded = mgr_fresh.get_state("user_test_1")
    assert loaded.temperature == 0.85
    assert loaded.top_p == 0.95


@pytest.mark.anyio
async def test_regulate_inference_use_case(tmp_path: Path):
    settings = SethSettings(project_root=tmp_path)

    # Mock embedding engine to return deterministic tensors
    mock_engine = MagicMock()
    # Return normalized 1D tensor
    mock_engine.encode.return_value = torch.tensor([1.0, 0.0, 0.0])

    use_case = RegulateInferenceUseCase(settings=settings, embedding_engine=mock_engine, alpha=0.5)

    # Adjust for query
    new_state = await use_case.adjust_for_query("user_reg_1", "Optimize Python algorithm")
    assert isinstance(new_state, RegulatorState)
    assert 0.0 <= new_state.temperature <= 2.0
