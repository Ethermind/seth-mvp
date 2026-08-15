"""
Use Case: Dynamic Inference Regulation based on semantic intent embeddings.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
import json
import logging
import os
from pathlib import Path
import re
from typing import Any, Dict, Optional

from sentence_transformers import SentenceTransformer
import torch
from src.config.settings import SethSettings, get_settings
from src.domain.models import RegulatorPresets, RegulatorState

logger = logging.getLogger(__name__)


class UserStateManager:
    """Manages on-disk state persistence per user session."""

    def __init__(self, settings: SethSettings) -> None:
        self.settings = settings
        self._states: Dict[str, RegulatorState] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        os.makedirs(self.settings.state_path.parent, exist_ok=True)

    def _safe_user_fragment(self, user_id: str) -> str:
        return re.sub(r"[^A-Za-z0-9_\-]", "_", str(user_id)) or "unknown"

    def _path_for(self, user_id: str) -> Path:
        base, ext = os.path.splitext(self.settings.state_path)
        return Path(f"{base}_{self._safe_user_fragment(user_id)}{ext or '.json'}")

    def get_lock(self, user_id: str) -> asyncio.Lock:
        if user_id not in self._locks:
            self._locks[user_id] = asyncio.Lock()
        return self._locks[user_id]

    def get_state(self, user_id: str) -> RegulatorState:
        if user_id in self._states:
            return self._states[user_id]

        state = RegulatorPresets.default()
        path = self._path_for(user_id)
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    state = RegulatorState.from_dict(data)
            except Exception as e:
                logger.error("Error reading user state file %s: %s", path, e)

        self._states[user_id] = state
        return state

    def save_state(self, user_id: str, state: RegulatorState) -> None:
        path = self._path_for(user_id)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(state.to_dict(), f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error("Error writing user state file %s: %s", path, e)


class RegulateInferenceUseCase:
    """Dynamically shifts LLM temperature, top_p, and presence_penalty based on query intent."""

    def __init__(self, settings: SethSettings | None = None, embedding_engine: Any | None = None, alpha: float = 0.2) -> None:
        self.settings = settings or get_settings()
        self.alpha = alpha
        self.state_manager = UserStateManager(self.settings)

        if embedding_engine is not None:
            self.engine = embedding_engine
        else:
            logger.info("Loading SentenceTransformer for Dynamic Regulator: %s", self.settings.embedding_model)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.engine = SentenceTransformer(self.settings.embedding_model, device=device)

        self.targets = {
            "rigorous": {
                "vector": self.engine.encode("Technical architecture, code precision, logic, systems design", convert_to_tensor=True, normalize_embeddings=True),
                "state": RegulatorPresets.rigorous(),
            },
            "chaotic": {
                "vector": self.engine.encode("Glitch aesthetics, humor, creative chaos, fertile glitch", convert_to_tensor=True, normalize_embeddings=True),
                "state": RegulatorPresets.chaotic(),
            },
            "verbose": {
                "vector": self.engine.encode("Deep philosophy, ontological analysis, long essay, legacy", convert_to_tensor=True, normalize_embeddings=True),
                "state": RegulatorPresets.verbose(),
            },
        }

    async def adjust_for_query(self, user_id: str, query: str) -> RegulatorState:
        """Calculates semantic vector similarity and shifts user hyperparameters smoothly."""
        lock = self.state_manager.get_lock(user_id)
        async with lock:
            return await asyncio.to_thread(self._adjust_sync, user_id, query)

    def _adjust_sync(self, user_id: str, query: str) -> RegulatorState:
        text = query.strip() or "neutral"
        q_vec = self.engine.encode(text, convert_to_tensor=True, normalize_embeddings=True)

        best_name, _ = max(
            ((n, torch.dot(q_vec, d["vector"]).item()) for n, d in self.targets.items()),
            key=lambda x: x[1],
        )

        state = self.state_manager.get_state(user_id)
        state.interpolate(self.targets[best_name]["state"], self.alpha)
        self.state_manager.save_state(user_id, state)

        logger.info("🌀 STATE ADJUSTMENT [%s] user=%s -> Temp: %.3f", best_name.upper(), user_id, state.temperature)
        return state
