"""
Context window token estimation and guardrails using AutoTokenizer.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from transformers import AutoTokenizer
from src.config.settings import SethSettings, get_settings
from src.domain.exceptions import ContextLimitExceededError

logger = logging.getLogger(__name__)


class ContextWindowGuard:
    """Estimates input token size against model context limit and adjusts max_tokens dynamically."""

    def __init__(self, settings: SethSettings | None = None) -> None:
        self.settings = settings or get_settings()
        self.tokenizer = AutoTokenizer.from_pretrained(self.settings.llm_model, trust_remote_code=True)
        self.max_allowed_context = self.settings.max_tokens

    def calculate_safe_output_tokens(
        self,
        raw_messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        requested_output_tokens: int = 4096,
        min_output_tokens: int = 256,
    ) -> int:
        """
        Estimates total token length of formatted prompt and ensures available output slots.
        Raises ContextLimitExceededError if input already exceeds max allowed context.
        """
        try:
            formatted_chat = self.tokenizer.apply_chat_template(
                raw_messages,
                tools=tools,
                tokenize=True,
                add_generation_prompt=True,
            )
            estimated_input_tokens = len(formatted_chat)
        except Exception as e:
            logger.debug("Chat template tokenization fallback: %s", e)
            # Rough character-based fallback (~4 chars per token)
            estimated_input_tokens = sum(len(str(m.get("content", ""))) for m in raw_messages) // 4

        available_output_slots = self.max_allowed_context - estimated_input_tokens

        if available_output_slots < min_output_tokens:
            raise ContextLimitExceededError(
                estimated_tokens=estimated_input_tokens,
                max_tokens=self.max_allowed_context,
            )

        if requested_output_tokens > available_output_slots:
            adjusted_output = max(min_output_tokens, available_output_slots - 50)
            logger.warning(
                "⚠️ [CONTEXT OVERFLOW GUARD] Adjusting requested output tokens from %d to %d (Input: %d, Max: %d).",
                requested_output_tokens,
                adjusted_output,
                estimated_input_tokens,
                self.max_allowed_context,
            )
            return adjusted_output

        return requested_output_tokens
