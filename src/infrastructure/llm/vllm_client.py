"""
vLLM / OpenAI-compatible client adapter implementing LLMProvider protocol.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Dict, List, Optional

from openai import AsyncOpenAI
from src.config.settings import SethSettings, get_settings
from src.domain.exceptions import InferenceEngineError
from src.domain.models import Message, RegulatorState, Role, StreamChunk, StreamEventType, ToolCall
from src.infrastructure.llm.accumulator import StreamAccumulator
from src.infrastructure.llm.context_guard import ContextWindowGuard

logger = logging.getLogger(__name__)


class VllmClient:
    """Inference client connected to vLLM (or any OpenAI-compatible server)."""

    def __init__(
        self,
        settings: SethSettings | None = None,
        client: AsyncOpenAI | None = None,
        context_guard: ContextWindowGuard | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.client = client or AsyncOpenAI(base_url=self.settings.vllm_url, api_key=self.settings.api_key)
        self.context_guard = context_guard or ContextWindowGuard(self.settings)

    def _prepare_kwargs(
        self,
        raw_messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        config: Optional[RegulatorState] = None,
    ) -> Dict[str, Any]:
        conf_dict = config.to_dict() if config else {}
        kwargs: Dict[str, Any] = {
            "model": self.settings.llm_model,
            "messages": raw_messages,
            **conf_dict,
        }

        requested_output = 4096
        safe_output = self.context_guard.calculate_safe_output_tokens(
            raw_messages=raw_messages,
            tools=tools,
            requested_output_tokens=requested_output,
        )
        kwargs["max_tokens"] = safe_output

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        if self.settings.llm_enable_thinking:
            kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": True},
                "skip_special_tokens": False,
            }

        return kwargs

    async def generate(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        config: Optional[RegulatorState] = None,
    ) -> Message:
        """Runs standard non-streaming completion."""
        raw_messages = [m.to_dict() for m in messages]
        kwargs = self._prepare_kwargs(raw_messages, tools, config)

        try:
            response = await self.client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            msg = choice.message

            tool_calls: List[ToolCall] = []
            if getattr(msg, "tool_calls", None):
                for tc in msg.tool_calls:
                    tool_calls.append(
                        ToolCall(
                            call_id=tc.id,
                            name=tc.function.name,
                            arguments=tc.function.arguments,
                        )
                    )

            return Message(
                role=Role.ASSISTANT,
                content=msg.content or "",
                tool_calls=tool_calls,
            )
        except Exception as e:
            logger.exception("❌ Error in vLLM generate: %s", e)
            raise InferenceEngineError(f"vLLM inference failed: {e}") from e

    async def generate_stream(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        config: Optional[RegulatorState] = None,
    ) -> AsyncIterator[StreamChunk]:
        """Streams completion tokens, thinking reasoning deltas, and tool calls."""
        raw_messages = [m.to_dict() for m in messages]
        kwargs = self._prepare_kwargs(raw_messages, tools, config)

        try:
            stream = await self.client.chat.completions.create(stream=True, **kwargs)
            accumulator = StreamAccumulator()

            async for chunk in stream:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                delta = choice.delta

                reasoning_piece = getattr(delta, "reasoning", None)
                if reasoning_piece:
                    accumulator.reasoning += reasoning_piece
                    yield StreamChunk(event_type=StreamEventType.REASONING, text=reasoning_piece)

                if getattr(delta, "content", None):
                    accumulator.content += delta.content
                    yield StreamChunk(event_type=StreamEventType.CONTENT, text=delta.content)

                if getattr(delta, "tool_calls", None):
                    accumulator.ingest_tool_call_deltas(delta.tool_calls)

                if choice.finish_reason:
                    accumulator.finish_reason = choice.finish_reason

            # Emit detected whole tool calls if any
            domain_tool_calls = accumulator.to_domain_tool_calls()
            for tc in domain_tool_calls:
                yield StreamChunk(
                    event_type=StreamEventType.TOOL_START,
                    tool_name=tc.name,
                    metadata={"tool_call": tc, "assistant_message_dict": accumulator.to_message_dict()},
                )

        except Exception as e:
            logger.exception("❌ Error in vLLM generate_stream: %s", e)
            yield StreamChunk(event_type=StreamEventType.ERROR, text=str(e), ok=False)
