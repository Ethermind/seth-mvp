"""
Streaming chunk accumulator for OpenAI/vLLM completion deltas.
Reconstructs fragmented tool call arguments and reasoning traces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.domain.models import Role, ToolCall


@dataclass(slots=True)
class FakeFunctionCall:
    name: str
    arguments: str


@dataclass(slots=True)
class FakeToolCall:
    id: str
    function: FakeFunctionCall


class StreamAccumulator:
    """Accumulates token deltas and fragmented tool calls from streaming completions."""

    def __init__(self) -> None:
        self.content: str = ""
        self.reasoning: str = ""
        self.finish_reason: Optional[str] = None
        self._tool_slots: Dict[int, Dict[str, str]] = {}

    def ingest_tool_call_deltas(self, tool_call_deltas: Any) -> None:
        """Accumulates tool call pieces streamed by chunk index."""
        for tc in tool_call_deltas:
            index = getattr(tc, "index", 0)
            slot = self._tool_slots.setdefault(index, {"id": "", "name": "", "arguments": ""})
            if getattr(tc, "id", None):
                slot["id"] = tc.id
            fn = getattr(tc, "function", None)
            if fn is not None:
                if getattr(fn, "name", None):
                    slot["name"] += fn.name
                if getattr(fn, "arguments", None):
                    slot["arguments"] += fn.arguments

    def to_domain_tool_calls(self) -> List[ToolCall]:
        """Converts accumulated tool slots into domain ToolCall instances."""
        calls: List[ToolCall] = []
        for idx, slot in sorted(self._tool_slots.items()):
            call_id = slot["id"] or f"call_{idx}"
            calls.append(ToolCall(call_id=call_id, name=slot["name"], arguments=slot["arguments"]))
        return calls

    def to_message_dict(self) -> Dict[str, Any]:
        """Converts accumulated state into an assistant message dict for vLLM context."""
        msg: Dict[str, Any] = {"role": Role.ASSISTANT.value, "content": self.content}
        if self._tool_slots:
            msg["tool_calls"] = [
                {
                    "id": slot["id"] or f"call_{idx}",
                    "type": "function",
                    "function": {"name": slot["name"], "arguments": slot["arguments"]},
                }
                for idx, slot in sorted(self._tool_slots.items())
            ]
        return msg
