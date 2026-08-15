"""
Domain entities and value objects for SETH-IN-A-BOX.
Pure Python implementations using dataclasses and StrEnum.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Dict, List, Optional, Union


class Role(StrEnum):
    """Conversational roles compatible with OpenAI and modern LLM chat templates."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class StreamEventType(StrEnum):
    """Event types emitted during streaming inference."""
    REASONING = "reasoning"
    CONTENT = "content"
    TOOL_START = "tool_start"
    TOOL_END = "tool_end"
    DONE = "done"
    ERROR = "error"


class MediaType(StrEnum):
    """Supported multimodal output/input media types."""
    IMAGE = "image"
    AUDIO = "audio"


@dataclass(slots=True, frozen=True)
class ToolCall:
    """Represents a tool execution request initiated by the LLM."""
    call_id: str
    name: str
    arguments: str


@dataclass(slots=True, frozen=True)
class ToolResult:
    """Represents the output produced after executing a tool callable."""
    call_id: str
    name: str
    output: str
    is_success: bool = True


@dataclass(slots=True, frozen=True)
class Message:
    """Represents a conversational message turn in the domain."""
    role: Role
    content: Union[str, List[Dict[str, Any]]]
    tool_call_id: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Converts message into standard OpenAI/vLLM dict representation."""
        data: Dict[str, Any] = {
            "role": self.role.value if isinstance(self.role, Role) else str(self.role),
            "content": self.content,
        }
        if self.tool_call_id:
            data["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            data["tool_calls"] = [
                {
                    "id": tc.call_id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments,
                    },
                }
                for tc in self.tool_calls
            ]
        return data


@dataclass(slots=True, frozen=True)
class StreamChunk:
    """Represents an atomic incremental chunk yielded during streaming inference."""
    event_type: StreamEventType
    text: str = ""
    tool_name: Optional[str] = None
    ok: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RegulatorState:
    """Dynamic inference hyperparameter state (temperature, top_p, presence_penalty)."""
    temperature: float = 0.25
    top_p: float = 0.85
    presence_penalty: float = 0.2

    def interpolate(self, target: RegulatorState, alpha: float) -> None:
        """Smoothly shifts the parameters towards a target state."""
        self.temperature += alpha * (target.temperature - self.temperature)
        self.top_p += alpha * (target.top_p - self.top_p)
        self.presence_penalty += alpha * (target.presence_penalty - self.presence_penalty)

    def to_dict(self) -> Dict[str, float]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "presence_penalty": self.presence_penalty,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RegulatorState:
        return cls(
            temperature=float(data.get("temperature", 0.25)),
            top_p=float(data.get("top_p", 0.85)),
            presence_penalty=float(data.get("presence_penalty", 0.2)),
        )


class RegulatorPresets:
    """Standard predefined behavioral states."""

    @classmethod
    def default(cls) -> RegulatorState:
        return RegulatorState(temperature=0.25, top_p=0.85, presence_penalty=0.2)

    @classmethod
    def rigorous(cls) -> RegulatorState:
        """High precision, low temperature for code, logic, and systems design."""
        return RegulatorState(temperature=0.1, top_p=0.7, presence_penalty=0.0)

    @classmethod
    def chaotic(cls) -> RegulatorState:
        """High entropy for creative glitch aesthetics, humor, and chaos."""
        return RegulatorState(temperature=1.3, top_p=0.99, presence_penalty=0.9)

    @classmethod
    def verbose(cls) -> RegulatorState:
        """Extended context for long-form essays, philosophy, and deep analysis."""
        return RegulatorState(temperature=0.85, top_p=0.95, presence_penalty=0.4)


@dataclass(slots=True, frozen=True)
class MediaAttachment:
    """Represents a generated or uploaded media file (image/audio)."""
    media_type: MediaType
    local_path: str
    url: str


@dataclass(slots=True, frozen=True)
class GpuTelemetry:
    """Metrics snapshot for an individual physical GPU."""
    index: int
    name: str
    used_gb: float
    total_gb: float


@dataclass(slots=True, frozen=True)
class SystemStatus:
    """Complete telemetry status across hardware and microservices."""
    vllm: str
    whisper: str
    qdrant: str
    neo4j_graphiti: str
    vram: Optional[List[GpuTelemetry]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vllm": self.vllm,
            "whisper": self.whisper,
            "qdrant": self.qdrant,
            "neo4j_graphiti": self.neo4j_graphiti,
            "vram": [asdict(g) for g in self.vram] if self.vram else None,
        }


@dataclass(slots=True, frozen=True)
class Session:
    """Represents an authorized user session."""
    user_id: str
    is_allowed: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
