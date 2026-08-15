"""
Domain protocols (structural contracts) for SETH-IN-A-BOX.
Uses PEP 544 typing.Protocol for clean, duck-typed abstractions without inheritance boilerplate.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Dict, List, Optional, Protocol
from src.domain.models import GpuTelemetry, Message, RegulatorState, StreamChunk, SystemStatus


class LLMProvider(Protocol):
    """Protocol for LLM inference engines (vLLM, OpenAI, etc.)."""

    async def generate(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        config: Optional[RegulatorState] = None,
    ) -> Message:
        """Runs non-streaming completion."""
        ...

    async def generate_stream(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        config: Optional[RegulatorState] = None,
    ) -> AsyncIterator[StreamChunk]:
        """Streams completion token-by-token and tool deltas."""
        ...


class SemanticMemory(Protocol):
    """Protocol for semantic long-term memory (Mem0 / Qdrant)."""

    async def search(self, user_id: str, query: str, limit: int = 10) -> List[str]:
        """Searches relevant long-term semantic memories for a user."""
        ...

    async def save(self, user_id: str, fact: str, response: str) -> bool:
        """Stores a new fact or preference for a user."""
        ...


class GraphMemory(Protocol):
    """Protocol for temporal and relational knowledge graph (Graphiti / Neo4j)."""

    async def add_episode(self, user_id: str, user_text: str, assistant_text: str) -> None:
        """Appends a conversational episode to the knowledge graph."""
        ...

    async def query_relations(self, user_id: str, query: str) -> List[str]:
        """Queries relational connections and temporal evolution."""
        ...


class ConversationHistory(Protocol):
    """Protocol for managing short-term conversational context."""

    async def get_history(self, user_id: str) -> List[Message]:
        """Retrieves recent message turns for a user."""
        ...

    async def append_turn(self, user_id: str, user_text: str, assistant_text: str) -> None:
        """Appends user input and assistant response to short-term history."""
        ...


class AudioTranscriber(Protocol):
    """Protocol for speech-to-text transcription (Whisper)."""

    async def transcribe(self, audio_bytes: bytes, filename: str = "audio.ogg") -> Optional[str]:
        """Transcribes raw audio bytes into text."""
        ...


class ImageGenerator(Protocol):
    """Protocol for text-to-image synthesis (Stable Diffusion)."""

    async def generate_image(self, prompt: str) -> str:
        """Generates an image from prompt and returns local filepath."""
        ...


class SpeechSynthesizer(Protocol):
    """Protocol for text-to-speech synthesis (Kokoro)."""

    async def synthesize(self, text: str) -> str:
        """Synthesizes Spanish audio from text and returns local MP3 filepath."""
        ...


class WebSearcher(Protocol):
    """Protocol for web search and page content extraction."""

    async def search_and_crawl(self, query: str, max_results: int = 5) -> str:
        """Performs live search and extracts markdown page content."""
        ...


class SessionStore(Protocol):
    """Protocol for user authentication and session allow-list."""

    def is_allowed(self, user_id: str) -> bool:
        """Checks if a session ID is authorized."""
        ...

    def register(self, user_id: str, token: str) -> bool:
        """Registers a new session ID if token is valid."""
        ...


class HardwareProbe(Protocol):
    """Protocol for querying system health and GPU telemetry."""

    async def probe_vram(self) -> Optional[List[GpuTelemetry]]:
        """Queries per-GPU VRAM usage."""
        ...

    async def collect_status(self) -> SystemStatus:
        """Gathers complete status across microservices and hardware."""
        ...
