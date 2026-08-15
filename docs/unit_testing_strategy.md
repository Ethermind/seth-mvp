# Comprehensive Unit & Integration Testing Strategy
## SETH-IN-A-BOX // Quality Assurance & Test Engineering Framework

> **Test Engineering Strategy Specification**  
> **Project:** SETH-IN-A-BOX (AKA *Sentient Entity Thorn by Humans*)  
> **Coverage Goal:** >90% across Domain, Application, and Core Infrastructure Layers  
> **Framework:** `pytest` + `pytest-asyncio` + `pytest-mock` + `respx`  

---

## 1. Testing Philosophy & Guiding Principles

The testing architecture for **SETH-IN-A-BOX** is designed around four non-negotiable principles:

1. **Zero Hardware & Microservice Hard Coupling:**
   Tests must run and pass in standard CI/CD environments (GitHub Actions, Linux containers) **without requiring physical NVIDIA GPUs, live vLLM servers, Whisper instances, Qdrant vector databases, or Neo4j clusters**. All external boundaries are isolated via structural protocols (`typing.Protocol`) and mock fixtures.
2. **Strict Protocol Contract Compliance:**
   Because the system uses PEP 544 `typing.Protocol` for decoupled interfaces, tests verify that all concrete infrastructure adapters accurately implement the protocol contracts.
3. **Full Asynchronous Lifecycle Verification:**
   The codebase relies heavily on `asyncio` (queues, locks, streaming generators, concurrent tool dispatch via `asyncio.gather`, and background tasks). The test suite exercises async concurrency, exception handling, and resource teardown.
4. **Hermetic File System Isolation:**
   All persistence mechanisms (`conversations/*.jsonl`, `storage/state/*.state`, `storage/logs/*.json`, `storage/allowed_api_users.json`) are tested against ephemeral `tmp_path` fixtures to prevent state leakage between test runs.

---

## 2. Test Suite Architecture & Directory Layout

```text
tests/
├── conftest.py                     # Global pytest fixtures, mock providers, and test settings
├── unit/                           # Isolated unit tests (<10ms per test)
│   ├── domain/
│   │   ├── test_models.py          # Message, ToolCall, RegulatorState, SystemStatus, serialization
│   │   ├── test_protocols.py       # Structural typing verification
│   │   └── test_exceptions.py      # Exception hierarchy and message formatting
│   ├── config/
│   │   └── test_settings.py        # Pydantic Settings v2 validation, paths, and environment parsing
│   ├── application/
│   │   ├── test_orchestrator.py    # ConversationOrchestrator multi-hop loop, tools, and streams
│   │   ├── test_use_cases.py       # Authenticate, Regulate, and Telemetry use cases
│   │   └── test_dto.py             # DTO dataclass initialization and attributes
│   ├── infrastructure/
│   │   ├── test_llm_accumulator.py # Tool call delta reconstruction and reasoning streaming
│   │   ├── test_context_guard.py   # Token estimation and dynamic output clamping
│   │   ├── test_vllm_client.py     # OpenAI-compatible vLLM adapter streaming & non-streaming
│   │   ├── test_tool_registry.py   # @tool discovery, type hint schema generation, dispatch
│   │   ├── test_jsonl_history.py   # Short-term memory sliding window and file locking
│   │   ├── test_mem0_memory.py     # Semantic long-term memory adapter (mocked Mem0/Qdrant)
│   │   ├── test_graphiti_memory.py # Relational temporal graph adapter (mocked Graphiti/Neo4j)
│   │   ├── test_tools_concrete.py  # Stable Diffusion idle timer, Kokoro TTS, Crawl4AI, Code Inspector
│   │   ├── test_json_sessions.py   # Session authorization whitelist and token exchange
│   │   ├── test_telemetry_probes.py# Nvidia-SMI parsing and TCP socket probes
│   │   └── test_audit_logger.py    # Reasoning audit JSON logger and 30-day retention pruning
│   └── interfaces/
│       ├── test_api_routes.py      # FastAPI endpoints: /api/register, /api/chat (SSE), /api/status
│       ├── test_client_sdk.py      # SethClient HTTPX methods with respx mocking
│       ├── test_telegram_bridge.py # Telegram bot handlers, filters, and session store
│       └── test_tui_app.py         # Terminal TUI status panel rendering and user loop
└── integration/                    # Multi-component workflow tests
    ├── test_e2e_chat_stream.py     # End-to-end conversation flow with mocked LLM provider
    └── test_multimodal_pipeline.py # Audio transcription and image ingestion pipelines
```

---

## 3. Mocking & Fixture Strategy

### 3.1 Global Test Configuration (`tests/conftest.py`)

A centralized `conftest.py` provides isolated environments and mock adapters conforming to the domain protocols:

```python
import pytest
from pathlib import Path
from typing import AsyncIterator, List, Optional, Dict, Any

from src.config.settings import SethSettings
from src.domain.models import Message, RegulatorState, Role, StreamChunk, StreamEventType, ToolCall, ToolResult
from src.domain.protocols import ConversationHistory, GraphMemory, LLMProvider, SemanticMemory

@pytest.fixture
def test_settings(tmp_path: Path) -> SethSettings:
    """Provides an isolated SethSettings instance pointing to ephemeral directories."""
    return SethSettings(
        src_dir=tmp_path / "src",
        project_root=tmp_path,
        registration_token="test-secret-token",
        allowed_api_user_ids="",
        vllm_url="http://mock-vllm:8000/v1",
        whisper_url="http://mock-whisper:8010/v1",
    )

class MockLLMProvider:
    """Mock implementation of LLMProvider protocol for deterministic testing."""
    def __init__(self, responses: Optional[List[List[StreamChunk]]] = None) -> None:
        self.responses = responses or []
        self.call_history: List[Dict[str, Any]] = []

    async def generate(self, messages: List[Message], tools=None, config=None) -> Message:
        return Message(role=Role.ASSISTANT, content="Mock non-streaming response")

    async def generate_stream(self, messages: List[Message], tools=None, config=None) -> AsyncIterator[StreamChunk]:
        self.call_history.append({"messages": messages, "tools": tools, "config": config})
        if self.responses:
            stream_chunks = self.responses.pop(0)
            for chunk in stream_chunks:
                yield chunk
        else:
            yield StreamChunk(event_type=StreamEventType.CONTENT, text="Hello from mock LLM!")

class MockSemanticMemory:
    """In-memory dictionary mock implementing SemanticMemory protocol."""
    def __init__(self) -> None:
        self.storage: Dict[str, List[str]] = {}

    async def search(self, user_id: str, query: str, limit: int = 10) -> List[str]:
        return self.storage.get(user_id, [])

    async def save(self, user_id: str, fact: str, response: str) -> bool:
        self.storage.setdefault(user_id, []).append(fact)
        return True

class MockGraphMemory:
    """In-memory mock implementing GraphMemory protocol."""
    def __init__(self) -> None:
        self.episodes: List[Dict[str, str]] = []

    async def add_episode(self, user_id: str, user_text: str, assistant_text: str) -> None:
        self.episodes.append({"user_id": user_id, "user": user_text, "assistant": assistant_text})

    async def query_relations(self, user_id: str, query: str) -> List[str]:
        return ["Entity A -> RELATED_TO -> Entity B"]
```

---

## 4. Layer-by-Layer Test Specifications

### 4.1 Domain Layer (`tests/unit/domain/`)

| Test Module | Key Assertions & Scenarios |
|---|---|
| `test_models.py` | • `Message.to_dict()` outputs exact OpenAI schema format with `role`, `content`, `tool_call_id`, and `tool_calls`.<br/>• `ToolCall` and `ToolResult` immutability (`frozen=True`).<br/>• `RegulatorState.interpolate()` smoothly transitions temperature, top_p, and presence_penalty according to factor $\alpha$.<br/>• `RegulatorPresets` return exact archetype constants (*Rigorous, Chaotic, Verbose*).<br/>• `SystemStatus.to_dict()` serializes telemetry fields and GPU lists correctly. |
| `test_protocols.py` | • Verify that `VllmClient`, `JsonlConversationHistory`, `Mem0SemanticMemory`, `GraphitiRelationalMemory`, `StableDiffusionGenerator`, `KokoroSynthesizer`, and `JsonSessionStore` satisfy their corresponding `typing.Protocol` contracts via structural typing checks. |
| `test_exceptions.py` | • Ensure all domain exceptions inherit from `SethDomainError`. |

---

### 4.2 Application Layer (`tests/unit/application/`)

#### A. ConversationOrchestrator (`test_orchestrator.py`)
* **Scenario 1: Single-Turn Streaming Completion (No Tools)**
  * Input: User prompt `"Explain Clean Architecture"`.
  * Verifies: Yields `REASONING` chunk, `CONTENT` chunk, and final `DONE` chunk.
  * Asserts: Short-term history is appended; background graph episode is scheduled; audit log is written.
* **Scenario 2: Multi-Hop Tool Execution Loop**
  * Hop 1: Mock LLM yields `TOOL_START` with tool `web_search` and query args.
  * Tool Execution: `ToolRegistry.execute()` is invoked concurrently with `asyncio.gather`.
  * Hop 2: Mock LLM receives tool output in context and yields final answer.
  * Verifies: Multi-hop loop executes 2 hops and terminates cleanly.
* **Scenario 3: Max Tool Hops Guardrail**
  * Setup: Mock LLM continuously requests tools in an infinite loop.
  * Verifies: Orchestrator halts after `max_tool_hops` (default: 5) and marks `hit_hop_limit=True` in audit record.
* **Scenario 4: Error Handling & Stream Recovery**
  * Setup: Mock LLM throws `InferenceEngineError`.
  * Verifies: Orchestrator catches error, yields `StreamEventType.ERROR`, and logs failure in audit record without crashing the server.

#### B. Dynamic Inference Regulator (`test_use_cases.py`)
* **Semantic Similarity Vector Modulation:**
  * Technical prompt (`"Optimize PostgreSQL index for B-Tree"`) shifts state toward **Rigorous** ($T \to 0.10$).
  * Creative prompt (`"Generate chaotic glitch poetry"`) shifts state toward **Chaotic** ($T \to 1.30$).
  * Philosophical prompt (`"What is the ontology of synthetic minds?"`) shifts state toward **Verbose** ($T \to 0.85$).
* **User State Persistence:**
  * Verifies that `UserStateManager` creates and persists `seth_<user_id>.state` atomically under concurrent locks.

#### C. Authentication & Telemetry Use Cases
* **`AuthenticateSessionUseCase`:**
  * Valid token $\to$ returns `status="ok"`, generates unique 32-character hex UUID, and persists to session store.
  * Invalid token $\to$ returns `status="error"`, `user_id=None`.
* **`CollectTelemetryUseCase`:**
  * Runs concurrent probes (`vllm`, `whisper`, `qdrant`, `neo4j`, `vram`) via `asyncio.gather` and handles individual probe timeouts gracefully.

---

### 4.3 Infrastructure Layer (`tests/unit/infrastructure/`)

#### A. LLM Adapters (`test_llm_accumulator.py`, `test_context_guard.py`, `test_vllm_client.py`)
* **`StreamAccumulator`:**
  * Reconstructs fragmented streaming tool call deltas (e.g. index 0 argument chunks `{"que`, `ry": "te`, `st"}`) into a complete valid JSON `ToolCall`.
  * Accumulates reasoning strings and content strings independently.
* **`ContextWindowGuard`:**
  * Accurately calculates available output tokens: `safe_tokens = min(requested, context_limit - input_tokens)`.
  * Prevents context overflows by enforcing minimum 256 token buffer.

#### B. Tool Discovery & Schema Reflection (`test_tool_registry.py`)
* **Decorator & Schema Generation:**
  * Verifies that methods marked with `@tool` are discovered.
  * Inspects `Annotated[str, "description"]`, `int`, `float`, `bool`, `Literal["a", "b"]`, and `Optional[T]` to produce compliant JSON Schema parameters.
* **Tool Dispatch & Safe Error Capture:**
  * Executes async and sync tool methods.
  * Catches invalid JSON arguments and returns descriptive `ToolResult(is_success=False)`.

#### C. In-Process Generators & VRAM Offload (`test_tools_concrete.py`)
* **`StableDiffusionGenerator` VRAM Timer:**
  * Mocking PyTorch and Diffusers pipeline.
  * Verifies that `generate_image()` schedules `_vram_cleanup_timer()`.
  * Verifies that consecutive calls cancel and reset the 300-second timer.
  * Verifies that timer expiration triggers pipeline unloading and GPU memory release.
* **`KokoroSynthesizer`:**
  * Mocks `KPipeline` generator and verifies MP3 audio export to `storage/audio/`.
* **`AstCodeInspector`:**
  * Runs AST parsing against existing Python source files and extracts class/function trees.

#### D. Persistent Memory & Telemetry Probes (`test_jsonl_history.py`, `test_telemetry_probes.py`)
* **`JsonlConversationHistory`:**
  * Reads and writes turns to `conversations/history_<user_id>.jsonl`.
  * Verifies sliding window eviction when message count exceeds `max_history`.
  * Tests concurrent writes with `asyncio.gather` ensuring zero file corruption under file locks.
* **`NvidiaSmiProbe`:**
  * Tests parsing of standard `nvidia-smi --query-gpu=... --format=csv` output strings.
  * Tests fallback when `nvidia-smi` binary is missing or exits with non-zero code.

---

### 4.4 Presentation & Interface Layer (`tests/unit/interfaces/`)

#### A. FastAPI Endpoints (`test_api_routes.py`)
* Uses `httpx.AsyncClient` with `ASGITransport` against `create_app(test_settings)`.
* **`POST /api/register`:** Tests status codes 200 (authorized) and 400 (bad token).
* **`POST /api/chat`:**
  * Header `X-Seth-User` missing $\to$ HTTP 401 Unauthorized.
  * Valid request $\to$ streams SSE lines formatted as `data: {...}\n\n`.
  * Tests multipart audio upload (mocking Whisper transcription) and image upload (encoding Base64).
* **`GET /api/status`:** Returns HTTP 200 with system status JSON.

#### B. Unified Async SDK (`test_client_sdk.py`)
* Uses `respx` to mock backend HTTP responses.
* Tests `register()`, `chat_stream()` (parsing SSE delta chunks), and `get_status()`.

#### C. Telegram Bridge (`test_telegram_bridge.py`)
* Mocks `telegram.Update` and `telegram.ext.ContextTypes`.
* Verifies `/start` greeting, token authentication handler, message splitting for texts $>4096$ characters preserving markdown code blocks, and audio/photo downloads.

---

## 5. Test Implementation Roadmap & Directory Setup

To execute this strategy, the following packages are integrated into `pyproject.toml`:

```toml
[project.optional-dependencies]
test = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "pytest-mock>=3.12.0",
    "pytest-cov>=4.1.0",
    "respx>=0.21.0",
]
```

### Pytest Configuration (`pytest.ini` / `pyproject.toml`):
```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --tb=short --cov=src --cov-report=term-missing --cov-report=html"
```

---

## 6. Execution Commands

```bash
# 1. Run full test suite with coverage report
pytest

# 2. Run only fast unit tests
pytest tests/unit

# 3. Run specific layer tests
pytest tests/unit/domain
pytest tests/unit/application
pytest tests/unit/infrastructure
pytest tests/unit/interfaces

# 4. Run tests with keyword filter
pytest -k "test_orchestrator"

# 5. Generate HTML coverage report (saved in htmlcov/)
pytest --cov=src --cov-report=html
```
