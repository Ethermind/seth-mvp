"""
ConversationOrchestrator: Core Multi-Hop LLM inference, memory injection, and tool dispatch loop.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
import logging
import re
from typing import Any, AsyncIterator, Dict, List, Optional
from uuid import uuid4

from src.application.use_cases.regulate import RegulateInferenceUseCase
from src.config.settings import SethSettings, get_settings
from src.domain.models import Message, Role, StreamChunk, StreamEventType
from src.domain.protocols import ConversationHistory, GraphMemory, LLMProvider, SemanticMemory
from src.infrastructure.logging.audit import ReasoningAuditLogger
from src.infrastructure.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class ConversationOrchestrator:
    """Orchestrates end-to-end conversation flow, memory retrieval, dynamic regulation, and multi-hop tools."""

    def __init__(
        self,
        llm: LLMProvider,
        tools: ToolRegistry,
        semantic_memory: SemanticMemory,
        graph_memory: GraphMemory,
        history_repo: ConversationHistory,
        regulator: RegulateInferenceUseCase,
        system_prompt: str,
        audit_logger: Optional[ReasoningAuditLogger] = None,
        settings: Optional[SethSettings] = None,
        max_tool_hops: int = 5,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.semantic_memory = semantic_memory
        self.graph_memory = graph_memory
        self.history_repo = history_repo
        self.regulator = regulator
        self.system_prompt = system_prompt
        self.settings = settings or get_settings()
        self.audit_logger = audit_logger or ReasoningAuditLogger(self.settings)
        self.max_tool_hops = max_tool_hops

    async def execute_stream(
        self,
        user_id: str,
        user_text: str,
        image_b64: Optional[str] = None,
    ) -> AsyncIterator[StreamChunk]:
        """
        Runs the full streaming pipeline:
        1. Context recovery
        2. Dynamic state calculation
        3. Semantic memory injection
        4. Multi-hop tool execution loop
        5. Asynchronous persistence and auditing
        """
        # 1. Recover short-term history
        history = await self.history_repo.get_history(user_id)

        # 2. Dynamically adjust inference configuration
        state = await self.regulator.adjust_for_query(user_id, user_text)

        # 3. Retrieve long-term memories
        memories = await self.semantic_memory.search(user_id, user_text)

        # 4. Construct messages payload
        messages = self._assemble_messages(history, user_text, image_b64, memories)
        tool_schemas = self.tools.get_openai_schemas()

        audit_record: Dict[str, Any] = {
            "audit_id": uuid4().hex[:8],
            "timestamp_start": datetime.now().isoformat(),
            "timestamp_end": None,
            "user_id": user_id,
            "query": user_text,
            "inference_config": state.to_dict(),
            "hops": [],
            "final_response": None,
            "hop_count": 0,
            "hit_hop_limit": False,
            "error": None,
        }

        tool_calls_used: List[str] = []
        generated_media: List[Dict[str, str]] = []
        final_content = ""

        try:
            for hop in range(self.max_tool_hops):
                logger.info("🧠 [HOP %d/%d] Executing LLM generation (user: %s)...", hop + 1, self.max_tool_hops, user_id)
                hop_tool_requests = []
                hop_reasoning = ""
                hop_content = ""
                assistant_msg_dict: Dict[str, Any] = {}

                async for chunk in self.llm.generate_stream(messages, tool_schemas, state):
                    if chunk.event_type == StreamEventType.REASONING:
                        hop_reasoning += chunk.text
                        yield chunk
                    elif chunk.event_type == StreamEventType.CONTENT:
                        hop_content += chunk.text
                        yield chunk
                    elif chunk.event_type == StreamEventType.TOOL_START:
                        if "tool_call" in chunk.metadata:
                            hop_tool_requests.append(chunk.metadata["tool_call"])
                        if "assistant_message_dict" in chunk.metadata:
                            assistant_msg_dict = chunk.metadata["assistant_message_dict"]
                    elif chunk.event_type == StreamEventType.ERROR:
                        yield chunk
                        audit_record["error"] = chunk.text
                        return

                final_content = hop_content
                audit_record["hop_count"] = hop + 1
                audit_record["hops"].append({
                    "hop_number": hop + 1,
                    "reasoning": hop_reasoning or None,
                    "tool_calls_requested": [tc.name for tc in hop_tool_requests] if hop_tool_requests else None,
                    "tool_results": None,
                })

                if not hop_tool_requests:
                    # Model completed turn without requesting tools
                    break

                # Append the assistant message with tool_calls to the context
                if assistant_msg_dict:
                    messages.append(
                        Message(
                            role=Role.ASSISTANT,
                            content=assistant_msg_dict.get("content", ""),
                            tool_calls=hop_tool_requests,
                        )
                    )

                # Execute requested tools concurrently
                for tc in hop_tool_requests:
                    tool_calls_used.append(tc.name)
                    yield StreamChunk(event_type=StreamEventType.TOOL_START, tool_name=tc.name)

                tool_results = await asyncio.gather(
                    *(self.tools.execute(tc.name, tc.arguments, user_id=user_id, call_id=tc.call_id) for tc in hop_tool_requests)
                )

                audit_record["hops"][-1]["tool_results"] = [r.output for r in tool_results]

                for result in tool_results:
                    yield StreamChunk(event_type=StreamEventType.TOOL_END, tool_name=result.name, ok=result.is_success)
                    messages.append(
                        Message(
                            role=Role.TOOL,
                            content=result.output,
                            tool_call_id=result.call_id,
                        )
                    )
                    # Extract media from tool execution output if present
                    if result.is_success and result.output:
                        self._collect_media_from_tool_output(result.output, generated_media)

            if hop_tool_requests and audit_record["hop_count"] >= self.max_tool_hops:
                audit_record["hit_hop_limit"] = True
                logger.warning("⚠️ Tool hop limit reached (%d hops).", self.max_tool_hops)

            audit_record["final_response"] = final_content

            # 5. Persist to short and graph memory
            if final_content:
                await self.history_repo.append_turn(user_id, user_text, final_content)
                asyncio.create_task(self.graph_memory.add_episode(user_id, user_text, final_content))

            # 6. Extract media references and combine with tool generated media
            text_media = self._extract_media_refs(final_content)
            all_media = self._merge_media(generated_media, text_media)

            yield StreamChunk(
                event_type=StreamEventType.DONE,
                text=final_content,
                metadata={
                    "media": all_media,
                    "tool_calls_used": tool_calls_used,
                    "hop_count": audit_record["hop_count"],
                },
            )

        except Exception as e:
            logger.exception("❌ Error during conversation stream: %s", e)
            audit_record["error"] = str(e)
            yield StreamChunk(event_type=StreamEventType.ERROR, text=str(e), ok=False)

        finally:
            audit_record["timestamp_end"] = datetime.now().isoformat()
            self.audit_logger.log_turn(audit_record)

    def _assemble_messages(
        self,
        history: List[Message],
        user_text: str,
        image_b64: Optional[str],
        memories: List[str],
    ) -> List[Message]:
        system_content = self.system_prompt
        if memories:
            system_content += "\n\n<MEMORY>\n" + "\n".join(f"- {m}" for m in memories) + "\n</MEMORY>"

        messages = [Message(role=Role.SYSTEM, content=system_content)]
        messages.extend(history)

        if image_b64:
            user_content = [
                {"type": "text", "text": user_text or "Describe the content of this image."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            ]
        else:
            user_content = user_text

        messages.append(Message(role=Role.USER, content=user_content))
        return messages

    def _extract_media_refs(self, response_text: str) -> List[Dict[str, str]]:
        """Extracts media references mentioned in text by regex."""
        media = []
        if not response_text:
            return media

        for match in re.finditer(r"(?:storage/)?images/([\w\-_]+\.png)", response_text):
            filename = match.group(1)
            fs_path = self.settings.storage_images_dir / filename
            if fs_path.exists():
                media.append({"type": "image", "path": str(fs_path.resolve()), "url": f"/storage/images/{filename}"})

        for match in re.finditer(r"(?:storage/)?audio/([\w\-_]+\.mp3)", response_text):
            filename = match.group(1)
            fs_path = self.settings.storage_audio_dir / filename
            if fs_path.exists():
                media.append({"type": "audio", "path": str(fs_path.resolve()), "url": f"/storage/audio/{filename}"})

        return media

    def _collect_media_from_tool_output(self, raw_output: str, media_list: List[Dict[str, str]]) -> None:
        """Parses tool output JSON and extracts generated image or audio file paths."""
        try:
            import json
            from pathlib import Path
            data = json.loads(raw_output)
            if isinstance(data, dict):
                local_path = data.get("local_path") or data.get("path")
                if local_path:
                    p = Path(local_path).resolve()
                    if p.exists():
                        ext = p.suffix.lower()
                        if ext in (".png", ".jpg", ".jpeg", ".webp"):
                            media_list.append({
                                "type": "image",
                                "path": str(p),
                                "url": f"/storage/images/{p.name}",
                            })
                        elif ext in (".mp3", ".wav", ".ogg"):
                            media_list.append({
                                "type": "audio",
                                "path": str(p),
                                "url": f"/storage/audio/{p.name}",
                            })
        except Exception:
            pass

    def _merge_media(self, tool_media: List[Dict[str, str]], text_media: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Merges and deduplicates media items from tool outputs and text references."""
        seen_paths = set()
        merged = []
        for item in tool_media + text_media:
            p = item.get("path")
            if p and p not in seen_paths:
                seen_paths.add(p)
                merged.append(item)
        return merged
