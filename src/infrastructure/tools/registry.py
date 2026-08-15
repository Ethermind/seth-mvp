"""
Unified Tool Discovery, Schema Reflection, and Execution Registry.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, fields, is_dataclass, MISSING
from datetime import date, datetime, time as _dt_time
import enum
from functools import lru_cache
import inspect
import json
import logging
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)
from uuid import UUID

from pydantic import BaseModel
from src.domain.models import ToolResult

logger = logging.getLogger(__name__)


def tool(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to mark an instance method as an LLM-invocable tool."""
    func.__tool__ = True  # type: ignore[attr-defined]
    return func


@dataclass(slots=True)
class RegisteredTool:
    name: str
    func: Callable[..., Any]
    description: str
    schema: Dict[str, Any]


class ToolRegistry:
    """Discovers, validates, and dispatches tool calls from LLM function calling."""

    def __init__(self) -> None:
        self.tools: Dict[str, RegisteredTool] = {}

    def register_instance(self, instance: Any) -> None:
        """Inspects and registers all methods marked with @tool on an instance."""
        for name, method in inspect.getmembers(instance, predicate=inspect.ismethod):
            if not getattr(method.__func__, "__tool__", False):
                continue

            description = inspect.getdoc(method) or ""
            schema = self._build_schema_from_signature(method)

            self.tools[name] = RegisteredTool(
                name=name,
                func=method,
                description=description,
                schema=schema,
            )
            logger.info("🛠️ [TOOL REGISTRY] Auto-registered tool: '%s'", name)

    def register_instances(self, instances: List[Any]) -> None:
        for inst in instances:
            self.register_instance(inst)

    def get_openai_schemas(self) -> List[Dict[str, Any]]:
        """Returns the OpenAI / vLLM tools payload format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool_obj.name,
                    "description": tool_obj.description,
                    "parameters": tool_obj.schema,
                },
            }
            for tool_obj in self.tools.values()
        ]

    async def execute(self, name: str, raw_arguments: str, user_id: Optional[str] = None, call_id: str = "") -> ToolResult:
        """Safely executes a tool call and captures errors into a ToolResult."""
        if name not in self.tools:
            return ToolResult(
                call_id=call_id or f"call_{name}",
                name=name,
                output=f"Error: Tool '{name}' is not registered.",
                is_success=False,
            )

        try:
            kwargs = json.loads(raw_arguments) if raw_arguments.strip() else {}
        except json.JSONDecodeError as err:
            return ToolResult(
                call_id=call_id or f"call_{name}",
                name=name,
                output=f"Error: Arguments for '{name}' were not valid JSON ({err}). Re-emit well-formed JSON.",
                is_success=False,
            )

        tool_obj = self.tools[name]
        logger.info("🛠️ Executing tool: %s with args: %s", name, kwargs)

        try:
            fn = tool_obj.func
            # If function expects user_id and it's not provided in kwargs, inject it
            sig = inspect.signature(fn)
            if "user_id" in sig.parameters and "user_id" not in kwargs and user_id:
                kwargs["user_id"] = user_id

            res = fn(**kwargs)
            if asyncio.iscoroutine(res):
                res = await res

            output_str = str(res)
            return ToolResult(
                call_id=call_id or f"call_{name}",
                name=name,
                output=output_str,
                is_success=not output_str.startswith("Error"),
            )

        except Exception as exc:
            logger.exception("Error executing tool '%s': %s", name, exc)
            return ToolResult(
                call_id=call_id or f"call_{name}",
                name=name,
                output=f"Error executing tool: {str(exc)}",
                is_success=False,
            )

    def _build_schema_from_signature(self, method: Callable[..., Any]) -> Dict[str, Any]:
        hints = get_type_hints(method, include_extras=True)
        properties = {}
        required = []

        for name, param in inspect.signature(method).parameters.items():
            schema = self._parameter_schema(name, hints)
            if schema is None:
                continue

            properties[name] = schema
            if param.default is inspect.Parameter.empty:
                required.append(name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    def _parameter_schema(self, name: str, hints: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if name in ("self", "cls"):
            return None

        hint = hints.get(name)
        if hint is None or get_origin(hint) is not Annotated:
            return None

        base_type, *metadata = get_args(hint)
        schema = self._schema_for_type(base_type)
        if metadata:
            schema["description"] = str(metadata[0])

        return schema

    @staticmethod
    @lru_cache(maxsize=256)
    def _schema_for_type(tp: Any) -> Dict[str, Any]:
        origin = get_origin(tp)
        args = get_args(tp)

        primitives = {str: "string", int: "integer", float: "number", bool: "boolean"}
        if tp in primitives:
            return {"type": primitives[tp]}

        if tp is Any:
            return {}

        if tp is UUID:
            return {"type": "string", "format": "uuid"}

        if tp is Path:
            return {"type": "string"}

        if tp is datetime:
            return {"type": "string", "format": "date-time"}

        if tp is date:
            return {"type": "string", "format": "date"}

        if tp is _dt_time:
            return {"type": "string", "format": "time"}

        if inspect.isclass(tp) and issubclass(tp, enum.Enum):
            values = [m.value for m in tp]
            if not values:
                return {"type": "string"}
            schema = dict(ToolRegistry._schema_for_type(type(values[0])))
            schema["enum"] = values
            return schema

        if origin is Literal:
            values = list(args)
            if not values:
                return {"type": "string"}
            schema = dict(ToolRegistry._schema_for_type(type(values[0])))
            schema["enum"] = values
            return schema

        if origin is Union:
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1:
                schema = dict(ToolRegistry._schema_for_type(non_none[0]))
                schema["nullable"] = True
                return schema
            return {"anyOf": [ToolRegistry._schema_for_type(a) for a in non_none]}

        if origin in (list, tuple, set):
            return {
                "type": "array",
                "items": ToolRegistry._schema_for_type(args[0] if args else str),
            }

        if origin is dict:
            value_type = args[1] if len(args) == 2 else Any
            return {
                "type": "object",
                "additionalProperties": ToolRegistry._schema_for_type(value_type),
            }

        if inspect.isclass(tp) and is_dataclass(tp):
            hints = get_type_hints(tp)
            return {
                "type": "object",
                "properties": {f.name: ToolRegistry._schema_for_type(hints.get(f.name, Any)) for f in fields(tp)},
                "required": [f.name for f in fields(tp) if f.default is MISSING and f.default_factory is MISSING],
            }

        try:
            if inspect.isclass(tp) and issubclass(tp, BaseModel):
                if hasattr(tp, "model_json_schema"):
                    return tp.model_json_schema()
                return tp.schema()
        except Exception:
            pass

        return {"type": "string"}
