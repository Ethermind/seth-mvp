"""
Unit tests for ToolRegistry (src.infrastructure.tools.registry).
"""

from __future__ import annotations

from typing import Annotated, Literal
import pytest

from src.infrastructure.tools.registry import ToolRegistry, tool


class MockService:
    @tool
    def sum_two(
        self,
        a: Annotated[int, "First number"],
        b: Annotated[int, "Second number"],
    ) -> int:
        """Sums two numbers."""
        return a + b

    @tool
    async def greet(
        self,
        name: Annotated[str, "Target name"],
        style: Annotated[Literal["short", "long"], "Greeting style"] = "short",
    ) -> str:
        """Greets someone."""
        if style == "long":
            return f"Hello there, {name}! Welcome."
        return f"Hi {name}."


def test_tool_registry_schemas():
    registry = ToolRegistry()
    registry.register_instance(MockService())

    schemas = registry.get_openai_schemas()
    assert len(schemas) == 2

    sum_tool = next(s for s in schemas if s["function"]["name"] == "sum_two")
    assert sum_tool["function"]["description"] == "Sums two numbers."
    props = sum_tool["function"]["parameters"]["properties"]
    assert props["a"]["description"] == "First number"
    assert props["b"]["description"] == "Second number"


@pytest.mark.anyio
async def test_tool_registry_execution():
    registry = ToolRegistry()
    registry.register_instance(MockService())

    # Sync tool
    res1 = await registry.execute("sum_two", '{"a": 10, "b": 20}')
    assert res1.is_success is True
    assert res1.output == "30"

    # Async tool
    res2 = await registry.execute("greet", '{"name": "Luis", "style": "long"}')
    assert res2.is_success is True
    assert res2.output == "Hello there, Luis! Welcome."


@pytest.mark.anyio
async def test_tool_registry_errors():
    registry = ToolRegistry()
    registry.register_instance(MockService())

    # Invalid JSON
    res = await registry.execute("sum_two", "{bad_json")
    assert res.is_success is False
    assert "not valid JSON" in res.output

    # Unregistered tool
    res_unknown = await registry.execute("missing_tool", "{}")
    assert res_unknown.is_success is False
    assert "is not registered" in res_unknown.output
