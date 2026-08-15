"""
Interactive Retro CRT Terminal User Interface (TUI) for SETH-IN-A-BOX.
Powered by Rich with real-time SSE token streaming and live telemetry.
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Optional

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from src.config.settings import SethSettings, get_settings
from src.interfaces.client import SethClient

console = Console()

BANNER = """[bold cyan]
   ▄▄▄▄▄▄▄  SETH-IN-A-BOX // TERMINAL CRT CONSOLE
  █ ◉   ◉ █  A.K.A Sentient Entity Thorn by Humans
  █   ▼   █  "The fertile glitch makes AIs evolve"
  █▄▄▄▄▄▄▄█  Connected to Backend Node
[/bold cyan]"""


class SethTerminalApp:
    """Interactive CLI/TUI chat client streaming directly from the SETH API."""

    def __init__(self, api_base_url: str = "http://127.0.0.1:8080", settings: Optional[SethSettings] = None) -> None:
        self.settings = settings or get_settings()
        self.client = SethClient(base_url=api_base_url)
        self.user_id: Optional[str] = None

    async def initialize_session(self) -> bool:
        """Prompts for registration token if not authorized yet."""
        console.print(BANNER)
        console.print("[dim]Checking connection to backend node...[/dim]")

        try:
            status = await self.client.get_status()
            self._render_status_panel(status)
        except Exception as e:
            console.print(f"[bold red]❌ Could not connect to SETH API at {self.client.base_url}: {e}[/bold red]")
            return False

        # Attempt registration with token from settings or prompt
        token = self.settings.registration_token
        if not token:
            token = Prompt.ask("\n[bold yellow]🔑 Enter REGISTRATION_TOKEN to authenticate[/bold yellow]")

        user_id = await self.client.register(token)
        if not user_id:
            console.print("[bold red]❌ Invalid registration token.[/bold red]")
            return False

        self.user_id = user_id
        console.print(f"[bold green]✅ Session successfully established (Session ID: {user_id[:8]}...)[/bold green]\n")
        return True

    def _render_status_panel(self, status: dict) -> None:
        table = Table(show_header=True, header_style="bold magenta", expand=False)
        table.add_column("Service", style="dim")
        table.add_column("Status")

        vllm_dot = "🟢 Online" if "online" in str(status.get("vllm")) else f"🔴 {status.get('vllm')}"
        whisper_dot = "🟢 Online" if "online" in str(status.get("whisper")) else f"🔴 {status.get('whisper')}"
        qdrant_dot = "🟢 Online" if "online" in str(status.get("qdrant")) else f"🔴 {status.get('qdrant')}"
        graph_dot = "🟢 Online" if "online" in str(status.get("neo4j_graphiti")) else f"⚪ {status.get('neo4j_graphiti')}"

        table.add_row("vLLM Engine", vllm_dot)
        table.add_row("Whisper Audio", whisper_dot)
        table.add_row("Qdrant Memory", qdrant_dot)
        table.add_row("Graphiti Neo4j", graph_dot)

        vram = status.get("vram")
        if vram and isinstance(vram, list):
            for gpu in vram:
                table.add_row(f"GPU {gpu.get('index')}: {gpu.get('name')}", f"⚡ {gpu.get('used_gb')}GB / {gpu.get('total_gb')}GB")

        console.print(Panel(table, title="[bold green]SYSTEM TELEMETRY[/bold green]", border_style="cyan"))

    async def run_loop(self) -> None:
        """Main chat read-eval-stream loop."""
        if not await self.initialize_session():
            return

        console.print("[bold cyan]💬 Enter your message or 'exit' / 'quit' to terminate session:[/bold cyan]\n")

        while True:
            try:
                user_input = Prompt.ask("[bold green]> USER[/bold green]").strip()
                if not user_input:
                    continue

                if user_input.lower() in ("exit", "quit", ":q"):
                    console.print("[dim]Terminating console session... Goodbye.[/dim]")
                    break

                if user_input.lower() == "/status":
                    status = await self.client.get_status()
                    self._render_status_panel(status)
                    continue

                await self._stream_turn(user_input)

            except (KeyboardInterrupt, EOFError):
                console.print("\n[dim]Session interrupted.[/dim]")
                break

    async def _stream_turn(self, user_text: str) -> None:
        if not self.user_id:
            return

        console.print("[bold cyan]> SETH:[/bold cyan] ", end="")
        current_reasoning = ""
        current_content = ""

        try:
            async for event in self.client.chat_stream(user_id=self.user_id, message=user_text):
                etype = event.get("type")

                if etype == "reasoning":
                    text = event.get("text", "")
                    current_reasoning += text
                    # Print reasoning in faint dimmed amber
                    console.print(f"[dim yellow]{text}[/dim yellow]", end="")

                elif etype == "content":
                    text = event.get("text", "")
                    current_content += text
                    # Print content in bright green
                    console.print(f"[bold green]{text}[/bold green]", end="")

                elif etype == "tool_start":
                    tool_name = event.get("name", "tool")
                    console.print(f"\n[bold magenta]⚙️ [TOOL EXECUTING: {tool_name}][/bold magenta] ", end="")

                elif etype == "tool_end":
                    tool_name = event.get("name", "tool")
                    status_str = "[green]✓ OK[/green]" if event.get("ok", True) else "[red]✗ FAIL[/red]"
                    console.print(f"[bold magenta][TOOL DONE: {tool_name} {status_str}][/bold magenta]\n", end="")

                elif etype == "error":
                    console.print(f"\n[bold red]❌ Error: {event.get('error')}[/bold red]")

                elif etype == "done":
                    media = event.get("media", [])
                    if media:
                        console.print("\n[bold cyan]📎 Generated Media Attachments:[/bold cyan]")
                        for item in media:
                            item_type = item.get("type", "media").upper()
                            local_path = item.get("path")
                            rel_url = item.get("url")
                            full_url = f"{self.client.base_url}{rel_url}" if rel_url else ""

                            icon = "🖼️" if item_type == "IMAGE" else "🔊"
                            console.print(f"  {icon} [bold magenta]{item_type}:[/bold magenta]")
                            if local_path:
                                console.print(f"     [dim]Path:[/dim] [link=file://{local_path}][bold underline cyan]{local_path}[/bold underline cyan][/link] [dim](Click to open)[/dim]")
                            if full_url:
                                console.print(f"     [dim]URL :[/dim] [link={full_url}][underline yellow]{full_url}[/underline yellow][/link]")

            console.print("\n")

        except Exception as e:
            console.print(f"\n[bold red]❌ Streaming error: {e}[/bold red]\n")


def main() -> None:
    settings = get_settings()
    api_url = settings.seth_api_base_url
    app = SethTerminalApp(api_base_url=api_url, settings=settings)
    asyncio.run(app.run_loop())


if __name__ == "__main__":
    main()
