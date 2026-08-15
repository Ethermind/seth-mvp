"""Tools and tool registry package for SETH."""

from src.infrastructure.tools.registry import ToolRegistry, tool
from src.infrastructure.tools.web_search import Crawl4AiSearcher
from src.infrastructure.tools.image_diffusion import StableDiffusionGenerator
from src.infrastructure.tools.speech_kokoro import KokoroSynthesizer
from src.infrastructure.tools.code_inspector import AstCodeInspector

__all__ = [
    "ToolRegistry",
    "tool",
    "Crawl4AiSearcher",
    "StableDiffusionGenerator",
    "KokoroSynthesizer",
    "AstCodeInspector",
]
