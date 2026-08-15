"""
Local image generation adapter using Stable Diffusion and Diffusers with VRAM auto-unload.
Implements ImageGenerator protocol and registers @tool methods.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import time
from typing import Annotated, Any, Dict, Optional

from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
import torch
from src.config.settings import SethSettings, get_settings
from src.infrastructure.tools.registry import tool

logger = logging.getLogger(__name__)


class StableDiffusionGenerator:
    """Generates images using local Stable Diffusion weights with automatic GPU idle cleanup."""

    def __init__(self, settings: SethSettings | None = None, idle_timeout_secs: int = 300) -> None:
        self.settings = settings or get_settings()
        self.idle_timeout_secs = idle_timeout_secs
        os.makedirs(self.settings.storage_images_dir, exist_ok=True)

        gpus = torch.cuda.device_count()
        self.device = torch.device("cuda:1") if gpus > 1 else torch.device("cuda:0")
        logger.info("🎨 [IMAGE SYSTEM INIT] Device: %s", self.device)

        self._pipe: Optional[StableDiffusionPipeline] = None
        self._cleanup_task: Optional[asyncio.Task[None]] = None
        self._lock = asyncio.Lock()

    def _init_pipeline(self) -> StableDiffusionPipeline:
        if self._pipe is None:
            model_path = str(self.settings.image_model_full_path)
            logger.info("⏳ Loading Image Pipeline from [%s] on %s...", model_path, self.device)
            pipe = StableDiffusionPipeline.from_single_file(
                model_path,
                torch_dtype=torch.float16,
                use_safetensors=True,
                safety_checker=None,
                requires_safety_checker=False,
            )
            pipe = pipe.to(self.device)
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config,
                use_karras_sigmas=True,
            )
            pipe.enable_attention_slicing()
            self._pipe = pipe
        return self._pipe

    def _sync_generate(self, prompt: str) -> str:
        pipe = self._init_pipeline()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gen_{timestamp}_{int(time.time()) % 10000}.png"
        output_path = os.path.join(self.settings.storage_images_dir, filename)

        logger.info("🚀 Rendering image from: '%s'", prompt)
        image = pipe(
            prompt=prompt,
            negative_prompt="bad anatomy, blurry, low quality, deformed, bad hands, mutated, disfigured",
            num_inference_steps=25,
            guidance_scale=7.5,
            width=512,
            height=512,
        ).images[0]

        image.save(output_path)
        return output_path

    async def generate_image(self, prompt: str) -> str:
        """Generates image file asynchronously and resets the VRAM idle timer."""
        async with self._lock:
            if self._cleanup_task and not self._cleanup_task.done():
                self._cleanup_task.cancel()

            try:
                file_path = await asyncio.to_thread(self._sync_generate, prompt)
                self._cleanup_task = asyncio.create_task(self._vram_cleanup_timer())
                return file_path
            except Exception as e:
                self._cleanup_task = asyncio.create_task(self._vram_cleanup_timer())
                raise e

    @tool
    async def create_image(
        self,
        prompt: Annotated[str, (
            "Expanded context-aware English prompt in comma-separated tag format. "
            "Example: '1girl, floating particles, void atmosphere, dark ambient, baroque, masterwork'"
        )],
    ) -> str:
        """
        Use this tool when requested to create, draw, or visualize images.
        ROLE: Creative Art Director. Expand user request into an English Booru-style tag list.
        """
        logger.info("🖼️ Image request received with prompt: '%s'", prompt)
        try:
            file_path = await self.generate_image(prompt)
            report = {
                "status": "success",
                "local_path": file_path,
                "message": f"Image generated successfully. File saved locally at {file_path}. Inform the user that the image is now available.",
            }
            return json.dumps(report, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False)

    async def _vram_cleanup_timer(self) -> None:
        try:
            await asyncio.sleep(self.idle_timeout_secs)
            async with self._lock:
                if self._pipe is not None:
                    self._pipe = None
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    logger.info("♻️ [VRAM CLEANUP] 5 min of inactivity reached. Unloaded Image Pipeline from %s", self.device)
        except asyncio.CancelledError:
            logger.info("♻️ [VRAM CLEANUP] Timer reset due to incoming request.")
