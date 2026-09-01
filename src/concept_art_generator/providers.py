from __future__ import annotations

import base64
import os
from abc import ABC, abstractmethod
from contextlib import ExitStack
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import httpx
from PIL import Image

from .models import DEFAULT_BACKGROUND_MODEL, Backend


@dataclass(slots=True)
class RenderSpec:
    prompt: str
    references: list[Path]
    width: int
    height: int
    transparent: bool
    lora_name: str | None = None
    seed: int | None = None
    negative_prompt: str = ""
    steps: int = 28
    guidance_scale: float = 4.0
    lora_scale: float = 1.25
    scheduler: str | None = None
    base_model: str | None = None
    background_model: str = DEFAULT_BACKGROUND_MODEL


@dataclass(slots=True)
class RenderedImage:
    png: bytes
    estimated_cost_usd: float
    provider_request_id: str | None = None
    generation_parameters: dict | None = None


class ArtProvider(ABC):
    backend: Backend

    @abstractmethod
    def render(self, spec: RenderSpec) -> RenderedImage: ...


class HuggingFaceSpaceProvider(ArtProvider):
    backend = Backend.HUGGING_FACE

    def __init__(self, url: str | None = None, token: str | None = None):
        self.url = (url or os.getenv("HF_SPACE_URL", "")).rstrip("/")
        self.token = token or os.getenv("HF_TOKEN")

    def render(self, spec: RenderSpec) -> RenderedImage:
        if not self.url or not spec.lora_name:
            raise ValueError("Hugging Face generation requires HF_SPACE_URL and --lora-name.")
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        upscale_to_2k = max(spec.width, spec.height) >= 2048
        # The Space's Swin2SR checkpoint performs a native x2 upscale. Final
        # requests therefore render at half size before neural super-resolution.
        request_width = (spec.width + 1) // 2 if upscale_to_2k else spec.width
        request_height = (spec.height + 1) // 2 if upscale_to_2k else spec.height
        payload = {
            "prompt": spec.prompt,
            "lora_name": spec.lora_name,
            "negative_prompt": spec.negative_prompt,
            "width": request_width,
            "height": request_height,
            "steps": spec.steps,
            "guidance_scale": spec.guidance_scale,
            "lora_scale": spec.lora_scale,
            "seed": spec.seed,
            "scheduler": spec.scheduler,
            "base_model": spec.base_model,
            "remove_background": spec.transparent,
            # Name the segmentation model explicitly; the Space rejects an unknown one with a 400.
            "background_model": spec.background_model,
            "upscale_to_2k": upscale_to_2k,
        }
        response = httpx.post(
            f"{self.url}/v1/generate",
            json=payload,
            headers=headers,
            timeout=900,
        )
        response.raise_for_status()
        result = response.json()
        image_b64 = result.get("image_base64") if isinstance(result, dict) else None
        if not isinstance(image_b64, str) or not image_b64:
            raise RuntimeError("Hugging Face Space returned no base64 PNG.")
        parameters = result.get("generation_parameters")
        if not isinstance(parameters, dict) or not isinstance(parameters.get("seed"), int):
            raise RuntimeError(  # noqa: TRY004 - malformed external provider response
                "Hugging Face Space returned no replayable generation parameters."
            )
        return RenderedImage(
            base64.b64decode(image_b64),
            0.0,
            response.headers.get("x-request-id"),
            parameters,
        )

class GPTImage2Provider(ArtProvider):
    backend = Backend.GPT_IMAGE_2

    def render(self, spec: RenderSpec) -> RenderedImage:
        if not spec.references:
            raise ValueError(
                "GPT Image 2 requires at least one reference image for style preservation."
            )
        if len(spec.references) > 16:
            raise ValueError("GPT Image 2 accepts at most 16 reference images.")
        from openai import OpenAI

        client = OpenAI()
        is_draft = max(spec.width, spec.height) <= 512
        # GPT Image 2 cannot render a 512x512 output. Keep the workflow's draft
        # intent, but request and retain the model's supported 1024px square draft.
        request_size = "1024x1024" if is_draft else f"{spec.width}x{spec.height}"
        # The edits endpoint accepts image inputs; only this game's bounded reference list is sent.
        with ExitStack() as stack:
            handles = [stack.enter_context(open(path, "rb")) for path in spec.references]
            response = client.images.edit(
                model="gpt-image-2",
                image=handles,
                prompt=spec.prompt,
                size=request_size,
                quality="low" if is_draft else "high",
                background="transparent" if spec.transparent else "auto",
                response_format="b64_json",
            )
        image_b64 = response.data[0].b64_json
        if not image_b64:
            raise RuntimeError("GPT Image 2 did not return base64 image data.")
        usage = getattr(response, "usage", None)
        # Providers do not expose a stable dollar field; retain usage in logs and avoid invented spend.
        _ = usage
        return RenderedImage(
            base64.b64decode(image_b64), 0.0, getattr(response, "_request_id", None)
        )


class DeterministicProvider(ArtProvider):
    """Offline provider for demos/tests; never selected by CLI/UI."""

    backend = Backend.GPT_IMAGE_2

    def render(self, spec: RenderSpec) -> RenderedImage:
        image = Image.new(
            "RGBA", (spec.width, spec.height), (39, 55, 79, 0 if spec.transparent else 255)
        )
        buffer = BytesIO()
        image.save(buffer, "PNG")
        return RenderedImage(buffer.getvalue(), 0.0, "offline")
