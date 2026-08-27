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

from .models import Backend


@dataclass(slots=True)
class RenderSpec:
    prompt: str
    references: list[Path]
    width: int
    height: int
    transparent: bool
    lora_name: str | None = None
    seed: int | None = None


@dataclass(slots=True)
class RenderedImage:
    png: bytes
    estimated_cost_usd: float
    provider_request_id: str | None = None


class ArtProvider(ABC):
    backend: Backend

    @abstractmethod
    def render(self, spec: RenderSpec) -> RenderedImage: ...


class HuggingFaceSpaceProvider(ArtProvider):
    backend = Backend.HUGGING_FACE

    def __init__(self, url: str | None = None, token: str | None = None):
        self.url = (url or os.getenv("HF_SPACE_URL", "")).rstrip("/")
        self.token = token or os.getenv("HF_API_TOKEN") or os.getenv("HF_TOKEN")

    def render(self, spec: RenderSpec) -> RenderedImage:
        if not self.url or not spec.lora_name:
            raise ValueError("Hugging Face generation requires HF_SPACE_URL and --lora-name.")
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        payload = {
            "prompt": spec.prompt,
            "lora_name": spec.lora_name,
            "width": spec.width,
            "height": spec.height,
            "steps": 16 if max(spec.width, spec.height) <= 512 else 28,
            "guidance_scale": 4.0,
            "lora_scale": 0.8,
            "seed": spec.seed,
            "remove_background": spec.transparent,
        }
        response = httpx.post(f"{self.url}/v1/generate", json=payload, headers=headers, timeout=600)
        response.raise_for_status()
        body = response.json()
        return RenderedImage(
            base64.b64decode(body["image_base64"]), 0.0, response.headers.get("x-request-id")
        )


class GPTImage2Provider(ArtProvider):
    backend = Backend.GPT_IMAGE_2

    def render(self, spec: RenderSpec) -> RenderedImage:
        if not spec.references:
            raise ValueError(
                "GPT Image 2 requires at least one reference image for style preservation."
            )
        from openai import OpenAI

        client = OpenAI()
        # The edits endpoint accepts image inputs; only this game's bounded reference list is sent.
        with ExitStack() as stack:
            handles = [stack.enter_context(open(path, "rb")) for path in spec.references]
            response = client.images.edit(
                model="gpt-image-2",
                image=handles,
                prompt=spec.prompt,
                size=f"{spec.width}x{spec.height}",
                quality="low" if max(spec.width, spec.height) <= 512 else "high",
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
