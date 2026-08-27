from __future__ import annotations

import base64
import json
import os
from abc import ABC, abstractmethod
from contextlib import ExitStack
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from urllib.parse import urljoin, urlparse

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
            "negative_prompt": "",
            "width": request_width,
            "height": request_height,
            "steps": 16 if max(spec.width, spec.height) <= 512 else 28,
            "guidance": 4.0,
            "scale": 0.8,
            "seed": spec.seed,
            "remove_background": spec.transparent,
            "upscale_to_2k": upscale_to_2k,
        }
        submission = httpx.post(
            f"{self.url}/gradio_api/call/v2/generate_ui",
            json=payload,
            headers=headers,
            timeout=60,
        )
        submission.raise_for_status()
        event_id = submission.json().get("event_id")
        if not isinstance(event_id, str) or not event_id:
            raise RuntimeError("Hugging Face Space did not return a generation event ID.")

        result = None
        event_name = None
        with httpx.stream(
            "GET",
            f"{self.url}/gradio_api/call/generate_ui/{event_id}",
            headers=headers,
            timeout=900,
        ) as events:
            events.raise_for_status()
            for line in events.iter_lines():
                if line.startswith("event:"):
                    event_name = line.partition(":")[2].strip()
                elif line.startswith("data:") and event_name == "error":
                    error = self._event_error(line.partition(":")[2].strip())
                    raise RuntimeError(f"Hugging Face Space generation failed: {error}")
                elif line.startswith("data:") and event_name == "complete":
                    result = json.loads(line.partition(":")[2].strip())
                    break
        if not isinstance(result, list) or not result or not isinstance(result[0], dict):
            raise RuntimeError("Hugging Face Space closed without a valid completed image result.")

        image_url = result[0].get("url")
        if not isinstance(image_url, str) or not image_url:
            raise RuntimeError("Hugging Face Space completed without an image URL.")
        image_url = urljoin(f"{self.url}/", image_url)
        if urlparse(image_url).netloc != urlparse(self.url).netloc:
            raise RuntimeError("Hugging Face Space returned an image URL on an unexpected host.")
        image_response = httpx.get(image_url, headers=headers, timeout=120, follow_redirects=True)
        image_response.raise_for_status()
        return RenderedImage(image_response.content, 0.0, event_id)

    @staticmethod
    def _event_error(raw_data: str) -> str:
        try:
            payload = json.loads(raw_data)
        except json.JSONDecodeError:
            return raw_data or "unknown queue error"
        if isinstance(payload, dict) and payload.get("error"):
            return str(payload["error"])
        return "the private Space returned a redacted queue error; inspect its runtime logs"


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
