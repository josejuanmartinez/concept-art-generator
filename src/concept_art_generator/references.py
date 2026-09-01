from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

MAX_REFERENCE_IMAGES = 16

# The Qwen Image LoRA Studio dataset convention: `cammy.txt` captions `cammy.png`.
# Here the same files set reference descriptions instead of training captions.
CAPTION_SUFFIX = ".txt"


def sidecar_caption(image_path: Path) -> str | None:
    """A caption .txt sitting beside the image, as the studio writes them."""
    neighbour = image_path.with_suffix(CAPTION_SUFFIX)
    if not neighbour.is_file():
        return None
    try:
        return neighbour.read_text(encoding="utf-8").strip() or None
    except (OSError, UnicodeDecodeError):
        return None


def load_caption_sidecars(paths: Iterable[Path]) -> dict[str, str]:
    """Map lowercased file stem -> caption text for a batch of uploaded .txt files."""
    captions: dict[str, str] = {}
    for path in paths:
        candidate = Path(path)
        if candidate.suffix.lower() != CAPTION_SUFFIX:
            continue
        try:
            text = candidate.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError):
            continue
        if text:
            captions[candidate.stem.lower()] = text
    return captions


def caption_for(filename: str, sidecars: dict[str, str]) -> str | None:
    """Match one image filename against loaded .txt captions by lowercased stem."""
    return sidecars.get(Path(filename).stem.lower())


class ReferenceAgent(Protocol):
    def describe(self, image_path: Path) -> str: ...

    def select(self, prompt: str, candidates: list[tuple[Path, str]], limit: int) -> list[str]: ...


class OpenAIReferenceAgent:
    """Uses a GPT model for reference descriptions and text-only relevance ranking."""

    def __init__(self, model: str | None = None):
        self.model = model or os.getenv("OPENAI_REFERENCE_MODEL", "gpt-5.6-luna")

    @staticmethod
    def _client():
        from openai import OpenAI

        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "OPENAI_API_KEY is required to describe or select reference images. "
                "Pass a description when adding each reference to avoid the description call."
            )
        return OpenAI()

    def describe(self, image_path: Path) -> str:
        mime = mimetypes.guess_type(image_path.name)[0] or "image/png"
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        response = self._client().responses.create(
            model=self.model,
            store=False,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "Describe this concept-art reference for later semantic matching. "
                                "Be concise but cover subject, silhouette, materials, palette, camera, "
                                "lighting, rendering style, and distinctive design motifs."
                            ),
                        },
                        {"type": "input_image", "image_url": f"data:{mime};base64,{encoded}"},
                    ],
                }
            ],
        )
        description = response.output_text.strip()
        if not description:
            raise RuntimeError("The GPT reference-description request returned no text.")
        return description

    def select(self, prompt: str, candidates: list[tuple[Path, str]], limit: int) -> list[str]:
        catalog = [
            {"filename": path.name, "description": description} for path, description in candidates
        ]
        response = self._client().responses.create(
            model=self.model,
            store=False,
            input=(
                f"Select exactly {limit} reference images whose descriptions best support this "
                f"concept-art request: {prompt!r}. Consider subject, shape language, materials, palette, "
                "camera, and rendering style. Return only a JSON array of filenames, best match first. "
                "Every filename must come from this catalog:\n"
                + json.dumps(catalog, ensure_ascii=False)
            ),
        )
        selected = self._json_array(response.output_text)
        allowed = {path.name for path, _ in candidates}
        unique = list(dict.fromkeys(name for name in selected if name in allowed))
        if len(unique) != limit:
            raise RuntimeError(
                f"GPT selected {len(unique)} valid unique references; expected exactly {limit}."
            )
        return unique

    @staticmethod
    def _json_array(text: str) -> list[str]:
        match = re.search(r"\[[\s\S]*\]", text)
        if not match:
            raise RuntimeError("GPT reference selection did not return a JSON array.")
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError as error:
            raise RuntimeError("GPT reference selection returned invalid JSON.") from error
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise RuntimeError("GPT reference selection must be a JSON array of filenames.")
        return value
