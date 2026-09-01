"""The one place a generation prompt is built.

The two backends need genuinely different prompts, so `build_prompt` dispatches on the backend
rather than wrapping one shape for both:

* **Hugging Face + LoRA** — the LoRA was trained on captions of one exact shape, so the prompt is
  restricted to that shape and nothing else is added: `<trigger> <subject>, <style keywords>`,
  ending in `plain white background`. The Space strips that background itself via
  `remove_background`, so no "transparent background" instruction is appended; asking a LoRA for
  something its captions never said only pushes it off-distribution.
* **GPT Image 2** — no trigger word exists. The style comes from this model's own references: the
  images are attached to the edit request, and the descriptions extracted from them are restated
  here as style notes so the requirement is explicit rather than implied.
"""

from __future__ import annotations

from .art_models import ArtModel, resolve_model
from .models import ArtRequest, Backend


def build_prompt(
    request: ArtRequest,
    descriptions: dict[str, str] | None = None,
    backend: Backend | None = None,
) -> str:
    """Build the prompt for `backend`, defaulting to the one the request asked for.

    `backend` is passed explicitly when a Hugging Face request falls back to GPT Image 2, so the
    fallback gets a reference-driven prompt instead of a meaningless trigger word.
    """
    effective = backend or Backend(request.backend)
    if effective == Backend.HUGGING_FACE:
        return lora_prompt(resolve_model(request.art_model), request.prompt)
    return reference_prompt(request, descriptions or {})


def lora_prompt(model: ArtModel, subject: str) -> str:
    """`<trigger> <subject>, <style keywords>` — the shape the LoRA was trained on.

    The trigger is separated by a space, exactly as the training captions were written.
    """
    text = subject.strip().rstrip(".").strip()
    if text.lower().startswith(model.trigger.lower()):
        text = text[len(model.trigger) :].lstrip(" ,").strip()
    if not text.lower().endswith(model.style_suffix.lower()):
        text = f"{text}, {model.style_suffix}" if text else model.style_suffix
    return f"{model.trigger} {text}"


def reference_prompt(request: ArtRequest, descriptions: dict[str, str]) -> str:
    """Style extracted from this model's own references, for a backend with no trigger word."""
    subject = request.prompt.strip().rstrip(".").strip()
    notes = [text.strip() for text in descriptions.values() if text.strip()]
    extracted = (
        "Style extracted from those references:\n"
        + "\n".join(f"- {note}" for note in notes)
        + "\n"
        if notes
        else ""
    )
    logo = (
        " Preserve the supplied logo exactly; do not redraw, warp, or invent lettering."
        if request.logo_path
        else ""
    )
    background = "transparent background" if request.transparent else "plain, uncluttered background"
    # The art model's name is a repo slug, meaningless to GPT Image 2 — the style has to come
    # from the references themselves, so the name is deliberately left out.
    return (
        f"Concept art. Subject: {subject}. "
        f"Match only the visual language of the attached reference images: "
        f"silhouette discipline, materials, palette, lighting and rendering style. "
        f"{extracted}"
        f"Single isolated subject, centered, no scenery, no text, no watermark, {background}."
        f"{logo}"
    )
