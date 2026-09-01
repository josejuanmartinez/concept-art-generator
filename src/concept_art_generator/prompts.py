"""The one place a generation prompt is built.

The two backends need genuinely different prompts, so `build_prompt` dispatches on the backend
rather than wrapping one shape for both. What they share is the leading trigger word: every prompt
starts `<art model> <subject>`, so a prompt reads the same whichever backend it went to.

* **Hugging Face + LoRA** — the LoRA was trained on captions of one exact shape, so the prompt is
  restricted to that shape and nothing else is added: `<trigger> <subject>, <style keywords>`,
  ending in `plain white background`. The Space strips that background itself via
  `remove_background`, so no "transparent background" instruction is appended; asking a LoRA for
  something its captions never said only pushes it off-distribution.
* **GPT Image 2** — the trigger carries no trained meaning here, so it rides along as a label on the
  subject line and the style still comes from this model's own references: the images are attached
  to the edit request, and the descriptions extracted from them are restated here as style notes so
  the requirement is explicit rather than implied.
"""

from __future__ import annotations

from .art_models import ART_MODEL_NAMES, ArtModel, resolve_model
from .models import ArtRequest, Backend

# Longest first, so `pilot-bc` is not half-matched by a shorter name that shares its opening.
_TRIGGERS = tuple(sorted(ART_MODEL_NAMES, key=len, reverse=True))


def build_prompt(
    request: ArtRequest,
    descriptions: dict[str, str] | None = None,
    backend: Backend | None = None,
) -> str:
    """Build the prompt for `backend`, defaulting to the one the request asked for.

    `backend` is passed explicitly when a Hugging Face request falls back to GPT Image 2, so the
    fallback gets a reference-driven prompt instead of only a trigger word.
    """
    effective = backend or Backend(request.backend)
    model = resolve_model(request.art_model)
    if effective == Backend.HUGGING_FACE:
        return lora_prompt(model, request.prompt)
    return reference_prompt(request, descriptions or {})


def with_trigger(model: ArtModel, subject: str) -> str:
    """`<trigger> <subject>`, with the trigger separated by a space and never repeated.

    Any catalogued trigger is stripped, not just this model's: a prompt carried over from
    another art model arrives still wearing that model's prefix, and prepending on top of it
    would send `drone-bc pilot-bc A pilot ...` — two styles named in one prompt.
    """
    text = subject.strip().rstrip(".").strip()
    for trigger in _TRIGGERS:
        if text.lower().startswith(trigger.lower()):
            text = text[len(trigger) :].lstrip(" ,").strip()
            break
    return f"{model.trigger} {text}" if text else model.trigger


def lora_prompt(model: ArtModel, subject: str) -> str:
    """`<trigger> <subject>, <style keywords>` — the shape the LoRA was trained on."""
    text = with_trigger(model, subject)
    if not text.lower().endswith(model.style_suffix.lower()):
        text = f"{text}, {model.style_suffix}"
    return text


def reference_prompt(request: ArtRequest, descriptions: dict[str, str]) -> str:
    """Style extracted from this model's own references, for a backend with no trained trigger."""
    subject = with_trigger(resolve_model(request.art_model), request.prompt)
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
    # The leading art model name is a repo slug with no trained meaning for GPT Image 2 — it keeps
    # both backends' prompts readable side by side, but the style still comes from the references.
    return (
        f"Concept art. Subject: {subject}. "
        f"Match only the visual language of the attached reference images: "
        f"silhouette discipline, materials, palette, lighting and rendering style. "
        f"{extracted}"
        f"Single isolated subject, centered, no scenery, no text, no watermark, {background}."
        f"{logo}"
    )
