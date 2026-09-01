"""The closed catalogue of trained private LoRAs.

Only these three exist. A slug is never inferred from a game slug, job name, folder, or
documentation example: anything outside this table is refused. Each LoRA carries the visual
language of exactly one game, so it is bound to that game here and cannot be used for another.
Edit this table when a LoRA is trained, added, or retired.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LoraModel:
    slug: str
    game: str
    subject_keyword: str
    style_suffix: str
    example_prompt: str

    @property
    def trigger(self) -> str:
        """The trained trigger word: the repo name without the owner prefix."""
        return self.slug.split("/")[-1]


LORA_MODELS: tuple[LoraModel, ...] = (
    LoraModel(
        "jjmcarrascosa/drone-bc",
        "battle-cars",
        "A drone",
        "stylized 3D rendered game asset, glossy cel-shaded surfaces, plain white background",
        "A drone with a smooth white and grey egg-shaped shell, lime green thruster rings on "
        "each side, a glowing orange vent slot on its face, two upright yellow-tipped blade fins "
        "above and one below, stylized 3D rendered game asset, glossy cel-shaded surfaces, "
        "plain white background",
    ),
    LoraModel(
        "jjmcarrascosa/pilot-bc",
        "battle-cars",
        "A pilot",
        "flat vector game art, plain white background",
        "A pilot in a horned imp mask and hood, wearing a green graffiti-covered hooded jacket "
        "and dark cargo trousers, holding a curved black blade, in colourful high-top sneakers, "
        "full body, flat vector game art, plain white background",
    ),
    LoraModel(
        "jjmcarrascosa/pilot-mw",
        "massive-warfare",
        "A pilot",
        "semi-realistic painted digital game art, soft rim lighting, plain white background",
        "A pilot undead soldier with a blazing skull head wreathed in orange fire, heavy charcoal "
        "combat armour lined with glowing orange lights, a rifle held at his hip, semi-realistic "
        "painted digital game art, soft rim lighting, plain white background",
    ),
)

LORA_SLUGS: tuple[str, ...] = tuple(model.slug for model in LORA_MODELS)

BY_SLUG: dict[str, LoraModel] = {model.slug: model for model in LORA_MODELS}


def catalogue() -> list[dict[str, str]]:
    """The three LoRAs as plain data, for `concept-art loras` and for agents."""
    return [
        {
            "lora_name": model.slug,
            "trigger": model.trigger,
            "game": model.game,
            "subject_keyword": model.subject_keyword,
            "style_suffix": model.style_suffix,
            "example_prompt": model.example_prompt,
        }
        for model in LORA_MODELS
    ]


def models_for(game: str) -> list[LoraModel]:
    return [model for model in LORA_MODELS if model.game == game]


def resolve_lora(slug: str | None, game: str) -> LoraModel:
    """Refuse an unknown LoRA, and refuse one trained on another game's art."""
    if not slug:
        raise ValueError(
            f"Hugging Face generation requires --lora-name. For {game}, choose one of: "
            f"{_listing(models_for(game)) or 'no LoRA is trained for this game'}."
        )
    model = BY_SLUG.get(slug)
    if model is None:
        raise ValueError(
            f"Unknown LoRA {slug!r}. Only these exist: {_listing(LORA_MODELS)}. "
            "Ask a human for the exact slug; never invent one."
        )
    if model.game != game:
        raise ValueError(
            f"{slug} carries {model.game} style and cannot be used for {game}. "
            f"For {game}, choose one of: "
            f"{_listing(models_for(game)) or 'no LoRA is trained for this game'}."
        )
    return model


def _listing(models: tuple[LoraModel, ...] | list[LoraModel]) -> str:
    return ", ".join(f"{model.slug} ({model.game})" for model in models)
