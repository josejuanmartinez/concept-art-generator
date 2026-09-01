"""The closed catalogue of art models.

An **art model** is the unit of work: a named visual style that owns its own reference images and
its own trained private LoRA. Exactly three exist. Choosing the art model chooses the LoRA — a slug
is never inferred or typed by hand, and anything outside this table is refused.

Isolation is per art model: references live under `data/models/<name>/references/` and are never
read, attached, or described for another model. Edit this table when a model is trained, added, or
retired.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ArtModel:
    name: str
    lora_slug: str
    subject_keyword: str
    style_suffix: str
    example_prompt: str

    @property
    def trigger(self) -> str:
        """The trained trigger word, which is the model name itself."""
        return self.name


ART_MODELS: tuple[ArtModel, ...] = (
    ArtModel(
        "drone-bc",
        "jjmcarrascosa/drone-bc",
        "A drone",
        "stylized 3D rendered game asset, glossy cel-shaded surfaces, plain white background",
        "A drone with a smooth white and grey egg-shaped shell, lime green thruster rings on "
        "each side, a glowing orange vent slot on its face, two upright yellow-tipped blade fins "
        "above and one below, stylized 3D rendered game asset, glossy cel-shaded surfaces, "
        "plain white background",
    ),
    ArtModel(
        "pilot-bc",
        "jjmcarrascosa/pilot-bc",
        "A pilot",
        "flat vector game art, plain white background",
        "A pilot in a horned imp mask and hood, wearing a green graffiti-covered hooded jacket "
        "and dark cargo trousers, holding a curved black blade, in colourful high-top sneakers, "
        "full body, flat vector game art, plain white background",
    ),
    ArtModel(
        "pilot-mw",
        "jjmcarrascosa/pilot-mw",
        "A pilot",
        "semi-realistic painted digital game art, soft rim lighting, plain white background",
        "A pilot undead soldier with a blazing skull head wreathed in orange fire, heavy charcoal "
        "combat armour lined with glowing orange lights, a rifle held at his hip, semi-realistic "
        "painted digital game art, soft rim lighting, plain white background",
    ),
)

ART_MODEL_NAMES: tuple[str, ...] = tuple(model.name for model in ART_MODELS)

BY_NAME: dict[str, ArtModel] = {model.name: model for model in ART_MODELS}


def catalogue() -> list[dict[str, str]]:
    """The three art models as plain data, for `concept-art models` and for agents."""
    return [
        {
            "art_model": model.name,
            "lora_name": model.lora_slug,
            "trigger": model.trigger,
            "subject_keyword": model.subject_keyword,
            "style_suffix": model.style_suffix,
            "example_prompt": model.example_prompt,
        }
        for model in ART_MODELS
    ]


def resolve_model(name: str | None) -> ArtModel:
    """Refuse anything that is not one of the three art models."""
    model = BY_NAME.get((name or "").strip())
    if model is None:
        raise ValueError(
            f"Unknown art model {name!r}. Only these exist: {', '.join(ART_MODEL_NAMES)}. "
            "Ask a human which one to use; never invent one."
        )
    return model
