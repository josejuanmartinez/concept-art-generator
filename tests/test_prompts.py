from pathlib import Path

import pytest
from PIL import Image

from concept_art_generator.loras import BY_SLUG, LORA_MODELS
from concept_art_generator.models import ArtRequest, Backend
from concept_art_generator.prompts import build_prompt, lora_prompt
from concept_art_generator.providers import DeterministicProvider, RenderedImage, RenderSpec
from concept_art_generator.workflow import ConceptArtWorkflow

DRONE = BY_SLUG["jjmcarrascosa/drone-bc"]


def hf_request(prompt, slug="jjmcarrascosa/drone-bc", game="battle-cars"):
    return ArtRequest(game, prompt, Backend.HUGGING_FACE, slug)


@pytest.mark.parametrize("model", LORA_MODELS)
def test_each_example_prompt_survives_unchanged_except_the_trigger(model):
    assert lora_prompt(model, model.example_prompt) == f"{model.trigger} {model.example_prompt}"


def test_lora_prompt_appends_the_trained_style_keywords_when_missing():
    built = lora_prompt(DRONE, "A drone with a squat black hull")
    assert built == f"drone-bc A drone with a squat black hull, {DRONE.style_suffix}"
    assert built.endswith("plain white background")


def test_lora_prompt_never_repeats_the_trigger_word():
    assert lora_prompt(DRONE, "drone-bc, A drone with a squat hull").count("drone-bc") == 1
    assert lora_prompt(DRONE, "drone-bc A drone with a squat hull").startswith("drone-bc A drone")


def test_lora_prompt_adds_nothing_else():
    built = build_prompt(hf_request(DRONE.example_prompt))
    for wrapper in ("Concept art for", "reference image", "transparent background", "watermark"):
        assert wrapper not in built
    assert built.startswith("drone-bc A drone")


def test_gpt_prompt_restates_the_style_extracted_from_the_references():
    request = ArtRequest("battle-cars", "An armored street racer.", Backend.GPT_IMAGE_2)
    built = build_prompt(request, {"a.png": "Lime thruster rings", "b.png": "Glossy cel-shaded"})
    assert "Subject: An armored street racer." in built
    assert "- Lime thruster rings" in built
    assert "- Glossy cel-shaded" in built
    assert "transparent background" in built
    assert "drone-bc" not in built


def test_gpt_prompt_honours_an_opaque_request():
    request = ArtRequest("battle-cars", "racer", Backend.GPT_IMAGE_2, transparent=False)
    assert "transparent background" not in build_prompt(request, {})


def test_backend_override_builds_the_gpt_prompt_for_a_hugging_face_request():
    built = build_prompt(hf_request("A drone"), {"a.png": "Lime rings"}, Backend.GPT_IMAGE_2)
    assert built.startswith("Concept art for battle-cars.")
    assert "drone-bc" not in built


def test_build_prompt_still_refuses_a_cross_game_lora():
    with pytest.raises(ValueError, match="cannot be used for massive-warfare"):
        build_prompt(hf_request("A drone", game="massive-warfare"))


class Replayable(DeterministicProvider):
    backend = Backend.HUGGING_FACE

    def __init__(self):
        self.specs = []

    def render(self, spec: RenderSpec):
        self.specs.append(spec)
        rendered = super().render(spec)
        return RenderedImage(
            rendered.png,
            0.0,
            "offline",
            {
                "prompt": spec.prompt,
                "lora_name": spec.lora_name,
                "negative_prompt": spec.negative_prompt,
                "steps": spec.steps,
                "guidance_scale": spec.guidance_scale,
                "lora_scale": spec.lora_scale,
                "seed": spec.seed,
                "scheduler": "FlowMatchEulerDiscreteScheduler",
                "base_model": "Qwen/Qwen-Image-2512",
                "background_model": spec.background_model,
            },
        )


class Capturing(DeterministicProvider):
    def __init__(self):
        self.specs = []

    def render(self, spec: RenderSpec):
        self.specs.append(spec)
        return super().render(spec)


def seeded(tmp_path, hf) -> tuple[ConceptArtWorkflow, Capturing]:
    gpt = Capturing()
    flow = ConceptArtWorkflow(tmp_path, {Backend.HUGGING_FACE: hf, Backend.GPT_IMAGE_2: gpt})
    source = tmp_path / "drone.png"
    Image.new("RGBA", (16, 16), (255, 0, 0, 255)).save(source)
    flow.add_reference("battle-cars", source, "White egg-shaped drone shell, lime thruster rings")
    return flow, gpt


def test_hugging_face_draft_and_final_send_the_identical_lora_prompt(tmp_path: Path):
    hf = Replayable()
    flow, _ = seeded(tmp_path, hf)
    job = flow.create_draft(hf_request(DRONE.example_prompt))
    flow.approve("battle-cars", job.id)
    flow.create_final("battle-cars", job.id)

    draft, final = hf.specs
    assert draft.prompt == f"drone-bc {DRONE.example_prompt}"
    assert final.prompt == draft.prompt


def test_gpt_fallback_rebuilds_the_prompt_without_the_trigger_word(tmp_path: Path):
    class Failing(DeterministicProvider):
        backend = Backend.HUGGING_FACE

        def render(self, spec: RenderSpec):
            raise TimeoutError("Space did not respond")

    flow, gpt = seeded(tmp_path, Failing())
    flow.create_draft(hf_request(DRONE.example_prompt))

    sent = gpt.specs[0].prompt
    assert "drone-bc" not in sent
    assert sent.startswith("Concept art for battle-cars.")
    assert "- White egg-shaped drone shell, lime thruster rings" in sent
