import pytest
from PIL import Image

from concept_art_generator.loras import LORA_SLUGS, catalogue, resolve_lora
from concept_art_generator.models import ArtRequest, Backend
from concept_art_generator.providers import DeterministicProvider
from concept_art_generator.workflow import ConceptArtWorkflow


def test_exactly_three_loras_exist():
    assert LORA_SLUGS == (
        "jjmcarrascosa/drone-bc",
        "jjmcarrascosa/pilot-bc",
        "jjmcarrascosa/pilot-mw",
    )
    assert [entry["game"] for entry in catalogue()] == [
        "battle-cars",
        "battle-cars",
        "massive-warfare",
    ]
    assert all(entry["example_prompt"].endswith("plain white background") for entry in catalogue())


def test_resolve_accepts_a_catalogued_lora_for_its_own_game():
    assert resolve_lora("jjmcarrascosa/pilot-mw", "massive-warfare").game == "massive-warfare"


def test_resolve_refuses_an_invented_slug():
    with pytest.raises(ValueError, match="Unknown LoRA"):
        resolve_lora("jjmcarrascosa/pilot-wc", "massive-warfare")


def test_resolve_refuses_another_games_lora():
    with pytest.raises(ValueError, match="carries battle-cars style"):
        resolve_lora("jjmcarrascosa/drone-bc", "massive-warfare")


def test_resolve_refuses_a_missing_slug_and_names_the_options():
    with pytest.raises(ValueError, match="jjmcarrascosa/drone-bc"):
        resolve_lora(None, "battle-cars")


def flow_with(tmp_path):
    flow = ConceptArtWorkflow(
        tmp_path,
        {
            Backend.HUGGING_FACE: DeterministicProvider(),
            Backend.GPT_IMAGE_2: DeterministicProvider(),
        },
    )
    source = tmp_path / "reference.png"
    Image.new("RGBA", (16, 16), (255, 0, 0, 255)).save(source)
    flow.add_reference("massive-warfare", source, "Tracked military vehicle")
    return flow


def test_draft_refuses_a_lora_trained_on_another_game(tmp_path):
    flow = flow_with(tmp_path)
    with pytest.raises(ValueError, match="cannot be used for massive-warfare"):
        flow.create_draft(
            ArtRequest(
                "massive-warfare", "artillery", Backend.HUGGING_FACE, "jjmcarrascosa/pilot-bc"
            )
        )


def test_draft_refuses_an_uncatalogued_lora(tmp_path):
    flow = flow_with(tmp_path)
    with pytest.raises(ValueError, match="Unknown LoRA"):
        flow.create_draft(
            ArtRequest("massive-warfare", "artillery", Backend.HUGGING_FACE, "owner/made-up")
        )


def test_gpt_image_2_needs_no_lora(tmp_path):
    flow = flow_with(tmp_path)
    assert flow.create_draft(
        ArtRequest("massive-warfare", "artillery", Backend.GPT_IMAGE_2)
    ).lora_name is None
