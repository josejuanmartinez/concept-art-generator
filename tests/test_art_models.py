import pytest
from PIL import Image

from concept_art_generator.art_models import ART_MODEL_NAMES, catalogue, resolve_model
from concept_art_generator.models import ArtRequest, Backend
from concept_art_generator.providers import DeterministicProvider
from concept_art_generator.workflow import ConceptArtWorkflow
from concept_art_generator.workspace import IsolationError, ModelWorkspace


def test_exactly_three_art_models_exist():
    assert ART_MODEL_NAMES == ("drone-bc", "pilot-bc", "pilot-mw")
    assert [entry["lora_name"] for entry in catalogue()] == [
        "jjmcarrascosa/drone-bc",
        "jjmcarrascosa/pilot-bc",
        "jjmcarrascosa/pilot-mw",
    ]


def test_the_trigger_word_is_the_model_name():
    assert [entry["trigger"] for entry in catalogue()] == list(ART_MODEL_NAMES)


def test_every_example_prompt_ends_in_its_own_style_keywords():
    for entry in catalogue():
        assert entry["example_prompt"].endswith(entry["style_suffix"])
        assert entry["example_prompt"].startswith(entry["subject_keyword"])


@pytest.mark.parametrize("name", ART_MODEL_NAMES)
def test_resolve_accepts_each_catalogued_model(name):
    assert resolve_model(name).name == name


@pytest.mark.parametrize("name", ["drone-mw", "pilot-wc", "owner/made-up", "", None])
def test_resolve_refuses_anything_else(name):
    with pytest.raises(ValueError, match="Unknown art model"):
        resolve_model(name)


def flow_with(tmp_path, art_model="pilot-mw") -> ConceptArtWorkflow:
    flow = ConceptArtWorkflow(
        tmp_path,
        {
            Backend.HUGGING_FACE: DeterministicProvider(),
            Backend.GPT_IMAGE_2: DeterministicProvider(),
        },
    )
    source = tmp_path / "reference.png"
    Image.new("RGBA", (16, 16), (255, 0, 0, 255)).save(source)
    flow.add_reference(art_model, source, "A reference for this model")
    return flow


def test_the_workspace_refuses_an_uncatalogued_model(tmp_path):
    with pytest.raises(IsolationError, match="Unknown art model"):
        ModelWorkspace(tmp_path, "drone-mw")


def test_choosing_the_model_chooses_its_lora(tmp_path):
    flow = flow_with(tmp_path)
    job = flow.create_draft(ArtRequest("pilot-mw", "A pilot", Backend.HUGGING_FACE))
    assert job.lora_name == "jjmcarrascosa/pilot-mw"


def test_gpt_image_2_records_no_lora(tmp_path):
    flow = flow_with(tmp_path)
    job = flow.create_draft(ArtRequest("pilot-mw", "A pilot", Backend.GPT_IMAGE_2))
    assert job.lora_name is None


def test_a_draft_for_an_uncatalogued_model_is_refused(tmp_path):
    flow = flow_with(tmp_path)
    with pytest.raises(ValueError, match="Unknown art model"):
        flow.create_draft(ArtRequest("drone-mw", "A drone", Backend.HUGGING_FACE))


def test_each_model_keeps_its_own_reference_folder(tmp_path):
    flow = flow_with(tmp_path, "pilot-mw")
    second = tmp_path / "other.png"
    Image.new("RGBA", (16, 16), (0, 255, 0, 255)).save(second)
    flow.add_reference("drone-bc", second, "A drone reference")

    assert list(flow.reference_descriptions("pilot-mw")) == ["reference.png"]
    assert list(flow.reference_descriptions("drone-bc")) == ["other.png"]
    assert sorted(flow.art_models()) == ["drone-bc", "pilot-mw"]
    assert (tmp_path / "models" / "pilot-mw" / "references").is_dir()
