"""The Gradio front end: it builds, and its handlers keep the approval gate shut."""

from pathlib import Path

import pytest
from PIL import Image

from concept_art_generator import ui
from concept_art_generator.art_models import ART_MODEL_NAMES, BY_NAME
from concept_art_generator.models import ArtRequest, Backend
from concept_art_generator.providers import DeterministicProvider, RenderedImage, RenderSpec
from concept_art_generator.workflow import ConceptArtWorkflow


class OfflineHuggingFace(DeterministicProvider):
    backend = Backend.HUGGING_FACE

    def render(self, spec: RenderSpec) -> RenderedImage:
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


class OfflineAgent:
    def describe(self, image_path: Path) -> str:
        return f"offline description for {image_path.name}"

    def select(self, prompt, candidates, limit):
        return [path.name for path, _ in candidates][:limit]


@pytest.fixture
def flow(tmp_path):
    workflow = ConceptArtWorkflow(
        tmp_path,
        {
            Backend.GPT_IMAGE_2: DeterministicProvider(),
            Backend.HUGGING_FACE: OfflineHuggingFace(),
        },
        reference_agent=OfflineAgent(),
    )
    source = tmp_path / "reference.png"
    Image.new("RGBA", (16, 16), (255, 0, 0, 255)).save(source)
    workflow.add_reference("pilot-mw", source, "A charcoal-armoured pilot reference")
    return workflow


def test_the_app_builds(flow):
    demo = ui.build_ui(flow)
    assert demo.title == "Concept Art Generator"


def test_caption_progress_wording():
    assert "No references yet" in ui.caption_progress({})
    assert ui.caption_progress({"a.png": "x"}) == "All 1 references captioned."
    assert "1 of 2 references captioned" in ui.caption_progress({"a.png": "x", "b.png": ""})


def test_every_art_model_offers_its_own_example_prompt():
    assert set(ui.EXAMPLE_PROMPTS) == set(ART_MODEL_NAMES)
    for name, prompt in ui.EXAMPLE_PROMPTS.items():
        assert prompt == BY_NAME[name].example_prompt


def test_the_gallery_is_paged_and_scoped_to_one_art_model(flow, tmp_path):
    for index in range(10):
        source = tmp_path / f"extra-{index:02}.png"
        Image.new("RGBA", (16, 16), (0, 0, 255, 255)).save(source)
        flow.add_reference("pilot-mw", source, f"Reference {index}")
    other = tmp_path / "drone.png"
    Image.new("RGBA", (16, 16), (0, 255, 0, 255)).save(other)
    flow.add_reference("drone-bc", other, "A drone reference")

    ui.build_ui(flow)  # exercises the same helpers the Blocks wire up
    workflow_names = list(flow.reference_descriptions("pilot-mw"))
    assert len(workflow_names) == 11
    assert "drone.png" not in workflow_names
    assert list(flow.reference_descriptions("drone-bc")) == ["drone.png"]


@pytest.mark.parametrize("backend", ["gpt-image-2", "huggingface"])
def test_the_export_button_unlocks_only_after_approval(flow, backend):
    job = flow.create_draft(ArtRequest("pilot-mw", "A pilot", Backend(backend)))
    approve, reject, export = ui.approval_gate(job)
    assert approve["interactive"] is True
    assert reject["interactive"] is True
    assert export["interactive"] is False, "the 2K final must be locked before approval"

    approve, reject, export = ui.approval_gate(flow.approve(job.art_model, job.id))
    assert export["interactive"] is True
    assert approve["interactive"] is False


def test_a_rejected_draft_locks_every_button(flow):
    job = flow.create_draft(ArtRequest("pilot-mw", "A pilot", Backend.GPT_IMAGE_2))
    rejected = flow.reject(job.art_model, job.id, "Wider silhouette")
    assert [update["interactive"] for update in ui.approval_gate(rejected)] == [False] * 3


def test_no_job_locks_every_button():
    assert [update["interactive"] for update in ui.approval_gate(None)] == [False] * 3


def test_the_job_summary_names_the_model_lora_and_prompt_sent(flow):
    job = flow.create_draft(ArtRequest("pilot-mw", "A pilot", Backend.HUGGING_FACE))
    summary = ui.describe(job)
    assert "pilot-mw" in summary
    assert "jjmcarrascosa/pilot-mw" in summary
    assert job.artifacts["draft"]["prompt"] in summary
    assert "draft_ready" in summary


class CountingAgent:
    """Records every provider call so a test can assert none happened."""

    def __init__(self):
        self.calls = []

    def describe(self, image_path: Path) -> str:
        self.calls.append(image_path.name)
        return f"GPT description for {image_path.name}"

    def select(self, prompt, candidates, limit):
        return [path.name for path, _ in candidates][:limit]


@pytest.fixture
def counted(tmp_path):
    agent = CountingAgent()
    workflow = ConceptArtWorkflow(
        tmp_path, {Backend.GPT_IMAGE_2: DeterministicProvider()}, reference_agent=agent
    )
    return workflow, agent


def sources(tmp_path, count=3):
    made = []
    for index in range(count):
        source = tmp_path / f"ref-{index}.png"
        Image.new("RGBA", (16, 16), (index * 40, 0, 0, 255)).save(source)
        made.append(source)
    return made


def test_adding_references_never_calls_the_provider(counted, tmp_path):
    """Uploading must not block on GPT: captioning is a separate, explicit step."""
    workflow, agent = counted
    for source in sources(tmp_path):
        workflow.add_reference("drone-bc", source, describe_missing=False)

    assert agent.calls == []
    assert workflow.uncaptioned_references("drone-bc") == [
        "ref-0.png",
        "ref-1.png",
        "ref-2.png",
    ]


def test_describing_only_touches_references_that_are_still_blank(counted, tmp_path):
    workflow, agent = counted
    made = sources(tmp_path)
    for source in made:
        workflow.add_reference("drone-bc", source, describe_missing=False)
    workflow.set_reference_description("drone-bc", "ref-1.png", "typed by hand")

    for filename in workflow.uncaptioned_references("drone-bc"):
        workflow.describe_reference("drone-bc", filename)

    assert agent.calls == ["ref-0.png", "ref-2.png"], "the captioned one must be left alone"
    descriptions = workflow.reference_descriptions("drone-bc")
    assert descriptions["ref-1.png"] == "typed by hand"
    assert descriptions["ref-0.png"] == "GPT description for ref-0.png"
    assert workflow.uncaptioned_references("drone-bc") == []


def test_txt_captions_remove_the_need_to_describe(counted, tmp_path):
    workflow, agent = counted
    made = sources(tmp_path, 2)
    for source in made:
        workflow.add_reference("drone-bc", source, describe_missing=False)
    for index in range(2):
        (tmp_path / f"ref-{index}.txt").write_text(f"caption {index}", encoding="utf-8")

    workflow.apply_caption_files("drone-bc", [tmp_path / f"ref-{i}.txt" for i in range(2)])

    assert workflow.uncaptioned_references("drone-bc") == []
    assert agent.calls == []
