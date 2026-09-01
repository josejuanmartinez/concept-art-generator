import json
from pathlib import Path

import pytest
from PIL import Image

from concept_art_generator.models import ArtRequest, Backend, JobState
from concept_art_generator.providers import DeterministicProvider, RenderedImage, RenderSpec
from concept_art_generator.workflow import ConceptArtWorkflow


def reference(path: Path) -> None:
    Image.new("RGBA", (16, 16), (255, 0, 0, 255)).save(path)


def workflow(tmp_path: Path) -> ConceptArtWorkflow:
    return ConceptArtWorkflow(tmp_path, {Backend.GPT_IMAGE_2: DeterministicProvider()})


def test_final_requires_explicit_approval(tmp_path: Path):
    flow = workflow(tmp_path)
    source = tmp_path / "reference.png"
    reference(source)
    flow.add_reference("pilot-mw", source, "Red armored vehicle concept art")
    job = flow.create_draft(ArtRequest("pilot-mw", "artillery vehicle", Backend.GPT_IMAGE_2))
    assert job.state == JobState.DRAFT_READY
    with pytest.raises(ValueError, match="Human approval"):
        flow.create_final("pilot-mw", job.id)
    flow.approve("pilot-mw", job.id)
    draft_sidecar = Path(job.draft_path).with_suffix(".png.prompt")
    assert json.loads(draft_sidecar.read_text())["feedback"][0]["decision"] == "approved"
    final = flow.create_final("pilot-mw", job.id)
    assert final.state == JobState.FINAL_READY
    assert Path(final.final_path).exists()
    assert max(Image.open(final.final_path).size) == 2048
    assert Path(final.final_path).with_suffix(".png.prompt").exists()


def test_rejection_feedback_is_written_to_draft_sidecar(tmp_path: Path):
    flow = workflow(tmp_path)
    source = tmp_path / "reference.png"
    reference(source)
    flow.add_reference("drone-bc", source, "Red battle car concept art")
    job = flow.create_draft(ArtRequest("drone-bc", "pilot", Backend.GPT_IMAGE_2))
    rejected = flow.reject(
        "drone-bc", job.id, "Make the silhouette wider and remove the spoiler."
    )
    sidecar = json.loads(Path(rejected.draft_path).with_suffix(".png.prompt").read_text())
    assert rejected.state == JobState.REJECTED
    assert sidecar["feedback"][0]["feedback"].startswith("Make the silhouette")


def test_hugging_face_failure_falls_back_to_the_models_own_gpt_references(tmp_path: Path):
    class FailingHuggingFace(DeterministicProvider):
        backend = Backend.HUGGING_FACE

        def render(self, spec: RenderSpec):
            raise TimeoutError("Space did not respond")

    flow = ConceptArtWorkflow(
        tmp_path,
        {
            Backend.HUGGING_FACE: FailingHuggingFace(),
            Backend.GPT_IMAGE_2: DeterministicProvider(),
        },
    )
    source = tmp_path / "reference.png"
    reference(source)
    flow.add_reference("pilot-mw", source, "Red artillery vehicle concept art")
    job = flow.create_draft(
        ArtRequest("pilot-mw", "artillery vehicle", Backend.HUGGING_FACE)
    )
    assert any("retried with GPT Image 2" in note for note in job.notes)


def test_art_model_references_are_isolated(tmp_path: Path):
    flow = workflow(tmp_path)
    first, second = tmp_path / "mw.png", tmp_path / "bc.png"
    reference(first)
    reference(second)
    flow.add_reference("pilot-mw", first, "Massive Warfare vehicle")
    flow.add_reference("drone-bc", second, "Battle Cars vehicle")
    assert len(flow.workspace("pilot-mw").reference_files(10)) == 1
    assert len(flow.workspace("drone-bc").reference_files(10)) == 1


def test_opaque_provider_is_rejected(tmp_path: Path):
    from concept_art_generator.providers import RenderedImage, RenderSpec

    class Opaque(DeterministicProvider):
        def render(self, spec: RenderSpec):
            import io

            image = Image.new("RGBA", (16, 16), (1, 2, 3, 255))
            buffer = io.BytesIO()
            image.save(buffer, "PNG")
            return RenderedImage(buffer.getvalue(), 0)

    flow = ConceptArtWorkflow(tmp_path, {Backend.GPT_IMAGE_2: Opaque()})
    source = tmp_path / "reference.png"
    reference(source)
    flow.add_reference("drone-bc", source, "Red pilot reference")
    with pytest.raises(ValueError, match="opaque"):
        flow.create_draft(ArtRequest("drone-bc", "pilot", Backend.GPT_IMAGE_2))


def test_more_than_16_references_are_selected_by_description_and_persisted(tmp_path: Path):
    class FakeReferenceAgent:
        def __init__(self):
            self.candidates = []

        def describe(self, image_path: Path) -> str:
            raise AssertionError("Descriptions were supplied explicitly")

        def select(self, prompt, candidates, limit):
            self.candidates = candidates
            assert prompt == "tracked artillery"
            assert limit == 16
            return [path.name for path, _description in reversed(candidates)][0:limit]

    agent = FakeReferenceAgent()
    flow = ConceptArtWorkflow(
        tmp_path, {Backend.GPT_IMAGE_2: DeterministicProvider()}, reference_agent=agent
    )
    for index in range(18):
        source = tmp_path / f"ref-{index:02}.png"
        reference(source)
        flow.add_reference("pilot-mw", source, f"Reference description {index}")

    job = flow.create_draft(ArtRequest("pilot-mw", "tracked artillery", Backend.GPT_IMAGE_2))

    assert len(agent.candidates) == 18
    assert len(job.reference_files) == 16
    assert job.reference_files[0] == "ref-17.png"
    assert list(job.reference_descriptions) == job.reference_files
    persisted = json.loads(flow.workspace(job.art_model).job_file(job.id).read_text())
    assert persisted["reference_files"] == job.reference_files


def test_missing_description_is_generated_and_saved(tmp_path: Path):
    class FakeReferenceAgent:
        def describe(self, image_path: Path) -> str:
            return f"GPT description for {image_path.name}"

        def select(self, prompt, candidates, limit):
            raise AssertionError("Only one reference exists")

    flow = ConceptArtWorkflow(
        tmp_path,
        {Backend.GPT_IMAGE_2: DeterministicProvider()},
        reference_agent=FakeReferenceAgent(),
    )
    source = tmp_path / "reference.png"
    reference(source)
    target = flow.add_reference("drone-bc", source)

    descriptions = flow.workspace("drone-bc").reference_descriptions()
    assert descriptions[target.name] == "GPT description for reference.png"


def test_hf_final_replays_approved_1024_parameters_and_only_adds_upscale(tmp_path: Path):
    class ReplayableHuggingFace(DeterministicProvider):
        backend = Backend.HUGGING_FACE

        def __init__(self):
            self.specs = []

        def render(self, spec: RenderSpec):
            self.specs.append(spec)
            rendered = super().render(spec)
            parameters = {
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
            }
            return RenderedImage(
                rendered.png, rendered.estimated_cost_usd, rendered.provider_request_id, parameters
            )

    provider = ReplayableHuggingFace()
    flow = ConceptArtWorkflow(tmp_path, {Backend.HUGGING_FACE: provider})
    source = tmp_path / "reference.png"
    reference(source)
    flow.add_reference("pilot-mw", source, "Tracked military vehicle")
    job = flow.create_draft(
        ArtRequest(
            "pilot-mw",
            "tracked artillery",
            Backend.HUGGING_FACE,
            seed=42,
            negative_prompt="blurry",
        )
    )
    flow.approve(job.art_model, job.id)
    flow.create_final(job.art_model, job.id)

    draft, final = provider.specs
    assert (draft.width, draft.height) == (1024, 1024)
    assert (final.width, final.height) == (2048, 2048)
    assert final.prompt == draft.prompt
    assert final.seed == draft.seed == 42
    assert final.negative_prompt == draft.negative_prompt == "blurry"
    assert final.steps == draft.steps == 28
    assert final.guidance_scale == draft.guidance_scale == 4.0
    assert final.lora_scale == draft.lora_scale == 1.25
    assert final.background_model == draft.background_model == "birefnet-general"
    assert final.scheduler == "FlowMatchEulerDiscreteScheduler"
    assert final.base_model == "Qwen/Qwen-Image-2512"


def test_gpt_final_uses_approved_draft_plus_at_most_15_style_references(tmp_path: Path):
    class CapturingGPT(DeterministicProvider):
        def __init__(self):
            self.specs = []

        def render(self, spec: RenderSpec):
            self.specs.append(spec)
            return super().render(spec)

    provider = CapturingGPT()
    flow = ConceptArtWorkflow(tmp_path, {Backend.GPT_IMAGE_2: provider})
    for index in range(16):
        source = tmp_path / f"reference-{index:02}.png"
        reference(source)
        flow.add_reference("drone-bc", source, f"Combat car style reference {index}")

    job = flow.create_draft(ArtRequest("drone-bc", "armored racer", Backend.GPT_IMAGE_2))
    flow.approve(job.art_model, job.id)
    flow.create_final(job.art_model, job.id)

    draft, final = provider.specs
    assert len(draft.references) == 16
    assert len(final.references) == 16
    assert final.references[0] == Path(job.draft_path)
    assert [path.name for path in final.references[1:]] == job.reference_files[:15]
