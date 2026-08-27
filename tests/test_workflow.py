import json
from pathlib import Path

import pytest
from PIL import Image

from concept_art_generator.models import ArtRequest, Backend, JobState
from concept_art_generator.providers import DeterministicProvider, RenderSpec
from concept_art_generator.workflow import ConceptArtWorkflow


def reference(path: Path) -> None:
    Image.new("RGBA", (16, 16), (255, 0, 0, 255)).save(path)


def workflow(tmp_path: Path) -> ConceptArtWorkflow:
    return ConceptArtWorkflow(tmp_path, {Backend.GPT_IMAGE_2: DeterministicProvider()})


def test_final_requires_explicit_approval(tmp_path: Path):
    flow = workflow(tmp_path)
    source = tmp_path / "reference.png"
    reference(source)
    flow.add_reference("massive-warfare", source)
    job = flow.create_draft(ArtRequest("massive-warfare", "artillery vehicle", Backend.GPT_IMAGE_2))
    assert job.state == JobState.DRAFT_READY
    with pytest.raises(ValueError, match="Human approval"):
        flow.create_final("massive-warfare", job.id)
    flow.approve("massive-warfare", job.id)
    draft_sidecar = Path(job.draft_path).with_suffix(".png.prompt")
    assert json.loads(draft_sidecar.read_text())["feedback"][0]["decision"] == "approved"
    final = flow.create_final("massive-warfare", job.id)
    assert final.state == JobState.FINAL_READY
    assert Path(final.final_path).exists()
    assert max(Image.open(final.final_path).size) == 2048
    assert Path(final.final_path).with_suffix(".png.prompt").exists()


def test_rejection_feedback_is_written_to_draft_sidecar(tmp_path: Path):
    flow = workflow(tmp_path)
    source = tmp_path / "reference.png"
    reference(source)
    flow.add_reference("battle-cars", source)
    job = flow.create_draft(ArtRequest("battle-cars", "pilot", Backend.GPT_IMAGE_2))
    rejected = flow.reject(
        "battle-cars", job.id, "Make the silhouette wider and remove the spoiler."
    )
    sidecar = json.loads(Path(rejected.draft_path).with_suffix(".png.prompt").read_text())
    assert rejected.state == JobState.REJECTED
    assert sidecar["feedback"][0]["feedback"].startswith("Make the silhouette")


def test_hugging_face_failure_falls_back_to_game_local_gpt(tmp_path: Path):
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
    flow.add_reference("massive-warfare", source)
    job = flow.create_draft(
        ArtRequest("massive-warfare", "artillery vehicle", Backend.HUGGING_FACE, "mw-v1")
    )
    assert any("retried with GPT Image 2" in note for note in job.notes)


def test_game_references_are_isolated(tmp_path: Path):
    flow = workflow(tmp_path)
    first, second = tmp_path / "mw.png", tmp_path / "bc.png"
    reference(first)
    reference(second)
    flow.add_reference("massive-warfare", first)
    flow.add_reference("battle-cars", second)
    assert len(flow.workspace("massive-warfare").reference_files(10)) == 1
    assert len(flow.workspace("battle-cars").reference_files(10)) == 1


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
    flow.add_reference("battle-cars", source)
    with pytest.raises(ValueError, match="opaque"):
        flow.create_draft(ArtRequest("battle-cars", "pilot", Backend.GPT_IMAGE_2))
