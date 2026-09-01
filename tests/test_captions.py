"""Reference captioning: manual text, sibling `.txt` files, and the UI caption editor.

Mirrors the Qwen Image LoRA Studio convention (`drone.txt` captions `drone.png`), used here to
set reference descriptions rather than training captions.
"""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from concept_art_generator import web
from concept_art_generator.models import Backend
from concept_art_generator.providers import DeterministicProvider
from concept_art_generator.workflow import ConceptArtWorkflow


class RefusingAgent:
    """Proves a code path never reaches GPT."""

    def describe(self, image_path):
        raise AssertionError(f"GPT should not have been asked to describe {image_path.name}")

    def select(self, prompt, candidates, limit):
        return [path.name for path, _ in candidates][:limit]


def image(path):
    Image.new("RGBA", (16, 16), (255, 0, 0, 255)).save(path)


def png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGBA", (16, 16), (255, 0, 0, 255)).save(buffer, "PNG")
    return buffer.getvalue()


@pytest.fixture
def flow(tmp_path):
    return ConceptArtWorkflow(
        tmp_path, {Backend.GPT_IMAGE_2: DeterministicProvider()}, reference_agent=RefusingAgent()
    )


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(
        web,
        "workflow",
        ConceptArtWorkflow(
            tmp_path,
            {Backend.GPT_IMAGE_2: DeterministicProvider()},
            reference_agent=RefusingAgent(),
        ),
    )
    return TestClient(web.app)


def test_a_sibling_txt_captions_the_image(flow, tmp_path):
    source = tmp_path / "drone.png"
    image(source)
    (tmp_path / "drone.txt").write_text("A drone with a smooth white shell", encoding="utf-8")
    target = flow.add_reference("drone-bc", source)
    assert flow.reference_descriptions("drone-bc")[target.name] == (
        "A drone with a smooth white shell"
    )


def test_an_explicit_description_beats_the_sibling_txt(flow, tmp_path):
    source = tmp_path / "drone.png"
    image(source)
    (tmp_path / "drone.txt").write_text("from the txt", encoding="utf-8")
    flow.add_reference("drone-bc", source, "typed by hand")
    assert flow.reference_descriptions("drone-bc")["drone.png"] == "typed by hand"


def test_describe_missing_false_leaves_the_caption_empty(flow, tmp_path):
    source = tmp_path / "drone.png"
    image(source)
    flow.add_reference("drone-bc", source, describe_missing=False)
    assert flow.reference_descriptions("drone-bc")["drone.png"] == ""


def test_caption_files_are_matched_by_stem_case_insensitively(flow, tmp_path):
    for name in ("drone.png", "pilot.png"):
        source = tmp_path / name
        image(source)
        flow.add_reference("drone-bc", source, describe_missing=False)
    (tmp_path / "DRONE.txt").write_text("egg-shaped shell", encoding="utf-8")
    (tmp_path / "unrelated.txt").write_text("ignored", encoding="utf-8")

    matched = flow.apply_caption_files(
        "drone-bc", [tmp_path / "DRONE.txt", tmp_path / "unrelated.txt"]
    )

    descriptions = flow.reference_descriptions("drone-bc")
    assert matched == 1
    assert descriptions["drone.png"] == "egg-shaped shell"
    assert descriptions["pilot.png"] == ""


def test_set_reference_description_refuses_a_foreign_filename(flow, tmp_path):
    source = tmp_path / "drone.png"
    image(source)
    flow.add_reference("drone-bc", source, "a drone")
    with pytest.raises(ValueError, match="missing"):
        flow.set_reference_description("drone-bc", "../../secret.png", "x")


def test_api_uploads_images_with_matching_caption_txt_files(client):
    response = client.post(
        "/api/models/drone-bc/references",
        data={"captioning": "manual"},
        files=[
            ("files", ("drone.png", png(), "image/png")),
            ("files", ("pilot.png", png(), "image/png")),
            ("caption_files", ("drone.txt", b"egg-shaped shell", "text/plain")),
            ("caption_files", ("pilot.txt", b"horned imp mask", "text/plain")),
        ],
    )
    assert response.status_code == 200
    stored = client.get("/api/models/drone-bc/references").json()["descriptions"]
    assert stored == {"drone.png": "egg-shaped shell", "pilot.png": "horned imp mask"}


def test_api_manual_captioning_never_calls_gpt(client):
    # The fixture's agent raises if asked; a 200 proves the upload avoided it.
    response = client.post(
        "/api/models/drone-bc/references",
        data={"captioning": "manual"},
        files=[("files", ("a.png", png(), "image/png"))],
    )
    assert response.status_code == 200
    assert client.get("/api/models/drone-bc/references").json()["descriptions"] == {"a.png": ""}


def test_api_saves_typed_captions(client):
    client.post(
        "/api/models/drone-bc/references",
        data={"captioning": "manual"},
        files=[("files", ("drone.png", png(), "image/png"))],
    )
    saved = client.post(
        "/api/models/drone-bc/captions",
        data={"drone.png": "  A drone with lime green thruster rings  "},
    )
    assert saved.json()["descriptions"]["drone.png"] == "A drone with lime green thruster rings"


def test_api_caption_file_upload_overwrites_stored_descriptions(client):
    client.post(
        "/api/models/drone-bc/references",
        data={"captioning": "manual"},
        files=[("files", ("drone.png", png(), "image/png"))],
    )
    client.post("/api/models/drone-bc/captions", data={"drone.png": "placeholder"})
    loaded = client.post(
        "/api/models/drone-bc/caption-files",
        files=[("caption_files", ("drone.txt", b"egg-shaped shell", "text/plain"))],
    )
    assert loaded.json() == {"matched": 1, "descriptions": {"drone.png": "egg-shaped shell"}}


def test_an_unknown_art_model_is_refused_by_the_api(client):
    assert client.get("/api/models/drone-mw/references").status_code == 404


def test_re_adding_the_same_reference_keeps_its_caption_and_calls_no_gpt(flow, tmp_path):
    """The fixture's agent raises if asked, so reaching the end proves GPT was never called."""
    source = tmp_path / "drone.png"
    image(source)
    flow.add_reference("drone-bc", source, "My careful hand-written caption")

    flow.add_reference("drone-bc", source)  # same bytes, no caption supplied

    descriptions = flow.reference_descriptions("drone-bc")
    assert list(descriptions) == ["drone.png"], "the image must not be duplicated"
    assert descriptions["drone.png"] == "My careful hand-written caption"


def test_re_adding_with_a_new_caption_still_updates_it(flow, tmp_path):
    source = tmp_path / "drone.png"
    image(source)
    flow.add_reference("drone-bc", source, "first caption")
    flow.add_reference("drone-bc", source, "second caption")
    assert flow.reference_descriptions("drone-bc")["drone.png"] == "second caption"


def test_re_adding_an_uncaptioned_reference_still_describes_it(tmp_path):
    """Idempotence must not strand a reference that never got a description."""

    class CountingAgent:
        def __init__(self):
            self.calls = 0

        def describe(self, image_path):
            self.calls += 1
            return f"GPT description {self.calls}"

        def select(self, prompt, candidates, limit):
            return [path.name for path, _ in candidates][:limit]

    agent = CountingAgent()
    flow = ConceptArtWorkflow(
        tmp_path, {Backend.GPT_IMAGE_2: DeterministicProvider()}, reference_agent=agent
    )
    source = tmp_path / "drone.png"
    image(source)
    flow.add_reference("drone-bc", source, describe_missing=False)
    assert flow.reference_descriptions("drone-bc")["drone.png"] == ""

    flow.add_reference("drone-bc", source)
    assert flow.reference_descriptions("drone-bc")["drone.png"] == "GPT description 1"
    assert agent.calls == 1


def test_a_different_image_with_the_same_name_is_kept_separately(flow, tmp_path):
    source = tmp_path / "drone.png"
    image(source)
    flow.add_reference("drone-bc", source, "the red one")
    Image.new("RGBA", (16, 16), (0, 0, 255, 255)).save(source)
    flow.add_reference("drone-bc", source, "the blue one")

    descriptions = flow.reference_descriptions("drone-bc")
    assert len(descriptions) == 2
    assert set(descriptions.values()) == {"the red one", "the blue one"}


def test_the_upload_endpoint_takes_no_shared_description(client):
    """Captions are per image; a blanket description would defeat 16-of-N ranking."""
    client.post(
        "/api/models/drone-bc/references",
        data={"captioning": "manual", "descriptions": "a blanket caption"},
        files=[
            ("files", ("a.png", png(), "image/png")),
            ("files", ("b.png", png(), "image/png")),
        ],
    )
    stored = client.get("/api/models/drone-bc/references").json()["descriptions"]
    assert stored == {"a.png": "", "b.png": ""}


def test_uploaded_txt_captions_are_never_overridden_by_anything_else(client):
    """The regression that motivated removing the shared box: .txt data was discarded."""
    client.post(
        "/api/models/drone-bc/references",
        data={"captioning": "manual", "descriptions": "a blanket caption"},
        files=[
            ("files", ("drone.png", png(), "image/png")),
            ("files", ("pilot.png", png(), "image/png")),
            ("caption_files", ("drone.txt", b"Egg-shaped white drone shell", "text/plain")),
            ("caption_files", ("pilot.txt", b"Horned imp mask pilot", "text/plain")),
        ],
    )
    stored = client.get("/api/models/drone-bc/references").json()["descriptions"]
    assert stored == {
        "drone.png": "Egg-shaped white drone shell",
        "pilot.png": "Horned imp mask pilot",
    }
