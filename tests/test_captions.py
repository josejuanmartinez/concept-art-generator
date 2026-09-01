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

BROWSER = {"accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"}


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
    target = flow.add_reference("battle-cars", source)
    assert flow.reference_descriptions("battle-cars")[target.name] == (
        "A drone with a smooth white shell"
    )


def test_an_explicit_description_beats_the_sibling_txt(flow, tmp_path):
    source = tmp_path / "drone.png"
    image(source)
    (tmp_path / "drone.txt").write_text("from the txt", encoding="utf-8")
    flow.add_reference("battle-cars", source, "typed by hand")
    assert flow.reference_descriptions("battle-cars")["drone.png"] == "typed by hand"


def test_describe_missing_false_leaves_the_caption_empty(flow, tmp_path):
    source = tmp_path / "drone.png"
    image(source)
    flow.add_reference("battle-cars", source, describe_missing=False)
    assert flow.reference_descriptions("battle-cars")["drone.png"] == ""


def test_caption_files_are_matched_by_stem_case_insensitively(flow, tmp_path):
    for name in ("drone.png", "pilot.png"):
        source = tmp_path / name
        image(source)
        flow.add_reference("battle-cars", source, describe_missing=False)
    (tmp_path / "DRONE.txt").write_text("egg-shaped shell", encoding="utf-8")
    (tmp_path / "unrelated.txt").write_text("ignored", encoding="utf-8")

    matched = flow.apply_caption_files(
        "battle-cars", [tmp_path / "DRONE.txt", tmp_path / "unrelated.txt"]
    )

    descriptions = flow.reference_descriptions("battle-cars")
    assert matched == 1
    assert descriptions["drone.png"] == "egg-shaped shell"
    assert descriptions["pilot.png"] == ""


def test_set_reference_description_refuses_a_foreign_filename(flow, tmp_path):
    source = tmp_path / "drone.png"
    image(source)
    flow.add_reference("battle-cars", source, "a drone")
    with pytest.raises(ValueError, match="missing"):
        flow.set_reference_description("battle-cars", "../../secret.png", "x")


def test_ui_uploads_images_with_matching_caption_txt_files(client):
    response = client.post(
        "/references",
        data={"game": "battle-cars", "descriptions": "", "captioning": "manual"},
        files=[
            ("files", ("drone.png", png(), "image/png")),
            ("files", ("pilot.png", png(), "image/png")),
            ("caption_files", ("drone.txt", b"egg-shaped shell", "text/plain")),
            ("caption_files", ("pilot.txt", b"horned imp mask", "text/plain")),
        ],
    )
    assert response.status_code == 200
    library = client.get("/references/battle-cars", headers=BROWSER)
    assert "egg-shaped shell" in library.text
    assert "horned imp mask" in library.text
    assert "2 of 2 references captioned" in library.text


def test_ui_shared_description_fills_every_uncaptioned_image(client):
    client.post(
        "/references",
        data={"game": "battle-cars", "descriptions": "shared caption", "captioning": "manual"},
        files=[
            ("files", ("a.png", png(), "image/png")),
            ("files", ("b.png", png(), "image/png")),
        ],
    )
    library = client.get("/references/battle-cars", headers=BROWSER)
    assert library.text.count("shared caption") == 2


def test_ui_manual_captioning_never_calls_gpt(client):
    # The fixture's agent raises if asked; a 200 proves the upload avoided it.
    response = client.post(
        "/references",
        data={"game": "battle-cars", "descriptions": "", "captioning": "manual"},
        files=[("files", ("a.png", png(), "image/png"))],
    )
    assert response.status_code == 200
    assert client.get("/references/battle-cars", headers=BROWSER).text.count(
        "0 of 1 references captioned"
    )


def test_ui_caption_editor_saves_typed_captions(client):
    client.post(
        "/references",
        data={"game": "battle-cars", "descriptions": "", "captioning": "manual"},
        files=[("files", ("drone.png", png(), "image/png"))],
    )
    saved = client.post(
        "/references/battle-cars/captions",
        data={"caption:drone.png": "  A drone with lime green thruster rings  "},
    )
    assert saved.json()["descriptions"]["drone.png"] == "A drone with lime green thruster rings"
    assert "lime green thruster rings" in client.get(
        "/references/battle-cars", headers=BROWSER
    ).text


def test_ui_caption_file_upload_overwrites_stored_descriptions(client):
    client.post(
        "/references",
        data={"game": "battle-cars", "descriptions": "placeholder", "captioning": "manual"},
        files=[("files", ("drone.png", png(), "image/png"))],
    )
    loaded = client.post(
        "/references/battle-cars/caption-files",
        files=[("caption_files", ("drone.txt", b"egg-shaped shell", "text/plain"))],
    )
    assert loaded.json() == {
        "matched": 1,
        "descriptions": {"drone.png": "egg-shaped shell"},
    }


def test_reference_thumbnails_are_served_and_traversal_is_refused(client):
    client.post(
        "/references",
        data={"game": "battle-cars", "descriptions": "a drone", "captioning": "manual"},
        files=[("files", ("drone.png", png(), "image/png"))],
    )
    assert client.get("/references/battle-cars/image/drone.png").status_code == 200
    assert client.get("/references/battle-cars/image/nope.png").status_code == 404
