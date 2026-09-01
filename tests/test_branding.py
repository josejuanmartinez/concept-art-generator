"""The per-art-model logo composite."""

from pathlib import Path

import pytest
from PIL import Image

from concept_art_generator.branding import (
    LOGO_MARGIN_AT_1024,
    LOGO_WIDTH_AT_1024,
    MODEL_LOGOS,
    logo_for,
    stamp,
    stamp_bytes,
)
from concept_art_generator.models import ArtRequest, Backend
from concept_art_generator.providers import DeterministicProvider
from concept_art_generator.workflow import ConceptArtWorkflow


def art(size: int = 1024) -> Image.Image:
    return Image.new("RGBA", (size, size), (0, 0, 0, 0))


def flow(tmp_path: Path) -> ConceptArtWorkflow:
    return ConceptArtWorkflow(tmp_path, {Backend.GPT_IMAGE_2: DeterministicProvider()})


def reference(path: Path) -> None:
    Image.new("RGBA", (16, 16), (255, 0, 0, 255)).save(path)


@pytest.mark.parametrize("art_model", sorted(MODEL_LOGOS))
def test_every_branded_model_ships_its_own_logo(art_model: str):
    path = logo_for(art_model)
    assert path is not None and path.exists()
    assert path.name == f"{art_model}.png"


@pytest.mark.parametrize("art_model", sorted(MODEL_LOGOS))
def test_no_logo_carries_its_own_background(art_model: str):
    """The supplied pilot-mw art was a white mark on an opaque black rectangle, which stamps a
    black box onto a transparent asset. It was matted before vendoring; this keeps a raw
    re-copy of either logo from quietly bringing the box back."""
    with Image.open(logo_for(art_model)) as logo:
        alpha = logo.convert("RGBA").getchannel("A")
    assert alpha.getextrema()[0] == 0, "the logo has no transparent pixels at all"
    corners = [(0, 0), (alpha.width - 1, 0), (0, alpha.height - 1), (alpha.width - 1, alpha.height - 1)]
    assert [alpha.getpixel(corner) for corner in corners] == [0, 0, 0, 0]


def test_drone_bc_is_unbranded_and_its_art_is_untouched():
    original = art()
    stamped, logo = stamp(original, "drone-bc")
    assert logo is None
    assert stamped is original


def test_the_logo_lands_in_the_bottom_left_corner():
    stamped, _ = stamp(art(), "pilot-bc")
    alpha = stamped.getchannel("A")
    marked = alpha.getbbox()
    assert marked is not None
    left, upper, right, lower = marked
    assert left == LOGO_MARGIN_AT_1024
    assert lower == 1024 - LOGO_MARGIN_AT_1024
    assert right - left == LOGO_WIDTH_AT_1024
    # Bottom-left, not centred: the stamp stays clear of the opposite half of the canvas.
    assert right < 1024 / 2 and upper > 1024 / 2


def test_the_logo_keeps_its_aspect_ratio():
    with Image.open(logo_for("pilot-bc")) as source:
        expected = round(source.height * LOGO_WIDTH_AT_1024 / source.width)
    stamped, _ = stamp(art(), "pilot-bc")
    left, upper, right, lower = stamped.getchannel("A").getbbox()
    assert (right - left, lower - upper) == (LOGO_WIDTH_AT_1024, expected)


@pytest.mark.parametrize(
    ("size", "width"), [(512, 125), (1024, LOGO_WIDTH_AT_1024), (2048, 500)]
)
def test_the_logo_scales_with_the_image_so_branding_reads_the_same(size: int, width: int):
    """250px at 1024 is a ratio, not a pixel count: the 2K final would otherwise be stamped
    at half the intended size."""
    stamped, _ = stamp(art(size), "pilot-bc")
    left, _, right, _ = stamped.getchannel("A").getbbox()
    assert right - left == width


def test_stamp_bytes_leaves_an_unbranded_models_bytes_byte_for_byte_alone():
    png = b"not even a real image"
    assert stamp_bytes(png, "drone-bc") == (png, None)


def test_a_branded_draft_records_the_logo_it_was_given(tmp_path: Path):
    workflow = flow(tmp_path)
    source = tmp_path / "reference.png"
    reference(source)
    workflow.add_reference("pilot-mw", source, "Burning skull soldier")

    job = workflow.create_draft(ArtRequest("pilot-mw", "A pilot", Backend.GPT_IMAGE_2))
    workflow.approve("pilot-mw", job.id)
    final = workflow.create_final("pilot-mw", job.id)

    assert final.artifacts["draft"]["logo"] == "pilot-mw.png"
    assert final.artifacts["final"]["logo"] == "pilot-mw.png"
    assert any("bottom-left corner of the final" in note for note in final.notes)
    assert Image.open(final.final_path).size == (2048, 2048)


def test_an_unbranded_final_records_no_logo(tmp_path: Path):
    workflow = flow(tmp_path)
    source = tmp_path / "reference.png"
    reference(source)
    workflow.add_reference("drone-bc", source, "White egg-shaped shell")

    job = workflow.create_draft(ArtRequest("drone-bc", "A drone", Backend.GPT_IMAGE_2))
    workflow.approve("drone-bc", job.id)
    final = workflow.create_final("drone-bc", job.id)

    assert final.artifacts["final"]["logo"] is None
    assert not any("Stamped" in note for note in final.notes)
