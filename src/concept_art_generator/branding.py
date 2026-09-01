"""The game logo stamped into every image an art model produces.

Branding is a property of the art model, exactly like its LoRA and its references: `pilot-bc`
carries the Battle Cars logo, `pilot-mw` carries the Mech Warfare one, and `drone-bc` carries
none. The logos are vendored under `assets/logos/` and named after the model they belong to, so
the mapping is not a lookup anyone has to keep in their head — and, as with references, one
model's logo is never placed on another's art.

The composite is a post-process, not a prompt instruction: the provider never sees the logo, so
it cannot redraw, warp or invent lettering.

A vendored logo must carry its own transparency. The supplied `pilot-mw` artwork was a white mark
on an opaque black rectangle, which stamps a black box onto a transparent asset; it was matted
before vendoring by taking its luminance as the alpha channel, which keeps the anti-aliased edges
a hard black key would have fringed, and cropped to the mark. `test_branding.py` holds both logos
to that.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from PIL import Image

LOGO_DIR = Path(__file__).parent / "assets" / "logos"

# Only these two art models are branded. A model absent from this table is left untouched.
MODEL_LOGOS: dict[str, str] = {
    "pilot-bc": "pilot-bc.png",
    "pilot-mw": "pilot-mw.png",
}

# The brief is "250px wide on a 1024px image". Both halves of that are kept rather than just the
# pixel count, because it is the ratio that has to hold on the 2K final: a logo pinned at 250px
# would read half its intended size there. 1024 → 250px, 2048 → 500px, 512 → 125px.
LOGO_WIDTH_AT_1024 = 250
LOGO_MARGIN_AT_1024 = 32
REFERENCE_WIDTH = 1024


def logo_for(art_model: str) -> Path | None:
    """This model's vendored logo, or None when it is unbranded."""
    filename = MODEL_LOGOS.get(art_model)
    if filename is None:
        return None
    path = LOGO_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"{art_model} is a branded art model but {path} is missing.")
    return path


def stamp(image: Image.Image, art_model: str) -> tuple[Image.Image, str | None]:
    """Composite this model's logo into the bottom-left corner, scaled to the image.

    Returns the image and the logo filename that was applied, which is None — and the image
    unchanged — for an unbranded art model.
    """
    path = logo_for(art_model)
    if path is None:
        return image, None
    scale = image.width / REFERENCE_WIDTH
    width = max(1, round(LOGO_WIDTH_AT_1024 * scale))
    margin = max(0, round(LOGO_MARGIN_AT_1024 * scale))
    with Image.open(path) as source:
        logo = source.convert("RGBA")
    height = max(1, round(logo.height * width / logo.width))
    logo = logo.resize((width, height), Image.Resampling.LANCZOS)
    canvas = image.convert("RGBA")
    # alpha_composite, not paste: a logo with soft or transparent edges blends into the art
    # instead of punching its own bounding box through a transparent asset.
    canvas.alpha_composite(logo, (margin, max(0, canvas.height - height - margin)))
    return canvas, path.name


def stamp_bytes(png: bytes, art_model: str) -> tuple[bytes, str | None]:
    """`stamp` over encoded PNG bytes; an unbranded model's bytes are returned untouched."""
    if art_model not in MODEL_LOGOS:
        return png, None
    with Image.open(BytesIO(png)) as image:
        stamped, filename = stamp(image, art_model)
    buffer = BytesIO()
    stamped.save(buffer, "PNG")
    return buffer.getvalue(), filename
