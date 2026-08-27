from __future__ import annotations

from .models import ArtRequest
from .workspace import GameWorkspace


class StyleDirector:
    """The read-only planning subagent: sees only one game's references."""

    def brief(self, request: ArtRequest, workspace: GameWorkspace) -> str:
        count = min(len(workspace.reference_files()), request.reference_count)
        if not count:
            raise ValueError(f"{request.game} has no references. Add them before generating.")
        logo = (
            " Preserve the supplied logo exactly; do not redraw, warp, or invent lettering."
            if request.logo_path
            else ""
        )
        return (
            f"Create production-ready concept art for {request.game}. Match only the visual language, "
            f"silhouette discipline, materials, palette and rendering cues from the {count} approved "
            f"reference image(s) supplied for this game. Subject: {request.prompt}. "
            "Single isolated subject, centered, no scenery, no text, no watermark, transparent background."
            f"{logo}"
        )


class QualityGate:
    """The verifier subagent. It blocks final delivery when alpha is absent."""

    @staticmethod
    def require_transparency(png: bytes) -> None:
        from io import BytesIO

        from PIL import Image

        image = Image.open(BytesIO(png)).convert("RGBA")
        alpha = image.getchannel("A")
        if alpha.getextrema()[0] == 255:
            raise ValueError(
                "Provider returned an opaque image; it cannot be exported as a transparent asset."
            )
