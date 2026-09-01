from __future__ import annotations


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
