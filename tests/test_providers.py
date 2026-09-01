import base64
from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image

from concept_art_generator.providers import GPTImage2Provider, HuggingFaceSpaceProvider, RenderSpec


class FakeResponse:
    def __init__(self, body=None, content=b"", headers=None):
        self.body = body
        self.content = content
        self.headers = headers or {}

    def raise_for_status(self):
        return None

    def json(self):
        return self.body


@pytest.mark.parametrize(
    ("width", "height", "transparent", "expected_width", "expected_height", "upscale"),
    [
        (1024, 1024, True, 1024, 1024, False),
        (2048, 2048, True, 1024, 1024, True),
        (2048, 2048, False, 1024, 1024, True),
    ],
)
def test_hugging_face_maps_options_and_returns_replayable_parameters(
    monkeypatch,
    width,
    height,
    transparent,
    expected_width,
    expected_height,
    upscale,
):
    captured = {}

    image = BytesIO()
    Image.new("RGBA", (2048 if upscale else 1024, 2048 if upscale else 1024), (1, 2, 3, 0)).save(
        image, "PNG"
    )
    parameters = {
        "prompt": "vehicle",
        "lora_name": "owner/game-style",
        "negative_prompt": "blurry",
        "steps": 28,
        "guidance_scale": 4.0,
        "lora_scale": 1.25,
        "seed": 42,
        "scheduler": "FlowMatchEulerDiscreteScheduler",
        "base_model": "Qwen/Qwen-Image-2512",
        "background_model": "birefnet-general",
    }

    def fake_post(url, *, json, headers, timeout):
        captured.update(submit_url=url, payload=json, headers=headers, submit_timeout=timeout)
        return FakeResponse(
            {
                "image_base64": base64.b64encode(image.getvalue()).decode(),
                "generation_parameters": parameters,
            },
            headers={"x-request-id": "request-123"},
        )

    monkeypatch.setattr("concept_art_generator.providers.httpx.post", fake_post)
    provider = HuggingFaceSpaceProvider("https://example.hf.space", "secret")
    result = provider.render(
        RenderSpec(
            "vehicle",
            [],
            width,
            height,
            transparent,
            "owner/game-style",
            seed=42,
            negative_prompt="blurry",
        )
    )

    assert Image.open(BytesIO(result.png)).size == (
        2048 if upscale else 1024,
        2048 if upscale else 1024,
    )
    assert result.provider_request_id == "request-123"
    assert result.generation_parameters == parameters
    assert captured["submit_url"].endswith("/v1/generate")
    assert captured["payload"]["width"] == expected_width
    assert captured["payload"]["height"] == expected_height
    assert captured["payload"]["remove_background"] is transparent
    assert captured["payload"]["upscale_to_2k"] is upscale
    assert captured["payload"]["guidance_scale"] == 4.0
    assert captured["payload"]["lora_scale"] == 1.25
    assert captured["payload"]["background_model"] == "birefnet-general"
    assert captured["payload"]["negative_prompt"] == "blurry"
    assert captured["payload"]["steps"] == 28
    assert captured["payload"]["seed"] == 42
    assert captured["headers"] == {"Authorization": "Bearer secret"}


def test_hugging_face_requires_replayable_parameters(monkeypatch):
    monkeypatch.setattr(
        "concept_art_generator.providers.httpx.post",
        lambda *_args, **_kwargs: FakeResponse({"image_base64": "cG5n"}),
    )

    provider = HuggingFaceSpaceProvider("https://example.hf.space", "secret")
    with pytest.raises(RuntimeError, match="replayable generation parameters"):
        provider.render(RenderSpec("vehicle", [], 1024, 1024, True, "owner/game-style"))


def test_gpt_image_2_rejects_more_than_16_references(tmp_path):
    references = [tmp_path / f"{index}.png" for index in range(17)]
    with pytest.raises(ValueError, match="at most 16"):
        GPTImage2Provider().render(RenderSpec("vehicle", references, 512, 512, True))


def test_gpt_image_2_uses_and_retains_supported_low_quality_draft_size(
    monkeypatch, tmp_path
):
    reference = tmp_path / "reference.png"
    Image.new("RGBA", (16, 16), (255, 0, 0, 255)).save(reference)
    generated = tmp_path / "generated.png"
    Image.new("RGBA", (1024, 1024), (1, 2, 3, 0)).save(generated)
    captured = {}

    class FakeImages:
        def edit(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                data=[SimpleNamespace(b64_json=base64.b64encode(generated.read_bytes()).decode())],
                _request_id="image-request",
            )

    monkeypatch.setattr("openai.OpenAI", lambda: SimpleNamespace(images=FakeImages()))

    result = GPTImage2Provider().render(
        RenderSpec("vehicle", [reference], 512, 512, True)
    )

    assert captured["size"] == "1024x1024"
    assert captured["quality"] == "low"
    assert Image.open(BytesIO(result.png)).size == (1024, 1024)
