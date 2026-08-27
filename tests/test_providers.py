import json

import pytest

from concept_art_generator.providers import HuggingFaceSpaceProvider, RenderSpec


class FakeResponse:
    def __init__(self, body=None, content=b""):
        self.body = body
        self.content = content

    def raise_for_status(self):
        return None

    def json(self):
        return self.body


class FakeEventStream(FakeResponse):
    def __init__(self, lines):
        super().__init__()
        self.lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def iter_lines(self):
        yield from self.lines


@pytest.mark.parametrize(
    ("width", "height", "transparent", "expected_width", "expected_height", "upscale"),
    [
        (512, 512, True, 512, 512, False),
        (2048, 1365, True, 1024, 683, True),
        (2048, 1365, False, 1024, 683, True),
    ],
)
def test_hugging_face_maps_options_and_completes_queued_request(
    monkeypatch,
    width,
    height,
    transparent,
    expected_width,
    expected_height,
    upscale,
):
    captured = {}

    def fake_post(url, *, json, headers, timeout):
        captured.update(submit_url=url, payload=json, headers=headers, submit_timeout=timeout)
        return FakeResponse({"event_id": "event-123"})

    def fake_stream(method, url, *, headers, timeout):
        captured.update(stream_method=method, stream_url=url, stream_timeout=timeout)
        completed = json.dumps(
            [{"url": "https://example.hf.space/gradio_api/file=result.png"}, 42, "details"]
        )
        return FakeEventStream(["event: heartbeat", "", "event: complete", f"data: {completed}"])

    def fake_get(url, *, headers, timeout, follow_redirects):
        captured.update(image_url=url, image_timeout=timeout, follow_redirects=follow_redirects)
        return FakeResponse(content=b"png")

    monkeypatch.setattr("concept_art_generator.providers.httpx.post", fake_post)
    monkeypatch.setattr("concept_art_generator.providers.httpx.stream", fake_stream)
    monkeypatch.setattr("concept_art_generator.providers.httpx.get", fake_get)
    provider = HuggingFaceSpaceProvider("https://example.hf.space", "secret")
    result = provider.render(
        RenderSpec("vehicle", [], width, height, transparent, "owner/game-style", seed=42)
    )

    assert result.png == b"png"
    assert result.provider_request_id == "event-123"
    assert captured["submit_url"].endswith("/gradio_api/call/v2/generate_ui")
    assert captured["stream_url"].endswith("/gradio_api/call/generate_ui/event-123")
    assert captured["payload"]["width"] == expected_width
    assert captured["payload"]["height"] == expected_height
    assert captured["payload"]["remove_background"] is transparent
    assert captured["payload"]["upscale_to_2k"] is upscale
    assert captured["payload"]["guidance"] == 4.0
    assert captured["payload"]["scale"] == 0.8
    assert captured["headers"] == {"Authorization": "Bearer secret"}
    assert captured["image_url"].startswith("https://example.hf.space/")


def test_hugging_face_surfaces_redacted_queue_error(monkeypatch):
    monkeypatch.setattr(
        "concept_art_generator.providers.httpx.post",
        lambda *_args, **_kwargs: FakeResponse({"event_id": "failed-event"}),
    )
    monkeypatch.setattr(
        "concept_art_generator.providers.httpx.stream",
        lambda *_args, **_kwargs: FakeEventStream(
            ["event: error", 'data: {"error": null}']
        ),
    )

    provider = HuggingFaceSpaceProvider("https://example.hf.space", "secret")
    with pytest.raises(RuntimeError, match="inspect its runtime logs"):
        provider.render(RenderSpec("vehicle", [], 512, 512, True, "owner/game-style"))
