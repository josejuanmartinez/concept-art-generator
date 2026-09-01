"""The JSON HTTP API used by curl, scripts and agents."""

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from concept_art_generator import web
from concept_art_generator.models import Backend
from concept_art_generator.providers import DeterministicProvider, RenderedImage, RenderSpec
from concept_art_generator.workflow import ConceptArtWorkflow


class OfflineHuggingFace(DeterministicProvider):
    """Offline stand-in that echoes the replayable parameters a real Space returns."""

    backend = Backend.HUGGING_FACE

    def render(self, spec: RenderSpec) -> RenderedImage:
        rendered = super().render(spec)
        return RenderedImage(
            rendered.png,
            rendered.estimated_cost_usd,
            rendered.provider_request_id,
            {
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
            },
        )


class OfflineReferenceAgent:
    def describe(self, image_path: Path) -> str:
        return f"offline description for {image_path.name}"

    def select(self, prompt, candidates, limit):
        return [path.name for path, _description in candidates][:limit]


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(
        web,
        "workflow",
        ConceptArtWorkflow(
            tmp_path,
            {
                Backend.GPT_IMAGE_2: DeterministicProvider(),
                Backend.HUGGING_FACE: OfflineHuggingFace(),
            },
            reference_agent=OfflineReferenceAgent(),
        ),
    )
    return TestClient(web.app)


def png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGBA", (16, 16), (255, 0, 0, 255)).save(buffer, "PNG")
    return buffer.getvalue()


def upload(client, art_model="pilot-mw", names=("a.png",)):
    return client.post(
        f"/api/models/{art_model}/references",
        data={"captioning": "gpt"},
        files=[("files", (name, png(), "image/png")) for name in names],
    )


def draft_form(**overrides) -> dict:
    return {
        "prompt": "A pilot undead soldier",
        "backend": "gpt-image-2",
        "negative_prompt": "",
        "seed": "",
        "steps": "28",
        "guidance_scale": "4.0",
        "lora_scale": "1.25",
        "background_model": "birefnet-general",
        "reference_count": "16",
        "transparent": "true",
        **overrides,
    }


def test_catalogue_is_served(client):
    entries = client.get("/api/models").json()["art_models"]
    assert [entry["art_model"] for entry in entries] == ["drone-bc", "pilot-bc", "pilot-mw"]


@pytest.mark.parametrize("backend", ["gpt-image-2", "huggingface"])
def test_draft_approve_final_for_either_backend(client, backend):
    upload(client)
    job = client.post("/api/models/pilot-mw/draft", data=draft_form(backend=backend)).json()
    assert job["state"] == "draft_ready"
    assert job["art_model"] == "pilot-mw"

    approved = client.post(f"/api/models/pilot-mw/jobs/{job['id']}/approve").json()
    assert approved["state"] == "approved"

    final = client.post(f"/api/models/pilot-mw/jobs/{job['id']}/final").json()
    assert final["state"] == "final_ready"
    assert client.get(f"/assets/pilot-mw/finals/{job['id']}").status_code == 200
    assert client.get(f"/assets/pilot-mw/drafts/{job['id']}").status_code == 200


def test_the_final_is_refused_before_approval(client):
    upload(client)
    job = client.post("/api/models/pilot-mw/draft", data=draft_form()).json()
    refused = client.post(f"/api/models/pilot-mw/jobs/{job['id']}/final")
    assert refused.status_code == 400
    assert "approval" in refused.json()["detail"]


def test_reject_records_feedback_and_blocks_the_final(client):
    upload(client)
    job = client.post("/api/models/pilot-mw/draft", data=draft_form()).json()
    rejected = client.post(
        f"/api/models/pilot-mw/jobs/{job['id']}/reject?feedback=Wider%20silhouette"
    ).json()
    assert rejected["state"] == "rejected"
    assert rejected["feedback"][0]["feedback"] == "Wider silhouette"
    assert client.post(f"/api/models/pilot-mw/jobs/{job['id']}/final").status_code == 400


def test_reject_without_feedback_is_refused(client):
    upload(client)
    job = client.post("/api/models/pilot-mw/draft", data=draft_form()).json()
    assert client.post(f"/api/models/pilot-mw/jobs/{job['id']}/reject").status_code == 400


def test_opaque_selection_reaches_the_job(client):
    upload(client)
    job = client.post("/api/models/pilot-mw/draft", data=draft_form(transparent="false")).json()
    assert job["transparent"] is False


def test_jobs_are_listed_per_art_model(client):
    upload(client)
    upload(client, art_model="drone-bc", names=("b.png",))
    client.post("/api/models/pilot-mw/draft", data=draft_form())
    client.post("/api/models/drone-bc/draft", data=draft_form(prompt="A drone"))

    mw = client.get("/api/models/pilot-mw/jobs").json()["jobs"]
    bc = client.get("/api/models/drone-bc/jobs").json()["jobs"]
    assert len(mw) == 1
    assert len(bc) == 1
    assert mw[0]["prompt"] != bc[0]["prompt"]


def test_an_uncatalogued_art_model_is_refused(client):
    assert client.get("/api/models/drone-mw/jobs").status_code == 404
    assert client.post("/api/models/drone-mw/draft", data=draft_form()).status_code == 400


def test_assets_route_rejects_a_non_image_stage(client):
    assert client.get("/assets/pilot-mw/references/anything").status_code == 404


def test_the_api_serves_no_markup():
    """The human front end is Gradio; this app is JSON and images only."""
    source = Path(web.__file__).read_text(encoding="utf-8")
    for markup in ("<form", "<div", "<table", "<html", "<script"):
        assert markup not in source


def test_the_mounted_gradio_app_exposes_the_stop_event_shutdown_needs(tmp_path):
    """Ctrl+C hangs on the heartbeat SSE streams unless this event is found and set.

    `stop_events` walks Gradio internals, so this is the test that fails loudly if a Gradio
    upgrade moves them — otherwise the hang comes back quietly.
    """
    import asyncio

    import gradio as gr
    from fastapi import FastAPI

    from concept_art_generator.ui import build_ui

    flow = ConceptArtWorkflow(tmp_path, {Backend.HUGGING_FACE: OfflineHuggingFace()})
    mounted = gr.mount_gradio_app(FastAPI(), build_ui(flow), path="/")

    events = web.stop_events(mounted)
    assert events, "Gradio no longer exposes stop_event on the mounted app"
    for event in events:
        assert isinstance(event, asyncio.Event)
        assert not event.is_set()
        event.set()


def test_stop_events_is_empty_rather_than_raising_on_a_plain_app():
    from fastapi import FastAPI

    assert web.stop_events(FastAPI()) == []
