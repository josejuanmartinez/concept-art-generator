import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from concept_art_generator import web
from concept_art_generator.models import Backend
from concept_art_generator.providers import DeterministicProvider, RenderedImage, RenderSpec
from concept_art_generator.workflow import ConceptArtWorkflow

# What a browser sends; the JSON API is what curl and agents send.
BROWSER = {"accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"}


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


def draft_form(**overrides) -> dict:
    form = {
        "game": "massive-warfare",
        "prompt": "scout tank",
        "backend": "gpt-image-2",
        "lora_name": "",
        "negative_prompt": "",
        "seed": "",
        "steps": "28",
        "guidance_scale": "4.0",
        "lora_scale": "1.25",
        "background_model": "birefnet-general",
        "reference_count": "16",
        "transparent": "true",
    }
    return {**form, **overrides}


def upload(client, game="massive-warfare", names=("a.png",), description=""):
    return client.post(
        "/references",
        data={"game": game, "descriptions": description},
        files=[("files", (name, png(), "image/png")) for name in names],
    )


def test_several_references_upload_with_the_form_description_left_blank(client):
    response = upload(client, names=("a.png", "b.png", "c.png"))
    assert response.status_code == 200
    assert len(response.json()["references"]) == 3


@pytest.mark.parametrize("backend", ["gpt-image-2", "huggingface"])
def test_browser_can_run_draft_approve_final_for_either_backend(client, backend):
    upload(client)
    lora = "jjmcarrascosa/pilot-mw" if backend == "huggingface" else ""
    created = client.post(
        "/draft",
        data=draft_form(backend=backend, lora_name=lora),
        headers=BROWSER,
        follow_redirects=False,
    )
    assert created.status_code == 303
    job_url = created.headers["location"]

    page = client.get(job_url, headers=BROWSER)
    assert "Approve this draft" in page.text
    assert f"/assets/massive-warfare/drafts/{job_url.rsplit('/', 1)[1]}" in page.text

    approved = client.post(
        job_url + "/approve", data={"feedback": "ok"}, headers=BROWSER, follow_redirects=False
    )
    assert approved.status_code == 303
    assert "Export 2K final" in client.get(job_url, headers=BROWSER).text
    assert (
        client.post(job_url + "/final", headers=BROWSER, follow_redirects=False).status_code == 303
    )

    final_page = client.get(job_url, headers=BROWSER)
    assert "final_ready" in final_page.text
    assert "/assets/massive-warfare/finals/" in final_page.text
    job_id = job_url.rsplit("/", 1)[1]
    assert client.get(f"/assets/massive-warfare/finals/{job_id}").status_code == 200


def test_browser_can_reject_a_draft_with_feedback(client):
    upload(client)
    job_url = client.post(
        "/draft", data=draft_form(), headers=BROWSER, follow_redirects=False
    ).headers["location"]
    client.post(job_url + "/reject", data={"feedback": "Wider silhouette"}, headers=BROWSER)
    assert "Wider silhouette" in client.get(job_url, headers=BROWSER).text


def test_opaque_selection_reaches_the_job(client):
    upload(client)
    response = client.post("/draft", data=draft_form(transparent="false"))
    assert response.json()["transparent"] is False


def test_json_api_stays_available_for_agents_and_curl(client):
    upload(client)
    job = client.post("/draft", data=draft_form()).json()
    approved = client.post(f"/jobs/massive-warfare/{job['id']}/approve").json()
    assert approved["state"] == "approved"
    assert client.post(f"/jobs/massive-warfare/{job['id']}/final").json()["state"] == "final_ready"


def test_documented_reject_query_parameter_still_works(client):
    upload(client)
    job = client.post("/draft", data=draft_form()).json()
    rejected = client.post(f"/jobs/massive-warfare/{job['id']}/reject?feedback=Wider")
    assert rejected.json()["feedback"][0]["feedback"] == "Wider"


def test_reject_without_feedback_is_refused(client):
    upload(client)
    job = client.post("/draft", data=draft_form()).json()
    assert client.post(f"/jobs/massive-warfare/{job['id']}/reject").status_code == 400


def test_dashboard_lists_every_game_job(client):
    upload(client)
    upload(client, game="battle-cars", names=("bc.png",))
    client.post("/draft", data=draft_form())
    client.post("/draft", data=draft_form(game="battle-cars", prompt="armored racer"))
    home = client.get("/", headers=BROWSER)
    assert "massive-warfare" in home.text
    assert "battle-cars" in home.text
    assert "armored racer" in home.text


def test_assets_route_rejects_a_non_image_stage(client):
    assert client.get("/assets/massive-warfare/references/anything").status_code == 404
