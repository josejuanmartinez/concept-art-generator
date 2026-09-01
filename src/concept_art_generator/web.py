"""The JSON HTTP API, for scripts, curl and agents.

The human front end is Gradio (`ui.py`), mounted onto this same app by `main()`. Nothing here
renders markup: every route returns JSON or an image file.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from .art_models import catalogue
from .models import DEFAULT_BACKGROUND_MODEL, ArtRequest, Backend
from .references import caption_for, load_caption_sidecars
from .workflow import ConceptArtWorkflow

app = FastAPI(title="Concept Art Generator")
workflow = ConceptArtWorkflow()

STAGING = Path("data") / ".uploads"


def fail(exc: Exception):
    raise HTTPException(status_code=400, detail=str(exc)) from exc


def not_found(exc: Exception):
    raise HTTPException(status_code=404, detail=str(exc)) from exc


def save_uploads(uploads: list[UploadFile], folder: Path) -> list[Path]:
    saved = []
    folder.mkdir(parents=True, exist_ok=True)
    for upload in uploads:
        target = folder / Path(upload.filename or "upload").name
        with target.open("wb") as handle:
            shutil.copyfileobj(upload.file, handle)
        saved.append(target)
    return saved


@app.get("/api/models")
def art_models():
    return {"art_models": catalogue()}


@app.get("/api/models/{art_model}/jobs")
def jobs(art_model: str):
    try:
        return {"jobs": [job.to_dict() for job in workflow.list_jobs(art_model)]}
    except (OSError, ValueError) as exc:
        not_found(exc)


@app.get("/api/models/{art_model}/references")
def references_index(art_model: str):
    try:
        return {"descriptions": workflow.reference_descriptions(art_model)}
    except (OSError, ValueError) as exc:
        not_found(exc)


@app.post("/api/models/{art_model}/references")
async def add_references(
    art_model: str,
    files: list[UploadFile] = File(),
    caption_files: list[UploadFile] | None = File(None),
    captioning: str = Form("gpt"),
):
    """Add reference images. Captions are per-image only: an uploaded `.txt` matched by
    filename, GPT, or typed afterwards through `POST /captions`."""
    sidecars: dict[str, str] = {}
    copied = []
    try:
        uploaded = [item for item in (caption_files or []) if (item.filename or "").strip()]
        if uploaded:
            sidecars = load_caption_sidecars(save_uploads(uploaded, STAGING))
        for upload in files:
            filename = Path(upload.filename or "reference.png").name
            target = STAGING / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as handle:
                shutil.copyfileobj(upload.file, handle)
            try:
                copied.append(
                    str(
                        workflow.add_reference(
                            art_model,
                            target,
                            caption_for(filename, sidecars),
                            describe_missing=captioning == "gpt",
                        )
                    )
                )
            except (OSError, RuntimeError, ValueError) as exc:
                fail(exc)
    finally:
        shutil.rmtree(STAGING, ignore_errors=True)
    return {"references": copied}


@app.post("/api/models/{art_model}/captions")
async def save_captions(art_model: str, request: Request):
    saved = {}
    for filename, value in (await request.form()).items():
        try:
            workflow.set_reference_description(art_model, filename, str(value).strip())
        except (OSError, ValueError) as exc:
            fail(exc)
        saved[filename] = str(value).strip()
    return {"descriptions": saved}


@app.post("/api/models/{art_model}/caption-files")
async def load_caption_files(art_model: str, caption_files: list[UploadFile] = File()):
    try:
        matched = workflow.apply_caption_files(art_model, save_uploads(caption_files, STAGING))
    except (OSError, ValueError) as exc:
        fail(exc)
    finally:
        shutil.rmtree(STAGING, ignore_errors=True)
    return {"matched": matched, "descriptions": workflow.reference_descriptions(art_model)}


@app.post("/api/models/{art_model}/draft")
def draft(
    art_model: str,
    prompt: str = Form(),
    backend: Backend = Form(),
    negative_prompt: str = Form(""),
    seed: int | None = Form(None),
    steps: int = Form(28),
    guidance_scale: float = Form(4.0),
    lora_scale: float = Form(1.25),
    background_model: str = Form(DEFAULT_BACKGROUND_MODEL),
    reference_count: int = Form(16),
    transparent: bool = Form(True),
):
    try:
        return workflow.create_draft(
            ArtRequest(
                art_model,
                prompt,
                backend,
                reference_count,
                transparent=transparent,
                negative_prompt=negative_prompt,
                seed=seed,
                steps=steps,
                guidance_scale=guidance_scale,
                lora_scale=lora_scale,
                background_model=background_model or DEFAULT_BACKGROUND_MODEL,
            )
        ).to_dict()
    except (OSError, RuntimeError, ValueError) as exc:
        fail(exc)


async def read_feedback(request: Request) -> str | None:
    """Accept feedback from a form body or from the documented `?feedback=` query."""
    if "feedback" in request.query_params:
        return request.query_params["feedback"]
    if request.headers.get("content-type", "").startswith(
        ("application/x-www-form-urlencoded", "multipart/form-data")
    ):
        value = (await request.form()).get("feedback")
        if value is not None:
            return str(value)
    return None


@app.post("/api/models/{art_model}/jobs/{job_id}/approve")
async def approve(art_model: str, job_id: str, request: Request):
    try:
        return workflow.approve(art_model, job_id, await read_feedback(request)).to_dict()
    except (OSError, RuntimeError, ValueError) as exc:
        fail(exc)


@app.post("/api/models/{art_model}/jobs/{job_id}/reject")
async def reject(art_model: str, job_id: str, request: Request):
    try:
        return workflow.reject(art_model, job_id, await read_feedback(request) or "").to_dict()
    except (OSError, RuntimeError, ValueError) as exc:
        fail(exc)


@app.post("/api/models/{art_model}/jobs/{job_id}/final")
def final(art_model: str, job_id: str):
    try:
        return workflow.create_final(art_model, job_id).to_dict()
    except (OSError, RuntimeError, ValueError) as exc:
        fail(exc)


@app.get("/assets/{art_model}/{stage}/{job_id}")
def asset(art_model: str, stage: str, job_id: str):
    if stage not in {"drafts", "finals"}:
        raise HTTPException(status_code=404, detail="Unknown asset stage")
    path = workflow.workspace(art_model).image_path(stage, job_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Asset not found")
    return FileResponse(path, media_type="image/png")


def main() -> None:
    """Serve the Gradio front end at / with this JSON API mounted alongside it."""
    import gradio as gr
    import uvicorn

    from .ui import APP_CSS, build_ui

    uvicorn.run(
        gr.mount_gradio_app(app, build_ui(workflow), path="/", css=APP_CSS),
        host="127.0.0.1",
        port=8000,
    )
