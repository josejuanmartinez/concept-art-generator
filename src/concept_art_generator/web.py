from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from .models import ArtRequest, Backend
from .workflow import ConceptArtWorkflow

app = FastAPI(title="Concept Art Generator")
workflow = ConceptArtWorkflow()


def fail(exc: Exception):
    raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/", response_class=HTMLResponse)
def home():
    return """<h1>Concept Art Generator</h1><p>1. Upload references per game. 2. Make a draft. 3. Approve it. 4. Export a 2K final.</p><form action='/references' method='post' enctype='multipart/form-data'><input name='game' placeholder='massive-warfare' required><input type='file' name='files' multiple required><input name='descriptions' placeholder='Optional description (one file only)'><button>Upload references</button></form><hr><form action='/draft' method='post'><input name='game' placeholder='massive-warfare' required><input name='prompt' placeholder='armored scout vehicle' required><select name='backend'><option value='gpt-image-2'>GPT Image 2 (references)</option><option value='huggingface'>Hugging Face Space (LoRA)</option></select><input name='lora_name' placeholder='LoRA name (HF only)'><input name='negative_prompt' placeholder='Negative prompt (HF only)'><input name='seed' type='number' placeholder='Seed (optional)'><input name='steps' type='number' value='28' min='1' max='80'><input name='guidance_scale' type='number' value='4.0' step='0.1'><input name='lora_scale' type='number' value='0.8' step='0.05'><input name='reference_count' type='number' value='16' min='1' max='16'><label><input name='transparent' type='checkbox' checked> Transparent PNG</label><button>Create 1024px draft</button></form><p>Use <code>POST /jobs/{game}/{job_id}/approve</code> then <code>POST /jobs/{game}/{job_id}/final</code>.</p>"""


@app.post("/references")
async def references(
    game: str = Form(),
    files: list[UploadFile] = File(),
    descriptions: list[str] | None = Form(None),
):
    supplied = descriptions or []
    if supplied and len(supplied) not in {1, len(files)}:
        fail(ValueError("Supply either no descriptions, one for one file, or one per file."))
    if len(files) > 1 and len(supplied) == 1:
        fail(ValueError("A single description can only be used with a single uploaded file."))
    copied = []
    for index, upload in enumerate(files):
        filename = Path(upload.filename or "reference.png").name
        temporary = Path("data") / ".uploads" / filename
        temporary.parent.mkdir(parents=True, exist_ok=True)
        try:
            with temporary.open("wb") as handle:
                shutil.copyfileobj(upload.file, handle)
            description = supplied[index] if index < len(supplied) else None
            copied.append(str(workflow.add_reference(game, temporary, description)))
        except (OSError, RuntimeError, ValueError) as exc:
            fail(exc)
        finally:
            temporary.unlink(missing_ok=True)
    return {"references": copied}


@app.post("/draft")
def draft(
    game: str = Form(),
    prompt: str = Form(),
    backend: Backend = Form(),
    lora_name: str | None = Form(None),
    negative_prompt: str = Form(""),
    seed: int | None = Form(None),
    steps: int = Form(28),
    guidance_scale: float = Form(4.0),
    lora_scale: float = Form(0.8),
    reference_count: int = Form(16),
    transparent: bool = Form(True),
):
    try:
        return workflow.create_draft(
            ArtRequest(
                game,
                prompt,
                backend,
                lora_name,
                reference_count,
                transparent=transparent,
                negative_prompt=negative_prompt,
                seed=seed,
                steps=steps,
                guidance_scale=guidance_scale,
                lora_scale=lora_scale,
            )
        ).to_dict()
    except (OSError, RuntimeError, ValueError) as exc:
        fail(exc)


@app.post("/jobs/{game}/{job_id}/approve")
def approve(game: str, job_id: str, feedback: str | None = None):
    try:
        return workflow.approve(game, job_id, feedback).to_dict()
    except (OSError, RuntimeError, ValueError) as exc:
        fail(exc)


@app.post("/jobs/{game}/{job_id}/reject")
def reject(game: str, job_id: str, feedback: str):
    try:
        return workflow.reject(game, job_id, feedback).to_dict()
    except (OSError, RuntimeError, ValueError) as exc:
        fail(exc)


@app.post("/jobs/{game}/{job_id}/final")
def final(game: str, job_id: str):
    try:
        return workflow.create_final(game, job_id).to_dict()
    except (OSError, RuntimeError, ValueError) as exc:
        fail(exc)


@app.get("/assets/{game}/{stage}/{job_id}")
def asset(game: str, stage: str, job_id: str):
    path = workflow.workspace(game).image_path(stage, job_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Asset not found")
    return FileResponse(path, media_type="image/png")


def main() -> None:
    import uvicorn

    uvicorn.run("concept_art_generator.web:app", host="127.0.0.1", port=8000, reload=True)
