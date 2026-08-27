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
    return """<h1>Concept Art Generator</h1><p>1. Upload references per game. 2. Make a draft. 3. Approve it. 4. Export a 2K final.</p><form action='/references' method='post' enctype='multipart/form-data'><input name='game' placeholder='massive-warfare' required><input type='file' name='files' multiple required><button>Upload references</button></form><hr><form action='/draft' method='post'><input name='game' placeholder='massive-warfare' required><input name='prompt' placeholder='armored scout vehicle' required><select name='backend'><option value='gpt-image-2'>GPT Image 2 (references)</option><option value='huggingface'>Hugging Face Space (LoRA)</option></select><input name='lora_name' placeholder='LoRA name (HF only)'><input name='reference_count' type='number' value='4' min='1' max='10'><label><input name='transparent' type='checkbox' checked> Transparent PNG</label><button>Create low-res draft</button></form><p>Use <code>POST /jobs/{game}/{job_id}/approve</code> then <code>POST /jobs/{game}/{job_id}/final</code>.</p>"""


@app.post("/references")
async def references(game: str = Form(), files: list[UploadFile] = File()):
    copied = []
    for upload in files:
        filename = Path(upload.filename or "reference.png").name
        temporary = Path("data") / ".uploads" / filename
        temporary.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open("wb") as handle:
            shutil.copyfileobj(upload.file, handle)
        copied.append(str(workflow.add_reference(game, temporary)))
        temporary.unlink(missing_ok=True)
    return {"references": copied}


@app.post("/draft")
def draft(
    game: str = Form(),
    prompt: str = Form(),
    backend: Backend = Form(),
    lora_name: str | None = Form(None),
    reference_count: int = Form(4),
    transparent: bool = Form(True),
):
    try:
        return workflow.create_draft(
            ArtRequest(game, prompt, backend, lora_name, reference_count, transparent=transparent)
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
