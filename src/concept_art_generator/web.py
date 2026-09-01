from __future__ import annotations

import html
import json
import shutil
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse

from .loras import LORA_MODELS
from .models import DEFAULT_BACKGROUND_MODEL, ArtJob, ArtRequest, Backend, JobState
from .references import caption_for, load_caption_sidecars
from .workflow import ConceptArtWorkflow

app = FastAPI(title="Concept Art Generator")
workflow = ConceptArtWorkflow()

STYLE = """
:root { color-scheme: light dark; }
body { font: 15px/1.5 system-ui, sans-serif; margin: 0 auto; max-width: 62rem; padding: 1.5rem; }
h1, h2, h3 { line-height: 1.2; }
form { border: 1px solid #8884; border-radius: 8px; padding: 1rem; margin: 0 0 1rem; }
label { display: block; margin: .5rem 0 .15rem; font-weight: 600; font-size: .85rem; }
input, select, textarea { width: 100%; padding: .4rem; box-sizing: border-box; font: inherit; }
textarea { min-height: 5rem; resize: vertical; }
button { margin-top: .8rem; padding: .5rem 1rem; cursor: pointer; }
.grid { display: grid; gap: .6rem 1rem; grid-template-columns: repeat(auto-fit, minmax(11rem, 1fr)); }
.row { display: flex; flex-wrap: wrap; gap: 1rem; align-items: flex-start; }
.row form { flex: 1 1 16rem; }
table { border-collapse: collapse; width: 100%; }
th, td { border-bottom: 1px solid #8884; padding: .35rem .5rem; text-align: left; font-size: .9rem;
  vertical-align: top; }
img.preview { max-width: 320px; border: 1px solid #8884; border-radius: 6px; }
img.thumb { width: 96px; height: 96px; object-fit: contain; border-radius: 4px; }
img.preview, img.thumb {
  background: repeating-conic-gradient(#8883 0% 25%, transparent 0% 50%) 50% / 16px 16px; }
code, pre { background: #8882; border-radius: 4px; padding: .1rem .3rem; }
pre { padding: .6rem; overflow-x: auto; white-space: pre-wrap; }
.state { font-weight: 700; }
.hint { color: #8889; font-size: .85rem; }
details { margin: .6rem 0; font-size: .85rem; }
details p { margin: .4rem 0; }
"""


def page(title: str, body: str) -> HTMLResponse:
    return HTMLResponse(
        f"<!doctype html><meta charset='utf-8'><meta name='viewport' "
        f"content='width=device-width,initial-scale=1'><title>{html.escape(title)}</title>"
        f"<style>{STYLE}</style><body><p><a href='/'>&larr; Concept Art Generator</a></p>{body}"
    )


def fail(exc: Exception):
    raise HTTPException(status_code=400, detail=str(exc)) from exc


def wants_html(request: Request) -> bool:
    return "text/html" in request.headers.get("accept", "")


def respond(request: Request, job: ArtJob):
    """Browsers get the job page; agents and curl get the same JSON as before."""
    if wants_html(request):
        return RedirectResponse(f"/jobs/{job.game}/{job.id}", status_code=303)
    return job.to_dict()


def save_uploads(uploads: list[UploadFile], folder: Path) -> list[Path]:
    saved = []
    folder.mkdir(parents=True, exist_ok=True)
    for upload in uploads:
        target = folder / Path(upload.filename or "upload").name
        with target.open("wb") as handle:
            shutil.copyfileobj(upload.file, handle)
        saved.append(target)
    return saved


def lora_examples() -> str:
    items = "".join(
        f"<p><code>{model.slug}</code> &middot; {html.escape(model.game)} &middot; starts "
        f"<em>{html.escape(model.subject_keyword)}</em><br>{html.escape(model.example_prompt)}</p>"
        for model in LORA_MODELS
    )
    return (
        "<details><summary>The three LoRAs and the prompt style each expects</summary>"
        f"{items}<p class='hint'>Write the subject the same way: the subject keyword first, then "
        "the distinguishing details, then that LoRA's render-style keywords. The trigger word "
        "(<code>drone-bc</code>, <code>pilot-bc</code>, <code>pilot-mw</code>) is added for you, "
        "and the style keywords are appended if you leave them out.</p></details>"
    )


def draft_form(game: str = "") -> str:
    backends = "".join(
        f"<option value='{b.value}'>{label}</option>"
        for b, label in (
            (Backend.GPT_IMAGE_2, "GPT Image 2 (sends game-local references)"),
            (Backend.HUGGING_FACE, "Hugging Face Space (private LoRA)"),
        )
    )
    loras = "<option value=''>None (GPT Image 2)</option>" + "".join(
        f"<option value='{model.slug}'>{model.slug} &mdash; {model.game}</option>"
        for model in LORA_MODELS
    )
    prompts = json.dumps({model.slug: model.example_prompt for model in LORA_MODELS})
    example = html.escape(LORA_MODELS[0].example_prompt, quote=True)
    return f"""<form action='/draft' method='post'>
<h3>2. Generate a 1024px draft</h3>
<div class='grid'>
<div><label>Game</label><input name='game' value='{html.escape(game)}'
  placeholder='battle-cars' required></div>
<div><label>Backend</label><select name='backend'>{backends}</select></div>
<div><label>Transparency</label><select name='transparent'>
  <option value='true'>Transparent PNG</option><option value='false'>Opaque</option>
</select></div>
<div><label>References to use</label>
  <input name='reference_count' type='number' value='16' min='1' max='16'></div>
</div>
<label>Prompt</label><textarea name='prompt' id='prompt' required
  placeholder='{example}'></textarea>
<div class='grid'>
<div><label>LoRA (Hugging Face only &mdash; only these three exist)</label>
  <select name='lora_name' id='lora'>{loras}</select></div>
<div><label>Negative prompt (HF only)</label><input name='negative_prompt'></div>
<div><label>Seed (HF only, optional)</label><input name='seed' type='number'></div>
<div><label>Steps (HF only)</label><input name='steps' type='number' value='28' min='1' max='80'></div>
<div><label>Guidance (HF only)</label>
  <input name='guidance_scale' type='number' value='4.0' step='0.1'></div>
<div><label>LoRA scale (HF only)</label>
  <input name='lora_scale' type='number' value='1.25' step='0.05' min='0' max='2'></div>
<div><label>Background model (HF only)</label>
  <input name='background_model' value='{DEFAULT_BACKGROUND_MODEL}'></div>
</div>
{lora_examples()}
<p class='hint'>Each LoRA may only be used for the game it was trained on. The prompt sent to the
Space is restricted to the shape that LoRA was trained on:
<code>&lt;trigger&gt; &lt;subject&gt;, &lt;style keywords&gt;</code>. GPT Image 2 takes no LoRA and
no trigger word &mdash; its prompt is built from the style extracted from this game's references.</p>
<button>Create draft</button></form>
<script>
const examplePrompts = {prompts};
const exampleValues = Object.values(examplePrompts);
document.getElementById('lora').addEventListener('change', function (event) {{
  const prompt = document.getElementById('prompt');
  const example = examplePrompts[event.target.value];
  // Follow the chosen LoRA while the box is untouched, but never discard typed text.
  const untouched = !prompt.value.trim() || exampleValues.includes(prompt.value);
  if (example && untouched) {{ prompt.value = example; }}
}});
</script>"""


def reference_form(game: str = "") -> str:
    return f"""<form action='/references' method='post' enctype='multipart/form-data'>
<h3>1. Add this game's reference images</h3>
<label>Game</label><input name='game' value='{html.escape(game)}'
  placeholder='battle-cars' required>
<label>Reference images</label><input type='file' name='files' multiple required
  accept='.png,.jpg,.jpeg,.webp'>
<label>Caption .txt files (optional)</label><input type='file' name='caption_files' multiple
  accept='.txt'>
<label>Shared description (optional)</label><input name='descriptions'
  placeholder='Blue compact combat car, cel-shaded, front three-quarter view'>
<label>Images left without a caption</label><select name='captioning'>
  <option value='gpt'>Describe with GPT</option>
  <option value='manual'>Leave blank &mdash; I will caption them myself</option>
</select>
<p class='hint'>Captions follow the Qwen Image LoRA Studio convention:
<code>drone.txt</code> captions <code>drone.png</code>, matched by filename. A shared
description fills any image without a <code>.txt</code>. You can also edit every caption by hand
afterwards. Describing with GPT needs <code>OPENAI_API_KEY</code>. References stay under
<code>data/games/&lt;game&gt;/references/</code> and are never shared with another game.</p>
<button>Add references</button></form>"""


@app.get("/", response_class=HTMLResponse)
def home(game: str = ""):
    rows = []
    for name in workflow.games():
        for job in workflow.list_jobs(name):
            rows.append(
                f"<tr><td>{html.escape(name)}</td>"
                f"<td><a href='/jobs/{name}/{job.id}'><code>{job.id}</code></a></td>"
                f"<td class='state'>{html.escape(job.state)}</td>"
                f"<td>{html.escape(job.backend)}</td>"
                f"<td>{html.escape(job.prompt)}</td>"
                f"<td>{html.escape(job.created_at)}</td></tr>"
            )
    table = (
        "<table><tr><th>Game</th><th>Job</th><th>State</th><th>Backend</th><th>Prompt</th>"
        f"<th>Created</th></tr>{''.join(rows)}</table>"
        if rows
        else "<p class='hint'>No jobs yet. Add references, then create a draft.</p>"
    )
    libraries = " ".join(
        f"<a href='/references/{name}'>{html.escape(name)}</a>" for name in workflow.games()
    )
    return page(
        "Concept Art Generator",
        "<h1>Concept Art Generator</h1><p>Add references per game &rarr; create a 1024px "
        "draft &rarr; a human approves it &rarr; export a transparent 2K final.</p>"
        f"<div class='row'>{reference_form(game)}{draft_form(game)}</div>"
        + (f"<p>Reference libraries: {libraries}</p>" if libraries else "")
        + f"<h2>Jobs</h2>{table}",
    )


@app.get("/references/{game}", response_class=HTMLResponse)
def reference_library(game: str, matched: int | None = None):
    try:
        descriptions = workflow.reference_descriptions(game)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    rows = "".join(
        f"<tr><td><img class='thumb' src='/references/{game}/image/{html.escape(name)}'"
        f" alt='{html.escape(name)}'></td>"
        f"<td><code>{html.escape(name)}</code></td>"
        f"<td><textarea name='caption:{html.escape(name)}'>{html.escape(text)}</textarea></td>"
        f"</tr>"
        for name, text in descriptions.items()
    )
    captioned = sum(1 for text in descriptions.values() if text.strip())
    total = len(descriptions)
    progress = (
        f"{captioned} of {total} references captioned."
        if total
        else "This game has no references yet."
    )
    note = f" Loaded captions for {matched} image(s)." if matched is not None else ""
    editor = (
        f"<form action='/references/{game}/captions' method='post'>"
        f"<h3>Captions</h3><table><tr><th>Image</th><th>File</th><th>Description</th></tr>"
        f"{rows}</table><button>Save captions</button></form>"
        if total
        else ""
    )
    return page(
        f"{game} references",
        f"<h1>{html.escape(game)} references</h1>"
        f"<p class='hint'>{progress}{html.escape(note)} A description is what GPT ranks when a "
        f"game has more than 16 references, so keep them specific.</p>"
        f"{editor}"
        f"<form action='/references/{game}/caption-files' method='post' "
        f"enctype='multipart/form-data'><h3>Load captions from .txt files</h3>"
        f"<label>Caption files</label><input type='file' name='caption_files' multiple required "
        f"accept='.txt'>"
        f"<p class='hint'>Matched to images by filename, so <code>drone.txt</code> captions "
        f"<code>drone.png</code> &mdash; the same convention the Qwen Image LoRA Studio uses for "
        f"training datasets. Loaded captions overwrite what is stored.</p>"
        f"<button>Load caption files</button></form>"
        f"{reference_form(game)}",
    )


@app.get("/references/{game}/image/{filename}")
def reference_image(game: str, filename: str):
    try:
        path = workflow.workspace(game).reference_paths([Path(filename).name])[0]
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(path)


@app.get("/jobs/{game}/{job_id}", response_class=HTMLResponse)
def job_page(game: str, job_id: str):
    try:
        job = workflow.get_job(game, job_id)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    actions = ""
    if job.state == JobState.DRAFT_READY:
        actions = (
            f"<div class='row'>"
            f"<form action='/jobs/{game}/{job.id}/approve' method='post'>"
            f"<h3>Approve</h3><label>Note (optional)</label><input name='feedback'>"
            f"<button>Approve this draft</button></form>"
            f"<form action='/jobs/{game}/{job.id}/reject' method='post'>"
            f"<h3>Request changes</h3><label>Feedback (required)</label>"
            f"<input name='feedback' required placeholder='Wider silhouette; less surface noise'>"
            f"<button>Reject this draft</button></form></div>"
        )
    elif job.state == JobState.APPROVED:
        actions = (
            f"<form action='/jobs/{game}/{job.id}/final' method='post'><h3>Export final</h3>"
            f"<p class='hint'>Renders 2048&times;2048 by replaying the approved parameters.</p>"
            f"<button>Export 2K final</button></form>"
        )
    images = "".join(
        f"<div><h3>{stage.title()}</h3>"
        f"<a href='/assets/{game}/{stage}s/{job.id}'>"
        f"<img class='preview' src='/assets/{game}/{stage}s/{job.id}' alt='{stage}'></a></div>"
        for stage in ("draft", "final")
        if getattr(job, f"{stage}_path")
    )
    feedback = "".join(
        f"<li><b>{html.escape(entry['decision'])}</b> {html.escape(entry.get('at', ''))} "
        f"&mdash; {html.escape(entry.get('feedback', '') or '(no note)')}</li>"
        for entry in job.feedback
    )
    references = ", ".join(html.escape(name) for name in job.reference_files) or "none recorded"
    lora = f" &middot; LoRA <code>{html.escape(job.lora_name)}</code>" if job.lora_name else ""
    return page(
        f"{game} / {job.id}",
        f"<h1>{html.escape(game)} &middot; <code>{job.id}</code></h1>"
        f"<p>State: <span class='state'>{html.escape(job.state)}</span> &middot; "
        f"backend <code>{html.escape(job.backend)}</code>{lora} &middot; "
        f"transparent: {job.transparent}</p>"
        f"<p><b>Prompt:</b> {html.escape(job.prompt)}</p>"
        f"<p><b>References:</b> {references} "
        f"(<a href='/references/{game}'>library</a>)</p>"
        f"<div class='row'>{images or '<p>No image yet.</p>'}</div>"
        f"{actions}"
        f"<h2>Decisions</h2><ul>{feedback or '<li>None yet.</li>'}</ul>"
        f"<h2>Notes</h2><pre>{html.escape(chr(10).join(job.notes))}</pre>",
    )


@app.post("/references")
async def references(
    request: Request,
    game: str = Form(),
    files: list[UploadFile] = File(),
    caption_files: list[UploadFile] | None = File(None),
    descriptions: list[str] | None = Form(None),
    captioning: str = Form("gpt"),
):
    # An untouched HTML text input still posts an empty string; treat it as "no description".
    supplied = [value for value in (descriptions or []) if value.strip()]
    if supplied and len(supplied) not in {1, len(files)}:
        fail(ValueError("Supply either no descriptions, one shared description, or one per file."))
    staging = Path("data") / ".uploads"
    sidecars: dict[str, str] = {}
    copied = []
    try:
        uploaded_captions = [
            upload for upload in (caption_files or []) if (upload.filename or "").strip()
        ]
        if uploaded_captions:
            sidecars = load_caption_sidecars(save_uploads(uploaded_captions, staging))
        for index, upload in enumerate(files):
            filename = Path(upload.filename or "reference.png").name
            target = staging / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as handle:
                shutil.copyfileobj(upload.file, handle)
            typed = supplied[index] if len(supplied) == len(files) else (supplied or [None])[0]
            description = typed or caption_for(filename, sidecars)
            try:
                copied.append(
                    str(
                        workflow.add_reference(
                            game,
                            target,
                            description,
                            describe_missing=captioning == "gpt",
                        )
                    )
                )
            except (OSError, RuntimeError, ValueError) as exc:
                fail(exc)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    if wants_html(request):
        return RedirectResponse(f"/references/{game}", status_code=303)
    return {"references": copied}


@app.post("/references/{game}/captions")
async def save_captions(request: Request, game: str):
    form = await request.form()
    saved = {}
    for key, value in form.items():
        if not key.startswith("caption:"):
            continue
        filename = key.split(":", 1)[1]
        try:
            workflow.set_reference_description(game, filename, str(value).strip())
        except (OSError, ValueError) as exc:
            fail(exc)
        saved[filename] = str(value).strip()
    if wants_html(request):
        return RedirectResponse(f"/references/{game}", status_code=303)
    return {"descriptions": saved}


@app.post("/references/{game}/caption-files")
async def load_captions(
    request: Request, game: str, caption_files: list[UploadFile] = File()
):
    staging = Path("data") / ".uploads"
    try:
        matched = workflow.apply_caption_files(game, save_uploads(caption_files, staging))
    except (OSError, ValueError) as exc:
        fail(exc)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    if wants_html(request):
        return RedirectResponse(f"/references/{game}?matched={matched}", status_code=303)
    return {"matched": matched, "descriptions": workflow.reference_descriptions(game)}


@app.post("/draft")
def draft(
    request: Request,
    game: str = Form(),
    prompt: str = Form(),
    backend: Backend = Form(),
    lora_name: str | None = Form(None),
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
        job = workflow.create_draft(
            ArtRequest(
                game,
                prompt,
                backend,
                lora_name or None,
                reference_count,
                transparent=transparent,
                negative_prompt=negative_prompt,
                seed=seed,
                steps=steps,
                guidance_scale=guidance_scale,
                lora_scale=lora_scale,
                background_model=background_model or DEFAULT_BACKGROUND_MODEL,
            )
        )
    except (OSError, RuntimeError, ValueError) as exc:
        fail(exc)
    return respond(request, job)


async def read_feedback(request: Request) -> str | None:
    """Accept feedback from the HTML form body or from the documented `?feedback=` query."""
    if "feedback" in request.query_params:
        return request.query_params["feedback"]
    if request.headers.get("content-type", "").startswith(
        ("application/x-www-form-urlencoded", "multipart/form-data")
    ):
        value = (await request.form()).get("feedback")
        if value is not None:
            return str(value)
    return None


@app.post("/jobs/{game}/{job_id}/approve")
async def approve(request: Request, game: str, job_id: str):
    try:
        job = workflow.approve(game, job_id, await read_feedback(request))
    except (OSError, RuntimeError, ValueError) as exc:
        fail(exc)
    return respond(request, job)


@app.post("/jobs/{game}/{job_id}/reject")
async def reject(request: Request, game: str, job_id: str):
    try:
        job = workflow.reject(game, job_id, await read_feedback(request) or "")
    except (OSError, RuntimeError, ValueError) as exc:
        fail(exc)
    return respond(request, job)


@app.post("/jobs/{game}/{job_id}/final")
def final(request: Request, game: str, job_id: str):
    try:
        job = workflow.create_final(game, job_id)
    except (OSError, RuntimeError, ValueError) as exc:
        fail(exc)
    return respond(request, job)


@app.get("/assets/{game}/{stage}/{job_id}")
def asset(game: str, stage: str, job_id: str):
    if stage not in {"drafts", "finals"}:
        raise HTTPException(status_code=404, detail="Unknown asset stage")
    path = workflow.workspace(game).image_path(stage, job_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Asset not found")
    return FileResponse(path, media_type="image/png")


def main() -> None:
    import uvicorn

    uvicorn.run("concept_art_generator.web:app", host="127.0.0.1", port=8000, reload=True)
