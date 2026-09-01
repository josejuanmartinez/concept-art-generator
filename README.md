# Concept Art Generator

![TinyBytes Art Pipeline — isolated references, two generation backends (HF LoRA or GPT Image 2), human approval, transparent 2K finals](docs/art-pipeline.png)

A production-minded TinyBytes assessment project: create game-specific concept art without mixing visual references between **Massive Warfare** and **Battle Cars**. It uses a human approval gate: **low-resolution draft → explicit approval → transparent final**.

## Architecture

There are **three independent ways to drive the pipeline**, and they all sit on top of one shared
core. The agent path is optional: the UI and the CLI involve no Claude at all.

```text
  Agent (Claude Code)          UI (concept-art-ui)          CLI (concept-art)
  ├─ style-director            browser forms, job           subcommands printing
  ├─ technical-artist          pages, reference             JSON on stdout
  └─ verifier                  library + JSON API
         │                             │                            │
         └──────────────┬──────────────┴────────────────────────────┘
                        ▼
                 ConceptArtWorkflow
      game isolation · LoRA catalogue · draft → approved → final
      transparency gate · provenance sidecars · usage ledger
                 │                          │
          GPT Image 2                HF Space + named LoRA
          reference edit             replayable JSON API
```

`ConceptArtWorkflow` is the only place a job is created, approved, or exported, so every rule below
holds identically no matter which of the three entry points you use.

- **Isolation:** each game lives under `data/games/<game>/`; references are selected only from that folder and hashes are recorded in each job. There is no global reference pool.
- **Human oversight:** `final` refuses all non-approved jobs.
- **Bounded LoRA choice:** exactly three trained LoRAs exist and each is bound to one game; anything else is refused rather than guessed.
- **Transparency:** both providers are asked for transparent output and a verifier rejects opaque PNGs before export.
- **Cost hygiene:** each paid stage appends a record to `data/usage.jsonl`; secrets are environment variables and never committed.
- **Providers:** GPT Image 2 uses the image-edit endpoint with up to 16 game-local references. Each reference has a stored description; GPT creates it through the Responses API when one is not supplied. If more than 16 references exist, GPT ranks their descriptions against the art request and the best 16 are attached. The HF backend calls the private Qwen Image LoRA Studio's `/v1/generate` endpoint and persists the returned replay parameters. GPT Image 2 supports image input/output through image endpoints, per [official OpenAI documentation](https://developers.openai.com/api/docs/models/gpt-image-2).

The final pass exports 2048×2048. The Hugging Face path repeats the approved 1024×1024 generation parameters and enables the Space's Swin2SR x2 upscaler; GPT Image 2 generates its high-quality 2K final directly.

## Setup

Requires Python 3.11+.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

That is enough to run the tests and start the UI. To generate images you also need provider
credentials — set them as **system environment variables**, not in a file; see
[Secrets](#secrets-the-os-environment-not-env).

## Secrets: the OS environment and `.env`

The project reads four environment variables. None of them are required to browse the code, run the
tests, or start the UI — they are only needed by the backend you actually call.

| Variable | Needed for | Notes |
|---|---|---|
| `OPENAI_API_KEY` | GPT Image 2 generation, and GPT-written reference descriptions/ranking | Provider credential. Not needed if you caption references yourself and only use Hugging Face. |
| `OPENAI_REFERENCE_MODEL` | Reference descriptions and 16-of-N selection | Not a secret. Optional; defaults to `gpt-5.6-luna`. |
| `HF_SPACE_URL` | Hugging Face generation | Not a secret. Example: `https://<owner>-qwen-image-lora-studio.hf.space` |
| `HF_TOKEN` | Private Space and private LoRA access | Token with read access. |

### Where they live

**The application reads ordinary environment variables.** `.env` is an optional local convenience,
not a requirement — nothing needs it, and nothing fails without it. Recommended practice is to keep
the two real secrets, `OPENAI_API_KEY` and `HF_TOKEN`, **in the system environment only**, and leave
them out of `.env` entirely.

`src/concept_art_generator/config.py` loads `.env` with `override=False`, so the resolution order is:

1. A variable already in the process environment — **always wins**.
2. Otherwise, a value from `.env` in the current working directory, if that file exists.
3. Otherwise, unset.

Because step 2 reads the **current working directory**, a `.env` in this repository is silently
ignored when you run `concept-art` from anywhere else. System environment variables work from any
directory, which is the second reason to prefer them.

#### Setting them at OS level

Windows, persistent for your user account (reopen the terminal afterwards):

```powershell
[Environment]::SetEnvironmentVariable('OPENAI_API_KEY', '<key>', 'User')
[Environment]::SetEnvironmentVariable('HF_TOKEN', '<token>', 'User')
```

Windows, current session only:

```powershell
$env:OPENAI_API_KEY = '<key>'
$env:HF_TOKEN = '<token>'
```

macOS / Linux, current session (add to `~/.zshrc` or `~/.bashrc` to persist):

```bash
export OPENAI_API_KEY='<key>'
export HF_TOKEN='<token>'
```

Confirm they are visible without revealing them:

```powershell
[bool]$env:OPENAI_API_KEY, [bool]$env:HF_TOKEN
```

#### Why this is the safer default

A `.env` is a plaintext file inside the project folder, which means it is exposed to a much wider
set of readers than a process environment variable:

- Any agent, editor extension, indexer, linter, or script with filesystem access to the repository
  can read it — including tools you did not think of as "having your keys".
- It can be committed by accident (a `git add -f`, a misconfigured `.gitignore`, a copy of the repo
  into a directory whose ignore rules differ), and it travels with the folder when the project is
  zipped, backed up, or shared.
- It shows up in screen shares, file-tree screenshots, and diffs.

Environment variables are not a vault — any process running as your user can still read them — so
the gain is specific rather than absolute: the secret is not sitting on disk in the project, and it
cannot leak by copying or committing the folder. That is exactly the risk the comment at the top of
`.env.example` is warning about.

If you do use `.env`, keep it to the **non-secret** settings, and let the OS supply the credentials:

```bash
# .env — safe to keep here
OPENAI_REFERENCE_MODEL=gpt-5.6-luna
HF_SPACE_URL=https://<owner>-qwen-image-lora-studio.hf.space
# OPENAI_API_KEY and HF_TOKEN: set these in the system environment instead.
```

`.env` is gitignored; `.env.example` is committed and contains no values.

### How they are handled

Keys are used as provider authentication only — `Authorization: Bearer <HF_TOKEN>` for the Space,
and the OpenAI SDK's own credential handling for `OPENAI_API_KEY`. They are never added to prompts,
job JSON, `.png.prompt` sidecars, `data/usage.jsonl`, output metadata, UI responses, or logs.

### Rules

- Never place API keys in prompts, source files, job JSON, logs, screenshots, or commits.
- Never `cat .env` or echo a key in an agent session; the orchestrator instructions in `CLAUDE.md`
  forbid it explicitly.
- Treat any agent or program with shell access as trusted code: use dedicated least-privilege
  provider keys, set a spend cap on the OpenAI key, and scope the HF token to only the private
  Space and LoRA repositories it needs.
- Rotate a key immediately if it is exposed.
- Keep invoices in the provider accounts; `data/usage.jsonl` is the local usage log.

## The three LoRAs

Exactly three trained LoRAs exist. They are listed in `src/concept_art_generator/loras.py`, which is
the single place to edit when one is trained, added, or retired. A slug is **never** inferred from a
game slug, job name, folder, or an example in this file — anything outside the table is refused.

| LoRA | Trigger | Game | Subject | Style keywords |
|---|---|---|---|---|
| `jjmcarrascosa/drone-bc` | `drone-bc` | `battle-cars` | A drone | stylized 3D rendered game asset, glossy cel-shaded surfaces |
| `jjmcarrascosa/pilot-bc` | `pilot-bc` | `battle-cars` | A pilot | flat vector game art |
| `jjmcarrascosa/pilot-mw` | `pilot-mw` | `massive-warfare` | A pilot | semi-realistic painted digital game art, soft rim lighting |

Each LoRA may only be used for the game it was trained on; using `drone-bc` for `massive-warfare` is
refused, because that is exactly the cross-game style leak this project exists to prevent. GPT Image
2 takes no LoRA at all — it uses the game's own reference images.

Run `concept-art loras` for the catalogue as JSON, including each example prompt in full. See
[Prompt building](#prompt-building) for the shape a LoRA prompt must take.

## Prompt building

The two backends need genuinely different prompts, so `src/concept_art_generator/prompts.py` has a
single `build_prompt(request, descriptions, backend)` that dispatches on the backend. It is the only
place a generation prompt is constructed.

### Hugging Face + LoRA — the trained caption shape

Each LoRA was trained on captions of one exact shape, so the prompt is restricted to that shape and
nothing else is added:

```text
<trigger> <subject keyword> <details>, <that LoRA's style keywords>, plain white background
```

The trigger is the LoRA name without the owner prefix, and it is prepended for you — do not type it.
That LoRA's style keywords are appended when you leave them out. No scene description, no framing
boilerplate, and deliberately **no "transparent background" instruction**: the trained captions say
`plain white background`, and the Space strips it server-side via `remove_background`, so asking the
LoRA for something its captions never said would only push it off-distribution.

```text
you type:  A drone with a squat black hull and cyan thrusters
sent:      drone-bc A drone with a squat black hull and cyan thrusters,
           stylized 3D rendered game asset, glossy cel-shaded surfaces, plain white background
```

Write the subject the same way that LoRA's own example does. **These examples apply to the Hugging
Face LoRA backend only** — they are the captions the adapters were trained on, and they mean nothing
to GPT Image 2.

`jjmcarrascosa/drone-bc`

> A drone with a smooth white and grey egg-shaped shell, lime green thruster rings on each side, a
> glowing orange vent slot on its face, two upright yellow-tipped blade fins above and one below,
> stylized 3D rendered game asset, glossy cel-shaded surfaces, plain white background

`jjmcarrascosa/pilot-bc`

> A pilot in a horned imp mask and hood, wearing a green graffiti-covered hooded jacket and dark
> cargo trousers, holding a curved black blade, in colourful high-top sneakers, full body, flat
> vector game art, plain white background

`jjmcarrascosa/pilot-mw`

> A pilot undead soldier with a blazing skull head wreathed in orange fire, heavy charcoal combat
> armour lined with glowing orange lights, a rifle held at his hip, semi-realistic painted digital
> game art, soft rim lighting, plain white background

### GPT Image 2 — our own style-director prompt from the references

There is no LoRA and no trigger word here, so **none of the caption shape above applies**. GPT Image
2 gets our own style-director prompt instead: the game's reference images are attached to the edit
request, and the style descriptions extracted from those references are restated in the prompt as
notes, so the requirement is explicit rather than implied. Write a plain subject line and let the
references carry the style.

```text
Concept art for battle-cars. Subject: An armored street racer. Match only the visual language
of the attached reference images for this game: silhouette discipline, materials, palette,
lighting and rendering style. Style extracted from those references:
- White egg-shaped drone shell with lime thruster rings
- Orange armoured hull panel, glossy cel-shaded
Single isolated subject, centered, no scenery, no text, no watermark, transparent background.
```

### Fallback and replay

When a Hugging Face request falls back to GPT Image 2, the prompt is **rebuilt** in the GPT shape
above — the trigger word means nothing to it, and it needs the reference-derived style instead. An HF
final replays the approved draft's prompt byte for byte.

## References and captions

Every reference lives under `data/games/<game>/references/`, and each one carries a text description
stored in `descriptions.json`. Descriptions are not decoration: when a game has more than 16
references, GPT ranks these descriptions against the art request and picks the 16 to attach, so
specific captions produce better drafts.

Captions follow the **Qwen Image LoRA Studio dataset convention** — `drone.txt` captions
`drone.png`, matched by filename — used here to set reference descriptions rather than training
captions. There are four ways to caption a reference, in priority order:

1. **Typed text** — the description argument on the CLI, or the shared-description field in the UI.
2. **A `.txt` file** — a sibling next to the image on the CLI, or uploaded alongside the images in
   the UI, matched by filename stem.
3. **By hand afterwards** — the UI's reference library at `/references/<game>` shows every image
   with an editable description box.
4. **GPT** — when nothing above supplies one, the configured GPT model writes it
   (needs `OPENAI_API_KEY`). In the UI this is the *Describe with GPT* option; choose *Leave blank*
   to caption by hand instead and avoid the API call entirely.

## Running a generation

The pipeline is the same in all three places, and both backends are available in all three:

| | Agent | UI | Command line |
|---|---|---|---|
| Add references | `concept-art add-reference` | reference form on `/` | `concept-art add-reference` |
| Caption references | `.txt` sibling or argument | caption editor + `.txt` upload | `.txt` sibling or argument |
| GPT Image 2 draft | `--backend gpt-image-2` | backend dropdown | `--backend gpt-image-2` |
| Hugging Face draft | `--backend huggingface --lora-name` | backend + LoRA dropdowns | `--backend huggingface --lora-name` |
| Approve / reject | `concept-art approve` / `reject` | buttons on the job page | `concept-art approve` / `reject` |
| 2K final | `concept-art final` | **Export 2K final** button | `concept-art final` |

### 1. From an agent

`CLAUDE.md` and `.claude/agents/` make Claude Code the orchestrator. Open the repository in Claude
Code and ask in natural language:

```text
Generate concept art for battle-cars using jjmcarrascosa/drone-bc:
a drone with a squat black and orange armoured hull, twin stubby side thrusters glowing cyan.
```

```text
Generate concept art for massive-warfare with GPT Image 2:
an undead artillery gunner in heavy charcoal armour.
```

The orchestrator then:

1. Confirms the single game slug in scope. It never reads, attaches, summarizes, or transmits
   another game's references.
2. For Hugging Face, asks which of the three LoRAs to use if you have not said. It never invents a
   fourth.
3. Delegates planning to `style-director` (read-only, game-scoped).
4. Has `technical-artist` produce one 1024×1024 draft, then stops.
5. Waits for your explicit approval — `concept-art approve <game> <job-id>` — or a rejection with
   feedback.
6. Delegates the final export, then invokes `verifier` last.
7. Routes any failure back to `technical-artist`, never to `verifier`.

Agents drive the same CLI documented below; every command prints JSON, so `concept-art jobs <game>`
and `concept-art show <game> <job-id>` are the read-back points. See `CLAUDE.md` for the full rule
set the orchestrator follows.

### 2. From the UI

```bash
concept-art-ui
# http://127.0.0.1:8000
```

The dashboard at `/` has the reference form, the draft form, and a table of every game's jobs.

- **Adding references** takes images, optional `.txt` caption files, an optional shared description,
  and a choice of what to do with anything left uncaptioned (describe with GPT, or leave blank).
- **The reference library** at `/references/<game>` lists every reference with a thumbnail and an
  editable description, plus a `.txt` upload that fills them by filename.
- **The draft form** has a **Backend** dropdown (GPT Image 2 or Hugging Face Space) and a **LoRA**
  dropdown containing only the three catalogued LoRAs. Choosing one fills the prompt box with that
  LoRA's example prompt, so you can edit rather than start blank; anything you type yourself is
  never overwritten. The HF-only fields (negative prompt, seed, steps, guidance `4.0`, LoRA scale
  `1.25`, background model `birefnet-general`) are ignored by GPT Image 2. **Transparency** is
  `Transparent PNG` by default; choose `Opaque` only for artwork that genuinely requires a
  background.
- **The job page** at `/jobs/<game>/<job-id>` shows the draft over a checkerboard so transparency is
  visible, with **Approve**, **Request changes**, and — once approved — **Export 2K final**. The
  final appears on the same page.

The same routes stay available as a JSON API, so scripts and agents can use the server too. They
return JSON to a non-browser client and redirect a browser to the relevant page:

```bash
# The draft response includes the job ID.
curl -X POST http://127.0.0.1:8000/jobs/battle-cars/<job-id>/approve
curl -X POST http://127.0.0.1:8000/jobs/battle-cars/<job-id>/final
curl -X POST "http://127.0.0.1:8000/jobs/battle-cars/<job-id>/reject?feedback=Wider%20silhouette"
```

Generated files are served at `/assets/<game>/drafts/<job-id>` and `/assets/<game>/finals/<job-id>`;
reference images at `/references/<game>/image/<filename>`.

### 3. From the command line

```bash
# A description argument, or a sibling drone.txt, or GPT writes one.
concept-art add-reference battle-cars path\to\drone.png "Egg-shaped white drone shell, lime thruster rings"
concept-art add-reference battle-cars path\to\pilot.png

# Hugging Face: one of the three LoRAs, matched to its own game.
concept-art draft battle-cars "A drone with a smooth white and grey egg-shaped shell, lime green thruster rings on each side, a glowing orange vent slot on its face, two upright yellow-tipped blade fins above and one below, stylized 3D rendered game asset, glossy cel-shaded surfaces, plain white background" --backend huggingface --lora-name jjmcarrascosa/drone-bc

concept-art draft battle-cars "A pilot in a horned imp mask and hood, wearing a green graffiti-covered hooded jacket and dark cargo trousers, holding a curved black blade, in colourful high-top sneakers, full body, flat vector game art, plain white background" --backend huggingface --lora-name jjmcarrascosa/pilot-bc --seed 42 --steps 28 --guidance-scale 4.0 --lora-scale 1.25

concept-art draft massive-warfare "A pilot undead soldier with a blazing skull head wreathed in orange fire, heavy charcoal combat armour lined with glowing orange lights, a rifle held at his hip, semi-realistic painted digital game art, soft rim lighting, plain white background" --backend huggingface --lora-name jjmcarrascosa/pilot-mw

# GPT Image 2 needs no LoRA; it uses this game's references.
concept-art draft battle-cars "A drone with a smooth white and grey egg-shaped shell, lime green thruster rings on each side, stylized 3D rendered game asset, glossy cel-shaded surfaces, plain white background" --backend gpt-image-2

concept-art approve battle-cars <job-id>
concept-art final battle-cars <job-id>

# Keep artist feedback with a rejected draft; no final can be made from it.
concept-art reject battle-cars <job-id> --feedback "Wider silhouette; reduce surface noise"

# Read-back
concept-art loras
concept-art games
concept-art jobs battle-cars --state draft_ready
concept-art show battle-cars <job-id>
```

Every command prints the job (or list) as JSON on stdout. `concept-art --help` reprints the runnable
example above for each LoRA. `--data-dir` relocates the workspace root, which defaults to `data`.
`--references N` caps how many references a draft may use (1–16), and `--opaque` disables the
transparent-PNG default.

Drafts use all available references up to 16. When there are more, GPT receives only their stored text descriptions and chooses the 16 best matches; the exact filenames, descriptions, and hashes are persisted so the final reuses the same set.

## Choosing a generation backend

| Backend | Use it when | Pros | Trade-offs |
|---|---|---|---|
| **GPT Image 2** | You have game references and need strong visual adherence without training first. | Sends up to 16 selected, game-local reference images to the edit endpoint; GPT description matching narrows larger pools; fast to start; requests transparent output. | Paid per generation; references leave the local machine for the provider; needs `OPENAI_API_KEY`; style consistency depends on reference selection and prompt quality. |
| **Hugging Face Space + named LoRA** | The game's visual style already has one of the three trained private Qwen Image LoRAs. | Does not transmit the game reference images for inference; reusable style adapter; the LoRA-to-game binding makes style selection explicit and auditable; supports Swin2SR 2K finals and server-side background removal. | Requires an available GPU Space and private-token access; first inference or upscaling can be slow; segmentation can trim delicate details, so quality must be reviewed. |

Both routes create a 1024×1024 draft first and require explicit approval before a final. GPT Image 2 uses `quality="low"`. HF/LoRA uses the configured prompt, negative prompt, LoRA, steps, guidance, LoRA scale, seed, scheduler, and base model; those exact values are persisted for the approved final. Both routes reject opaque PNGs rather than silently exporting non-transparent assets.

### Reliability and 2K final policy

When Hugging Face is selected, it is always tried first. A timeout, forbidden response, malformed response, or other provider error automatically retries through GPT Image 2 using only the selected, game-local references; the job audit notes record the fallback. If GPT Image 2 is unavailable or the game has no references, the job fails safely instead of using another game's style.

Final exports are **2048×2048** and preserve their PNG alpha channel. For an HF final, the Space repeats the same deterministic 1024×1024 generation and applies tiled Swin2SR x2. The only stage-specific HF inference switch is `upscale_to_2k`: `false` for the draft and `true` after approval. `remove_background` remains identical between stages and follows the optional Transparent PNG selection. GPT Image 2 uses the approved draft as its primary edit input plus up to 15 selected style references, then generates the high-quality 2K final.

The equivalent Space request options are:

```json
{
  "width": 1024,
  "height": 1024,
  "guidance_scale": 4.0,
  "lora_scale": 1.25,
  "remove_background": true,
  "background_model": "birefnet-general",
  "upscale_to_2k": true
}
```

`remove_background` controls PNG transparency independently from 2K upscaling, so either option can be enabled without the other. `background_model` names the Space's segmentation model and is always sent explicitly, so a change of server-side default cannot silently alter a game's cutouts; it defaults to `birefnet-general` and the Space rejects an unknown name with a 400. Guidance defaults to `4.0` and LoRA scale to `1.25`, matching the studio's own defaults.

### Hugging Face replay API protocol

The client sends every inference control to `POST <HF_SPACE_URL>/v1/generate`, including the bearer `HF_TOKEN`. The response contains the PNG and a canonical `generation_parameters` record with the actual seed and scheduler. That record is stored in the job and replayed for the final; generation fails closed if the active scheduler, LoRA scale, guidance, or background removal model no longer matches. The final request changes only `upscale_to_2k` from `false` to `true`.

## Provenance sidecars and feedback

Every generated PNG receives a sibling sidecar using the studio-friendly convention `image.png.prompt`. It contains the effective model/backend, complete generation prompt, LoRA name where applicable, game-local reference hashes, request ID, dimensions, transparency selection, job state, and every approval/rejection comment.

The canonical job record remains in `data/games/<game>/jobs/<job-id>.json`; sidecars are refreshed when a draft is approved or rejected, so moving a generated asset does not lose its prompt or artist feedback.

## Project files

### Repository

| Path | What it is |
|---|---|
| `README.md` | This document. |
| `CLAUDE.md` | Orchestrator instructions Claude Code follows: isolation rules, the three LoRAs, the approval loop, allowed commands, and the secret-handling ban. Used only by the agent entry point. |
| `pyproject.toml` | Package metadata, dependencies, the `concept-art` / `concept-art-ui` entry points, and the pytest and ruff configuration. |
| `.env.example` | Committed template for the four environment variables, with the two credentials commented out. Contains no values. |
| `.gitignore` | Excludes `.env`, `data/`, caches, and build output. |
| `docs/art-pipeline.png` | The pipeline diagram at the top of this README. |

### Source — `src/concept_art_generator/`

| Module | Responsibility |
|---|---|
| `__init__.py` | Loads an optional `.env` on import, then exposes `ConceptArtWorkflow`. |
| `config.py` | `load_settings()`: optionally reads a `.env` from the working directory with `override=False`, so system environment variables always win. |
| `models.py` | Dataclasses and enums: `ArtRequest`, `ArtJob`, `Backend`, `JobState`, `DEFAULT_BACKGROUND_MODEL`, plus job JSON load/save. |
| `loras.py` | The closed catalogue of the three trained LoRAs — trigger word, game, subject keyword, style keywords, example prompt — and `resolve_lora()`, which refuses an unknown slug or a cross-game pairing. |
| `prompts.py` | `build_prompt()` — the one place a prompt is built, dispatching on backend: the trained LoRA caption shape for Hugging Face, reference-extracted style for GPT Image 2. |
| `workspace.py` | `GameWorkspace` — the isolation boundary. Validates the game slug, owns every path under `data/games/<game>/`, stores reference descriptions, and computes reference hashes. |
| `references.py` | `OpenAIReferenceAgent` (GPT-written descriptions and text-only 16-of-N ranking) and the `.txt` caption helpers that implement the studio's `drone.txt → drone.png` convention. |
| `agents.py` | `QualityGate` — rejects an opaque PNG before export. |
| `providers.py` | `RenderSpec` / `RenderedImage`, `HuggingFaceSpaceProvider` (`POST /v1/generate`), `GPTImage2Provider` (images-edit endpoint), and `DeterministicProvider` for offline tests. |
| `workflow.py` | `ConceptArtWorkflow` — the shared core behind all three entry points: reference and caption handling, LoRA validation, draft, approve/reject, final with parameter replay, HF→GPT fallback, sidecars, and the usage ledger. |
| `cli.py` | The `concept-art` command: `add-reference`, `draft`, `approve`, `reject`, `final`, `show`, `games`, `loras`, `jobs`. |
| `web.py` | The `concept-art-ui` FastAPI app: dashboard, reference library and caption editor, job pages with approve/reject/final controls, asset serving, and the JSON API for scripts and agents. |

### Agent definitions — `.claude/agents/`

| File | Role |
|---|---|
| `style-director.md` | Read-only art-direction planner, scoped to exactly one game. |
| `technical-artist.md` | Runs the CLI workflow for one game; carries the three LoRAs and the prompt-style exemplars, and stops after the draft for human approval. |
| `verifier.md` | Read-only final check: job state, reference-hash isolation, alpha channel, usage ledger. |

### Tests — `tests/`

| File | Covers |
|---|---|
| `test_workflow.py` | Approval gate, rejection feedback, game isolation, opaque rejection, 16-of-N reference selection, HF parameter replay, GPT final input composition, HF→GPT fallback. |
| `test_providers.py` | HF request mapping and replay-parameter validation; GPT Image 2 reference limit and supported draft size. |
| `test_loras.py` | The catalogue is exactly three; an invented slug and a cross-game pairing are both refused, at the helper and through `create_draft`. |
| `test_prompts.py` | Both prompt shapes: trigger prepended once, style keywords appended when missing, no wrapper text on the LoRA prompt, reference style notes in the GPT prompt, identical draft/final HF prompt, and a rebuilt prompt on the GPT fallback. |
| `test_captions.py` | Sibling `.txt` captions, caption priority, stem matching, the UI upload/caption-editor routes, and reference-image path traversal. |
| `test_web.py` | The browser flow end to end for both backends, the reference-upload form, the transparency control, and the JSON API used by curl and agents. |

### Generated at runtime — `data/` (gitignored)

```text
data/
├─ usage.jsonl                       # append-only local cost ledger
└─ games/<game>/
   ├─ references/                    # this game's reference images only
   │  └─ descriptions.json           # filename → stored description
   ├─ jobs/<job-id>.json             # canonical job record
   ├─ drafts/<job-id>.png            # 1024×1024 draft (+ .png.prompt sidecar)
   └─ finals/<job-id>.png            # 2048×2048 final (+ .png.prompt sidecar)
```

## Test

```bash
pytest -q
ruff check src tests
```