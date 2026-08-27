# Concept Art Generator

A production-minded TinyBytes assessment project: create game-specific concept art without mixing visual references between **Massive Warfare** and **Battle Cars**. It uses a human approval gate: **low-resolution draft → explicit approval → transparent final**.

## Architecture

```text
Claude orchestrator
  ├─ Style director subagent (one game's references only)
  ├─ Technical artist subagent (runs CLI/UI; creates draft/final)
  └─ Verifier subagent (checks isolation, approval state, alpha)
                         │
                         ▼
              ConceptArtWorkflow
             draft → approved → final
                 │          │
       GPT Image 2      HF Space + named LoRA
       reference edit   replayable JSON API
```

- **Isolation:** each game lives under `data/games/<game>/`; references are selected only from that folder and hashes are recorded in each job. There is no global reference pool.
- **Human oversight:** `final` refuses all non-approved jobs.
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
copy .env.example .env
```

The application loads `.env` automatically; values already present in the shell take precedence.

| Variable | Needed for | Notes |
|---|---|---|
| `OPENAI_API_KEY` | GPT Image 2 | Provider credential |
| `OPENAI_REFERENCE_MODEL` | Reference descriptions/selection | Optional; defaults to `gpt-5.6-luna` |
| `HF_SPACE_URL` | Hugging Face | Example: `https://<owner>-qwen-image-lora-studio.hf.space` |
| `HF_TOKEN` | Private Space/LoRA | Token with read access |

Keys are loaded from `.env` or the shell; shell values take precedence. They are used as provider authentication headers only and are not added to prompts, job JSON, usage logs, output metadata, or UI responses. `.env` is gitignored. Still treat any agent or program with shell/environment access as trusted code: use dedicated least-privilege provider keys, set a spend cap for OpenAI, scope the HF token to the private Space/LoRA access required, and rotate a key if it is exposed.

# Instructions for agents

This game can be ran inside an orchestrator agent like Claude, that will use the project subagents and report only verified outcomes.

If you are an agent:

1. Ask which single game slug is in scope. Never read, attach, summarize, or transmit another game's references.
2. If Hugging Face is selected, ask the user for the exact LoRA model slug in Hugging Face before making a request. Prefer the fully qualified `owner/repo` slug. Never infer it from the game slug, job name, local folder, or examples in this README.
3. Delegate planning to `style-director`; it is read-only and game-scoped.
4. Have `technical-artist` make one 1024×1024 draft. Do not produce a final in the same step.
5. Wait for explicit human approval via `concept-art approve <game> <job-id>`.
6. Delegate final export then invoke `verifier` last.
7. Route failures back to `technical-artist`, never to `verifier`.

## Backends

- **Hugging Face Space:** requires the exact user-confirmed Hugging Face LoRA model slug and `HF_SPACE_URL` / `HF_TOKEN`. Ask for the slug when it has not been supplied; do not guess it. Do not send reference images to the Space.
- **GPT Image 2:** sends only the selected references inside `data/games/<game>/references/` to the edit API.

Never place API keys in prompts, source files, job JSON, logs, screenshots, or commits. Keep invoices in provider accounts; use `data/usage.jsonl` as the local usage log.


## Choosing a generation backend

| Backend | Use it when | Pros | Trade-offs |
|---|---|---|---|
| **GPT Image 2** | You have game references and need strong visual adherence without training first. | Sends up to 16 selected, game-local reference images to the edit endpoint; GPT description matching narrows larger pools; fast to start; requests transparent output. | Paid per generation; references leave the local machine for the provider; needs `OPENAI_API_KEY`; style consistency depends on reference selection and prompt quality. |
| **Hugging Face Space + named LoRA** | A game's visual style already has a trained private Qwen Image LoRA. | Does not transmit the game reference images for inference; reusable style adapter; private LoRA naming makes game-specific style selection explicit; supports Swin2SR 2K finals and server-side background removal. | Requires an available GPU Space and private-token access; first inference or upscaling can be slow; segmentation can trim delicate details, so quality must be reviewed. |

Both routes create a 1024×1024 draft first and require explicit approval before a final. GPT Image 2 uses `quality="low"`. HF/LoRA uses the configured prompt, negative prompt, LoRA, steps, guidance, LoRA scale, seed, scheduler, and base model; those exact values are persisted for the approved final. Both routes reject opaque PNGs rather than silently exporting non-transparent assets.

### Reliability and 2K final policy

When Hugging Face is selected, it is always tried first. A timeout, forbidden response, malformed response, or other provider error automatically retries through GPT Image 2 using only the selected, game-local references; the job audit notes record the fallback. If GPT Image 2 is unavailable or the game has no references, the job fails safely instead of using another game's style.

Final exports are **2048×2048** and preserve their PNG alpha channel. For an HF final, the Space repeats the same deterministic 1024×1024 generation and applies tiled Swin2SR x2. The only stage-specific HF inference switch is `upscale_to_2k`: `false` for the draft and `true` after approval. `remove_background` remains identical between stages and follows the optional Transparent PNG selection. GPT Image 2 uses the approved draft as its primary edit input plus up to 15 selected style references, then generates the high-quality 2K final.

The equivalent Space request options are:

```json
{
  "width": 1024,
  "height": 1024,
  "remove_background": true,
  "upscale_to_2k": true
}
```

`remove_background` controls PNG transparency independently from 2K upscaling, so either option can be enabled without the other.

### Hugging Face replay API protocol

The client sends every inference control to `POST <HF_SPACE_URL>/v1/generate`, including the bearer `HF_TOKEN`. The response contains the PNG and a canonical `generation_parameters` record with the actual seed and scheduler. That record is stored in the job and replayed for the final; generation fails closed if the active scheduler no longer matches. The final request changes only `upscale_to_2k` from `false` to `true`.

## CLI

```bash
concept-art add-reference massive-warfare path\to\mw_reference.png
concept-art add-reference battle-cars path\to\bc_reference.png "Blue compact combat car, cel-shaded, front three-quarter view"

concept-art draft massive-warfare "heavy tracked artillery drone" --backend gpt-image-2
concept-art draft massive-warfare "heavy tracked artillery drone" --backend huggingface --lora-name <exact-owner>/<exact-lora-repo> --seed 42 --steps 28 --guidance-scale 4.0 --lora-scale 0.8

concept-art approve massive-warfare <job-id>
concept-art final massive-warfare <job-id>

# Keep artist feedback with a rejected draft; no final can be made from it.
concept-art reject massive-warfare <job-id> --feedback "Wider silhouette; reduce surface noise"
```

The optional last `add-reference` argument is the image description. If it is omitted, the configured GPT model describes the image using `OPENAI_API_KEY`. Drafts use all available references up to 16. When there are more, GPT receives only their stored text descriptions and chooses the 16 best matches; the exact filenames, descriptions, and hashes are persisted so the final reuses the same set.

## Provenance sidecars and feedback

Every generated PNG receives a sibling sidecar using the studio-friendly convention `image.png.prompt`. It contains the effective model/backend, complete generation prompt, LoRA name where applicable, game-local reference hashes, request ID, dimensions, transparency selection, job state, and every approval/rejection comment.

The canonical job record remains in `data/games/<game>/jobs/<job-id>.json`; sidecars are refreshed when a draft is approved or rejected, so moving a generated asset does not lose its prompt or artist feedback.

## Minimal UI

```bash
concept-art-ui
# http://127.0.0.1:8000
```

The UI has reference upload and draft-generation forms. It deliberately exposes approval and final export as small, auditable HTTP actions:

```bash
# The draft response includes the job ID.
curl -X POST http://127.0.0.1:8000/jobs/massive-warfare/<job-id>/approve
curl -X POST http://127.0.0.1:8000/jobs/massive-warfare/<job-id>/final
curl -X POST "http://127.0.0.1:8000/jobs/massive-warfare/<job-id>/reject?feedback=Wider%20silhouette"
```

Generated files are available at `/assets/<game>/drafts/<job-id>` and `/assets/<game>/finals/<job-id>`.

The UI's **Transparent PNG** control is on by default. The CLI has the equivalent default; pass `--opaque` only for artwork that genuinely requires a background.

## Claude loop and verification

`CLAUDE.md` and `.claude/agents/` make Claude Code the orchestrator:

1. `style-director` plans from one game's references only.
2. `technical-artist` produces one 1024×1024 draft and stops.
3. A human approves/rejects it.
4. `technical-artist` produces the final only after approval.
5. `verifier` runs last, read-only, to check job state, reference hashes and alpha.

## Test and submission checklist

```bash
pytest -q
ruff check src tests
```

For the demo video, show each game's isolated reference folder, a draft, the blocked-final error before approval, approval, final export, and `data/usage.jsonl`. Record actual invoices with that local log before submission.
