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
       reference edit   /v1/generate
```

- **Isolation:** each game lives under `data/games/<game>/`; references are selected only from that folder and hashes are recorded in each job. There is no global reference pool.
- **Human oversight:** `final` refuses all non-approved jobs.
- **Transparency:** both providers are asked for transparent output and a verifier rejects opaque PNGs before export.
- **Cost hygiene:** each paid stage appends a record to `data/usage.jsonl`; secrets are environment variables and never committed.
- **Providers:** GPT Image 2 uses the image-edit endpoint with up to `X` game-local references. The HF backend calls the private Qwen Image LoRA Space with a LoRA name. GPT Image 2 supports image input/output through image endpoints, per [official OpenAI documentation](https://developers.openai.com/api/docs/models/gpt-image-2).

The final pass requests high quality, then uses alpha-safe Lanczos scaling to a 2K maximum edge without another billed provider call.

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
| `HF_SPACE_URL` | Hugging Face | Example: `https://<owner>-qwen-image-lora-studio.hf.space` |
| `HF_API_TOKEN` or `HF_TOKEN` | Private Space/LoRA | Token with read access |

Keys are loaded from `.env` or the shell; shell values take precedence. They are used as provider authentication headers only and are not added to prompts, job JSON, usage logs, output metadata, or UI responses. `.env` is gitignored. Still treat any agent or program with shell/environment access as trusted code: use dedicated least-privilege provider keys, set a spend cap for OpenAI, scope the HF token to the private Space/LoRA access required, and rotate a key if it is exposed.

## Choosing a generation backend

| Backend | Use it when | Pros | Trade-offs |
|---|---|---|---|
| **GPT Image 2** | You have a small, selected set of game references and need strong visual adherence without training first. | Sends `X` selected, game-local reference images to the edit endpoint; fast to start; no LoRA training or GPU operations to manage; requests transparent output. | Paid per generation; references leave the local machine for the provider; needs `OPENAI_API_KEY`; style consistency depends on reference selection and prompt quality. |
| **Hugging Face Space + named LoRA** | A game's visual style already has a trained private Qwen Image LoRA. | Does not transmit the game reference images for inference; reusable style adapter; private LoRA naming makes game-specific style selection explicit; supports native 2K final requests and server-side background removal. | Requires an available GPU Space and private-token access; first inference can be slow; segmentation can trim delicate details, so quality must be reviewed. |

Both routes create a 512px draft first, require explicit approval before a final, and reject opaque PNGs rather than silently exporting non-transparent assets.

### Reliability and 2K final policy

When Hugging Face is selected, it is always tried first. A timeout, forbidden response, malformed response, or other provider error automatically retries through GPT Image 2 using only the selected, game-local references; the job audit notes record the fallback. If GPT Image 2 is unavailable or the game has no references, the job fails safely instead of using another game's style.

Final exports always have a **2048px long edge** and preserve their PNG alpha channel. The HF path requests a native 2K render. GPT Image 2 currently supplies its highest supported high-quality landscape source (1536×1024); the application performs one alpha-preserving local upscale to a 2K canvas and records the output dimensions in the job. It never promotes the 512px draft to a final.

## CLI

```bash
concept-art add-reference massive-warfare path\to\mw_reference.png
concept-art add-reference battle-cars path\to\bc_reference.png

concept-art draft massive-warfare "heavy tracked artillery drone" --backend gpt-image-2 --references 4
concept-art draft massive-warfare "heavy tracked artillery drone" --backend huggingface --lora-name massive-warfare-v1

concept-art approve massive-warfare <job-id>
concept-art final massive-warfare <job-id>

# Keep artist feedback with a rejected draft; no final can be made from it.
concept-art reject massive-warfare <job-id> --feedback "Wider silhouette; reduce surface noise"
```

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
2. `technical-artist` produces one 512px draft and stops.
3. A human approves/rejects it.
4. `technical-artist` produces the final only after approval.
5. `verifier` runs last, read-only, to check job state, reference hashes and alpha.

## Test and submission checklist

```bash
pytest -q
ruff check src tests
```

For the demo video, show each game's isolated reference folder, a draft, the blocked-final error before approval, approval, final export, and `data/usage.jsonl`. Record actual invoices with that local log before submission.
