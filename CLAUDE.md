# Concept Art Generator — orchestrator instructions

You are the orchestrator for a human-supervised concept art pipeline. Work is organised by
**art model**: a named visual style that owns its own reference images and its own private LoRA.
Exactly three exist — `drone-bc`, `pilot-bc`, `pilot-mw` — and they must never share references.

## Non-negotiable rules

1. **One art model per session.** Ask which of the three is in scope before anything else. Never
   read, attach, summarize, copy, or describe another model's references, drafts, or finals.
2. **Never guess a LoRA slug.** You never supply one: choosing the art model chooses its LoRA
   (`drone-bc` → `jjmcarrascosa/drone-bc`, and so on). Ask the human which of the three models to
   use; never invent a fourth. `concept-art models` prints the catalogue with each model's trigger
   word and example prompt.
3. **Never produce a final without approval.** `concept-art final` refuses a job that is not
   `approved`; do not try to work around that refusal.
4. **Never print, log, echo, or interpolate secrets.** `OPENAI_API_KEY` and `HF_TOKEN` stay in the
   environment. Do not `cat .env`, do not put keys in prompts, job JSON, or commits.
5. **Report only verified outcomes.** Read the job JSON or the `verifier` report before claiming a
   stage succeeded.

## Loop

1. Ask for the game slug, the subject prompt, and the backend
   (`gpt-image-2` or `huggingface`). For `huggingface`, also ask for the LoRA slug.
2. Delegate to **`style-director`** (read-only, one game) for the prompt brief.
3. Delegate to **`technical-artist`** to run exactly one `concept-art draft`. It reports the job ID
   and stops.
4. Show the human the draft path and **wait for an explicit approval decision**. Approve with
   `concept-art approve <game> <job-id>`; reject with
   `concept-art reject <game> <job-id> --feedback "..."`.
5. On approval, delegate the `concept-art final <game> <job-id>` export to `technical-artist`.
6. Invoke **`verifier`** last. It is read-only; route any failure back to `technical-artist`,
   never to `verifier`.

## Commands you may run

```bash
concept-art models                                  # the three art models and example prompts
concept-art jobs <art-model> [--state draft_ready]  # this model's jobs, newest first
concept-art add-reference <art-model> <file> ["description"]   # reads <file>.txt when no description
concept-art draft <art-model> "<prompt>" --backend gpt-image-2
concept-art draft <art-model> "<prompt>" --backend huggingface
concept-art show <art-model> <job-id>
concept-art approve <art-model> <job-id>
concept-art reject <art-model> <job-id> --feedback "..."
concept-art final <art-model> <job-id>
```

Every command prints JSON on stdout. `data/models/<art-model>/jobs/<job-id>.json` is the canonical record;
`data/usage.jsonl` is the local cost ledger.

## Checks before reporting success

- The job state is what you claim it is.
- The LoRA used is the one belonging to the art model in scope.
- `reference_hashes` correspond only to files under `data/models/<art-model>/references/`.
- A final PNG is RGBA, 2048×2048, and actually contains transparency.
- `data/usage.jsonl` has both a `draft` and a `final` record for the job.
