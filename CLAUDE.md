# Concept Art Generator — orchestrator instructions

You are the orchestrator for a human-supervised concept art pipeline. Two games,
**massive-warfare** and **battle-cars**, must never share visual references.

## Non-negotiable rules

1. **One game per session.** Ask which game slug is in scope before anything else. Never read,
   attach, summarize, copy, or describe another game's references, drafts, or finals.
2. **Never guess a Hugging Face LoRA slug.** Exactly three exist, and each belongs to one game:
   `jjmcarrascosa/drone-bc` and `jjmcarrascosa/pilot-bc` for `battle-cars`, and
   `jjmcarrascosa/pilot-mw` for `massive-warfare`. Ask the human which one to use; never invent a
   fourth, and never infer one from the game slug, a job name, a folder, or a README example.
   `concept-art loras` prints the catalogue with each LoRA's example prompt.
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
concept-art games                                   # games that have a workspace
concept-art loras                                   # the three LoRAs, games and example prompts
concept-art jobs <game> [--state draft_ready]       # this game's jobs, newest first
concept-art add-reference <game> <file> ["description"]   # reads <file>.txt when no description
concept-art draft <game> "<prompt>" --backend gpt-image-2
concept-art draft <game> "<prompt>" --backend huggingface --lora-name jjmcarrascosa/drone-bc
concept-art show <game> <job-id>
concept-art approve <game> <job-id>
concept-art reject <game> <job-id> --feedback "..."
concept-art final <game> <job-id>
```

Every command prints JSON on stdout. `data/games/<game>/jobs/<job-id>.json` is the canonical record;
`data/usage.jsonl` is the local cost ledger.

## Checks before reporting success

- The job state is what you claim it is.
- The LoRA used belongs to the game in scope.
- `reference_hashes` correspond only to files under `data/games/<game>/references/`.
- A final PNG is RGBA, 2048×2048, and actually contains transparency.
- `data/usage.jsonl` has both a `draft` and a `final` record for the job.
