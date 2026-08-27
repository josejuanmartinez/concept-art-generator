# Concept Art Generator agent operating manual

The interactive Claude Code session is the **orchestrator**. It must use the project subagents and report only verified outcomes.

1. Ask which single game slug is in scope. Never read, attach, summarize, or transmit another game's references.
2. Delegate planning to `style-director`; it is read-only and game-scoped.
3. Have `technical-artist` make one 512px draft. Do not produce a final in the same step.
4. Wait for explicit human approval via `concept-art approve <game> <job-id>`.
5. Delegate final export then invoke `verifier` last.
6. Route failures back to `technical-artist`, never to `verifier`.

## Backends

- **Hugging Face Space:** requires a LoRA name and `HF_SPACE_URL` / `HF_API_TOKEN`. Do not send reference images to it.
- **GPT Image 2:** sends only the selected references inside `data/games/<game>/references/` to the edit API.

Never place API keys in prompts, source files, job JSON, logs, screenshots, or commits. Keep invoices in provider accounts; use `data/usage.jsonl` as the local usage log.
