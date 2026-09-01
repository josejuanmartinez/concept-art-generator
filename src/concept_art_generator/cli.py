from __future__ import annotations

import argparse
import json

from .art_models import ART_MODEL_NAMES, ART_MODELS, catalogue
from .models import DEFAULT_BACKGROUND_MODEL, ArtRequest, Backend, JobState
from .workflow import ConceptArtWorkflow


def examples() -> str:
    """Show a real, runnable command per art model, using that model's own prompt style."""
    wrap = " \\"
    lines = ["Only three art models exist; each owns its references and its LoRA:", ""]
    for model in ART_MODELS:
        lines += [
            f"  concept-art draft {model.name}{wrap}",
            f'    "{model.example_prompt}"{wrap}',
            "    --backend huggingface",
            "",
        ]
    first = ART_MODELS[0]
    lines += [
        "GPT Image 2 uses no LoRA; it works from the same model's references:",
        "",
        f"  concept-art draft {first.name}{wrap}",
        f'    "{first.example_prompt}"{wrap}',
        "    --backend gpt-image-2",
        "",
        "Run `concept-art models` for the catalogue as JSON.",
    ]
    return "\n".join(lines)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="concept-art",
        description="Human-supervised concept art generator",
        epilog=examples(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    root.add_argument("--data-dir", default="data")
    actions = root.add_subparsers(dest="action", required=True)
    ref = actions.add_parser("add-reference")
    ref.add_argument("art_model", choices=ART_MODEL_NAMES)
    ref.add_argument("file")
    ref.add_argument(
        "description",
        nargs="?",
        help="Optional image description; when omitted GPT creates one with OPENAI_API_KEY.",
    )
    draft = actions.add_parser("draft")
    draft.add_argument("art_model", choices=ART_MODEL_NAMES)
    draft.add_argument("prompt")
    draft.add_argument("--backend", choices=[b.value for b in Backend], required=True)
    draft.add_argument("--negative-prompt", default="")
    draft.add_argument("--seed", type=int)
    draft.add_argument("--steps", type=int, default=28)
    draft.add_argument("--guidance-scale", type=float, default=4.0)
    draft.add_argument("--lora-scale", type=float, default=1.25)
    draft.add_argument("--background-model", default=DEFAULT_BACKGROUND_MODEL)
    draft.add_argument("--references", type=int, default=16, choices=range(1, 17))
    draft.add_argument(
        "--opaque", action="store_true", help="Allow an opaque output instead of transparent PNG."
    )
    approve = actions.add_parser("approve")
    approve.add_argument("art_model", choices=ART_MODEL_NAMES)
    approve.add_argument("job_id")
    approve.add_argument("--feedback")
    reject = actions.add_parser("reject")
    reject.add_argument("art_model", choices=ART_MODEL_NAMES)
    reject.add_argument("job_id")
    reject.add_argument("--feedback", required=True)
    final = actions.add_parser("final")
    final.add_argument("art_model", choices=ART_MODEL_NAMES)
    final.add_argument("job_id")
    show = actions.add_parser("show")
    show.add_argument("art_model", choices=ART_MODEL_NAMES)
    show.add_argument("job_id")
    actions.add_parser("models")
    jobs = actions.add_parser("jobs")
    jobs.add_argument("art_model", choices=ART_MODEL_NAMES)
    jobs.add_argument("--state", choices=[s.value for s in JobState])
    return root


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    flow = ConceptArtWorkflow(args.data_dir)
    if args.action == "add-reference":
        result = {
            "reference": str(flow.add_reference(args.art_model, args.file, args.description))
        }
    elif args.action == "draft":
        result = flow.create_draft(
            ArtRequest(
                args.art_model,
                args.prompt,
                Backend(args.backend),
                args.references,
                transparent=not args.opaque,
                negative_prompt=args.negative_prompt,
                seed=args.seed,
                steps=args.steps,
                guidance_scale=args.guidance_scale,
                lora_scale=args.lora_scale,
                background_model=args.background_model,
            )
        ).to_dict()
    elif args.action == "approve":
        result = flow.approve(args.art_model, args.job_id, args.feedback).to_dict()
    elif args.action == "reject":
        result = flow.reject(args.art_model, args.job_id, args.feedback).to_dict()
    elif args.action == "final":
        result = flow.create_final(args.art_model, args.job_id).to_dict()
    elif args.action == "models":
        result = {"art_models": catalogue()}
    elif args.action == "jobs":
        jobs = flow.list_jobs(args.art_model)
        if args.state:
            jobs = [job for job in jobs if job.state == args.state]
        result = {
            "jobs": [
                {
                    "id": job.id,
                    "state": job.state,
                    "backend": job.backend,
                    "prompt": job.prompt,
                    "created_at": job.created_at,
                }
                for job in jobs
            ]
        }
    else:
        result = flow.get_job(args.art_model, args.job_id).to_dict()
    print(json.dumps(result, indent=2))
