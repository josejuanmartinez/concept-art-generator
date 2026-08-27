from __future__ import annotations

import argparse
import json

from .models import ArtRequest, Backend
from .workflow import ConceptArtWorkflow


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="concept-art", description="Human-supervised concept art generator"
    )
    root.add_argument("--data-dir", default="data")
    actions = root.add_subparsers(dest="action", required=True)
    ref = actions.add_parser("add-reference")
    ref.add_argument("game")
    ref.add_argument("file")
    draft = actions.add_parser("draft")
    draft.add_argument("game")
    draft.add_argument("prompt")
    draft.add_argument("--backend", choices=[b.value for b in Backend], required=True)
    draft.add_argument("--lora-name")
    draft.add_argument("--references", type=int, default=4)
    draft.add_argument(
        "--opaque", action="store_true", help="Allow an opaque output instead of transparent PNG."
    )
    approve = actions.add_parser("approve")
    approve.add_argument("game")
    approve.add_argument("job_id")
    approve.add_argument("--feedback")
    reject = actions.add_parser("reject")
    reject.add_argument("game")
    reject.add_argument("job_id")
    reject.add_argument("--feedback", required=True)
    final = actions.add_parser("final")
    final.add_argument("game")
    final.add_argument("job_id")
    show = actions.add_parser("show")
    show.add_argument("game")
    show.add_argument("job_id")
    return root


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    flow = ConceptArtWorkflow(args.data_dir)
    if args.action == "add-reference":
        result = {"reference": str(flow.add_reference(args.game, args.file))}
    elif args.action == "draft":
        result = flow.create_draft(
            ArtRequest(
                args.game,
                args.prompt,
                Backend(args.backend),
                args.lora_name,
                args.references,
                transparent=not args.opaque,
            )
        ).to_dict()
    elif args.action == "approve":
        result = flow.approve(args.game, args.job_id, args.feedback).to_dict()
    elif args.action == "reject":
        result = flow.reject(args.game, args.job_id, args.feedback).to_dict()
    elif args.action == "final":
        result = flow.create_final(args.game, args.job_id).to_dict()
    else:
        result = flow.get_job(args.game, args.job_id).to_dict()
    print(json.dumps(result, indent=2))
