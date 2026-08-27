from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

from PIL import Image

from .agents import QualityGate, StyleDirector
from .models import ArtJob, ArtRequest, Backend, JobState
from .providers import (
    ArtProvider,
    GPTImage2Provider,
    HuggingFaceSpaceProvider,
    RenderedImage,
    RenderSpec,
)
from .workspace import GameWorkspace


class ConceptArtWorkflow:
    """Orchestrator: draft → explicit human approval → transparent final."""

    def __init__(
        self, data_root: str | Path = "data", providers: dict[Backend, ArtProvider] | None = None
    ):
        self.root = Path(data_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.providers = providers or {
            Backend.HUGGING_FACE: HuggingFaceSpaceProvider(),
            Backend.GPT_IMAGE_2: GPTImage2Provider(),
        }
        self.director, self.gate = StyleDirector(), QualityGate()

    def workspace(self, game: str) -> GameWorkspace:
        return GameWorkspace(self.root, game)

    def add_reference(self, game: str, source: str | Path) -> Path:
        source = Path(source)
        if source.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            raise ValueError("References must be PNG, JPG, JPEG or WebP.")
        target = self.workspace(game).references / source.name
        if target.exists() and target.read_bytes() != source.read_bytes():
            target = target.with_stem(f"{source.stem}-{source.stat().st_size}")
        target.write_bytes(source.read_bytes())
        return target

    def create_draft(self, request: ArtRequest) -> ArtJob:
        ws = self.workspace(request.game)
        references = ws.reference_files(request.reference_count)
        prompt = self.director.brief(request, ws)
        job = ArtJob(
            game=request.game,
            prompt=request.prompt,
            backend=request.backend,
            lora_name=request.lora_name,
            reference_hashes=ws.fingerprints(request.reference_count),
            transparent=request.transparent,
            notes=["Draft generated; final is blocked until explicit approval."],
        )
        rendered, effective_backend = self._render(
            request.backend,
            RenderSpec(prompt, references, 512, 512, request.transparent, request.lora_name),
            job,
        )
        if request.transparent:
            self.gate.require_transparency(rendered.png)
        destination = ws.image_path("drafts", job.id)
        destination.write_bytes(rendered.png)
        job.draft_path = str(destination)
        job.cost_usd += rendered.estimated_cost_usd
        job.artifacts["draft"] = self._artifact(
            prompt, effective_backend, rendered.provider_request_id, destination
        )
        job.notes.append(
            f"Effective backend: {effective_backend}; provider request: "
            f"{rendered.provider_request_id or 'not supplied'}"
        )
        job.save(ws.job_file(job.id))
        self._write_sidecar(ws, job, "draft")
        self._ledger(job, "draft")
        return job

    def approve(self, game: str, job_id: str, feedback: str | None = None) -> ArtJob:
        ws = self.workspace(game)
        job = ArtJob.load(ws.job_file(job_id))
        if job.state != JobState.DRAFT_READY:
            raise ValueError(f"Only draft_ready jobs can be approved (current: {job.state}).")
        from .models import now

        job.state, job.approved_at = JobState.APPROVED, now()
        job.feedback.append(
            {"decision": "approved", "at": job.approved_at, "feedback": feedback or ""}
        )
        job.save(ws.job_file(job.id))
        self._refresh_sidecars(ws, job)
        return job

    def reject(self, game: str, job_id: str, feedback: str) -> ArtJob:
        ws = self.workspace(game)
        job = ArtJob.load(ws.job_file(job_id))
        if job.state != JobState.DRAFT_READY:
            raise ValueError(f"Only draft_ready jobs can be rejected (current: {job.state}).")
        if not feedback.strip():
            raise ValueError("Rejection feedback is required so the next draft is actionable.")
        from .models import now

        job.state = JobState.REJECTED
        job.feedback.append({"decision": "rejected", "at": now(), "feedback": feedback.strip()})
        job.save(ws.job_file(job.id))
        self._refresh_sidecars(ws, job)
        return job

    def create_final(self, game: str, job_id: str) -> ArtJob:
        ws = self.workspace(game)
        job = ArtJob.load(ws.job_file(job_id))
        if job.state != JobState.APPROVED:
            raise ValueError("Human approval is required before final generation.")
        references = ws.reference_files(len(job.reference_hashes))
        request = ArtRequest(
            game,
            job.prompt,
            Backend(job.backend),
            job.lora_name,
            len(references),
            transparent=job.transparent,
        )
        prompt = self.director.brief(request, ws) + " Final-quality, 2K long-edge deliverable."
        rendered, effective_backend = self._render(
            request.backend,
            RenderSpec(prompt, references, 2048, 1365, job.transparent, job.lora_name),
            job,
        )
        if job.transparent:
            self.gate.require_transparency(rendered.png)
        destination = ws.image_path("finals", job.id)
        image = Image.open(BytesIO(rendered.png)).convert("RGBA")
        # HF can render a native 2K long edge. GPT Image 2's supported maximum is 1536,
        # so its high-quality source is alpha-preservingly upscaled to the final 2K canvas.
        factor = 2048 / max(image.size)
        if factor != 1:
            image = image.resize(
                (round(image.width * factor), round(image.height * factor)),
                Image.Resampling.LANCZOS,
            )
        image.save(destination, "PNG")
        job.final_path, job.state = str(destination), JobState.FINAL_READY
        job.cost_usd += rendered.estimated_cost_usd
        job.artifacts["final"] = self._artifact(
            prompt, effective_backend, rendered.provider_request_id, destination
        )
        job.notes.append(
            f"Final effective backend: {effective_backend}; exported {image.width}x{image.height} PNG."
        )
        job.save(ws.job_file(job.id))
        self._write_sidecar(ws, job, "final")
        self._ledger(job, "final")
        return job

    def get_job(self, game: str, job_id: str) -> ArtJob:
        return ArtJob.load(self.workspace(game).job_file(job_id))

    def _render(
        self, requested: Backend, spec: RenderSpec, job: ArtJob
    ) -> tuple[RenderedImage, Backend]:
        """Use HF first; retry with game-local GPT Image 2 references when it fails."""
        try:
            return self.providers[requested].render(spec), requested
        except Exception as error:
            if requested != Backend.HUGGING_FACE or Backend.GPT_IMAGE_2 not in self.providers:
                raise
            if not spec.references:
                raise RuntimeError(
                    "HF failed and no game-local references are available for GPT fallback."
                ) from error
            fallback_spec = RenderSpec(
                prompt=spec.prompt,
                references=spec.references,
                # GPT Image 2's largest supported landscape source is 1536x1024.
                width=min(spec.width, 1536),
                height=min(spec.height, 1024),
                transparent=spec.transparent,
                lora_name=None,
                seed=spec.seed,
            )
            job.notes.append(
                f"Hugging Face failed ({type(error).__name__}); retried with GPT Image 2."
            )
            return self.providers[Backend.GPT_IMAGE_2].render(fallback_spec), Backend.GPT_IMAGE_2

    def _ledger(self, job: ArtJob, stage: str) -> None:
        record = {
            "job_id": job.id,
            "game": job.game,
            "stage": stage,
            "backend": job.backend,
            "cost_usd": job.cost_usd,
            "at": job.created_at,
        }
        with (self.root / "usage.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    @staticmethod
    def _artifact(prompt: str, backend: Backend, request_id: str | None, image_path: Path) -> dict:
        model = (
            "gpt-image-2" if backend == Backend.GPT_IMAGE_2 else "Qwen-Image-2512 (private LoRA)"
        )
        with Image.open(image_path) as image:
            dimensions = list(image.size)
        return {
            "tool": "concept-art-generator",
            "model": model,
            "effective_backend": str(backend),
            "prompt": prompt,
            "provider_request_id": request_id,
            "dimensions": dimensions,
        }

    def _write_sidecar(self, ws: GameWorkspace, job: ArtJob, stage: str) -> None:
        image_path = Path(job.draft_path if stage == "draft" else job.final_path or "")
        if not image_path.exists():
            return
        payload = {
            **job.artifacts[stage],
            "job_id": job.id,
            "game": job.game,
            "requested_backend": str(job.backend),
            "lora_name": job.lora_name,
            "reference_hashes": job.reference_hashes,
            "transparent": job.transparent,
            "state": job.state,
            "feedback": job.feedback,
        }
        ws.sidecar_path(image_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _refresh_sidecars(self, ws: GameWorkspace, job: ArtJob) -> None:
        for stage in job.artifacts:
            self._write_sidecar(ws, job, stage)
