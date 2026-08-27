from __future__ import annotations

import json
import secrets
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
from .references import MAX_REFERENCE_IMAGES, OpenAIReferenceAgent, ReferenceAgent
from .workspace import GameWorkspace


class ConceptArtWorkflow:
    """Orchestrator: draft → explicit human approval → transparent final."""

    def __init__(
        self,
        data_root: str | Path = "data",
        providers: dict[Backend, ArtProvider] | None = None,
        reference_agent: ReferenceAgent | None = None,
    ):
        self.root = Path(data_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.providers = providers or {
            Backend.HUGGING_FACE: HuggingFaceSpaceProvider(),
            Backend.GPT_IMAGE_2: GPTImage2Provider(),
        }
        self.reference_agent = reference_agent or OpenAIReferenceAgent()
        self.director, self.gate = StyleDirector(), QualityGate()

    def workspace(self, game: str) -> GameWorkspace:
        return GameWorkspace(self.root, game)

    def add_reference(self, game: str, source: str | Path, description: str | None = None) -> Path:
        source = Path(source)
        if source.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            raise ValueError("References must be PNG, JPG, JPEG or WebP.")
        target = self.workspace(game).references / source.name
        if target.exists() and target.read_bytes() != source.read_bytes():
            target = target.with_stem(f"{source.stem}-{source.stat().st_size}")
        target.write_bytes(source.read_bytes())
        effective_description = (description or "").strip()
        if not effective_description:
            effective_description = self.reference_agent.describe(target)
        self.workspace(game).set_reference_description(target.name, effective_description)
        return target

    def create_draft(self, request: ArtRequest) -> ArtJob:
        if not 1 <= request.reference_count <= MAX_REFERENCE_IMAGES:
            raise ValueError(f"reference_count must be between 1 and {MAX_REFERENCE_IMAGES}.")
        ws = self.workspace(request.game)
        references, descriptions = self._select_references(
            ws, request.prompt, request.reference_count
        )
        prompt = self.director.brief(request, ws)
        job = ArtJob(
            game=request.game,
            prompt=request.prompt,
            backend=request.backend,
            lora_name=request.lora_name,
            reference_hashes=ws.fingerprints_for(references),
            reference_files=[path.name for path in references],
            reference_descriptions=descriptions,
            transparent=request.transparent,
            notes=["Draft generated; final is blocked until explicit approval."],
        )
        seed = request.seed
        if request.backend == Backend.HUGGING_FACE and seed is None:
            seed = secrets.randbelow(2**31 - 1)
        draft_size = 1024 if request.backend == Backend.HUGGING_FACE else 512
        rendered, effective_backend = self._render(
            request.backend,
            RenderSpec(
                prompt,
                references,
                draft_size,
                draft_size,
                request.transparent,
                request.lora_name,
                seed,
                request.negative_prompt,
                request.steps,
                request.guidance_scale,
                request.lora_scale,
                request.scheduler,
                request.base_model,
            ),
            job,
        )
        if request.transparent:
            self.gate.require_transparency(rendered.png)
        destination = ws.image_path("drafts", job.id)
        destination.write_bytes(rendered.png)
        job.draft_path = str(destination)
        job.cost_usd += rendered.estimated_cost_usd
        job.generation_parameters = rendered.generation_parameters or {}
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
        references = self._job_references(ws, job)
        parameters = job.generation_parameters
        draft_backend = Backend(
            job.artifacts.get("draft", {}).get("effective_backend", job.backend)
        )
        if draft_backend == Backend.HUGGING_FACE and not parameters:
            raise ValueError(
                "This HF draft has no replayable parameters. Generate a new draft before finalizing."
            )
        request = ArtRequest(
            game,
            job.prompt,
            draft_backend,
            job.lora_name,
            len(references),
            transparent=job.transparent,
            negative_prompt=str(parameters.get("negative_prompt", "")),
            seed=parameters.get("seed"),
            steps=int(parameters.get("steps", 28)),
            guidance_scale=float(parameters.get("guidance_scale", 4.0)),
            lora_scale=float(parameters.get("lora_scale", 0.8)),
            scheduler=parameters.get("scheduler"),
            base_model=parameters.get("base_model"),
        )
        draft_prompt = job.artifacts.get("draft", {}).get("prompt")
        prompt = str(draft_prompt or self.director.brief(request, ws))
        if request.backend == Backend.GPT_IMAGE_2:
            prompt += " Final-quality, 2K deliverable preserving the approved concept."
        render_references = references
        if request.backend == Backend.GPT_IMAGE_2:
            approved_draft = Path(job.draft_path or "")
            if not approved_draft.exists():
                raise ValueError("The approved draft is missing; GPT final consistency is unsafe.")
            render_references = [approved_draft, *references[:15]]
        rendered, effective_backend = self._render(
            request.backend,
            RenderSpec(
                prompt,
                render_references,
                2048,
                2048,
                job.transparent,
                job.lora_name,
                request.seed,
                request.negative_prompt,
                request.steps,
                request.guidance_scale,
                request.lora_scale,
                request.scheduler,
                request.base_model,
            ),
            job,
        )
        if job.transparent:
            self.gate.require_transparency(rendered.png)
        if rendered.generation_parameters:
            replayed = {
                key: rendered.generation_parameters.get(key)
                for key in (
                    "prompt",
                    "lora_name",
                    "negative_prompt",
                    "steps",
                    "guidance_scale",
                    "lora_scale",
                    "seed",
                    "scheduler",
                    "base_model",
                )
            }
            original = {key: parameters.get(key) for key in replayed}
            if request.backend == Backend.HUGGING_FACE and replayed != original:
                raise RuntimeError("Hugging Face final did not replay the approved draft parameters.")
        destination = ws.image_path("finals", job.id)
        image = Image.open(BytesIO(rendered.png)).convert("RGBA")
        # Normalize a provider result only when it does not already have a 2K long edge.
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

    def _select_references(
        self, ws: GameWorkspace, prompt: str, limit: int
    ) -> tuple[list[Path], dict[str, str]]:
        candidates = ws.reference_files()
        if not candidates:
            raise ValueError(f"{ws.game} has no references. Add them before generating.")
        metadata = ws.reference_descriptions()
        for path in candidates:
            if not metadata.get(path.name, "").strip():
                metadata[path.name] = self.reference_agent.describe(path)
                ws.set_reference_description(path.name, metadata[path.name])

        if len(candidates) > limit:
            selected_names = self.reference_agent.select(
                prompt, [(path, metadata[path.name]) for path in candidates], limit
            )
            references = ws.reference_paths(selected_names)
        else:
            references = candidates
        descriptions = {path.name: metadata[path.name] for path in references}
        return references, descriptions

    @staticmethod
    def _job_references(ws: GameWorkspace, job: ArtJob) -> list[Path]:
        if job.reference_files:
            references = ws.reference_paths(job.reference_files)
        else:
            # Compatibility for jobs created before filenames were persisted.
            by_hash = {ws.fingerprints_for([path])[0]: path for path in ws.reference_files()}
            try:
                references = [by_hash[value] for value in job.reference_hashes]
            except KeyError as error:
                raise ValueError("A reference used by this job is no longer available.") from error
        if ws.fingerprints_for(references) != job.reference_hashes:
            raise ValueError("A selected reference has changed since the draft was generated.")
        return references

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
                width=spec.width,
                height=spec.height,
                transparent=spec.transparent,
                lora_name=None,
                seed=spec.seed,
                negative_prompt=spec.negative_prompt,
                steps=spec.steps,
                guidance_scale=spec.guidance_scale,
                lora_scale=spec.lora_scale,
                scheduler=spec.scheduler,
                base_model=spec.base_model,
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
            "reference_files": job.reference_files,
            "reference_descriptions": job.reference_descriptions,
            "transparent": job.transparent,
            "state": job.state,
            "feedback": job.feedback,
            "generation_parameters": job.generation_parameters,
        }
        ws.sidecar_path(image_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _refresh_sidecars(self, ws: GameWorkspace, job: ArtJob) -> None:
        for stage in job.artifacts:
            self._write_sidecar(ws, job, stage)
