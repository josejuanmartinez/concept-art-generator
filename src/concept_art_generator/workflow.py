from __future__ import annotations

import json
import secrets
from collections.abc import Callable
from io import BytesIO
from pathlib import Path

from PIL import Image

from .agents import QualityGate
from .art_models import resolve_model
from .branding import stamp, stamp_bytes
from .models import DEFAULT_BACKGROUND_MODEL, ArtJob, ArtRequest, Backend, JobState
from .prompts import build_prompt
from .providers import (
    ArtProvider,
    GPTImage2Provider,
    HuggingFaceSpaceProvider,
    RenderedImage,
    RenderSpec,
)
from .references import (
    MAX_REFERENCE_IMAGES,
    OpenAIReferenceAgent,
    ReferenceAgent,
    caption_for,
    load_caption_sidecars,
    sidecar_caption,
)
from .workspace import ModelWorkspace


class ConceptArtWorkflow:
    """Orchestrator: draft → explicit human approval → transparent final, per art model."""

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
        self.gate = QualityGate()

    def workspace(self, art_model: str) -> ModelWorkspace:
        return ModelWorkspace(self.root, art_model)

    def add_reference(
        self,
        art_model: str,
        source: str | Path,
        description: str | None = None,
        *,
        describe_missing: bool = True,
    ) -> Path:
        """Caption priority: the supplied text, then a sibling `.txt`, then GPT.

        Adding a file this art model already holds is idempotent: the image is not duplicated,
        and its stored caption is left alone unless you supply a new one. Re-adding a folder
        therefore never overwrites hand-written captions, and never spends a GPT call on a
        reference that already has a description.

        `describe_missing=False` stores an empty description instead of calling GPT, for the
        UI flow where a human captions the references by hand afterwards.
        """
        source = Path(source)
        if source.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            raise ValueError("References must be PNG, JPG, JPEG or WebP.")
        workspace = self.workspace(art_model)
        content = source.read_bytes()
        target = workspace.references / source.name
        if target.exists() and target.read_bytes() != content:
            # A different image under a name this model already uses; keep both.
            target = target.with_stem(f"{source.stem}-{source.stat().st_size}")
        already_held = target.exists() and target.read_bytes() == content
        stored = workspace.reference_descriptions().get(target.name, "").strip()
        target.write_bytes(content)

        effective_description = (description or "").strip() or (sidecar_caption(source) or "")
        if not effective_description and already_held and stored:
            return target
        if not effective_description and describe_missing:
            effective_description = self.reference_agent.describe(target)
        workspace.set_reference_description(target.name, effective_description)
        return target

    def reference_descriptions(self, art_model: str) -> dict[str, str]:
        """Every reference filename for this art model mapped to its stored description."""
        ws = self.workspace(art_model)
        stored = ws.reference_descriptions()
        return {path.name: stored.get(path.name, "") for path in ws.reference_files()}

    def set_reference_description(self, art_model: str, filename: str, description: str) -> None:
        ws = self.workspace(art_model)
        ws.reference_paths([filename])  # refuse a name that is not this model's reference
        ws.set_reference_description(filename, description)

    def uncaptioned_references(self, art_model: str) -> list[str]:
        """Reference filenames that still have no description."""
        return [
            name
            for name, text in self.reference_descriptions(art_model).items()
            if not text.strip()
        ]

    def describe_reference(self, art_model: str, filename: str) -> str:
        """Ask the configured GPT model to describe one stored reference.

        Captioning is a deliberate, separate step from adding images: nothing calls this unless
        a human asks for it, so uploading references never blocks on the provider.
        """
        ws = self.workspace(art_model)
        path = ws.reference_paths([filename])[0]
        description = self.reference_agent.describe(path)
        ws.set_reference_description(filename, description)
        return description

    def apply_caption_files(self, art_model: str, caption_paths: list[Path]) -> int:
        """Fill descriptions from uploaded `.txt` files matched by stem; returns matches."""
        sidecars = load_caption_sidecars(caption_paths)
        matched = 0
        for filename in self.reference_descriptions(art_model):
            text = caption_for(filename, sidecars)
            if text:
                self.set_reference_description(art_model, filename, text)
                matched += 1
        return matched

    def create_draft(self, request: ArtRequest) -> ArtJob:
        if not 1 <= request.reference_count <= MAX_REFERENCE_IMAGES:
            raise ValueError(f"reference_count must be between 1 and {MAX_REFERENCE_IMAGES}.")
        # Choosing the art model chooses the LoRA; a slug is never supplied by a caller.
        model = resolve_model(request.art_model)
        ws = self.workspace(request.art_model)
        references, descriptions = self._select_references(
            ws,
            request.prompt,
            request.reference_count,
            required=request.backend != Backend.HUGGING_FACE,
        )
        prompt = build_prompt(request, descriptions)
        job = ArtJob(
            art_model=request.art_model,
            prompt=request.prompt,
            backend=request.backend,
            lora_name=model.lora_slug if request.backend == Backend.HUGGING_FACE else None,
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
        rendered, effective_backend, sent_prompt = self._render(
            request.backend,
            RenderSpec(
                prompt,
                references,
                draft_size,
                draft_size,
                request.transparent,
                job.lora_name,
                seed,
                request.negative_prompt,
                request.steps,
                request.guidance_scale,
                request.lora_scale,
                request.scheduler,
                request.base_model,
                request.background_model,
            ),
            job,
            lambda backend: build_prompt(request, descriptions, backend),
        )
        if request.transparent:
            self.gate.require_transparency(rendered.png)
        destination = ws.image_path("drafts", job.id)
        # The logo goes on after the transparency gate, so the gate still judges what the
        # provider actually returned rather than a corner this pipeline painted itself.
        png, logo = stamp_bytes(rendered.png, job.art_model)
        destination.write_bytes(png)
        job.draft_path = str(destination)
        job.cost_usd += rendered.estimated_cost_usd
        job.generation_parameters = rendered.generation_parameters or {}
        job.artifacts["draft"] = self._artifact(
            sent_prompt, effective_backend, rendered.provider_request_id, destination, logo
        )
        job.notes.append(
            f"Effective backend: {effective_backend}; provider request: "
            f"{rendered.provider_request_id or 'not supplied'}"
        )
        if logo:
            job.notes.append(f"Stamped {logo} into the bottom-left corner of the draft.")
        job.save(ws.job_file(job.id))
        self._write_sidecar(ws, job, "draft")
        self._ledger(job, "draft")
        return job

    def approve(self, art_model: str, job_id: str, feedback: str | None = None) -> ArtJob:
        ws = self.workspace(art_model)
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

    def reject(self, art_model: str, job_id: str, feedback: str) -> ArtJob:
        ws = self.workspace(art_model)
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

    def create_final(self, art_model: str, job_id: str) -> ArtJob:
        ws = self.workspace(art_model)
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
            art_model,
            job.prompt,
            draft_backend,
            reference_count=max(1, len(references)),
            transparent=job.transparent,
            negative_prompt=str(parameters.get("negative_prompt", "")),
            seed=parameters.get("seed"),
            steps=int(parameters.get("steps", 28)),
            guidance_scale=float(parameters.get("guidance_scale", 4.0)),
            lora_scale=float(parameters.get("lora_scale", 1.25)),
            scheduler=parameters.get("scheduler"),
            base_model=parameters.get("base_model"),
            background_model=str(
                parameters.get("background_model") or DEFAULT_BACKGROUND_MODEL
            ),
        )
        # The HF final must replay the approved draft's prompt byte for byte.
        draft_prompt = job.artifacts.get("draft", {}).get("prompt")
        prompt = str(draft_prompt or build_prompt(request, job.reference_descriptions))
        if request.backend == Backend.GPT_IMAGE_2:
            prompt += " Final-quality, 2K deliverable preserving the approved concept."
        render_references = references
        if request.backend == Backend.GPT_IMAGE_2:
            approved_draft = Path(job.draft_path or "")
            if not approved_draft.exists():
                raise ValueError("The approved draft is missing; GPT final consistency is unsafe.")
            render_references = [approved_draft, *references[:15]]
        rendered, effective_backend, sent_prompt = self._render(
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
                request.background_model,
            ),
            job,
            lambda backend: build_prompt(request, job.reference_descriptions, backend),
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
                    "background_model",
                )
            }
            original = {key: parameters.get(key) for key in replayed}
            if effective_backend == Backend.HUGGING_FACE and replayed != original:
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
        image, logo = stamp(image, job.art_model)
        image.save(destination, "PNG")
        job.final_path, job.state = str(destination), JobState.FINAL_READY
        job.cost_usd += rendered.estimated_cost_usd
        job.artifacts["final"] = self._artifact(
            sent_prompt, effective_backend, rendered.provider_request_id, destination, logo
        )
        job.notes.append(
            f"Final effective backend: {effective_backend}; exported {image.width}x{image.height} PNG."
        )
        if logo:
            job.notes.append(f"Stamped {logo} into the bottom-left corner of the final.")
        job.save(ws.job_file(job.id))
        self._write_sidecar(ws, job, "final")
        self._ledger(job, "final")
        return job

    def _select_references(
        self, ws: ModelWorkspace, prompt: str, limit: int, required: bool = True
    ) -> tuple[list[Path], dict[str, str]]:
        """Pick this model's references, captioning and ranking them if needed.

        `required` is False for Hugging Face: only `GPTImage2Provider` reads `spec.references`,
        so a LoRA draft needs none. They are still gathered when present, because the GPT
        fallback attaches them if the Space fails.
        """
        candidates = ws.reference_files()
        if not candidates:
            if not required:
                return [], {}
            raise ValueError(f"{ws.art_model} has no references. Add them before generating.")
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
    def _job_references(ws: ModelWorkspace, job: ArtJob) -> list[Path]:
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

    def get_job(self, art_model: str, job_id: str) -> ArtJob:
        return ArtJob.load(self.workspace(art_model).job_file(job_id))

    def art_models(self) -> list[str]:
        """Every art model that already owns an isolated workspace folder."""
        container = self.root / "models"
        if not container.exists():
            return []
        return sorted(path.name for path in container.iterdir() if path.is_dir())

    def list_jobs(self, art_model: str) -> list[ArtJob]:
        """This model's jobs, newest first; used by the CLI, the UI and agents."""
        ws = self.workspace(art_model)
        jobs = [ArtJob.load(path) for path in (ws.path / "jobs").glob("*.json")]
        return sorted(jobs, key=lambda job: job.created_at, reverse=True)

    def _render(
        self,
        requested: Backend,
        spec: RenderSpec,
        job: ArtJob,
        rebuild_prompt: Callable[[Backend], str] | None = None,
    ) -> tuple[RenderedImage, Backend, str]:
        """Use HF first; retry with the model's own GPT Image 2 references when it fails.

        The fallback rebuilds the prompt for GPT Image 2: the LoRA trigger word means nothing
        to it, and it needs the reference-derived style instead. That rebuilt prompt is returned
        alongside the backend, because it — not the one we set out to send — is what the job
        record has to name.
        """
        try:
            return self.providers[requested].render(spec), requested, spec.prompt
        except Exception as error:
            if requested != Backend.HUGGING_FACE or Backend.GPT_IMAGE_2 not in self.providers:
                raise
            if not spec.references:
                raise RuntimeError(
                    "HF failed and no references are available for this art model's GPT fallback."
                ) from error
            fallback_prompt = rebuild_prompt(Backend.GPT_IMAGE_2) if rebuild_prompt else spec.prompt
            fallback_spec = RenderSpec(
                prompt=fallback_prompt,
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
                background_model=spec.background_model,
            )
            job.notes.append(
                f"Hugging Face failed ({type(error).__name__}); retried with GPT Image 2."
            )
            return (
                self.providers[Backend.GPT_IMAGE_2].render(fallback_spec),
                Backend.GPT_IMAGE_2,
                fallback_prompt,
            )

    def _ledger(self, job: ArtJob, stage: str) -> None:
        record = {
            "job_id": job.id,
            "art_model": job.art_model,
            "stage": stage,
            "backend": job.backend,
            "cost_usd": job.cost_usd,
            "at": job.created_at,
        }
        with (self.root / "usage.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    @staticmethod
    def _artifact(
        prompt: str,
        backend: Backend,
        request_id: str | None,
        image_path: Path,
        logo: str | None = None,
    ) -> dict:
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
            "logo": logo,
        }

    def _write_sidecar(self, ws: ModelWorkspace, job: ArtJob, stage: str) -> None:
        image_path = Path(job.draft_path if stage == "draft" else job.final_path or "")
        if not image_path.exists():
            return
        payload = {
            **job.artifacts[stage],
            "job_id": job.id,
            "art_model": job.art_model,
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

    def _refresh_sidecars(self, ws: ModelWorkspace, job: ArtJob) -> None:
        for stage in job.artifacts:
            self._write_sidecar(ws, job, stage)
