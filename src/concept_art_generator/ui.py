"""The Gradio front end.

Two tabs, adapted from the Qwen Image LoRA Studio:

* **References** reuses the studio's Train gallery — a paged grid of thumbnails, each with its own
  caption box, plus `.txt` caption upload matched by filename. There it captions a training
  dataset; here it sets the reference descriptions for one art model.
* **Generate** reuses the studio's Generate panel, with the approval gate added: a draft is always
  produced first, and the 2K final stays locked until a human approves it.

Inference is not done here. This calls `ConceptArtWorkflow`, which calls the Space's
`/v1/generate` or GPT Image 2, so none of the studio's torch/diffusers stack is needed.
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr

from .art_models import ART_MODEL_NAMES, ART_MODELS, BY_NAME
from .models import DEFAULT_BACKGROUND_MODEL, ArtRequest, Backend, JobState
from .workflow import ConceptArtWorkflow

GALLERY_COLUMNS = 4
GALLERY_ROWS = 2
GALLERY_PAGE_SIZE = GALLERY_COLUMNS * GALLERY_ROWS

EXAMPLE_PROMPTS = {model.name: model.example_prompt for model in ART_MODELS}

APP_CSS = """
/* Hidden columns free their space, so a partly filled last row stretched its one visible
   thumbnail across the whole width. Pin every slot to a quarter of the row. */
.gallery-slot {
    flex: 0 1 calc(25% - 0.75rem) !important;
}

/* Gradio draws its progress bar inside the event's output component; an empty Markdown
   collapses to nothing, which made a long captioning run a barely visible sliver. */
.status-panel {
    min-height: 4.5rem;
    display: flex;
    align-items: center;
    justify-content: center;
}

.transparent-result .image-container,
.transparent-result .wrap {
    background-color: #f4f4f4 !important;
    background-image:
        linear-gradient(45deg, #d8d8d8 25%, transparent 25%),
        linear-gradient(-45deg, #d8d8d8 25%, transparent 25%),
        linear-gradient(45deg, transparent 75%, #d8d8d8 75%),
        linear-gradient(-45deg, transparent 75%, #d8d8d8 75%) !important;
    background-position: 0 0, 0 10px, 10px -10px, -10px 0 !important;
    background-size: 20px 20px !important;
}
"""


def caption_progress(descriptions: dict[str, str]) -> str:
    total = len(descriptions)
    if not total:
        return "No references yet. Add images for this art model to start."
    done = sum(1 for text in descriptions.values() if text.strip())
    if done == total:
        return f"All {total} references captioned."
    return (
        f"{done} of {total} references captioned. Descriptions are what GPT ranks when a model "
        "has more than 16 references, so keep them specific."
    )


def describe(job) -> str:
    """The job summary shown under the images, including the exact prompt that was sent."""
    model = BY_NAME[job.art_model]
    lora = f" · LoRA `{job.lora_name}`" if job.lora_name else ""
    sent = job.artifacts.get("draft", {}).get("prompt", "")
    return (
        f"### `{job.id}` — **{job.state}**\n"
        f"{job.art_model} · backend `{job.backend}`{lora} · "
        f"transparent: {job.transparent} · trigger `{model.trigger}`\n\n"
        f"Prompt sent to the provider:\n\n```text\n{sent}\n```"
    )


def approval_gate(job):
    """Approve / reject / export button states.

    Only a `draft_ready` job can be judged, and only an `approved` one can be exported, so the
    2K final stays locked until a human has approved that exact draft.
    """
    judging = job is not None and job.state == JobState.DRAFT_READY
    exporting = job is not None and job.state == JobState.APPROVED
    return (
        gr.update(interactive=judging),
        gr.update(interactive=judging),
        gr.update(interactive=exporting),
    )


def build_ui(workflow: ConceptArtWorkflow) -> gr.Blocks:
    """Build the Blocks app against one workflow instance."""

    # --- References tab helpers ---------------------------------------------

    def gallery_state(art_model: str, page: int):
        descriptions = workflow.reference_descriptions(art_model)
        names = list(descriptions)
        pages = max(1, -(-len(names) // GALLERY_PAGE_SIZE))
        page = min(max(1, int(page or 1)), pages)
        window = names[(page - 1) * GALLERY_PAGE_SIZE : page * GALLERY_PAGE_SIZE]
        workspace = workflow.workspace(art_model)

        slots = []
        for offset in range(GALLERY_PAGE_SIZE):
            if offset < len(window):
                name = window[offset]
                path = str(workspace.reference_paths([name])[0])
                slots += [
                    gr.update(visible=True),
                    gr.update(value=path, label=name),
                    gr.update(value=descriptions[name]),
                ]
            else:
                slots += [
                    gr.update(visible=False),
                    gr.update(value=None, label=""),
                    gr.update(value=""),
                ]
        summary = f"Page {page} of {pages} · {len(names)} reference(s)"
        return (
            *slots,
            page,
            window,
            summary,
            gr.update(interactive=page > 1),
            gr.update(interactive=page < pages),
            caption_progress(descriptions),
        )

    def add_images(art_model, files, page):
        """Store the images and nothing else.

        Captioning is a separate, explicit step, so adding references never waits on a provider
        and never invents a caption for an image the human has not looked at yet.
        """
        for path in files or []:
            source = Path(str(path))
            if source.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                continue
            workflow.add_reference(art_model, source, describe_missing=False)
        return gallery_state(art_model, page)

    def describe_uncaptioned(art_model, progress=gr.Progress()):
        """Caption the references that are still blank, on request.

        This returns only the status line. Gradio draws its progress bar on an event's output
        components, so keeping the gallery out of them confines the bar to one readable panel
        instead of repeating it inside every thumbnail.
        """
        pending = workflow.uncaptioned_references(art_model)
        if not pending:
            return "Every reference already has a description; nothing to do."
        for filename in progress.tqdm(pending, desc="Asking GPT to describe references"):
            workflow.describe_reference(art_model, filename)
        return f"Described {len(pending)} reference(s) with GPT."

    def add_caption_files(art_model, files, page):
        workflow.apply_caption_files(art_model, [Path(str(item)) for item in files or []])
        return gallery_state(art_model, page)

    def save_page_captions(art_model, shown, *texts):
        for name, text in zip(shown or [], texts, strict=False):
            workflow.set_reference_description(art_model, name, (text or "").strip())
        return gallery_state(art_model, 1)

    # --- Generate tab helpers -----------------------------------------------

    def follow_model(art_model, current_prompt):
        """Offer the model's own example while the box is untouched; never discard typed text."""
        example = EXAMPLE_PROMPTS.get(art_model, "")
        untouched = not (current_prompt or "").strip() or current_prompt in EXAMPLE_PROMPTS.values()
        return gr.update(value=example) if untouched else gr.update()

    def create_draft(
        art_model,
        prompt,
        backend,
        transparent,
        reference_count,
        negative_prompt,
        seed,
        steps,
        guidance,
        lora_scale,
        background_model,
    ):
        job = workflow.create_draft(
            ArtRequest(
                art_model,
                prompt,
                Backend(backend),
                int(reference_count),
                transparent=bool(transparent),
                negative_prompt=negative_prompt or "",
                seed=int(seed) if seed is not None else None,
                steps=int(steps),
                guidance_scale=float(guidance),
                lora_scale=float(lora_scale),
                background_model=background_model or DEFAULT_BACKGROUND_MODEL,
            )
        )
        return (job.id, job.draft_path, None, describe(job), *approval_gate(job))

    def approve(art_model, job_id, note):
        job = workflow.approve(art_model, job_id, note or None)
        return (job.id, describe(job), *approval_gate(job))

    def reject(art_model, job_id, note):
        job = workflow.reject(art_model, job_id, note or "")
        return (job.id, describe(job), *approval_gate(job))

    def export_final(art_model, job_id):
        job = workflow.create_final(art_model, job_id)
        return (job.final_path, describe(job), *approval_gate(job))

    def list_jobs(art_model):
        return [
            [job.id, job.state, job.backend, job.prompt, job.created_at]
            for job in workflow.list_jobs(art_model)
        ]

    # --- Blocks --------------------------------------------------------------

    with gr.Blocks(title="Concept Art Generator") as demo:
        gr.Markdown(
            "# Concept Art Generator\n"
            "Three art models, each owning its own references and its own private LoRA. "
            "Every asset goes draft → human approval → transparent 2K final."
        )

        with gr.Tab("References"):
            gr.Markdown(
                "**1. Add the images.** They are stored immediately; nothing is sent anywhere.\n\n"
                "**2. Caption them, separately.** Type in the box under each thumbnail, upload "
                "`.txt` files named after the images (`drone.txt` captions `drone.png`, the Qwen "
                "Image LoRA Studio convention), or press *Describe blank ones with GPT* to fill "
                "only the ones still empty. References are stored per art model and never shared "
                "with another."
            )
            ref_model = gr.Dropdown(
                ART_MODEL_NAMES, value=ART_MODEL_NAMES[0], label="Art model",
                info="References are isolated per model; this picks the folder they land in.",
            )
            with gr.Row():
                upload_images = gr.UploadButton(
                    "Add reference images",
                    file_count="multiple",
                    file_types=["image"],
                    type="filepath",
                    variant="primary",
                )
                upload_captions = gr.UploadButton(
                    "Add caption .txt files",
                    file_count="multiple",
                    file_types=[".txt"],
                    type="filepath",
                )
                describe_button = gr.Button("Describe blank ones with GPT")
            caption_status = gr.Markdown(elem_classes="status-panel")

            page_state = gr.State(1)
            shown_state = gr.State([])
            slot_columns, slot_images, slot_captions = [], [], []
            for _row in range(GALLERY_ROWS):
                with gr.Row():
                    for _column in range(GALLERY_COLUMNS):
                        with gr.Column(visible=False, min_width=180, elem_classes="gallery-slot") as column:
                            image = gr.Image(interactive=False, height=200)
                            caption = gr.Textbox(
                                placeholder="Describe this reference…",
                                lines=2,
                                max_lines=4,
                                show_label=False,
                                container=False,
                            )
                        slot_columns.append(column)
                        slot_images.append(image)
                        slot_captions.append(caption)
            with gr.Row():
                previous_page = gr.Button("Previous", interactive=False)
                gallery_summary = gr.Markdown()
                next_page = gr.Button("Next", interactive=False)
            save_captions = gr.Button("Save captions on this page", variant="primary")

            slot_outputs = []
            for column, image, caption in zip(
                slot_columns, slot_images, slot_captions, strict=True
            ):
                slot_outputs += [column, image, caption]
            gallery_outputs = [
                *slot_outputs,
                page_state,
                shown_state,
                gallery_summary,
                previous_page,
                next_page,
                caption_status,
            ]

            ref_model.change(gallery_state, [ref_model, page_state], gallery_outputs)
            upload_images.upload(
                add_images, [ref_model, upload_images, page_state], gallery_outputs
            )
            describe_button.click(describe_uncaptioned, ref_model, caption_status).then(
                gallery_state, [ref_model, page_state], gallery_outputs
            )
            upload_captions.upload(
                add_caption_files, [ref_model, upload_captions, page_state], gallery_outputs
            )
            save_captions.click(
                save_page_captions, [ref_model, shown_state, *slot_captions], gallery_outputs
            )
            previous_page.click(
                lambda model, page: gallery_state(model, int(page) - 1),
                [ref_model, page_state],
                gallery_outputs,
            )
            next_page.click(
                lambda model, page: gallery_state(model, int(page) + 1),
                [ref_model, page_state],
                gallery_outputs,
            )

        with gr.Tab("Generate"):
            gen_model = gr.Dropdown(
                ART_MODEL_NAMES, value=ART_MODEL_NAMES[0], label="Art model",
                info="Choosing the model chooses its LoRA and its reference folder.",
            )
            prompt = gr.Textbox(
                label="Prompt",
                value=EXAMPLE_PROMPTS[ART_MODEL_NAMES[0]],
                lines=3,
                info=(
                    "Subject first, then the details, then that model's style keywords. The "
                    "trigger word is prepended for you on the Hugging Face backend."
                ),
            )
            with gr.Row():
                backend = gr.Radio(
                    [b.value for b in Backend],
                    value=Backend.HUGGING_FACE.value,
                    label="Backend",
                    info=(
                        "huggingface uses this model's private LoRA. gpt-image-2 uses no LoRA and "
                        "sends this model's reference images instead."
                    ),
                )
                transparent = gr.Checkbox(
                    value=True,
                    label="Transparent PNG",
                    info="Rejects an opaque result rather than exporting a non-transparent asset.",
                )
                reference_count = gr.Slider(
                    1, 16, value=16, step=1, label="References to use",
                    info="Upper bound; GPT ranks descriptions when the model has more.",
                )
            with gr.Accordion("Hugging Face inference settings", open=False):
                negative_prompt = gr.Textbox(label="Negative prompt")
                with gr.Row():
                    steps = gr.Slider(1, 80, value=28, step=1, label="Steps")
                    guidance = gr.Slider(0, 20, value=4.0, step=0.1, label="Guidance")
                    lora_scale = gr.Slider(0, 2, value=1.25, step=0.05, label="LoRA scale")
                with gr.Row():
                    seed = gr.Number(label="Seed (empty = random)", precision=0)
                    background_model = gr.Textbox(
                        value=DEFAULT_BACKGROUND_MODEL, label="Background removal model"
                    )
            draft_button = gr.Button("Create 1024px draft", variant="primary")

            job_state = gr.State(None)
            job_details = gr.Markdown("No draft yet.")
            with gr.Row():
                draft_image = gr.Image(
                    label="Draft (awaiting approval)",
                    format="png",
                    elem_classes="transparent-result",
                )
                final_image = gr.Image(
                    label="Approved 2K final", format="png", elem_classes="transparent-result"
                )

            gr.Markdown("### Human approval\nThe 2K final stays locked until a human approves.")
            with gr.Row():
                decision_note = gr.Textbox(
                    label="Note / feedback",
                    placeholder="Wider silhouette; reduce surface noise",
                    scale=3,
                )
                approve_button = gr.Button("Approve draft", interactive=False)
                reject_button = gr.Button("Request changes", interactive=False)
            final_button = gr.Button("Export 2K final", variant="primary", interactive=False)

            gates = [approve_button, reject_button, final_button]
            gen_model.change(follow_model, [gen_model, prompt], prompt)
            draft_button.click(
                create_draft,
                [
                    gen_model, prompt, backend, transparent, reference_count,
                    negative_prompt, seed, steps, guidance, lora_scale, background_model,
                ],
                [job_state, draft_image, final_image, job_details, *gates],
            )
            approve_button.click(
                approve, [gen_model, job_state, decision_note], [job_state, job_details, *gates]
            )
            reject_button.click(
                reject, [gen_model, job_state, decision_note], [job_state, job_details, *gates]
            )
            final_button.click(
                export_final, [gen_model, job_state], [final_image, job_details, *gates]
            )

        with gr.Tab("Jobs"):
            jobs_model = gr.Dropdown(
                ART_MODEL_NAMES, value=ART_MODEL_NAMES[0], label="Art model"
            )
            refresh = gr.Button("Refresh")
            jobs_table = gr.Dataframe(
                headers=["Job", "State", "Backend", "Prompt", "Created"],
                column_count=5,
                interactive=False,
                wrap=True,
            )
            jobs_model.change(list_jobs, jobs_model, jobs_table)
            refresh.click(list_jobs, jobs_model, jobs_table)
            demo.load(list_jobs, jobs_model, jobs_table)

        demo.load(gallery_state, [ref_model, page_state], gallery_outputs)

    return demo


def main() -> None:
    """Launch the Gradio app on its own, without the JSON API."""
    build_ui(ConceptArtWorkflow()).launch(server_name="127.0.0.1", css=APP_CSS)
