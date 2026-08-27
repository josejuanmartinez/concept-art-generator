from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from uuid import uuid4


def now() -> str:
    return datetime.now(UTC).isoformat()


class Backend(StrEnum):
    HUGGING_FACE = "huggingface"
    GPT_IMAGE_2 = "gpt-image-2"


class JobState(StrEnum):
    DRAFT_READY = "draft_ready"
    APPROVED = "approved"
    FINAL_READY = "final_ready"
    REJECTED = "rejected"


@dataclass(slots=True)
class ArtRequest:
    game: str
    prompt: str
    backend: Backend
    lora_name: str | None = None
    reference_count: int = 16
    logo_path: str | None = None
    transparent: bool = True
    negative_prompt: str = ""
    seed: int | None = None
    steps: int = 28
    guidance_scale: float = 4.0
    lora_scale: float = 0.8
    scheduler: str | None = None
    base_model: str | None = None


@dataclass(slots=True)
class ArtJob:
    game: str
    prompt: str
    backend: str
    id: str = field(default_factory=lambda: uuid4().hex[:12])
    state: str = JobState.DRAFT_READY
    created_at: str = field(default_factory=now)
    approved_at: str | None = None
    draft_path: str | None = None
    final_path: str | None = None
    lora_name: str | None = None
    reference_hashes: list[str] = field(default_factory=list)
    reference_files: list[str] = field(default_factory=list)
    reference_descriptions: dict[str, str] = field(default_factory=dict)
    cost_usd: float = 0.0
    transparent: bool = True
    notes: list[str] = field(default_factory=list)
    feedback: list[dict[str, str]] = field(default_factory=list)
    artifacts: dict[str, dict] = field(default_factory=dict)
    generation_parameters: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ArtJob:
        return cls(**data)

    def save(self, path: Path) -> None:
        import json

        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> ArtJob:
        import json

        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
