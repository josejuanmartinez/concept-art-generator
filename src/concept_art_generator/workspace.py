from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


class IsolationError(ValueError):
    pass


class GameWorkspace:
    """Owns every path used by one game; no provider receives other-game references."""

    def __init__(self, root: Path, game: str):
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", game):
            raise IsolationError("Game must be a lowercase slug (letters, numbers, hyphens).")
        self.root = root.resolve()
        self.game = game
        self.path = (self.root / "games" / game).resolve()
        if self.root not in self.path.parents:
            raise IsolationError("Invalid workspace path.")
        for name in ("references", "jobs", "drafts", "approved", "finals"):
            (self.path / name).mkdir(parents=True, exist_ok=True)

    @property
    def references(self) -> Path:
        return self.path / "references"

    def reference_files(self, limit: int | None = None) -> list[Path]:
        allowed = {".png", ".jpg", ".jpeg", ".webp"}
        files = sorted(p for p in self.references.iterdir() if p.suffix.lower() in allowed)
        return files if limit is None else files[:limit]

    @property
    def reference_metadata_file(self) -> Path:
        return self.references / "descriptions.json"

    def reference_descriptions(self) -> dict[str, str]:
        if not self.reference_metadata_file.exists():
            return {}
        value = json.loads(self.reference_metadata_file.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise TypeError("Reference descriptions metadata must be a JSON object.")
        return {str(name): str(description) for name, description in value.items()}

    def set_reference_description(self, filename: str, description: str) -> None:
        metadata = self.reference_descriptions()
        metadata[filename] = description.strip()
        self.reference_metadata_file.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def reference_paths(self, filenames: list[str]) -> list[Path]:
        available = {path.name: path for path in self.reference_files()}
        missing = [name for name in filenames if name not in available]
        if missing:
            raise ValueError(f"Selected reference files are missing: {', '.join(missing)}")
        return [available[name] for name in filenames]

    def fingerprints(self, limit: int) -> list[str]:
        return [
            hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in self.reference_files(limit)
        ]

    @staticmethod
    def fingerprints_for(paths: list[Path]) -> list[str]:
        return [hashlib.sha256(path.read_bytes()).hexdigest()[:16] for path in paths]

    def job_file(self, job_id: str) -> Path:
        return self.path / "jobs" / f"{job_id}.json"

    def image_path(self, stage: str, job_id: str) -> Path:
        return self.path / stage / f"{job_id}.png"

    @staticmethod
    def sidecar_path(image_path: Path) -> Path:
        """Match the studio convention: `asset.png.prompt`."""
        return image_path.with_suffix(image_path.suffix + ".prompt")
