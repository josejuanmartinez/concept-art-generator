from __future__ import annotations

import hashlib
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

    def reference_files(self, limit: int) -> list[Path]:
        allowed = {".png", ".jpg", ".jpeg", ".webp"}
        return sorted(p for p in self.references.iterdir() if p.suffix.lower() in allowed)[:limit]

    def fingerprints(self, limit: int) -> list[str]:
        return [
            hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in self.reference_files(limit)
        ]

    def job_file(self, job_id: str) -> Path:
        return self.path / "jobs" / f"{job_id}.json"

    def image_path(self, stage: str, job_id: str) -> Path:
        return self.path / stage / f"{job_id}.png"

    @staticmethod
    def sidecar_path(image_path: Path) -> Path:
        """Match the studio convention: `asset.png.prompt`."""
        return image_path.with_suffix(image_path.suffix + ".prompt")
