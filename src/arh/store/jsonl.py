"""Append-only JSONL trial store. Same fields as the eventual Postgres trials
table - migration is a bulk load, not a rewrite."""

from pathlib import Path

from arh.models import TrialResult


class JsonlStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, result: TrialResult) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(result.model_dump_json() + "\n")

    def load(self) -> list[TrialResult]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as f:
            return [TrialResult.model_validate_json(line) for line in f if line.strip()]
