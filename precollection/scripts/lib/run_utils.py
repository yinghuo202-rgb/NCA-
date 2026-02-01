from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunDirs:
    run_id: str
    root: Path
    logs: Path
    outputs: Path
    data_snapshots: Path


def make_run_id(run_id: str | None = None) -> str:
    if run_id:
        return run_id
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def init_run_dirs(root: Path, run_id: str) -> RunDirs:
    run_root = root / "runs" / run_id
    logs = run_root / "logs"
    outputs = run_root / "outputs"
    data_snapshots = run_root / "data_snapshots"
    logs.mkdir(parents=True, exist_ok=True)
    outputs.mkdir(parents=True, exist_ok=True)
    data_snapshots.mkdir(parents=True, exist_ok=True)
    return RunDirs(run_id=run_id, root=run_root, logs=logs, outputs=outputs, data_snapshots=data_snapshots)


def write_params_yaml(path: Path, params: dict[str, Any]) -> None:
    try:
        import yaml  # type: ignore

        text = yaml.safe_dump(params, allow_unicode=True, sort_keys=False)
    except Exception:
        # fallback to JSON-like text
        lines = []
        for k, v in params.items():
            lines.append(f"{k}: {v}")
        text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")

