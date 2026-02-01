from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    config: Path
    inputs: Path
    resources: Path
    secrets: Path

    data_raw: Path
    data_raw_meta: Path
    data_raw_subtitles: Path

    data_processed: Path
    data_processed_text: Path
    data_processed_qc: Path
    data_creators: Path
    data_index: Path

    analysis: Path
    analysis_figures: Path

    outputs: Path
    runs: Path
    logs: Path


def get_project_paths() -> ProjectPaths:
    root = Path(__file__).resolve().parents[2]
    return ProjectPaths(
        root=root,
        config=root / "config",
        inputs=root / "inputs",
        resources=root / "resources",
        secrets=root / "secrets",
        data_raw=root / "data" / "raw",
        data_raw_meta=root / "data" / "raw" / "meta",
        data_raw_subtitles=root / "data" / "raw" / "subtitles",
        data_processed=root / "data" / "processed",
        data_processed_text=root / "data" / "processed" / "text",
        data_processed_qc=root / "data" / "processed" / "qc",
        data_creators=root / "data" / "creators",
        data_index=root / "data" / "index",
        analysis=root / "analysis",
        analysis_figures=root / "analysis" / "figures",
        outputs=root / "outputs",
        runs=root / "runs",
        logs=root / "logs",
    )


def ensure_dirs(paths: ProjectPaths) -> None:
    dirs = [
        paths.config,
        paths.inputs,
        paths.resources,
        paths.secrets,
        paths.data_raw_meta,
        paths.data_raw_subtitles,
        paths.data_processed_text,
        paths.data_processed_qc,
        paths.data_creators,
        paths.data_index,
        paths.analysis,
        paths.analysis_figures,
        paths.outputs,
        paths.runs,
        paths.logs,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
