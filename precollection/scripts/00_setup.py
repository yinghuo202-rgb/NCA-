from __future__ import annotations

from pathlib import Path

from lib.paths import ensure_dirs, get_project_paths


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(str(path))


def main() -> int:
    paths = get_project_paths()
    ensure_dirs(paths)

    _require_file(paths.config / "breakpoints.json")
    _require_file(paths.config / "sample_manifest_seed.csv")
    _require_file(paths.resources / "onomatopoeia.txt")
    _require_file(paths.resources / "connectors.txt")
    _require_file(paths.resources / "frame_markers.txt")
    _require_file(paths.resources / "modality.txt")

    print(f"Project root: {paths.root}")
    print("OK: 目录与必要配置已就绪")
    cookies_path = paths.secrets / "cookies.json"
    if not cookies_path.exists():
        print(f"NOTE: 未找到 {cookies_path}（可选，但会影响字幕可得率）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

