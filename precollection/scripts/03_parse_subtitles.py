from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from lib.paths import ensure_dirs, get_project_paths
from lib.run_utils import init_run_dirs


def _read_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"未找到 manifest：{path}（先运行 scripts/01_build_manifest.py）")
    return pd.read_csv(path, dtype=str).fillna("")


def _extract_lines(bcc: object) -> list[str]:
    if isinstance(bcc, dict):
        body = bcc.get("body")
        if isinstance(body, list):
            return [str(seg.get("content") or "") for seg in body if str(seg.get("content") or "").strip()]
        return []
    if isinstance(bcc, list):
        return [str(seg.get("content") or "") for seg in bcc if isinstance(seg, dict) and str(seg.get("content") or "").strip()]
    return []


def _load_bvids_from_log(path: Path) -> set[str]:
    if not path or not path.exists():
        return set()
    bvids: set[str] = set()
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if str(obj.get("status", "")).strip() != "OK":
            continue
        bvid = str(obj.get("bvid", "")).strip()
        if bvid:
            bvids.add(bvid)
    return bvids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="", help="manifest path (default data/raw/meta/videos_manifest.csv)")
    parser.add_argument("--run-id", default="", help="run id to write into data/processed/text/<run_id>/...")
    parser.add_argument("--only-new", action="store_true", help="仅处理尚未生成 raw.txt 的样本")
    parser.add_argument("--since-log", default="", help="仅处理 download.jsonl 中 status=OK 的 bvid")
    args = parser.parse_args()

    paths = get_project_paths()
    ensure_dirs(paths)

    manifest_path = paths.data_raw_meta / "videos_manifest.csv"
    if args.manifest:
        manifest_path = Path(args.manifest)
    df = _read_manifest(manifest_path)

    run_id = str(args.run_id).strip()
    since_bvids = _load_bvids_from_log(Path(args.since_log)) if args.since_log else set()
    if run_id:
        run_dirs = init_run_dirs(paths.root, run_id)

    new_raw = 0
    for _, r in tqdm(df.iterrows(), total=len(df), desc="Parse subtitles"):
        bvid = str(r.get("bvid", "")).strip()
        unique_key = str(r.get("unique_key", "")).strip() or bvid
        creator_id = str(r.get("creator_id", "")).strip()
        subtitle_status = str(r.get("subtitle_status", "")).strip()
        if not bvid or not creator_id:
            continue
        if since_bvids and bvid not in since_bvids:
            continue

        if run_id:
            out_dir = paths.data_processed_text / run_id / creator_id / unique_key
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "raw.txt"
        else:
            out_dir = paths.data_processed_text / creator_id
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"raw_{bvid}.txt"

        if subtitle_status != "OK":
            if out_path.exists():
                out_path.unlink()
            continue

        if args.only_new and out_path.exists():
            continue

        raw_sub_path = paths.data_raw_subtitles / creator_id / f"{bvid}.bcc.json"
        if not raw_sub_path.exists():
            if out_path.exists():
                out_path.unlink()
            continue

        try:
            bcc = json.loads(raw_sub_path.read_text(encoding="utf-8"))
        except Exception:
            # 保守：遇到异常跳过，后续在报告里会反映 NO_SUBTITLE/ERR
            if out_path.exists():
                out_path.unlink()
            continue

        lines = _extract_lines(bcc)
        text = "\n".join([ln.strip() for ln in lines if ln.strip()])
        if text:
            text = text.strip() + "\n"

        out_path.write_text(text, encoding="utf-8")
        new_raw += 1

    if run_id:
        if run_id:
            total_raw = len(list((paths.data_processed_text / run_id).rglob("raw.txt")))
        else:
            total_raw = len(list(paths.data_processed_text.rglob("raw_*.txt")))
        inc_path = run_dirs.logs / "run_log_incremental.txt"
        with inc_path.open("a", encoding="utf-8") as f:
            f.write(
                f"[{Path(manifest_path).name}] parse_subtitles: new_raw={new_raw} total_raw={total_raw} "
                f"(only_new={args.only_new}, since_log={bool(args.since_log)})\n"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
