from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from lib.lexicon import load_terms
from lib.paths import ensure_dirs, get_project_paths
from lib.run_utils import init_run_dirs
from lib.text_processing import clean_text, count_chars, count_tokens, split_sentences


def _read_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"未找到 manifest：{path}（先运行 scripts/01_build_manifest.py）")
    return pd.read_csv(path, dtype=str).fillna("")


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
    parser.add_argument("--run-id", default="", help="run id to read/write data/processed/text/<run_id>/...")
    parser.add_argument("--only-new", action="store_true", help="仅处理尚未生成 clean.txt 的样本")
    parser.add_argument("--since-log", default="", help="仅处理 download.jsonl 中 status=OK 的 bvid")
    args = parser.parse_args()

    paths = get_project_paths()
    ensure_dirs(paths)

    manifest_path = paths.data_raw_meta / "videos_manifest.csv"
    if args.manifest:
        manifest_path = Path(args.manifest)
    df = _read_manifest(manifest_path)

    onomatopoeia = load_terms(paths.resources / "onomatopoeia.txt")

    run_id = str(args.run_id).strip()
    since_bvids = _load_bvids_from_log(Path(args.since_log)) if args.since_log else set()
    if run_id:
        run_dirs = init_run_dirs(paths.root, run_id)

    new_clean = 0
    for _, r in tqdm(df.iterrows(), total=len(df), desc="Clean text"):
        bvid = str(r.get("bvid", "")).strip()
        unique_key = str(r.get("unique_key", "")).strip() or bvid
        creator_id = str(r.get("creator_id", "")).strip()
        series = str(r.get("series", "")).strip()
        subtitle_status = str(r.get("subtitle_status", "")).strip()
        if not bvid or not creator_id:
            continue
        if since_bvids and bvid not in since_bvids:
            continue

        if run_id:
            out_dir = paths.data_processed_text / run_id / creator_id / unique_key
            out_dir.mkdir(parents=True, exist_ok=True)
            raw_txt_path = out_dir / "raw.txt"
            clean_txt_path = out_dir / "clean.txt"
            qc_path = out_dir / "qc.json"
        else:
            raw_txt_path = paths.data_processed_text / creator_id / f"raw_{bvid}.txt"
            out_dir = paths.data_processed_text / creator_id
            out_dir.mkdir(parents=True, exist_ok=True)
            clean_txt_path = out_dir / f"clean_{bvid}.txt"
            qc_path = paths.data_processed_qc / f"{bvid}_qc.json"

        if subtitle_status != "OK" or (not raw_txt_path.exists()):
            if clean_txt_path.exists():
                clean_txt_path.unlink()
            if qc_path.exists():
                qc_path.unlink()
            continue

        if args.only_new and clean_txt_path.exists():
            continue

        raw_text = raw_txt_path.read_text(encoding="utf-8")
        cleaned, removed_count = clean_text(raw_text, onomatopoeia_terms=onomatopoeia)

        clean_txt_path.write_text(cleaned, encoding="utf-8")

        raw_chars = count_chars(raw_text)
        clean_chars = count_chars(cleaned)
        raw_tokens = count_tokens(raw_text)
        clean_tokens = count_tokens(cleaned)
        raw_sentences = len(split_sentences(raw_text))
        clean_sentences = len(split_sentences(cleaned))

        warnings: list[str] = []
        if raw_chars > 0 and clean_chars / raw_chars < 0.65:
            warnings.append("clean_chars/raw_chars < 0.65（拟声词表可能过宽，或字幕异常）")

        qc = {
            "bvid": bvid,
            "unique_key": unique_key,
            "series": series,
            "creator_id": creator_id,
            "raw_chars": raw_chars,
            "clean_chars": clean_chars,
            "removed_onomatopoeia_count": removed_count,
            "raw_tokens": raw_tokens,
            "clean_tokens": clean_tokens,
            "raw_sentences": raw_sentences,
            "clean_sentences": clean_sentences,
            "warnings": warnings,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }

        qc_path.write_text(json.dumps(qc, ensure_ascii=False, indent=2), encoding="utf-8")
        new_clean += 1

    if run_id:
        if run_id:
            total_clean = len(list((paths.data_processed_text / run_id).rglob("clean.txt")))
        else:
            total_clean = len(list(paths.data_processed_text.rglob("clean_*.txt")))
        inc_path = run_dirs.logs / "run_log_incremental.txt"
        with inc_path.open("a", encoding="utf-8") as f:
            f.write(
                f"[{Path(manifest_path).name}] clean_text: new_clean={new_clean} total_clean={total_clean} "
                f"(only_new={args.only_new}, since_log={bool(args.since_log)})\n"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
