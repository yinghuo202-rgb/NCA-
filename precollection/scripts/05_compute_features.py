from __future__ import annotations

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from lib.lexicon import load_terms
from lib.paths import ensure_dirs, get_project_paths
from lib.text_processing import (
    count_chars,
    count_punct,
    count_tokens,
    split_sentences,
    substring_hits,
)


def _read_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"未找到 manifest：{path}（先运行 scripts/01_build_manifest.py）")
    return pd.read_csv(path, dtype=str).fillna("")


def _per_1k(value: int, tokens: int) -> float:
    if tokens <= 0:
        return 0.0
    return value / tokens * 1000.0


def main() -> int:
    paths = get_project_paths()
    ensure_dirs(paths)

    manifest_path = paths.data_raw_meta / "videos_manifest.csv"
    df = _read_manifest(manifest_path)

    connectors = load_terms(paths.resources / "connectors.txt")
    frame_markers = load_terms(paths.resources / "frame_markers.txt")
    modality = load_terms(paths.resources / "modality.txt")

    rows: list[dict[str, object]] = []
    out_columns = [
        "series",
        "creator_id",
        "bvid",
        "pub_date",
        "actual_stage",
        "duration_sec",
        "chars",
        "tokens",
        "sentences",
        "mean_sentence_chars",
        "punct_per_1k_tokens",
        "connectors_per_1k_tokens",
        "frame_markers_per_1k_tokens",
        "modality_per_1k_tokens",
    ]
    for _, r in tqdm(df.iterrows(), total=len(df), desc="Compute features"):
        bvid = str(r.get("bvid", "")).strip()
        creator_id = str(r.get("creator_id", "")).strip()
        series = str(r.get("series", "")).strip()
        pub_date = str(r.get("pub_date", "")).strip()
        actual_stage = str(r.get("actual_stage", "")).strip()
        duration_sec = str(r.get("duration_sec", "")).strip()
        subtitle_status = str(r.get("subtitle_status", "")).strip()

        if not bvid or not creator_id:
            continue
        if subtitle_status != "OK":
            continue

        clean_path = paths.data_processed_text / creator_id / f"clean_{bvid}.txt"
        if not clean_path.exists():
            continue

        text = clean_path.read_text(encoding="utf-8")
        chars = count_chars(text)
        tokens = count_tokens(text)
        sentences = len(split_sentences(text))
        mean_sentence_chars = (chars / sentences) if sentences > 0 else 0.0

        punct = count_punct(text)
        connectors_hit = substring_hits(text, connectors)
        frame_hit = substring_hits(text, frame_markers)
        modality_hit = substring_hits(text, modality)

        rows.append(
            {
                "series": series,
                "creator_id": creator_id,
                "bvid": bvid,
                "pub_date": pub_date,
                "actual_stage": actual_stage,
                "duration_sec": int(duration_sec) if duration_sec.isdigit() else "",
                "chars": chars,
                "tokens": tokens,
                "sentences": sentences,
                "mean_sentence_chars": round(mean_sentence_chars, 3),
                "punct_per_1k_tokens": round(_per_1k(punct, tokens), 3),
                "connectors_per_1k_tokens": round(_per_1k(connectors_hit, tokens), 3),
                "frame_markers_per_1k_tokens": round(_per_1k(frame_hit, tokens), 3),
                "modality_per_1k_tokens": round(_per_1k(modality_hit, tokens), 3),
            }
        )

    out_path = paths.analysis / "features.csv"
    pd.DataFrame(rows, columns=out_columns).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
