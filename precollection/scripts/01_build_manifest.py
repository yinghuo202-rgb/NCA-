from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from lib.bilibili import fetch_video_meta
from lib.breakpoints import compute_stage, load_breakpoints
from lib.http import requests_session
from lib.paths import ensure_dirs, get_project_paths


def _read_seed(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    required = {"series", "creator_id", "bvid", "expected_stage"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"seed 缺少字段: {sorted(missing)}")
    return df


def main() -> int:
    paths = get_project_paths()
    ensure_dirs(paths)

    seed_path = paths.config / "sample_manifest_seed.csv"
    breakpoints = load_breakpoints(paths.config / "breakpoints.json")

    df_seed = _read_seed(seed_path)
    sess = requests_session()

    rows: list[dict[str, object]] = []
    for _, r in tqdm(df_seed.iterrows(), total=len(df_seed), desc="Build manifest"):
        series = str(r["series"]).strip()
        creator_id = str(r["creator_id"]).strip()
        bvid = str(r["bvid"]).strip()
        expected_stage = str(r["expected_stage"]).strip()

        meta = None
        err = None
        for attempt in range(4):
            try:
                meta = fetch_video_meta(sess, bvid)
                break
            except Exception as e:  # noqa: BLE001 - 预收集脚本：记录并重试
                err = f"{type(e).__name__}: {e}"
                time.sleep(1.0 + attempt * 1.5)

        if meta is None:
            rows.append(
                {
                    "series": series,
                    "creator_id": creator_id,
                    "bvid": bvid,
                    "title": "",
                    "aid": "",
                    "cid": "",
                    "pages_count": "",
                    "page_index": "",
                    "part_name": "",
                    "page_duration": "",
                    "up_mid": "",
                    "up_name": "",
                    "pub_datetime": "",
                    "pub_date": "",
                    "duration_sec": "",
                    "expected_stage": expected_stage,
                    "actual_stage": "",
                    "stage_match": "",
                    "needs_review": True,
                    "subtitle_status": "",
                    "error": err or "UNKNOWN_ERROR",
                    "retrieved_at": datetime.now().isoformat(timespec="seconds"),
                }
            )
            continue

        pub_date = meta.pub_datetime.date()
        actual_stage = compute_stage(pub_date, breakpoints)
        stage_match = actual_stage == expected_stage

        rows.append(
            {
                "series": series,
                "creator_id": creator_id,
                "bvid": bvid,
                "title": meta.title,
                "aid": meta.aid,
                "cid": meta.cid,
                "pages_count": meta.pages_count,
                "page_index": meta.page_index,
                "part_name": meta.part_name,
                "page_duration": meta.page_duration,
                "up_mid": meta.up_mid if meta.up_mid is not None else "",
                "up_name": meta.up_name if meta.up_name is not None else "",
                "pub_datetime": meta.pub_datetime.isoformat(timespec="seconds"),
                "pub_date": pub_date.isoformat(),
                "duration_sec": meta.duration_sec,
                "expected_stage": expected_stage,
                "actual_stage": actual_stage,
                "stage_match": stage_match,
                "needs_review": (not stage_match),
                "subtitle_status": "",
                "error": "",
                "retrieved_at": datetime.now().isoformat(timespec="seconds"),
            }
        )

        time.sleep(0.3)

    out_path = paths.data_raw_meta / "videos_manifest.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
