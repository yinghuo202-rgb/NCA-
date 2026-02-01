from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from lib.http import get_json, load_cookies_json, requests_session
from lib.paths import ensure_dirs, get_project_paths
from lib.run_utils import init_run_dirs, make_run_id


def _append_run_log(path: Path, lines: list[str]) -> None:
    existing = ""
    if path.exists():
        try:
            existing = path.read_text(encoding="utf-8-sig")
        except Exception:
            existing = path.read_text(encoding="utf-8", errors="ignore")
    merged = (existing.rstrip("\n") + "\n" + "\n".join(lines)).strip() + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(merged, encoding="utf-8-sig")


def _log(lines: list[str], msg: str) -> None:
    stamp = datetime.now().isoformat(timespec="seconds")
    line = f"[{stamp}] {msg}"
    print(line)
    lines.append(line)


def _find_name_col(df: pd.DataFrame, override: str | int | None) -> str | int:
    if override is not None and override != "":
        return override
    candidates = ["creator_name", "name", "up_name", "uname", "title"]
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: first column
    return df.columns[0]


def _search_users(
    sess,
    keyword: str,
    *,
    page: int,
    page_size: int = 5,
    retries: int = 3,
    sleep_sec: float = 1.2,
) -> tuple[list[dict[str, Any]], str]:
    last_err = ""
    for attempt in range(max(retries, 1)):
        try:
            payload = get_json(
                sess,
                "https://api.bilibili.com/x/web-interface/search/type",
                params={
                    "search_type": "bili_user",
                    "keyword": keyword,
                    "page": page,
                    "page_size": page_size,
                },
                headers={"Referer": "https://www.bilibili.com/"},
            )
            data = payload.get("data") or {}
            return (data.get("result") or []), ""
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(sleep_sec + attempt * 1.5)
    return [], last_err


def _score_candidate(name: str, cand_name: str, fans: int) -> float:
    score = 0.0
    if cand_name == name:
        score += 0.6
    elif name in cand_name or cand_name in name:
        score += 0.4
    # followers boost (log scale)
    score += min(0.4, (fans ** 0.5) / 1000.0)
    return score


def _candidate_stats(cand: dict[str, Any]) -> str:
    fans = cand.get("fans", "")
    videos = cand.get("videos", "")
    level = cand.get("level", "")
    verify = cand.get("official_verify") or {}
    verify_desc = verify.get("desc") or ""
    return f"fans={fans}; videos={videos}; level={level}; verify={verify_desc}"


def build_review(
    *,
    input_path: Path,
    output_path: Path,
    name_col: str | int | None,
    group_col: str | int | None,
    notes_col: str | int | None,
    no_header: bool,
    topk: int,
    sleep_sec: float,
    retries: int,
    cookies_path: Path | None,
) -> None:
    df = pd.read_excel(input_path, header=None if no_header else 0)
    name_col = _find_name_col(df, name_col)
    group_col = group_col if group_col in df.columns else None
    notes_col = notes_col if notes_col in df.columns else None

    cookies = load_cookies_json(cookies_path) if cookies_path else {}
    sess = requests_session(cookies)

    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        name = str(r.get(name_col, "")).strip()
        if not name:
            continue

        candidates, err = _search_users(
            sess,
            name,
            page=1,
            page_size=max(topk, 5),
            retries=max(retries, 1),
            sleep_sec=max(sleep_sec, 0.8),
        )
        scored: list[tuple[float, dict[str, Any]]] = []
        for cand in candidates:
            cand_name = str(cand.get("uname") or "")
            fans = int(cand.get("fans") or 0)
            scored.append((_score_candidate(name, cand_name, fans), cand))
        scored.sort(key=lambda x: x[0], reverse=True)
        scored = scored[:topk]

        row: dict[str, Any] = {
            "creator_name": name,
            "creator_group": str(r.get(group_col, "")).strip() if group_col else "",
            "notes": str(r.get(notes_col, "")).strip() if notes_col else "",
        }

        for i, (score, cand) in enumerate(scored, start=1):
            row[f"candidate_mid_{i}"] = cand.get("mid", "")
            row[f"candidate_name_{i}"] = cand.get("uname", "")
            row[f"candidate_desc_{i}"] = cand.get("usign", "")
            row[f"candidate_stats_{i}"] = _candidate_stats(cand)
            row[f"candidate_score_{i}"] = round(score, 4)

        # status heuristic
        status = "UNRESOLVED"
        if err:
            status = "ERROR"
        if scored:
            top = scored[0][0]
            second = scored[1][0] if len(scored) > 1 else 0.0
            if top >= 0.85 and (top - second) >= 0.15:
                status = "RESOLVED"
            elif top >= 0.6:
                status = "AMBIGUOUS"
        row["confidence"] = f"top_score={scored[0][0]:.3f}" if scored else "no_candidate"
        row["manual_mid"] = ""
        row["status"] = status
        row["error"] = err
        rows.append(row)
        time.sleep(max(sleep_sec, 0.4))

    out_df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_excel(output_path, index=False)


def apply_review(
    *, review_path: Path, output_path: Path, duplicates_path: Path, stats_path: Path
) -> int:
    df = pd.read_excel(review_path, dtype=str).fillna("")
    if "manual_mid" not in df.columns:
        raise ValueError("review 表缺少 manual_mid 列")

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    # gather duplicates with source row ids
    grouped: dict[str, list[dict[str, Any]]] = {}
    manual_mid_empty = 0
    for _, r in df.iterrows():
        mid = str(r.get("manual_mid", "")).strip()
        if not mid:
            manual_mid_empty += 1
            continue
        source_row = str(r.name + 1)
        item = {
            "creator_id": mid,
            "creator_name": str(r.get("creator_name", "")).strip(),
            "creator_group": str(r.get("creator_group", "")).strip(),
            "notes": str(r.get("notes", "")).strip(),
            "source": "manual_mid=candidate_mid_1",
            "source_rows": source_row,
        }
        grouped.setdefault(mid, []).append(item)

    duplicates: list[dict[str, Any]] = []
    for mid, items in grouped.items():
        # keep first
        first = items[0]
        if mid not in seen:
            seen.add(mid)
            rows.append(first)
        if len(items) > 1:
            duplicates.append(
                {
                    "creator_id": mid,
                    "creator_names": ";".join([it["creator_name"] for it in items if it["creator_name"]]),
                    "creator_groups": ";".join([it["creator_group"] for it in items if it["creator_group"]]),
                    "source_rows": ";".join([it["source_rows"] for it in items]),
                    "count": len(items),
                }
            )

    out_df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    dup_df = pd.DataFrame(duplicates)
    duplicates_path.parent.mkdir(parents=True, exist_ok=True)
    dup_df.to_csv(duplicates_path, index=False, encoding="utf-8-sig")

    # stats
    stats = {
        "total_rows": int(len(df)),
        "manual_mid_empty": int(manual_mid_empty),
        "dedup_count": int(len(out_df)),
        "duplicate_groups": int(len(duplicates)),
        "duplicate_top": sorted(
            [{"creator_id": d.get("creator_id"), "count": int(d.get("count", 0))} for d in duplicates],
            key=lambda x: x["count"],
            reverse=True,
        )[:10],
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8-sig")

    return len(out_df)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="inputs/creators.xlsx", help="input excel path")
    parser.add_argument("--output", default="outputs/creators_id_review.xlsx", help="output excel path")
    parser.add_argument("--name-col", default="", help="creator name column")
    parser.add_argument("--group-col", default="", help="creator group column")
    parser.add_argument("--notes-col", default="", help="notes column")
    parser.add_argument("--no-header", action="store_true", help="treat excel as no-header and use column indices")
    parser.add_argument("--name-col-index", type=int, default=-1, help="creator name column index (0-based)")
    parser.add_argument("--group-col-index", type=int, default=-1, help="creator group column index (0-based)")
    parser.add_argument("--notes-col-index", type=int, default=-1, help="notes column index (0-based)")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--sleep", type=float, default=0.8)
    parser.add_argument("--retries", type=int, default=1, help="search retries per name")
    parser.add_argument("--cookies", default="secrets/cookies.json", help="cookies json path (optional)")
    parser.add_argument(
        "--apply-review",
        action="store_true",
        help="use outputs/creators_id_review.xlsx with manual_mid to build data/creators/creators.csv",
    )
    parser.add_argument("--review", default="outputs/creators_id_review.xlsx")
    parser.add_argument("--creators-csv", default="data/creators/creators.csv")
    parser.add_argument("--run-id", default="", help="run id for archiving under runs/<run_id>")
    args = parser.parse_args()

    paths = get_project_paths()
    ensure_dirs(paths)

    log_lines: list[str] = []
    _log(log_lines, "Start 00_resolve_creator_ids")

    run_id = make_run_id(args.run_id or None)
    run_dirs = init_run_dirs(paths.root, run_id)

    if args.apply_review:
        review_path = (paths.root / args.review).resolve()
        creators_csv = (paths.root / args.creators_csv).resolve()
        duplicates_csv = paths.data_creators / "creators_duplicates.csv"
        stats_json = paths.data_creators / "creators_stats.json"
        count = apply_review(
            review_path=review_path,
            output_path=creators_csv,
            duplicates_path=duplicates_csv,
            stats_path=stats_json,
        )
        _log(log_lines, f"Wrote creators.csv: {creators_csv}")
        _log(log_lines, f"Wrote creators_stats.json: {stats_json}")
        _log(log_lines, f"Creators count: {count}")
        # snapshot
        try:
            import shutil

            shutil.copy2(creators_csv, run_dirs.data_snapshots / "creators.csv")
            if duplicates_csv.exists():
                shutil.copy2(duplicates_csv, run_dirs.data_snapshots / "creators_duplicates.csv")
        except Exception as e:  # noqa: BLE001
            _log(log_lines, f"WARN: snapshot copy failed: {type(e).__name__}: {e}")
        _append_run_log(paths.outputs / "run_log.txt", log_lines)
        _append_run_log(run_dirs.logs / "run_log_creators.txt", log_lines)
        return 0

    input_path = (paths.root / args.input).resolve()
    output_path = (paths.root / args.output).resolve()
    cookies_path = (paths.root / args.cookies).resolve()
    if not cookies_path.exists():
        cookies_path = None

    if args.no_header:
        name_col = args.name_col_index if args.name_col_index >= 0 else None
        group_col = args.group_col_index if args.group_col_index >= 0 else None
        notes_col = args.notes_col_index if args.notes_col_index >= 0 else None
        _log(
            log_lines,
            f"Detected no-header file; using column indices name={args.name_col_index} group={args.group_col_index} notes={args.notes_col_index}",
        )
    else:
        name_col = args.name_col or None
        group_col = args.group_col or None
        notes_col = args.notes_col or None

    _log(log_lines, f"Using name_col={name_col or 'auto'} group_col={group_col or ''} notes_col={notes_col or ''}")
    build_review(
        input_path=input_path,
        output_path=output_path,
        name_col=name_col,
        group_col=group_col,
        notes_col=notes_col,
        no_header=args.no_header,
        topk=max(args.topk, 1),
        sleep_sec=max(args.sleep, 0.2),
        retries=max(args.retries, 1),
        cookies_path=cookies_path,
    )
    _log(log_lines, f"Wrote creators_id_review.xlsx: {output_path}")
    # archive to run outputs
    try:
        import shutil

        shutil.copy2(output_path, run_dirs.outputs / "creators_id_review.xlsx")
    except Exception as e:  # noqa: BLE001
        _log(log_lines, f"WARN: archive copy failed: {type(e).__name__}: {e}")
    _append_run_log(paths.outputs / "run_log.txt", log_lines)
    _append_run_log(run_dirs.logs / "run_log_creators.txt", log_lines)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
