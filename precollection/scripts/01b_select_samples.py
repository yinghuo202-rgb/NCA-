from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lib.breakpoints_ext import BreakpointsConfig, load_breakpoints_yaml
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


_DELIMS = ["｜", "|", "—", "-", "–", ":", "：", " "]


def _extract_title_pattern(title: str) -> str:
    t = (title or "").strip()
    if not t:
        return ""
    m = re.match(r"^[【\\[]([^】\\]]{2,20})[】\\]]", t)
    if m:
        return m.group(1).strip()
    for d in _DELIMS:
        if d in t:
            t = t.split(d, 1)[0].strip()
            break
    t = re.sub(r"\\d+$", "", t).strip()
    if len(t) < 2 or len(t) > 20:
        return ""
    return t


def _even_sample(df: pd.DataFrame, *, n: int) -> pd.DataFrame:
    if len(df) <= n:
        return df
    idxs = [int(round(x)) for x in list(np.linspace(0, len(df) - 1, n))]
    idxs = sorted(set(min(max(i, 0), len(df) - 1) for i in idxs))
    return df.iloc[idxs]


def _pick_style_group_rule(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]] | None:
    if df.empty:
        return None
    work = df.copy()
    work["title_pattern"] = work["title"].map(_extract_title_pattern)
    work = work[work["title_pattern"].astype(str).str.strip() != ""]
    if work.empty:
        return None

    group_cols = ["tname", "title_pattern"]
    grp = work.groupby(group_cols + ["phase_base"]).size().reset_index(name="cnt")
    if grp.empty:
        return None

    summary = grp.groupby(group_cols)["cnt"].agg(["min", "sum"]).reset_index()
    summary = summary.sort_values(["min", "sum"], ascending=[False, False])
    best = summary.iloc[0]
    tname = best["tname"]
    pattern = best["title_pattern"]
    subset = work[(work["tname"] == tname) & (work["title_pattern"] == pattern)].copy()
    meta = {
        "selection_reason": "rule",
        "tname": tname,
        "title_pattern": pattern,
        "min_count": int(best["min"]),
        "sum_count": int(best["sum"]),
    }
    return subset, meta


def _pick_style_group_cluster(df: pd.DataFrame, log_lines: list[str] | None = None) -> tuple[pd.DataFrame, dict[str, Any]] | None:
    if df.empty:
        return None
    try:
        from sklearn.cluster import KMeans  # type: ignore
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    except Exception:
        if log_lines is not None:
            _log(log_lines, "WARN: sklearn not available; skip clustering fallback")
        return None

    corpus = (
        df["title"].fillna("").astype(str)
        + " "
        + df["desc"].fillna("").astype(str)
        + " "
        + df["tname"].fillna("").astype(str)
    ).tolist()

    if len(corpus) < 5:
        return None

    max_features = 3000
    vec = TfidfVectorizer(max_features=max_features, min_df=2)
    X = vec.fit_transform(corpus)
    k = max(2, min(8, int(math.sqrt(len(corpus)))))

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    work = df.copy()
    work["cluster_id"] = labels

    grp = work.groupby(["cluster_id", "phase_base"]).size().reset_index(name="cnt")
    summary = grp.groupby("cluster_id")["cnt"].agg(["min", "sum"]).reset_index()
    summary = summary.sort_values(["min", "sum"], ascending=[False, False])
    if summary.empty:
        return None
    best = summary.iloc[0]
    cluster_id = int(best["cluster_id"])
    subset = work[work["cluster_id"] == cluster_id].copy()
    meta = {
        "selection_reason": "cluster",
        "cluster_id": cluster_id,
        "min_count": int(best["min"]),
        "sum_count": int(best["sum"]),
    }
    return subset, meta


def _filter_candidates(
    df: pd.DataFrame,
    *,
    duration_min: int,
    duration_max: int,
    exclude_keywords: list[str],
    allow_buffer: bool,
) -> pd.DataFrame:
    work = df.copy()
    work["duration"] = pd.to_numeric(work["duration"], errors="coerce").fillna(0).astype(int)

    if not allow_buffer and "in_buffer_base" in work.columns:
        work = work[~work["in_buffer_base"].astype(str).isin(["True", "true", "1"])]

    work = work[(work["duration"] >= duration_min) & (work["duration"] <= duration_max)]
    if exclude_keywords:
        pattern = "|".join([re.escape(x) for x in exclude_keywords if x])
        if pattern:
            work = work[~work["title"].fillna("").astype(str).str.contains(pattern)]

    work = work[work["phase_base"].isin(["S0", "S1", "S2"])]
    return work


def _make_unique_key(row: pd.Series) -> str:
    bvid = str(row.get("bvid", "")).strip()
    if not bvid:
        return ""
    cid = str(row.get("cid", "")).strip()
    if cid:
        return f"{bvid}_cid{cid}"
    part = str(row.get("part", "")).strip()
    if part:
        return f"{bvid}_part{part}"
    page = str(row.get("page", "")).strip()
    if page:
        return f"{bvid}_p{page}"
    return bvid


def _estimate_asr_ratio(paths) -> float:
    # Use the latest run manifest with subtitle_source if available.
    runs_dir = paths.runs
    if not runs_dir.exists():
        return 1.0
    manifests = sorted(runs_dir.glob("*/outputs/final_manifest.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in manifests:
        try:
            df = pd.read_csv(p, dtype=str).fillna("")
        except Exception:
            continue
        if "subtitle_source" not in df.columns:
            continue
        total = len(df)
        if total == 0:
            continue
        asr_n = int(df["subtitle_source"].astype(str).str.startswith("asr").sum())
        return min(1.0, max(0.0, asr_n / float(total)))
    return 1.0


def _build_row_base(row: pd.Series, *, selection_reason: str, title_pattern: str, cluster_id: str) -> dict[str, Any]:
    return {
        "creator_id": row.get("creator_id", ""),
        "creator_name": row.get("creator_name", ""),
        "creator_group": row.get("creator_group", ""),
        "phase_base": row.get("phase_base", ""),
        "bvid": row.get("bvid", ""),
        "unique_key": _make_unique_key(row),
        "pubdate": row.get("pubdate", ""),
        "title": row.get("title", ""),
        "duration": row.get("duration", ""),
        "tname": row.get("tname", ""),
        "selection_reason": selection_reason,
        "title_pattern": title_pattern,
        "cluster_id": cluster_id,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="data/index/videos_index.csv")
    parser.add_argument("--breakpoints", default="config/breakpoints.yaml")
    parser.add_argument("--output-review", default="outputs/selection_review.xlsx")
    parser.add_argument("--selected-manifest", default="outputs/selected_manifest.csv")
    parser.add_argument("--final-manifest", default="outputs/final_manifest.csv")
    parser.add_argument("--n-per-phase", type=int, default=10)
    parser.add_argument("--run-id", default="", help="run id for archiving under runs/<run_id>")
    args = parser.parse_args()

    paths = get_project_paths()
    ensure_dirs(paths)

    log_lines: list[str] = []
    run_id = make_run_id(args.run_id or None)
    run_dirs = init_run_dirs(paths.root, run_id)
    _log(log_lines, f"Start 01b_select_samples run_id={run_id}")

    index_path = (paths.root / args.index).resolve()
    breakpoints_path = (paths.root / args.breakpoints).resolve()
    review_path = (paths.root / args.output_review).resolve()
    selected_path = (paths.root / args.selected_manifest).resolve()
    final_path = (paths.root / args.final_manifest).resolve()

    cfg = load_breakpoints_yaml(breakpoints_path)
    _log(
        log_lines,
        f"Params: duration_min_sec={cfg.duration_min_sec} duration_max_sec={cfg.duration_max_sec} exclude_title_keywords={cfg.exclude_title_keywords}",
    )

    df = pd.read_csv(index_path, dtype=str).fillna("")
    required = {
        "creator_id",
        "creator_name",
        "creator_group",
        "bvid",
        "pubdate",
        "title",
        "tname",
        "duration",
        "phase_base",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"videos_index 缺少字段: {sorted(missing)}")

    # Parse pubdate for sorting
    df["pubdate_dt"] = pd.to_datetime(df["pubdate"], errors="coerce")

    # strict filter config
    strict_cfg = {
        "duration_min": int(cfg.duration_min_sec),
        "duration_max": int(cfg.duration_max_sec),
        "exclude_keywords": list(cfg.exclude_title_keywords),
        "allow_buffer": False,
    }
    # fill level configs
    fill1_cfg = {
        "duration_min": 180,
        "duration_max": 2700,
        "exclude_keywords": list(cfg.exclude_title_keywords),
        "allow_buffer": False,
    }
    fill2_cfg = {
        "duration_min": 180,
        "duration_max": 2700,
        "exclude_keywords": list(cfg.exclude_title_keywords),
        "allow_buffer": True,
    }
    strong_exclude = [k for k in cfg.exclude_title_keywords if any(x in k for x in ["直播", "回放"])]
    if not strong_exclude:
        strong_exclude = ["直播", "回放"]
    fill3_cfg = {
        "duration_min": 180,
        "duration_max": 2700,
        "exclude_keywords": strong_exclude,
        "allow_buffer": True,
    }

    all_selected_rows: list[dict[str, Any]] = []
    creators_summary: list[dict[str, Any]] = []

    for creator_id, g in df.groupby("creator_id"):
        creator_df = g.copy()
        creator_df = creator_df.sort_values("pubdate_dt")
        base_phase_counts = {p: int((creator_df["phase_base"] == p).sum()) for p in ["S0", "S1", "S2"]}

        strict_filtered = _filter_candidates(creator_df, **strict_cfg)
        style_meta = {"selection_reason": "fallback", "title_pattern": "", "cluster_id": ""}
        style_df = strict_filtered.copy()

        rule_pick = _pick_style_group_rule(strict_filtered)
        if rule_pick:
            style_df, meta = rule_pick
            style_meta.update(meta)
        else:
            cluster_pick = _pick_style_group_cluster(strict_filtered, log_lines=log_lines)
            if cluster_pick:
                style_df, meta = cluster_pick
                style_meta.update(meta)

        creator_summary: dict[str, Any] = {
            "creator_id": creator_df["creator_id"].iloc[0],
            "creator_name": creator_df["creator_name"].iloc[0],
            "creator_group": creator_df["creator_group"].iloc[0],
            "style_reason": style_meta.get("selection_reason", ""),
            "style_tname": style_meta.get("tname", ""),
            "style_title_pattern": style_meta.get("title_pattern", ""),
            "style_cluster_id": style_meta.get("cluster_id", ""),
        }

        selected_rows_creator: list[dict[str, Any]] = []

        for phase in ["S0", "S1", "S2"]:
            phase_base = creator_df[creator_df["phase_base"] == phase].copy()
            candidates_total = len(phase_base)

            strict_phase = strict_filtered[strict_filtered["phase_base"] == phase].copy()
            strict_remaining = len(strict_phase)

            # style preference if not too narrow
            style_phase = style_df[style_df["phase_base"] == phase].copy()
            if len(style_phase) < max(3, min(10, strict_remaining)):
                style_phase = strict_phase.copy()

            # strict selection
            selected = _even_sample(style_phase, n=args.n_per_phase)
            selected_keys = set(selected["bvid"].astype(str).tolist())

            fill_reason = ""
            if candidates_total < args.n_per_phase:
                fill_reason = "index_shortage"
            elif strict_remaining < args.n_per_phase:
                fill_reason = "filter_too_strict"
            else:
                fill_reason = "phase_shortage"

            for _, r in selected.iterrows():
                row = _build_row_base(
                    r,
                    selection_reason=style_meta.get("selection_reason", "fallback"),
                    title_pattern=style_meta.get("title_pattern", ""),
                    cluster_id=str(style_meta.get("cluster_id", "")),
                )
                row.update(
                    {
                        "auto_selected": 1,
                        "manual_keep": "",
                        "notes": "",
                        "strict_ok": 1,
                        "fill_level": 0,
                        "fill_reason": "",
                    }
                )
                selected_rows_creator.append(row)

            missing_n = args.n_per_phase - len(selected)
            if missing_n > 0:
                pools = [
                    (_filter_candidates(phase_base, **fill1_cfg), 1),
                    (_filter_candidates(phase_base, **fill2_cfg), 2),
                    (_filter_candidates(phase_base, **fill3_cfg), 3),
                ]
                for pool_df, level in pools:
                    pool_df = pool_df[~pool_df["bvid"].astype(str).isin(selected_keys)].copy()
                    if pool_df.empty:
                        continue
                    sampled = _even_sample(pool_df, n=missing_n)
                    for _, r in sampled.iterrows():
                        row = _build_row_base(
                            r,
                            selection_reason=style_meta.get("selection_reason", "fallback"),
                            title_pattern=style_meta.get("title_pattern", ""),
                            cluster_id=str(style_meta.get("cluster_id", "")),
                        )
                        row.update(
                            {
                                "auto_selected": 1,
                                "manual_keep": "",
                                "notes": "",
                                "strict_ok": 0,
                                "fill_level": level,
                                "fill_reason": fill_reason,
                            }
                        )
                        selected_rows_creator.append(row)
                        selected_keys.add(str(r.get("bvid", "")))
                    missing_n = args.n_per_phase - len([r for r in selected_rows_creator if r.get("phase_base") == phase])
                    if missing_n <= 0:
                        break

            selected_n = len([r for r in selected_rows_creator if r.get("phase_base") == phase])
            gap = max(args.n_per_phase - selected_n, 0)
            creator_summary[f"{phase}_candidates_total"] = candidates_total
            creator_summary[f"{phase}_strict_remaining"] = strict_remaining
            creator_summary[f"{phase}_selected_n"] = selected_n
            creator_summary[f"{phase}_gap"] = gap

        # fill level 4: phase softening across phases to reach total 30
        total_selected = len(selected_rows_creator)
        total_target = args.n_per_phase * 3
        if total_selected < total_target:
            pool4 = _filter_candidates(creator_df, **fill3_cfg)
            selected_keys = {r.get("bvid") for r in selected_rows_creator}
            pool4 = pool4[~pool4["bvid"].astype(str).isin(selected_keys)].copy()
            pool4 = pool4.sort_values("pubdate_dt")
            needed = total_target - total_selected
            sampled = _even_sample(pool4, n=needed)
            for _, r in sampled.iterrows():
                row = _build_row_base(
                    r,
                    selection_reason=style_meta.get("selection_reason", "fallback"),
                    title_pattern=style_meta.get("title_pattern", ""),
                    cluster_id=str(style_meta.get("cluster_id", "")),
                )
                row.update(
                    {
                        "auto_selected": 1,
                        "manual_keep": "",
                        "notes": "",
                        "strict_ok": 0,
                        "fill_level": 4,
                        "fill_reason": f"phase_soften_from_{r.get('phase_base', '')}",
                    }
                )
                selected_rows_creator.append(row)

        creators_summary.append(creator_summary)
        all_selected_rows.extend(selected_rows_creator)

    selected_df = pd.DataFrame(all_selected_rows)
    if selected_df.empty:
        raise ValueError("未选出任何样本（selected_df为空）")

    # Ensure unique_key
    if "unique_key" not in selected_df.columns:
        selected_df["unique_key"] = selected_df.apply(_make_unique_key, axis=1)
    dup_mask = selected_df["unique_key"].astype(str).duplicated(keep="first")
    dup_count = int(dup_mask.sum())
    if dup_count > 0:
        _log(log_lines, f"WARN: duplicate unique_key detected: {dup_count} (dropping duplicates)")
        selected_df = selected_df[~dup_mask].copy()

    # final manifest equals selected rows
    selected_manifest = selected_df.copy()
    selected_manifest = selected_manifest.sort_values(["creator_id", "phase_base", "pubdate"])

    # summary sheet
    summary_df = pd.DataFrame(creators_summary)
    fill_dist = selected_manifest["fill_level"].value_counts().sort_index().to_dict()
    strict_ratio = float((selected_manifest["strict_ok"] == 1).sum()) / float(len(selected_manifest))
    est_asr_ratio = _estimate_asr_ratio(paths)
    est_asr_n = int(round(len(selected_manifest) * est_asr_ratio))
    summary_rows = [
        {"metric": "total_selected", "value": int(len(selected_manifest))},
        {"metric": "strict_ok_ratio", "value": round(strict_ratio, 4)},
        {"metric": "fill_level_distribution", "value": json.dumps(fill_dist, ensure_ascii=False)},
        {"metric": "estimated_asr_ratio", "value": round(est_asr_ratio, 4)},
        {"metric": "estimated_asr_n", "value": est_asr_n},
        {"metric": "duplicate_unique_key_dropped", "value": dup_count},
    ]
    summary_sheet = pd.DataFrame(summary_rows)

    review_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(review_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="creators", index=False)
        selected_manifest.to_excel(writer, sheet_name="selected_videos", index=False)
        summary_sheet.to_excel(writer, sheet_name="summary", index=False)

    selected_manifest.to_csv(selected_path, index=False, encoding="utf-8-sig")
    selected_manifest.to_csv(final_path, index=False, encoding="utf-8-sig")

    _log(log_lines, f"Wrote selection_review.xlsx: {review_path}")
    _log(log_lines, f"Wrote selected_manifest: {selected_path}")
    _log(log_lines, f"Wrote final_manifest: {final_path}")
    _append_run_log(paths.outputs / "run_log.txt", log_lines)
    _append_run_log(run_dirs.logs / "run_log_select.txt", log_lines)

    try:
        import shutil

        shutil.copy2(review_path, run_dirs.outputs / "selection_review.xlsx")
        shutil.copy2(selected_path, run_dirs.outputs / "selected_manifest.csv")
        shutil.copy2(final_path, run_dirs.outputs / "final_manifest.csv")
    except Exception as e:  # noqa: BLE001
        _log(log_lines, f"WARN: archive copy failed: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
