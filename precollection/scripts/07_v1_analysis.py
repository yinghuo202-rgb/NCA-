from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from lib.lexicon import load_terms
from lib.paths import ensure_dirs, get_project_paths
from lib.run_utils import init_run_dirs, make_run_id
from lib.text_processing import count_chars, normalize_punctuation, split_sentences, substring_hits


_RE_HAS_CJK = re.compile(r"[\u4e00-\u9fff]")
_RE_HAS_ALNUM = re.compile(r"[A-Za-z0-9]")


@dataclass(frozen=True)
class MappingRow:
    series: str
    creator_group: str
    bvid: str
    phase: str
    title: str


def _append_log(lines: list[str], message: str) -> None:
    stamp = datetime.now().isoformat(timespec="seconds")
    line = f"[{stamp}] {message}"
    print(line)
    lines.append(line)


def _read_phase_mapping(path: Path) -> dict[str, MappingRow]:
    df = pd.read_csv(path, dtype=str).fillna("")
    required = ["series", "creator_group", "bvid", "phase", "title"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"phase mapping missing columns: {missing} ({path})")

    df["bvid"] = df["bvid"].astype(str).str.strip()
    df["phase"] = df["phase"].astype(str).str.strip()
    df["creator_group"] = df["creator_group"].astype(str).str.strip()
    df["series"] = df["series"].astype(str).str.strip()
    df["title"] = df["title"].astype(str).str.strip()

    bad_phase = df[~df["phase"].isin(["S0", "S1", "S2"])]
    if not bad_phase.empty:
        bad = ", ".join(sorted(set(bad_phase["phase"].tolist())))
        raise ValueError(f"phase mapping has invalid phase values: {bad}")

    if df["bvid"].duplicated().any():
        dup = df[df["bvid"].duplicated(keep=False)]["bvid"].tolist()
        raise ValueError(f"phase mapping has duplicate bvid: {dup}")

    out: dict[str, MappingRow] = {}
    for _, r in df.iterrows():
        bvid = str(r["bvid"]).strip()
        if not bvid:
            continue
        out[bvid] = MappingRow(
            series=str(r["series"]).strip(),
            creator_group=str(r["creator_group"]).strip(),
            bvid=bvid,
            phase=str(r["phase"]).strip(),
            title=str(r["title"]).strip(),
        )
    return out


def _load_bvids_from_log(path: Path) -> set[str]:
    if not path or not path.exists():
        return set()
    bvids: set[str] = set()
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        status = str(obj.get("status", "")).strip()
        if status != "OK":
            continue
        bvid = str(obj.get("bvid", "")).strip()
        if bvid:
            bvids.add(bvid)
    return bvids


def _load_text_for_analysis(path: Path) -> str:
    raw = path.read_text(encoding="utf-8-sig")
    raw = raw.replace("\ufeff", "")
    raw = normalize_punctuation(raw)
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.strip() for ln in raw.split("\n")]
    lines = [ln for ln in lines if ln]
    merged = " ".join(lines)
    merged = re.sub(r"\\s+", " ", merged).strip()
    return merged


def _tokenize(text: str) -> list[str]:
    try:
        import jieba  # type: ignore

        try:
            import logging

            jieba.setLogLevel(logging.ERROR)
        except Exception:
            pass

        toks = jieba.lcut(text, cut_all=False)
    except Exception:
        # Fallback: character-level for CJK + ASCII alnum groups
        toks = re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9]+", text)

    out: list[str] = []
    for t in toks:
        s = str(t).strip()
        if not s:
            continue
        if not (_RE_HAS_CJK.search(s) or _RE_HAS_ALNUM.search(s)):
            continue
        out.append(s)
    return out


def _mattr(tokens: list[str], *, window: int = 100) -> float:
    n = len(tokens)
    if n == 0:
        return 0.0
    if n <= window:
        return len(set(tokens)) / float(n)

    # Sliding window with incremental counts (O(n)).
    counts: dict[str, int] = {}
    unique = 0
    for t in tokens[:window]:
        prev = counts.get(t, 0)
        counts[t] = prev + 1
        if prev == 0:
            unique += 1

    ttr_sum = unique / float(window)
    windows = 1
    for i in range(window, n):
        out_t = tokens[i - window]
        counts[out_t] -= 1
        if counts[out_t] == 0:
            del counts[out_t]
            unique -= 1

        in_t = tokens[i]
        prev = counts.get(in_t, 0)
        counts[in_t] = prev + 1
        if prev == 0:
            unique += 1

        ttr_sum += unique / float(window)
        windows += 1

    return ttr_sum / float(windows)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values)) / float(len(values))


def _plot_phase_boxplot(df: pd.DataFrame, *, metric: str, out_path: Path, title: str) -> None:
    order = ["S0", "S1", "S2"]
    if df.empty or metric not in df.columns:
        return

    tmp = df.copy()
    tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce")
    tmp = tmp[tmp["phase"].isin(order)]
    tmp = tmp.dropna(subset=[metric])
    if tmp.empty:
        return

    data = [tmp.loc[tmp["phase"] == p, metric].tolist() for p in order]
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
    ax.boxplot(data, tick_labels=order, showmeans=True)
    ax.set_title(title)
    ax.set_xlabel("Phase")
    ax.set_ylabel(metric)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="v1 analysis: clean text -> master.csv/features.csv/plots")
    parser.add_argument("--manifest", default="outputs/final_manifest.csv", help="manifest csv path (preferred)")
    parser.add_argument("--phase-col", default="phase_base", help="phase column in manifest (e.g., phase_base)")
    parser.add_argument("--phase-shift", type=int, default=0, help="override phase column using phase_shift_{+/-N}")
    parser.add_argument("--mapping", default="config/phase_mapping_v1.csv", help="legacy phase mapping csv path (fallback)")
    parser.add_argument("--run-id", default="", help="run id for archiving under runs/<run_id>")
    parser.add_argument("--input-dir", default="data/processed/text", help="directory containing Creator_F/M clean_*.txt")
    parser.add_argument("--out-dir", default="outputs", help="output directory")
    parser.add_argument("--mattr-window", type=int, default=100, help="MATTR window size")
    parser.add_argument("--no-plots", action="store_true", help="skip plot generation")
    parser.add_argument("--only-new", action="store_true", help="仅追加新样本（已存在则跳过）")
    parser.add_argument("--since-log", default="", help="仅处理 download.jsonl 中 status=OK 的 bvid")
    args = parser.parse_args()

    paths = get_project_paths()
    ensure_dirs(paths)

    run_id = make_run_id(args.run_id or None)
    run_dirs = init_run_dirs(paths.root, run_id)

    out_dir = (paths.root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    run_log: list[str] = []
    _append_log(run_log, "Start v1 analysis pipeline")
    _append_log(run_log, "Self-check (reuse/new)")
    _append_log(run_log, "REUSE: scripts/lib/text_processing.py (punctuation normalize, sentence split, counts)")
    _append_log(run_log, "REUSE: scripts/lib/lexicon.py (load_terms)")
    _append_log(run_log, "REFERENCE: scripts/06_generate_report.py (matplotlib style patterns)")
    _append_log(run_log, "NEW: scripts/07_v1_analysis.py (master/features/plots + run_log)")
    _append_log(run_log, "NEW: config/phase_mapping_v1.csv (fixed BV→phase mapping)")
    _append_log(run_log, "NEW: resources/stopwords_zh.txt + resources/function_words_zh.txt (v1 lexicons)")
    _append_log(run_log, "NEW: resources/templates_v1.txt + resources/connectives_*.txt (v1 lexicons)")

    manifest_path = (paths.root / args.manifest).resolve()
    phase_map: dict[str, MappingRow] = {}

    phase_col = args.phase_col
    if args.phase_shift != 0:
        phase_col = f"phase_shift_{args.phase_shift:+d}"

    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path, dtype=str).fillna("")
        if phase_col not in manifest.columns:
            raise ValueError(f"manifest 缺少 phase 列: {phase_col}")
        for _, r in manifest.iterrows():
            vid = str(r.get("unique_key") or r.get("bvid") or r.get("video_id") or "").strip()
            if not vid:
                continue
            phase = str(r.get(phase_col) or "").strip()
            if phase not in {"S0", "S1", "S2"}:
                continue
            phase_map[vid] = MappingRow(
                series=str(r.get("series") or "").strip(),
                creator_group=str(r.get("creator_group") or r.get("creator_id") or "").strip(),
                bvid=vid,
                phase=phase,
                title=str(r.get("title") or "").strip(),
            )
        _append_log(run_log, f"Loaded manifest: {manifest_path} (rows={len(phase_map)}) phase_col={phase_col}")
    else:
        mapping_path = (paths.root / args.mapping).resolve()
        phase_map = _read_phase_mapping(mapping_path)
        _append_log(run_log, f"Loaded phase mapping: {mapping_path} (rows={len(phase_map)})")

    input_dir = (paths.root / args.input_dir).resolve()
    clean_paths = sorted(input_dir.rglob("clean*.txt"))
    _append_log(run_log, f"Discovered clean files: {input_dir} (n={len(clean_paths)})")

    bvid_to_path: dict[str, Path] = {}
    for p in clean_paths:
        bvid = ""
        if p.name == "clean.txt":
            bvid = p.parent.name
        elif p.stem.startswith("clean_"):
            bvid = p.stem[len("clean_") :]
        if not bvid:
            continue
        if bvid in bvid_to_path:
            _append_log(run_log, f"WARN: duplicate clean file for {bvid}: {bvid_to_path[bvid]} and {p} (keeping first)")
            continue
        bvid_to_path[bvid] = p

    # Log extra clean files (not in mapping)
    extra_bvids = sorted([b for b in bvid_to_path.keys() if b not in phase_map])
    for b in extra_bvids:
        _append_log(run_log, f"MISSING_PHASE: {b} (skip) path={bvid_to_path[b]}")

    stop_terms = set(load_terms(paths.resources / "stopwords_zh.txt"))
    func_terms = set(load_terms(paths.resources / "function_words_zh.txt"))
    template_terms = load_terms(paths.resources / "templates_v1.txt")
    conn_explain = load_terms(paths.resources / "connectives_explain.txt")
    conn_contrast = load_terms(paths.resources / "connectives_contrast.txt")
    conn_progress = load_terms(paths.resources / "connectives_progression.txt")
    connectives_total_terms = []
    seen: set[str] = set()
    for t in [*conn_explain, *conn_contrast, *conn_progress]:
        if t and t not in seen:
            seen.add(t)
            connectives_total_terms.append(t)

    master_rows: list[dict] = []
    feature_rows: list[dict] = []

    master_path = out_dir / "master.csv"
    features_path = out_dir / "features.csv"
    existing_master = pd.read_csv(master_path, dtype=str).fillna("") if master_path.exists() else pd.DataFrame()
    existing_features = pd.read_csv(features_path, dtype=str).fillna("") if features_path.exists() else pd.DataFrame()
    existing_ids: set[str] = set()
    if not existing_master.empty and "video_id" in existing_master.columns:
        existing_ids = set(existing_master["video_id"].astype(str).tolist())

    since_bvids = _load_bvids_from_log(Path(args.since_log)) if args.since_log else set()

    missing_files = [b for b in phase_map.keys() if b not in bvid_to_path]
    for b in missing_files:
        _append_log(run_log, f"MISSING_CLEAN_FILE: {b} (expected in mapping)")

    target_bvids = set(phase_map.keys())
    if since_bvids:
        target_bvids = target_bvids.intersection(since_bvids)
    if args.only_new and existing_ids:
        target_bvids = target_bvids.difference(existing_ids)

    for bvid, mr in phase_map.items():
        if bvid not in target_bvids:
            continue
        p = bvid_to_path.get(bvid)
        if not p:
            continue

        text = _load_text_for_analysis(p)
        tokens = _tokenize(text)
        length_chars = count_chars(text)
        length_tokens = len(tokens)

        master_rows.append(
            {
                "video_id": bvid,
                "series": mr.series,
                "creator_group": mr.creator_group,
                "phase": mr.phase,
                "text": text,
                "length_chars": length_chars,
                "length_tokens": length_tokens,
            }
        )

        stop_cnt = sum(1 for t in tokens if t in stop_terms)
        func_cnt = sum(1 for t in tokens if t in func_terms)

        sentences = split_sentences(text)
        if not sentences and text.strip():
            sentences = [text.strip()]
        sent_lens_chars = [float(count_chars(s)) for s in sentences]

        templates_hits = substring_hits(text, template_terms)
        connectives_hits = substring_hits(text, connectives_total_terms)

        denom = float(length_tokens) if length_tokens > 0 else 1.0
        comma_n = text.count("，")
        period_n = text.count("。")
        comma_period_ratio = float(comma_n) / float(period_n if period_n > 0 else 1)

        feature_rows.append(
            {
                "video_id": bvid,
                "series": mr.series,
                "creator_group": mr.creator_group,
                "phase": mr.phase,
                "length_tokens": length_tokens,
                "mattr": round(_mattr(tokens, window=args.mattr_window), 6),
                "mean_word_len": round(_mean([float(len(t)) for t in tokens]), 6),
                "mean_sent_len_chars": round(_mean(sent_lens_chars), 6),
                "stop_ratio": round(float(stop_cnt) / denom, 6),
                "func_ratio": round(float(func_cnt) / denom, 6),
                "templates_density": round(float(templates_hits) * 1000.0 / denom, 6),
                "connectives_total": round(float(connectives_hits) * 1000.0 / denom, 6),
                "comma_period_ratio": round(comma_period_ratio, 6),
            }
        )

    new_master_df = pd.DataFrame(master_rows)
    new_features_df = pd.DataFrame(feature_rows)

    if args.only_new or args.since_log:
        master_df = pd.concat([existing_master, new_master_df], ignore_index=True)
        features_df = pd.concat([existing_features, new_features_df], ignore_index=True)
        if "video_id" in master_df.columns:
            master_df = master_df.drop_duplicates(subset=["video_id"], keep="first")
        if "video_id" in features_df.columns:
            features_df = features_df.drop_duplicates(subset=["video_id"], keep="first")
    else:
        master_df = new_master_df
        features_df = new_features_df

    master_df.to_csv(master_path, index=False, encoding="utf-8-sig")
    features_df.to_csv(features_path, index=False, encoding="utf-8-sig")
    _append_log(run_log, f"Wrote master: {master_path} (rows={len(master_df)}) new={len(new_master_df)}")
    _append_log(run_log, f"Wrote features: {features_path} (rows={len(features_df)}) new={len(new_features_df)}")

    if not args.no_plots:
        plots_dir = out_dir / "plots"
        _plot_phase_boxplot(
            features_df,
            metric="mattr",
            out_path=plots_dir / "box_mattr.png",
            title="MATTR by phase (window=100)",
        )
        _plot_phase_boxplot(
            features_df,
            metric="mean_sent_len_chars",
            out_path=plots_dir / "box_mean_sent_len_chars.png",
            title="Mean sentence length (chars) by phase",
        )
        _plot_phase_boxplot(
            features_df,
            metric="connectives_total",
            out_path=plots_dir / "box_connectives_total.png",
            title="Connectives density (per 1k tokens) by phase",
        )
        _plot_phase_boxplot(
            features_df,
            metric="comma_period_ratio",
            out_path=plots_dir / "box_comma_period_ratio.png",
            title="Comma/period ratio by phase",
        )
        _append_log(run_log, f"Wrote plots: {plots_dir}")

    log_path = out_dir / "run_log.txt"
    log_path.write_text("\n".join(run_log).strip() + "\n", encoding="utf-8-sig")
    _append_log(run_log, f"Wrote run log: {log_path}")

    # Update the file after the last log line.
    log_path.write_text("\n".join(run_log).strip() + "\n", encoding="utf-8-sig")

    if args.only_new or args.since_log:
        inc_path = run_dirs.logs / "run_log_incremental.txt"
        with inc_path.open("a", encoding="utf-8") as f:
            f.write(
                f"[{datetime.now().isoformat(timespec='seconds')}] v1_analysis: "
                f"new_master={len(new_master_df)} total_master={len(master_df)} "
                f"new_features={len(new_features_df)} total_features={len(features_df)} "
                f"(only_new={args.only_new}, since_log={bool(args.since_log)})\n"
            )

    # archive to run outputs
    try:
        import shutil

        shutil.copy2(master_path, run_dirs.outputs / "master.csv")
        shutil.copy2(features_path, run_dirs.outputs / "features.csv")
        if (out_dir / "plots").exists():
            dst_plots = run_dirs.outputs / "plots"
            if dst_plots.exists():
                shutil.rmtree(dst_plots)
            shutil.copytree(out_dir / "plots", dst_plots)
        shutil.copy2(log_path, run_dirs.logs / "run_log_analysis.txt")
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
