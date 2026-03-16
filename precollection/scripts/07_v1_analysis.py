from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from pandas.errors import EmptyDataError

from lib.lexicon import load_terms
from lib.paths import ensure_dirs, get_project_paths
from lib.run_utils import init_run_dirs, make_run_id
from lib.text_processing import clean_text, count_chars, normalize_punctuation, split_sentences, substring_hits


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
        if not line or not line.startswith("{"):
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


def _load_text_for_analysis(path: Path) -> str:
    raw = path.read_text(encoding="utf-8-sig")
    raw = raw.replace("\ufeff", "")
    raw = normalize_punctuation(raw)
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.strip() for ln in raw.split("\n")]
    lines = [ln for ln in lines if ln]
    merged = " ".join(lines)
    merged = re.sub(r"\s+", " ", merged).strip()
    return merged


def _tokenize(text: str) -> list[str]:
    try:
        import jieba  # type: ignore

        try:
            import logging

            jieba.setLogLevel(logging.ERROR)
        except Exception:
            pass

        tokens = jieba.lcut(text, cut_all=False)
    except Exception:
        tokens = re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9]+", text)

    out: list[str] = []
    for token in tokens:
        s = str(token).strip()
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

    counts: dict[str, int] = {}
    unique = 0
    for token in tokens[:window]:
        prev = counts.get(token, 0)
        counts[token] = prev + 1
        if prev == 0:
            unique += 1

    ttr_sum = unique / float(window)
    windows = 1
    for i in range(window, n):
        out_token = tokens[i - window]
        counts[out_token] -= 1
        if counts[out_token] == 0:
            del counts[out_token]
            unique -= 1

        in_token = tokens[i]
        prev = counts.get(in_token, 0)
        counts[in_token] = prev + 1
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

    data = [tmp.loc[tmp["phase"] == phase, metric].tolist() for phase in order]
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


def _collect_clean_maps(input_dir: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    by_video_id: dict[str, Path] = {}
    by_bvid: dict[str, Path] = {}
    if not input_dir.exists():
        return by_video_id, by_bvid

    for path in sorted(input_dir.rglob("clean*.txt")):
        video_id = ""
        bvid = ""
        if path.name == "clean.txt":
            video_id = path.parent.name.strip()
            if video_id.startswith("BV"):
                bvid = video_id.split("_", 1)[0]
        elif path.stem.startswith("clean_"):
            bvid = path.stem[len("clean_") :].strip()
            video_id = bvid

        if video_id and video_id not in by_video_id:
            by_video_id[video_id] = path
        if bvid and bvid not in by_bvid:
            by_bvid[bvid] = path

    return by_video_id, by_bvid


def _resolve_manifest_rows(manifest_path: Path, *, phase_col: str) -> list[dict[str, str]]:
    manifest = pd.read_csv(manifest_path, dtype=str).fillna("")
    if phase_col not in manifest.columns:
        raise ValueError(f"manifest missing phase column: {phase_col}")

    rows: list[dict[str, str]] = []
    for _, row in manifest.iterrows():
        phase = str(row.get(phase_col) or "").strip()
        if phase not in {"S0", "S1", "S2"}:
            continue
        bvid = str(row.get("bvid") or row.get("video_id") or "").strip()
        video_id = str(row.get("unique_key") or bvid or row.get("video_id") or "").strip()
        if not video_id:
            continue
        rows.append(
            {
                "video_id": video_id,
                "bvid": bvid or video_id.split("_", 1)[0],
                "creator_id": str(row.get("creator_id") or "").strip(),
                "creator_name": str(row.get("creator_name") or "").strip(),
                "creator_group": str(row.get("creator_group") or "").strip(),
                "series": str(row.get("series") or "").strip(),
                "phase": phase,
                "title": str(row.get("title") or "").strip(),
                "pubdate": str(row.get("pubdate") or row.get("pub_date") or "").strip(),
            }
        )
    return rows


def _resolve_clean_path(
    *,
    paths,
    run_id: str,
    creator_id: str,
    video_id: str,
    bvid: str,
    by_video_id: dict[str, Path],
    by_bvid: dict[str, Path],
    onomatopoeia_terms: list[str],
) -> Path | None:
    if run_id and creator_id and video_id:
        run_path = paths.data_processed_text / run_id / creator_id / video_id / "clean.txt"
        if run_path.exists():
            return run_path
        raw_run_path = paths.data_processed_text / run_id / creator_id / video_id / "raw.txt"
        if raw_run_path.exists():
            cleaned, _ = clean_text(raw_run_path.read_text(encoding="utf-8"), onomatopoeia_terms=onomatopoeia_terms)
            run_path.write_text(cleaned, encoding="utf-8")
            return run_path

    if creator_id and bvid:
        legacy_path = paths.data_processed_text / creator_id / f"clean_{bvid}.txt"
        if legacy_path.exists():
            return legacy_path
        legacy_raw_path = paths.data_processed_text / creator_id / f"raw_{bvid}.txt"
        if legacy_raw_path.exists():
            cleaned, _ = clean_text(legacy_raw_path.read_text(encoding="utf-8"), onomatopoeia_terms=onomatopoeia_terms)
            legacy_path.write_text(cleaned, encoding="utf-8")
            return legacy_path

    if video_id in by_video_id:
        return by_video_id[video_id]
    if bvid in by_bvid:
        return by_bvid[bvid]
    return None


def _build_feature_row(item: dict[str, str], text: str, *, mattr_window: int, stop_terms: set[str], func_terms: set[str], template_terms: list[str], connectives_total_terms: list[str]) -> dict[str, Any]:
    tokens = _tokenize(text)
    length_chars = count_chars(text)
    length_tokens = len(tokens)
    stop_count = sum(1 for token in tokens if token in stop_terms)
    func_count = sum(1 for token in tokens if token in func_terms)
    sentences = split_sentences(text)
    if not sentences and text.strip():
        sentences = [text.strip()]
    sentence_lengths = [float(count_chars(sentence)) for sentence in sentences]
    templates_hits = substring_hits(text, template_terms)
    connectives_hits = substring_hits(text, connectives_total_terms)
    denom = float(length_tokens) if length_tokens > 0 else 1.0
    comma_n = text.count("，") + text.count(",")
    period_n = text.count("。") + text.count(".")
    comma_period_ratio = float(comma_n) / float(period_n if period_n > 0 else 1)

    return {
        "video_id": item["video_id"],
        "bvid": item["bvid"],
        "creator_id": item["creator_id"],
        "creator_name": item["creator_name"],
        "series": item["series"],
        "creator_group": item["creator_group"],
        "phase": item["phase"],
        "pubdate": item["pubdate"],
        "title": item["title"],
        "length_chars": length_chars,
        "length_tokens": length_tokens,
        "mattr": round(_mattr(tokens, window=mattr_window), 6),
        "mean_word_len": round(_mean([float(len(token)) for token in tokens]), 6),
        "mean_sent_len_chars": round(_mean(sentence_lengths), 6),
        "stop_ratio": round(float(stop_count) / denom, 6),
        "func_ratio": round(float(func_count) / denom, 6),
        "templates_density": round(float(templates_hits) * 1000.0 / denom, 6),
        "connectives_total": round(float(connectives_hits) * 1000.0 / denom, 6),
        "comma_period_ratio": round(comma_period_ratio, 6),
    }


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=str).fillna("")
    except EmptyDataError:
        return pd.DataFrame()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build full-sample master/features tables from clean text")
    parser.add_argument("--manifest", default="outputs/final_manifest.csv", help="manifest csv path")
    parser.add_argument("--phase-col", default="phase_base", help="phase column in manifest")
    parser.add_argument("--phase-shift", type=int, default=0, help="override phase column using phase_shift_{+/-N}")
    parser.add_argument("--mapping", default="config/phase_mapping_v1.csv", help="legacy phase mapping csv path")
    parser.add_argument("--run-id", default="", help="run id for archiving under runs/<run_id>")
    parser.add_argument("--input-dir", default="data/processed/text", help="directory containing clean text")
    parser.add_argument("--out-dir", default="outputs", help="output directory")
    parser.add_argument("--mattr-window", type=int, default=100, help="MATTR window size")
    parser.add_argument("--no-plots", action="store_true", help="skip plot generation")
    parser.add_argument("--only-new", action="store_true", help="append only new samples")
    parser.add_argument("--since-log", default="", help="restrict to bvids marked OK in a download log")
    args = parser.parse_args()

    paths = get_project_paths()
    ensure_dirs(paths)

    requested_run_id = str(args.run_id).strip()
    run_id = make_run_id(requested_run_id or None)
    run_dirs = init_run_dirs(paths.root, run_id)

    out_dir = (paths.root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    run_log: list[str] = []
    _append_log(run_log, "Start v1 analysis pipeline")
    _append_log(run_log, "Build features directly from manifest + clean text")

    phase_col = args.phase_col
    if args.phase_shift != 0:
        phase_col = f"phase_shift_{args.phase_shift:+d}"

    manifest_path = (paths.root / args.manifest).resolve()
    items: list[dict[str, str]] = []
    if manifest_path.exists():
        items = _resolve_manifest_rows(manifest_path, phase_col=phase_col)
        _append_log(run_log, f"Loaded manifest rows: {manifest_path} (rows={len(items)}) phase_col={phase_col}")
    else:
        mapping_path = (paths.root / args.mapping).resolve()
        phase_map = _read_phase_mapping(mapping_path)
        items = [
            {
                "video_id": row.bvid,
                "bvid": row.bvid,
                "creator_id": "",
                "creator_name": "",
                "creator_group": row.creator_group,
                "series": row.series,
                "phase": row.phase,
                "title": row.title,
                "pubdate": "",
            }
            for row in phase_map.values()
        ]
        _append_log(run_log, f"Loaded legacy phase mapping: {mapping_path} (rows={len(items)})")

    input_dir = (paths.root / args.input_dir).resolve()
    by_video_id, by_bvid = _collect_clean_maps(input_dir)
    _append_log(
        run_log,
        f"Indexed clean text files from {input_dir} (video_ids={len(by_video_id)}, bvids={len(by_bvid)})",
    )

    stop_terms = set(load_terms(paths.resources / "stopwords_zh.txt"))
    func_terms = set(load_terms(paths.resources / "function_words_zh.txt"))
    template_terms = load_terms(paths.resources / "templates_v1.txt")
    onomatopoeia_terms = load_terms(paths.resources / "onomatopoeia.txt")
    connectives_total_terms: list[str] = []
    seen_terms: set[str] = set()
    for name in ["connectives_explain.txt", "connectives_contrast.txt", "connectives_progression.txt"]:
        for term in load_terms(paths.resources / name):
            if term and term not in seen_terms:
                seen_terms.add(term)
                connectives_total_terms.append(term)

    master_path = out_dir / "master.csv"
    features_path = out_dir / "features.csv"
    existing_master = _read_csv_if_exists(master_path)
    existing_features = _read_csv_if_exists(features_path)
    existing_ids: set[str] = set()
    if not existing_master.empty and "video_id" in existing_master.columns:
        existing_ids = set(existing_master["video_id"].astype(str).tolist())

    since_bvids = _load_bvids_from_log(Path(args.since_log)) if args.since_log else set()

    master_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    missing_paths = 0

    for item in items:
        video_id = item["video_id"]
        bvid = item["bvid"]
        if since_bvids and bvid not in since_bvids:
            continue
        if args.only_new and video_id in existing_ids:
            continue

        clean_path = _resolve_clean_path(
            paths=paths,
            run_id=requested_run_id,
            creator_id=item["creator_id"],
            video_id=video_id,
            bvid=bvid,
            by_video_id=by_video_id,
            by_bvid=by_bvid,
            onomatopoeia_terms=onomatopoeia_terms,
        )
        if clean_path is None:
            missing_paths += 1
            _append_log(run_log, f"MISSING_CLEAN_FILE: video_id={video_id} bvid={bvid}")
            continue

        text = _load_text_for_analysis(clean_path)
        master_rows.append(
            {
                "video_id": video_id,
                "bvid": bvid,
                "creator_id": item["creator_id"],
                "creator_name": item["creator_name"],
                "series": item["series"],
                "creator_group": item["creator_group"],
                "phase": item["phase"],
                "pubdate": item["pubdate"],
                "title": item["title"],
                "clean_path": str(clean_path),
                "text": text,
                "length_chars": count_chars(text),
                "length_tokens": len(_tokenize(text)),
            }
        )
        feature_rows.append(
            _build_feature_row(
                item,
                text,
                mattr_window=args.mattr_window,
                stop_terms=stop_terms,
                func_terms=func_terms,
                template_terms=template_terms,
                connectives_total_terms=connectives_total_terms,
            )
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
    _append_log(run_log, f"Missing clean files skipped: {missing_paths}")

    if not args.no_plots:
        plots_dir = out_dir / "plots"
        _plot_phase_boxplot(
            features_df,
            metric="mattr",
            out_path=plots_dir / "box_mattr.png",
            title="MATTR by phase",
        )
        _plot_phase_boxplot(
            features_df,
            metric="mean_sent_len_chars",
            out_path=plots_dir / "box_mean_sent_len_chars.png",
            title="Mean sentence length by phase",
        )
        _plot_phase_boxplot(
            features_df,
            metric="connectives_total",
            out_path=plots_dir / "box_connectives_total.png",
            title="Connectives density by phase",
        )
        _plot_phase_boxplot(
            features_df,
            metric="comma_period_ratio",
            out_path=plots_dir / "box_comma_period_ratio.png",
            title="Comma / period ratio by phase",
        )
        _append_log(run_log, f"Wrote plots: {plots_dir}")

    log_path = out_dir / "run_log.txt"
    log_path.write_text("\n".join(run_log).strip() + "\n", encoding="utf-8-sig")

    if args.only_new or args.since_log:
        inc_path = run_dirs.logs / "run_log_incremental.txt"
        with inc_path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"[{datetime.now().isoformat(timespec='seconds')}] v1_analysis: "
                f"new_master={len(new_master_df)} total_master={len(master_df)} "
                f"new_features={len(new_features_df)} total_features={len(features_df)} "
                f"(only_new={args.only_new}, since_log={bool(args.since_log)})\n"
            )

    try:
        import shutil

        shutil.copy2(master_path, run_dirs.outputs / "master.csv")
        shutil.copy2(features_path, run_dirs.outputs / "features.csv")
        plots_dir = out_dir / "plots"
        if plots_dir.exists():
            dst_plots = run_dirs.outputs / "plots"
            if dst_plots.exists():
                shutil.rmtree(dst_plots)
            shutil.copytree(plots_dir, dst_plots)
        shutil.copy2(log_path, run_dirs.logs / "run_log_analysis.txt")
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
