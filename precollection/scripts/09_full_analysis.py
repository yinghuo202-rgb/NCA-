from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st

from lib.breakpoints_ext import load_breakpoints_yaml
from lib.paths import ensure_dirs, get_project_paths
from lib.run_utils import init_run_dirs, make_run_id


PRIMARY_STYLE_METRICS = [
    "mattr",
    "mean_word_len",
    "mean_sent_len_chars",
    "stop_ratio",
    "func_ratio",
    "templates_density",
    "connectives_total",
    "comma_period_ratio",
]
PAIRWISE_PHASES = [("S0", "S1"), ("S1", "S2"), ("S0", "S2")]


def _log(lines: list[str], message: str) -> None:
    stamp = datetime.now().isoformat(timespec="seconds")
    line = f"[{stamp}] {message}"
    print(line)
    lines.append(line)


def _bh_fdr(values: list[float]) -> list[float]:
    if not values:
        return []
    values_arr = np.asarray(values, dtype=float)
    order = np.argsort(values_arr)
    adjusted = np.zeros(len(values_arr), dtype=float)
    prev = 1.0
    m = float(len(values_arr))
    for rank_rev, idx in enumerate(order[::-1], start=1):
        rank = len(values_arr) - rank_rev + 1
        candidate = min(prev, values_arr[idx] * m / float(rank))
        adjusted[idx] = min(candidate, 1.0)
        prev = candidate
    return adjusted.tolist()


def _bootstrap_ci(values: np.ndarray, *, n_boot: int = 1000, alpha: float = 0.05) -> tuple[float, float]:
    if values.size == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(42)
    draws = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=values.size, replace=True)
        draws.append(float(np.mean(sample)))
    lo = float(np.percentile(draws, 100 * (alpha / 2.0)))
    hi = float(np.percentile(draws, 100 * (1.0 - alpha / 2.0)))
    return (lo, hi)


def _dz(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    std = float(np.std(values, ddof=1))
    if std <= 1e-12:
        return 0.0
    return float(np.mean(values) / std)


def _ols_cluster_robust(
    X: np.ndarray,
    y: np.ndarray,
    *,
    clusters: np.ndarray,
    param_names: list[str],
) -> dict[str, Any] | None:
    n, k = X.shape
    if n <= k or len(np.unique(clusters)) < 5:
        return None

    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    fitted = X @ beta
    resid = y - fitted

    meat = np.zeros((k, k), dtype=float)
    unique_clusters = np.unique(clusters)
    for cluster in unique_clusters:
        idx = np.where(clusters == cluster)[0]
        Xg = X[idx, :]
        eg = resid[idx].reshape(-1, 1)
        meat += Xg.T @ (eg @ eg.T) @ Xg

    g = float(len(unique_clusters))
    correction = 1.0
    if g > 1 and n > k:
        correction = (g / (g - 1.0)) * ((n - 1.0) / (n - k))
    cov = correction * (xtx_inv @ meat @ xtx_inv)
    se = np.sqrt(np.clip(np.diag(cov), a_min=0.0, a_max=None))

    t_stats = np.full(k, np.nan, dtype=float)
    p_values = np.full(k, np.nan, dtype=float)
    valid = se > 1e-12
    t_stats[valid] = beta[valid] / se[valid]
    df = max(int(g) - 1, 1)
    p_values[valid] = 2.0 * st.t.sf(np.abs(t_stats[valid]), df=df)

    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "param_names": param_names,
        "coef": beta,
        "se": se,
        "t": t_stats,
        "p": p_values,
        "n": int(n),
        "k": int(k),
        "clusters": int(g),
        "r2": float(r2),
    }


def _numeric_columns(df: pd.DataFrame, *, excluded: set[str]) -> list[str]:
    cols: list[str] = []
    for column in df.columns:
        if column in excluded:
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        if int(numeric.notna().sum()) >= 5:
            df[column] = numeric
            cols.append(column)
    return cols


def _prepare_dataset(features: pd.DataFrame, manifest: pd.DataFrame, cfg) -> tuple[pd.DataFrame, list[str], list[str]]:
    key_feat = "video_id" if "video_id" in features.columns else "bvid"
    key_man = "unique_key" if "unique_key" in manifest.columns else "bvid"
    merge_cols = [
        key_man,
        "bvid",
        "creator_id",
        "creator_name",
        "creator_group",
        "phase_base",
        "pubdate",
        "pub_date",
        "strict_ok",
        "fill_level",
        "fill_reason",
        "subtitle_status",
    ]
    merge_cols = [column for column in merge_cols if column in manifest.columns]
    df = features.merge(
        manifest[merge_cols],
        how="left",
        left_on=key_feat,
        right_on=key_man,
        suffixes=("", "_m"),
    )

    if "phase" not in df.columns:
        df["phase"] = df.get("phase_base", "")
    df["phase"] = df["phase"].astype(str)
    df["creator_id"] = df.get("creator_id", "").astype(str)
    df["creator_name"] = df.get("creator_name", "").astype(str)
    df["creator_group"] = df.get("creator_group", "").astype(str)
    df["fill_level"] = pd.to_numeric(df.get("fill_level", 0), errors="coerce").fillna(0).astype(int)
    df["strict_ok"] = pd.to_numeric(df.get("strict_ok", 0), errors="coerce").fillna(0).astype(int)
    df["subtitle_status"] = df.get("subtitle_status", "").astype(str)
    df["pubdate_raw"] = df.get("pubdate", df.get("pub_date", "")).astype(str)
    df["pubdate_dt"] = pd.to_datetime(df["pubdate_raw"], errors="coerce")
    df = df[df["phase"].isin(["S0", "S1", "S2"])].copy()
    df = df.dropna(subset=["pubdate_dt"])

    if "length_tokens" in df.columns:
        df["length_tokens"] = pd.to_numeric(df["length_tokens"], errors="coerce")
    df["log_length_tokens"] = np.log1p(pd.to_numeric(df.get("length_tokens", 0), errors="coerce").fillna(0.0))

    df["time_weeks"] = (df["pubdate_dt"] - df["pubdate_dt"].min()).dt.days.astype(float) / 7.0
    t1 = pd.Timestamp(cfg.t1)
    t2 = pd.Timestamp(cfg.t2)
    df["post_t1"] = (df["pubdate_dt"] >= t1).astype(float)
    df["post_t2"] = (df["pubdate_dt"] >= t2).astype(float)
    df["weeks_after_t1"] = ((df["pubdate_dt"] - t1).dt.days.clip(lower=0).astype(float)) / 7.0
    df["weeks_after_t2"] = ((df["pubdate_dt"] - t2).dt.days.clip(lower=0).astype(float)) / 7.0

    feature_identifiers = {
        key_feat,
        "bvid",
        "series",
        "creator_group",
        "creator_id",
        "creator_name",
        "phase",
        "pubdate",
        "title",
        "clean_path",
        "text",
    }
    feature_metric_candidates = [column for column in features.columns if column not in feature_identifiers]
    numeric_cols: list[str] = []
    for column in feature_metric_candidates:
        numeric = pd.to_numeric(df[column], errors="coerce")
        if int(numeric.notna().sum()) >= 5:
            df[column] = numeric
            numeric_cols.append(column)
    style_metrics = [metric for metric in PRIMARY_STYLE_METRICS if metric in numeric_cols]

    if style_metrics:
        valid = df[style_metrics].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
        if int(valid.sum()) >= 10:
            values = df.loc[valid, style_metrics].astype(float)
            means = values.mean(axis=0)
            stds = values.std(axis=0, ddof=0).replace(0.0, 1.0)
            z = (values - means) / stds
            _, _, vt = np.linalg.svd(z.to_numpy(dtype=float), full_matrices=False)
            pc1 = z.to_numpy(dtype=float) @ vt[0]
            if "connectives_total" in style_metrics:
                ref = np.corrcoef(pc1, z["connectives_total"].to_numpy(dtype=float))[0, 1]
                if np.isfinite(ref) and ref < 0:
                    pc1 = -pc1
            df["style_index_pc1"] = np.nan
            df.loc[valid, "style_index_pc1"] = pc1
            numeric_cols.append("style_index_pc1")
            style_metrics.append("style_index_pc1")

    return df, numeric_cols, style_metrics


def _creator_pairwise_tests(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        pivot = (
            df.groupby(["creator_id", "phase"])[metric]
            .mean()
            .unstack()
        )
        for phase_a, phase_b in PAIRWISE_PHASES:
            if phase_a not in pivot.columns or phase_b not in pivot.columns:
                continue
            subset = pivot[[phase_a, phase_b]].dropna()
            if len(subset) < 10:
                continue
            diff = (subset[phase_b] - subset[phase_a]).to_numpy(dtype=float)
            if diff.size == 0:
                continue
            try:
                t_res = st.ttest_1samp(diff, popmean=0.0, nan_policy="omit")
                t_p = float(t_res.pvalue)
            except Exception:
                t_p = np.nan
            try:
                if np.allclose(diff, diff[0]):
                    wilcoxon_p = np.nan
                else:
                    wilcoxon_p = float(st.wilcoxon(diff).pvalue)
            except Exception:
                wilcoxon_p = np.nan
            ci_low, ci_high = _bootstrap_ci(diff)
            rows.append(
                {
                    "metric": metric,
                    "pair": f"{phase_a} vs {phase_b}",
                    "n_creators": int(len(subset)),
                    "mean_phase_a": float(subset[phase_a].mean()),
                    "mean_phase_b": float(subset[phase_b].mean()),
                    "mean_diff": float(np.mean(diff)),
                    "median_diff": float(np.median(diff)),
                    "diff_ci_low": ci_low,
                    "diff_ci_high": ci_high,
                    "effect_dz": _dz(diff),
                    "t_p": t_p,
                    "wilcoxon_p": wilcoxon_p,
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["t_p_fdr"] = _bh_fdr([float(v) if pd.notna(v) else 1.0 for v in out["t_p"].tolist()])
    out["wilcoxon_p_fdr"] = _bh_fdr([float(v) if pd.notna(v) else 1.0 for v in out["wilcoxon_p"].tolist()])
    return out.sort_values(["t_p_fdr", "metric", "pair"]).reset_index(drop=True)


def _phase_fe_tests(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        cols = ["creator_id", "phase", metric, "fill_level"]
        if metric != "length_tokens" and "log_length_tokens" in df.columns:
            cols.append("log_length_tokens")
        work = df[cols].copy()
        work[metric] = pd.to_numeric(work[metric], errors="coerce")
        work = work.dropna(subset=[metric])
        if len(work) < 30:
            continue

        creators = sorted(work["creator_id"].astype(str).unique().tolist())
        phases = ["S0", "S1", "S2"]
        param_names = ["intercept", "phase_S1", "phase_S2"]
        matrix = [
            np.ones(len(work), dtype=float),
            (work["phase"] == "S1").astype(float).to_numpy(),
            (work["phase"] == "S2").astype(float).to_numpy(),
        ]
        if metric != "length_tokens" and "log_length_tokens" in work.columns:
            matrix.append(pd.to_numeric(work["log_length_tokens"], errors="coerce").fillna(0.0).to_numpy(dtype=float))
            param_names.append("log_length_tokens")
        matrix.append(pd.to_numeric(work["fill_level"], errors="coerce").fillna(0.0).to_numpy(dtype=float))
        param_names.append("fill_level")

        for creator in creators[1:]:
            matrix.append((work["creator_id"] == creator).astype(float).to_numpy())
            param_names.append(f"creator_{creator}")

        X = np.column_stack(matrix)
        y = work[metric].to_numpy(dtype=float)
        clusters = work["creator_id"].to_numpy()
        result = _ols_cluster_robust(X, y, clusters=clusters, param_names=param_names)
        if result is None:
            continue

        for term in ["phase_S1", "phase_S2"]:
            idx = param_names.index(term)
            rows.append(
                {
                    "metric": metric,
                    "term": term,
                    "coef": float(result["coef"][idx]),
                    "se": float(result["se"][idx]),
                    "t": float(result["t"][idx]),
                    "p": float(result["p"][idx]),
                    "n": int(result["n"]),
                    "creator_clusters": int(result["clusters"]),
                    "r2": float(result["r2"]),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_fdr"] = _bh_fdr(out["p"].tolist())
    return out.sort_values(["p_fdr", "metric", "term"]).reset_index(drop=True)


def _segmented_breakpoint_tests(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        cols = [
            "creator_id",
            "time_weeks",
            "post_t1",
            "weeks_after_t1",
            "post_t2",
            "weeks_after_t2",
            metric,
            "fill_level",
        ]
        if metric != "length_tokens" and "log_length_tokens" in df.columns:
            cols.append("log_length_tokens")
        work = df[cols].copy()
        work[metric] = pd.to_numeric(work[metric], errors="coerce")
        work = work.dropna(subset=[metric])
        if len(work) < 30:
            continue

        creators = sorted(work["creator_id"].astype(str).unique().tolist())
        param_names = ["intercept", "time_weeks", "post_t1", "weeks_after_t1", "post_t2", "weeks_after_t2"]
        matrix = [
            np.ones(len(work), dtype=float),
            work["time_weeks"].to_numpy(dtype=float),
            work["post_t1"].to_numpy(dtype=float),
            work["weeks_after_t1"].to_numpy(dtype=float),
            work["post_t2"].to_numpy(dtype=float),
            work["weeks_after_t2"].to_numpy(dtype=float),
        ]
        if metric != "length_tokens" and "log_length_tokens" in work.columns:
            matrix.append(pd.to_numeric(work["log_length_tokens"], errors="coerce").fillna(0.0).to_numpy(dtype=float))
            param_names.append("log_length_tokens")
        matrix.append(pd.to_numeric(work["fill_level"], errors="coerce").fillna(0.0).to_numpy(dtype=float))
        param_names.append("fill_level")

        for creator in creators[1:]:
            matrix.append((work["creator_id"] == creator).astype(float).to_numpy())
            param_names.append(f"creator_{creator}")

        X = np.column_stack(matrix)
        y = work[metric].to_numpy(dtype=float)
        clusters = work["creator_id"].to_numpy()
        result = _ols_cluster_robust(X, y, clusters=clusters, param_names=param_names)
        if result is None:
            continue

        for term in ["post_t1", "weeks_after_t1", "post_t2", "weeks_after_t2"]:
            idx = param_names.index(term)
            rows.append(
                {
                    "metric": metric,
                    "term": term,
                    "coef": float(result["coef"][idx]),
                    "se": float(result["se"][idx]),
                    "t": float(result["t"][idx]),
                    "p": float(result["p"][idx]),
                    "n": int(result["n"]),
                    "creator_clusters": int(result["clusters"]),
                    "r2": float(result["r2"]),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_fdr"] = _bh_fdr(out["p"].tolist())
    return out.sort_values(["p_fdr", "metric", "term"]).reset_index(drop=True)


def _phase_stats(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        for phase in ["S0", "S1", "S2"]:
            subset = pd.to_numeric(df.loc[df["phase"] == phase, metric], errors="coerce").dropna()
            if subset.empty:
                continue
            rows.append(
                {
                    "phase": phase,
                    "metric": metric,
                    "n": int(subset.shape[0]),
                    "mean": float(subset.mean()),
                    "median": float(subset.median()),
                    "std": float(subset.std(ddof=1) if subset.shape[0] > 1 else 0.0),
                    "min": float(subset.min()),
                    "max": float(subset.max()),
                    "iqr": float(subset.quantile(0.75) - subset.quantile(0.25)),
                }
            )
    return pd.DataFrame(rows)


def _plot_boxplot(df: pd.DataFrame, *, metric: str, out_path: Path, title: str) -> None:
    tmp = df.copy()
    tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce")
    tmp = tmp.dropna(subset=[metric])
    if tmp.empty:
        return
    order = ["S0", "S1", "S2"]
    data = [tmp.loc[tmp["phase"] == phase, metric].tolist() for phase in order]
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
    ax.boxplot(data, tick_labels=order, showmeans=True)
    ax.set_title(title)
    ax.set_xlabel("Phase")
    ax.set_ylabel(metric)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_time_trend(df: pd.DataFrame, *, metric: str, out_path: Path, title: str, t1: pd.Timestamp, t2: pd.Timestamp) -> None:
    tmp = df[["pubdate_dt", metric]].copy()
    tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce")
    tmp = tmp.dropna(subset=[metric, "pubdate_dt"])
    if tmp.empty:
        return
    tmp["month"] = tmp["pubdate_dt"].dt.to_period("M").dt.to_timestamp()
    grouped = tmp.groupby("month")[metric].mean().reset_index()
    if grouped.empty:
        return
    grouped["rolling_3m"] = grouped[metric].rolling(window=3, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=150)
    ax.plot(grouped["month"], grouped[metric], color="#9ecae1", linewidth=1.0, alpha=0.8, label="monthly mean")
    ax.plot(grouped["month"], grouped["rolling_3m"], color="#08519c", linewidth=2.0, label="3-month rolling mean")
    ax.axvline(t1, color="#ef3b2c", linestyle="--", linewidth=1.2, label="T1")
    ax.axvline(t2, color="#31a354", linestyle="--", linewidth=1.2, label="T2")
    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel(metric)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend(frameon=False, ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _creator_coverage_table(df: pd.DataFrame) -> pd.DataFrame:
    coverage = (
        df.groupby(["creator_id", "phase"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for phase in ["S0", "S1", "S2"]:
        if phase not in coverage.columns:
            coverage[phase] = 0
    coverage["has_all_phases"] = ((coverage["S0"] > 0) & (coverage["S1"] > 0) & (coverage["S2"] > 0)).astype(int)
    coverage["full_10x3"] = ((coverage["S0"] >= 10) & (coverage["S1"] >= 10) & (coverage["S2"] >= 10)).astype(int)
    return coverage.sort_values(["has_all_phases", "full_10x3", "creator_id"], ascending=[False, False, True]).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Full breakpoint analysis with creator-level and segmented models")
    parser.add_argument("--features", default="outputs/features.csv", help="features.csv path")
    parser.add_argument("--manifest", default="outputs/final_manifest.csv", help="final_manifest.csv path")
    parser.add_argument("--breakpoints", default="config/breakpoints.yaml", help="breakpoints yaml path")
    parser.add_argument("--out-dir", default="", help="output dir (default runs/<run_id>/outputs)")
    parser.add_argument("--run-id", default="", help="run id for archiving under runs/<run_id>")
    parser.add_argument("--min-n", type=int, default=30, help="minimum rows per metric model")
    args = parser.parse_args()

    paths = get_project_paths()
    ensure_dirs(paths)

    run_id = make_run_id(args.run_id or None)
    run_dirs = init_run_dirs(paths.root, run_id)
    out_dir = Path(args.out_dir) if args.out_dir else run_dirs.outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = out_dir / "tables"
    plots_dir = out_dir / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = []
    _log(log_lines, f"Start full analysis run_id={run_id}")

    features_path = (paths.root / args.features).resolve()
    manifest_path = (paths.root / args.manifest).resolve()
    breakpoints_path = (paths.root / args.breakpoints).resolve()
    cfg = load_breakpoints_yaml(breakpoints_path)

    features = pd.read_csv(features_path, dtype=str).fillna("")
    manifest = pd.read_csv(manifest_path, dtype=str).fillna("")
    df, numeric_metrics, style_metrics = _prepare_dataset(features, manifest, cfg)
    if df.empty:
        raise ValueError("No analyzable rows after merging features and manifest")

    if args.min_n > 0:
        numeric_metrics = [
            metric
            for metric in numeric_metrics
            if int(pd.to_numeric(df[metric], errors="coerce").notna().sum()) >= int(args.min_n)
        ]

    phase_counts = df["phase"].value_counts().reindex(["S0", "S1", "S2"]).fillna(0).astype(int)
    phase_counts.rename_axis("phase").reset_index(name="count").to_csv(
        tables_dir / "phase_counts.csv", index=False, encoding="utf-8-sig"
    )

    fill_counts = df["fill_level"].value_counts().sort_index()
    fill_counts.rename_axis("fill_level").reset_index(name="count").to_csv(
        tables_dir / "fill_level_counts.csv", index=False, encoding="utf-8-sig"
    )

    sample_structure = (
        df.groupby("phase")
        .agg(n=("phase", "count"), creators=("creator_id", "nunique"))
        .reset_index()
    )
    sample_structure.to_csv(tables_dir / "sample_structure.csv", index=False, encoding="utf-8-sig")

    coverage = _creator_coverage_table(df)
    coverage.to_csv(tables_dir / "creator_phase_coverage.csv", index=False, encoding="utf-8-sig")

    phase_stats = _phase_stats(df, numeric_metrics)
    phase_stats.to_csv(tables_dir / "phase_stats.csv", index=False, encoding="utf-8-sig")

    pairwise_df = _creator_pairwise_tests(df, numeric_metrics)
    if not pairwise_df.empty:
        pairwise_df.to_csv(tables_dir / "creator_phase_tests.csv", index=False, encoding="utf-8-sig")

    phase_fe_df = _phase_fe_tests(df, numeric_metrics)
    if not phase_fe_df.empty:
        phase_fe_df.to_csv(tables_dir / "creator_fe_phase.csv", index=False, encoding="utf-8-sig")

    segmented_df = _segmented_breakpoint_tests(df, numeric_metrics)
    if not segmented_df.empty:
        segmented_df.to_csv(tables_dir / "segmented_breakpoints.csv", index=False, encoding="utf-8-sig")

    corr_cols = [metric for metric in numeric_metrics if metric in df.columns]
    if corr_cols:
        corr = df[corr_cols].corr(method="spearman")
        corr.to_csv(tables_dir / "spearman_corr.csv", encoding="utf-8-sig")

    fig_phase_counts = plt.figure(figsize=(6.0, 4.0), dpi=150)
    ax = fig_phase_counts.add_subplot(111)
    ax.bar(phase_counts.index.tolist(), phase_counts.values.tolist(), color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax.set_title("Phase counts")
    ax.set_xlabel("Phase")
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig_phase_counts.tight_layout()
    fig_phase_counts.savefig(plots_dir / "fig01_phase_counts.png")
    plt.close(fig_phase_counts)

    for idx, metric in enumerate([m for m in ["mattr", "connectives_total", "mean_sent_len_chars", "style_index_pc1"] if m in df.columns], start=2):
        _plot_boxplot(
            df,
            metric=metric,
            out_path=plots_dir / f"fig0{idx}_{metric}_box.png",
            title=f"{metric} by phase",
        )

    t1 = pd.Timestamp(cfg.t1)
    t2 = pd.Timestamp(cfg.t2)
    if "mattr" in df.columns:
        _plot_time_trend(
            df,
            metric="mattr",
            out_path=plots_dir / "fig06_mattr_time_trend.png",
            title="MATTR monthly trend with breakpoints",
            t1=t1,
            t2=t2,
        )
    if "style_index_pc1" in df.columns:
        _plot_time_trend(
            df,
            metric="style_index_pc1",
            out_path=plots_dir / "fig07_style_index_time_trend.png",
            title="Style index monthly trend with breakpoints",
            t1=t1,
            t2=t2,
        )

    phase_sig_metrics = 0
    if not phase_fe_df.empty:
        phase_sig_metrics = int(phase_fe_df.loc[phase_fe_df["p_fdr"] < 0.05, "metric"].nunique())
    segmented_sig_metrics = 0
    if not segmented_df.empty:
        segmented_sig_metrics = int(segmented_df.loc[segmented_df["p_fdr"] < 0.05, "metric"].nunique())
    pair_sig = 0
    if not pairwise_df.empty:
        pair_sig = int((pairwise_df["t_p_fdr"] < 0.05).sum())

    top_lines: list[str] = []
    if not phase_fe_df.empty:
        for _, row in phase_fe_df.head(5).iterrows():
            top_lines.append(
                f"- FE phase: {row['metric']} / {row['term']} coef={row['coef']:.4f}, p_fdr={row['p_fdr']:.4g}"
            )
    if not segmented_df.empty:
        for _, row in segmented_df.head(5).iterrows():
            top_lines.append(
                f"- Breakpoint: {row['metric']} / {row['term']} coef={row['coef']:.4f}, p_fdr={row['p_fdr']:.4g}"
            )
    if not pairwise_df.empty:
        for _, row in pairwise_df.head(5).iterrows():
            top_lines.append(
                f"- Creator paired: {row['metric']} / {row['pair']} mean_diff={row['mean_diff']:.4f}, p_fdr={row['t_p_fdr']:.4g}"
            )

    report_lines = [
        f"# Analysis Report ({run_id})",
        "",
        f"- features_path: {features_path}",
        f"- manifest_path: {manifest_path}",
        f"- breakpoints_path: {breakpoints_path}",
        f"- breakpoints: T1={cfg.t1.isoformat()}, T2={cfg.t2.isoformat()}",
        f"- samples_ok: {len(df)}",
        f"- creators: {df['creator_id'].nunique()}",
        f"- creators_with_all_phases: {int(coverage['has_all_phases'].sum())}",
        f"- creators_with_10x3_ok: {int(coverage['full_10x3'].sum())}",
        f"- numeric_metrics: {', '.join(numeric_metrics)}",
        "",
        "## Methods",
        "- Creator-level paired comparisons on creator mean values across phases.",
        "- Creator fixed-effects panel models with cluster-robust standard errors.",
        "- Segmented breakpoint regressions around T1 and T2, with creator fixed effects and length/fill controls.",
        "",
        "## Key Results",
        f"- Creator FE significant metrics (FDR<0.05): {phase_sig_metrics}",
        f"- Segmented breakpoint significant metrics (FDR<0.05): {segmented_sig_metrics}",
        f"- Creator paired significant comparisons (FDR<0.05): {pair_sig}",
    ]
    if top_lines:
        report_lines.extend(["", "## Top Findings"])
        report_lines.extend(top_lines)
    report_lines.extend(
        [
            "",
            "## Outputs",
            f"- tables: {tables_dir}",
            f"- plots: {plots_dir}",
        ]
    )
    (out_dir / "analysis_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    (out_dir / "analysis_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
