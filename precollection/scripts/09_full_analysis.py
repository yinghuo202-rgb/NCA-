from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lib.paths import ensure_dirs, get_project_paths
from lib.run_utils import init_run_dirs, make_run_id


def _log(lines: list[str], msg: str) -> None:
    stamp = datetime.now().isoformat(timespec="seconds")
    line = f"[{stamp}] {msg}"
    print(line)
    lines.append(line)


def _bh_fdr(pvals: list[float]) -> list[float]:
    m = len(pvals)
    if m == 0:
        return []
    order = np.argsort(pvals)
    adj = [0.0] * m
    prev = 1.0
    for i in range(m - 1, -1, -1):
        idx = order[i]
        rank = i + 1
        val = pvals[idx] * m / rank
        prev = min(prev, val)
        adj[idx] = min(prev, 1.0)
    return adj


def _welch_anova(groups: list[np.ndarray]) -> tuple[float, float, float, float]:
    # Returns F, df1, df2, p
    import scipy.stats as st  # type: ignore

    k = len(groups)
    means = np.array([g.mean() for g in groups])
    ns = np.array([len(g) for g in groups], dtype=float)
    vars_ = np.array([g.var(ddof=1) if len(g) > 1 else 0.0 for g in groups])
    # avoid zero variance
    vars_[vars_ <= 1e-12] = 1e-12
    weights = ns / vars_
    w_sum = weights.sum()
    mean_w = (weights * means).sum() / w_sum
    num = (weights * (means - mean_w) ** 2).sum() / (k - 1)
    term = ((1 - weights / w_sum) ** 2) / (ns - 1)
    term = np.where(np.isfinite(term), term, 0.0)
    denom = 1 + (2 * (k - 2) / (k**2 - 1)) * term.sum()
    f = num / denom
    df1 = k - 1
    df2 = (k**2 - 1) / (3 * term.sum()) if term.sum() > 0 else 1e9
    p = st.f.sf(f, df1, df2)
    return float(f), float(df1), float(df2), float(p)


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return 0.0
    va = a.var(ddof=1) if len(a) > 1 else 0.0
    vb = b.var(ddof=1) if len(b) > 1 else 0.0
    denom = math.sqrt((va + vb) / 2.0) if (va + vb) > 0 else 1.0
    return float((a.mean() - b.mean()) / denom)


def _bootstrap_ci(a: np.ndarray, b: np.ndarray, n: int = 500, alpha: float = 0.05) -> tuple[float, float]:
    if len(a) == 0 or len(b) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(42)
    vals = []
    for _ in range(n):
        sa = rng.choice(a, size=len(a), replace=True)
        sb = rng.choice(b, size=len(b), replace=True)
        vals.append(_cohens_d(sa, sb))
    lo = np.percentile(vals, 100 * (alpha / 2))
    hi = np.percentile(vals, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def _vif(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = []
    X = df[cols].to_numpy(dtype=float)
    for i, c in enumerate(cols):
        y = X[:, i]
        X_others = np.delete(X, i, axis=1)
        if X_others.size == 0:
            out.append({"feature": c, "vif": 1.0})
            continue
        # add intercept
        Xo = np.column_stack([np.ones(X_others.shape[0]), X_others])
        coef, *_ = np.linalg.lstsq(Xo, y, rcond=None)
        y_hat = Xo @ coef
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        vif = 1.0 / max(1e-6, 1 - r2)
        out.append({"feature": c, "vif": float(vif), "r2": float(r2)})
    return pd.DataFrame(out)


def _fixed_effects(df: pd.DataFrame, metric: str) -> dict[str, Any]:
    # OLS with creator_id fixed effects + phase dummies
    y = pd.to_numeric(df[metric], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(y)
    work = df.loc[mask].copy()
    y = y[mask]
    if len(y) < 5:
        return {"metric": metric, "n": len(y), "r2": 0.0}
    creators = work["creator_id"].astype(str).fillna("NA").tolist()
    phases = work["phase"].astype(str).fillna("NA").tolist()

    creator_cats = sorted(set(creators))
    phase_cats = [p for p in ["S0", "S1", "S2"] if p in set(phases)]
    # drop first category for baseline
    creator_map = {c: i for i, c in enumerate(creator_cats[1:])}
    phase_map = {p: i for i, p in enumerate(phase_cats[1:])}

    rows = []
    for c, p in zip(creators, phases, strict=False):
        row = [1.0]  # intercept
        # phase dummies
        for ph in phase_cats[1:]:
            row.append(1.0 if p == ph else 0.0)
        # creator dummies
        for cr in creator_cats[1:]:
            row.append(1.0 if c == cr else 0.0)
        rows.append(row)

    X = np.array(rows, dtype=float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ coef
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # phase coefficients (relative to baseline phase)
    phase_coeffs = {}
    for i, ph in enumerate(phase_cats[1:], start=1):
        phase_coeffs[ph] = float(coef[i])

    return {"metric": metric, "n": len(y), "r2": float(r2), "phase_coeffs": phase_coeffs}


def main() -> int:
    parser = argparse.ArgumentParser(description="Full analysis package for #011")
    parser.add_argument("--features", default="outputs/features.csv", help="features.csv path")
    parser.add_argument("--manifest", default="outputs/final_manifest.csv", help="final_manifest.csv path")
    parser.add_argument("--out-dir", default="", help="output dir (default runs/<run_id>/outputs)")
    parser.add_argument("--run-id", default="", help="run id for archiving under runs/<run_id>")
    parser.add_argument("--min-n-per-phase", type=int, default=50, help="minimum samples per phase to run inference")
    parser.add_argument("--min-creators", type=int, default=20, help="minimum creators to run inference")
    parser.add_argument("--quick-summary-only", action="store_true", help="only output descriptive stats")
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
    features = pd.read_csv(features_path, dtype=str).fillna("")
    manifest = pd.read_csv(manifest_path, dtype=str).fillna("")

    # Merge
    key_feat = "video_id" if "video_id" in features.columns else "bvid"
    key_man = "unique_key" if "unique_key" in manifest.columns else "bvid"
    merge_cols = [key_man, "creator_id", "phase_base", "pubdate", "strict_ok", "fill_level", "fill_reason", "creator_group"]
    merge_cols = [c for c in merge_cols if c in manifest.columns]
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
    df["creator_id"] = df.get("creator_id", df.get("creator_group", "")).astype(str)

    # Numeric metrics
    metric_cols = [c for c in df.columns if c not in {key_feat, "series", "creator_group", "phase", "creator_id"}]
    metric_cols = [c for c in metric_cols if c in features.columns]
    numeric_cols = []
    for c in metric_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() >= 5:
            df[c] = s
            numeric_cols.append(c)

    phases = [p for p in ["S0", "S1", "S2"] if p in df["phase"].unique().tolist()]
    df_phase = df[df["phase"].isin(phases)].copy()

    # Phase counts
    phase_counts = df_phase["phase"].value_counts().reindex(phases).fillna(0).astype(int)
    phase_counts.to_csv(tables_dir / "phase_counts.csv", encoding="utf-8-sig")

    # Fill level distribution
    if "fill_level" in df_phase.columns:
        fill_counts = df_phase["fill_level"].value_counts().sort_index()
        fill_counts.to_csv(tables_dir / "fill_level_counts.csv", encoding="utf-8-sig")

    creator_count = df_phase["creator_id"].nunique() if "creator_id" in df_phase.columns else 0
    sample_structure = (
        df_phase.groupby("phase")
        .agg(n=("phase", "count"), creators=("creator_id", "nunique"))
        .reset_index()
    )
    sample_structure.to_csv(tables_dir / "sample_structure.csv", index=False, encoding="utf-8-sig")

    # quick summary only / insufficient sample
    insufficient = args.quick_summary_only
    if not insufficient:
        if len(phases) < 2:
            insufficient = True
        elif (phase_counts.min() if len(phase_counts) else 0) < int(args.min_n_per_phase):
            insufficient = True
        elif creator_count < int(args.min_creators):
            insufficient = True

    if insufficient:
        # descriptive stats by phase
        desc_rows = []
        for metric in numeric_cols:
            for p in phases:
                vals = pd.to_numeric(
                    df_phase.loc[df_phase["phase"] == p, metric], errors="coerce"
                ).dropna()
                if vals.empty:
                    continue
                desc_rows.append(
                    {
                        "phase": p,
                        "metric": metric,
                        "n": int(vals.shape[0]),
                        "mean": float(vals.mean()),
                        "median": float(vals.median()),
                        "std": float(vals.std(ddof=1) if vals.shape[0] > 1 else 0.0),
                        "min": float(vals.min()),
                        "max": float(vals.max()),
                        "iqr": float(vals.quantile(0.75) - vals.quantile(0.25)),
                    }
                )
        pd.DataFrame(desc_rows).to_csv(tables_dir / "phase_stats.csv", index=False, encoding="utf-8-sig")

        # fig01: phase counts
        import matplotlib.pyplot as plt

        fig1 = plt.figure(figsize=(6.0, 4.0), dpi=150)
        ax = fig1.add_subplot(111)
        ax.bar(phase_counts.index.tolist(), phase_counts.values.tolist(), color=["#1f77b4", "#ff7f0e", "#2ca02c"])
        ax.set_title("Phase counts")
        ax.set_xlabel("Phase")
        ax.set_ylabel("Count")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        fig1.tight_layout()
        fig1.savefig(plots_dir / "fig01_phase_counts.png")
        plt.close(fig1)

        report_lines = [
            f"# Analysis Report ({run_id})",
            "",
            "## INSUFFICIENT SAMPLE FOR INFERENCE",
            f"- min_n_per_phase: {args.min_n_per_phase}",
            f"- min_creators: {args.min_creators}",
            f"- phases: {', '.join(phases)}",
            f"- phase_counts: {phase_counts.to_dict()}",
            f"- creators: {creator_count}",
            "",
            f"- features_path: {features_path}",
            f"- manifest_path: {manifest_path}",
            f"- samples: {len(df_phase)}",
            f"- numeric_metrics: {', '.join(numeric_cols)}",
            "",
            "## Outputs",
            f"- tables: {tables_dir}",
            f"- plots: {plots_dir}",
        ]
        report_path = out_dir / "analysis_report.md"
        report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
        (out_dir / "analysis_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        return 0

    # ANOVA / Welch / Kruskal
    import scipy.stats as st  # type: ignore

    anova_rows = []
    pair_rows = []
    for metric in numeric_cols:
        groups = [df_phase.loc[df_phase["phase"] == p, metric].dropna().to_numpy(dtype=float) for p in phases]
        if len(groups) < 2:
            continue
        if min(len(g) for g in groups) < 3:
            continue
        f_stat, p_val = st.f_oneway(*groups)
        welch_f, df1, df2, welch_p = _welch_anova(groups)
        kruskal = st.kruskal(*groups)
        anova_rows.append(
            {
                "metric": metric,
                "anova_f": float(f_stat),
                "anova_p": float(p_val),
                "welch_f": float(welch_f),
                "welch_df1": float(df1),
                "welch_df2": float(df2),
                "welch_p": float(welch_p),
                "kruskal_h": float(kruskal.statistic),
                "kruskal_p": float(kruskal.pvalue),
            }
        )

        # pairwise
        phase_pairs = [("S0", "S1"), ("S0", "S2"), ("S1", "S2")]
        for a, b in phase_pairs:
            if a not in phases or b not in phases:
                continue
            ga = df_phase.loc[df_phase["phase"] == a, metric].dropna().to_numpy(dtype=float)
            gb = df_phase.loc[df_phase["phase"] == b, metric].dropna().to_numpy(dtype=float)
            if len(ga) < 3 or len(gb) < 3:
                continue
            t = st.ttest_ind(ga, gb, equal_var=False)
            mw = st.mannwhitneyu(ga, gb, alternative="two-sided")
            d = _cohens_d(ga, gb)
            ci_lo, ci_hi = _bootstrap_ci(ga, gb, n=300)
            pair_rows.append(
                {
                    "metric": metric,
                    "pair": f"{a} vs {b}",
                    "t_p": float(t.pvalue),
                    "mw_p": float(mw.pvalue),
                    "cohens_d": float(d),
                    "d_ci_low": ci_lo,
                    "d_ci_high": ci_hi,
                }
            )

    anova_df = pd.DataFrame(anova_rows)
    if not anova_df.empty:
        anova_df["anova_p_fdr"] = _bh_fdr(anova_df["anova_p"].tolist())
        anova_df["welch_p_fdr"] = _bh_fdr(anova_df["welch_p"].tolist())
        anova_df["kruskal_p_fdr"] = _bh_fdr(anova_df["kruskal_p"].tolist())
        anova_df.to_csv(tables_dir / "anova_summary.csv", index=False, encoding="utf-8-sig")

    pair_df = pd.DataFrame(pair_rows)
    if not pair_df.empty:
        pair_df["t_p_fdr"] = _bh_fdr(pair_df["t_p"].tolist())
        pair_df["mw_p_fdr"] = _bh_fdr(pair_df["mw_p"].tolist())
        pair_df.to_csv(tables_dir / "pairwise_tests.csv", index=False, encoding="utf-8-sig")

    # Fixed effects
    fe_rows = []
    for metric in numeric_cols:
        fe = _fixed_effects(df_phase, metric)
        fe_rows.append(fe)
    if fe_rows:
        fe_df = pd.DataFrame(
            [
                {
                    "metric": r.get("metric"),
                    "n": r.get("n"),
                    "r2": r.get("r2"),
                    "phase_coeffs": json.dumps(r.get("phase_coeffs", {}), ensure_ascii=False),
                }
                for r in fe_rows
            ]
        )
        fe_df.to_csv(tables_dir / "fixed_effects.csv", index=False, encoding="utf-8-sig")

    # Spearman correlation
    if numeric_cols:
        corr = df_phase[numeric_cols].corr(method="spearman")
        corr.to_csv(tables_dir / "spearman_corr.csv", encoding="utf-8-sig")

        # VIF
        vif_df = _vif(df_phase.dropna(subset=numeric_cols), numeric_cols)
        vif_df.to_csv(tables_dir / "vif.csv", index=False, encoding="utf-8-sig")

        # PCA
        try:
            from sklearn.decomposition import PCA  # type: ignore
            from sklearn.preprocessing import StandardScaler  # type: ignore

            X = df_phase[numeric_cols].fillna(0.0).to_numpy(dtype=float)
            Xs = StandardScaler().fit_transform(X)
            pca = PCA(n_components=min(5, Xs.shape[1]))
            comps = pca.fit_transform(Xs)
            pca_df = pd.DataFrame(comps, columns=[f"PC{i+1}" for i in range(comps.shape[1])])
            pca_df["phase"] = df_phase["phase"].values
            pca_df.to_csv(tables_dir / "pca_scores.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame(
                {"component": [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
                 "explained_variance_ratio": pca.explained_variance_ratio_}
            ).to_csv(tables_dir / "pca_variance.csv", index=False, encoding="utf-8-sig")
        except Exception:
            pass

    # Plots
    import matplotlib.pyplot as plt

    # fig01: phase counts
    fig1 = plt.figure(figsize=(6.0, 4.0), dpi=150)
    ax = fig1.add_subplot(111)
    ax.bar(phase_counts.index.tolist(), phase_counts.values.tolist(), color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax.set_title("Phase counts")
    ax.set_xlabel("Phase")
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(plots_dir / "fig01_phase_counts.png")
    plt.close(fig1)

    # fig02-04: boxplots for key metrics
    for idx, metric in enumerate([c for c in ["mattr", "mean_sent_len_chars", "connectives_total"] if c in df_phase.columns], start=2):
        data = [df_phase.loc[df_phase["phase"] == p, metric].dropna().tolist() for p in phases]
        fig = plt.figure(figsize=(6.4, 4.0), dpi=150)
        ax = fig.add_subplot(111)
        ax.boxplot(data, tick_labels=phases, showmeans=True)
        ax.set_title(f"{metric} by phase")
        ax.set_xlabel("Phase")
        ax.set_ylabel(metric)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots_dir / f"fig0{idx}_{metric}_box.png")
        plt.close(fig)

    # fig05: time trend (monthly)
    if "pubdate" in df_phase.columns:
        tmp = df_phase.copy()
        tmp["pubdate"] = pd.to_datetime(tmp["pubdate"], errors="coerce")
        tmp = tmp.dropna(subset=["pubdate"])
        if not tmp.empty and "mattr" in tmp.columns:
            tmp["month"] = tmp["pubdate"].dt.to_period("M").dt.to_timestamp()
            grp = tmp.groupby("month")["mattr"].mean().reset_index()
            fig = plt.figure(figsize=(6.8, 4.0), dpi=150)
            ax = fig.add_subplot(111)
            ax.plot(grp["month"], grp["mattr"], marker="o", linewidth=1.5)
            ax.set_title("MATTR time trend (monthly)")
            ax.set_xlabel("Month")
            ax.set_ylabel("MATTR")
            ax.grid(True, axis="y", linestyle="--", alpha=0.3)
            fig.autofmt_xdate()
            fig.tight_layout()
            fig.savefig(plots_dir / "fig05_mattr_time_trend.png")
            plt.close(fig)

    # Report
    sig_anova = 0
    sig_pair = 0
    effect_range = ""
    if not anova_df.empty:
        sig_anova = int((anova_df["anova_p_fdr"] < 0.05).sum())
    if not pair_df.empty:
        sig_pair = int((pair_df["t_p_fdr"] < 0.05).sum())
        if sig_pair > 0:
            d_vals = pair_df.loc[pair_df["t_p_fdr"] < 0.05, "cohens_d"].tolist()
            effect_range = f"{min(d_vals):.3f} ~ {max(d_vals):.3f}" if d_vals else ""

    report_lines = [
        f"# Analysis Report ({run_id})",
        "",
        f"- features_path: {features_path}",
        f"- manifest_path: {manifest_path}",
        f"- samples: {len(df_phase)}",
        f"- phases: {', '.join(phases)}",
        f"- numeric_metrics: {', '.join(numeric_cols)}",
        "",
        "## Key Results (FDR-controlled)",
        f"- ANOVA significant metrics (FDR<0.05): {sig_anova}",
        f"- Pairwise significant comparisons (FDR<0.05): {sig_pair}",
        f"- Effect size range (Cohen's d, significant pairs): {effect_range or 'n/a'}",
        "",
        "## Outputs",
        f"- tables: {tables_dir}",
        f"- plots: {plots_dir}",
    ]
    report_path = out_dir / "analysis_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    # Log
    (out_dir / "analysis_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
