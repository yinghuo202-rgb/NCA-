from __future__ import annotations

import argparse
import importlib.util
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from lib.paths import get_project_paths
from lib.run_utils import init_run_dirs, make_run_id


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _log(lines: list[str], message: str) -> None:
    line = f"[{_now()}] {message}"
    print(line)
    lines.append(line)


def _append_run_log(path: Path, new_lines: list[str]) -> None:
    existing = ""
    if path.exists():
        try:
            existing = path.read_text(encoding="utf-8-sig")
        except Exception:
            existing = path.read_text(encoding="utf-8", errors="ignore")
    content = existing.rstrip("\n")
    block = "\n".join(new_lines).strip()
    merged = (content + "\n" + block).strip() + "\n" if content else block + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(merged, encoding="utf-8-sig")


def _load_report06_helpers(log_lines: list[str]):
    """Reuse CSV reader helpers from scripts/06_generate_report.py if available."""
    report_path = Path(__file__).resolve().parent / "06_generate_report.py"
    if not report_path.exists():
        _log(log_lines, f"REUSE-SKIP: missing {report_path}, fall back to pandas.read_csv")
        return None

    spec = importlib.util.spec_from_file_location("_report06", report_path)
    if not spec or not spec.loader:
        _log(log_lines, "REUSE-SKIP: failed to build import spec for 06_generate_report.py")
        return None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]

    if not hasattr(module, "_read_csv_if_exists"):
        _log(log_lines, "REUSE-SKIP: 06_generate_report.py has no _read_csv_if_exists")
        return None

    _log(log_lines, f"REUSE: {report_path}._read_csv_if_exists")
    return module


def _read_csv(path: Path, report06_module, log_lines: list[str]) -> pd.DataFrame:
    if report06_module is not None:
        return report06_module._read_csv_if_exists(path)  # type: ignore[attr-defined]
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _phase_order() -> list[str]:
    return ["S0", "S1", "S2"]


def _plot_hist_by_phase(df: pd.DataFrame, *, phase_col: str, metric: str, out_path: Path) -> bool:
    if metric not in df.columns or phase_col not in df.columns:
        return False

    tmp = df[[phase_col, metric]].copy()
    tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce")
    tmp = tmp.dropna(subset=[metric])
    tmp = tmp[tmp[phase_col].isin(_phase_order())]
    if tmp.empty:
        return False

    fig, ax = plt.subplots(figsize=(6.8, 4.0), dpi=150)
    colors = {"S0": "#1f77b4", "S1": "#ff7f0e", "S2": "#2ca02c"}
    bins = 10

    for ph in _phase_order():
        vals = tmp.loc[tmp[phase_col] == ph, metric].tolist()
        if not vals:
            continue
        ax.hist(vals, bins=bins, density=True, alpha=0.35, label=ph, color=colors.get(ph))

    ax.set_title(f"Distribution of {metric} by phase")
    ax.set_xlabel(metric)
    ax.set_ylabel("Density")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False, title="phase")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return True


def _plot_ecdf(df: pd.DataFrame, *, phase_col: str, metric: str, out_path: Path) -> bool:
    if metric not in df.columns or phase_col not in df.columns:
        return False

    tmp = df[[phase_col, metric]].copy()
    tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce")
    tmp = tmp.dropna(subset=[metric])
    tmp = tmp[tmp[phase_col].isin(_phase_order())]
    if tmp.empty:
        return False

    fig, ax = plt.subplots(figsize=(6.8, 4.2), dpi=150)
    colors = {"S0": "#1f77b4", "S1": "#ff7f0e", "S2": "#2ca02c"}

    any_line = False
    for ph in _phase_order():
        vals = sorted(tmp.loc[tmp[phase_col] == ph, metric].tolist())
        n = len(vals)
        if n == 0:
            continue
        y = [(i + 1) / n for i in range(n)]
        ax.step(vals, y, where="post", label=ph, color=colors.get(ph))
        any_line = True

    if not any_line:
        plt.close(fig)
        return False

    ax.set_title(f"ECDF of {metric} by phase")
    ax.set_xlabel(metric)
    ax.set_ylabel("ECDF")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=False, title="phase")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return True


def _plot_scatter(
    df: pd.DataFrame,
    *,
    phase_col: str,
    x_col: str,
    y_col: str,
    out_path: Path,
    title: str,
) -> bool:
    if phase_col not in df.columns or x_col not in df.columns or y_col not in df.columns:
        return False

    tmp = df[[phase_col, x_col, y_col]].copy()
    tmp[x_col] = pd.to_numeric(tmp[x_col], errors="coerce")
    tmp[y_col] = pd.to_numeric(tmp[y_col], errors="coerce")
    tmp = tmp.dropna(subset=[x_col, y_col])
    tmp = tmp[tmp[phase_col].isin(_phase_order())]
    if tmp.empty:
        return False

    fig, ax = plt.subplots(figsize=(6.8, 4.2), dpi=150)
    colors = {"S0": "#1f77b4", "S1": "#ff7f0e", "S2": "#2ca02c"}
    for ph in _phase_order():
        g = tmp[tmp[phase_col] == ph]
        if g.empty:
            continue
        ax.scatter(g[x_col], g[y_col], label=ph, alpha=0.85, s=36, color=colors.get(ph))

    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=False, title="phase")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return True


def _plot_box_facet_creator(
    df: pd.DataFrame,
    *,
    phase_col: str,
    creator_col: str,
    metric: str,
    out_path: Path,
) -> bool:
    if phase_col not in df.columns or creator_col not in df.columns or metric not in df.columns:
        return False

    tmp = df[[phase_col, creator_col, metric]].copy()
    tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce")
    tmp = tmp.dropna(subset=[metric])
    tmp = tmp[tmp[phase_col].isin(_phase_order())]
    if tmp.empty:
        return False

    creators = [c for c in ["Creator_F", "Creator_M"] if c in tmp[creator_col].unique().tolist()]
    if len(creators) < 2:
        return False

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2), dpi=150, sharey=True)
    fig.suptitle(f"{metric} by phase (facet by creator_group)", y=1.02)

    for ax, cr in zip(axes, creators, strict=False):
        g = tmp[tmp[creator_col] == cr]
        data = [g.loc[g[phase_col] == ph, metric].tolist() for ph in _phase_order()]
        ax.boxplot(data, tick_labels=_phase_order(), showmeans=True)
        ax.set_title(cr)
        ax.set_xlabel("phase")
        ax.set_ylabel(metric)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_connectives_composition(
    df: pd.DataFrame,
    *,
    phase_col: str,
    out_path: Path,
) -> tuple[bool, list[str]]:
    if phase_col not in df.columns:
        return False, []

    # Heuristic: any numeric columns starting with "connectives_" excluding total.
    candidates = [
        c
        for c in df.columns
        if c.startswith("connectives_")
        and c not in {"connectives_total"}
    ]
    if not candidates:
        return False, []

    tmp = df[[phase_col, *candidates]].copy()
    for c in candidates:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce")

    # Keep columns with at least one valid number.
    keep = [c for c in candidates if tmp[c].notna().any()]
    if len(keep) < 2:
        return False, keep

    tmp = tmp[tmp[phase_col].isin(_phase_order())]
    if tmp.empty:
        return False, keep

    sums = tmp.groupby(phase_col)[keep].sum(numeric_only=True).reindex(_phase_order()).fillna(0.0)
    denom = sums.sum(axis=1).replace({0.0: 1.0})
    ratios = sums.div(denom, axis=0)

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    bottom = [0.0] * len(ratios.index)
    colors = plt.cm.tab20.colors

    for i, c in enumerate(keep):
        vals = ratios[c].tolist()
        ax.bar(ratios.index.tolist(), vals, bottom=bottom, label=c, color=colors[i % len(colors)], alpha=0.9)
        bottom = [b + v for b, v in zip(bottom, vals, strict=False)]

    ax.set_title("Connective type composition by phase")
    ax.set_xlabel("phase")
    ax.set_ylabel("Proportion")
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return True, keep


def _plot_corr_heatmap(
    df: pd.DataFrame,
    *,
    out_path: Path,
    exclude_cols: set[str],
) -> tuple[bool, list[str]]:
    if df.empty:
        return False, []

    work = df.copy()
    cols = [c for c in work.columns if c not in exclude_cols]
    if not cols:
        return False, []

    numeric: dict[str, pd.Series] = {}
    for c in cols:
        s = pd.to_numeric(work[c], errors="coerce")
        if s.notna().sum() >= 2:
            numeric[c] = s

    if len(numeric) < 2:
        return False, list(numeric.keys())

    num_df = pd.DataFrame(numeric)
    corr = num_df.corr(method="spearman")

    labels = corr.columns.tolist()
    mat = corr.to_numpy()

    fig, ax = plt.subplots(figsize=(0.55 * len(labels) + 3.0, 0.55 * len(labels) + 2.6), dpi=150)
    im = ax.imshow(mat, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_title("Spearman correlation heatmap")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7, color="black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return True, labels


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate extra descriptive plots from outputs/features.csv")
    parser.add_argument("--features", default="outputs/features.csv", help="path to features.csv")
    parser.add_argument("--master", default="outputs/master.csv", help="optional master.csv for creator_group/length_tokens")
    parser.add_argument("--out-dir", default="outputs/plots_extra", help="output directory for plots")
    parser.add_argument("--run-id", default="", help="run id for archiving under runs/<run_id>")
    args = parser.parse_args()

    paths = get_project_paths()
    root = paths.root

    log_lines: list[str] = []
    run_id = make_run_id(args.run_id or None)
    run_dirs = init_run_dirs(root, run_id)
    _log(log_lines, "Start extra descriptive plots")

    # Self-check: reuse existing scripts where possible.
    _log(log_lines, "Self-check (reuse/new)")
    _log(log_lines, "FOUND: scripts/06_generate_report.py (features.csv reader + plotting patterns)")
    _log(log_lines, "FOUND: scripts/07_v1_analysis.py (produces outputs/features.csv + outputs/master.csv)")
    _log(log_lines, "NEW: scripts/08_plots_extra.py (extra plots into outputs/plots_extra)")

    report06 = _load_report06_helpers(log_lines)

    features_path = (root / args.features).resolve()
    features = _read_csv(features_path, report06, log_lines).fillna("")
    _log(log_lines, f"Loaded features: {features_path} (rows={len(features)}, cols={len(features.columns)})")
    _log(log_lines, f"features.csv columns: {features.columns.tolist()}")

    if features.empty:
        _log(log_lines, "ERR: features.csv is empty; skip plot generation")
        _append_run_log((root / "outputs" / "run_log.txt").resolve(), log_lines)
        return 2

    # Column mapping +补齐：尽量从 features.csv 取，缺失时用 master.csv 补齐。
    col_map: dict[str, str] = {}
    for canonical, candidates in {
        "video_id": ["video_id", "bvid"],
        "phase": ["phase", "actual_stage", "stage"],
        "creator_group": ["creator_group", "creator_id", "creator"],
        "length_tokens": ["length_tokens", "tokens", "clean_tokens"],
    }.items():
        for c in candidates:
            if c in features.columns:
                col_map[canonical] = c
                break

    if "video_id" not in col_map:
        _log(log_lines, "ERR: cannot find video identifier column (video_id/bvid).")
        _append_run_log((root / "outputs" / "run_log.txt").resolve(), log_lines)
        return 2

    # If creator_group / length_tokens missing, join from master.csv
    need_master = ("creator_group" not in col_map) or ("length_tokens" not in col_map) or ("phase" not in col_map)
    if need_master:
        master_path = (root / args.master).resolve()
        if master_path.exists():
            try:
                master = pd.read_csv(
                    master_path,
                    dtype=str,
                    usecols=["video_id", "creator_group", "length_tokens", "phase"],
                ).fillna("")
                features = features.merge(
                    master,
                    how="left",
                    left_on=col_map["video_id"],
                    right_on="video_id",
                    suffixes=("", "_from_master"),
                )
                if col_map["video_id"] != "video_id" and "video_id" in features.columns:
                    # Avoid duplicate id columns after later canonical renaming.
                    features = features.drop(columns=["video_id"])
                _log(log_lines, f"Joined master for missing columns: {master_path}")
            except Exception as e:
                _log(log_lines, f"WARN: failed to join master.csv ({master_path}): {e}")
        else:
            _log(log_lines, f"WARN: master.csv not found: {master_path} (cannot backfill creator_group/length_tokens)")

        # Refresh mapping after join.
        for canonical, candidates in {
            "phase": ["phase", "actual_stage", "stage"],
            "creator_group": ["creator_group", "creator_id", "creator"],
            "length_tokens": ["length_tokens", "tokens", "clean_tokens"],
        }.items():
            if canonical in col_map:
                continue
            for c in candidates:
                if c in features.columns:
                    col_map[canonical] = c
                    break

    _log(log_lines, f"Column mapping: {col_map}")

    # Normalize to canonical column names for plotting.
    df = features.copy()
    df = df.rename(
        columns={
            col_map.get("video_id", "video_id"): "video_id",
            col_map.get("phase", "phase"): "phase",
            col_map.get("creator_group", "creator_group"): "creator_group",
            col_map.get("length_tokens", "length_tokens"): "length_tokens",
        }
    )

    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    generated: list[str] = []
    skipped: list[str] = []

    # 1) Distributions (hist by phase)
    for metric in ["mattr", "mean_sent_len_chars", "connectives_total", "comma_period_ratio"]:
        out_path = out_dir / f"dist_{metric}_hist.png"
        ok = _plot_hist_by_phase(df, phase_col="phase", metric=metric, out_path=out_path)
        if ok:
            generated.append(out_path.name)
        else:
            skipped.append(f"dist_{metric}_hist (missing column or empty)")

    # 2) ECDF
    for metric in ["mattr", "connectives_total"]:
        out_path = out_dir / f"ecdf_{metric}.png"
        ok = _plot_ecdf(df, phase_col="phase", metric=metric, out_path=out_path)
        if ok:
            generated.append(out_path.name)
        else:
            skipped.append(f"ecdf_{metric} (missing column or empty)")

    # 3) Length vs metrics scatter
    if "length_tokens" in df.columns and df["length_tokens"].astype(str).str.strip().ne("").any():
        out_path = out_dir / "scatter_length_tokens_vs_mean_sent_len_chars.png"
        ok = _plot_scatter(
            df,
            phase_col="phase",
            x_col="length_tokens",
            y_col="mean_sent_len_chars",
            out_path=out_path,
            title="Length effect check: length_tokens vs mean_sent_len_chars",
        )
        if ok:
            generated.append(out_path.name)
        else:
            skipped.append("scatter_length_tokens_vs_mean_sent_len_chars (missing column or empty)")

        out_path = out_dir / "scatter_length_tokens_vs_connectives_total.png"
        ok = _plot_scatter(
            df,
            phase_col="phase",
            x_col="length_tokens",
            y_col="connectives_total",
            out_path=out_path,
            title="Length effect check: length_tokens vs connectives_total",
        )
        if ok:
            generated.append(out_path.name)
        else:
            skipped.append("scatter_length_tokens_vs_connectives_total (missing column or empty)")
    else:
        skipped.append("scatter_length_* (length_tokens missing)")

    # 4) Creator-faceted boxplots
    for metric in ["mattr", "mean_sent_len_chars", "connectives_total", "comma_period_ratio"]:
        out_path = out_dir / f"box_facet_creator_group_{metric}.png"
        ok = _plot_box_facet_creator(
            df,
            phase_col="phase",
            creator_col="creator_group",
            metric=metric,
            out_path=out_path,
        )
        if ok:
            generated.append(out_path.name)
        else:
            skipped.append(f"box_facet_creator_group_{metric} (missing columns or insufficient creators)")

    # 5) Connective type composition (optional)
    out_path = out_dir / "connectives_composition_stacked.png"
    ok, used_cols = _plot_connectives_composition(df, phase_col="phase", out_path=out_path)
    if ok:
        generated.append(out_path.name)
        _log(log_lines, f"Connective type columns used: {used_cols}")
    else:
        skipped.append("connectives_composition_stacked (no connective type columns)")

    # 6) Correlation heatmap (Spearman)
    exclude = {"video_id", "phase", "creator_group"}
    ok, used_cols = _plot_corr_heatmap(df, out_path=out_dir / "corr_heatmap_spearman.png", exclude_cols=exclude)
    if ok:
        generated.append("corr_heatmap_spearman.png")
        _log(log_lines, f"Correlation heatmap columns (Spearman): {used_cols}")
    else:
        skipped.append("corr_heatmap_spearman (not enough numeric columns)")

    _log(log_lines, f"Wrote plots to: {out_dir}")
    if generated:
        _log(log_lines, f"Generated ({len(generated)}): {generated}")
    if skipped:
        _log(log_lines, f"Skipped ({len(skipped)}): {skipped}")

    # Append to outputs/run_log.txt
    run_log_path = (root / "outputs" / "run_log.txt").resolve()
    _append_run_log(run_log_path, log_lines)
    # archive plots
    try:
        import shutil

        dst = run_dirs.outputs / "plots_extra"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(out_dir, dst)
        _append_run_log(run_dirs.logs / "run_log_plots_extra.txt", log_lines)
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
