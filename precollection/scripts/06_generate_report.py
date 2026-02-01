from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from pandas.errors import EmptyDataError

from lib.paths import ensure_dirs, get_project_paths


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=str).fillna("")
    except EmptyDataError:
        return pd.DataFrame()


def _read_jsonl_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except Exception:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return pd.DataFrame(rows).fillna("")


def _safe_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _df_to_md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    headers = [str(c) for c in df.columns.tolist()]
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in df.astype(str).values.tolist():
        lines.append("| " + " | ".join([str(x).replace("\n", " ").strip() for x in row]) + " |")
    return "\n".join(lines)


def _plot_connectors(features: pd.DataFrame, out_path: Path) -> None:
    if features.empty:
        return

    df = features.copy()
    df["connectors_per_1k_tokens"] = _safe_float(df["connectors_per_1k_tokens"])
    df = df.dropna(subset=["connectors_per_1k_tokens"])
    if df.empty:
        return

    stage_order = ["S0", "S1", "S2"]
    stage_to_x = {s: i for i, s in enumerate(stage_order)}
    df = df[df["actual_stage"].isin(stage_order)]
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    colors = {"fanshi_newanime": "#1f77b4", "shuianxiaoxi": "#d62728"}

    for series, g in df.groupby("series"):
        xs = [stage_to_x[s] for s in g["actual_stage"].tolist()]
        ys = g["connectors_per_1k_tokens"].tolist()
        # 轻微抖动避免重叠
        xs_j = [x + (0.06 if series == "fanshi_newanime" else -0.06) for x in xs]
        ax.scatter(xs_j, ys, label=series, alpha=0.85, s=32, color=colors.get(series, None))

        means = (
            g.groupby("actual_stage")["connectors_per_1k_tokens"]
            .mean()
            .reindex(stage_order)
            .tolist()
        )
        ax.plot(list(range(len(stage_order))), means, linewidth=2.0, alpha=0.6, color=colors.get(series, None))

    ax.set_xticks(list(range(len(stage_order))), stage_order)
    ax.set_xlabel("Stage")
    ax.set_ylabel("connectors_per_1k_tokens")
    ax.set_title("Connectors density by stage (pilot)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_download_status(manifest: pd.DataFrame, out_path: Path) -> None:
    if manifest.empty or "subtitle_status" not in manifest.columns:
        return
    counts = manifest["subtitle_status"].fillna("").astype(str).replace({"": "UNKNOWN"}).value_counts()
    if counts.empty:
        return

    fig, ax = plt.subplots(figsize=(6.0, 3.6), dpi=150)
    ax.bar(counts.index.tolist(), counts.values.tolist(), color="#4c78a8", alpha=0.9)
    ax.set_title("Subtitle download status (pilot)")
    ax.set_xlabel("status")
    ax.set_ylabel("count")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> int:
    paths = get_project_paths()
    ensure_dirs(paths)

    manifest_path = paths.data_raw_meta / "videos_manifest.csv"
    features_path = paths.analysis / "features.csv"
    download_log_path = paths.logs / "download.jsonl"

    manifest = _read_csv_if_exists(manifest_path)
    features = _read_csv_if_exists(features_path)
    download_log = _read_jsonl_if_exists(download_log_path)

    fig_path = paths.analysis_figures / "connectors_by_stage.png"
    _plot_connectors(features, fig_path)

    status_fig_path = paths.analysis_figures / "download_status.png"
    _plot_download_status(manifest, status_fig_path)

    lines: list[str] = []
    lines.append("# 预收集报告（pilot）\n")

    lines.append("## 分层核验（expected vs actual）\n")
    if manifest.empty:
        lines.append(f"- 缺少 manifest：`{manifest_path}`\n")
    else:
        cols = [
            "series",
            "creator_id",
            "bvid",
            "pub_date",
            "duration_sec",
            "page_duration",
            "expected_stage",
            "actual_stage",
            "stage_match",
            "needs_review",
            "subtitle_status",
            "subtitle_source",
        ]
        show = manifest[[c for c in cols if c in manifest.columns]].copy()
        lines.append(_df_to_md_table(show))
        lines.append("")

    lines.append("\n## 字幕下载状态汇总\n")
    if manifest.empty or "subtitle_status" not in manifest.columns:
        lines.append("- 未找到 subtitle_status（先运行 `scripts/02_download_subtitles.py`）\n")
    else:
        status_counts = manifest["subtitle_status"].value_counts(dropna=False).rename_axis("status").reset_index(name="count")
        lines.append(_df_to_md_table(status_counts))
        lines.append("")

        by_series = (
            manifest.groupby(["series", "subtitle_status"])
            .size()
            .reset_index(name="count")
            .sort_values(["series", "subtitle_status"])
        )
        lines.append("\n按 series 细分：\n")
        lines.append(_df_to_md_table(by_series))
        lines.append("")

    if not download_log.empty and "status" in download_log.columns:
        per_video = download_log[download_log.get("bvid", "").astype(str) != ""].copy()
        if not per_video.empty:
            lines.append("\n下载日志（download.jsonl）状态统计：\n")
            log_counts = per_video["status"].value_counts(dropna=False).rename_axis("status").reset_index(name="count")
            lines.append(_df_to_md_table(log_counts))
            lines.append("")

    lines.append("\n## 字幕来源统计（web_cc / ai_player）\n")
    if manifest.empty or ("subtitle_source" not in manifest.columns) or ("subtitle_status" not in manifest.columns):
        lines.append("- 缺少 subtitle_source（先运行 `scripts/02_download_subtitles.py`）\n")
    else:
        ok = manifest[manifest["subtitle_status"] == "OK"].copy()
        lines.append(
            "- 策略：优先网页 CC（`x/player/wbi/v2`）；AI 字幕（`x/player/v2`）存在串台/错配风险，默认不启用；为满足“12 条 BV 全部有字幕且一一对应”，提供本地 ASR（faster-whisper）兜底（`scripts/02_download_subtitles.py --asr-mode fallback/force`）。\n"
        )
        if ok.empty:
            lines.append("- OK 样本为 0：默认只取网页 CC 字幕（`x/player/wbi/v2`），如需回退 AI 字幕需手动加 `--allow-player-v2`。\n")
        else:
            src_counts = (
                ok["subtitle_source"]
                .fillna("")
                .replace({"": "UNKNOWN"})
                .value_counts()
                .rename_axis("subtitle_source")
                .reset_index(name="count")
            )
            lines.append(_df_to_md_table(src_counts))
            lines.append("")

    if not download_log.empty and "sha1" in download_log.columns:
        per_video = download_log[download_log.get("bvid", "").astype(str) != ""].copy()
        sha = per_video.get("sha1", "").astype(str)
        per_video = per_video[sha != ""].copy()
        if not per_video.empty:
            dup = (
                per_video.groupby("sha1")["bvid"]
                .apply(lambda s: ",".join(sorted(set([str(x) for x in s.tolist() if str(x).strip()]))))
                .reset_index(name="bvids")
            )
            dup["n"] = dup["bvids"].apply(lambda s: len([x for x in s.split(",") if x]))
            dup = dup[dup["n"] > 1].sort_values(["n", "sha1"], ascending=[False, True])
            if not dup.empty:
                lines.append("\n## 重复字幕告警（sha1 相同）\n")
                lines.append("以下 BV 下载到相同字幕内容，存在错配/缓存风险：\n")
                lines.append(_df_to_md_table(dup[["sha1", "n", "bvids"]]))
                lines.append("")

    lines.append("\n## 指标均值（series × stage）\n")
    numeric_cols = [
        "chars",
        "tokens",
        "sentences",
        "mean_sentence_chars",
        "punct_per_1k_tokens",
        "connectors_per_1k_tokens",
        "frame_markers_per_1k_tokens",
        "modality_per_1k_tokens",
    ]
    if not features_path.exists():
        lines.append(f"- 缺少 features：`{features_path}`（先运行 `scripts/05_compute_features.py`）\n")
    elif features.empty:
        lines.append("- features 为空（通常意味着字幕缺失或未生成 clean 文本）\n")
        empty_tbl = pd.DataFrame(columns=["series", "actual_stage", *numeric_cols])
        lines.append(_df_to_md_table(empty_tbl))
        lines.append("")
    else:
        df = features.copy()
        for c in numeric_cols:
            if c in df.columns:
                df[c] = _safe_float(df[c])
        agg = (
            df.groupby(["series", "actual_stage"])[[c for c in numeric_cols if c in df.columns]]
            .mean(numeric_only=True)
            .round(3)
            .reset_index()
            .sort_values(["series", "actual_stage"])
        )
        lines.append(_df_to_md_table(agg))
        lines.append("")

    lines.append("\n## 可视化\n")
    if status_fig_path.exists():
        lines.append(f"![download_status]({status_fig_path.relative_to(paths.analysis).as_posix()})\n")
    if fig_path.exists():
        lines.append(f"![connectors_by_stage]({fig_path.relative_to(paths.analysis).as_posix()})\n")
    if (not status_fig_path.exists()) and (not fig_path.exists()):
        lines.append("- 图未生成（manifest/features 缺失）\n")

    lines.append("\n## 风险提示（预收集必须记录）\n")
    if not manifest.empty and "subtitle_status" in manifest.columns:
        ok = manifest[manifest["subtitle_status"] == "OK"]
        missing = manifest[manifest["subtitle_status"] != "OK"]
        total = len(manifest)
        lines.append(f"- 字幕可得率：{len(ok)}/{total}（缺失：{len(missing)}）\n")
        if not missing.empty:
            bvs = ", ".join(missing["bvid"].tolist())
            lines.append(f"- 缺失样本 BV：{bvs}\n")
        if len(ok) == 0:
            lines.append("- 说明：脚本默认仅下载网页端可见的 CC 字幕（`x/player/wbi/v2`），并默认禁用 AI 字幕；如需回退到 AI 字幕（`x/player/v2`），请使用 `--allow-player-v2`。\n")
        else:
            if "subtitle_source" in ok.columns and ok["subtitle_source"].astype(str).str.startswith("asr").any():
                lines.append("- ASR 风险：本地语音识别可能引入分段/错字偏差，正式研究需在报告中明确标注 subtitle_source，并做敏感性分析。\n")

        if "subtitle_status" in manifest.columns:
            blocked_n = int((manifest["subtitle_status"] == "BLOCKED_OR_EMPTYLIST").sum())
            if blocked_n >= 10:
                lines.append("- 风控/接口异常风险：BLOCKED_OR_EMPTYLIST 数量较多，可能触发风控/需要降低速率/更换网络/更新 cookies。\n")

        if (not ok.empty) and ("subtitle_to_max_sec" in ok.columns) and ("page_duration" in ok.columns):
            tmp = ok.copy()
            tmp["subtitle_to_max_sec"] = _safe_float(tmp["subtitle_to_max_sec"])
            tmp["page_duration"] = _safe_float(tmp["page_duration"])
            tmp = tmp.dropna(subset=["subtitle_to_max_sec", "page_duration"])
            tmp = tmp[tmp["page_duration"] > 0]
            if not tmp.empty:
                tmp["coverage"] = tmp["subtitle_to_max_sec"] / tmp["page_duration"]
                low = tmp[tmp["coverage"] < 0.4].copy()
                if not low.empty:
                    bvs = ", ".join(low["bvid"].tolist())
                    lines.append(f"- 覆盖率风险：部分 OK 字幕仅覆盖视频前段（subtitle_to_max/page_duration < 0.4），可能影响指标代表性：{bvs}\n")

    if not manifest.empty and "subtitle_type" in manifest.columns:
        t = manifest["subtitle_type"].fillna("").astype(str)
        known = t[t != ""]
        if len(known) == 0:
            lines.append("- subtitle_type：未知（平台/API 未返回或未能解析）；报告中需提示“字幕类型偏差风险”。\n")
        else:
            type_counts = known.value_counts().rename_axis("subtitle_type").reset_index(name="count")
            lines.append("- subtitle_type 分布：\n")
            lines.append(_df_to_md_table(type_counts))
            lines.append("")

    if not manifest.empty and "needs_review" in manifest.columns:
        nr = manifest["needs_review"].astype(str).str.lower().isin(["true", "1", "yes"])
        lines.append(f"- 分层不匹配（needs_review=true）：{int(nr.sum())}/{len(manifest)}（本版本不替换样本，仅标记）\n")

    lines.append("- 长视频长度效应：正式研究建议考虑固定窗口（前 N 分钟/前 N 字）或加入长度控制变量。\n")
    lines.append("- 字幕类型偏差：若自动字幕/上传字幕比例随时间变化，可能造成“语言漂移”假象；需在正式研究中控制或分层。\n")

    out_md = paths.analysis / "pilot_report.md"
    out_md.write_text("\n".join(lines).strip() + "\n", encoding="utf-8-sig")
    print(f"Wrote: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
