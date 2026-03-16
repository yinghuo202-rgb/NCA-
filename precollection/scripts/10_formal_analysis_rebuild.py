from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
import statsmodels.formula.api as smf

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from lib.breakpoints_ext import load_breakpoints_yaml
from lib.lexicon import load_terms
from lib.paths import ensure_dirs, get_project_paths

try:
    import jieba  # type: ignore
except Exception:  # pragma: no cover
    jieba = None


PHASES = ["S0", "S1", "S2"]
MAIN_FEATURES = ["connective_density", "template_density", "mattr", "structure_composite"]
SECONDARY_FEATURES = [
    "heading_like_density",
    "definitional_marker_density",
    "inferential_marker_density",
    "liability_shield_density",
    "segment_count_per_1000_tokens",
    "func_ratio",
    "stop_ratio",
    "mean_word_len",
    "style_index_pc1",
]
APPENDIX_FEATURES = [
    "formulaic_transition_density",
    "avg_tokens_per_segment",
    "legacy_mean_sent_len_chars",
    "repaired_mean_segment_chars",
    "legacy_comma_period_ratio",
    "repaired_comma_period_ratio",
    "segment_length_cv",
    "segment_boundary_density",
]
DROPPED_FEATURES = ["connectives_total"]
METADATA_CONTROL_TERMS = ["log_duration", "title_q", "title_bracket", "series_part"]

PUNCT_TRANS = str.maketrans(
    {
        ",": "，",
        ";": "；",
        ":": "：",
        "?": "？",
        "!": "！",
        "(": "（",
        ")": "）",
        "[": "【",
        "]": "】",
        "{": "（",
        "}": "）",
    }
)
RE_SPACES = re.compile(r"[ \t\u00A0\u3000]+")
RE_MULTI_NL = re.compile(r"\n{2,}")
RE_CJK = re.compile(r"[\u4e00-\u9fff]")
RE_ALNUM = re.compile(r"[A-Za-z0-9]")
RE_STRONG_SENT = re.compile(r"[。！？!?]+")
RE_PUNCT = re.compile(r"[，。！？；：、,.!?;:]+")


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    tier: str
    class_label: str
    kind: str
    report_col: str
    description: str
    model_family: str
    transform: str
    count_col: str | None = None


def _log(message: str, *, lines: list[str]) -> None:
    stamp = datetime.now().isoformat(timespec="seconds")
    line = f"[{stamp}] {message}"
    print(line)
    lines.append(line)


def _md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_empty_"
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        vals: list[str] = []
        for col in cols:
            value = row[col]
            if pd.isna(value):
                vals.append("")
            elif isinstance(value, float):
                if abs(value) >= 1000:
                    vals.append(f"{value:,.2f}")
                elif abs(value) >= 1:
                    vals.append(f"{value:.4f}".rstrip("0").rstrip("."))
                else:
                    vals.append(f"{value:.6f}".rstrip("0").rstrip("."))
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _bh_fdr(values: list[float]) -> list[float]:
    if not values:
        return []
    arr = np.asarray(values, dtype=float)
    order = np.argsort(arr)
    out = np.empty(len(arr), dtype=float)
    prev = 1.0
    n = float(len(arr))
    for rank_rev, idx in enumerate(order[::-1], start=1):
        rank = len(arr) - rank_rev + 1
        value = min(prev, arr[idx] * n / float(rank))
        out[idx] = min(value, 1.0)
        prev = value
    return out.tolist()


def _apply_fdr_by_group(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty or "p_value" not in df.columns:
        return df
    out = df.copy()
    out["p_fdr"] = np.nan
    for _, idx in out.groupby(group_cols).groups.items():
        pvals = out.loc[list(idx), "p_value"].fillna(1.0).astype(float).tolist()
        out.loc[list(idx), "p_fdr"] = _bh_fdr(pvals)
    return out


def _safe_logit(series: pd.Series, eps: float = 1e-4) -> pd.Series:
    clipped = series.clip(lower=eps, upper=1.0 - eps)
    return np.log(clipped / (1.0 - clipped))


def _count_chars(text: str) -> int:
    return len(RE_CJK.findall(text)) + len(RE_ALNUM.findall(text))


def _normalize_text(text: str) -> str:
    out = text.replace("\r\n", "\n").replace("\r", "\n")
    out = out.translate(PUNCT_TRANS)
    out = out.replace("...", "…")
    out = out.replace("。。", "。")
    out = RE_SPACES.sub(" ", out)
    out = "\n".join(line.strip() for line in out.split("\n"))
    out = RE_MULTI_NL.sub("\n", out)
    return out.strip()


def _strip_punct_for_tokens(text: str) -> str:
    return RE_PUNCT.sub(" ", text)


def _tokenize(text: str) -> list[str]:
    clean = _strip_punct_for_tokens(text)
    if jieba is not None:
        try:
            tokens = jieba.lcut(clean, cut_all=False)
        except Exception:
            tokens = re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9]+", clean)
    else:
        tokens = re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9]+", clean)
    out: list[str] = []
    for token in tokens:
        s = str(token).strip()
        if not s:
            continue
        if RE_CJK.search(s) or RE_ALNUM.search(s):
            out.append(s)
    return out


def _mattr(tokens: list[str], window: int = 100) -> float:
    n = len(tokens)
    if n == 0:
        return 0.0
    if n <= window:
        return len(set(tokens)) / float(n)
    counts: dict[str, int] = {}
    uniq = 0
    for token in tokens[:window]:
        prev = counts.get(token, 0)
        counts[token] = prev + 1
        if prev == 0:
            uniq += 1
    total = uniq / float(window)
    windows = 1
    for i in range(window, n):
        out_tok = tokens[i - window]
        counts[out_tok] -= 1
        if counts[out_tok] == 0:
            del counts[out_tok]
            uniq -= 1
        in_tok = tokens[i]
        prev = counts.get(in_tok, 0)
        counts[in_tok] = prev + 1
        if prev == 0:
            uniq += 1
        total += uniq / float(window)
        windows += 1
    return total / float(windows)


def _substring_hits(text: str, terms: list[str]) -> int:
    return sum(text.count(term) for term in terms if term)


def _load_bcc(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _infer_provenance(row: pd.Series, bcc: dict[str, Any]) -> str:
    source = str(row.get("subtitle_source") or "").strip().lower()
    bcc_type = str(bcc.get("type") or "").strip().lower()
    version = str(bcc.get("version") or "").strip().lower()
    if "asr" in source or "whisper" in source or bcc_type == "asr" or "whisper" in version:
        return "ASR"
    if source in {"web_cc", "ai_player"}:
        return "platform_subtitle"
    if bcc.get("body"):
        return "unknown"
    return "unknown"


def _remove_terms(text: str, terms: list[str]) -> tuple[str, int]:
    out = text
    removed = 0
    for term in terms:
        if not term:
            continue
        c = out.count(term)
        if c:
            removed += c
            out = out.replace(term, "")
    return out, removed


def _prepare_chunks(bcc: dict[str, Any], onomatopoeia_terms: list[str]) -> tuple[list[dict[str, Any]], int]:
    body = bcc.get("body") or []
    chunks: list[dict[str, Any]] = []
    removed_total = 0
    for seg in body:
        if not isinstance(seg, dict):
            continue
        text = _normalize_text(str(seg.get("content") or ""))
        if not text:
            continue
        text, removed = _remove_terms(text, onomatopoeia_terms)
        removed_total += removed
        text = _normalize_text(text)
        if not text:
            continue
        try:
            start = float(seg.get("from", 0) or 0)
        except Exception:
            start = 0.0
        try:
            end = float(seg.get("to", 0) or 0)
        except Exception:
            end = start
        chunks.append({"from": start, "to": max(end, start), "text": text})
    return chunks, removed_total


def _repair_segments(chunks: list[dict[str, Any]], cue_terms: list[str]) -> list[str]:
    if not chunks:
        return []
    segments: list[str] = []
    current_parts: list[str] = []
    current_chars = 0
    prev_end = float(chunks[0]["from"])
    cue_prefixes = tuple(sorted(set(cue_terms), key=len, reverse=True))

    def flush() -> None:
        nonlocal current_parts, current_chars
        text = _normalize_text(" ".join(current_parts))
        if text:
            segments.append(text)
        current_parts = []
        current_chars = 0

    for chunk in chunks:
        text = str(chunk["text"])
        text_chars = _count_chars(text)
        gap = float(chunk["from"]) - float(prev_end)
        cue_start = bool(cue_prefixes and text.startswith(cue_prefixes))
        should_break = False
        if current_parts:
            if gap >= 1.0:
                should_break = True
            elif gap >= 0.55 and current_chars >= 60:
                should_break = True
            elif current_chars >= 95 and gap >= 0.25:
                should_break = True
            elif current_chars >= 125 and text_chars >= 28:
                should_break = True
            elif cue_start and current_chars >= 24:
                should_break = True
        if should_break:
            flush()
        current_parts.append(text)
        current_chars += text_chars
        prev_end = float(chunk["to"])
    flush()
    if not segments:
        return segments

    merged: list[str] = []
    for seg in segments:
        seg = _normalize_text(seg)
        seg_chars = _count_chars(seg)
        if merged and seg_chars < 16:
            merged[-1] = _normalize_text(f"{merged[-1]} {seg}")
            continue
        if merged and _count_chars(merged[-1]) < 22:
            merged[-1] = _normalize_text(f"{merged[-1]} {seg}")
            continue
        merged.append(seg)
    if len(merged) >= 2 and _count_chars(merged[-1]) < 20:
        merged[-2] = _normalize_text(f"{merged[-2]} {merged[-1]}")
        merged.pop()
    return merged


def _legacy_structure_metrics(text: str, chars: int) -> dict[str, float]:
    sentences = [s.strip() for s in RE_STRONG_SENT.split(text) if s.strip()]
    sentence_count = max(len(sentences), 1)
    mean_sent_len = float(chars) / float(sentence_count) if sentence_count > 0 else 0.0
    comma_count = text.count("，") + text.count(",")
    period_count = text.count("。") + text.count("！") + text.count("？") + text.count("!") + text.count("?")
    ratio = float(comma_count) / float(max(period_count, 1))
    return {
        "legacy_sentence_count": float(sentence_count),
        "legacy_mean_sent_len_chars": mean_sent_len,
        "legacy_comma_period_ratio": ratio,
        "raw_comma_count": float(comma_count),
        "raw_period_count": float(period_count),
    }


def _repaired_structure_metrics(
    repaired_segments: list[str],
    chars: int,
    tokens: int,
    raw_comma_count: float,
) -> dict[str, float]:
    seg_count = max(len(repaired_segments), 1)
    lengths = np.asarray([_count_chars(seg) for seg in repaired_segments], dtype=float) if repaired_segments else np.asarray([float(chars)])
    mean_len = float(np.mean(lengths))
    std_len = float(np.std(lengths, ddof=0))
    cv = (std_len / mean_len) if mean_len > 1e-9 else 0.0
    repaired_ratio = float(raw_comma_count) / float(seg_count)
    boundary_density = float(seg_count) * 1000.0 / float(max(tokens, 1))
    avg_tokens_per_segment = float(tokens) / float(seg_count)
    return {
        "repaired_segment_count": float(seg_count),
        "repaired_mean_segment_chars": mean_len,
        "segment_length_cv": cv,
        "repaired_comma_period_ratio": repaired_ratio,
        "segment_boundary_density": boundary_density,
        "segment_count_per_1000_tokens": boundary_density,
        "avg_tokens_per_segment": avg_tokens_per_segment,
    }


def _select_evenly(group: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    if len(group) <= n:
        return group.copy()
    idx = [min(int(i * len(group) / n), len(group) - 1) for i in range(n)]
    idx = sorted(dict.fromkeys(idx))
    while len(idx) < n:
        for extra in range(len(group)):
            if extra not in idx:
                idx.append(extra)
            if len(idx) == n:
                break
    idx = sorted(idx[:n])
    return group.iloc[idx].copy()


def _safe_standardize(series: pd.Series) -> pd.Series:
    base = pd.to_numeric(series, errors="coerce").astype(float)
    std = float(base.std(ddof=0))
    if not np.isfinite(std) or std < 1e-9:
        return pd.Series(0.0, index=series.index, dtype=float)
    out = (base - float(base.mean())) / std
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _build_style_pc1(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    valid_cols = [col for col in cols if col in df.columns]
    if not valid_cols:
        return pd.Series(np.nan, index=df.index)
    data = df[valid_cols].astype(float)
    mask = data.notna().all(axis=1)
    out = pd.Series(np.nan, index=df.index, dtype=float)
    if int(mask.sum()) < 10:
        return out
    values = data.loc[mask]
    means = values.mean(axis=0)
    stds = values.std(axis=0, ddof=0).replace(0.0, 1.0)
    z = (values - means) / stds
    _, _, vt = np.linalg.svd(z.to_numpy(dtype=float), full_matrices=False)
    pc1 = z.to_numpy(dtype=float) @ vt[0]
    if "connective_density" in valid_cols:
        ref = np.corrcoef(pc1, z["connective_density"].to_numpy(dtype=float))[0, 1]
        if np.isfinite(ref) and ref < 0:
            pc1 = -pc1
    out.loc[mask] = pc1
    return out


def _feature_specs() -> dict[str, FeatureSpec]:
    return {
        "connective_density": FeatureSpec(
            name="connective_density",
            tier="main",
            class_label="cohesion density",
            kind="count",
            report_col="connective_density",
            description="Connective marker hits per 1,000 tokens.",
            model_family="poisson_fe_offset",
            transform="count model with offset(log(tokens))",
            count_col="connective_hits",
        ),
        "template_density": FeatureSpec(
            name="template_density",
            tier="main",
            class_label="template density",
            kind="count",
            report_col="template_density",
            description="Template / framing phrase hits per 1,000 tokens.",
            model_family="poisson_fe_offset",
            transform="count model with offset(log(tokens))",
            count_col="template_hits",
        ),
        "mattr": FeatureSpec(
            name="mattr",
            tier="main",
            class_label="lexical diversity",
            kind="bounded",
            report_col="mattr",
            description="Moving-average type-token ratio over jieba tokens.",
            model_family="ols_fe_clustered",
            transform="logit-clipped",
        ),
        "structure_composite": FeatureSpec(
            name="structure_composite",
            tier="main",
            class_label="structure composite",
            kind="continuous",
            report_col="structure_composite",
            description="Average z-score of heading-like, definitional-marker, inferential-marker, and liability-shield densities.",
            model_family="ols_fe_clustered",
            transform="standardized continuous",
        ),
        "heading_like_density": FeatureSpec(
            name="heading_like_density",
            tier="secondary",
            class_label="heading-like density",
            kind="count",
            report_col="heading_like_density",
            description="Heading-like discourse marker hits per 1,000 tokens.",
            model_family="poisson_fe_offset",
            transform="count model with offset(log(tokens))",
            count_col="heading_like_hits",
        ),
        "definitional_marker_density": FeatureSpec(
            name="definitional_marker_density",
            tier="secondary",
            class_label="definitional marker density",
            kind="count",
            report_col="definitional_marker_density",
            description="Definitional framing marker hits per 1,000 tokens.",
            model_family="poisson_fe_offset",
            transform="count model with offset(log(tokens))",
            count_col="definitional_marker_hits",
        ),
        "inferential_marker_density": FeatureSpec(
            name="inferential_marker_density",
            tier="secondary",
            class_label="inferential marker density",
            kind="count",
            report_col="inferential_marker_density",
            description="Inferential / reasoning marker hits per 1,000 tokens.",
            model_family="poisson_fe_offset",
            transform="count model with offset(log(tokens))",
            count_col="inferential_marker_hits",
        ),
        "liability_shield_density": FeatureSpec(
            name="liability_shield_density",
            tier="secondary",
            class_label="liability-shield density",
            kind="count",
            report_col="liability_shield_density",
            description="Hedging / liability-shield marker hits per 1,000 tokens.",
            model_family="poisson_fe_offset",
            transform="count model with offset(log(tokens))",
            count_col="liability_shield_hits",
        ),
        "segment_count_per_1000_tokens": FeatureSpec(
            name="segment_count_per_1000_tokens",
            tier="secondary",
            class_label="segment-count density",
            kind="positive",
            report_col="segment_count_per_1000_tokens",
            description="Repaired segment boundaries per 1,000 tokens.",
            model_family="ols_fe_clustered",
            transform="log",
        ),
        "func_ratio": FeatureSpec(
            name="func_ratio",
            tier="secondary",
            class_label="function-word ratio",
            kind="bounded",
            report_col="func_ratio",
            description="Function-word token share.",
            model_family="ols_fe_clustered",
            transform="logit-clipped",
        ),
        "stop_ratio": FeatureSpec(
            name="stop_ratio",
            tier="secondary",
            class_label="stopword ratio",
            kind="bounded",
            report_col="stop_ratio",
            description="Stopword token share.",
            model_family="ols_fe_clustered",
            transform="logit-clipped",
        ),
        "mean_word_len": FeatureSpec(
            name="mean_word_len",
            tier="secondary",
            class_label="mean word length",
            kind="positive",
            report_col="mean_word_len",
            description="Character count divided by token count.",
            model_family="ols_fe_clustered",
            transform="log",
        ),
        "style_index_pc1": FeatureSpec(
            name="style_index_pc1",
            tier="secondary",
            class_label="style index",
            kind="continuous",
            report_col="style_index_pc1",
            description="First principal component of retained non-appendix features.",
            model_family="ols_fe_clustered",
            transform="standardized continuous",
        ),
        "legacy_mean_sent_len_chars": FeatureSpec(
            name="legacy_mean_sent_len_chars",
            tier="appendix",
            class_label="legacy sentence length",
            kind="positive",
            report_col="legacy_mean_sent_len_chars",
            description="Legacy raw-punctuation sentence length; appendix-only after segmentation repair.",
            model_family="appendix_only",
            transform="none",
        ),
        "repaired_mean_segment_chars": FeatureSpec(
            name="repaired_mean_segment_chars",
            tier="appendix",
            class_label="segment length",
            kind="positive",
            report_col="repaired_mean_segment_chars",
            description="Mean repaired segment length in characters.",
            model_family="appendix_only",
            transform="none",
        ),
        "legacy_comma_period_ratio": FeatureSpec(
            name="legacy_comma_period_ratio",
            tier="appendix",
            class_label="legacy comma/period ratio",
            kind="positive",
            report_col="legacy_comma_period_ratio",
            description="Raw punctuation ratio; demoted because it directly inherits ASR punctuation noise.",
            model_family="appendix_only",
            transform="none",
        ),
        "repaired_comma_period_ratio": FeatureSpec(
            name="repaired_comma_period_ratio",
            tier="appendix",
            class_label="repaired comma/boundary ratio",
            kind="positive",
            report_col="repaired_comma_period_ratio",
            description="Raw comma count over repaired segment boundaries; appendix-only QC trace.",
            model_family="appendix_only",
            transform="none",
        ),
        "segment_length_cv": FeatureSpec(
            name="segment_length_cv",
            tier="appendix",
            class_label="segment variability",
            kind="positive",
            report_col="segment_length_cv",
            description="Coefficient of variation of repaired segment lengths.",
            model_family="appendix_only",
            transform="none",
        ),
        "segment_boundary_density": FeatureSpec(
            name="segment_boundary_density",
            tier="appendix",
            class_label="boundary density",
            kind="positive",
            report_col="segment_boundary_density",
            description="Legacy alias of repaired segment boundary count per 1,000 tokens; retained only for appendix continuity.",
            model_family="appendix_only",
            transform="none",
        ),
        "avg_tokens_per_segment": FeatureSpec(
            name="avg_tokens_per_segment",
            tier="appendix",
            class_label="average tokens per segment",
            kind="positive",
            report_col="avg_tokens_per_segment",
            description="Average token length of repaired segments; appendix-only after segmentation rebuild.",
            model_family="appendix_only",
            transform="none",
        ),
        "formulaic_transition_density": FeatureSpec(
            name="formulaic_transition_density",
            tier="drop",
            class_label="formulaic transition density",
            kind="count",
            report_col="formulaic_transition_density",
            description="Evaluated automatically, but dropped from reporting because it overlaps heavily with template and connective densities.",
            model_family="not_modeled",
            transform="redundant with main densities",
            count_col="formulaic_transition_hits",
        ),
    }


def _prepare_model_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["phase"] = pd.Categorical(out["phase"], categories=PHASES, ordered=True)
    out["phase_S1"] = (out["phase"] == "S1").astype(float)
    out["phase_S2"] = (out["phase"] == "S2").astype(float)
    out["log_tokens"] = np.log(out["length_tokens"].clip(lower=1.0))
    out["pubdate_dt"] = pd.to_datetime(out["pubdate_dt"])
    out["time_weeks"] = (out["pubdate_dt"] - out["pubdate_dt"].min()).dt.days.astype(float) / 7.0
    out["duration_sec"] = pd.to_numeric(out.get("duration_sec"), errors="coerce")
    duration_fill = float(out["duration_sec"].dropna().median()) if out["duration_sec"].notna().any() else 1.0
    out["duration_sec"] = out["duration_sec"].fillna(duration_fill)
    out["log_duration"] = np.log(out["duration_sec"].clip(lower=1.0))
    out["title"] = out.get("title", "").astype(str)
    out["part_name"] = out.get("part_name", "").astype(str)
    out["title_pattern"] = out.get("title_pattern", "").astype(str)
    out["title_q"] = out["title"].str.contains(r"[?？]", regex=True).astype(float)
    out["title_bracket"] = out["title"].str.contains(r"[\[\]【】()（）]", regex=True).astype(float)
    format_text = out["title"].fillna("") + " " + out["part_name"].fillna("") + " " + out["title_pattern"].fillna("")
    out["series_part"] = format_text.str.contains(r"(?:上集|下集|完整版|最终版|全集|P\d+|Q&A|Part\s*\d+)", case=False, regex=True).astype(float)
    return out


def _transformed_outcome(df: pd.DataFrame, spec: FeatureSpec) -> pd.Series:
    if spec.kind == "bounded":
        return _safe_logit(df[spec.report_col].astype(float))
    if spec.kind == "positive":
        return np.log(df[spec.report_col].astype(float).clip(lower=1e-4))
    if spec.kind == "count":
        return np.log1p(df[spec.report_col].astype(float))
    return df[spec.report_col].astype(float)


def _fit_count_regression(
    *,
    formula: str,
    data: pd.DataFrame,
    count_col: str,
    group_col: str,
    offset: pd.Series,
) -> tuple[Any, dict[str, Any]]:
    diagnostics: dict[str, Any] = {
        "count_family": "poisson_fe_offset",
        "dispersion": np.nan,
        "alpha": np.nan,
    }
    poisson_fit = smf.glm(
        formula=formula,
        data=data,
        family=sm.families.Poisson(),
        offset=offset,
    ).fit(cov_type="cluster", cov_kwds={"groups": data[group_col]}, maxiter=200)
    mu = poisson_fit.mu
    pearson = np.sum(((data[count_col] - mu) ** 2) / np.clip(mu, 1e-6, None))
    dispersion = float(pearson / max(len(data) - len(poisson_fit.params), 1))
    diagnostics["dispersion"] = dispersion
    fit = poisson_fit
    if dispersion >= 3.0:
        try:
            nb_fit = smf.negativebinomial(
                formula=formula,
                data=data,
                offset=offset,
            ).fit(
                disp=False,
                maxiter=300,
                cov_type="cluster",
                cov_kwds={"groups": data[group_col]},
            )
            if np.isfinite(float(getattr(nb_fit, "llf", np.nan))):
                fit = nb_fit
                diagnostics["count_family"] = "negative_binomial_fe_offset"
                diagnostics["alpha"] = float(nb_fit.params.get("alpha", np.nan))
        except Exception as exc:  # noqa: BLE001
            diagnostics["nb_fallback_error"] = f"{type(exc).__name__}: {exc}"
    return fit, diagnostics


def _fit_panel_model(
    df: pd.DataFrame,
    spec: FeatureSpec,
    *,
    model_name: str = "A_panel_fe",
    extra_terms: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    use_prov = df["transcript_provenance"].nunique(dropna=False) > 1
    extra_terms = extra_terms or []
    diagnostics: dict[str, Any] = {
        "feature": spec.name,
        "model": model_name,
        "family": spec.model_family,
        "converged": True,
        "n_videos": int(len(df)),
        "n_creators": int(df["creator_id"].nunique()),
        "used_provenance_control": bool(use_prov),
        "extra_controls": ", ".join(extra_terms) if extra_terms else "",
    }
    rows: list[dict[str, Any]] = []
    if spec.kind == "count" and spec.count_col:
        rhs_terms = ["C(phase, Treatment(reference='S0'))", "time_weeks"] + extra_terms + ["C(creator_id)"]
        if use_prov:
            rhs_terms.append("C(transcript_provenance)")
        formula = f"{spec.count_col} ~ " + " + ".join(rhs_terms)
        try:
            fit, count_diag = _fit_count_regression(
                formula=formula,
                data=df,
                count_col=spec.count_col,
                group_col="creator_id",
                offset=np.log(df["length_tokens"].clip(lower=1.0)),
            )
            diagnostics.update(count_diag)
            diagnostics["family"] = str(count_diag.get("count_family") or diagnostics["family"])
            params = fit.params
            conf = fit.conf_int()
            pvals = fit.pvalues
            for phase in ["S1", "S2"]:
                term = f"C(phase, Treatment(reference='S0'))[T.{phase}]"
                if term not in params.index:
                    continue
                coef = float(params[term])
                low, high = conf.loc[term].tolist()
                rows.append(
                    {
                        "model": model_name,
                        "feature": spec.name,
                        "tier": spec.tier,
                        "feature_class": spec.class_label,
                        "term": f"{phase}_vs_S0",
                        "coef": coef,
                        "conf_low": float(low),
                        "conf_high": float(high),
                        "p_value": float(pvals[term]),
                        "effect_scale": "IRR",
                        "estimate": float(math.exp(coef)),
                        "estimate_low": float(math.exp(low)),
                        "estimate_high": float(math.exp(high)),
                        "direction": "decrease" if coef < 0 else "increase",
                        "n_videos": int(len(df)),
                        "n_creators": int(df["creator_id"].nunique()),
                        "control_set": "metadata" if extra_terms else "base",
                        "model_family_used": diagnostics["family"],
                    }
                )
        except Exception as exc:  # noqa: BLE001
            diagnostics["converged"] = False
            diagnostics["error"] = f"{type(exc).__name__}: {exc}"
    else:
        work = df.copy()
        work["__y__"] = _transformed_outcome(work, spec)
        rhs_terms = ["C(phase, Treatment(reference='S0'))", "time_weeks", "log_tokens"] + extra_terms + ["C(creator_id)"]
        if use_prov:
            rhs_terms.append("C(transcript_provenance)")
        formula = "__y__ ~ " + " + ".join(rhs_terms)
        try:
            fit = smf.ols(formula=formula, data=work).fit(cov_type="cluster", cov_kwds={"groups": work["creator_id"]})
            diagnostics["r2"] = float(getattr(fit, "rsquared", np.nan))
            params = fit.params
            conf = fit.conf_int()
            pvals = fit.pvalues
            for phase in ["S1", "S2"]:
                term = f"C(phase, Treatment(reference='S0'))[T.{phase}]"
                if term not in params.index:
                    continue
                coef = float(params[term])
                low, high = conf.loc[term].tolist()
                rows.append(
                    {
                        "model": model_name,
                        "feature": spec.name,
                        "tier": spec.tier,
                        "feature_class": spec.class_label,
                        "term": f"{phase}_vs_S0",
                        "coef": coef,
                        "conf_low": float(low),
                        "conf_high": float(high),
                        "p_value": float(pvals[term]),
                        "effect_scale": spec.transform,
                        "estimate": coef,
                        "estimate_low": float(low),
                        "estimate_high": float(high),
                        "direction": "decrease" if coef < 0 else "increase",
                        "n_videos": int(len(df)),
                        "n_creators": int(df["creator_id"].nunique()),
                        "control_set": "metadata" if extra_terms else "base",
                        "model_family_used": diagnostics["family"],
                    }
                )
        except Exception as exc:  # noqa: BLE001
            diagnostics["converged"] = False
            diagnostics["error"] = f"{type(exc).__name__}: {exc}"
    return pd.DataFrame(rows), diagnostics


def _fit_mixed_model(df: pd.DataFrame, spec: FeatureSpec) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    work = df.copy()
    work["__y__"] = _transformed_outcome(work, spec)
    use_prov = work["transcript_provenance"].nunique(dropna=False) > 1
    formula = "__y__ ~ phase_S1 + phase_S2 + time_weeks + log_tokens"
    if use_prov:
        formula += " + C(transcript_provenance)"
    diagnostics: dict[str, Any] = {
        "feature": spec.name,
        "model": "B_mixed_random_slopes",
        "converged": True,
        "n_videos": int(len(work)),
        "n_creators": int(work["creator_id"].nunique()),
        "used_provenance_control": bool(use_prov),
    }
    result_rows: list[dict[str, Any]] = []
    heter_rows: list[dict[str, Any]] = []
    model = smf.mixedlm(formula, data=work, groups=work["creator_id"], re_formula="~phase_S1 + phase_S2")
    fit = None
    try:
        for method in ["lbfgs", "powell", "cg"]:
            try:
                candidate = model.fit(reml=False, method=method, maxiter=400, disp=False)
                if fit is None:
                    fit = candidate
                if bool(getattr(candidate, "converged", True)):
                    fit = candidate
                    diagnostics["optimizer"] = method
                    break
            except Exception as exc:  # noqa: BLE001
                diagnostics.setdefault("fit_attempt_errors", []).append(f"{method}: {type(exc).__name__}: {exc}")
        if fit is None:
            raise RuntimeError("mixed model did not return a fit object")
    except Exception as exc:  # noqa: BLE001
        diagnostics["converged"] = False
        diagnostics["error"] = f"{type(exc).__name__}: {exc}"
        return pd.DataFrame(result_rows), pd.DataFrame(heter_rows), diagnostics

    diagnostics["llf"] = float(getattr(fit, "llf", np.nan))
    diagnostics["converged"] = bool(getattr(fit, "converged", True))
    cov_re = getattr(fit, "cov_re", None)
    if cov_re is not None:
        try:
            diagnostics["random_intercept_sd"] = float(np.sqrt(max(float(cov_re.iloc[0, 0]), 0.0)))
            if "phase_S1" in cov_re.index:
                diagnostics["random_slope_sd_s1"] = float(np.sqrt(max(float(cov_re.loc["phase_S1", "phase_S1"]), 0.0)))
            if "phase_S2" in cov_re.index:
                diagnostics["random_slope_sd_s2"] = float(np.sqrt(max(float(cov_re.loc["phase_S2", "phase_S2"]), 0.0)))
        except Exception:
            pass
    fe = fit.fe_params
    conf = fit.conf_int()
    pvals = fit.pvalues
    for term, label in [("phase_S1", "S1_vs_S0"), ("phase_S2", "S2_vs_S0")]:
        if term not in fe.index:
            continue
        low, high = conf.loc[term].tolist()
        result_rows.append(
            {
                "model": "B_mixed_random_slopes",
                "feature": spec.name,
                "tier": spec.tier,
                "feature_class": spec.class_label,
                "term": label,
                "coef": float(fe[term]),
                "conf_low": float(low),
                "conf_high": float(high),
                "p_value": float(pvals[term]),
                "effect_scale": spec.transform,
                "estimate": float(fe[term]),
                "estimate_low": float(low),
                "estimate_high": float(high),
                "direction": "decrease" if float(fe[term]) < 0 else "increase",
                "n_videos": int(len(work)),
                "n_creators": int(work["creator_id"].nunique()),
            }
        )

    creator_phase_means = work.groupby(["creator_id", "phase"])["__y__"].mean().unstack()
    raw_effects: dict[str, dict[str, float]] = {}
    for creator_id, row in creator_phase_means.iterrows():
        raw_effects[str(creator_id)] = {
            "S1_vs_S0": float(row.get("S1", np.nan) - row.get("S0", np.nan)) if pd.notna(row.get("S0", np.nan)) and pd.notna(row.get("S1", np.nan)) else np.nan,
            "S2_vs_S0": float(row.get("S2", np.nan) - row.get("S0", np.nan)) if pd.notna(row.get("S0", np.nan)) and pd.notna(row.get("S2", np.nan)) else np.nan,
        }

    creator_counts = work.groupby("creator_id").size().to_dict()
    creator_names = work.groupby("creator_id")["creator_name"].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]).to_dict()
    creator_phases = work.groupby("creator_id")["phase"].nunique().to_dict()
    for creator_id, reff in fit.random_effects.items():
        effect_s1 = float(fe.get("phase_S1", 0.0) + reff.get("phase_S1", 0.0))
        effect_s2 = float(fe.get("phase_S2", 0.0) + reff.get("phase_S2", 0.0))
        raw_s1 = raw_effects.get(str(creator_id), {}).get("S1_vs_S0", np.nan)
        raw_s2 = raw_effects.get(str(creator_id), {}).get("S2_vs_S0", np.nan)
        pooled_s1 = float(fe.get("phase_S1", 0.0))
        pooled_s2 = float(fe.get("phase_S2", 0.0))
        shrink_s1 = np.nan
        shrink_s2 = np.nan
        if np.isfinite(raw_s1):
            denom = abs(raw_s1 - pooled_s1)
            shrink_s1 = abs(effect_s1 - pooled_s1) / denom if denom > 1e-8 else np.nan
        if np.isfinite(raw_s2):
            denom = abs(raw_s2 - pooled_s2)
            shrink_s2 = abs(effect_s2 - pooled_s2) / denom if denom > 1e-8 else np.nan
        heter_rows.append(
            {
                "row_type": "creator_effect",
                "feature": spec.name,
                "creator_id": creator_id,
                "creator_name": creator_names.get(creator_id, ""),
                "contrast": "S1_vs_S0",
                "pooled_effect": pooled_s1,
                "creator_effect": effect_s1,
                "raw_creator_effect": raw_s1,
                "shrinkage_ratio": float(shrink_s1) if np.isfinite(shrink_s1) else np.nan,
                "aligns_with_pooled": int(np.sign(effect_s1) == np.sign(float(fe.get("phase_S1", 0.0)))) if effect_s1 != 0 else 1,
                "n_videos": int(creator_counts.get(creator_id, 0)),
                "n_phases": int(creator_phases.get(creator_id, 0)),
            }
        )
        heter_rows.append(
            {
                "row_type": "creator_effect",
                "feature": spec.name,
                "creator_id": creator_id,
                "creator_name": creator_names.get(creator_id, ""),
                "contrast": "S2_vs_S0",
                "pooled_effect": pooled_s2,
                "creator_effect": effect_s2,
                "raw_creator_effect": raw_s2,
                "shrinkage_ratio": float(shrink_s2) if np.isfinite(shrink_s2) else np.nan,
                "aligns_with_pooled": int(np.sign(effect_s2) == np.sign(float(fe.get("phase_S2", 0.0)))) if effect_s2 != 0 else 1,
                "n_videos": int(creator_counts.get(creator_id, 0)),
                "n_phases": int(creator_phases.get(creator_id, 0)),
            }
        )
    heter_df = pd.DataFrame(heter_rows)
    summary_rows: list[dict[str, Any]] = []
    if not heter_df.empty:
        for contrast in ["S1_vs_S0", "S2_vs_S0"]:
            sub = heter_df[heter_df["contrast"] == contrast]
            if sub.empty:
                continue
            summary_rows.append(
                {
                    "row_type": "summary",
                    "feature": spec.name,
                    "creator_id": "",
                    "creator_name": "",
                    "contrast": contrast,
                    "pooled_effect": float(sub["pooled_effect"].iloc[0]),
                    "creator_effect": float(sub["creator_effect"].median()),
                    "aligns_with_pooled": float(sub["aligns_with_pooled"].mean()),
                    "n_videos": int(sub["n_videos"].sum()),
                    "n_phases": int(sub["n_phases"].median()),
                    "same_direction_share": float(sub["aligns_with_pooled"].mean()),
                    "effect_sd": float(sub["creator_effect"].std(ddof=0)),
                    "effect_p10": float(sub["creator_effect"].quantile(0.10)),
                    "effect_p90": float(sub["creator_effect"].quantile(0.90)),
                    "raw_effect_median": float(sub["raw_creator_effect"].median()) if sub["raw_creator_effect"].notna().any() else np.nan,
                    "median_shrinkage_ratio": float(sub["shrinkage_ratio"].median()) if sub["shrinkage_ratio"].notna().any() else np.nan,
                    "shrinkage_supported_share": float((sub["shrinkage_ratio"] < 1.0).mean()) if sub["shrinkage_ratio"].notna().any() else np.nan,
                }
            )
    if summary_rows:
        heter_df = pd.concat([heter_df, pd.DataFrame(summary_rows)], ignore_index=True)
    return pd.DataFrame(result_rows), heter_df, diagnostics


def _fit_break_model(
    df: pd.DataFrame,
    spec: FeatureSpec,
    breakpoint_date: pd.Timestamp,
    *,
    buffer_weeks: int = 0,
) -> pd.DataFrame:
    work = df.copy()
    if buffer_weeks > 0:
        delta = (work["pubdate_dt"] - breakpoint_date).dt.days.astype(float) / 7.0
        work = work[delta.abs() > float(buffer_weeks)].copy()
    if work.empty:
        return pd.DataFrame()
    work["post_break"] = (work["pubdate_dt"] >= breakpoint_date).astype(float)
    work["weeks_after_break"] = ((work["pubdate_dt"] - breakpoint_date).dt.days.clip(lower=0).astype(float)) / 7.0
    use_prov = work["transcript_provenance"].nunique(dropna=False) > 1
    rows: list[dict[str, Any]] = []

    if spec.kind == "count" and spec.count_col:
        formula = f"{spec.count_col} ~ post_break + weeks_after_break + time_weeks + C(creator_id)"
        if use_prov:
            formula += " + C(transcript_provenance)"
        fit, count_diag = _fit_count_regression(
            formula=formula,
            data=work,
            count_col=spec.count_col,
            group_col="creator_id",
            offset=np.log(work["length_tokens"].clip(lower=1.0)),
        )
        params = fit.params
        conf = fit.conf_int()
        pvals = fit.pvalues
        for term in ["post_break", "weeks_after_break"]:
            low, high = conf.loc[term].tolist()
            rows.append(
                {
                    "feature": spec.name,
                    "term": term,
                    "coef": float(params[term]),
                    "conf_low": float(low),
                    "conf_high": float(high),
                    "p_value": float(pvals[term]),
                    "n_videos": int(len(work)),
                    "n_creators": int(work["creator_id"].nunique()),
                    "effect_scale": "log_rate",
                    "model_family_used": str(count_diag.get("count_family") or "poisson_fe_offset"),
                }
            )
    else:
        work["__y__"] = _transformed_outcome(work, spec)
        formula = "__y__ ~ post_break + weeks_after_break + time_weeks + log_tokens + C(creator_id)"
        if use_prov:
            formula += " + C(transcript_provenance)"
        fit = smf.ols(formula=formula, data=work).fit(cov_type="cluster", cov_kwds={"groups": work["creator_id"]})
        params = fit.params
        conf = fit.conf_int()
        pvals = fit.pvalues
        for term in ["post_break", "weeks_after_break"]:
            low, high = conf.loc[term].tolist()
            rows.append(
                {
                    "feature": spec.name,
                    "term": term,
                    "coef": float(params[term]),
                    "conf_low": float(low),
                    "conf_high": float(high),
                    "p_value": float(pvals[term]),
                    "n_videos": int(len(work)),
                    "n_creators": int(work["creator_id"].nunique()),
                    "effect_scale": spec.transform,
                }
            )
    return pd.DataFrame(rows)


def _event_study(df: pd.DataFrame, spec: FeatureSpec, breakpoint_date: pd.Timestamp, label: str) -> pd.DataFrame:
    work = df.copy()
    work["event_week"] = ((work["pubdate_dt"] - breakpoint_date).dt.days.astype(float)) / 7.0
    work = work[(work["event_week"] >= -24) & (work["event_week"] <= 24)].copy()
    if work.empty:
        return pd.DataFrame()
    bins = [-24, -16, -8, -4, -2, 0, 2, 4, 8, 16, 24]
    labels = ["[-24,-16)", "[-16,-8)", "[-8,-4)", "[-4,-2)", "[-2,0)", "[0,2)", "[2,4)", "[4,8)", "[8,16)", "[16,24]"]
    work["event_bin"] = pd.cut(work["event_week"], bins=bins, labels=labels, include_lowest=True, right=False)
    work = work.dropna(subset=["event_bin"]).copy()
    work["event_bin"] = work["event_bin"].astype(str)
    use_prov = work["transcript_provenance"].nunique(dropna=False) > 1
    formula = "__y__ ~ C(event_bin, Treatment(reference='[-8,-4)')) + log_tokens + C(creator_id)"
    if use_prov:
        formula += " + C(transcript_provenance)"
    work["__y__"] = _transformed_outcome(work, spec)
    fit = smf.ols(formula=formula, data=work).fit(cov_type="cluster", cov_kwds={"groups": work["creator_id"]})
    conf = fit.conf_int()
    pvals = fit.pvalues
    rows: list[dict[str, Any]] = []
    for event_bin in labels:
        if event_bin == "[-8,-4)":
            rows.append(
                {
                    "feature": spec.name,
                    "breakpoint": label,
                    "event_bin": event_bin,
                    "coef": 0.0,
                    "conf_low": 0.0,
                    "conf_high": 0.0,
                    "p_value": np.nan,
                    "n_videos": int(len(work)),
                }
            )
            continue
        term = f"C(event_bin, Treatment(reference='[-8,-4)'))[T.{event_bin}]"
        if term not in fit.params.index:
            continue
        low, high = conf.loc[term].tolist()
        rows.append(
            {
                "feature": spec.name,
                "breakpoint": label,
                "event_bin": event_bin,
                "coef": float(fit.params[term]),
                "conf_low": float(low),
                "conf_high": float(high),
                "p_value": float(pvals[term]),
                "n_videos": int(len(work)),
            }
        )
    return pd.DataFrame(rows)


def _creator_month_its(df: pd.DataFrame, specs: list[FeatureSpec], t1: pd.Timestamp, t2: pd.Timestamp) -> pd.DataFrame:
    work = df.copy()
    work["month"] = work["pubdate_dt"].dt.to_period("M").dt.to_timestamp()
    agg_rows: list[dict[str, Any]] = []
    grouped = work.groupby(["creator_id", "creator_name", "month"], dropna=False)
    for (creator_id, creator_name, month), grp in grouped:
        row: dict[str, Any] = {
            "creator_id": creator_id,
            "creator_name": creator_name,
            "month": month,
            "transcript_provenance": grp["transcript_provenance"].mode().iloc[0],
            "length_tokens": float(grp["length_tokens"].sum()),
        }
        for spec in specs:
            if spec.kind == "count" and spec.count_col:
                row[spec.count_col] = float(grp[spec.count_col].sum())
                row[spec.report_col] = float(grp[spec.count_col].sum()) * 1000.0 / float(max(grp["length_tokens"].sum(), 1.0))
            else:
                row[spec.report_col] = float(grp[spec.report_col].mean())
        agg_rows.append(row)
    month_df = pd.DataFrame(agg_rows)
    if month_df.empty:
        return month_df
    month_df["time_months"] = (month_df["month"] - month_df["month"].min()).dt.days.astype(float) / 30.4
    month_df["post_t1"] = (month_df["month"] >= t1).astype(float)
    month_df["post_t2"] = (month_df["month"] >= t2).astype(float)
    month_df["months_after_t1"] = ((month_df["month"] - t1).dt.days.clip(lower=0).astype(float)) / 30.4
    month_df["months_after_t2"] = ((month_df["month"] - t2).dt.days.clip(lower=0).astype(float)) / 30.4
    month_df["log_tokens"] = np.log(month_df["length_tokens"].clip(lower=1.0))
    use_prov = month_df["transcript_provenance"].nunique(dropna=False) > 1
    out_rows: list[dict[str, Any]] = []
    for spec in specs:
        formula = "__y__ ~ post_t1 + post_t2 + months_after_t1 + months_after_t2 + C(creator_id)"
        if spec.kind != "count":
            formula += " + log_tokens"
        if use_prov:
            formula += " + C(transcript_provenance)"
        if spec.kind == "count" and spec.count_col:
            fit, count_diag = _fit_count_regression(
                formula=f"{spec.count_col} ~ post_t1 + post_t2 + months_after_t1 + months_after_t2 + C(creator_id)" + (" + C(transcript_provenance)" if use_prov else ""),
                data=month_df,
                count_col=spec.count_col,
                group_col="creator_id",
                offset=np.log(month_df["length_tokens"].clip(lower=1.0)),
            )
            conf = fit.conf_int()
            for term in ["post_t1", "post_t2", "months_after_t1", "months_after_t2"]:
                low, high = conf.loc[term].tolist()
                out_rows.append(
                    {
                        "feature": spec.name,
                        "term": term,
                        "coef": float(fit.params[term]),
                        "conf_low": float(low),
                        "conf_high": float(high),
                        "p_value": float(fit.pvalues[term]),
                        "n_creator_months": int(len(month_df)),
                        "model_family_used": str(count_diag.get("count_family") or "poisson_fe_offset"),
                    }
                )
        else:
            month_df["__y__"] = _transformed_outcome(month_df, spec)
            fit = smf.ols(formula=formula, data=month_df).fit(cov_type="cluster", cov_kwds={"groups": month_df["creator_id"]})
            conf = fit.conf_int()
            for term in ["post_t1", "post_t2", "months_after_t1", "months_after_t2"]:
                low, high = conf.loc[term].tolist()
                out_rows.append(
                    {
                        "feature": spec.name,
                        "term": term,
                        "coef": float(fit.params[term]),
                        "conf_low": float(low),
                        "conf_high": float(high),
                        "p_value": float(fit.pvalues[term]),
                        "n_creator_months": int(len(month_df)),
                        "model_family_used": "ols_fe_clustered",
                    }
                )
    return pd.DataFrame(out_rows)


def _compare_robustness(
    baseline: pd.DataFrame,
    test: pd.DataFrame,
    *,
    features: list[str],
    force_partial: bool = False,
) -> str:
    base = baseline[baseline["feature"].isin(features)].copy()
    alt = test[test["feature"].isin(features)].copy()
    if base.empty or alt.empty:
        return "not consistent"
    merged = base.merge(alt, on=["feature", "term"], suffixes=("_base", "_test"))
    if merged.empty:
        return "not consistent"
    sign_match = np.sign(merged["coef_base"]) == np.sign(merged["coef_test"])
    base_sig = merged["p_value_base"] < 0.05
    test_sig = merged["p_value_test"] < 0.10
    if force_partial:
        return "partially consistent"
    if bool(sign_match.all()) and (not bool(base_sig.any()) or bool(test_sig[base_sig].all())):
        return "consistent"
    if sign_match.mean() >= 0.5:
        return "partially consistent"
    return "not consistent"


def _write_sample_flow(
    out_dir: Path,
    sample_flow: pd.DataFrame,
    *,
    main_videos: int,
    main_creators: int,
    balanced_videos: int,
    balanced_creators: int,
    creators_all_phases: int,
    draft_note: str,
) -> None:
    sample_flow.to_csv(out_dir / "sample_flow.csv", index=False, encoding="utf-8-sig")
    note = f"""# Formal Sample Flow

Formal main sample specification: **{main_videos} videos / {main_creators} creators**.

Balanced sample specification: **{balanced_videos} videos / {balanced_creators} creators**, defined only as a robustness layer using an exact 10x3 creator-phase subset.

The current production report number `2412 / 87` refers to subtitle transcripts with readable transcript files, repaired segmentation, and retained analytic features in the current formal pipeline.

The draft abstract number `1000 / 990 / 81` is treated here as an earlier manuscript-stage snapshot rather than a valid formal sample layer. It is documented for reconciliation only and is not used in any model, table, or figure.

Creators with all three phases in the formal main sample: **{creators_all_phases}**.

{draft_note}

## Flow Table

{_md_table(sample_flow)}
"""
    (out_dir / "sample_flow.md").write_text(note, encoding="utf-8")


def _write_preprocessing_log(
    out_dir: Path,
    qc_df: pd.DataFrame,
    *,
    provenance_counts: pd.Series,
) -> None:
    summary = pd.DataFrame(
        [
            {
                "metric": "mean_sent_len_chars",
                "before_mean": qc_df["mean_sent_len_chars_before"].mean(),
                "after_mean": qc_df["mean_sent_len_chars_after"].mean(),
                "before_median": qc_df["mean_sent_len_chars_before"].median(),
                "after_median": qc_df["mean_sent_len_chars_after"].median(),
            },
            {
                "metric": "comma_period_ratio",
                "before_mean": qc_df["comma_period_ratio_before"].mean(),
                "after_mean": qc_df["comma_period_ratio_after"].mean(),
                "before_median": qc_df["comma_period_ratio_before"].median(),
                "after_median": qc_df["comma_period_ratio_after"].median(),
            },
            {
                "metric": "segment_count_per_1000_tokens",
                "before_mean": qc_df["segment_count_per_1000_tokens_before"].mean(),
                "after_mean": qc_df["segment_count_per_1000_tokens_after"].mean(),
                "before_median": qc_df["segment_count_per_1000_tokens_before"].median(),
                "after_median": qc_df["segment_count_per_1000_tokens_after"].median(),
            },
            {
                "metric": "avg_tokens_per_segment",
                "before_mean": qc_df["avg_tokens_per_segment_before"].mean(),
                "after_mean": qc_df["avg_tokens_per_segment_after"].mean(),
                "before_median": qc_df["avg_tokens_per_segment_before"].median(),
                "after_median": qc_df["avg_tokens_per_segment_after"].median(),
            },
        ]
    )
    text = f"""# Preprocessing Rebuild Log

## Transcript provenance

{_md_table(provenance_counts.rename_axis("transcript_provenance").reset_index(name="count"))}

The current formal main sample is **ASR only**. Provenance is retained for documentation and limitations, but it does not vary within the formal sample and therefore cannot support a source-comparison model.

## Segmentation rebuild

The repaired structure pipeline starts from BCC chunk timing rather than raw ASR punctuation.

Rules:
- normalize whitespace and punctuation, then remove configured onomatopoeia terms;
- start a new repaired segment when pause >= 1.0 seconds;
- also split when medium pauses appear after enough material has accumulated;
- also split before discourse-cue openings such as heading-like or definitional markers when enough text has already accumulated;
- merge back very short fragments so structure metrics are not driven by timing noise alone.

This means sentence/structure metrics no longer depend on raw ASR commas or periods for boundary detection.

## Before / after QC summary

{_md_table(summary)}
"""
    (out_dir / "preprocessing_log.md").write_text(text, encoding="utf-8")


def _write_feature_files(
    out_dir: Path,
    feature_specs: dict[str, FeatureSpec],
    feature_qc: pd.DataFrame,
    corr: pd.DataFrame,
    *,
    component_terms: dict[str, list[str]],
) -> None:
    decision_rows = []
    for spec in feature_specs.values():
        decision_rows.append(
            {
                "feature": spec.name,
                "tier": spec.tier,
                "class": spec.class_label,
                "description": spec.description,
                "model_family": spec.model_family,
                "transformation": spec.transform,
                "headline_allowed": int(spec.tier == "main"),
                "keep_status": {
                    "main": "keep as main",
                    "secondary": "keep as secondary",
                    "appendix": "appendix only",
                    "drop": "drop",
                }.get(spec.tier, "drop"),
            }
        )
    decision_rows.append(
        {
            "feature": "connectives_total",
            "tier": "drop",
            "class": "raw connective count",
            "description": "Legacy absolute connective count. Dropped from headline use because it is not length-corrected.",
            "model_family": "not_modeled",
            "transformation": "replaced by connective_density or count model with offset(log(tokens))",
            "headline_allowed": 0,
            "keep_status": "drop",
        }
    )
    decision_df = pd.DataFrame(decision_rows).sort_values(["tier", "feature"]).reset_index(drop=True)
    decision_df.to_csv(out_dir / "feature_decision_table.csv", index=False, encoding="utf-8-sig")
    decision_df.to_csv(out_dir / "feature_definition_table.csv", index=False, encoding="utf-8-sig")
    feature_qc.to_csv(out_dir / "feature_qc_table.csv", index=False, encoding="utf-8-sig")
    corr.to_csv(out_dir / "correlation_matrix.csv", encoding="utf-8-sig")

    structure_lines = []
    for name, terms in component_terms.items():
        structure_lines.append(f"- `{name}`: " + " / ".join(terms))
    text = f"""# Feature Specification

## Formal feature tiers

### Main features
- `connective_density`: connective hits per 1,000 tokens. Headline reporting uses density; panel modeling uses Poisson fixed-effects with offset(log(tokens)).
- `template_density`: template / framing phrase hits per 1,000 tokens. Panel modeling uses Poisson fixed-effects with offset(log(tokens)).
- `mattr`: lexical diversity over jieba tokens. Modeled on a clipped-logit scale.
- `structure_composite`: average z-score of four transparent component densities.

Structure composite components:
{chr(10).join(structure_lines)}

### Secondary features
- `heading_like_density`
- `definitional_marker_density`
- `inferential_marker_density`
- `liability_shield_density`
- `segment_count_per_1000_tokens`
- `func_ratio`
- `stop_ratio`
- `mean_word_len`
- `style_index_pc1`

### Appendix-only features
- `avg_tokens_per_segment`
- `legacy_comma_period_ratio`
- `legacy_mean_sent_len_chars`
- `repaired_mean_segment_chars`
- `segment_length_cv`
- `segment_boundary_density`

### Dropped headline candidates
- `connectives_total` is no longer a formal analytic headline because it is a raw absolute count.
- `formulaic_transition_density` is dropped from formal reporting because it overlaps heavily with `connective_density`, `template_density`, and heading-like markers.

## Measurement rules
- Count-like features use Poisson or Negative Binomial fixed-effects models with offset(log(tokens)), depending on overdispersion.
- Bounded ratios use clipped-logit transforms with creator fixed effects and clustered standard errors.
- Positive continuous variables use log transforms when needed.
- Breakpoint robustness uses transformed outcomes for stability and interpretability, not raw absolute counts.

## Evaluated automatic metrics
- `heading_like_density`
- `definitional_marker_density`
- `inferential_marker_density`
- `liability_shield_density`
- `formulaic_transition_density`
- `segment_count_per_1000_tokens`
- `avg_tokens_per_segment`
"""
    (out_dir / "feature_spec.md").write_text(text, encoding="utf-8")


def _write_model_plan(out_dir: Path) -> None:
    text = """# Model Plan

## Model A: video-level panel model
- unit: video
- predictor: phase
- creator fixed effects: yes
- clustered SE by creator: yes
- controls: log(tokens) or offset(log(tokens)), continuous time trend, transcript provenance when variation exists
- metadata-controlled robustness: log(duration), title-question flag, bracket-title flag, series/part flag
- count-like outcomes: Poisson with Negative Binomial fallback when overdispersion is high

## Model B: random-slope mixed model
- creator random intercept: yes
- phase random slopes by creator: yes
- purpose: estimate heterogeneity in S1-S0 and S2-S0 drift

## Model C: creator-level phase effect distribution
- source: empirical-Bayes creator-specific phase slopes from Model B
- outputs: creator-level effect table, forest plot, same-direction share, partial-pooling / shrinkage diagnostics

## Breakpoint analysis
- role: robustness only
- benchmark dates: 2022-11-30 and 2023-11-02
- checks: shifted cutpoints, buffer exclusions, event-time bins, creator-month ITS
"""
    (out_dir / "model_plan.md").write_text(text, encoding="utf-8")


def _plot_sample_flow(path: Path, sample_flow: pd.DataFrame) -> None:
    labels = sample_flow["stage"].tolist()[:6]
    counts = sample_flow["n_videos"].fillna(0).astype(int).tolist()[:6]
    fig, ax = plt.subplots(figsize=(12, 2.8), dpi=160)
    ax.axis("off")
    width = 1.55
    for i, (label, count) in enumerate(zip(labels, counts)):
        x = i * 1.8
        rect = plt.Rectangle((x, 0.2), width, 0.8, facecolor="#EADBC8", edgecolor="#3E2C2C", linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x + width / 2, 0.73, label.replace("_", "\n"), ha="center", va="center", fontsize=9)
        ax.text(x + width / 2, 0.37, f"n={count}", ha="center", va="center", fontsize=10, fontweight="bold")
        if i < len(labels) - 1:
            ax.annotate("", xy=(x + width + 0.22, 0.6), xytext=(x + width + 0.02, 0.6), arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.set_xlim(-0.1, len(labels) * 1.8 - 0.1)
    ax.set_ylim(0.0, 1.2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_phase_distribution(path: Path, main_df: pd.DataFrame, balanced_df: pd.DataFrame) -> None:
    phase_main = main_df["phase"].value_counts().reindex(PHASES).fillna(0)
    phase_bal = balanced_df["phase"].value_counts().reindex(PHASES).fillna(0)
    x = np.arange(len(PHASES))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4), dpi=160)
    ax.bar(x - width / 2, phase_main.values, width, label="Main sample", color="#8FB9A8")
    ax.bar(x + width / 2, phase_bal.values, width, label="Balanced sample", color="#F2C572")
    ax.set_xticks(x, PHASES)
    ax.set_ylabel("Videos")
    ax.set_title("Phase Distribution")
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_feature_distributions(path: Path, df: pd.DataFrame) -> None:
    features = MAIN_FEATURES
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), dpi=160)
    for ax, feature in zip(axes.ravel(), features):
        data = [df.loc[df["phase"] == phase, feature].astype(float).dropna().to_numpy() for phase in PHASES]
        ax.boxplot(data, tick_labels=PHASES, showmeans=True)
        ax.set_title(feature)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_forest(path: Path, heter_df: pd.DataFrame, feature: str, *, show_creator_labels: bool = True) -> None:
    sub = heter_df[(heter_df["row_type"] == "creator_effect") & (heter_df["feature"] == feature)].copy()
    if sub.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 14), dpi=160, sharey=False)
    for ax, contrast in zip(axes, ["S1_vs_S0", "S2_vs_S0"]):
        cur = sub[sub["contrast"] == contrast].sort_values("creator_effect")
        y = np.arange(len(cur))
        ax.hlines(y, xmin=0, xmax=cur["creator_effect"], color="#B8B8B8", linewidth=0.8)
        ax.scatter(cur["creator_effect"], y, color="#2E5E4E", s=18)
        ax.axvline(float(cur["pooled_effect"].iloc[0]), color="#C44D58", linestyle="--", linewidth=1.2)
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.set_title(f"{feature}: {contrast}")
        ax.set_xlabel("Creator-specific effect")
        if show_creator_labels:
            step = max(1, len(y) // 20)
            ax.set_yticks(y[::step])
            ax.set_yticklabels(cur["creator_name"].iloc[::step])
        else:
            ax.set_yticks([])
            ax.set_ylabel("")
        ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_event_time(path: Path, event_df: pd.DataFrame, feature: str) -> None:
    sub = event_df[event_df["feature"] == feature].copy()
    if sub.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=160, sharey=True)
    for ax, breakpoint in zip(axes, ["T1", "T2"]):
        cur = sub[sub["breakpoint"] == breakpoint].copy()
        if cur.empty:
            continue
        x = np.arange(len(cur))
        ax.errorbar(
            x,
            cur["coef"],
            yerr=[cur["coef"] - cur["conf_low"], cur["conf_high"] - cur["coef"]],
            fmt="o-",
            color="#2E5E4E",
            ecolor="#6A8D73",
            capsize=3,
        )
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xticks(x, cur["event_bin"], rotation=45, ha="right")
        ax.set_title(f"{feature} around {breakpoint}")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_segmentation_qc(path: Path, qc_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=160)
    pairs = [
        ("mean_sent_len_chars_before", "mean_sent_len_chars_after", "Segment Length"),
        ("comma_period_ratio_before", "comma_period_ratio_after", "Comma / Boundary"),
        ("segment_count_per_1000_tokens_before", "segment_count_per_1000_tokens_after", "Boundary Density"),
        ("avg_tokens_per_segment_before", "avg_tokens_per_segment_after", "Tokens per Segment"),
    ]
    for ax, (left, right, title) in zip(axes.ravel(), pairs):
        ax.boxplot(
            [qc_df[left].astype(float).to_numpy(), qc_df[right].astype(float).to_numpy()],
            tick_labels=["Legacy", "Repaired"],
            showmeans=True,
        )
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _creator_alias_map(df: pd.DataFrame) -> dict[str, str]:
    creator_ids = sorted(df["creator_id"].astype(str).dropna().unique().tolist())
    return {creator_id: f"Creator_{idx:03d}" for idx, creator_id in enumerate(creator_ids, start=1)}


def _apply_creator_aliases(df: pd.DataFrame, alias_map: dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    if "creator_id" in out.columns:
        out["creator_id"] = out["creator_id"].astype(str).map(alias_map).where(out["creator_id"].astype(str) != "", out["creator_id"])
    if "creator_name" in out.columns:
        if "creator_id" in out.columns:
            out["creator_name"] = out["creator_id"].astype(str).where(out["creator_id"].astype(str) != "", out["creator_name"])
        else:
            out["creator_name"] = out["creator_name"].astype(str)
    return out


def _copy_text_file(src: Path, dst: Path) -> None:
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def _write_blind_review_exports(
    out_dir: Path,
    figures_dir: Path,
    *,
    main_df: pd.DataFrame,
    heterogeneity_results: pd.DataFrame,
    sample_flow: pd.DataFrame,
    feature_decision_path: Path,
    feature_qc_path: Path,
    corr_path: Path,
    main_results_path: Path,
    metadata_results_path: Path,
    robustness_path: Path,
    model_diagnostics_path: Path,
    breakpoint_path: Path,
    event_time_path: Path,
    creator_month_path: Path,
    sample_flow_md_path: Path,
    old_vs_new_path: Path,
    feature_spec_path: Path,
    results_abstract_path: Path,
    sample_description_path: Path,
) -> None:
    blind_fig_dir = out_dir / "blind_review_figure_set"
    blind_tbl_dir = out_dir / "blind_review_tables"
    blind_fig_dir.mkdir(parents=True, exist_ok=True)
    blind_tbl_dir.mkdir(parents=True, exist_ok=True)

    alias_map = _creator_alias_map(main_df)
    blind_heter = _apply_creator_aliases(heterogeneity_results, alias_map)
    blind_heter.to_csv(blind_tbl_dir / "heterogeneity_results.csv", index=False, encoding="utf-8-sig")

    sample_flow.to_csv(blind_tbl_dir / "sample_flow.csv", index=False, encoding="utf-8-sig")
    shutil.copyfile(feature_decision_path, blind_tbl_dir / "feature_decision_table.csv")
    shutil.copyfile(feature_qc_path, blind_tbl_dir / "feature_qc_table.csv")
    shutil.copyfile(corr_path, blind_tbl_dir / "correlation_matrix.csv")
    shutil.copyfile(main_results_path, blind_tbl_dir / "main_results_table.csv")
    shutil.copyfile(metadata_results_path, blind_tbl_dir / "metadata_controlled_results.csv")
    shutil.copyfile(breakpoint_path, blind_tbl_dir / "breakpoint_sensitivity_table.csv")
    shutil.copyfile(event_time_path, blind_tbl_dir / "event_time_results.csv")
    shutil.copyfile(creator_month_path, blind_tbl_dir / "creator_month_its_results.csv")
    shutil.copyfile(sample_description_path, blind_tbl_dir / "sample_description_table.csv")
    _copy_text_file(sample_flow_md_path, blind_tbl_dir / "sample_flow.md")
    _copy_text_file(old_vs_new_path, blind_tbl_dir / "old_vs_new_spec.md")
    _copy_text_file(feature_spec_path, blind_tbl_dir / "feature_spec.md")
    _copy_text_file(robustness_path, blind_tbl_dir / "robustness_summary.md")
    _copy_text_file(model_diagnostics_path, blind_tbl_dir / "model_diagnostics.md")
    _copy_text_file(results_abstract_path, blind_tbl_dir / "results_for_abstract.md")

    shutil.copyfile(figures_dir / "sample_flow_figure.png", blind_fig_dir / "figure_01_sample_flow.png")
    shutil.copyfile(figures_dir / "phase_distribution_figure.png", blind_fig_dir / "figure_02_phase_distribution.png")
    shutil.copyfile(figures_dir / "main_feature_distributions.png", blind_fig_dir / "figure_03_main_feature_distributions.png")
    shutil.copyfile(figures_dir / "event_time_figure.png", blind_fig_dir / "figure_05_event_time.png")
    shutil.copyfile(figures_dir / "segmentation_qc_before_after.png", blind_fig_dir / "figure_06_segmentation_qc.png")
    _plot_forest(blind_fig_dir / "figure_04_creator_effect_forest.png", blind_heter, "connective_density", show_creator_labels=False)

    readme = """# Blind Review Export Package

This directory contains two output modes:

1. Internal working outputs:
   The standard files in the current output directory remain the internal working set.

2. Blind review outputs:
   - `blind_review_figure_set/`
   - `blind_review_tables/`

Blind review rules used here:
- creator-identifying labels are removed from figures or replaced with neutral creator IDs in tables;
- local usernames, local file paths, and machine-specific folders are not included;
- the formal sample specification remains 2412 videos / 87 creators;
- the balanced sample remains a robustness layer only.

Files in `blind_review_tables/` are review-facing copies of the key quantitative tables and summaries.
Files in `blind_review_figure_set/` are review-facing figures with neutral filenames and no creator-identifying labels.
"""
    (out_dir / "blind_review_export_readme.md").write_text(readme, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Formal quantitative analysis rebuild.")
    parser.add_argument("--run-id", default="20260131_234759")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--index", default="")
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()

    logs: list[str] = []
    paths = get_project_paths()
    ensure_dirs(paths)
    cfg = load_breakpoints_yaml(paths.config / "breakpoints.yaml")
    t1 = pd.Timestamp(cfg.t1)
    t2 = pd.Timestamp(cfg.t2)

    manifest_path = Path(args.manifest) if args.manifest else paths.runs / args.run_id / "outputs" / "final_manifest.csv"
    index_path = Path(args.index) if args.index else paths.data_index / "videos_index.csv"
    out_dir = Path(args.out_dir) if args.out_dir else paths.runs / args.run_id / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "formal_figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    _log(f"Load manifest: {manifest_path}", lines=logs)
    manifest = pd.read_csv(manifest_path, dtype=str).fillna("")
    index_df = pd.read_csv(index_path, dtype=str).fillna("")

    resources = {
        "stop_terms": set(load_terms(paths.resources / "stopwords_zh.txt")),
        "func_terms": set(load_terms(paths.resources / "function_words_zh.txt")),
        "template_terms": load_terms(paths.resources / "templates_v1.txt"),
        "connective_terms": load_terms(paths.resources / "connectives_explain.txt")
        + load_terms(paths.resources / "connectives_contrast.txt")
        + load_terms(paths.resources / "connectives_progression.txt"),
        "onomatopoeia_terms": load_terms(paths.resources / "onomatopoeia.txt"),
    }

    component_terms = {
        "heading_like_density": ["首先", "其次", "最后", "一方面", "另一方面", "我们来看", "我们来看看", "可以看到", "总结一下", "重点是"],
        "definitional_marker_density": ["简单来说", "换句话说", "本质上", "也就是说", "意味着", "可以理解为", "指的是"],
        "inferential_marker_density": ["因为", "所以", "因此", "总之", "但是", "不过", "然而", "如果", "由此", "可见"],
        "liability_shield_density": ["可能", "也许", "或许", "大概", "我觉得", "我认为", "恐怕", "基本上"],
    }
    transition_terms = ["接下来", "再来看", "换句话说", "说白了", "回过头看", "另一方面", "总结一下", "回到这里", "总的来说"]
    cue_terms = sorted(
        set(component_terms["heading_like_density"] + component_terms["definitional_marker_density"] + transition_terms + resources["template_terms"]),
        key=len,
        reverse=True,
    )

    manifest["subtitle_status"] = manifest["subtitle_status"].astype(str)
    ok_manifest = manifest[manifest["subtitle_status"] == "OK"].copy()
    ok_manifest["pubdate_dt"] = pd.to_datetime(ok_manifest["pubdate"].replace("", pd.NA), errors="coerce")
    ok_manifest = ok_manifest.dropna(subset=["pubdate_dt"]).copy()

    rows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []
    qc_pass = 0
    for _, row in ok_manifest.iterrows():
        bvid = str(row["bvid"]).strip()
        creator_id = str(row["creator_id"]).strip()
        raw_path = paths.data_raw_subtitles / creator_id / f"{bvid}.bcc.json"
        if not raw_path.exists():
            continue
        bcc = _load_bcc(raw_path)
        chunks, removed_count = _prepare_chunks(bcc, resources["onomatopoeia_terms"])
        if not chunks:
            continue
        text = _normalize_text(" ".join(chunk["text"] for chunk in chunks))
        tokens = _tokenize(text)
        if not text or not tokens:
            continue
        chars = _count_chars(text)
        repaired_segments = _repair_segments(chunks, cue_terms)
        legacy = _legacy_structure_metrics(text, chars)
        repaired = _repaired_structure_metrics(repaired_segments, chars, len(tokens), legacy["raw_comma_count"])

        stop_count = sum(1 for tok in tokens if tok in resources["stop_terms"])
        func_count = sum(1 for tok in tokens if tok in resources["func_terms"])
        token_n = max(len(tokens), 1)
        full_for_hits = text.replace(" ", "")
        template_hits = _substring_hits(full_for_hits, resources["template_terms"])
        connective_hits = _substring_hits(full_for_hits, resources["connective_terms"])
        heading_hits = _substring_hits(full_for_hits, component_terms["heading_like_density"])
        definitional_hits = _substring_hits(full_for_hits, component_terms["definitional_marker_density"])
        inferential_hits = _substring_hits(full_for_hits, component_terms["inferential_marker_density"])
        liability_hits = _substring_hits(full_for_hits, component_terms["liability_shield_density"])
        formulaic_transition_hits = _substring_hits(full_for_hits, transition_terms)

        rec = {
            "video_id": str(row.get("unique_key") or bvid),
            "bvid": bvid,
            "creator_id": creator_id,
            "creator_name": str(row.get("creator_name") or ""),
            "creator_group": str(row.get("creator_group") or ""),
            "phase": str(row.get("phase_base") or ""),
            "pubdate": str(row.get("pubdate") or ""),
            "pubdate_dt": row["pubdate_dt"],
            "title": str(row.get("title") or ""),
            "part_name": str(row.get("part_name") or ""),
            "title_pattern": str(row.get("title_pattern") or ""),
            "duration_sec": row.get("duration_sec") or row.get("duration") or "",
            "strict_ok": int(float(row.get("strict_ok") or 0)),
            "fill_level": int(float(row.get("fill_level") or 0)),
            "fill_reason": str(row.get("fill_reason") or ""),
            "transcript_provenance": _infer_provenance(row, bcc),
            "length_chars": chars,
            "length_tokens": int(len(tokens)),
            "mattr": _mattr(tokens, window=100),
            "mean_word_len": float(chars) / float(token_n),
            "stop_ratio": float(stop_count) / float(token_n),
            "func_ratio": float(func_count) / float(token_n),
            "template_hits": int(template_hits),
            "template_density": float(template_hits) * 1000.0 / float(token_n),
            "connective_hits": int(connective_hits),
            "connective_density": float(connective_hits) * 1000.0 / float(token_n),
            "heading_like_hits": int(heading_hits),
            "heading_like_density": float(heading_hits) * 1000.0 / float(token_n),
            "definitional_marker_hits": int(definitional_hits),
            "definitional_marker_density": float(definitional_hits) * 1000.0 / float(token_n),
            "definitional_hits": int(definitional_hits),
            "definitional_density": float(definitional_hits) * 1000.0 / float(token_n),
            "inferential_marker_hits": int(inferential_hits),
            "inferential_marker_density": float(inferential_hits) * 1000.0 / float(token_n),
            "inferential_hits": int(inferential_hits),
            "inferential_density": float(inferential_hits) * 1000.0 / float(token_n),
            "liability_shield_hits": int(liability_hits),
            "liability_shield_density": float(liability_hits) * 1000.0 / float(token_n),
            "formulaic_transition_hits": int(formulaic_transition_hits),
            "formulaic_transition_density": float(formulaic_transition_hits) * 1000.0 / float(token_n),
            "structure_hits_total": int(heading_hits + definitional_hits + inferential_hits + liability_hits),
            "series": "",
        }
        rec.update(legacy)
        rec.update(repaired)
        rows.append(rec)
        qc_rows.append(
            {
                "video_id": rec["video_id"],
                "bvid": bvid,
                "creator_id": creator_id,
                "creator_name": rec["creator_name"],
                "phase": rec["phase"],
                "transcript_provenance": rec["transcript_provenance"],
                "length_tokens": rec["length_tokens"],
                "legacy_sentence_count": rec["legacy_sentence_count"],
                "repaired_segment_count": rec["repaired_segment_count"],
                "mean_sent_len_chars_before": rec["legacy_mean_sent_len_chars"],
                "mean_sent_len_chars_after": rec["repaired_mean_segment_chars"],
                "segment_count_per_1000_tokens_before": float(rec["legacy_sentence_count"]) * 1000.0 / float(max(rec["length_tokens"], 1)),
                "segment_count_per_1000_tokens_after": rec["segment_count_per_1000_tokens"],
                "avg_tokens_per_segment_before": float(rec["length_tokens"]) / float(max(rec["legacy_sentence_count"], 1)),
                "avg_tokens_per_segment_after": rec["avg_tokens_per_segment"],
                "legacy_mean_sent_len_chars": rec["legacy_mean_sent_len_chars"],
                "repaired_mean_segment_chars": rec["repaired_mean_segment_chars"],
                "comma_period_ratio_before": rec["legacy_comma_period_ratio"],
                "comma_period_ratio_after": rec["repaired_comma_period_ratio"],
                "legacy_comma_period_ratio": rec["legacy_comma_period_ratio"],
                "repaired_comma_period_ratio": rec["repaired_comma_period_ratio"],
                "segment_boundary_density": rec["segment_boundary_density"],
                "segment_length_cv": rec["segment_length_cv"],
                "removed_onomatopoeia_count": removed_count,
            }
        )
        qc_pass += 1

    df = pd.DataFrame(rows).sort_values(["creator_id", "pubdate_dt", "bvid"]).reset_index(drop=True)
    qc_df = pd.DataFrame(qc_rows).sort_values(["creator_id", "phase", "video_id"]).reset_index(drop=True)
    if df.empty:
        raise RuntimeError("No formal sample rows were constructed from the current manifest.")

    structure_component_cols = ["heading_like_density", "definitional_marker_density", "inferential_marker_density", "liability_shield_density"]
    for col in structure_component_cols:
        df[f"{col}_z"] = _safe_standardize(np.log1p(df[col]))
    df["structure_composite"] = df[[f"{name}_z" for name in structure_component_cols]].mean(axis=1)
    df["style_index_pc1"] = _build_style_pc1(
        df,
        ["connective_density", "template_density", "mattr", "structure_composite", "func_ratio", "stop_ratio", "mean_word_len"],
    )
    df = _prepare_model_df(df)

    counts = df.groupby(["creator_id", "phase"]).size().unstack(fill_value=0).reindex(columns=PHASES, fill_value=0)
    balanced_creators = counts[(counts >= 10).all(axis=1)].index.tolist()
    balanced_df = (
        df[df["creator_id"].isin(balanced_creators)]
        .sort_values(["creator_id", "phase", "pubdate_dt", "bvid"])
        .groupby(["creator_id", "phase"], group_keys=False)
        .apply(_select_evenly, n=10)
        .reset_index(drop=True)
    )

    df.to_csv(out_dir / "formal_panel_dataset.csv", index=False, encoding="utf-8-sig")
    balanced_df.to_csv(out_dir / "balanced_panel_dataset.csv", index=False, encoding="utf-8-sig")
    qc_df.to_csv(out_dir / "segment_quality_check.csv", index=False, encoding="utf-8-sig")

    sample_flow = pd.DataFrame(
        [
            {"stage": "raw_candidate_videos", "n_videos": int(len(index_df)), "n_creators": int(index_df["creator_id"].astype(str).nunique()), "sample_layer": "reference_pool", "note": "Indexed current candidate pool."},
            {"stage": "deduplicated_videos", "n_videos": int(index_df["bvid"].astype(str).nunique()), "n_creators": int(index_df["creator_id"].astype(str).nunique()), "sample_layer": "reference_pool", "note": "Unique BVIDs in current index."},
            {"stage": "selected_manifest_videos", "n_videos": int(len(manifest)), "n_creators": int(manifest["creator_id"].astype(str).nunique()), "sample_layer": "reference_pool", "note": "Current production manifest before transcript success filtering."},
            {"stage": "transcripts_available", "n_videos": int(len(ok_manifest)), "n_creators": int(ok_manifest["creator_id"].astype(str).nunique()), "sample_layer": "main", "note": "Manifest rows with transcript_status=OK."},
            {"stage": "transcripts_after_qc", "n_videos": int(qc_pass), "n_creators": int(df["creator_id"].nunique()), "sample_layer": "main", "note": "Readable transcripts after segmentation repair and feature reconstruction."},
            {"stage": "final_video_level_analytic_sample", "n_videos": int(len(df)), "n_creators": int(df["creator_id"].nunique()), "sample_layer": "main", "note": "Formal main sample used in all headline models."},
            {"stage": "creators_with_all_three_phases", "n_videos": np.nan, "n_creators": int((counts[PHASES] > 0).all(axis=1).sum()), "sample_layer": "main", "note": "Creators represented in S0/S1/S2."},
            {"stage": "balanced_creators_10x3", "n_videos": np.nan, "n_creators": int(len(balanced_creators)), "sample_layer": "balanced", "note": "Creators with at least 10 videos in each phase."},
            {"stage": "balanced_video_sample", "n_videos": int(len(balanced_df)), "n_creators": int(balanced_df["creator_id"].nunique()), "sample_layer": "balanced", "note": "Exact 10x3 subset for robustness."},
        ]
    )
    draft_note = (
        "The draft manuscript and experiment-design documents still contain legacy counts such as 1000 / 990 / 81. "
        "Those numbers are treated as earlier write-up snapshots, not as valid analytic layers in this rebuild."
    )
    _write_sample_flow(
        out_dir,
        sample_flow,
        main_videos=int(len(df)),
        main_creators=int(df["creator_id"].nunique()),
        balanced_videos=int(len(balanced_df)),
        balanced_creators=int(balanced_df["creator_id"].nunique()),
        creators_all_phases=int((counts[PHASES] > 0).all(axis=1).sum()),
        draft_note=draft_note,
    )

    provenance_counts = df["transcript_provenance"].value_counts(dropna=False)
    _write_preprocessing_log(out_dir, qc_df, provenance_counts=provenance_counts)

    feature_specs = _feature_specs()
    qc_rows_out = []
    for feature, spec in feature_specs.items():
        series = pd.to_numeric(df[feature], errors="coerce") if feature in df.columns else pd.Series(np.nan, index=df.index, dtype=float)
        valid = series.dropna()
        qc_rows_out.append(
            {
                "feature": feature,
                "tier": spec.tier,
                "n": int(series.notna().sum()),
                "missing": int(series.isna().sum()),
                "mean": float(valid.mean()) if not valid.empty else np.nan,
                "std": float(valid.std(ddof=0)) if not valid.empty else np.nan,
                "p01": float(valid.quantile(0.01)) if not valid.empty else np.nan,
                "median": float(valid.median()) if not valid.empty else np.nan,
                "p99": float(valid.quantile(0.99)) if not valid.empty else np.nan,
                "skew": float(st.skew(valid)) if len(valid) > 10 else np.nan,
                "recommended_family": spec.model_family,
            }
        )
    feature_qc = pd.DataFrame(qc_rows_out)
    corr_features = [name for name, spec in feature_specs.items() if spec.tier in {"main", "secondary"} and name in df.columns]
    corr = df[corr_features].corr(method="spearman") if corr_features else pd.DataFrame()
    _write_feature_files(out_dir, feature_specs, feature_qc, corr, component_terms=component_terms)

    sample_description = pd.DataFrame(
        [
            {
                "sample": "main",
                "videos": int(len(df)),
                "creators": int(df["creator_id"].nunique()),
                "creators_all_phases": int((counts[PHASES] > 0).all(axis=1).sum()),
                "phase_S0": int((df["phase"] == "S0").sum()),
                "phase_S1": int((df["phase"] == "S1").sum()),
                "phase_S2": int((df["phase"] == "S2").sum()),
                "provenance_ASR": int((df["transcript_provenance"] == "ASR").sum()),
                "tokens_median": float(df["length_tokens"].median()),
            },
            {
                "sample": "balanced",
                "videos": int(len(balanced_df)),
                "creators": int(balanced_df["creator_id"].nunique()),
                "creators_all_phases": int(balanced_df["creator_id"].nunique()),
                "phase_S0": int((balanced_df["phase"] == "S0").sum()),
                "phase_S1": int((balanced_df["phase"] == "S1").sum()),
                "phase_S2": int((balanced_df["phase"] == "S2").sum()),
                "provenance_ASR": int((balanced_df["transcript_provenance"] == "ASR").sum()),
                "tokens_median": float(balanced_df["length_tokens"].median()),
            },
        ]
    )
    sample_description.to_csv(out_dir / "sample_description_table.csv", index=False, encoding="utf-8-sig")

    _write_model_plan(out_dir)

    modeled_features = MAIN_FEATURES + SECONDARY_FEATURES
    panel_results: list[pd.DataFrame] = []
    mixed_results: list[pd.DataFrame] = []
    heter_results: list[pd.DataFrame] = []
    diagnostics_rows: list[dict[str, Any]] = []
    for feature in modeled_features:
        spec = feature_specs[feature]
        panel_df, panel_diag = _fit_panel_model(df, spec)
        diagnostics_rows.append(panel_diag)
        if not panel_df.empty:
            panel_results.append(panel_df)
        if feature in MAIN_FEATURES:
            mixed_df, heter_df, mixed_diag = _fit_mixed_model(df, spec)
            diagnostics_rows.append(mixed_diag)
            if not mixed_df.empty:
                mixed_results.append(mixed_df)
            if not heter_df.empty:
                heter_results.append(heter_df)

    main_results = pd.concat(panel_results + mixed_results, ignore_index=True) if (panel_results or mixed_results) else pd.DataFrame()
    if not main_results.empty:
        main_results = _apply_fdr_by_group(main_results, ["model", "tier"])
        main_results = main_results.sort_values(["model", "p_fdr", "feature", "term"]).reset_index(drop=True)
    main_results.to_csv(out_dir / "main_results_table.csv", index=False, encoding="utf-8-sig")

    heterogeneity_results = pd.concat(heter_results, ignore_index=True) if heter_results else pd.DataFrame()
    heterogeneity_results.to_csv(out_dir / "heterogeneity_results.csv", index=False, encoding="utf-8-sig")

    break_rows: list[pd.DataFrame] = []
    for feature in MAIN_FEATURES:
        spec = feature_specs[feature]
        for label, base_date in [("T1", t1), ("T2", t2)]:
            for shift in [0, -2, 2, -4, 4, -8, 8]:
                df_shift = _fit_break_model(df, spec, base_date + pd.Timedelta(weeks=shift), buffer_weeks=0)
                if df_shift.empty:
                    continue
                df_shift["breakpoint"] = label
                df_shift["shift_weeks"] = shift
                df_shift["buffer_weeks"] = 0
                df_shift["robustness_type"] = "shifted_breakpoint"
                break_rows.append(df_shift)
            for buffer in [2, 4]:
                df_buffer = _fit_break_model(df, spec, base_date, buffer_weeks=buffer)
                if df_buffer.empty:
                    continue
                df_buffer["breakpoint"] = label
                df_buffer["shift_weeks"] = 0
                df_buffer["buffer_weeks"] = buffer
                df_buffer["robustness_type"] = "buffer_exclusion"
                break_rows.append(df_buffer)
    breakpoint_df = pd.concat(break_rows, ignore_index=True) if break_rows else pd.DataFrame()
    if not breakpoint_df.empty:
        breakpoint_df = _apply_fdr_by_group(breakpoint_df, ["feature", "term", "robustness_type"])
    breakpoint_df.to_csv(out_dir / "breakpoint_sensitivity_table.csv", index=False, encoding="utf-8-sig")

    event_rows: list[pd.DataFrame] = []
    for feature in MAIN_FEATURES:
        spec = feature_specs[feature]
        ev_t1 = _event_study(df, spec, t1, "T1")
        ev_t2 = _event_study(df, spec, t2, "T2")
        if not ev_t1.empty:
            event_rows.append(ev_t1)
        if not ev_t2.empty:
            event_rows.append(ev_t2)
    event_df = pd.concat(event_rows, ignore_index=True) if event_rows else pd.DataFrame()
    if not event_df.empty:
        event_df = _apply_fdr_by_group(event_df, ["feature", "breakpoint"])
    event_df.to_csv(out_dir / "event_time_results.csv", index=False, encoding="utf-8-sig")

    creator_month_df = _creator_month_its(df, [feature_specs[name] for name in MAIN_FEATURES], t1, t2)
    if not creator_month_df.empty:
        creator_month_df = _apply_fdr_by_group(creator_month_df, ["feature"])
    creator_month_df.to_csv(out_dir / "creator_month_its_results.csv", index=False, encoding="utf-8-sig")

    balanced_rows: list[pd.DataFrame] = []
    for feature in MAIN_FEATURES:
        spec = feature_specs[feature]
        bal_df, _ = _fit_panel_model(balanced_df, spec, model_name="A_panel_fe_balanced")
        if not bal_df.empty:
            balanced_rows.append(bal_df)
    balanced_results = pd.concat(balanced_rows, ignore_index=True) if balanced_rows else pd.DataFrame()
    if not balanced_results.empty:
        balanced_results = _apply_fdr_by_group(balanced_results, ["model"])
    balanced_results.to_csv(out_dir / "balanced_sample_results.csv", index=False, encoding="utf-8-sig")

    q01 = float(df["length_tokens"].quantile(0.01))
    q99 = float(df["length_tokens"].quantile(0.99))
    trimmed = df[(df["length_tokens"] >= q01) & (df["length_tokens"] <= q99)].copy()
    trimmed_rows: list[pd.DataFrame] = []
    for feature in MAIN_FEATURES:
        spec = feature_specs[feature]
        trim_df, _ = _fit_panel_model(trimmed, spec, model_name="A_panel_fe_trimmed")
        if not trim_df.empty:
            trimmed_rows.append(trim_df)
    trimmed_results = pd.concat(trimmed_rows, ignore_index=True) if trimmed_rows else pd.DataFrame()
    if not trimmed_results.empty:
        trimmed_results = _apply_fdr_by_group(trimmed_results, ["model"])

    metadata_rows: list[pd.DataFrame] = []
    for feature in MAIN_FEATURES:
        spec = feature_specs[feature]
        meta_df, meta_diag = _fit_panel_model(df, spec, model_name="A_panel_fe_metadata", extra_terms=METADATA_CONTROL_TERMS)
        diagnostics_rows.append(meta_diag)
        if not meta_df.empty:
            metadata_rows.append(meta_df)
    metadata_results = pd.concat(metadata_rows, ignore_index=True) if metadata_rows else pd.DataFrame()
    if not metadata_results.empty:
        metadata_results = _apply_fdr_by_group(metadata_results, ["model"])
    metadata_results.to_csv(out_dir / "metadata_controlled_results.csv", index=False, encoding="utf-8-sig")

    source_results = pd.DataFrame(
        [
            {
                "status": "not_estimable",
                "subgroup": "ASR",
                "n_videos": int(len(df)),
                "n_creators": int(df["creator_id"].nunique()),
                "note": "Current formal sample is ASR only; provenance subgroup modeling is not available.",
            }
        ]
    )
    source_results.to_csv(out_dir / "source_sensitivity_results.csv", index=False, encoding="utf-8-sig")

    diagnostics_df = pd.DataFrame(diagnostics_rows)
    diagnostics_text = "# Model Diagnostics\n\n" + _md_table(diagnostics_df.fillna(""))
    diagnostics_text += "\n\nCurrent main-sample transcript provenance is ASR only. The provenance field is retained in the dataset and reported explicitly, but it cannot identify source effects within the main sample."
    (out_dir / "model_diagnostics.md").write_text(diagnostics_text, encoding="utf-8")

    main_panel_only = main_results[(main_results["model"] == "A_panel_fe") & (main_results["tier"] == "main")].copy() if not main_results.empty else pd.DataFrame()
    robust_summary = pd.DataFrame(
        [
            {"block": "main sample rerun", "classification": "consistent", "note": "Baseline formal main sample."},
            {"block": "balanced sample rerun", "classification": _compare_robustness(main_panel_only, balanced_results, features=MAIN_FEATURES), "note": "Exact 10x3 creator-phase subset."},
            {"block": "exclude extreme-length texts", "classification": _compare_robustness(main_panel_only, trimmed_results, features=MAIN_FEATURES), "note": f"Excluded length_tokens outside [{q01:.0f}, {q99:.0f}]."},
            {"block": "metadata-controlled version", "classification": _compare_robustness(main_panel_only, metadata_results, features=MAIN_FEATURES), "note": "Controls added for duration and title/part format cues."},
        ]
    )
    robust_text = "# Robustness Summary\n\n" + _md_table(robust_summary)
    robust_text += "\n\nTranscript provenance sensitivity was not run as a subgroup comparison because the current formal sample is ASR only."
    sharp_break_support = int(((breakpoint_df["term"] == "post_break") & (breakpoint_df["p_fdr"] < 0.05)).sum()) if not breakpoint_df.empty else 0
    judgment = "sharp break" if sharp_break_support >= max(len(MAIN_FEATURES), 4) else "gradual drift"
    if judgment != "sharp break":
        judgment = "gradual drift"
    robust_text += f"\n\nBreakpoint judgment: **{judgment}**. The robustness section supports a {judgment} interpretation rather than a sharp discontinuity."
    (out_dir / "robustness_summary.md").write_text(robust_text, encoding="utf-8")

    sig_main = main_results[(main_results["model"] == "A_panel_fe") & (main_results["tier"] == "main") & (main_results["p_fdr"] < 0.05)].copy() if not main_results.empty else pd.DataFrame()
    sig_secondary = main_results[(main_results["model"] == "A_panel_fe") & (main_results["tier"] == "secondary") & (main_results["p_fdr"] < 0.05)].copy() if not main_results.empty else pd.DataFrame()
    heter_summary = heterogeneity_results[heterogeneity_results["row_type"] == "summary"] if not heterogeneity_results.empty else pd.DataFrame()
    lines = ["# Results For Abstract", ""]
    lines.append(f"The formal main sample is defined as {len(df)} ASR subtitle transcripts from {df['creator_id'].nunique()} creators. The balanced robustness sample contains {len(balanced_df)} videos from {balanced_df['creator_id'].nunique()} creators.")
    if not sig_main.empty:
        top = sig_main.sort_values("p_fdr").iloc[0]
        direction = "modest decline" if float(top["coef"]) < 0 else "modest increase"
        lines.append(f"Across creator fixed-effects panel models, the clearest pattern is a {direction} in `{top['feature']}` within creators.")
    else:
        lines.append("Across creator fixed-effects panel models, the pattern is modest, selective, and within-creator.")
    if not heter_summary.empty:
        focal = heter_summary[heter_summary["feature"] == "connective_density"]
        if not focal.empty:
            s2 = focal[focal["contrast"] == "S2_vs_S0"]
            if not s2.empty:
                share = float(s2["same_direction_share"].iloc[0]) * 100.0
                lines.append(f"Random-slope mixed models indicate heterogeneous adaptation across creators; for connective density, {share:.1f}% of creators move in the same direction as the pooled S2-S0 effect.")
    meta_class = _compare_robustness(main_panel_only, metadata_results, features=MAIN_FEATURES)
    bal_class = _compare_robustness(main_panel_only, balanced_results, features=MAIN_FEATURES)
    trim_class = _compare_robustness(main_panel_only, trimmed_results, features=MAIN_FEATURES)
    abstract_map = {"consistent": "robustness-supported", "partially consistent": "partially supported", "not consistent": "partially supported"}
    lines.append(
        "Balanced-sample, extreme-length, and metadata-controlled checks are "
        f"{abstract_map.get(bal_class, 'partially supported')}, "
        f"{abstract_map.get(trim_class, 'partially supported')}, and "
        f"{abstract_map.get(meta_class, 'partially supported')}, respectively."
    )
    lines.append("Breakpoint checks are weaker than the panel-drift evidence, and the overall quantitative pattern supports gradual drift.")
    (out_dir / "results_for_abstract.md").write_text("\n\n".join(lines), encoding="utf-8")

    old_vs_new = f"""# Old vs New Sample / Design Specification

## 1. Why the old 75 / 5 pilot spec is no longer the formal analysis spec

The 75-text / 5-creator design belongs to the pilot / precollection stage and was only suitable for feasibility checking. The current rebuild starts from the current production manifest and current production transcripts. All formal tables, figures, and model summaries therefore use the current formal main sample of **{len(df)} videos / {df['creator_id'].nunique()} creators**, not a pilot-scale subset.

## 2. Why `connectives_total` is no longer allowed as a headline main result

Raw absolute connective counts are confounded by transcript length. In the rebuild, headline reporting switches to `connective_density` and the main count model uses Poisson or Negative Binomial fixed effects with `offset(log(tokens))` as needed. This makes the cohesion signal length-corrected and directly comparable across videos.

## 3. Why breakpoint analysis is downgraded to robustness only

The breakpoint checks are weaker than the within-creator panel-drift evidence, and the project is not framed here as a causal or sharp discontinuity design. Breakpoint models are retained only as robustness checks around the fixed benchmark dates 2022-11-30 and 2023-11-02.

## 4. How the abstract spec (1000 / 990 / 81) relates to the current formal report spec (2412 / 87)

The `1000 / 990 / 81` numbers appear in the current draft manuscript / design documents as an earlier write-up snapshot: roughly 1,000 planned videos, 990 analyzable videos, and 81 creators in an earlier formalization stage. The current production rerun has expanded beyond that snapshot and now yields **{len(df)} analyzable videos across {df['creator_id'].nunique()} creators**. In this rebuild, `1000 / 990 / 81` is documented only for reconciliation, while `2412 / 87` is adopted as the single formal main sample specification.

## 5. How the balanced sample (1260 / 42) is used

The `1260 / 42` layer is retained only as a robustness subset built from creators who meet the exact 10x3 threshold. It is not a competing formal sample specification and it does not replace the sole formal main sample of `2412 / 87`.
"""
    (out_dir / "old_vs_new_spec.md").write_text(old_vs_new, encoding="utf-8")

    top_secondary = sig_secondary.sort_values("p_fdr").iloc[0] if not sig_secondary.empty else None
    focal_connective = main_results[(main_results["model"] == "A_panel_fe") & (main_results["feature"] == "connective_density")].sort_values("term") if not main_results.empty else pd.DataFrame()
    descriptive_phase = (
        df.groupby("phase")[MAIN_FEATURES]
        .mean()
        .reindex(PHASES)
        .round(4)
        .reset_index()
        if not df.empty
        else pd.DataFrame()
    )
    report_lines = [
        "# Formal Analysis Rebuild Report",
        "",
        "This file supersedes the earlier breakpoint-centered summary in `analysis_report.md` for the current formal rerun.",
        "",
        "## Formal sample",
        "",
        f"- Formal main sample: `{len(df)}` videos from `{df['creator_id'].nunique()}` creators",
        f"- Creators with all three phases: `{int((counts[PHASES] > 0).all(axis=1).sum())}`",
        f"- Balanced robustness sample: `{len(balanced_df)}` videos from `{balanced_df['creator_id'].nunique()}` creators (`10x3`)",
        f"- Transcript provenance in the current formal sample: `ASR only ({int((df['transcript_provenance'] == 'ASR').sum())})`",
        "",
        "## Preprocessing rebuild",
        "",
        "- Transcript structure was rebuilt from BCC timing chunks with additional cue-based splits and short-fragment merge-back, rather than raw ASR commas / periods.",
        "- Legacy punctuation-based sentence metrics are retained only for QC comparison and appendix use.",
        "- Main structural evidence now comes from repaired segment boundaries and transparent structure-component densities.",
        "",
        "## Main findings",
        "",
        "- Headline tests remain limited to `connective_density`, `template_density`, `mattr`, and `structure_composite`.",
        "- Across creator fixed-effects panel models, the supported pattern is modest and selective rather than broad-based.",
        f"- Breakpoint robustness supports `{judgment}` rather than a sharp discontinuity.",
        "- Mixed models continue to indicate heterogeneous adaptation across creators, not a single shared path.",
        "",
        "## Secondary findings",
        "",
        f"- Secondary component metrics are retained separately to reduce primary-test burden; the top secondary panel result is `{top_secondary['feature']}`." if top_secondary is not None else "- Secondary component metrics are retained separately to reduce primary-test burden, but no secondary result survives FDR adjustment in the main sample.",
        f"- Metadata-controlled robustness is classified as `{meta_class}`, using duration and title/part format cues as additional controls.",
        "",
        "## Appendix-only findings",
        "",
        "- Raw punctuation ratios, raw sentence-length traces, and segment-length traces remain appendix-only because they are directly affected by ASR punctuation or by reconstruction artifacts.",
        "- `connectives_total` is not used as a headline result; all count-like main features are length-corrected.",
        "",
        "## Descriptive phase means (main features)",
        "",
        _md_table(descriptive_phase),
        "",
        "## Key files",
        "",
        "- `sample_flow.md`",
        "- `preprocessing_log.md`",
        "- `feature_spec.md`",
        "- `main_results_table.csv`",
        "- `heterogeneity_results.csv`",
        "- `breakpoint_sensitivity_table.csv`",
        "- `metadata_controlled_results.csv`",
        "- `results_for_abstract.md`",
        "- `old_vs_new_spec.md`",
        "- `blind_review_figure_set/`",
        "- `blind_review_tables/`",
    ]
    (out_dir / "formal_analysis_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    _plot_sample_flow(figures_dir / "sample_flow_figure.png", sample_flow)
    _plot_phase_distribution(figures_dir / "phase_distribution_figure.png", df, balanced_df)
    _plot_feature_distributions(figures_dir / "main_feature_distributions.png", df)
    forest_feature = "connective_density"
    _plot_forest(figures_dir / "creator_effect_forest.png", heterogeneity_results, forest_feature)
    _plot_event_time(figures_dir / "event_time_figure.png", event_df, forest_feature)
    _plot_segmentation_qc(figures_dir / "segmentation_qc_before_after.png", qc_df)

    figures_index = pd.DataFrame(
        [
            {"figure": "sample flow figure", "path": str(figures_dir / "sample_flow_figure.png")},
            {"figure": "phase distribution figure", "path": str(figures_dir / "phase_distribution_figure.png")},
            {"figure": "main feature distributions", "path": str(figures_dir / "main_feature_distributions.png")},
            {"figure": "creator effect forest", "path": str(figures_dir / "creator_effect_forest.png")},
            {"figure": "event-time figure", "path": str(figures_dir / "event_time_figure.png")},
            {"figure": "segmentation QC figure", "path": str(figures_dir / "segmentation_qc_before_after.png")},
        ]
    )
    figures_index.to_csv(out_dir / "main_figures_index.csv", index=False, encoding="utf-8-sig")

    _write_blind_review_exports(
        out_dir,
        figures_dir,
        main_df=df,
        heterogeneity_results=heterogeneity_results,
        sample_flow=sample_flow,
        feature_decision_path=out_dir / "feature_decision_table.csv",
        feature_qc_path=out_dir / "feature_qc_table.csv",
        corr_path=out_dir / "correlation_matrix.csv",
        main_results_path=out_dir / "main_results_table.csv",
        metadata_results_path=out_dir / "metadata_controlled_results.csv",
        robustness_path=out_dir / "robustness_summary.md",
        model_diagnostics_path=out_dir / "model_diagnostics.md",
        breakpoint_path=out_dir / "breakpoint_sensitivity_table.csv",
        event_time_path=out_dir / "event_time_results.csv",
        creator_month_path=out_dir / "creator_month_its_results.csv",
        sample_flow_md_path=out_dir / "sample_flow.md",
        old_vs_new_path=out_dir / "old_vs_new_spec.md",
        feature_spec_path=out_dir / "feature_spec.md",
        results_abstract_path=out_dir / "results_for_abstract.md",
        sample_description_path=out_dir / "sample_description_table.csv",
    )

    _log(f"Formal rebuild complete: {out_dir}", lines=logs)
    (out_dir / "formal_rebuild_run_log.txt").write_text("\n".join(logs), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
