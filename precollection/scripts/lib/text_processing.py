from __future__ import annotations

import re
from dataclasses import dataclass


_RE_CJK = re.compile(r"[\u4e00-\u9fff]")
_RE_ASCII_ALNUM = re.compile(r"[A-Za-z0-9]")
_RE_ASCII_TOKEN = re.compile(r"[A-Za-z0-9]+")


_PUNCT_TRANSLATION = str.maketrans(
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
        "､": "，",
        "﹐": "，",
        "﹑": "，",
        "。": "。",
        "｡": "。",
        "．": ".",
    }
)

_RE_ELLIPSIS = re.compile(r"(…{1,}|\.{3,})")
_RE_NON_DECIMAL_DOT = re.compile(r"(?<!\\d)\\.(?!\\d)")
_RE_REPEAT_PUNCT = re.compile(r"([，。！？；：])\\1{1,}")
_RE_SPACES = re.compile(r"[ \\t\\u00A0\\u3000]+")
_RE_MULTI_NEWLINES = re.compile(r"\\n{3,}")


_PUNCTS = set("，。！？；：、（）【】《》“”‘’—…·～-")


@dataclass(frozen=True)
class RemoveResult:
    text: str
    removed_total: int


def remove_terms(text: str, terms: list[str]) -> RemoveResult:
    removed_total = 0
    out = text
    for t in terms:
        if not t:
            continue
        c = out.count(t)
        if c:
            removed_total += c
            out = out.replace(t, "")
    return RemoveResult(text=out, removed_total=removed_total)


def normalize_punctuation(text: str) -> str:
    out = text
    out = out.replace("\r\n", "\n").replace("\r", "\n")
    out = _RE_ELLIPSIS.sub("……", out)
    out = out.translate(_PUNCT_TRANSLATION)
    out = _RE_NON_DECIMAL_DOT.sub("。", out)
    out = _RE_REPEAT_PUNCT.sub(r"\1", out)
    return out


def normalize_whitespace(text: str) -> str:
    out = text
    out = out.replace("\r\n", "\n").replace("\r", "\n")
    out = _RE_SPACES.sub(" ", out)
    out = "\n".join([ln.strip() for ln in out.split("\n")])
    out = _RE_MULTI_NEWLINES.sub("\n\n", out)
    return out.strip() + "\n" if out.strip() else ""


def clean_text(text: str, *, onomatopoeia_terms: list[str]) -> tuple[str, int]:
    removed = remove_terms(text, onomatopoeia_terms)
    out = normalize_punctuation(removed.text)
    out = normalize_whitespace(out)
    return out, removed.removed_total


def count_chars(text: str) -> int:
    # 字数：CJK 字符 + ASCII 字母数字（逐字符）
    return len(_RE_CJK.findall(text)) + len(_RE_ASCII_ALNUM.findall(text))


def count_tokens(text: str) -> int:
    # 预收集 token：每个 CJK 字符算 1；连续 ASCII 字母数字算 1 token
    return len(_RE_CJK.findall(text)) + len(_RE_ASCII_TOKEN.findall(text))


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"[。！？!?]+", text)
    return [p.strip() for p in parts if p.strip()]


def count_punct(text: str) -> int:
    return sum(1 for ch in text if ch in _PUNCTS)


def substring_hits(text: str, terms: list[str]) -> int:
    return sum(text.count(t) for t in terms if t)

