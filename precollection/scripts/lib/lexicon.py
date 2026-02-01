from __future__ import annotations

from pathlib import Path


def load_terms(path: Path) -> list[str]:
    terms: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        terms.append(s)
    # 去重但保序
    seen: set[str] = set()
    dedup: list[str] = []
    for t in terms:
        if t in seen:
            continue
        seen.add(t)
        dedup.append(t)
    return dedup

