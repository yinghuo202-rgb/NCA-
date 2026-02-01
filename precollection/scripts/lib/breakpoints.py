from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass(frozen=True)
class Breakpoints:
    t1: date
    t2: date


def load_breakpoints(path: Path) -> Breakpoints:
    data = json.loads(path.read_text(encoding="utf-8"))
    return Breakpoints(
        t1=date.fromisoformat(data["T1"]),
        t2=date.fromisoformat(data["T2"]),
    )


def compute_stage(pub_date: date, breakpoints: Breakpoints) -> str:
    if pub_date <= breakpoints.t1:
        return "S0"
    if pub_date < breakpoints.t2:
        return "S1"
    return "S2"
