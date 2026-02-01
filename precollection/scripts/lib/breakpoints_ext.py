from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BreakpointsConfig:
    t1: date
    t2: date
    window_years: int
    shift_weeks: list[int]
    buffer_weeks: list[int]
    default_buffer_weeks: int
    duration_min_sec: int
    duration_max_sec: int
    exclude_title_keywords: list[str]
    batch_size: int
    batch_cooldown_sec: int
    mid_cooldown_sec: int
    sleep_range_sec: list[float]
    max_retries: int


def _parse_yaml_simple(text: str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        key = key.strip()
        val = val.strip()
        if val.startswith("[") and val.endswith("]"):
            inner = val[1:-1].strip()
            if not inner:
                data[key] = []
            else:
                items = [x.strip().strip('"').strip("'") for x in inner.split(",")]
                # try int
                out: list[Any] = []
                for it in items:
                    try:
                        out.append(int(it))
                    except Exception:
                        out.append(it)
                data[key] = out
        else:
            v = val.strip('"').strip("'")
            data[key] = v
    return data


def load_breakpoints_yaml(path: Path) -> BreakpointsConfig:
    raw = path.read_text(encoding="utf-8")
    data: dict[str, Any]
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(raw)
        if not isinstance(data, dict):
            data = {}
    except Exception:
        data = _parse_yaml_simple(raw)

    def _get_int(key: str, default: int) -> int:
        v = data.get(key, default)
        try:
            return int(v)
        except Exception:
            return default

    def _get_list_int(key: str, default: list[int]) -> list[int]:
        v = data.get(key, default)
        if isinstance(v, list):
            out: list[int] = []
            for item in v:
                try:
                    out.append(int(item))
                except Exception:
                    continue
            return out
        if isinstance(v, str):
            return [int(x.strip()) for x in v.strip("[]").split(",") if x.strip().lstrip("-").isdigit()]
        return default

    def _get_list_str(key: str, default: list[str]) -> list[str]:
        v = data.get(key, default)
        if isinstance(v, list):
            return [str(x) for x in v if str(x)]
        if isinstance(v, str):
            return [s.strip() for s in v.strip("[]").split(",") if s.strip()]
        return default

    t1 = date.fromisoformat(str(data.get("t1_chatgpt", "2022-11-30")))
    t2_raw = str(data.get("t2_deepseek", "YYYY-MM-DD"))
    # allow placeholder
    t2 = date.fromisoformat(t2_raw) if t2_raw != "YYYY-MM-DD" else date.fromisoformat("2023-11-02")

    batching = data.get("batching") if isinstance(data, dict) else {}
    if not isinstance(batching, dict):
        batching = {}

    def _get_float_list(key: str, default: list[float]) -> list[float]:
        v = batching.get(key, data.get(key, default))
        if isinstance(v, list):
            out: list[float] = []
            for item in v:
                try:
                    out.append(float(item))
                except Exception:
                    continue
            return out or default
        if isinstance(v, str):
            items = [x.strip() for x in v.strip("[]").split(",") if x.strip()]
            out = []
            for it in items:
                try:
                    out.append(float(it))
                except Exception:
                    continue
            return out or default
        return default

    return BreakpointsConfig(
        t1=t1,
        t2=t2,
        window_years=_get_int("window_years", 6),
        shift_weeks=_get_list_int("shift_weeks", [0, -2, 2, -4, 4, -8, 8]),
        buffer_weeks=_get_list_int("buffer_weeks", [0, 2, 4]),
        default_buffer_weeks=_get_int("default_buffer_weeks", 2),
        duration_min_sec=_get_int("duration_min_sec", 300),
        duration_max_sec=_get_int("duration_max_sec", 1800),
        exclude_title_keywords=_get_list_str(
            "exclude_title_keywords",
            ["直播", "回放", "合集", "联动", "广告", "宣传"],
        ),
        batch_size=_get_int("batch_size", int(batching.get("batch_size", 10)) if batching else 10),
        batch_cooldown_sec=_get_int("batch_cooldown_sec", int(batching.get("batch_cooldown_sec", 300)) if batching else 300),
        mid_cooldown_sec=_get_int("mid_cooldown_sec", int(batching.get("mid_cooldown_sec", 120)) if batching else 120),
        sleep_range_sec=_get_float_list("sleep_range_sec", [1.2, 2.5]),
        max_retries=_get_int("max_retries", int(batching.get("max_retries", 3)) if batching else 3),
    )


def compute_phase(pub_date: date, *, t1: date, t2: date) -> str:
    if pub_date <= t1:
        return "S0"
    if pub_date < t2:
        return "S1"
    return "S2"


def shift_date(d: date, weeks: int) -> date:
    return d + timedelta(weeks=weeks)


def in_buffer(pub_date: date, *, t1: date, t2: date, buffer_weeks: int) -> bool:
    if buffer_weeks <= 0:
        return False
    delta = timedelta(weeks=buffer_weeks)
    return (t1 - delta) <= pub_date <= (t1 + delta) or (t2 - delta) <= pub_date <= (t2 + delta)
