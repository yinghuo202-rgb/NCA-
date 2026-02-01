from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

from .http import get_json


@dataclass(frozen=True)
class VideoMeta:
    bvid: str
    aid: int
    cid: int
    pages_count: int
    page_index: int
    part_name: str
    page_duration: int
    title: str
    pub_datetime: datetime  # Asia/Shanghai (UTC+8), 固定换算，避免机器时区差异
    duration_sec: int
    up_mid: int | None
    up_name: str | None


def _dt_from_bili_pubdate(pubdate_ts: int) -> datetime:
    # bilibili 的 pubdate 是 epoch seconds；用 UTC+8 固定换算保证可复现
    dt_utc = datetime.fromtimestamp(pubdate_ts, tz=timezone.utc)
    return (dt_utc + timedelta(hours=8)).replace(tzinfo=timezone(timedelta(hours=8)))


def fetch_view(sess: requests.Session, bvid: str) -> dict[str, Any]:
    return get_json(
        sess,
        "https://api.bilibili.com/x/web-interface/view",
        params={"bvid": bvid},
        headers={"Referer": "https://www.bilibili.com/"},
    )


def fetch_video_meta(sess: requests.Session, bvid: str) -> VideoMeta:
    payload = fetch_view(sess, bvid)
    if payload.get("code") != 0:
        raise RuntimeError(f"view API code={payload.get('code')} message={payload.get('message')}")

    data = payload["data"]
    pages = data.get("pages") or []
    if not pages:
        raise RuntimeError("view API 缺少 pages/cid")

    # 多P：选择时长最长的分P作为“主分P”，避免 pages[0] 是片头/预告导致 cid 不对应
    best_idx = 0
    best_dur = -1
    for idx, p in enumerate(pages):
        try:
            dur = int(p.get("duration") or 0)
        except Exception:
            dur = 0
        if dur > best_dur:
            best_dur = dur
            best_idx = idx
    best_page = pages[best_idx]

    owner = data.get("owner") or {}
    return VideoMeta(
        bvid=bvid,
        aid=int(data["aid"]),
        cid=int(best_page["cid"]),
        pages_count=len(pages),
        page_index=int(best_idx),
        part_name=str(best_page.get("part") or ""),
        page_duration=int(best_page.get("duration") or 0),
        title=str(data.get("title") or ""),
        pub_datetime=_dt_from_bili_pubdate(int(data["pubdate"])),
        duration_sec=int(data.get("duration") or 0),
        up_mid=int(owner["mid"]) if owner.get("mid") is not None else None,
        up_name=str(owner["name"]) if owner.get("name") is not None else None,
    )


def fetch_player(sess: requests.Session, aid: int, cid: int, *, bvid_for_referer: str) -> dict[str, Any]:
    return get_json(
        sess,
        "https://api.bilibili.com/x/player/v2",
        params={"aid": aid, "cid": cid},
        headers={"Referer": f"https://www.bilibili.com/video/{bvid_for_referer}"},
    )


def fetch_player_wbi(sess: requests.Session, aid: int, cid: int, *, bvid_for_referer: str) -> dict[str, Any]:
    return get_json(
        sess,
        "https://api.bilibili.com/x/player/wbi/v2",
        params={"aid": aid, "cid": cid},
        headers={"Referer": f"https://www.bilibili.com/video/{bvid_for_referer}"},
    )
