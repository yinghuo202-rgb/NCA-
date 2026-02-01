from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from lib.breakpoints_ext import BreakpointsConfig, compute_phase, in_buffer, load_breakpoints_yaml, shift_date
from lib.http import get_json, load_cookies_json, requests_session
from lib.paths import ensure_dirs, get_project_paths
from lib.run_utils import init_run_dirs, make_run_id, write_params_yaml
from lib.wbi import sign_wbi_params


def _append_run_log(path: Path, lines: list[str]) -> None:
    existing = ""
    if path.exists():
        try:
            existing = path.read_text(encoding="utf-8-sig")
        except Exception:
            existing = path.read_text(encoding="utf-8", errors="ignore")
    merged = (existing.rstrip("\n") + "\n" + "\n".join(lines)).strip() + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(merged, encoding="utf-8-sig")


def _log(lines: list[str], msg: str) -> None:
    stamp = datetime.now().isoformat(timespec="seconds")
    line = f"[{stamp}] {msg}"
    print(line)
    lines.append(line)


def _parse_duration_to_sec(s: str) -> int:
    if not s:
        return 0
    parts = s.split(":")
    try:
        if len(parts) == 3:
            h, m, sec = parts
            return int(h) * 3600 + int(m) * 60 + int(sec)
        if len(parts) == 2:
            m, sec = parts
            return int(m) * 60 + int(sec)
        return int(float(s))
    except Exception:
        return 0


def _fetch_creator_page(sess, *, mid: str, pn: int, ps: int) -> dict[str, Any]:
    base_params = {"mid": mid, "pn": pn, "ps": ps, "order": "pubdate"}
    headers = {"Referer": f"https://space.bilibili.com/{mid}/video"}
    try:
        signed = sign_wbi_params(sess, base_params)
        payload = get_json(
            sess,
            "https://api.bilibili.com/x/space/wbi/arc/search",
            params=signed,
            headers=headers,
        )
        if payload.get("code") in {-352, -403, -412}:
            raise RuntimeError(f"wbi_blocked code={payload.get('code')}")
        return payload
    except Exception:
        return get_json(
            sess,
            "https://api.bilibili.com/x/space/arc/search",
            params=base_params,
            headers=headers,
        )


def _cache_path(cache_dir: Path, mid: str, pn: int) -> Path:
    return cache_dir / f"{mid}_pn{pn}.json"


def _load_page_cached(
    sess,
    *,
    mid: str,
    pn: int,
    ps: int,
    cache_dir: Path,
    use_cache: bool,
    sleep_sec: float,
    retries: int,
) -> dict[str, Any]:
    cache_path = _cache_path(cache_dir, mid, pn)
    if use_cache and cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    err: str | None = None
    for attempt in range(retries):
        try:
            payload = _fetch_creator_page(sess, mid=mid, pn=pn, ps=ps)
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            return payload
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
            time.sleep(sleep_sec + attempt * 0.8)

    raise RuntimeError(err or "fetch failed")


def _fetch_bvids_via_ytdlp(
    *,
    mid: str,
    sleep_range: list[float],
    cookies_netscape: Path | None,
    timeout_sec: int = 1200,
    socket_timeout_sec: int = 10,
    retries: int = 2,
) -> tuple[list[str], str]:
    url = f"https://space.bilibili.com/{mid}/video"
    cmd = ["yt-dlp", "--flat-playlist", "--dump-json", "--ignore-errors", "--no-warnings", url]
    if socket_timeout_sec > 0:
        cmd += ["--socket-timeout", str(int(socket_timeout_sec))]
    if retries >= 0:
        cmd += ["--retries", str(int(retries))]
    if sleep_range:
        lo = float(min(sleep_range))
        hi = float(max(sleep_range))
        cmd += ["--sleep-interval", f"{max(lo, 0.5):.2f}", "--max-sleep-interval", f"{max(hi, lo, 0.5):.2f}"]
    if cookies_netscape and cookies_netscape.exists():
        cmd += ["--cookies", str(cookies_netscape)]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec, check=False)
    except subprocess.TimeoutExpired as e:
        return [], f"TimeoutExpired: {e}"
    bvids: list[str] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        bvid = str(obj.get("id") or "").strip()
        if bvid and bvid.startswith("BV"):
            bvids.append(bvid)

    return bvids, (proc.stderr or "").strip()


def _collect_creator_rows_api(
    sess,
    *,
    creator_id: str,
    creator_name: str,
    creator_group: str,
    cfg: BreakpointsConfig,
    window_start: date,
    page_size: int,
    cache_dir: Path,
    use_cache: bool,
    sleep_range: list[float],
    max_retries: int,
    mid_cooldown_sec: int,
    blocked_counts: dict[str, int],
    failed_rows: list[dict[str, Any]],
    log_lines: list[str],
) -> tuple[list[dict[str, Any]], bool]:
    creator_rows: list[dict[str, Any]] = []
    pn = 1
    blocked = False
    attempts = 0
    while True:
        oldest_pub_date: date | None = None
        attempts += 1
        try:
            payload = _load_page_cached(
                sess,
                mid=creator_id,
                pn=pn,
                ps=page_size,
                cache_dir=cache_dir,
                use_cache=use_cache,
                sleep_sec=_sleep_jitter(sleep_range),
                retries=max_retries,
            )
        except Exception as e:  # noqa: BLE001
            failed_rows.append(
                {
                    "mid": creator_id,
                    "creator_id": creator_id,
                    "creator_name": creator_name,
                    "code": "",
                    "status": "FAILED",
                    "cooldown_sec": "",
                    "page": pn,
                    "message": f"{type(e).__name__}: {e}",
                    "attempts": attempts,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                }
            )
            _log(log_lines, f"FAILED: creator_id={creator_id} page={pn} error={type(e).__name__}: {e}")
            break

        code = payload.get("code")
        if code == -799:
            blocked_counts[creator_id] = blocked_counts.get(creator_id, 0) + 1
            is_deferred = blocked_counts[creator_id] > 1
            cooldown = 600 if is_deferred else mid_cooldown_sec
            failed_rows.append(
                {
                    "mid": creator_id,
                    "creator_id": creator_id,
                    "creator_name": creator_name,
                    "code": code,
                    "status": "DEFERRED" if is_deferred else "TEMP_BLOCKED",
                    "cooldown_sec": cooldown,
                    "page": pn,
                    "attempts": attempts,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                }
            )
            _log(
                log_lines,
                f"{'DEFERRED' if is_deferred else 'TEMP_BLOCKED'}: creator_id={creator_id} page={pn} code=-799",
            )
            blocked = True
            time.sleep(cooldown)
            break

        if code != 0:
            failed_rows.append(
                {
                    "mid": creator_id,
                    "creator_id": creator_id,
                    "creator_name": creator_name,
                    "code": code,
                    "status": "ERROR",
                    "cooldown_sec": "",
                    "page": pn,
                    "message": payload.get("message"),
                    "attempts": attempts,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                }
            )
            _log(log_lines, f"ERROR: creator_id={creator_id} page={pn} code={code} msg={payload.get('message')}")
            break

        data = payload.get("data") or {}
        vlist = ((data.get("list") or {}).get("vlist") or []) if isinstance(data, dict) else []
        if not vlist:
            break

        for v in vlist:
            if not isinstance(v, dict):
                continue
            bvid = str(v.get("bvid", "")).strip()
            if not bvid:
                continue
            pub_ts = int(v.get("created") or 0)
            pub_date = datetime.fromtimestamp(pub_ts).date() if pub_ts else None
            if not pub_date:
                continue
            if oldest_pub_date is None or pub_date < oldest_pub_date:
                oldest_pub_date = pub_date
            if not _within_window(pub_date, window_years=cfg.window_years, window_start=window_start):
                continue

            row = {
                "creator_id": creator_id,
                "creator_name": creator_name,
                "creator_group": creator_group,
                "bvid": bvid,
                "aid": str(v.get("aid", "")),
                "pubdate": pub_date.isoformat(),
                "title": str(v.get("title", "")),
                "desc": str(v.get("description", "")),
                "tname": str(v.get("typename", "")),
                "duration": _parse_duration_to_sec(str(v.get("length", ""))),
                "part_count": "",
            }
            row.update(_compute_phase_columns(pub_date, cfg=cfg))
            creator_rows.append(row)

        if oldest_pub_date is not None and oldest_pub_date < window_start:
            break

        pn += 1
        time.sleep(_sleep_jitter(sleep_range))

    return creator_rows, blocked


def _fetch_view_meta(sess, *, bvid: str, sleep_sec: float, retries: int) -> dict[str, Any]:
    err: str | None = None
    headers = {"Referer": f"https://www.bilibili.com/video/{bvid}"}
    for attempt in range(max(retries, 1)):
        try:
            return get_json(
                sess,
                "https://api.bilibili.com/x/web-interface/view",
                params={"bvid": bvid},
                headers=headers,
            )
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
            time.sleep(sleep_sec + attempt * 0.8)
    raise RuntimeError(err or "view fetch failed")


def _within_window(pub_date: date, *, window_years: int, window_start: date | None = None) -> bool:
    if window_start is None:
        today = date.today()
        window_start = today - timedelta(days=365 * window_years)
    return pub_date >= window_start


def _compute_phase_columns(
    pub_date: date,
    *,
    cfg: BreakpointsConfig,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    out["phase_base"] = compute_phase(pub_date, t1=cfg.t1, t2=cfg.t2)
    out["in_buffer_base"] = in_buffer(
        pub_date,
        t1=cfg.t1,
        t2=cfg.t2,
        buffer_weeks=cfg.default_buffer_weeks,
    )
    for w in cfg.shift_weeks:
        if w == 0:
            continue
        t1s = shift_date(cfg.t1, w)
        t2s = shift_date(cfg.t2, w)
        key = f"phase_shift_{w:+d}"
        out[key] = compute_phase(pub_date, t1=t1s, t2=t2s)
    return out


def _sleep_jitter(sleep_range: list[float]) -> float:
    if not sleep_range:
        return 1.2
    lo = float(min(sleep_range))
    hi = float(max(sleep_range))
    if hi <= 0:
        return 0.5
    return random.uniform(lo, hi)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--creators", default="data/creators/creators.csv")
    parser.add_argument("--breakpoints", default="config/breakpoints.yaml")
    parser.add_argument("--output", default="data/index/videos_index.csv")
    parser.add_argument("--page-size", type=int, default=50)
    parser.add_argument("--sleep", type=float, default=0.0, help="override sleep min/max by fixed value (sec)")
    parser.add_argument("--retries", type=int, default=0, help="override max_retries")
    parser.add_argument("--use-cache", action="store_true", help="use cached API pages if available")
    parser.add_argument("--resume", action="store_true", help="skip creators already in output")
    parser.add_argument("--run-id", default="", help="run id for archiving under runs/<run_id>")
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--batch-cooldown-sec", type=int, default=0)
    parser.add_argument("--mid-cooldown-sec", type=int, default=0)
    parser.add_argument("--sleep-min", type=float, default=0.0)
    parser.add_argument("--sleep-max", type=float, default=0.0)
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--max-creators", type=int, default=0, help="limit creators processed this run (0=all)")
    parser.add_argument("--cookies", default="secrets/cookies.json")
    parser.add_argument("--backend", choices=["ytdlp", "api"], default="ytdlp", help="index backend (default ytdlp)")
    parser.add_argument("--ytdlp-timeout-sec", type=int, default=300)
    parser.add_argument("--ytdlp-socket-timeout", type=int, default=10)
    parser.add_argument("--ytdlp-retries", type=int, default=2)
    parser.add_argument("--ytdlp-fallback-api", action="store_true")
    args = parser.parse_args()

    paths = get_project_paths()
    ensure_dirs(paths)

    log_lines: list[str] = []
    run_id = make_run_id(args.run_id or None)
    run_dirs = init_run_dirs(paths.root, run_id)
    _log(log_lines, f"Start 01a_build_video_index run_id={run_id}")

    creators_path = (paths.root / args.creators).resolve()
    breakpoints_path = (paths.root / args.breakpoints).resolve()
    output_path = (paths.root / args.output).resolve()

    cfg = load_breakpoints_yaml(breakpoints_path)
    _log(log_lines, f"Loaded breakpoints: {breakpoints_path}")
    window_start = date.today() - timedelta(days=365 * cfg.window_years)

    creators = pd.read_csv(creators_path, dtype=str).fillna("")
    if "creator_id" not in creators.columns:
        raise ValueError("creators.csv 缺少 creator_id")

    cookies_path = (paths.root / args.cookies).resolve()
    cookies = load_cookies_json(cookies_path) if cookies_path.exists() else {}
    sess = requests_session(cookies)
    cookies_netscape = paths.secrets / "cookies.netscape.txt"
    try:
        nav = get_json(sess, "https://api.bilibili.com/x/web-interface/nav")
        data = nav.get("data") or {}
        is_login = bool(isinstance(data, dict) and data.get("isLogin") is True)
        if not is_login:
            _log(log_lines, "WARN: nav isLogin=false (cookies invalid or not logged in); expect higher risk-control")
    except Exception as e:  # noqa: BLE001
        _log(log_lines, f"WARN: nav check failed: {type(e).__name__}: {e}")

    existing = pd.DataFrame()
    processed_ids: set[str] = set()
    if output_path.exists() and args.resume:
        existing = pd.read_csv(output_path, dtype=str).fillna("")
        processed_ids = set(existing["creator_id"].astype(str).tolist()) if "creator_id" in existing.columns else set()
        _log(log_lines, f"Resume enabled: found {len(processed_ids)} creators in existing index")

    rows: list[dict[str, Any]] = []
    cache_dir = paths.data_index / "cache"

    # batching params (config first, then CLI override)
    batch_size = args.batch_size if args.batch_size > 0 else cfg.batch_size
    batch_cooldown_sec = args.batch_cooldown_sec if args.batch_cooldown_sec > 0 else cfg.batch_cooldown_sec
    mid_cooldown_sec = args.mid_cooldown_sec if args.mid_cooldown_sec > 0 else cfg.mid_cooldown_sec
    sleep_range = cfg.sleep_range_sec
    if args.sleep_min > 0 or args.sleep_max > 0:
        sleep_range = [max(args.sleep_min, 0.1), max(args.sleep_max, args.sleep_min, 0.1)]
    if args.sleep > 0:
        sleep_range = [args.sleep, args.sleep]
    max_retries = args.max_retries if args.max_retries > 0 else cfg.max_retries

    params_path = run_dirs.logs / "params.yaml"
    write_params_yaml(
        params_path,
        {
            "run_id": run_id,
            "backend": args.backend,
            "ytdlp_timeout_sec": args.ytdlp_timeout_sec,
            "ytdlp_socket_timeout": args.ytdlp_socket_timeout,
            "ytdlp_retries": args.ytdlp_retries,
            "ytdlp_fallback_api": args.ytdlp_fallback_api,
            "batch_size": batch_size,
            "batch_cooldown_sec": batch_cooldown_sec,
            "mid_cooldown_sec": mid_cooldown_sec,
            "sleep_range_sec": sleep_range,
            "max_retries": max_retries,
            "window_years": cfg.window_years,
            "shift_weeks": cfg.shift_weeks,
            "default_buffer_weeks": cfg.default_buffer_weeks,
        },
    )
    _log(log_lines, f"Params written: {params_path}")
    _log(
        log_lines,
        f"Params: backend={args.backend} batch_size={batch_size} cooldown={batch_cooldown_sec}s sleep_range={sleep_range} retries={max_retries}",
    )

    failed_rows: list[dict[str, Any]] = []
    failed_path = run_dirs.logs / "failed_mids.csv"
    blocked_counts: dict[str, int] = {}

    total = len(creators)
    if args.max_creators and args.max_creators > 0:
        creators = creators.head(args.max_creators)
        total = len(creators)
    batches = [creators.iloc[i : i + batch_size] for i in range(0, total, batch_size)] if batch_size > 0 else [creators]

    for b_idx, batch in enumerate(batches, start=1):
        _log(log_lines, f"Batch {b_idx}/{len(batches)} start (size={len(batch)})")
        for _, r in tqdm(batch.iterrows(), total=len(batch), desc="Build video index"):
            creator_id = str(r.get("creator_id", "")).strip()
            if not creator_id:
                continue
            if args.resume and creator_id in processed_ids:
                continue

            creator_name = str(r.get("creator_name", "")).strip()
            creator_group = str(r.get("creator_group", "")).strip()

            creator_rows: list[dict[str, Any]] = []
            pn = 1
            blocked = False

            attempts = 0
            if args.backend == "api":
                creator_rows, blocked = _collect_creator_rows_api(
                    sess,
                    creator_id=creator_id,
                    creator_name=creator_name,
                    creator_group=creator_group,
                    cfg=cfg,
                    window_start=window_start,
                    page_size=args.page_size,
                    cache_dir=cache_dir,
                    use_cache=args.use_cache,
                    sleep_range=sleep_range,
                    max_retries=max_retries,
                    mid_cooldown_sec=mid_cooldown_sec,
                    blocked_counts=blocked_counts,
                    failed_rows=failed_rows,
                    log_lines=log_lines,
                )
            else:
                # yt-dlp backend: list BV IDs from space page, then enrich via view API
                try:
                    bvids, ytdlp_err = _fetch_bvids_via_ytdlp(
                        mid=creator_id,
                        sleep_range=sleep_range,
                        cookies_netscape=cookies_netscape if cookies_netscape.exists() else None,
                        timeout_sec=args.ytdlp_timeout_sec,
                        socket_timeout_sec=args.ytdlp_socket_timeout,
                        retries=args.ytdlp_retries,
                    )
                except Exception as e:  # noqa: BLE001
                    bvids = []
                    ytdlp_err = f"{type(e).__name__}: {e}"
                if not bvids:
                    failed_rows.append(
                        {
                            "mid": creator_id,
                            "creator_id": creator_id,
                            "creator_name": creator_name,
                            "code": "YTDLP",
                            "status": "FAILED_YTDLP",
                            "cooldown_sec": "",
                            "page": "",
                            "message": ytdlp_err or "yt-dlp returned empty list",
                            "attempts": attempts,
                            "timestamp": datetime.now().isoformat(timespec="seconds"),
                        }
                    )
                    _log(log_lines, f"FAILED_YTDLP: creator_id={creator_id} msg={ytdlp_err or 'empty list'}")
                    if args.ytdlp_fallback_api:
                        _log(log_lines, f"Fallback to API: creator_id={creator_id}")
                        creator_rows, blocked = _collect_creator_rows_api(
                            sess,
                            creator_id=creator_id,
                            creator_name=creator_name,
                            creator_group=creator_group,
                            cfg=cfg,
                            window_start=window_start,
                            page_size=args.page_size,
                            cache_dir=cache_dir,
                            use_cache=args.use_cache,
                            sleep_range=sleep_range,
                            max_retries=max_retries,
                            mid_cooldown_sec=mid_cooldown_sec,
                            blocked_counts=blocked_counts,
                            failed_rows=failed_rows,
                            log_lines=log_lines,
                        )
                    else:
                        blocked = True
                        continue

                if bvids:
                    oldest_pub_date = None
                    for bvid in bvids:
                        attempts += 1
                        try:
                            payload = _fetch_view_meta(
                                sess,
                                bvid=bvid,
                                sleep_sec=_sleep_jitter(sleep_range),
                                retries=max_retries,
                            )
                        except Exception as e:  # noqa: BLE001
                            failed_rows.append(
                                {
                                    "mid": creator_id,
                                    "creator_id": creator_id,
                                    "creator_name": creator_name,
                                    "bvid": bvid,
                                    "code": "",
                                    "status": "FAILED_VIEW",
                                    "cooldown_sec": "",
                                    "page": "",
                                    "message": f"{type(e).__name__}: {e}",
                                    "attempts": attempts,
                                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                                }
                            )
                            _log(log_lines, f"FAILED_VIEW: creator_id={creator_id} bvid={bvid} err={type(e).__name__}: {e}")
                            continue

                        code = payload.get("code")
                        if code == -799:
                            blocked_counts[creator_id] = blocked_counts.get(creator_id, 0) + 1
                            is_deferred = blocked_counts[creator_id] > 1
                            cooldown = 600 if is_deferred else mid_cooldown_sec
                            failed_rows.append(
                                {
                                    "mid": creator_id,
                                    "creator_id": creator_id,
                                    "creator_name": creator_name,
                                    "bvid": bvid,
                                    "code": code,
                                    "status": "DEFERRED" if is_deferred else "TEMP_BLOCKED",
                                    "cooldown_sec": cooldown,
                                    "page": "",
                                    "attempts": attempts,
                                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                                }
                            )
                            _log(
                                log_lines,
                                f"{'DEFERRED' if is_deferred else 'TEMP_BLOCKED'}: creator_id={creator_id} bvid={bvid} code=-799",
                            )
                            blocked = True
                            time.sleep(cooldown)
                            break

                        if code != 0:
                            failed_rows.append(
                                {
                                    "mid": creator_id,
                                    "creator_id": creator_id,
                                    "creator_name": creator_name,
                                    "bvid": bvid,
                                    "code": code,
                                    "status": "ERROR_VIEW",
                                    "cooldown_sec": "",
                                    "page": "",
                                    "message": payload.get("message"),
                                    "attempts": attempts,
                                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                                }
                            )
                            _log(log_lines, f"ERROR_VIEW: creator_id={creator_id} bvid={bvid} code={code} msg={payload.get('message')}")
                            continue

                        data = payload.get("data") or {}
                        pub_ts = int(data.get("pubdate") or 0)
                        pub_date = datetime.fromtimestamp(pub_ts).date() if pub_ts else None
                        if not pub_date:
                            continue
                        if oldest_pub_date is None or pub_date < oldest_pub_date:
                            oldest_pub_date = pub_date
                        if not _within_window(pub_date, window_years=cfg.window_years, window_start=window_start):
                            if oldest_pub_date is not None and oldest_pub_date < window_start:
                                break
                            continue

                        row = {
                            "creator_id": creator_id,
                            "creator_name": creator_name,
                            "creator_group": creator_group,
                            "bvid": str(data.get("bvid") or bvid),
                            "aid": str(data.get("aid", "")),
                            "pubdate": pub_date.isoformat(),
                            "title": str(data.get("title", "")),
                            "desc": str(data.get("desc", "")),
                            "tname": str(data.get("tname", "")),
                            "duration": int(data.get("duration") or 0),
                            "part_count": len(data.get("pages") or []),
                        }
                        row.update(_compute_phase_columns(pub_date, cfg=cfg))
                        creator_rows.append(row)
                        time.sleep(_sleep_jitter(sleep_range))

            if creator_rows:
                out_df = pd.DataFrame(creator_rows)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                out_df.to_csv(output_path, mode="a", header=not output_path.exists(), index=False, encoding="utf-8-sig")
                processed_ids.add(creator_id)

            if blocked:
                continue

        if b_idx < len(batches):
            _log(log_lines, f"Batch {b_idx} cooldown {batch_cooldown_sec}s")
            time.sleep(batch_cooldown_sec)

    # Merge existing + new for reporting
    out_df = pd.read_csv(output_path, dtype=str).fillna("") if output_path.exists() else pd.DataFrame(rows)
    _log(log_lines, f"Wrote index: {output_path} (rows={len(out_df)})")

    if failed_rows:
        pd.DataFrame(failed_rows).to_csv(failed_path, index=False, encoding="utf-8-sig")
        _log(log_lines, f"Wrote failed mids: {failed_path}")
        try:
            failed_df = pd.DataFrame(failed_rows)
            summary = failed_df.groupby(["code", "status"]).size().reset_index(name="count")
            _log(log_lines, f"Failed summary: {summary.to_dict(orient='records')}")
        except Exception:
            pass

    # snapshot to run directory
    try:
        import shutil

        shutil.copy2(output_path, run_dirs.data_snapshots / "videos_index.csv")
    except Exception as e:  # noqa: BLE001
        _log(log_lines, f"WARN: snapshot copy failed: {type(e).__name__}: {e}")

    _append_run_log(paths.outputs / "run_log.txt", log_lines)
    _append_run_log(run_dirs.logs / "run_log_video_index.txt", log_lines)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
