from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from lib.bilibili import fetch_player, fetch_player_wbi, fetch_video_meta
from lib.http import get_json, load_cookies_json, requests_session
from lib.paths import ensure_dirs, get_project_paths
from lib.run_utils import init_run_dirs, make_run_id


STATUS_OK = "OK"
STATUS_NO_WEB_SUBTITLE = "NO_WEB_SUBTITLE"
STATUS_NOT_LOGGED_IN = "NOT_LOGGED_IN"
STATUS_BLOCKED = "BLOCKED_OR_EMPTYLIST"
STATUS_ERR = "ERR"
STATUS_SKIP_EXISTS = "SKIP_EXISTS"


class SuspectMismatchError(ValueError):
    def __init__(self, *, to_max: float, limit: float) -> None:
        super().__init__(f"SUSPECT_MISMATCH max_to={to_max:.3f} > limit={limit:.0f}")
        self.to_max = float(to_max)
        self.limit = float(limit)


def _run_cmd(cmd: list[str], *, cwd: Path, timeout_sec: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        check=False,
    )


@dataclass(frozen=True)
class SubtitleCandidate:
    source: str  # web_cc / ai_player
    api_code: int | None
    api_message: str | None
    subtitle_count: int
    lan: str
    lan_doc: str
    subtitle_url: str
    subtitle_type: str
    subtitle_id: str


def _read_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"未找到 manifest：{path}")
    df = pd.read_csv(path, dtype=str).fillna("")
    required = {"creator_id", "bvid"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"manifest 缺少字段: {sorted(missing)}")
    return df


def _to_int(value: object, default: int = 0) -> int:
    s = str(value).strip() if value is not None else ""
    try:
        return int(float(s))
    except Exception:
        return default


def _sha1_hex(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def _lang_rank(lan: str, lan_doc: str) -> int:
    lan = (lan or "").strip()
    lan_doc = (lan_doc or "").strip()
    preferred = [
        "zh-Hans",
        "zh-CN",
        "zh",
        "zh-Hant",
        "zh-TW",
        "zh-HK",
    ]
    if lan in preferred:
        return preferred.index(lan)
    if "中文" in lan_doc:
        return 50
    return 999


def _normalize_subtitle_source(value: str) -> str:
    v = (value or "").strip()
    if v in {"web_cc", "ai_player"}:
        return v
    if v in {"wbi_v2", "wbi", "web"}:
        return "web_cc"
    if v in {"player_v2", "player", "ai"}:
        return "ai_player"
    return v


def _canonical_subtitle_url(url: str) -> str:
    u = (url or "").strip()
    if u.startswith("//"):
        u = "https:" + u
    return u


def _choose_subtitle(subtitles: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not subtitles:
        return None
    return sorted(subtitles, key=lambda s: _lang_rank(str(s.get("lan", "")), str(s.get("lan_doc", ""))))[0]


def _parse_bcc_json(raw: bytes) -> dict[str, Any]:
    try:
        obj = json.loads(raw.decode("utf-8-sig"))
    except Exception:
        obj = json.loads(raw.decode("utf-8", errors="replace"))
    if not isinstance(obj, dict):
        raise ValueError("BCC 不是 JSON 对象")
    body = obj.get("body")
    if not isinstance(body, list):
        raise ValueError("BCC 缺少 body(list)")
    return obj


def _bcc_max_to_sec(bcc: dict[str, Any]) -> float:
    body = bcc.get("body") if isinstance(bcc, dict) else None
    if not isinstance(body, list):
        return 0.0
    to_max = 0.0
    for seg in body:
        if not isinstance(seg, dict):
            continue
        try:
            to_v = float(seg.get("to", 0) or 0)
        except Exception:
            to_v = 0.0
        if to_v > to_max:
            to_max = to_v
    return to_max


def _validate_bcc_bytes(raw: bytes, *, page_duration_sec: int) -> float:
    bcc = _parse_bcc_json(raw)
    to_max = _bcc_max_to_sec(bcc)
    if page_duration_sec > 0:
        limit = float(page_duration_sec + 5)
        if to_max > limit:
            raise SuspectMismatchError(to_max=to_max, limit=limit)
    return to_max


def _is_blocked_payload(code: int | None, message: str | None) -> bool:
    if code is None:
        return False
    if code in {-401, -403, -412, -429}:
        return True
    msg = (message or "").lower()
    keywords = [
        "风控",
        "risk",
        "请求过于频繁",
        "too many",
        "访问",
        "权限",
        "blocked",
        "wbi",
        "sign",
        "signature",
        "非法",
    ]
    return any(k in msg for k in keywords)


def _is_blocked_exception(exc: Exception) -> bool:
    resp = getattr(exc, "response", None)
    status_code = getattr(resp, "status_code", None)
    if status_code in {403, 412, 429}:
        return True
    msg = str(exc)
    return any(s in msg for s in ["412", "429", "403", "请求过于频繁", "风控"])


def _fetch_subtitle_candidates(
    sess,
    *,
    source: str,
    bvid: str,
    aid: int,
    cid: int,
    retry_empty_once: bool,
    sleep_sec: float,
) -> tuple[list[SubtitleCandidate], dict[str, object]]:
    """
    返回：候选列表（按语言优先级排序） + 诊断信息（用于写日志）
    """

    def fetch_once() -> dict[str, Any]:
        if source == "web_cc":
            return fetch_player_wbi(sess, aid, cid, bvid_for_referer=bvid)
        if source == "ai_player":
            return fetch_player(sess, aid, cid, bvid_for_referer=bvid)
        raise ValueError(f"unknown source: {source}")

    payload: dict[str, Any] | None = None
    try:
        payload = fetch_once()
    except Exception as e:  # noqa: BLE001
        diag = {
            "api_source": source,
            "api_code": "",
            "api_message": "",
            "subtitle_count": "",
            "error": f"{type(e).__name__}: {e}",
            "blocked_like": _is_blocked_exception(e),
        }
        return [], diag

    code = payload.get("code")
    message = payload.get("message")

    if code != 0:
        diag = {
            "api_source": source,
            "api_code": code,
            "api_message": message,
            "subtitle_count": "",
            "error": f"API code={code} message={message}",
            "blocked_like": _is_blocked_payload(code, str(message)),
        }
        return [], diag

    data = payload.get("data") or {}
    subtitles = ((data.get("subtitle") or {}).get("subtitles") or []) if isinstance(data, dict) else []

    if (not subtitles) and retry_empty_once:
        time.sleep(sleep_sec)
        try:
            payload2 = fetch_once()
        except Exception:
            payload2 = None
        if isinstance(payload2, dict) and payload2.get("code") == 0:
            data2 = payload2.get("data") or {}
            subs2 = ((data2.get("subtitle") or {}).get("subtitles") or []) if isinstance(data2, dict) else []
            subtitles = subs2
            payload = payload2
            code = payload2.get("code")
            message = payload2.get("message")

    subtitles = subtitles if isinstance(subtitles, list) else []
    subtitles_sorted = sorted(subtitles, key=lambda s: _lang_rank(str(s.get("lan", "")), str(s.get("lan_doc", ""))))

    cands: list[SubtitleCandidate] = []
    for s in subtitles_sorted:
        if not isinstance(s, dict):
            continue
        url = _canonical_subtitle_url(str(s.get("subtitle_url") or ""))
        if not url:
            continue
        cands.append(
            SubtitleCandidate(
                source=source,
                api_code=int(code) if isinstance(code, int) else None,
                api_message=str(message) if message is not None else None,
                subtitle_count=len(subtitles_sorted),
                lan=str(s.get("lan") or ""),
                lan_doc=str(s.get("lan_doc") or ""),
                subtitle_url=url,
                subtitle_type=str(s.get("type") or ""),
                subtitle_id=str(s.get("id") or ""),
            )
        )

    diag = {
        "api_source": source,
        "api_code": code,
        "api_message": message,
        "subtitle_count": len(subtitles_sorted),
        "error": "",
        "blocked_like": False,
    }
    return cands, diag


def _download_bcc_bytes(
    sess,
    *,
    bvid: str,
    subtitle_url: str,
) -> bytes:
    resp = sess.get(subtitle_url, headers={"Referer": f"https://www.bilibili.com/video/{bvid}"}, timeout=30)
    resp.raise_for_status()
    return resp.content


def _download_audio_via_ytdlp(*, bvid: str, workdir: Path, no_proxy: bool) -> Path:
    video_url = f"https://www.bilibili.com/video/{bvid}"
    out_tpl = str(workdir / f"{bvid}.%(ext)s")
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--no-part",
        "--retries",
        "3",
        "--fragment-retries",
        "3",
        "-f",
        "ba/best",
        "-o",
        out_tpl,
        video_url,
    ]
    if no_proxy:
        cmd.extend(["--proxy", ""])
    proc = _run_cmd(cmd, cwd=workdir, timeout_sec=600)
    if proc.returncode != 0:
        raise RuntimeError(f"yt-dlp failed rc={proc.returncode} stderr={proc.stderr.strip()[:400]}")

    candidates = [p for p in workdir.glob(f"{bvid}.*") if p.is_file() and (not p.name.endswith(".part"))]
    if not candidates:
        raise FileNotFoundError("yt-dlp 音频下载未产出文件")
    candidates.sort(key=lambda p: p.stat().st_size, reverse=True)
    return candidates[0]


def _asr_bcc_via_faster_whisper(
    *,
    audio_path: Path,
    model_name: str,
    compute_type: str,
    language: str,
) -> dict[str, Any]:
    try:
        from faster_whisper import WhisperModel  # type: ignore[import-not-found]
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("缺少依赖 faster-whisper；请先 pip install -r requirements.txt") from e

    global _ASR_MODEL_CACHE
    key = (model_name, compute_type)
    model = _ASR_MODEL_CACHE.get(key)
    if model is None:
        model = WhisperModel(model_name, device="cpu", compute_type=compute_type)
        _ASR_MODEL_CACHE[key] = model
    segments, _info = model.transcribe(
        str(audio_path),
        language=language,
        vad_filter=True,
    )

    body: list[dict[str, object]] = []
    for seg in segments:
        text = str(getattr(seg, "text", "") or "").strip()
        if not text:
            continue
        start = float(getattr(seg, "start", 0.0) or 0.0)
        end = float(getattr(seg, "end", 0.0) or 0.0)
        if end <= start:
            continue
        body.append(
            {
                "from": round(start, 3),
                "to": round(end, 3),
                "content": text,
            }
        )

    if not body:
        raise RuntimeError("ASR 输出为空（无有效分段）")

    return {
        "type": "ASR",
        "lang": language,
        "version": f"faster-whisper:{model_name}:{compute_type}",
        "body": body,
    }


_ASR_MODEL_CACHE: dict[tuple[str, str], Any] = {}


def _validate_existing_bcc_file(
    *,
    path: Path,
    page_duration_sec: int,
) -> tuple[str, float]:
    raw = path.read_bytes()
    to_max = _validate_bcc_bytes(raw, page_duration_sec=page_duration_sec)
    return _sha1_hex(raw), to_max


def _check_login(sess) -> tuple[bool, dict[str, Any]]:
    payload = get_json(sess, "https://api.bilibili.com/x/web-interface/nav")
    data = payload.get("data") or {}
    is_login = bool(isinstance(data, dict) and data.get("isLogin") is True)
    return is_login, payload


def _ensure_manifest_cols(df: pd.DataFrame) -> None:
    cols = [
        "subtitle_status",
        "subtitle_source",
        "subtitle_url",
        "subtitle_lang",
        "subtitle_lang_doc",
        "subtitle_type",
        "subtitle_id",
        "subtitle_sha1",
        "subtitle_to_max_sec",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="", help="指定 manifest 路径（如 outputs/final_manifest.csv）")
    parser.add_argument("--run-id", default="", help="run id for archiving under runs/<run_id>")
    parser.add_argument("--force", action="store_true", help="忽略缓存，强制重新下载并覆盖")
    parser.add_argument(
        "--allow-player-v2",
        action="store_true",
        help="允许回退到 legacy 的 x/player/v2（AI/自动字幕）；默认只取网页 CC（x/player/wbi/v2）",
    )
    parser.add_argument(
        "--trust-ai-player",
        action="store_true",
        help="信任并使用 AI 字幕（高串台/错配风险）；未开启时即使 --allow-player-v2 也不会使用 AI 字幕",
    )
    parser.add_argument(
        "--asr-mode",
        choices=["off", "fallback", "force"],
        default="fallback",
        help="本地 ASR 生成字幕：off=禁用，fallback=无平台字幕则补全，force=强制对全部 BV 生成（确保 12/12 对应）",
    )
    parser.add_argument(
        "--force-asr",
        action="store_true",
        help="别名：等同于 --asr-mode force",
    )
    parser.add_argument("--asr-model", default="tiny", help="faster-whisper 模型：tiny/base/small/medium/...")
    parser.add_argument("--asr-compute-type", default="int8", help="faster-whisper compute_type：int8/int8_float16/float16/...")
    parser.add_argument("--asr-language", default="zh", help="ASR 语言（默认 zh）")
    parser.add_argument("--only-bvid", default="", help="仅处理指定 BV（逗号分隔），用于分批/调试")
    parser.add_argument("--only-pending", action="store_true", help="仅处理 subtitle_status != OK 的样本")
    parser.add_argument("--offset", type=int, default=0, help="处理起始 offset（筛选后）")
    parser.add_argument("--limit", type=int, default=0, help="最多处理条数（筛选后，0=不限）")
    parser.add_argument("--shard-index", type=int, default=0, help="分片索引（1-based，与 --shard-total 配合）")
    parser.add_argument("--shard-total", type=int, default=0, help="分片总数（>0 启用分片）")
    parser.add_argument("--sleep", type=float, default=1.2, help="节流 sleep 秒数（>=1.2）")
    parser.add_argument("--mismatch-retries", type=int, default=3, help="遇到 SUSPECT_MISMATCH 时的重试次数（仅重试下载，不会放行错配）")
    parser.add_argument("--flush-every", type=int, default=20, help="每处理 N 条写回 manifest，避免中断丢进度")
    parser.add_argument("--no-proxy", action="store_true", help="禁用系统代理（requests trust_env=False；yt-dlp --proxy ''）")
    args = parser.parse_args()

    if args.force_asr:
        args.asr_mode = "force"

    sleep_sec = max(float(args.sleep), 1.2)
    mismatch_retries = max(int(args.mismatch_retries), 1)

    paths = get_project_paths()
    ensure_dirs(paths)

    run_id = make_run_id(args.run_id or None)
    run_dirs = init_run_dirs(paths.root, run_id)

    manifest_path = paths.data_raw_meta / "videos_manifest.csv"
    if args.manifest:
        manifest_path = Path(args.manifest)
    df = _read_manifest(manifest_path)
    _ensure_manifest_cols(df)
    only_bvid = {s.strip() for s in str(args.only_bvid).split(",") if s.strip()}

    cookies = load_cookies_json(paths.secrets / "cookies.json")
    sess = requests_session(cookies, trust_env=not args.no_proxy)

    log_path = run_dirs.logs / "download.jsonl"
    log_path.write_text("", encoding="utf-8")
    # run log text
    run_log_path = run_dirs.logs / "run_log_download.txt"
    run_log_path.write_text(f"[{datetime.now().isoformat(timespec='seconds')}] Start download manifest={manifest_path}\n", encoding="utf-8")
    processed = 0
    flush_every = max(int(args.flush_every), 1)
    asr_failed_rows: list[dict[str, object]] = []
    with log_path.open("a", encoding="utf-8") as log_f:
        # 登录态检查：若启用 ASR（fallback/force），即使未登录也允许继续（用 ASR 兜底保证 12/12）。
        is_login = False
        nav: dict[str, Any] = {}
        try:
            is_login, nav = _check_login(sess)
        except Exception as e:  # noqa: BLE001
            obj = {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "status": STATUS_NOT_LOGGED_IN,
                "error": f"{type(e).__name__}: {e}",
            }
            log_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            log_f.flush()
            if args.asr_mode == "off":
                print(f"{STATUS_NOT_LOGGED_IN}: nav 检查失败：{type(e).__name__}: {e}")
                return 2

        if not is_login:
            obj = {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "status": STATUS_NOT_LOGGED_IN,
                "api_code": nav.get("code"),
                "api_message": nav.get("message"),
            }
            log_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            log_f.flush()
            if args.asr_mode == "off":
                print(f"{STATUS_NOT_LOGGED_IN}: cookies 无效或未登录（x/web-interface/nav isLogin=false）")
                return 2

        sha1_to_bvid: dict[str, str] = {}

        indices = list(df.index)
        if only_bvid:
            indices = [idx for idx in indices if str(df.at[idx, "bvid"]).strip() in only_bvid]
        if args.only_pending:
            indices = [
                idx
                for idx in indices
                if str(df.at[idx, "subtitle_status"]).strip() != STATUS_OK
            ]
        if args.shard_total and args.shard_total > 0:
            if args.shard_index <= 0 or args.shard_index > args.shard_total:
                raise ValueError("--shard-index 必须在 [1, shard_total] 之间")
            indices = [
                idx
                for pos, idx in enumerate(indices)
                if pos % int(args.shard_total) == int(args.shard_index) - 1
            ]
        if args.offset and args.offset > 0:
            indices = indices[int(args.offset) :]
        if args.limit and args.limit > 0:
            indices = indices[: int(args.limit)]

        try:
            with run_log_path.open("a", encoding="utf-8") as rf:
                rf.write(
                    f"[{datetime.now().isoformat(timespec='seconds')}] Filtered rows={len(indices)} "
                    f"(only_pending={args.only_pending}, shard={args.shard_index}/{args.shard_total}, "
                    f"offset={args.offset}, limit={args.limit}, no_proxy={args.no_proxy})\n"
                )
        except Exception:
            pass

        for i in tqdm(indices, total=len(indices), desc="Download subtitles"):
            r = df.loc[i]
            bvid = str(r.get("bvid", "")).strip()
            creator_id = str(r.get("creator_id", "")).strip()
            series = str(r.get("series", "")).strip()
            unique_key = str(r.get("unique_key", "")).strip() or bvid

            aid = _to_int(r.get("aid", ""), 0)
            cid = _to_int(r.get("cid", ""), 0)
            duration_sec = _to_int(r.get("duration_sec", ""), 0)
            page_duration = _to_int(r.get("page_duration", ""), 0)
            page_duration_sec = page_duration if page_duration > 0 else duration_sec

            out_dir = paths.data_raw_subtitles / creator_id
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{bvid}.bcc.json"
            clean_path = paths.data_processed_text / creator_id / f"clean_{bvid}.txt"

            log_obj: dict[str, object] = {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "series": series,
                "creator_id": creator_id,
                "bvid": bvid,
                "unique_key": unique_key,
                "aid": aid,
                "cid": cid,
                "status": "",
                "api_source": "",
                "subtitle_count": "",
                "chosen_lan": "",
                "chosen_url": "",
                "code": "",
                "message": "",
                "error": "",
                "sha1": "",
                "warning_same_subtitle_as": "",
            }

            def clear_manifest_fields() -> None:
                df.at[i, "subtitle_status"] = ""
                for c in [
                    "subtitle_source",
                    "subtitle_url",
                    "subtitle_lang",
                    "subtitle_lang_doc",
                    "subtitle_type",
                    "subtitle_id",
                    "subtitle_sha1",
                    "subtitle_to_max_sec",
                ]:
                    if c in df.columns:
                        df.at[i, c] = ""

            try:
                if not bvid or not creator_id:
                    clear_manifest_fields()
                    if out_path.exists():
                        out_path.unlink()
                    log_obj.update(
                        {
                            "status": STATUS_ERR,
                            "error": "MISSING_REQUIRED_FIELDS (bvid/creator_id)",
                        }
                    )
                    log_f.write(json.dumps(log_obj, ensure_ascii=False) + "\n")
                    log_f.flush()
                    time.sleep(sleep_sec)
                    continue

                # clean.txt 已存在 -> 跳过（除非 --force）
                if clean_path.exists() and (not args.force):
                    if str(df.at[i, "subtitle_status"]).strip() != STATUS_OK:
                        df.at[i, "subtitle_status"] = STATUS_OK
                    log_obj.update(
                        {
                            "status": STATUS_SKIP_EXISTS,
                            "api_source": "clean_exists",
                            "error": "CLEAN_EXISTS",
                        }
                    )
                    log_f.write(json.dumps(log_obj, ensure_ascii=False) + "\n")
                    log_f.flush()
                    time.sleep(sleep_sec)
                    continue

                # 若 manifest 未提供 aid/cid，则回填（仅一次）
                if aid <= 0 or cid <= 0:
                    try:
                        meta = fetch_video_meta(sess, bvid)
                        aid = meta.aid
                        cid = meta.cid
                        duration_sec = meta.duration_sec
                        page_duration_sec = meta.page_duration
                        df.at[i, "aid"] = aid
                        df.at[i, "cid"] = cid
                        df.at[i, "duration_sec"] = duration_sec
                        df.at[i, "page_duration"] = meta.page_duration
                        df.at[i, "pages_count"] = meta.pages_count
                        df.at[i, "page_index"] = meta.page_index
                        df.at[i, "part_name"] = meta.part_name
                        df.at[i, "title"] = meta.title
                        df.at[i, "up_mid"] = meta.up_mid if meta.up_mid is not None else ""
                        df.at[i, "up_name"] = meta.up_name if meta.up_name is not None else ""
                        df.at[i, "pub_date"] = meta.pub_datetime.date().isoformat()
                        df.at[i, "pub_datetime"] = meta.pub_datetime.isoformat(timespec="seconds")
                    except Exception as e:  # noqa: BLE001
                        clear_manifest_fields()
                        if out_path.exists():
                            out_path.unlink()
                        log_obj.update(
                            {
                                "status": STATUS_ERR,
                                "error": f"FETCH_META_FAILED: {type(e).__name__}: {e}",
                            }
                        )
                        log_f.write(json.dumps(log_obj, ensure_ascii=False) + "\n")
                        log_f.flush()
                        time.sleep(sleep_sec)
                        continue

                def write_ok_from_asr() -> None:
                    nonlocal log_obj
                    try:
                        with tempfile.TemporaryDirectory(prefix=f"asr_{bvid}_") as tmpdir:
                            tmp = Path(tmpdir)
                            audio_path = _download_audio_via_ytdlp(bvid=bvid, workdir=tmp, no_proxy=args.no_proxy)
                            bcc = _asr_bcc_via_faster_whisper(
                                audio_path=audio_path,
                                model_name=str(args.asr_model),
                                compute_type=str(args.asr_compute_type),
                                language=str(args.asr_language),
                            )

                            # 轻度约束：避免末尾超出 page_duration 触发校验
                            if page_duration_sec > 0:
                                limit = float(page_duration_sec + 5)
                                for seg in bcc.get("body") or []:
                                    if not isinstance(seg, dict):
                                        continue
                                    try:
                                        seg_to = float(seg.get("to", 0) or 0)
                                        seg_from = float(seg.get("from", 0) or 0)
                                    except Exception:
                                        continue
                                    if seg_to > limit:
                                        seg["to"] = round(limit, 3)
                                    if seg_from > float(seg.get("to", seg_to) or seg_to):
                                        seg["from"] = round(
                                            max(0.0, float(seg.get("to", seg_to) or seg_to) - 0.2),
                                            3,
                                        )

                            raw_text = json.dumps(bcc, ensure_ascii=False, indent=2)
                            raw_bytes = raw_text.encode("utf-8")
                            to_max = _bcc_max_to_sec(bcc)
                            sha1 = _sha1_hex(raw_bytes)
                            out_path.write_bytes(raw_bytes)
                    except Exception as e:  # noqa: BLE001
                        asr_failed_rows.append(
                            {
                                "ts": datetime.now().isoformat(timespec="seconds"),
                                "creator_id": creator_id,
                                "bvid": bvid,
                                "unique_key": unique_key,
                                "error": f"{type(e).__name__}: {e}",
                            }
                        )
                        raise

                    same_as = sha1_to_bvid.get(sha1, "")
                    if same_as and same_as != bvid:
                        log_obj["warning_same_subtitle_as"] = same_as
                    else:
                        sha1_to_bvid[sha1] = bvid

                    df.at[i, "subtitle_status"] = STATUS_OK
                    df.at[i, "subtitle_source"] = "asr_faster_whisper"
                    df.at[i, "subtitle_url"] = ""
                    df.at[i, "subtitle_lang"] = str(args.asr_language)
                    df.at[i, "subtitle_lang_doc"] = "ASR"
                    df.at[i, "subtitle_type"] = "asr"
                    df.at[i, "subtitle_id"] = ""
                    df.at[i, "subtitle_sha1"] = sha1
                    df.at[i, "subtitle_to_max_sec"] = round(to_max, 3)

                    log_obj.update(
                        {
                            "status": STATUS_OK,
                            "api_source": "asr",
                            "subtitle_count": len(bcc.get("body") or []),
                            "chosen_lan": str(args.asr_language),
                            "chosen_url": "",
                            "code": "",
                            "message": "",
                            "sha1": sha1,
                            "asr_model": str(args.asr_model),
                            "asr_compute_type": str(args.asr_compute_type),
                        }
                    )
                    log_f.write(json.dumps(log_obj, ensure_ascii=False) + "\n")
                    log_f.flush()

                existing_source_raw = str(r.get("subtitle_source") or "").strip()
                existing_source_norm = _normalize_subtitle_source(existing_source_raw)
                existing_is_asr = existing_source_raw.startswith("asr_") or existing_source_raw.startswith("asr")
                cache_allowed = False
                if args.asr_mode == "off":
                    cache_allowed = True
                elif args.asr_mode == "force":
                    cache_allowed = existing_is_asr
                else:  # fallback
                    if existing_is_asr or (existing_source_norm == "web_cc"):
                        cache_allowed = True
                    elif existing_source_norm == "ai_player":
                        cache_allowed = bool(args.allow_player_v2 and args.trust_ai_player)

                # 缓存：也要做结构/时长一致性校验；若不允许缓存则先清掉旧文件避免污染
                if out_path.exists() and (not args.force) and cache_allowed:
                    sha1, to_max = _validate_existing_bcc_file(path=out_path, page_duration_sec=page_duration_sec)
                    df.at[i, "subtitle_status"] = STATUS_OK
                    df.at[i, "subtitle_source"] = _normalize_subtitle_source(str(r.get("subtitle_source") or ""))
                    df.at[i, "subtitle_sha1"] = sha1
                    df.at[i, "subtitle_to_max_sec"] = round(to_max, 3)

                    same_as = sha1_to_bvid.get(sha1, "")
                    if same_as and same_as != bvid:
                        log_obj["warning_same_subtitle_as"] = same_as
                    else:
                        sha1_to_bvid[sha1] = bvid

                    log_obj.update(
                        {
                            "status": STATUS_SKIP_EXISTS,
                            "api_source": _normalize_subtitle_source(str(r.get("subtitle_source") or "")),
                            "sha1": sha1,
                        }
                    )
                    log_f.write(json.dumps(log_obj, ensure_ascii=False) + "\n")
                    log_f.flush()
                    time.sleep(sleep_sec)
                    continue

                if out_path.exists() and (args.force or (not cache_allowed)):
                    out_path.unlink()

                # 强制 ASR：忽略平台字幕，保证每个 BV 都生成自己的字幕
                if args.asr_mode == "force":
                    write_ok_from_asr()
                    time.sleep(sleep_sec)
                    continue

                # 未登录：禁用平台字幕时，用 ASR 兜底
                if (not is_login) and args.asr_mode != "off":
                    write_ok_from_asr()
                    time.sleep(sleep_sec)
                    continue

                # 1) 默认：web_cc（x/player/wbi/v2）
                web_cands, web_diag = _fetch_subtitle_candidates(
                    sess,
                    source="web_cc",
                    bvid=bvid,
                    aid=aid,
                    cid=cid,
                    retry_empty_once=True,
                    sleep_sec=sleep_sec,
                )
                time.sleep(sleep_sec)

                if web_cands:
                    cand = web_cands[0]
                    sha1 = ""
                    try:
                        raw = _download_bcc_bytes(sess, bvid=bvid, subtitle_url=cand.subtitle_url)
                        sha1 = _sha1_hex(raw)
                        to_max = _validate_bcc_bytes(raw, page_duration_sec=page_duration_sec)
                        out_path.write_bytes(raw)
                    except Exception as e:  # noqa: BLE001
                        clear_manifest_fields()
                        if out_path.exists():
                            out_path.unlink()
                        status = STATUS_BLOCKED if _is_blocked_exception(e) else STATUS_ERR
                        df.at[i, "subtitle_status"] = status
                        log_obj.update(
                            {
                                "status": status,
                                "api_source": "web_cc",
                                "subtitle_count": cand.subtitle_count,
                                "chosen_lan": cand.lan,
                                "chosen_url": cand.subtitle_url,
                                "code": cand.api_code if cand.api_code is not None else "",
                                "message": cand.api_message if cand.api_message is not None else "",
                                "error": f"{type(e).__name__}: {e}",
                                "sha1": sha1,
                            }
                        )
                        log_f.write(json.dumps(log_obj, ensure_ascii=False) + "\n")
                        log_f.flush()
                        time.sleep(sleep_sec)
                        continue

                    same_as = sha1_to_bvid.get(sha1, "")
                    if same_as and same_as != bvid:
                        log_obj["warning_same_subtitle_as"] = same_as
                    else:
                        sha1_to_bvid[sha1] = bvid

                    df.at[i, "subtitle_status"] = STATUS_OK
                    df.at[i, "subtitle_source"] = "web_cc"
                    df.at[i, "subtitle_url"] = cand.subtitle_url
                    df.at[i, "subtitle_lang"] = cand.lan
                    df.at[i, "subtitle_lang_doc"] = cand.lan_doc
                    df.at[i, "subtitle_type"] = cand.subtitle_type
                    df.at[i, "subtitle_id"] = cand.subtitle_id
                    df.at[i, "subtitle_sha1"] = sha1
                    df.at[i, "subtitle_to_max_sec"] = round(to_max, 3)

                    log_obj.update(
                        {
                            "status": STATUS_OK,
                            "api_source": "web_cc",
                            "subtitle_count": cand.subtitle_count,
                            "chosen_lan": cand.lan,
                            "chosen_url": cand.subtitle_url,
                            "code": cand.api_code if cand.api_code is not None else "",
                            "message": cand.api_message if cand.api_message is not None else "",
                            "sha1": sha1,
                        }
                    )
                    log_f.write(json.dumps(log_obj, ensure_ascii=False) + "\n")
                    log_f.flush()
                    time.sleep(sleep_sec)
                    continue

                # web_cc 无候选：区分“确实无网页字幕” vs “请求异常/风控”
                web_blocked_like = bool(web_diag.get("blocked_like"))
                web_error = str(web_diag.get("error") or "").strip()
                web_sub_count = web_diag.get("subtitle_count", "")

                # 2) 可控回退：ai_player（x/player/v2）
                if args.allow_player_v2 and args.trust_ai_player:
                    last_ai_diag: dict[str, object] = {}
                    last_ai_error: str = ""
                    last_ai_candidate: SubtitleCandidate | None = None
                    last_ai_sha1: str = ""
                    last_ai_to_max: float | None = None

                    ai_ok = False
                    for attempt in range(mismatch_retries):
                        ai_cands, ai_diag = _fetch_subtitle_candidates(
                            sess,
                            source="ai_player",
                            bvid=bvid,
                            aid=aid,
                            cid=cid,
                            retry_empty_once=False,
                            sleep_sec=sleep_sec,
                        )
                        last_ai_diag = ai_diag
                        time.sleep(sleep_sec)

                        if not ai_cands:
                            break

                        cand = ai_cands[0]
                        last_ai_candidate = cand
                        sha1 = ""
                        try:
                            raw = _download_bcc_bytes(sess, bvid=bvid, subtitle_url=cand.subtitle_url)
                            sha1 = _sha1_hex(raw)
                            to_max = _validate_bcc_bytes(raw, page_duration_sec=page_duration_sec)
                            out_path.write_bytes(raw)
                            last_ai_sha1 = sha1
                            last_ai_to_max = to_max
                            ai_ok = True
                            break
                        except SuspectMismatchError as e:
                            last_ai_error = f"{type(e).__name__}: {e}"
                            last_ai_sha1 = sha1
                            last_ai_to_max = e.to_max
                            if out_path.exists():
                                out_path.unlink()
                            if attempt < mismatch_retries - 1:
                                time.sleep(sleep_sec)
                                continue
                            break
                        except Exception as e:  # noqa: BLE001
                            last_ai_error = f"{type(e).__name__}: {e}"
                            last_ai_sha1 = sha1
                            last_ai_to_max = None
                            if out_path.exists():
                                out_path.unlink()
                            break

                    if ai_ok and last_ai_candidate is not None:
                        cand = last_ai_candidate
                        sha1 = last_ai_sha1
                        to_max = float(last_ai_to_max or 0.0)

                        same_as = sha1_to_bvid.get(sha1, "")
                        if same_as and same_as != bvid:
                            log_obj["warning_same_subtitle_as"] = same_as
                        else:
                            sha1_to_bvid[sha1] = bvid

                        df.at[i, "subtitle_status"] = STATUS_OK
                        df.at[i, "subtitle_source"] = "ai_player"
                        df.at[i, "subtitle_url"] = cand.subtitle_url
                        df.at[i, "subtitle_lang"] = cand.lan
                        df.at[i, "subtitle_lang_doc"] = cand.lan_doc
                        df.at[i, "subtitle_type"] = cand.subtitle_type
                        df.at[i, "subtitle_id"] = cand.subtitle_id
                        df.at[i, "subtitle_sha1"] = sha1
                        df.at[i, "subtitle_to_max_sec"] = round(to_max, 3)

                        log_obj.update(
                            {
                                "status": STATUS_OK,
                                "api_source": "ai_player",
                                "subtitle_count": cand.subtitle_count,
                                "chosen_lan": cand.lan,
                                "chosen_url": cand.subtitle_url,
                                "code": cand.api_code if cand.api_code is not None else "",
                                "message": cand.api_message if cand.api_message is not None else "",
                                "sha1": sha1,
                                "mismatch_retries": mismatch_retries,
                                "web_cc_code": web_diag.get("api_code", ""),
                                "web_cc_message": web_diag.get("api_message", ""),
                                "web_cc_subtitle_count": web_sub_count,
                                "web_cc_error": web_error,
                            }
                        )
                        log_f.write(json.dumps(log_obj, ensure_ascii=False) + "\n")
                        log_f.flush()
                        time.sleep(sleep_sec)
                        continue

                    if last_ai_candidate is not None:
                        # 有候选但下载/校验失败：记录到日志（不放行错配）
                        status = STATUS_ERR
                        clear_manifest_fields()
                        if out_path.exists():
                            out_path.unlink()
                        df.at[i, "subtitle_status"] = status
                        log_obj.update(
                            {
                                "ai_player_error": last_ai_error,
                                "ai_player_sha1": last_ai_sha1,
                                "ai_player_subtitle_count": last_ai_candidate.subtitle_count,
                                "ai_player_lan": last_ai_candidate.lan,
                                "ai_player_url": last_ai_candidate.subtitle_url,
                                "ai_player_code": last_ai_candidate.api_code if last_ai_candidate.api_code is not None else "",
                                "ai_player_message": last_ai_candidate.api_message if last_ai_candidate.api_message is not None else "",
                                "mismatch_retries": mismatch_retries,
                                "web_cc_code": web_diag.get("api_code", ""),
                                "web_cc_message": web_diag.get("api_message", ""),
                                "web_cc_subtitle_count": web_sub_count,
                                "web_cc_error": web_error,
                            }
                        )
                        if args.asr_mode == "fallback":
                            write_ok_from_asr()
                            time.sleep(sleep_sec)
                            continue

                        log_obj.update(
                            {
                                "status": status,
                                "api_source": "ai_player",
                                "subtitle_count": last_ai_candidate.subtitle_count,
                                "chosen_lan": last_ai_candidate.lan,
                                "chosen_url": last_ai_candidate.subtitle_url,
                                "code": last_ai_candidate.api_code if last_ai_candidate.api_code is not None else "",
                                "message": last_ai_candidate.api_message if last_ai_candidate.api_message is not None else "",
                                "error": last_ai_error,
                                "sha1": last_ai_sha1,
                            }
                        )
                        log_f.write(json.dumps(log_obj, ensure_ascii=False) + "\n")
                        log_f.flush()
                        time.sleep(sleep_sec)
                        continue

                    # ai_player 也无候选 -> 仍然用 web_cc 的“无字幕/异常”作为主因
                    clear_manifest_fields()
                    if out_path.exists():
                        out_path.unlink()

                    log_obj.update(
                        {
                            "web_cc_code": web_diag.get("api_code", ""),
                            "web_cc_message": web_diag.get("api_message", ""),
                            "web_cc_subtitle_count": web_sub_count,
                            "web_cc_error": web_error,
                            "ai_player_code": last_ai_diag.get("api_code", ""),
                            "ai_player_message": last_ai_diag.get("api_message", ""),
                            "ai_player_subtitle_count": last_ai_diag.get("subtitle_count", ""),
                            "ai_player_error": last_ai_diag.get("error", ""),
                        }
                    )

                    if args.asr_mode == "fallback":
                        write_ok_from_asr()
                        time.sleep(sleep_sec)
                        continue

                    df.at[i, "subtitle_status"] = STATUS_NO_WEB_SUBTITLE if not web_blocked_like else STATUS_BLOCKED
                    log_obj.update(
                        {
                            "status": STATUS_NO_WEB_SUBTITLE if not web_blocked_like else STATUS_BLOCKED,
                            "api_source": "web_cc",
                            "subtitle_count": web_sub_count,
                            "code": web_diag.get("api_code", ""),
                            "message": web_diag.get("api_message", ""),
                            "error": web_error,
                        }
                    )
                    log_f.write(json.dumps(log_obj, ensure_ascii=False) + "\n")
                    log_f.flush()
                    time.sleep(sleep_sec)
                    continue

                # 未拿到可信平台字幕：ASR fallback 兜底（确保 12/12 对应）
                if args.asr_mode == "fallback":
                    write_ok_from_asr()
                    time.sleep(sleep_sec)
                    continue

                # 不允许 ASR：直接按 web_cc 结果写状态
                clear_manifest_fields()
                if out_path.exists():
                    out_path.unlink()
                df.at[i, "subtitle_status"] = STATUS_NO_WEB_SUBTITLE if not web_blocked_like else STATUS_BLOCKED
                log_obj.update(
                    {
                        "status": STATUS_NO_WEB_SUBTITLE if not web_blocked_like else STATUS_BLOCKED,
                        "api_source": "web_cc",
                        "subtitle_count": web_sub_count,
                        "code": web_diag.get("api_code", ""),
                        "message": web_diag.get("api_message", ""),
                        "error": web_error,
                    }
                )
                log_f.write(json.dumps(log_obj, ensure_ascii=False) + "\n")
                log_f.flush()
                time.sleep(sleep_sec)
            except Exception as e:  # noqa: BLE001
                clear_manifest_fields()
                if out_path.exists():
                    out_path.unlink()
                blocked_like = _is_blocked_exception(e)
                status = STATUS_BLOCKED if blocked_like else STATUS_ERR
                df.at[i, "subtitle_status"] = status
                log_obj.update({"status": status, "error": f"{type(e).__name__}: {e}"})
                log_f.write(json.dumps(log_obj, ensure_ascii=False) + "\n")
                log_f.flush()
                time.sleep(sleep_sec)
            finally:
                processed += 1
                if processed % flush_every == 0:
                    df.to_csv(manifest_path, index=False, encoding="utf-8-sig")
                    try:
                        with run_log_path.open("a", encoding="utf-8") as rf:
                            rf.write(
                                f"[{datetime.now().isoformat(timespec='seconds')}] Flush manifest after {processed} items\n"
                            )
                    except Exception:
                        pass

    df.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    # mirror to standard logs path (no append history)
    try:
        import shutil

        shutil.copy2(log_path, paths.logs / "download.jsonl")
    except Exception:
        pass
    if asr_failed_rows:
        pd.DataFrame(asr_failed_rows).to_csv(run_dirs.logs / "asr_failed.csv", index=False, encoding="utf-8-sig")
    print(f"Wrote: {manifest_path}")
    print(f"Wrote: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
