from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests


DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)


def load_cookies_json(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("cookies.json 必须是 JSON 对象（cookie_name -> cookie_value）")
    cookies: dict[str, str] = {}
    for k, v in raw.items():
        if v is None:
            continue
        name = str(k)
        value = str(v)
        try:
            # HTTP 头/requests 最终需要 latin-1；提前过滤掉异常字符，避免整条请求失败
            name.encode("latin-1")
            value.encode("latin-1")
        except UnicodeEncodeError:
            print(f"WARNING: cookie 含非 latin-1 字符，已跳过：{name}")
            continue
        cookies[name] = value
    return cookies


def cookies_to_header(cookies: dict[str, str]) -> str:
    return "; ".join([f"{k}={v}" for k, v in cookies.items() if v])


def requests_session(
    cookies: dict[str, str] | None = None,
    *,
    trust_env: bool = True,
) -> requests.Session:
    sess = requests.Session()
    sess.trust_env = bool(trust_env)
    sess.headers.update({"User-Agent": DEFAULT_UA})
    if cookies:
        sess.cookies.update(cookies)
    return sess


def get_json(
    sess: requests.Session,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout_sec: int = 20,
) -> dict[str, Any]:
    resp = sess.get(url, params=params, headers=headers, timeout=timeout_sec)
    resp.raise_for_status()
    return resp.json()
