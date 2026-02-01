from __future__ import annotations

import hashlib
import re
import time
from typing import Any

from .http import get_json


_MIXIN_KEY_ENC_TAB = [
    46, 47, 18, 2, 53, 8, 23, 32,
    15, 50, 10, 31, 58, 3, 45, 35,
    27, 43, 5, 49, 33, 9, 42, 19,
    29, 28, 14, 39, 12, 38, 41, 13,
    37, 48, 7, 16, 24, 55, 40, 61,
    26, 17, 0, 1, 60, 51, 30, 4,
    22, 25, 54, 21, 56, 59, 6, 63,
    57, 62, 11, 36, 20, 34, 44, 52,
]


_CACHE: dict[str, Any] = {"img_key": "", "sub_key": "", "expires": 0.0}


def _extract_key(url: str) -> str:
    if not url:
        return ""
    name = url.rsplit("/", 1)[-1]
    if "." in name:
        name = name.split(".", 1)[0]
    return name


def _get_wbi_keys(sess) -> tuple[str, str]:
    now = time.time()
    if _CACHE.get("img_key") and _CACHE.get("sub_key") and now < float(_CACHE.get("expires", 0.0)):
        return str(_CACHE["img_key"]), str(_CACHE["sub_key"])

    payload = get_json(sess, "https://api.bilibili.com/x/web-interface/nav")
    data = payload.get("data") or {}
    wbi_img = data.get("wbi_img") or {}
    img_key = _extract_key(str(wbi_img.get("img_url") or ""))
    sub_key = _extract_key(str(wbi_img.get("sub_url") or ""))
    if not img_key or not sub_key:
        raise RuntimeError("missing wbi_img keys from nav")

    _CACHE["img_key"] = img_key
    _CACHE["sub_key"] = sub_key
    _CACHE["expires"] = now + 3600
    return img_key, sub_key


def _mixin_key(img_key: str, sub_key: str) -> str:
    origin = img_key + sub_key
    mixed = "".join([origin[i] for i in _MIXIN_KEY_ENC_TAB if i < len(origin)])
    return mixed[:32]


def _filter_value(value: object) -> str:
    s = str(value)
    return re.sub(r"[!'()*]", "", s)


def sign_wbi_params(sess, params: dict[str, object]) -> dict[str, object]:
    img_key, sub_key = _get_wbi_keys(sess)
    mixin_key = _mixin_key(img_key, sub_key)

    signed = {k: v for k, v in params.items() if v is not None}
    signed["wts"] = int(time.time())

    items = sorted(signed.items(), key=lambda kv: kv[0])
    query = "&".join([f"{k}={_filter_value(v)}" for k, v in items])
    w_rid = hashlib.md5((query + mixin_key).encode("utf-8")).hexdigest()
    signed["w_rid"] = w_rid
    return signed
