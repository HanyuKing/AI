#!/usr/bin/env python3
import os
import json
import time
import signal
import requests

BASE_URL = os.getenv("UC_BASE_URL", "https://zaohaowu.com")
# fetchToken GET 接口路径（无参数，带 cookie）
FETCH_PATH = os.getenv("UC_FETCH_PATH", "/aigc/api/auth/fetchToken")
# refreshToken GET 接口路径（原 renewalToken）
REFRESH_PATH = os.getenv("UC_REFRESH_PATH", "/aigc/api/auth/renewalToken")

# AIGC 默认 cookie 名常见是 MM_AIGC，但以你环境配置为准
COOKIE_NAME = os.getenv("UC_COOKIE_NAME", "MM_AIGC")
REFRESH_INTERVAL_SECONDS = int(os.getenv("UC_REFRESH_INTERVAL_SECONDS", "180"))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COOKIE_FILE = os.getenv("UC_COOKIE_FILE", os.path.join(SCRIPT_DIR, "cookies.json"))
COOKIE_INDEX = os.getenv("UC_COOKIE_INDEX", "").strip()
COOKIE_ENTRY_NAME = os.getenv("UC_COOKIE_ENTRY_NAME", "").strip()
RUN_ONCE = os.getenv("UC_REFRESH_ONCE", "").strip()

_stop = False


def _handle_signal(signum, frame):
    global _stop
    _stop = True


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)


def _log(message: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {message}", flush=True)


def _parse_cookie_string(cookie_str: str):
    cookie_map = {}
    if not cookie_str:
        return cookie_map
    for part in cookie_str.split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        name, value = part.split("=", 1)
        cookie_map[name.strip()] = value.strip()
    return cookie_map


def _build_cookie_string(cookie_map):
    parts = []
    for k, v in cookie_map.items():
        if v is None:
            continue
        v = str(v)
        if v == "":
            continue
        parts.append(f"{k}={v}")
    return "; ".join(parts)


def _ensure_cookie_file_exists():
    if os.path.exists(COOKIE_FILE):
        return
    template = {
        "cookies": [
            {
                "name": "Cookie 1",
                "cookie": f"{COOKIE_NAME}=PASTE_YOUR_COOKIE_HERE"
            }
        ],
        "active_index": 0,
        "note": "在此文件中添加多个 Cookie，每个 entry 的 cookie 字段写完整 Cookie 头内容"
    }
    with open(COOKIE_FILE, "w", encoding="utf-8") as f:
        json.dump(template, f, ensure_ascii=False, indent=2)


def _load_cookie_file():
    _ensure_cookie_file_exists()
    with open(COOKIE_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not isinstance(data.get("cookies"), list):
        raise ValueError(f"cookie file format invalid: expect {{cookies: [...]}} ({COOKIE_FILE})")
    if len(data["cookies"]) == 0:
        raise ValueError(f"cookie list is empty ({COOKIE_FILE})")
    return data


def _select_cookie_entry(data):
    cookies = data["cookies"]

    if COOKIE_ENTRY_NAME:
        for idx, entry in enumerate(cookies):
            if entry.get("name") == COOKIE_ENTRY_NAME:
                return idx, entry
        raise ValueError(f"Cookie entry name not found: {COOKIE_ENTRY_NAME}")

    if COOKIE_INDEX:
        idx = int(COOKIE_INDEX)
        if idx < 0 or idx >= len(cookies):
            raise IndexError(f"COOKIE_INDEX out of range: {idx}, total={len(cookies)}")
        return idx, cookies[idx]

    idx = data.get("active_index", 0)
    if not isinstance(idx, int) or idx < 0 or idx >= len(cookies):
        idx = 0
    return idx, cookies[idx]


def load_cookie(session: requests.Session):
    data = _load_cookie_file()
    idx, entry = _select_cookie_entry(data)

    cookie_str = entry.get("cookie", "")
    cookie_map = _parse_cookie_string(cookie_str)
    if not cookie_map:
        raise ValueError("Selected cookie entry is empty. Paste your cookie first.")

    for k, v in cookie_map.items():
        session.cookies.set(k, str(v), path="/")

    return data, idx, cookie_map


def save_cookie(data, idx, cookie_map):
    new_cookie_string = _build_cookie_string(cookie_map)
    data["cookies"][idx]["cookie"] = new_cookie_string
    if "active_index" in data or COOKIE_INDEX or COOKIE_ENTRY_NAME:
        data["active_index"] = idx

    with open(COOKIE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return new_cookie_string


def refresh_once():
    data = _load_cookie_file()
    cookies = data.get("cookies", [])
    if not cookies:
        raise ValueError(f"cookie list is empty ({COOKIE_FILE})")

    if COOKIE_ENTRY_NAME or COOKIE_INDEX or "active_index" in data:
        target_indexes = [ _select_cookie_entry(data)[0] ]
    else:
        target_indexes = list(range(len(cookies)))

    base = BASE_URL.rstrip("/")

    for idx in target_indexes:
        entry = cookies[idx]
        entry_name = entry.get("name") or f"Cookie {idx + 1}"
        cookie_str = entry.get("cookie", "")
        cookie_map = _parse_cookie_string(cookie_str)
        if not cookie_map:
            _log(f"[{entry_name}] skip empty cookie entry at index {idx}")
            continue

        session = requests.Session()
        for k, v in cookie_map.items():
            session.cookies.set(k, str(v), path="/")

        fetch_url = base + FETCH_PATH
        r = session.get(fetch_url, timeout=10)
        _log(f"fetchToken status: {r.status_code} body: {r.text[:200]}")

        refresh_url = base + REFRESH_PATH
        r = session.get(refresh_url, timeout=10)
        _log(f"refreshToken status: {r.status_code} body: {r.text[:200]}")

        if r.status_code == 200:
            response_cookie_map = requests.utils.dict_from_cookiejar(r.cookies)
            mm_cookie = response_cookie_map.get(COOKIE_NAME)
            if not mm_cookie:
                _log(f"[{entry_name}] 未返回{COOKIE_NAME}")
                continue

            cookie_map[COOKIE_NAME] = str(mm_cookie)
            save_cookie(data, idx, cookie_map)
            _log(f"[{entry_name}] latest_cookie: {COOKIE_NAME}={mm_cookie}")


def refresh_loop():
    interval = max(1, REFRESH_INTERVAL_SECONDS)
    while not _stop:
        start_ts = time.time()
        try:
            refresh_once()
        except Exception as e:
            _log(f"refresh_error: {str(e)}")
        elapsed = time.time() - start_ts
        sleep_for = interval - elapsed
        if sleep_for > 0:
            time.sleep(sleep_for)


if __name__ == "__main__":
    if RUN_ONCE:
        refresh_once()
    else:
        refresh_loop()
