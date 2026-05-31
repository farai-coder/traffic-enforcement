"""
Mutable runtime settings shared across the app.

Stored in settings.json next to this file; all modules read the same values
via load(), and the GUI persists user changes via save(). Standalone — no
imports from config.py — to avoid an import cycle.
"""

import json
import os
import threading

_DIR = os.path.dirname(os.path.abspath(__file__))
SETTINGS_FILE = os.path.join(_DIR, "settings.json")

DEFAULTS = {
    "phone_ip": "192.168.60.208",
    "phone_port": 8080,
    "phone_path": "/video",
    "serial_port": "COM5",
    "serial_baud": 9600,
    # Camera source: "" = use the phone IP stream above; otherwise a webcam
    # index ("0", "1") or a full stream URL. Persisted per machine so each PC
    # remembers its own camera instead of always defaulting to the phone IP.
    "camera_source": "",
}

_lock = threading.Lock()


def load() -> dict:
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {**DEFAULTS, **data}
        except Exception:
            pass
    return dict(DEFAULTS)


def save(data: dict) -> None:
    merged = {**DEFAULTS, **data}
    with _lock:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2)


def phone_url(s: dict | None = None) -> str:
    """Build the phone IP-webcam stream URL from settings."""
    s = s or load()
    path = s.get("phone_path", "/video")
    if not path.startswith("/"):
        path = "/" + path
    return f"http://{s['phone_ip']}:{s['phone_port']}{path}"


def effective_camera_source(s: dict | None = None):
    """The camera the app should actually use.

    Returns the saved 'camera_source' (a webcam index as int, or a URL string)
    if set; otherwise falls back to the phone IP stream. This is what lets each
    machine remember its own camera instead of always using the phone IP.
    """
    s = s or load()
    src = s.get("camera_source", "")
    if src not in (None, ""):
        return int(src) if str(src).isdigit() else src
    return phone_url(s)


def set_camera_source(src) -> None:
    """Persist the chosen camera source (called by the GUI selector)."""
    data = load()
    data["camera_source"] = "" if src is None else str(src)
    save(data)


def camera_source(s: dict | None = None) -> str:
    """Backwards-compatible alias kept for older callers (phone URL)."""
    return phone_url(s)
