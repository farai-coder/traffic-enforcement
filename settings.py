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


def camera_source(s: dict | None = None) -> str:
    s = s or load()
    path = s.get("phone_path", "/video")
    if not path.startswith("/"):
        path = "/" + path
    return f"http://{s['phone_ip']}:{s['phone_port']}{path}"
