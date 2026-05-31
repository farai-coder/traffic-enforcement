"""
Persistent board calibration (stop line, lanes, light ROI).

Saved to board_calibration.json so calibrate.py does not rely on patching config.py.
"""

from __future__ import annotations

import json
import os

import config

_DIR = os.path.dirname(os.path.abspath(__file__))
CALIBRATION_FILE = os.path.join(os.path.dirname(_DIR), "board_calibration.json")


def defaults() -> dict:
    return {
        "stop_line_points": getattr(config, "STOP_LINE_POINTS", None),
        "lane_lines": list(getattr(config, "LANE_LINES", [])),
        "light_roi": getattr(config, "TRAFFIC_LIGHT_ROI", None),
        "stop_line_y": getattr(config, "STOP_LINE_Y", 0),
    }


def load() -> dict:
    data = defaults()
    if os.path.exists(CALIBRATION_FILE):
        try:
            with open(CALIBRATION_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
            data.update({k: v for k, v in saved.items() if v is not None})
        except (json.JSONDecodeError, OSError):
            pass
    return data


def save(stop_line_points, lane_lines, light_roi=None) -> None:
    payload = {
        "stop_line_points": stop_line_points,
        "lane_lines": lane_lines,
        "light_roi": light_roi,
    }
    if stop_line_points and len(stop_line_points) == 2:
        payload["stop_line_y"] = (
            stop_line_points[0][1] + stop_line_points[1][1]
        ) // 2
    with open(CALIBRATION_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[SAVED] {CALIBRATION_FILE}")
