"""
HTTP client for posting violations to the configured API endpoint.

Spec: api_format.md (POST /api/violations, multipart/form-data).
"""

from datetime import datetime
import os

import cv2
import requests

import config


def post_violation(
    violation_type: str,
    plate_number: str | None,
    track_id: int,
    confidence: float,
    light_state: str,
    image_bgr,
):
    """POST a violation to config.VIOLATIONS_API_URL.

    Returns (success, response_or_error). No-ops (returns (False, "disabled"))
    if VIOLATIONS_API_URL is None or empty.
    """
    url = getattr(config, "VIOLATIONS_API_URL", None)
    if not url:
        return False, "disabled"

    timestamp_iso = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    headers = {}
    token = getattr(config, "VIOLATIONS_API_TOKEN", None)
    if token:
        headers["Authorization"] = f"Bearer {token}"

    data = {
        "violation_type": violation_type,
        "timestamp": timestamp_iso,
        "vehicle_id": str(track_id),
        "confidence": f"{confidence:.4f}",
        "light_state": light_state,
    }
    if plate_number:
        data["plate_number"] = plate_number

    ok, jpg = cv2.imencode(".jpg", image_bgr)
    if not ok:
        return False, "image encode failed"
    filename = f"{violation_type}_{track_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    files = {"image": (filename, jpg.tobytes(), "image/jpeg")}

    try:
        resp = requests.post(
            url, data=data, files=files, headers=headers,
            timeout=getattr(config, "VIOLATIONS_API_TIMEOUT", 5),
        )
        return resp.ok, resp
    except requests.RequestException as e:
        return False, str(e)
