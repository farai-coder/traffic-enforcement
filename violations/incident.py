"""
Incident helpers: many photos while violating, one video/API per cooldown window.
"""

import time

# Lower number = higher priority when collapsing types in the same frame.
_VIOLATION_ORDER = ("red_light", "stop_line", "lane_change")


def format_violation_types(violated) -> str:
    """Single label for CSV/API/video (e.g. red_light+lane_change)."""
    if not violated:
        return "unknown"
    if isinstance(violated, str):
        return violated
    parts = [t for t in _VIOLATION_ORDER if t in violated]
    for t in sorted(violated):
        if t not in parts:
            parts.append(t)
    return "+".join(parts)


def types_from_tracker(violation_detector, track_id: int, fallback: str) -> str:
    tracker = violation_detector.tracked_vehicles.get(track_id)
    if tracker and tracker.get("violated"):
        return format_violation_types(tracker["violated"])
    return fallback


def label_from_subject(violation_detector, subject, light_state: str) -> str:
    """All committed types for CSV/API (red_light+stop_line+lane_change when applicable)."""
    types = subject.get("types")
    if types:
        return format_violation_types(types)
    return types_from_tracker(
        violation_detector, subject["track_id"], subject.get("type", "unknown")
    )


def collapse_by_vehicle(violations):
    """One loop entry per track_id (overlay / processing); types merged at log time."""
    by_id = {}
    for v in violations:
        by_id[v["track_id"]] = v
    return list(by_id.values())


class PhotoSampler:
    """Rate-limit follow-up photos; first offense frame is always immediate (see main)."""

    def __init__(self, interval_seconds=0.5):
        self.interval = float(interval_seconds)
        self._last: dict[int, float] = {}

    def should_capture(self, track_id: int, *, immediate: bool = False) -> bool:
        if immediate:
            return True
        now = time.time()
        if self.interval <= 0:
            self._last[track_id] = now
            return True
        last = self._last.get(track_id, 0.0)
        if now - last >= self.interval:
            self._last[track_id] = now
            return True
        return False

    def mark_captured(self, track_id: int) -> None:
        self._last[track_id] = time.time()

    def clear(self) -> None:
        self._last.clear()


class IncidentGate:
    """Cooldown for one video clip and one API post per vehicle."""

    def __init__(self, cooldown_seconds=60.0):
        self.cooldown = float(cooldown_seconds)
        self._by_track: dict[int, float] = {}
        self._by_plate: dict[str, float] = {}

    def allow(self, track_id: int, plate_number: str | None = None) -> bool:
        now = time.time()
        if track_id in self._by_track and now - self._by_track[track_id] < self.cooldown:
            return False
        plate = (plate_number or "").strip().upper()
        if plate and plate != "UNKNOWN":
            if plate in self._by_plate and now - self._by_plate[plate] < self.cooldown:
                return False
        return True

    def mark(self, track_id: int, plate_number: str | None = None) -> None:
        now = time.time()
        self._by_track[track_id] = now
        plate = (plate_number or "").strip().upper()
        if plate and plate != "UNKNOWN":
            self._by_plate[plate] = now

    def clear(self) -> None:
        self._by_track.clear()
        self._by_plate.clear()
