"""Per-vehicle plate reads: retry until a valid plate is found."""

from anpr.plate_format import normalise_plate


class PlateReadCache:
    def __init__(self, max_attempts: int = 20):
        self.max_attempts = max_attempts
        self._entries: dict[int, dict] = {}

    def resolve(self, track_id: int, read_fn) -> str | None:
        """Return a valid plate string, or None if not read yet / given up."""
        entry = self._entries.setdefault(
            track_id, {"plate": None, "attempts": 0, "given_up": False}
        )

        if entry["plate"]:
            return entry["plate"]

        if entry["given_up"]:
            return None

        if entry["attempts"] >= self.max_attempts:
            entry["given_up"] = True
            return None

        entry["attempts"] += 1
        raw = read_fn()
        valid = normalise_plate(raw)
        if valid:
            entry["plate"] = valid
            return valid

        return None

    def skip_capture(self, track_id: int) -> bool:
        """True when we should not log photos/video (no plate and retries exhausted)."""
        entry = self._entries.get(track_id)
        if not entry:
            return False
        if entry.get("plate"):
            return False
        return entry.get("given_up", False)

    def get(self, track_id: int) -> str | None:
        entry = self._entries.get(track_id)
        if entry and entry["plate"]:
            return entry["plate"]
        return None

    def clear(self) -> None:
        self._entries.clear()

    def valid_count(self) -> int:
        return sum(1 for e in self._entries.values() if e.get("plate"))
