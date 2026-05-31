"""Keep one stable track_id per physical vehicle across frames (IoU matching)."""


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / max(area_a + area_b - inter, 1)


class StableTracker:
    """Assign consistent integer IDs by matching boxes frame-to-frame."""

    def __init__(self, iou_threshold=0.25, max_lost_frames=20):
        self.iou_threshold = float(iou_threshold)
        self.max_lost_frames = int(max_lost_frames)
        self._tracks: dict[int, dict] = {}
        self._next_id = 1

    def update(self, detections: list) -> list:
        if not detections:
            for tid in list(self._tracks):
                self._tracks[tid]["lost"] += 1
                if self._tracks[tid]["lost"] > self.max_lost_frames:
                    del self._tracks[tid]
            return []

        used_tracks = set()
        used_dets = set()

        # Match existing tracks to detections (best IoU first).
        pairs = []
        for tid, track in self._tracks.items():
            for j, det in enumerate(detections):
                score = _iou(track["bbox"], det["bbox"])
                if score >= self.iou_threshold:
                    pairs.append((score, tid, j))
        pairs.sort(reverse=True)

        for score, tid, j in pairs:
            if tid in used_tracks or j in used_dets:
                continue
            det = detections[j]
            det["track_id"] = tid
            self._tracks[tid]["bbox"] = det["bbox"]
            self._tracks[tid]["lost"] = 0
            used_tracks.add(tid)
            used_dets.add(j)

        # New tracks for unmatched detections.
        for j, det in enumerate(detections):
            if j in used_dets:
                continue
            tid = self._next_id
            self._next_id += 1
            det["track_id"] = tid
            self._tracks[tid] = {"bbox": det["bbox"], "lost": 0}
            used_dets.add(j)

        # Age out tracks that were not matched this frame.
        for tid in list(self._tracks):
            if tid in used_tracks:
                continue
            self._tracks[tid]["lost"] += 1
            if self._tracks[tid]["lost"] > self.max_lost_frames:
                del self._tracks[tid]

        return detections

    def active_ids(self) -> set[int]:
        return set(self._tracks.keys())

    def clear(self) -> None:
        self._tracks.clear()
        self._next_id = 1
