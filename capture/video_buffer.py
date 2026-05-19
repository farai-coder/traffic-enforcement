import time
from collections import deque
import cv2


class VideoBuffer:
    """Rolling pre-roll + on-trigger post-roll.

    Keeps the last `pre_seconds` of frames at all times. When `trigger(path)` is
    called, continues recording for `post_seconds` more, then writes the whole
    window (pre + post) to MP4 on the next `add()` after the deadline.

    If no trigger fires, old frames are silently dropped.
    """

    def __init__(self, pre_seconds=1.5, post_seconds=1.5, max_fps=30):
        self.pre_seconds = float(pre_seconds)
        self.post_seconds = float(post_seconds)
        total = self.pre_seconds + self.post_seconds
        self._frames = deque(maxlen=int(total * max_fps) + 10)
        self._pending = None  # {"path": str, "deadline": float}

    def add(self, frame):
        """Append a frame. Returns the saved path if a pending clip just flushed."""
        if frame is None:
            return None
        now = time.time()
        self._frames.append((now, frame))

        if self._pending is None:
            cutoff = now - self.pre_seconds
            while self._frames and self._frames[0][0] < cutoff:
                self._frames.popleft()
            return None

        if now >= self._pending["deadline"]:
            return self._flush()
        return None

    def trigger(self, path):
        """Arm a save: write the buffer to `path` after `post_seconds` more frames.

        Returns True if newly armed, False if a save is already pending (so the
        caller doesn't double-log)."""
        if self._pending is not None:
            return False
        self._pending = {"path": path, "deadline": time.time() + self.post_seconds}
        return True

    @property
    def pending(self):
        return self._pending is not None

    def _flush(self):
        path = self._pending["path"]
        self._pending = None

        if len(self._frames) < 2:
            return None

        first_ts = self._frames[0][0]
        last_ts = self._frames[-1][0]
        duration = max(last_ts - first_ts, 1e-3)
        fps = max(1.0, (len(self._frames) - 1) / duration)

        h, w = self._frames[0][1].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
        if not writer.isOpened():
            return None
        for _, f in self._frames:
            writer.write(f)
        writer.release()
        return path
