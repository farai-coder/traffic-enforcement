import threading

import cv2
import config

_ROTATION_CODES = {
    "cw": cv2.ROTATE_90_CLOCKWISE,
    "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE,
    "180": cv2.ROTATE_180,
}


def preprocess_frame(frame):
    """Apply the configured rotation + resize to a captured frame.

    Centralizes the rotate/resize so the live detector, the calibrator, and the
    dataset tools all see frames in exactly the same orientation and scale.
    Controlled by config.CAMERA_ROTATION and config.TARGET_FRAME_HEIGHT.
    """
    rot = getattr(config, "CAMERA_ROTATION", "cw")
    if rot:
        code = _ROTATION_CODES.get(str(rot).lower())
        if code is not None:
            frame = cv2.rotate(frame, code)
    h, w = frame.shape[:2]
    target = getattr(config, "TARGET_FRAME_HEIGHT", 500)
    scale = target / h
    return cv2.resize(frame, (int(w * scale), target))


class Camera:
    """Handles webcam video capture."""

    def __init__(self, source=None):
        self.source = source if source is not None else config.CAMERA_SOURCE
        self.cap = None
        self._lock = threading.Lock()

    @staticmethod
    def _coerce(source):
        """Allow integer indices passed as strings ('0' -> 0)."""
        if isinstance(source, str) and source.strip().isdigit():
            return int(source.strip())
        return source

    def start(self):
        """Open the camera. Never raises — if it can't open, the camera stays
        closed and can be (re)connected later via set_source()."""
        with self._lock:
            self.cap = cv2.VideoCapture(self._coerce(self.source))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
            if not self.cap.isOpened():
                print(f"[CAMERA] Could not open source: {self.source} "
                      f"(start anyway — set it later)")
        return self

    def is_opened(self):
        return self.cap is not None and self.cap.isOpened()

    def set_source(self, source):
        """Switch the camera source at runtime (from the GUI selector).

        Returns (ok: bool, message: str).
        """
        with self._lock:
            if self.cap is not None:
                self.cap.release()
            self.source = source
            self.cap = cv2.VideoCapture(self._coerce(source))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
            ok = self.cap.isOpened()
        if ok:
            return True, f"Camera connected: {source}"
        return False, f"Could not open camera: {source}"

    def read(self):
        """Return a preprocessed frame, or None if no camera is available."""
        with self._lock:
            if self.cap is None or not self.cap.isOpened():
                return None
            # Try multiple times for unreliable streams (MJPEG/HTTP)
            for _ in range(5):
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    return preprocess_frame(frame)
            # Reconnect once if stream dropped
            self.cap.release()
            self.cap = cv2.VideoCapture(self._coerce(self.source))
            ret, frame = self.cap.read()
            if ret and frame is not None:
                return preprocess_frame(frame)
        return None

    def release(self):
        with self._lock:
            if self.cap is not None:
                self.cap.release()

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
