import sys

import cv2
import config


def _open_capture(source):
    """Open VideoCapture; use DirectShow on Windows for USB webcams."""
    if isinstance(source, int) and sys.platform == "win32":
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture(source)


def preprocess_frame(frame):
    """Shared transform for live run + calibration (same pixel coordinates)."""
    if frame is None:
        return None
    if getattr(config, "CAMERA_ROTATE_90", True):
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    h, w = frame.shape[:2]
    out_h = getattr(config, "FRAME_OUTPUT_HEIGHT", 500)
    scale = out_h / h
    return cv2.resize(frame, (int(w * scale), out_h))


class Camera:
    """Handles webcam video capture."""

    def __init__(self, source=None):
        self.source = source if source is not None else config.CAMERA_SOURCE
        self.cap = None

    def start(self):
        self.cap = _open_capture(self.source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        if sys.platform == "win32":
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera source: {self.source!r}. "
                f"Try CAMERA_USB_INDEX 0 or 1 in config.py."
            )
        return self

    def read(self):
        if self.cap is None:
            raise RuntimeError("Camera not started. Call start() first.")
        # Try multiple times for unreliable streams (MJPEG/HTTP)
        for _ in range(5):
            ret, frame = self.cap.read()
            if ret and frame is not None:
                return preprocess_frame(frame)
        # Reconnect if stream dropped
        self.cap.release()
        self.cap = _open_capture(self.source)
        ret, frame = self.cap.read()
        if ret:
            return preprocess_frame(frame)
        return None

    def release(self):
        if self.cap is not None:
            self.cap.release()

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
