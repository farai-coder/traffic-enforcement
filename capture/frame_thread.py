"""Background camera reader — always keeps only the newest frame (reduces lag)."""

import sys
import threading
import time

import cv2

import config
from capture.camera import _open_capture, preprocess_frame


class FrameCaptureThread:
    """Reads camera in a daemon thread; main loop always gets the latest frame."""

    def __init__(self, source=None, name="main", warmup=3):
        self.source = source if source is not None else config.CAMERA_SOURCE
        self.name = name
        self.warmup = warmup
        self.cap = None
        self._latest = None
        self._latest_id = 0
        self._lock = threading.Lock()
        self.running = False
        self.thread = None

    def start(self):
        self.cap = _open_capture(self.source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        if sys.platform == "win32":
            # Smaller driver buffer = less "frozen then jump" lag on USB cams.
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera source: {self.source!r}. "
                f"Try CAMERA_USB_INDEX 0 or 1 in config.py."
            )
        for _ in range(self.warmup):
            self.cap.read()
        self.running = True
        self.thread = threading.Thread(
            target=self._loop, daemon=True, name=f"capture-{self.name}",
        )
        self.thread.start()
        return self

    def _loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue
            processed = preprocess_frame(frame)
            with self._lock:
                self._latest = processed
                self._latest_id += 1

    def read(self):
        with self._lock:
            if self._latest is None:
                return None, -1
            return self._latest.copy(), self._latest_id

    def release(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        if self.cap is not None:
            self.cap.release()
            self.cap = None
