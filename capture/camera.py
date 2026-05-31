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

    def start(self):
        self.cap = cv2.VideoCapture(self.source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera source: {self.source}")
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
        self.cap = cv2.VideoCapture(self.source)
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
