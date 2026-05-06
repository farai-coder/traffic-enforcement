import cv2
import config


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
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                h, w = frame.shape[:2]
                scale = 500 / h
                frame = cv2.resize(frame, (int(w * scale), 500))
                return frame
        # Reconnect if stream dropped
        self.cap.release()
        self.cap = cv2.VideoCapture(self.source)
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            h, w = frame.shape[:2]
            scale = 500 / h
            return cv2.resize(frame, (int(w * scale), 500))
        return None

    def release(self):
        if self.cap is not None:
            self.cap.release()

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
