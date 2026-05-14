"""
Stop-line + lane-change violation test with ANPR (threaded).

Threads:
- Phone capture thread: reads MJPEG into a latest-frame slot.
- ANPR capture thread: reads local webcam into a latest-frame slot.
- ANPR/API worker thread: drains a queue of violation events, runs plate
  detection + OCR, and POSTs to the API. Doesn't block the main loop.

The main thread runs YOLO tracking + stop-line/lane-change checks +
OpenCV UI (imshow / waitKey must be on the main thread on Windows).

Controls: Q = quit, S = screenshot, T = reset capture history,
          R/Y/G = manual override of light state.
"""

import queue
import threading
import time

import cv2
from ultralytics import YOLO
import config
from anpr.plate_detector import PlateDetector
from anpr.ocr import PlateOCR
from detection.traffic_state import TrafficLightDetector
from api_client import post_violation


def _side_of_line(point, line_start, line_end):
    return ((line_end[0] - line_start[0]) * (point[1] - line_start[1]) -
            (line_end[1] - line_start[1]) * (point[0] - line_start[0]))


def _crossed_line(p_prev, p_curr, line_start, line_end):
    s1 = _side_of_line(p_prev, line_start, line_end)
    s2 = _side_of_line(p_curr, line_start, line_end)
    return (s1 > 0 and s2 < 0) or (s1 < 0 and s2 > 0)


class FrameCaptureThread:
    """Reads from a VideoCapture in a background thread and exposes only the
    latest frame. A glitchy stream no longer blocks the main detection loop —
    the main thread sees a stale frame instead of stalling."""

    def __init__(self, source, name="capture", api_pref=None, warmup=0):
        self.source = source
        self.name = name
        self.api_pref = api_pref
        self.warmup = warmup
        self.cap = None
        self._latest = None
        self._latest_id = 0
        self._lock = threading.Lock()
        self.running = False
        self.drop_count = 0
        self.thread = None

    def start(self):
        if self.api_pref is not None:
            self.cap = cv2.VideoCapture(self.source, self.api_pref)
        else:
            self.cap = cv2.VideoCapture(self.source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        if not self.cap.isOpened():
            return False
        for _ in range(self.warmup):
            self.cap.read()
        self.running = True
        self.thread = threading.Thread(
            target=self._loop, daemon=True, name=f"capture-{self.name}"
        )
        self.thread.start()
        return True

    def _loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.drop_count += 1
                if self.drop_count == 1 or self.drop_count % 100 == 0:
                    print(
                        f"[WARN] [{self.name}] Dropped frame, retrying... "
                        f"({self.drop_count} so far)",
                        flush=True,
                    )
                time.sleep(0.05)
                continue
            if self.drop_count:
                print(
                    f"[INFO] [{self.name}] Camera recovered after "
                    f"{self.drop_count} dropped frames",
                    flush=True,
                )
                self.drop_count = 0
            with self._lock:
                self._latest = frame
                self._latest_id += 1

    def read(self):
        with self._lock:
            if self._latest is None:
                return None, -1
            return self._latest.copy(), self._latest_id

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        if self.cap is not None:
            self.cap.release()


def anpr_worker(
    anpr_queue,
    anpr_capture,
    plate_detector,
    ocr,
    light_detector,
    preview_slot,
    preview_lock,
    stop_event,
):
    """Drain ANPR/API events: read ANPR frame, detect plate, OCR, POST."""
    tag_to_violation_type = {
        "STOPLINE": "stop_line",
        "REDLIGHT": "red_light",
        "LANECHANGE": "lane_change",
    }
    while not stop_event.is_set():
        try:
            event = anpr_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        if event is None:
            break
        track_id, tag, conf, fallback_frame = event

        if anpr_capture is not None:
            anpr_frame, _ = anpr_capture.read()
        else:
            anpr_frame = fallback_frame
        if anpr_frame is None:
            print(
                f"[ANPR] {tag} #{track_id} -> ANPR camera frame unavailable",
                flush=True,
            )
            anpr_queue.task_done()
            continue

        cv2.imwrite(
            f"data/violations/{tag.lower()}_vehicle_{track_id}.jpg", anpr_frame
        )
        anpr_preview = anpr_frame.copy()
        cv2.putText(
            anpr_preview, f"{tag} #{track_id}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2,
        )

        plate_text = None
        plate_img = plate_detector.detect_plate(anpr_frame)
        plate_preview = None
        if plate_img is None:
            print(f"[ANPR] {tag} #{track_id} -> no plate detected", flush=True)
        else:
            cv2.imwrite(
                f"data/violations/{tag.lower()}_plate_{track_id}.jpg", plate_img
            )
            text = ocr.read_plate(plate_img)
            if text:
                plate_text = text
                print(f"[ANPR] {tag} #{track_id} -> plate: {text}", flush=True)
            else:
                text = "(unreadable)"
                print(f"[ANPR] {tag} #{track_id} -> plate unreadable", flush=True)
            ph, pw = plate_img.shape[:2]
            target_w = max(360, pw)
            scale_p = target_w / pw
            preview = cv2.resize(
                plate_img, (target_w, int(ph * scale_p)),
                interpolation=cv2.INTER_CUBIC,
            )
            cv2.putText(
                preview, f"{tag} #{track_id}: {text}", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )
            plate_preview = preview

        with preview_lock:
            preview_slot["anpr"] = anpr_preview
            if plate_preview is not None:
                preview_slot["plate"] = plate_preview

        violation_type = tag_to_violation_type.get(tag, tag.lower())
        ok, resp = post_violation(
            violation_type=violation_type,
            plate_number=plate_text,
            track_id=track_id,
            confidence=conf,
            light_state=light_detector.current_state,
            image_bgr=anpr_frame,
        )
        if ok:
            print(
                f"[API] {tag} #{track_id} posted OK (HTTP {resp.status_code})",
                flush=True,
            )
        else:
            print(f"[API] {tag} #{track_id} POST failed: {resp}", flush=True)

        anpr_queue.task_done()


def main():
    print(f"[INFO] Camera source: {config.CAMERA_SOURCE}")
    print(f"[INFO] Stop line: {config.STOP_LINE_POINTS}")
    print(f"[INFO] Model: {config.YOLO_MODEL}, conf >= {config.CONFIDENCE_THRESHOLD}")

    main_capture = FrameCaptureThread(config.CAMERA_SOURCE, name="phone")
    if not main_capture.start():
        print("[ERROR] Cannot open camera source")
        return
    print("[INFO] Phone capture thread started")

    if config.ANPR_CAMERA_SOURCE == config.CAMERA_SOURCE:
        print("[INFO] ANPR camera = main camera; reusing frames (no second capture)")
        anpr_capture = None
    else:
        print(f"[INFO] ANPR camera source: {config.ANPR_CAMERA_SOURCE}")
        anpr_capture = FrameCaptureThread(
            config.ANPR_CAMERA_SOURCE, name="anpr",
            api_pref=cv2.CAP_DSHOW, warmup=10,
        )
        if not anpr_capture.start():
            print("[ERROR] Cannot open ANPR camera source")
            main_capture.stop()
            return
        actual_w = int(anpr_capture.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(anpr_capture.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[INFO] ANPR camera resolution: {actual_w}x{actual_h}")

    model = YOLO(config.YOLO_MODEL)
    print("[INFO] Loading ANPR plate detector + OCR...", flush=True)
    plate_detector = PlateDetector()
    ocr = PlateOCR()
    print("[INFO] Initializing traffic light (serial mode, ESP32)...", flush=True)
    light_detector = TrafficLightDetector(mode="serial")
    print("[INFO] Model loaded. Press Q=quit, S=screenshot, T=reset, R/Y/G=manual override.")

    p1 = tuple(config.STOP_LINE_POINTS[0])
    p2 = tuple(config.STOP_LINE_POINTS[1])
    lane_lines = [
        (tuple(line[0]), tuple(line[1]))
        for line in getattr(config, "LANE_LINES", [])
    ]

    last_capture_t = {}    # track_id -> last STOPLINE capture timestamp
    last_lane_t = {}       # track_id -> last LANECHANGE capture timestamp
    capture_cooldown_s = 1.0
    track_positions = {}   # track_id -> recent (cx, cy) positions
    frame_idx = 0
    last_seen_id = -1

    anpr_queue = queue.Queue(maxsize=20)
    preview_slot = {"anpr": None, "plate": None}
    preview_lock = threading.Lock()
    stop_event = threading.Event()
    anpr_thread = threading.Thread(
        target=anpr_worker,
        args=(
            anpr_queue, anpr_capture, plate_detector, ocr, light_detector,
            preview_slot, preview_lock, stop_event,
        ),
        daemon=True,
        name="anpr-worker",
    )
    anpr_thread.start()

    def enqueue_anpr(track_id, tag, conf, fallback_frame):
        try:
            anpr_queue.put_nowait((track_id, tag, conf, fallback_frame.copy()))
        except queue.Full:
            print(f"[ANPR] queue full, dropping {tag} #{track_id}", flush=True)

    while True:
        frame, frame_id = main_capture.read()
        if frame is None or frame_id == last_seen_id:
            # No new frame yet: pump UI so windows stay responsive.
            key = cv2.waitKey(15) & 0xFF
            if key == ord("q"):
                break
            continue
        last_seen_id = frame_id

        # Match capture/camera.py: rotate 90 CW, resize to height 500.
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        h, w = frame.shape[:2]
        scale = 500 / h
        frame = cv2.resize(frame, (int(w * scale), 500))

        # YOLO runs on the clean copy so detection isn't biased by overlay pixels.
        clean_frame = frame.copy()
        cv2.imshow("Detection Frame (clean)", clean_frame)

        cv2.line(frame, p1, p2, (0, 0, 255), 3)
        cv2.putText(frame, "STOP LINE", (p1[0], max(15, p1[1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        for ll_a, ll_b in lane_lines:
            cv2.line(frame, ll_a, ll_b, (0, 255, 255), 2)

        n_detections = 0
        n_touching = 0
        skip = max(1, getattr(config, "FRAME_SKIP", 1))
        run_yolo = (frame_idx % skip == 0)
        results = None
        if run_yolo:
            results = model.track(
                clean_frame,
                persist=True,
                conf=config.CONFIDENCE_THRESHOLD,
                classes=config.VEHICLE_CLASSES,
                verbose=False,
            )

        now = time.time()
        if (
            run_yolo
            and results
            and results[0].boxes is not None
            and results[0].boxes.id is not None
        ):
            boxes = results[0].boxes
            n_detections = len(boxes)
            for i in range(n_detections):
                x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[i].tolist()]
                track_id = int(boxes.id[i])
                conf = float(boxes.conf[i])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                history = track_positions.setdefault(track_id, [])
                history.append((cx, cy))
                if len(history) > 30:
                    del history[:-30]

                rect = (x1, y1, x2 - x1, y2 - y1)
                touches, _, _ = cv2.clipLine(rect, p1, p2)
                lane_crossed_now = False
                if touches:
                    n_touching += 1
                    last_t = last_capture_t.get(track_id, 0.0)
                    if (now - last_t) >= capture_cooldown_s:
                        last_capture_t[track_id] = now
                        light_state = light_detector.current_state
                        if light_state == "red":
                            tag = "REDLIGHT"
                            print(
                                f"[REDLIGHT] Vehicle #{track_id} ran a red light "
                                f"(bbox={x1},{y1},{x2},{y2}, conf={conf:.2f})",
                                flush=True,
                            )
                        else:
                            tag = "STOPLINE"
                            print(
                                f"[STOPLINE] Vehicle #{track_id} crossed the stop line "
                                f"(bbox={x1},{y1},{x2},{y2}, conf={conf:.2f}, "
                                f"light={light_state})",
                                flush=True,
                            )
                        enqueue_anpr(track_id, tag, conf, frame)

                if lane_lines and len(history) >= 2:
                    prev = history[-2]
                    curr = history[-1]
                    sides = []
                    for j, (ll_a, ll_b) in enumerate(lane_lines):
                        s_prev = _side_of_line(prev, ll_a, ll_b)
                        s_curr = _side_of_line(curr, ll_a, ll_b)
                        sgn_prev = "+" if s_prev > 0 else "-" if s_prev < 0 else "0"
                        sgn_curr = "+" if s_curr > 0 else "-" if s_curr < 0 else "0"
                        sides.append(f"L{j}:{sgn_prev}->{sgn_curr}")
                    print(
                        f"[TRACK] #{track_id} c=({cx},{cy}) prev=({prev[0]},{prev[1]}) "
                        f"hist={len(history)} {' '.join(sides)}",
                        flush=True,
                    )
                    last_lane = last_lane_t.get(track_id, 0.0)
                    if (now - last_lane) >= capture_cooldown_s:
                        for ll_a, ll_b in lane_lines:
                            if _crossed_line(prev, curr, ll_a, ll_b):
                                lane_crossed_now = True
                                last_lane_t[track_id] = now
                                print(
                                    f"[LANECHANGE] Vehicle #{track_id} crossed a lane line "
                                    f"(bbox={x1},{y1},{x2},{y2}, conf={conf:.2f})",
                                    flush=True,
                                )
                                enqueue_anpr(track_id, "LANECHANGE", conf, frame)
                                break

                color = (0, 0, 255) if (touches or lane_crossed_now) else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"ID:{track_id} {conf:.2f}"
                cv2.putText(frame, label, (x1, max(15, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        light_state = light_detector.current_state
        light_color_map = {
            "red": (0, 0, 255),
            "yellow": (0, 255, 255),
            "green": (0, 255, 0),
        }
        light_color = light_color_map.get(light_state, (180, 180, 180))
        cv2.putText(frame, f"LIGHT: {light_state.upper()}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, light_color, 2)
        cv2.putText(
            frame,
            f"dets={n_detections} touching={n_touching} "
            f"lane_caps={len(last_lane_t)} stop_caps={len(last_capture_t)} "
            f"q={anpr_queue.qsize()}",
            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2,
        )

        cv2.imshow("Stop Line Test", frame)

        # Pull any preview images written by the ANPR worker thread (must
        # imshow on the main thread on Windows).
        with preview_lock:
            anpr_prev = preview_slot.get("anpr")
            plate_prev = preview_slot.get("plate")
        if anpr_prev is not None:
            cv2.imshow("ANPR Frame", anpr_prev)
        if plate_prev is not None:
            cv2.imshow("Detected Plate", plate_prev)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            path = f"data/violations/stopline_test_{frame_idx}.jpg"
            cv2.imwrite(path, frame)
            print(f"[SCREENSHOT] {path}")
        if key == ord("t"):
            last_capture_t.clear()
            last_lane_t.clear()
            track_positions.clear()
            print("[RESET] Cleared capture history + lane state.")
        if key == ord("r"):
            light_detector.set_state("red")
            print("[LIGHT] manual override: RED")
        if key == ord("y"):
            light_detector.set_state("yellow")
            print("[LIGHT] manual override: YELLOW")
        if key == ord("g"):
            light_detector.set_state("green")
            print("[LIGHT] manual override: GREEN")

        frame_idx += 1

    print("[INFO] Shutting down...")
    stop_event.set()
    try:
        anpr_queue.put_nowait(None)
    except queue.Full:
        pass
    anpr_thread.join(timeout=5.0)
    main_capture.stop()
    if anpr_capture is not None:
        anpr_capture.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
