"""
Traffic Enforcement System — Main Pipeline (with ANPR)
======================================================
Captures video, detects traffic violations using YOLOv8, reads license plates
via ANPR/OCR for any violating vehicle, and logs violations + plate numbers
to CSV with evidence screenshots.

Controls:
    R       - Set light to RED
    G       - Set light to GREEN
    Y       - Set light to YELLOW
    T       - Reset tracked violations
    S       - Screenshot current frame
    SPACE   - Pause/Resume
    Q       - Quit
"""

import cv2
import config
from capture.camera import Camera
from capture.video_buffer import VideoBuffer
from detection.traffic_state import TrafficLightDetector
from detection.violation_detector import ViolationDetector
from violations.logger import ViolationLogger
from anpr.plate_detector import PlateDetector
from anpr.ocr import PlateOCR


def crop_vehicle(frame, bbox, pad_ratio=0.05):
    """Crop a vehicle from the frame with a small padding around the bbox."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    pad_x = int((x2 - x1) * pad_ratio)
    pad_y = int((y2 - y1) * pad_ratio)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)
    return frame[y1:y2, x1:x2]


def read_plate_for_vehicle(frame, bbox, plate_detector, ocr):
    """Run plate detection + OCR on a single vehicle crop. Returns (plate_text, plate_crop)."""
    vehicle_crop = crop_vehicle(frame, bbox)
    if vehicle_crop is None or vehicle_crop.size == 0:
        return None, None

    plate_crop = plate_detector.detect_plate(vehicle_crop)
    if plate_crop is None:
        return None, None

    text = ocr.read_plate(plate_crop)
    return text, plate_crop


def main():
    print("=" * 60)
    print("  TRAFFIC ENFORCEMENT SYSTEM (with ANPR)")
    print("  Press Q to quit | SPACE to pause | R to reset tracking")
    print("=" * 60)

    print("[INIT] Starting camera...")
    camera = Camera()
    camera.start()

    video_buffer = VideoBuffer(pre_seconds=1.5, post_seconds=1.5)

    print("[INIT] Traffic light in serial mode (ESP32 over COM5)...")
    light_detector = TrafficLightDetector(mode="serial")

    print("[INIT] Loading YOLOv8 vehicle detector...")
    violation_detector = ViolationDetector()

    print("[INIT] Loading ANPR plate detector...")
    plate_detector = PlateDetector()

    print("[INIT] Loading OCR model (this can take a moment)...")
    ocr = PlateOCR()

    print("[INIT] Initializing violation logger...")
    logger = ViolationLogger()

    print("[READY] System running. Monitoring traffic...")
    print("[CONTROLS] R=Red | Y=Yellow | G=Green | Q=Quit | SPACE=Pause | T=Reset | S=Screenshot\n")

    cv2.namedWindow("Traffic Enforcement System", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Traffic Enforcement System", 720, 960)

    paused = False
    frame_count = 0
    logged_violations = set()       # (track_id, type) pairs already logged
    plate_cache = {}                # track_id -> plate string (or "UNKNOWN")

    try:
        while True:
            if not paused:
                frame = camera.read()
                if frame is None:
                    print("[WARN] Dropped frame, retrying...", flush=True)
                    continue

                saved_clip = video_buffer.add(frame)
                if saved_clip:
                    print(f"[VIDEO] Saved {saved_clip}", flush=True)
                frame_count += 1

                light_state = light_detector.detect(frame)

                detections = violation_detector.detect_vehicles(frame)
                violations = violation_detector.check_violations(detections, light_state)

                # Build a quick lookup of bboxes by track id for ANPR
                bbox_by_id = {d["track_id"]: d["bbox"] for d in detections}
                conf_by_id = {d["track_id"]: d["conf"] for d in detections}

                for v in violations:
                    key = (v["track_id"], v["type"])
                    if key in logged_violations:
                        continue
                    logged_violations.add(key)

                    track_id = v["track_id"]

                    # Run ANPR once per vehicle, cache result
                    if track_id not in plate_cache:
                        bbox = bbox_by_id.get(track_id, v["bbox"])
                        plate_text, _ = read_plate_for_vehicle(
                            frame, bbox, plate_detector, ocr
                        )
                        plate_cache[track_id] = plate_text or "UNKNOWN"
                        if plate_text:
                            print(f"[ANPR] Vehicle #{track_id} -> plate: {plate_text}", flush=True)
                        else:
                            print(f"[ANPR] Vehicle #{track_id} -> plate unreadable", flush=True)

                    plate_number = plate_cache[track_id]
                    confidence = conf_by_id.get(track_id, 0.0)

                    print(f"[VIOLATION] {v['type'].upper()} | Vehicle #{track_id} | "
                          f"Plate: {plate_number}", flush=True)

                    image_path = logger.log(
                        violation_type=v["type"],
                        track_id=track_id,
                        plate_number=plate_number,
                        confidence=confidence,
                        frame=frame,
                    )

                    if image_path:
                        video_path = image_path.rsplit(".", 1)[0] + ".mp4"
                        if video_buffer.trigger(video_path):
                            print(f"[VIDEO] Recording post-roll → {video_path}", flush=True)

                # Draw annotations
                frame = light_detector.draw(frame)
                frame = violation_detector.draw(frame, detections, violations, light_state)

                # Overlay plate numbers on violating vehicles
                violation_ids = {v["track_id"] for v in violations}
                for det in detections:
                    if det["track_id"] in violation_ids and det["track_id"] in plate_cache:
                        plate = plate_cache[det["track_id"]]
                        x1, y1, _, _ = det["bbox"]
                        cv2.putText(frame, f"PLATE: {plate}",
                                    (x1, max(20, y1 - 30)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                cv2.putText(frame, f"Frame: {frame_count} | Light: {light_state.upper()}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Vehicles: {len(detections)} | Violations: {len(logged_violations)}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow("Traffic Enforcement System", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                paused = not paused
                print("[PAUSED]" if paused else "[RESUMED]")
            elif key == ord("r"):
                light_detector.set_state("red")
                print("[LIGHT] Switched to RED")
            elif key == ord("g"):
                light_detector.set_state("green")
                print("[LIGHT] Switched to GREEN")
            elif key == ord("y"):
                light_detector.set_state("yellow")
                print("[LIGHT] Switched to YELLOW")
            elif key == ord("t"):
                logged_violations.clear()
                plate_cache.clear()
                violation_detector.tracked_vehicles.clear()
                print("[RESET] Tracking, plates and violations cleared.")
            elif key == ord("s"):
                path = f"data/violations/screenshot_{frame_count}.jpg"
                cv2.imwrite(path, frame)
                print(f"[SCREENSHOT] Saved to {path}")

    except KeyboardInterrupt:
        print("\n[STOPPED] Interrupted by user.")
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print(f"\n[DONE] Total violations logged: {len(logged_violations)}")
        print(f"[DONE] Plates read: {sum(1 for p in plate_cache.values() if p != 'UNKNOWN')}/{len(plate_cache)}")
        print(f"[DONE] Evidence saved in: {config.VIOLATIONS_DIR}/")


if __name__ == "__main__":
    main()
