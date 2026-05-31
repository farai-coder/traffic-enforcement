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
from capture.frame_thread import FrameCaptureThread
from capture.camera import Camera
from capture.video_buffer import VideoBuffer
from detection.traffic_state import TrafficLightDetector
from detection.violation_detector import ViolationDetector
from violations.logger import ViolationLogger
from violations.incident import (
    IncidentGate,
    PhotoSampler,
    collapse_by_vehicle,
    label_from_subject,
)
from anpr.plate_detector import PlateDetector
from anpr.plate_cache import PlateReadCache
from anpr.reader import read_plate_for_vehicle
from anpr.ocr import load_plate_ocr
from api_client import post_violation


def main():
    print("=" * 60)
    print("  TRAFFIC ENFORCEMENT SYSTEM (with ANPR)")
    print("  Press Q to quit | SPACE to pause | R to reset tracking")
    print("=" * 60)

    print("[INIT] Starting camera...")
    print(f"[INIT] CAMERA_SOURCE = {config.CAMERA_SOURCE}")
    from calibration.store import load as load_board_calibration
    cal = load_board_calibration()
    print(f"[INIT] STOP_LINE_POINTS = {cal.get('stop_line_points')}")
    print(f"[INIT] LANE_LINES = {len(cal.get('lane_lines') or [])} line(s)")
    if getattr(config, "USE_ASYNC_CAMERA", True):
        camera = FrameCaptureThread()
        camera.start()
        print("[INIT] Async camera (latest frame — low display lag)", flush=True)
    else:
        camera = Camera()
        camera.start()

    video_buffer = VideoBuffer(
        pre_seconds=getattr(config, "VIOLATION_VIDEO_PRE_SECONDS", 1.0),
        post_seconds=getattr(config, "VIOLATION_VIDEO_POST_SECONDS", 1.0),
    )
    incident_gate = IncidentGate(
        cooldown_seconds=getattr(config, "VIOLATION_COOLDOWN_SECONDS", 60),
    )
    photo_sampler = PhotoSampler(
        interval_seconds=getattr(config, "VIOLATION_PHOTO_INTERVAL_SECONDS", 0.5),
    )

    print("[INIT] Traffic light in serial mode (ESP32, 2 pairs)...")
    print(f"[INIT] CAMERA_APPROACH = {getattr(config, 'CAMERA_APPROACH', 'east_west')}")
    light_detector = TrafficLightDetector(mode="serial")

    print("[INIT] Loading YOLOv8 vehicle detector...")
    violation_detector = ViolationDetector()

    print("[INIT] Loading ANPR plate detector...")
    plate_detector = PlateDetector()

    print("[INIT] Loading OCR model (this can take a moment)...")
    ocr = load_plate_ocr()

    print("[INIT] Initializing violation logger...")
    logger = ViolationLogger()

    print("[READY] System running. Monitoring traffic...")
    print("[CONTROLS] R=Red | Y=Yellow | G=Green | Q=Quit | SPACE=Pause | T=Reset | S=Screenshot\n")

    # Do not force window size — stretching makes overlays look misaligned vs the board.
    cv2.namedWindow("Traffic Enforcement System", cv2.WINDOW_NORMAL)

    paused = False
    frame_count = 0
    plate_cache = PlateReadCache(max_attempts=20)
    videos_recorded = 0             # one short clip per vehicle per cooldown
    photos_saved = 0
    detect_skip = max(1, getattr(config, "DETECTION_FRAME_SKIP", 2))
    detect_tick = 0
    overlay = {
        "detections": [],
        "violations": [],
        "light_state": "unknown",
        "violation_ids": set(),
    }

    try:
        while True:
            if not paused:
                raw = camera.read()
                frame = raw[0] if isinstance(raw, tuple) else raw
                if frame is None:
                    cv2.waitKey(1)
                    continue

                saved_clip = video_buffer.add(frame)
                if saved_clip:
                    print(f"[VIDEO] Saved {saved_clip}", flush=True)
                frame_count += 1
                detect_tick += 1
                run_detection = detect_tick % detect_skip == 0

                if run_detection:
                    light_state = light_detector.detect(frame)
                    detections = violation_detector.detect_vehicles(frame)
                    violations = collapse_by_vehicle(
                        violation_detector.check_violations(detections, light_state)
                    )
                    overlay["detections"] = detections
                    overlay["violations"] = violations
                    overlay["light_state"] = light_state
                    overlay["violation_ids"] = {v["track_id"] for v in violations}
                else:
                    light_state = overlay["light_state"]
                    detections = overlay["detections"]
                    violations = overlay["violations"]

                evidence_subjects = violation_detector.subjects_for_evidence(
                    detections, light_state
                )
                new_offense_ids = (
                    {v["track_id"] for v in violations} if run_detection else set()
                )

                bbox_by_id = {d["track_id"]: d["bbox"] for d in detections}
                conf_by_id = {d["track_id"]: d["conf"] for d in detections}

                require_plate = getattr(config, "VIOLATION_REQUIRE_VALID_PLATE", True)

                capture_queue = []
                if run_detection:
                    for v in violations:
                        tid = v["track_id"]
                        capture_queue.append((
                            True,
                            {
                                "track_id": tid,
                                "bbox": v["bbox"],
                                "types": violation_detector.committed_violation_types(
                                    tid, {"track_id": tid, "bbox": v["bbox"]}, light_state
                                ),
                                "conf": v.get("conf", conf_by_id.get(tid, 0.0)),
                            },
                        ))
                for sub in evidence_subjects:
                    if sub["track_id"] in new_offense_ids:
                        continue
                    capture_queue.append((False, sub))

                for immediate_flag, sub in capture_queue:
                    track_id = sub["track_id"]
                    if require_plate and plate_cache.skip_capture(track_id):
                        continue
                    # Offense frame + every frame until plate is read (no interval wait).
                    urgent = immediate_flag or (
                        require_plate
                        and not plate_cache.get(track_id)
                        and not plate_cache.skip_capture(track_id)
                    )
                    if not photo_sampler.should_capture(track_id, immediate=urgent):
                        continue

                    bbox = bbox_by_id.get(track_id, sub["bbox"])

                    def _read_plate(bbox=bbox):
                        text, _ = read_plate_for_vehicle(
                            frame, bbox, plate_detector, ocr
                        )
                        return text

                    had_plate = plate_cache.get(track_id)
                    plate_number = plate_cache.resolve(track_id, _read_plate)
                    if require_plate and not plate_number:
                        continue
                    if not had_plate and plate_number:
                        print(f"[ANPR] Vehicle #{track_id} -> plate: {plate_number}", flush=True)

                    photo_sampler.mark_captured(track_id)

                    violation_label = label_from_subject(
                        violation_detector, sub, light_state
                    )
                    confidence = sub.get("conf", conf_by_id.get(track_id, 0.0))

                    image_path = logger.log(
                        violation_type=violation_label,
                        track_id=track_id,
                        plate_number=plate_number,
                        confidence=confidence,
                        frame=frame,
                    )
                    if image_path:
                        photos_saved += 1
                        if immediate_flag:
                            print(f"[CAPTURE] Offense frame | {violation_label} | "
                                  f"#{track_id} | {plate_number}", flush=True)

                    record_video = incident_gate.allow(track_id, plate_number)
                    if record_video:
                        incident_gate.mark(track_id, plate_number)
                        videos_recorded += 1

                        if image_path and getattr(config, "VIOLATION_SAVE_VIDEO", True):
                            video_path = image_path.rsplit(".", 1)[0] + ".mp4"
                            if video_buffer.trigger(video_path):
                                print(f"[VIDEO] Short clip → {video_path}", flush=True)

                        if getattr(config, "VIOLATION_POST_API", True):
                            ok, resp = post_violation(
                                violation_label, plate_number, track_id,
                                confidence, light_state, frame,
                            )
                            if ok:
                                print("[API] Violation sent (one per incident)", flush=True)
                            elif resp != "disabled":
                                print(f"[API] Post failed: {resp}", flush=True)

                # Draw annotations
                frame = light_detector.draw(frame)
                frame = violation_detector.draw(frame, detections, violations, light_state)

                # Overlay plate numbers on violating vehicles
                for det in detections:
                    plate = plate_cache.get(det["track_id"])
                    if det["track_id"] in overlay["violation_ids"] and plate:
                        x1, y1, _, _ = det["bbox"]
                        cv2.putText(frame, f"PLATE: {plate}",
                                    (x1, max(20, y1 - 30)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                cv2.putText(frame, f"Frame: {frame_count} | Approach: {light_state.upper()}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Vehicles: {len(detections)} | Photos: {photos_saved} | Videos: {videos_recorded}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                if light_state == "unknown":
                    cv2.putText(frame, "Light UNKNOWN — wait for ESP32 or press R for red",
                                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
                elif light_state not in ("red",):
                    cv2.putText(frame, "Stop-line enforced on RED only (press R to test)",
                                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                elif len(detections) == 0:
                    cv2.putText(frame, "No vehicles detected — move car into view",
                                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

                cv2.imshow("Traffic Enforcement System", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            elif key == ord(" "):
                paused = not paused
                print("[PAUSED]" if paused else "[RESUMED]")
            elif key in (ord("r"), ord("R")):
                light_detector.set_state("red")
                print("[LIGHT] Switched to RED (manual override for testing)")
            elif key in (ord("g"), ord("G")):
                light_detector.set_state("green")
                print("[LIGHT] Switched to GREEN")
            elif key in (ord("y"), ord("Y")):
                light_detector.set_state("yellow")
                print("[LIGHT] Switched to YELLOW")
            elif key in (ord("t"), ord("T")):
                plate_cache.clear()
                violation_detector.reset_tracking()
                incident_gate.clear()
                photo_sampler.clear()
                videos_recorded = 0
                photos_saved = 0
                print("[RESET] Tracking, plates, photos, and video cooldown cleared.")
            elif key in (ord("s"), ord("S")):
                path = f"data/violations/screenshot_{frame_count}.jpg"
                cv2.imwrite(path, frame)
                print(f"[SCREENSHOT] Saved to {path}")

    except KeyboardInterrupt:
        print("\n[STOPPED] Interrupted by user.")
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print(f"\n[DONE] Photos saved: {photos_saved} | Videos recorded: {videos_recorded}")
        n_plates = plate_cache.valid_count()
        print(f"[DONE] Plates read: {n_plates}")
        print(f"[DONE] Evidence saved in: {config.VIOLATIONS_DIR}/")


if __name__ == "__main__":
    main()
