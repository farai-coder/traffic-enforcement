"""
Minimal YOLO vehicle detection test.

Shows the camera feed with bounding boxes, track IDs, and confidence scores.
No violations, no ANPR. Use this to tune detection alone.

Controls: Q to quit, S to screenshot.
"""

import cv2
from ultralytics import YOLO
import config


def main():
    print(f"[INFO] Camera source: {config.CAMERA_SOURCE}")
    print(f"[INFO] Model: {config.YOLO_MODEL}, conf >= {config.CONFIDENCE_THRESHOLD}")
    print(f"[INFO] Vehicle classes (COCO): {config.VEHICLE_CLASSES}")

    cap = cv2.VideoCapture(config.CAMERA_SOURCE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera source")
        return

    model = YOLO(config.YOLO_MODEL)
    print("[INFO] Model loaded. Press Q to quit, S to screenshot.")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[WARN] Dropped frame, retrying...")
            continue

        # Match capture/camera.py: rotate 90 CW, resize to height 500
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        h, w = frame.shape[:2]
        scale = 500 / h
        frame = cv2.resize(frame, (int(w * scale), 500))

        results = model.track(
            frame,
            persist=True,
            conf=config.CONFIDENCE_THRESHOLD,
            classes=config.VEHICLE_CLASSES,
            verbose=False,
        )

        n_detections = 0
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes
            n_detections = len(boxes)
            for i in range(n_detections):
                x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[i].tolist()]
                conf = float(boxes.conf[i])
                track_id = int(boxes.id[i])
                cls = int(boxes.cls[i])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"ID:{track_id} cls:{cls} {conf:.2f}"
                cv2.putText(
                    frame, label, (x1, max(15, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
                )

        cv2.putText(
            frame, f"frame={frame_idx} dets={n_detections} conf>={config.CONFIDENCE_THRESHOLD}",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2,
        )

        cv2.imshow("Detection Test", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            path = f"data/violations/det_test_{frame_idx}.jpg"
            cv2.imwrite(path, frame)
            print(f"[SCREENSHOT] {path}")

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
