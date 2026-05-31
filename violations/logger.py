import os
import csv
import threading
import cv2
from datetime import datetime
import config


class ViolationLogger:
    """Logs traffic violations to CSV and saves evidence screenshots."""

    def __init__(self):
        os.makedirs(config.VIOLATIONS_DIR, exist_ok=True)
        self._init_csv()

    def _init_csv(self):
        if not os.path.exists(config.VIOLATIONS_CSV):
            with open(config.VIOLATIONS_CSV, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "violation_type", "track_id",
                    "plate_number", "confidence", "image_path"
                ])

    def log(self, violation_type, track_id, plate_number, confidence, frame,
            light_state="unknown"):
        """Log a violation and save the evidence frame.

        Args:
            violation_type: 'red_light', 'stop_line', or 'lane_change'
            track_id: vehicle tracking ID
            plate_number: extracted plate text (or 'UNKNOWN')
            confidence: detection confidence score
            frame: the full video frame as evidence
            light_state: traffic light state at the time ('red'/'yellow'/'green')
        """
        timestamp = datetime.now()
        timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        filename = f"{violation_type}_{track_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
        image_path = os.path.join(config.VIOLATIONS_DIR, filename)

        # Save evidence screenshot
        cv2.imwrite(image_path, frame)

        # Append to CSV
        try:
            with open(config.VIOLATIONS_CSV, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp_str,
                    violation_type,
                    track_id,
                    plate_number or "UNKNOWN",
                    f"{confidence:.2f}",
                    image_path,
                ])
        except PermissionError:
            print(f"[WARNING] Could not write to CSV (file locked). Evidence image still saved.")

        print(f"[VIOLATION] {timestamp_str} | {violation_type.upper()} | "
              f"Vehicle #{track_id} | Plate: {plate_number or 'UNKNOWN'}")

        # Send to the violations API (best-effort, off the main thread so a slow
        # or down endpoint never stalls the video loop).
        self._post_to_api(violation_type, plate_number, track_id, confidence,
                          light_state, frame.copy())

        return image_path

    def _post_to_api(self, violation_type, plate_number, track_id, confidence,
                     light_state, frame):
        if not getattr(config, "VIOLATIONS_API_URL", None):
            return

        def _send():
            try:
                import api_client
                ok, resp = api_client.post_violation(
                    violation_type=violation_type,
                    plate_number=plate_number,
                    track_id=track_id,
                    confidence=confidence,
                    light_state=light_state,
                    image_bgr=frame,
                )
                if ok:
                    print(f"[API] Sent {violation_type} #{track_id} to receiver.", flush=True)
                else:
                    print(f"[API] Send failed for #{track_id}: {resp}", flush=True)
            except Exception as e:
                print(f"[API] Send error for #{track_id}: {e}", flush=True)

        threading.Thread(target=_send, daemon=True).start()
