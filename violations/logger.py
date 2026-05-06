import os
import csv
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

    def log(self, violation_type, track_id, plate_number, confidence, frame):
        """Log a violation and save the evidence frame.

        Args:
            violation_type: 'red_light', 'stop_line', or 'lane_change'
            track_id: vehicle tracking ID
            plate_number: extracted plate text (or 'UNKNOWN')
            confidence: detection confidence score
            frame: the full video frame as evidence
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

        return image_path
