"""Debug a single saved ANPR frame: which strategy finds the plate, and why."""

import sys
import cv2
import numpy as np
from anpr.plate_detector import PlateDetector


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/violations/stopline_vehicle_1.jpg"
    print(f"[DEBUG] Loading: {path}")
    img = cv2.imread(path)
    if img is None:
        print("[ERROR] Cannot read image")
        return
    print(f"[DEBUG] Image: {img.shape[1]}x{img.shape[0]}")

    det = PlateDetector()

    for strategy in ("_crop_white_region", "_detect_by_color",
                     "_detect_morphological", "_detect_by_contour"):
        result = getattr(det, strategy)(img.copy())
        status = "FOUND" if result is not None else "miss"
        print(f"[{strategy}] -> {status}", end="")
        if result is not None:
            print(f"  shape={result.shape[1]}x{result.shape[0]}")
            cv2.imwrite(f"data/violations/debug_{strategy}.jpg", result)
        else:
            print()

    # Also dump the HSV white mask so we can see what the white-region path saw.
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([180, 80, 255]))
    cv2.imwrite("data/violations/debug_white_mask.jpg", white_mask)
    print(f"[DEBUG] White mask saved (V>=150, S<=80): "
          f"{cv2.countNonZero(white_mask)} / {img.shape[0]*img.shape[1]} pixels")


if __name__ == "__main__":
    main()
