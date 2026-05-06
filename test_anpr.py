"""
Test ANPR on a single image or live webcam.

Usage:
    python test_anpr.py                  # Live webcam - click on a vehicle to read plate
    python test_anpr.py image.jpg        # Test on a single image
"""

import sys
import os
import cv2
from anpr.plate_detector import PlateDetector
from anpr.ocr import PlateOCR

# Force print output to flush immediately
os.environ["PYTHONUNBUFFERED"] = "1"

detector = PlateDetector()
ocr = PlateOCR()


def test_image(path):
    """Test ANPR on a single image file."""
    image = cv2.imread(path)
    if image is None:
        print(f"[ERROR] Cannot read image: {path}")
        return

    print(f"[INFO] Testing ANPR on: {path}")
    print(f"[INFO] Image size: {image.shape[1]}x{image.shape[0]}")

    # Detect plate
    plate_img = detector.detect_plate(image)
    if plate_img is None:
        print("[RESULT] No plate detected")
    else:
        print(f"[RESULT] Plate region found: {plate_img.shape[1]}x{plate_img.shape[0]}")
        cv2.imshow("Detected Plate", plate_img)

        # Read text
        text = ocr.read_plate(plate_img)
        if text:
            print(f"[RESULT] Plate text: {text}")
        else:
            print("[RESULT] Could not read plate text")

    cv2.imshow("Input Image", image)
    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_webcam():
    """Live webcam test - draws detected plates in real-time."""
    import config
    cap = cv2.VideoCapture(config.CAMERA_SOURCE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam")
        return

    print("[INFO] Webcam ANPR test running", flush=True)
    print("[INFO] Press SPACE to capture and read plate, Q to quit\n", flush=True)

    last_result = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()

        # Show last result on the video frame
        if last_result:
            cv2.putText(display, last_result, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        cv2.imshow("ANPR Test - Webcam", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            print("\n[CAPTURE] Processing frame...", flush=True)
            cv2.imwrite("data/violations/debug_capture.jpg", frame)

            plate_img = detector.detect_plate(frame)
            if plate_img is None:
                last_result = "NO PLATE DETECTED"
                print("[RESULT] No plate detected in frame", flush=True)
            else:
                cv2.imwrite("data/violations/debug_plate.jpg", plate_img)
                cv2.imshow("Detected Plate", plate_img)
                text = ocr.read_plate(plate_img)
                if text:
                    last_result = f"PLATE: {text}"
                    print(f"[RESULT] *** PLATE: {text} ***", flush=True)
                else:
                    last_result = "PLATE FOUND - TEXT UNREADABLE"
                    print("[RESULT] Could not read plate text", flush=True)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_image(sys.argv[1])
    else:
        test_webcam()
