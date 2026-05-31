"""
Capture toy-car frames for training a custom YOLO detector.
==========================================================
Unlike capture_toys.py (which saves raw frames to data/plates/ for plate
generation), this saves frames that match the LIVE pipeline's preprocessing
(rotate 90 CW + resize to height 500, exactly like capture/camera.py), so the
model trains on the same image distribution it will see at inference.

Frames are written to:  data/datasets/toy_cars/images/all/

Controls (in the preview window):
    SPACE   - Save the current frame
    A       - Toggle auto-capture (saves every --interval seconds)
    Q / ESC - Quit

After capturing, annotate the images (see TOY_TRAINING.md), then run
prepare_toy_dataset.py and train_toy_model.py.

Usage:
    python dataset_tools/capture_toy_frames.py
    python dataset_tools/capture_toy_frames.py --source 0 --interval 1.0 --auto
"""

import argparse
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from capture.camera import preprocess_frame as preprocess

OUT_DIR = os.path.join("data", "datasets", "toy_cars", "images", "all")


def next_index(out_dir):
    """Continue numbering after any frames already captured."""
    existing = [f for f in os.listdir(out_dir) if f.startswith("toy_") and f.endswith(".jpg")]
    nums = []
    for f in existing:
        try:
            nums.append(int(f[len("toy_"):-len(".jpg")]))
        except ValueError:
            pass
    return max(nums, default=0) + 1


def main():
    parser = argparse.ArgumentParser(description="Capture toy-car frames for training.")
    parser.add_argument("--source", default=None,
                        help="Camera index or stream URL (default: config.CAMERA_SOURCE)")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Seconds between auto-captures (default: 1.0)")
    parser.add_argument("--auto", action="store_true",
                        help="Start with auto-capture enabled")
    parser.add_argument("--rotation", default="none", choices=["none", "cw", "ccw", "180"],
                        help="Frame rotation to match the live system (webcam=none, phone=cw)")
    args = parser.parse_args()

    # Match the orientation the live detector will actually run in, so training
    # data and inference frames line up. The webcam runs upright (none).
    config.CAMERA_ROTATION = None if args.rotation == "none" else args.rotation

    os.makedirs(OUT_DIR, exist_ok=True)

    source = args.source if args.source is not None else config.CAMERA_SOURCE
    # Allow integer indices passed as strings ("0" -> 0)
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera source: {source}")
        return

    idx = next_index(OUT_DIR)
    auto = args.auto
    last_auto = 0.0
    saved = 0

    print("\n" + "=" * 60)
    print("  TOY-CAR FRAME CAPTURE")
    print(f"  Saving to: {OUT_DIR}")
    print("  SPACE=save | A=toggle auto | Q/ESC=quit")
    print("=" * 60 + "\n")

    window = "Capture Toy Frames"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[WARN] Dropped frame, retrying...")
            continue

        frame = preprocess(frame)
        display = frame.copy()

        do_save = False
        if auto and (time.time() - last_auto) >= args.interval:
            do_save = True
            last_auto = time.time()

        status = f"SPACE=save A=auto({'ON' if auto else 'off'}) Q=quit | saved: {saved}"
        cv2.rectangle(display, (0, 0), (display.shape[1], 28), (0, 0, 0), -1)
        cv2.putText(display, status, (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.imshow(window, display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            do_save = True
        elif key == ord("a"):
            auto = not auto
            last_auto = time.time()
            print(f"[AUTO] {'enabled' if auto else 'disabled'}")

        if do_save:
            path = os.path.join(OUT_DIR, f"toy_{idx:05d}.jpg")
            cv2.imwrite(path, frame)
            idx += 1
            saved += 1
            print(f"[SAVED] {path}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[DONE] {saved} frames saved in {OUT_DIR}")
    print("Next: annotate them (see TOY_TRAINING.md), then run prepare_toy_dataset.py")


if __name__ == "__main__":
    main()
