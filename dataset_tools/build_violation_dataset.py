"""
build_violation_dataset.py
==========================
Build a TRAFFIC-VIOLATION dataset by replaying footage through the project's
own ViolationDetector (YOLOv8 tracking + the stop-line / red-light / lane-change
zone logic from config.py).

Because recorded footage has no live signal, you tell the script which light
state to assume for the clip (red enforces the stop line; green/yellow do not).
For every violation the detector reports, it saves:
    <out>/frames/<type>_<track>_<idx>.jpg        raw evidence frame
    <out>/annotated/<type>_<track>_<idx>.jpg     same frame with boxes/lines drawn
    <out>/crops/<type>_<track>_<idx>.jpg         the violating vehicle, cropped
    <out>/<violation>.json                       per-violation metadata
    <out>/manifest.csv                           one row per violation

This is a labeled set of violation events for evaluation/regression of the
detector. The detector, config, and models are used exactly as-is — nothing
is modified.

Examples:
    python dataset_tools/build_violation_dataset.py --input data/clips/redlight.mp4 --light red
    python dataset_tools/build_violation_dataset.py --input data/clips/lanechange.mp4 --light green --every 2
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

import config
from detection.violation_detector import ViolationDetector
from dataset_tools import common


def main():
    parser = argparse.ArgumentParser(description="Build a traffic-violation event dataset.")
    parser.add_argument("--input", required=True, help="Video file (or folder/image) to replay.")
    parser.add_argument("--output", default="data/datasets/violations", help="Output dataset dir.")
    parser.add_argument("--light", choices=["red", "green", "yellow"], default="red",
                        help="Light state to assume for this clip (red enforces the stop line).")
    parser.add_argument("--every", type=int, default=1, help="For video: process 1 of every N frames.")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames.")
    args = parser.parse_args()

    frames_dir = common.ensure_dir(os.path.join(args.output, "frames"))
    annot_dir = common.ensure_dir(os.path.join(args.output, "annotated"))
    crops_dir = common.ensure_dir(os.path.join(args.output, "crops"))

    print("[INIT] Loading ViolationDetector (YOLOv8 + zone logic)...")
    detector = ViolationDetector()

    rows = []
    n_frames = 0
    n_violations = 0

    for idx, name, frame in common.iter_frames(args.input, every_n=args.every,
                                               max_frames=args.max_frames):
        n_frames += 1
        detections = detector.detect_vehicles(frame)
        violations = detector.check_violations(detections, args.light)

        if not violations:
            continue

        # Annotated copy reuses the project's own drawing routine
        annotated = detector.draw(frame.copy(), detections, violations, args.light)

        for v in violations:
            n_violations += 1
            track_id = v["track_id"]
            vtype = v["type"]
            x1, y1, x2, y2 = v["bbox"]
            stem = f"{vtype}_{track_id}_{idx:06d}"

            frame_path = os.path.join(frames_dir, f"{stem}.jpg")
            annot_path = os.path.join(annot_dir, f"{stem}.jpg")
            cv2.imwrite(frame_path, frame)
            cv2.imwrite(annot_path, annotated)

            crop_path = ""
            crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            if crop.size > 0:
                crop_path = os.path.join(crops_dir, f"{stem}.jpg")
                cv2.imwrite(crop_path, crop)

            meta = {
                "source": name,
                "frame_index": idx,
                "violation_type": vtype,
                "track_id": track_id,
                "bbox_xyxy": [x1, y1, x2, y2],
                "assumed_light": args.light,
                "frame_image": frame_path,
                "annotated_image": annot_path,
                "vehicle_crop": crop_path,
            }
            common.write_manifest_json(os.path.join(args.output, f"{stem}.json"), meta)
            rows.append({
                "source": name,
                "frame_index": idx,
                "violation_type": vtype,
                "track_id": track_id,
                "bbox": f"{x1} {y1} {x2} {y2}",
                "assumed_light": args.light,
                "frame_image": frame_path,
                "annotated_image": annot_path,
                "vehicle_crop": crop_path,
            })

        if n_frames % 25 == 0:
            print(f"[..] {n_frames} frames processed, {n_violations} violations so far")

    common.write_manifest_csv(
        os.path.join(args.output, "manifest.csv"), rows,
        header=["source", "frame_index", "violation_type", "track_id", "bbox",
                "assumed_light", "frame_image", "annotated_image", "vehicle_crop"],
    )

    print(f"\n[DONE] {n_frames} frames processed, {n_violations} violations captured "
          f"(assumed light = {args.light}).")
    print(f"[DONE] Frames    : {frames_dir}")
    print(f"[DONE] Annotated : {annot_dir}")
    print(f"[DONE] Crops     : {crops_dir}")
    print(f"[DONE] Manifest  : {os.path.join(args.output, 'manifest.csv')}")


if __name__ == "__main__":
    main()
