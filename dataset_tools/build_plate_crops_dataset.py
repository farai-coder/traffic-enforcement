"""
build_plate_crops_dataset.py
============================
Build a NUMBER-PLATE CROPPING dataset using the project's own pipeline:
YOLOv8 finds vehicles -> each vehicle is cropped -> PlateDetector (anpr) locates
and crops the plate inside it.

Output layout:
    <out>/vehicles/<name>_v<k>.jpg   each detected vehicle crop
    <out>/plates/<name>_v<k>.jpg     the plate crop found inside that vehicle
    <out>/manifest.csv               vehicle_crop, plate_crop, source, bbox

Only vehicles where a plate was successfully localized produce a plate crop;
the manifest records whether each vehicle yielded a plate. This is the raw
material for training/evaluating a dedicated plate-detection model later.
Nothing in the project is modified.

Examples:
    python dataset_tools/build_plate_crops_dataset.py --input data/plates
    python dataset_tools/build_plate_crops_dataset.py --input data/clips/cars.mp4 --every 10
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from ultralytics import YOLO

import config
from anpr.plate_detector import PlateDetector
from dataset_tools import common


def crop_vehicle(frame, bbox, pad_ratio=0.05):
    """Crop a vehicle from the frame with small padding (mirrors main.py)."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    pad_x = int((x2 - x1) * pad_ratio)
    pad_y = int((y2 - y1) * pad_ratio)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)
    return frame[y1:y2, x1:x2]


def main():
    parser = argparse.ArgumentParser(description="Build a vehicle + plate-crop dataset.")
    parser.add_argument("--input", required=True, help="Image file, folder of images, or video.")
    parser.add_argument("--output", default="data/datasets/plate_crops", help="Output dataset dir.")
    parser.add_argument("--every", type=int, default=1, help="For video: keep 1 of every N frames.")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames.")
    parser.add_argument("--conf", type=float, default=config.CONFIDENCE_THRESHOLD,
                        help="Vehicle detection confidence threshold.")
    parser.add_argument("--save-empty", action="store_true",
                        help="Also save vehicle crops where no plate was found.")
    args = parser.parse_args()

    classes = list(config.VEHICLE_CLASSES)
    veh_dir = common.ensure_dir(os.path.join(args.output, "vehicles"))
    plate_dir = common.ensure_dir(os.path.join(args.output, "plates"))

    print(f"[INIT] Loading YOLOv8 model: {config.YOLO_MODEL}")
    model = YOLO(config.YOLO_MODEL)
    print("[INIT] Loading PlateDetector...")
    plate_detector = PlateDetector()

    rows = []
    n_vehicles = 0
    n_plates = 0

    for idx, name, frame in common.iter_frames(args.input, every_n=args.every,
                                               max_frames=args.max_frames):
        results = model(frame, conf=args.conf, classes=classes, verbose=False)
        boxes = results[0].boxes
        if boxes is None:
            continue

        for k in range(len(boxes)):
            x1, y1, x2, y2 = (int(v) for v in boxes.xyxy[k].tolist())
            bbox = (x1, y1, x2, y2)
            vehicle_crop = crop_vehicle(frame, bbox)
            if vehicle_crop is None or vehicle_crop.size == 0:
                continue

            plate_crop = plate_detector.detect_plate(vehicle_crop)
            has_plate = plate_crop is not None and plate_crop.size > 0

            if not has_plate and not args.save_empty:
                continue

            n_vehicles += 1
            veh_name = f"{name}_v{k}.jpg"
            veh_path = os.path.join(veh_dir, veh_name)
            cv2.imwrite(veh_path, vehicle_crop)

            plate_path = ""
            if has_plate:
                n_plates += 1
                plate_path = os.path.join(plate_dir, veh_name)
                cv2.imwrite(plate_path, plate_crop)

            rows.append({
                "source": name,
                "vehicle_crop": veh_path,
                "plate_crop": plate_path,
                "has_plate": int(has_plate),
                "bbox": f"{x1} {y1} {x2} {y2}",
            })

        if len(rows) and n_vehicles % 25 == 0:
            print(f"[..] {n_vehicles} vehicle crops, {n_plates} plate crops so far")

    manifest = os.path.join(args.output, "manifest.csv")
    common.write_manifest_csv(
        manifest, rows,
        header=["source", "vehicle_crop", "plate_crop", "has_plate", "bbox"],
    )

    print(f"\n[DONE] {n_vehicles} vehicle crops, {n_plates} plate crops "
          f"({n_plates}/{n_vehicles} had a detectable plate).")
    print(f"[DONE] Vehicles : {veh_dir}")
    print(f"[DONE] Plates   : {plate_dir}")
    print(f"[DONE] Manifest : {manifest}")


if __name__ == "__main__":
    main()
