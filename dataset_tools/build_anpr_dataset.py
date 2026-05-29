"""
build_anpr_dataset.py
=====================
Build an ANPR (plate-text) dataset end-to-end with the project's own models:
YOLOv8 -> vehicle crop -> PlateDetector -> plate crop -> PlateOCR (TrOCR,
microsoft/trocr-base-printed) -> plate text.

Each readable plate is saved as an image plus a sidecar label, and every
attempt is recorded in a manifest:
    <out>/images/<name>.jpg          the plate crop
    <out>/labels/<name>.txt          the OCR text (one line) — pairs with the image
    <out>/manifest.csv               source, image, ocr_text, looks_like_plate
    <out>/gt.csv                     filename,text  (handy for OCR eval/fine-tuning)

The OCR result is a *weak* (model-generated) label — review/correct it before
using it as ground truth. Nothing in the project is modified.

Examples:
    python dataset_tools/build_anpr_dataset.py --input data/plates
    python dataset_tools/build_anpr_dataset.py --input data/clips/cars.mp4 --every 10
"""

import os
import sys
import re
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from ultralytics import YOLO

import config
from anpr.plate_detector import PlateDetector
from anpr.ocr import PlateOCR
from dataset_tools import common

# Zimbabwean-style plate sanity check (2-3 letters then 3-4 digits)
ZIM_PLATE_RE = re.compile(r"^[A-Z]{2,3}\d{3,4}$")


def crop_vehicle(frame, bbox, pad_ratio=0.05):
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
    parser = argparse.ArgumentParser(description="Build an ANPR (plate-crop + OCR text) dataset.")
    parser.add_argument("--input", required=True, help="Image file, folder of images, or video.")
    parser.add_argument("--output", default="data/datasets/anpr", help="Output dataset dir.")
    parser.add_argument("--every", type=int, default=1, help="For video: keep 1 of every N frames.")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames.")
    parser.add_argument("--conf", type=float, default=config.CONFIDENCE_THRESHOLD,
                        help="Vehicle detection confidence threshold.")
    parser.add_argument("--from-plate-crops", action="store_true",
                        help="Treat each input image as an already-cropped plate "
                             "(skip vehicle + plate detection, OCR directly).")
    parser.add_argument("--keep-unreadable", action="store_true",
                        help="Also save plate crops the OCR could not read.")
    args = parser.parse_args()

    img_dir = common.ensure_dir(os.path.join(args.output, "images"))
    lbl_dir = common.ensure_dir(os.path.join(args.output, "labels"))

    model = None
    plate_detector = None
    if not args.from_plate_crops:
        print(f"[INIT] Loading YOLOv8 model: {config.YOLO_MODEL}")
        model = YOLO(config.YOLO_MODEL)
        print("[INIT] Loading PlateDetector...")
        plate_detector = PlateDetector()

    print("[INIT] Loading OCR model (microsoft/trocr-base-printed)...")
    ocr = PlateOCR()

    classes = list(config.VEHICLE_CLASSES)
    rows = []
    gt_rows = []
    n_saved = 0
    n_readable = 0

    def handle_plate(plate_crop, base_name, source):
        """OCR a plate crop and save it + its label. Returns True if saved."""
        nonlocal n_saved, n_readable
        if plate_crop is None or plate_crop.size == 0:
            return False
        text = ocr.read_plate(plate_crop)
        readable = bool(text)
        if not readable and not args.keep_unreadable:
            return False

        text = text or ""
        looks_like_plate = int(bool(ZIM_PLATE_RE.match(text)))
        out_name = f"{base_name}.jpg"
        img_path = os.path.join(img_dir, out_name)
        cv2.imwrite(img_path, plate_crop)
        with open(os.path.join(lbl_dir, f"{base_name}.txt"), "w", encoding="utf-8") as f:
            f.write(text)

        rows.append({
            "source": source,
            "image": img_path,
            "ocr_text": text,
            "looks_like_plate": looks_like_plate,
        })
        gt_rows.append({"filename": out_name, "text": text})
        n_saved += 1
        if readable:
            n_readable += 1
        return True

    for idx, name, frame in common.iter_frames(args.input, every_n=args.every,
                                               max_frames=args.max_frames):
        if args.from_plate_crops:
            handle_plate(frame, name, name)
        else:
            results = model(frame, conf=args.conf, classes=classes, verbose=False)
            boxes = results[0].boxes
            if boxes is None:
                continue
            for k in range(len(boxes)):
                x1, y1, x2, y2 = (int(v) for v in boxes.xyxy[k].tolist())
                vehicle_crop = crop_vehicle(frame, (x1, y1, x2, y2))
                if vehicle_crop is None or vehicle_crop.size == 0:
                    continue
                plate_crop = plate_detector.detect_plate(vehicle_crop)
                handle_plate(plate_crop, f"{name}_v{k}", name)

        if n_saved and n_saved % 25 == 0:
            print(f"[..] {n_saved} plate crops saved, {n_readable} readable")

    common.write_manifest_csv(
        os.path.join(args.output, "manifest.csv"), rows,
        header=["source", "image", "ocr_text", "looks_like_plate"],
    )
    common.write_manifest_csv(
        os.path.join(args.output, "gt.csv"), gt_rows,
        header=["filename", "text"],
    )

    print(f"\n[DONE] {n_saved} plate crops saved, {n_readable} with readable text.")
    print(f"[DONE] Images   : {img_dir}")
    print(f"[DONE] Labels   : {lbl_dir}")
    print(f"[DONE] Manifest : {os.path.join(args.output, 'manifest.csv')}")
    print("[NOTE] OCR text is a model-generated weak label — review before using as ground truth.")


if __name__ == "__main__":
    main()
