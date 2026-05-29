"""
build_vehicle_dataset.py
========================
Auto-label raw footage into a YOLO-format VEHICLE-DETECTION dataset using the
exact YOLOv8 model this project already runs (config.YOLO_MODEL,
config.VEHICLE_CLASSES, config.CONFIDENCE_THRESHOLD).

For every input frame it runs YOLO, keeps the vehicle classes, and writes:
    <out>/images/<name>.jpg          the frame
    <out>/labels/<name>.txt          YOLO labels: "<cls> <cx> <cy> <w> <h>" (normalized)
    <out>/dataset.yaml               Ultralytics dataset descriptor
    <out>/preview/<name>.jpg         (optional) boxes drawn, for eyeballing

COCO vehicle ids (2 car, 3 motorcycle, 5 bus, 7 truck) are remapped to a
contiguous 0..N index so the dataset trains cleanly. Nothing in the project is
modified — this only reads frames in and writes a dataset out.

Examples:
    python dataset_tools/build_vehicle_dataset.py --input data/clips/intersection.mp4 --every 5
    python dataset_tools/build_vehicle_dataset.py --input data/raw_images --preview
"""

import os
import sys
import argparse

# Make the project root importable when run as `python dataset_tools/...`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from ultralytics import YOLO

import config
from dataset_tools import common

# COCO class id -> human-readable name (the project's VEHICLE_CLASSES)
COCO_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def main():
    parser = argparse.ArgumentParser(description="Build a YOLO-format vehicle-detection dataset.")
    parser.add_argument("--input", required=True, help="Image file, folder of images, or video.")
    parser.add_argument("--output", default="data/datasets/vehicles", help="Output dataset dir.")
    parser.add_argument("--every", type=int, default=1, help="For video: keep 1 of every N frames.")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames.")
    parser.add_argument("--conf", type=float, default=config.CONFIDENCE_THRESHOLD,
                        help="Detection confidence threshold.")
    parser.add_argument("--preview", action="store_true", help="Also save frames with boxes drawn.")
    args = parser.parse_args()

    classes = list(config.VEHICLE_CLASSES)
    # Build a stable contiguous remap: COCO id -> 0..N-1
    remap = {coco_id: i for i, coco_id in enumerate(classes)}
    names = [COCO_NAMES.get(c, str(c)) for c in classes]

    img_dir = common.ensure_dir(os.path.join(args.output, "images"))
    lbl_dir = common.ensure_dir(os.path.join(args.output, "labels"))
    prev_dir = common.ensure_dir(os.path.join(args.output, "preview")) if args.preview else None

    print(f"[INIT] Loading YOLOv8 model: {config.YOLO_MODEL}")
    model = YOLO(config.YOLO_MODEL)

    total_frames = 0
    total_boxes = 0

    for idx, name, frame in common.iter_frames(args.input, every_n=args.every,
                                               max_frames=args.max_frames):
        results = model(frame, conf=args.conf, classes=classes, verbose=False)
        boxes = results[0].boxes
        h, w = frame.shape[:2]

        lines = []
        preview = frame.copy() if args.preview else None

        if boxes is not None:
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                coco_cls = int(boxes.cls[i])
                if coco_cls not in remap:
                    continue
                cls = remap[coco_cls]
                # Normalize to YOLO format (center x/y, width, height)
                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                total_boxes += 1

                if preview is not None:
                    conf = float(boxes.conf[i])
                    cv2.rectangle(preview, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(preview, f"{names[cls]} {conf:.2f}", (int(x1), int(y1) - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Always write the image + a label file (empty label = valid negative sample)
        cv2.imwrite(os.path.join(img_dir, f"{name}.jpg"), frame)
        with open(os.path.join(lbl_dir, f"{name}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        if preview is not None:
            cv2.imwrite(os.path.join(prev_dir, f"{name}.jpg"), preview)

        total_frames += 1
        if total_frames % 25 == 0:
            print(f"[..] {total_frames} frames, {total_boxes} boxes so far")

    # Ultralytics dataset descriptor
    yaml_path = os.path.join(args.output, "dataset.yaml")
    abs_out = os.path.abspath(args.output).replace("\\", "/")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {abs_out}\n")
        f.write("train: images\n")
        f.write("val: images\n")
        f.write(f"nc: {len(names)}\n")
        f.write(f"names: {names}\n")

    print(f"\n[DONE] {total_frames} frames, {total_boxes} vehicle boxes.")
    print(f"[DONE] Images : {img_dir}")
    print(f"[DONE] Labels : {lbl_dir}")
    print(f"[DONE] Config : {yaml_path}")


if __name__ == "__main__":
    main()
