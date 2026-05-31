"""
Auto-label toy-car frames by color, producing YOLO-format labels.
=================================================================
The toys are bright, highly-saturated colors on a dark mat, so we segment each
color band, take bounding boxes of large blobs, and merge heavily-overlapping
fragments (e.g. a car split by its windshield/plate). One class: 0 = car.

This avoids hand annotation. It is approximate: cars that overlap heavily may be
merged into one box. Use --preview to write boxed images for review.

Outputs YOLO labels to: data/datasets/toy_cars/labels/all/<same-basename>.txt

Usage:
    python dataset_tools/autolabel_toys.py
    python dataset_tools/autolabel_toys.py --preview
"""

import argparse
import glob
import os

import cv2
import numpy as np

IMG_DIR = os.path.join("data", "datasets", "toy_cars", "images", "all")
LBL_DIR = os.path.join("data", "datasets", "toy_cars", "labels", "all")
PREVIEW_DIR = os.path.join("data", "datasets", "toy_cars", "preview")

# OpenCV hue is 0-179. One mask per distinct toy color so differently-colored
# cars that touch still split into separate boxes.
BANDS = [(0, 12), (12, 38), (38, 90), (90, 135), (135, 170), (170, 180)]
MIN_SAT = 140    # toys are far more saturated than skin/wood/markings
MIN_VAL = 70
MIN_AREA_FRAC = 0.012
MIN_SIDE = 25
MERGE_OVERLAP = 0.35   # merge boxes whose overlap (rel. to smaller) exceeds this


def _overlap_ratio(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix, iy = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, ix2 - ix) * max(0, iy2 - iy)
    if inter == 0:
        return 0.0
    return inter / min(aw * ah, bw * bh)


def _merge(boxes):
    boxes = list(boxes)
    changed = True
    while changed:
        changed = False
        res, used = [], [False] * len(boxes)
        for i in range(len(boxes)):
            if used[i]:
                continue
            x, y, w, h = boxes[i]
            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                if _overlap_ratio((x, y, w, h), boxes[j]) > MERGE_OVERLAP:
                    bx, by, bw, bh = boxes[j]
                    nx, ny = min(x, bx), min(y, by)
                    x2, y2 = max(x + w, bx + bw), max(y + h, by + bh)
                    x, y, w, h = nx, ny, x2 - nx, y2 - ny
                    used[j] = True
                    changed = True
            res.append((x, y, w, h))
            used[i] = True
        boxes = res
    return boxes


def boxes_for(img):
    h, w = img.shape[:2]
    area = h * w
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    out = []
    for lo, hi in BANDS:
        m = ((hue >= lo) & (hue < hi) & (sat >= MIN_SAT) & (val >= MIN_VAL)).astype("uint8") * 255
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            x, y, bw, bh = cv2.boundingRect(c)
            if bw * bh > MIN_AREA_FRAC * area and bw > MIN_SIDE and bh > MIN_SIDE:
                out.append((x, y, bw, bh))
    return _merge(out)


def main():
    parser = argparse.ArgumentParser(description="Color auto-label toy cars (YOLO format).")
    parser.add_argument("--preview", action="store_true", help="Also write boxed preview images")
    args = parser.parse_args()

    os.makedirs(LBL_DIR, exist_ok=True)
    if args.preview:
        os.makedirs(PREVIEW_DIR, exist_ok=True)

    images = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))
    if not images:
        print(f"[ERROR] No images in {IMG_DIR}. Run capture_toy_frames.py first.")
        return

    total_boxes = 0
    empty = 0
    for path in images:
        img = cv2.imread(path)
        h, w = img.shape[:2]
        boxes = boxes_for(img)
        total_boxes += len(boxes)
        if not boxes:
            empty += 1

        stem = os.path.splitext(os.path.basename(path))[0]
        with open(os.path.join(LBL_DIR, stem + ".txt"), "w") as f:
            for (x, y, bw, bh) in boxes:
                cx = (x + bw / 2) / w
                cy = (y + bh / 2) / h
                f.write(f"0 {cx:.6f} {cy:.6f} {bw / w:.6f} {bh / h:.6f}\n")

        if args.preview:
            prev = img.copy()
            for (x, y, bw, bh) in boxes:
                cv2.rectangle(prev, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.imwrite(os.path.join(PREVIEW_DIR, os.path.basename(path)), prev)

    print(f"[DONE] Labeled {len(images)} images, {total_boxes} boxes total "
          f"({empty} images had no car detected).")
    print(f"[DONE] Labels written to {LBL_DIR}")
    if args.preview:
        print(f"[DONE] Previews in {PREVIEW_DIR}")
    print("Next: python dataset_tools/prepare_toy_dataset.py")


if __name__ == "__main__":
    main()
