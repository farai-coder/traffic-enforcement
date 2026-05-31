"""
Split annotated toy-car frames into train/val and write data.yaml.
==================================================================
Reads images + YOLO-format labels you produced with an annotation tool
(Roboflow / CVAT / LabelImg) and lays them out the way Ultralytics expects:

    data/datasets/toy_cars/
        images/all/    <- captured frames (input)
        labels/all/    <- your YOLO .txt labels (input, same basenames)
        images/train/  images/val/   (output)
        labels/train/  labels/val/   (output)
        data.yaml      (output)

Single class:  0 = car

Usage:
    python dataset_tools/prepare_toy_dataset.py
    python dataset_tools/prepare_toy_dataset.py --val-split 0.2 --seed 42
"""

import argparse
import os
import random
import shutil

ROOT = os.path.join("data", "datasets", "toy_cars")
IMG_ALL = os.path.join(ROOT, "images", "all")
LBL_ALL = os.path.join(ROOT, "labels", "all")
CLASS_NAMES = ["car"]

IMG_EXTS = (".jpg", ".jpeg", ".png")


def main():
    parser = argparse.ArgumentParser(description="Split toy-car dataset + write data.yaml.")
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="Fraction of images used for validation (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed (default: 42)")
    parser.add_argument("--allow-unlabeled", action="store_true",
                        help="Include images that have no matching label file (as background)")
    args = parser.parse_args()

    if not os.path.isdir(IMG_ALL):
        print(f"[ERROR] {IMG_ALL} not found. Run capture_toy_frames.py first.")
        return

    images = sorted(f for f in os.listdir(IMG_ALL) if f.lower().endswith(IMG_EXTS))
    if not images:
        print(f"[ERROR] No images in {IMG_ALL}.")
        return

    def label_for(img_name):
        return os.path.splitext(img_name)[0] + ".txt"

    labeled, unlabeled = [], []
    for img in images:
        lbl_path = os.path.join(LBL_ALL, label_for(img))
        (labeled if os.path.isfile(lbl_path) else unlabeled).append(img)

    if unlabeled:
        print(f"[WARN] {len(unlabeled)} image(s) have no label file in {LBL_ALL}.")
        if not args.allow_unlabeled:
            print("       They will be SKIPPED. Pass --allow-unlabeled to keep them as background.")

    use = labeled + (unlabeled if args.allow_unlabeled else [])
    if not use:
        print("[ERROR] No labeled images to split. Annotate first (see TOY_TRAINING.md).")
        return

    rng = random.Random(args.seed)
    rng.shuffle(use)
    n_val = max(1, int(len(use) * args.val_split)) if len(use) > 1 else 0
    val_set = set(use[:n_val])

    # Reset output split dirs
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            d = os.path.join(ROOT, kind, split)
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d, exist_ok=True)

    counts = {"train": 0, "val": 0}
    for img in use:
        split = "val" if img in val_set else "train"
        shutil.copy2(os.path.join(IMG_ALL, img),
                     os.path.join(ROOT, "images", split, img))
        src_lbl = os.path.join(LBL_ALL, label_for(img))
        dst_lbl = os.path.join(ROOT, "labels", split, label_for(img))
        if os.path.isfile(src_lbl):
            shutil.copy2(src_lbl, dst_lbl)
        else:
            open(dst_lbl, "w").close()  # empty label = background image
        counts[split] += 1

    data_yaml = os.path.join(ROOT, "data.yaml")
    abs_root = os.path.abspath(ROOT).replace("\\", "/")
    with open(data_yaml, "w") as f:
        f.write(f"path: {abs_root}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"nc: {len(CLASS_NAMES)}\n")
        f.write("names:\n")
        for i, name in enumerate(CLASS_NAMES):
            f.write(f"  {i}: {name}\n")

    print(f"[DONE] train={counts['train']}  val={counts['val']}")
    print(f"[DONE] Wrote {data_yaml}")
    print("Next: python dataset_tools/train_toy_model.py")


if __name__ == "__main__":
    main()
