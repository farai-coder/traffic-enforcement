"""
Train a custom YOLOv8 toy-car detector.
=======================================
Fine-tunes yolov8n.pt on the dataset produced by prepare_toy_dataset.py.
The result (best.pt) is copied to the repo root as `toy_cars.pt` and the
script prints exactly how to wire it into config.py.

Usage:
    python dataset_tools/train_toy_model.py
    python dataset_tools/train_toy_model.py --epochs 100 --imgsz 640 --base yolov8s.pt
"""

import argparse
import os
import shutil

DATA_YAML = os.path.join("data", "datasets", "toy_cars", "data.yaml")
OUTPUT_WEIGHTS = "toy_cars.pt"


def main():
    parser = argparse.ArgumentParser(description="Train a toy-car YOLOv8 detector.")
    parser.add_argument("--data", default=DATA_YAML, help="Path to data.yaml")
    parser.add_argument("--base", default="yolov8n.pt", help="Base weights to fine-tune")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--name", default="toy_cars", help="Run name under runs/detect/")
    args = parser.parse_args()

    if not os.path.isfile(args.data):
        print(f"[ERROR] {args.data} not found. Run prepare_toy_dataset.py first.")
        return

    from ultralytics import YOLO

    print(f"[TRAIN] base={args.base} data={args.data} epochs={args.epochs} imgsz={args.imgsz}")
    model = YOLO(args.base)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
    )

    # Locate best.pt from the run and copy to repo root for easy use
    save_dir = getattr(results, "save_dir", None) or os.path.join("runs", "detect", args.name)
    best = os.path.join(str(save_dir), "weights", "best.pt")
    if os.path.isfile(best):
        shutil.copy2(best, OUTPUT_WEIGHTS)
        print(f"\n[DONE] Best weights: {best}")
        print(f"[DONE] Copied to:    {OUTPUT_WEIGHTS}")
        print("\nNow wire it into config.py:")
        print(f'    YOLO_MODEL = "{OUTPUT_WEIGHTS}"')
        print("    VEHICLE_CLASSES = [0]   # custom model has a single class: 0 = car")
    else:
        print(f"[WARN] Could not find best.pt under {save_dir}. Check the run output above.")


if __name__ == "__main__":
    main()
