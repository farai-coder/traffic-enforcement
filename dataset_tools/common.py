"""
Shared helpers for the dataset builders.

Read-only with respect to the rest of the project: this module imports the
project's config/models but never mutates them. All output goes under an
output directory you pass on the command line (default: data/datasets/...).
"""

import os
import csv
import json
import glob
import cv2

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".m4v")


def ensure_dir(path):
    """Create a directory (and parents) if it does not already exist."""
    os.makedirs(path, exist_ok=True)
    return path


def iter_frames(input_path, every_n=1, max_frames=None):
    """Yield (frame_index, source_name, frame) tuples from a flexible input.

    `input_path` may be:
        * a single image file
        * a directory of images (searched recursively)
        * a single video file

    Args:
        every_n: keep 1 of every N frames (videos only; images are always kept).
        max_frames: stop after yielding this many frames (None = no limit).
    """
    count = 0

    def _done():
        return max_frames is not None and count >= max_frames

    if os.path.isdir(input_path):
        files = []
        for ext in IMAGE_EXTS:
            files.extend(glob.glob(os.path.join(input_path, "**", f"*{ext}"), recursive=True))
        for f in sorted(files):
            if _done():
                return
            img = cv2.imread(f)
            if img is None:
                continue
            yield count, os.path.splitext(os.path.basename(f))[0], img
            count += 1
        return

    ext = os.path.splitext(input_path)[1].lower()

    if ext in IMAGE_EXTS:
        img = cv2.imread(input_path)
        if img is not None:
            yield 0, os.path.splitext(os.path.basename(input_path))[0], img
        return

    if ext in VIDEO_EXTS:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {input_path}")
        stem = os.path.splitext(os.path.basename(input_path))[0]
        idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % every_n == 0:
                    if _done():
                        break
                    yield count, f"{stem}_f{idx:06d}", frame
                    count += 1
                idx += 1
        finally:
            cap.release()
        return

    raise ValueError(f"Unsupported input (not an image/video/dir): {input_path}")


def write_manifest_csv(path, rows, header):
    """Write a list of dict rows to a CSV with the given header columns."""
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in header})


def write_manifest_json(path, obj):
    """Write a JSON manifest (pretty-printed)."""
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
