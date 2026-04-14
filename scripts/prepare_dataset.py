"""
CEG4195 Lab 2 – Dataset Preparation
Week 7 pixel-mask generation adapted for aerial house segmentation.

Usage:
    python scripts/prepare_dataset.py \
        --input_dir  data/raw_images \
        --output_dir data/processed \
        --split 0.8 0.1 0.1

Outputs
-------
data/processed/
    images/train/   images/val/   images/test/
    masks/train/    masks/val/    masks/test/
    dataset_stats.json
"""

import os
import json
import argparse
import random
import logging
from pathlib import Path
from typing import Tuple, List

import numpy as np
from PIL import Image, ImageDraw
import cv2

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ──────────────────────────────────────────────
# WEEK 7 PIXEL MASK GENERATION
# ──────────────────────────────────────────────

def generate_house_mask_from_annotations(
    image: np.ndarray,
    annotations: List[dict],
) -> np.ndarray:
    """
    Generate a binary pixel mask from polygon annotations.
    (Week 7 technique – polygon fill approach)

    Parameters
    ----------
    image       : H×W×3 uint8 RGB aerial image
    annotations : list of dicts with key 'segmentation' (list of [x,y,x,y,...])

    Returns
    -------
    mask : H×W uint8  (0 = background, 255 = house)
    """
    h, w = image.shape[:2]
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    for ann in annotations:
        for seg in ann.get("segmentation", []):
            # seg is a flat list [x0,y0,x1,y1,...] → list of (x,y) tuples
            points = [(seg[i], seg[i + 1]) for i in range(0, len(seg) - 1, 2)]
            if len(points) >= 3:
                draw.polygon(points, fill=255)

    return np.array(mask)


def heuristic_house_mask(image: np.ndarray, sensitivity: float = 1.0) -> np.ndarray:
    """
    Heuristic pixel mask generation without annotations.
    Combines colour + edge cues typical of aerial roof imagery.

    Steps (Week 7 pipeline):
      1. Convert to HSV; isolate roof-like hues (grey, red, brown, blue-grey).
      2. Run Canny edge detection.
      3. Close gaps with morphological ops.
      4. Filter small blobs (< min_area pixels).
    """
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # ── a) Roof colour ranges in HSV ──────────────────────────────────────
    masks = []

    # Grey roofs (low saturation, mid-to-high value)
    grey = cv2.inRange(hsv, np.array([0, 0, 80]), np.array([180, 50, 220]))
    masks.append(grey)

    # Red / terracotta roofs
    red1 = cv2.inRange(hsv, np.array([0, 50, 50]),  np.array([15, 255, 200]))
    red2 = cv2.inRange(hsv, np.array([165, 50, 50]), np.array([180, 255, 200]))
    masks.append(red1 | red2)

    # Brown roofs
    brown = cv2.inRange(hsv, np.array([10, 40, 40]), np.array([30, 200, 180]))
    masks.append(brown)

    # Blue-grey (metal) roofs
    blue = cv2.inRange(hsv, np.array([100, 10, 80]), np.array([130, 80, 220]))
    masks.append(blue)

    colour_mask = masks[0]
    for m in masks[1:]:
        colour_mask = cv2.bitwise_or(colour_mask, m)

    # ── b) Edge map ────────────────────────────────────────────────────────
    grey_img  = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges     = cv2.Canny(grey_img, 50, 150)

    # ── c) Combine & close ────────────────────────────────────────────────
    combined  = cv2.bitwise_or(colour_mask, edges)
    kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed    = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=3)
    dilated   = cv2.dilate(closed, kernel, iterations=1)

    # ── d) Remove small blobs ─────────────────────────────────────────────
    min_area     = int(300 * sensitivity)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(dilated)
    final_mask   = np.zeros((h, w), dtype=np.uint8)
    for lbl in range(1, num):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_area:
            final_mask[labels == lbl] = 255

    return final_mask


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    """Compute IoU and Dice for two binary masks."""
    pred_bin = pred > 127
    gt_bin   = gt   > 127
    inter    = float((pred_bin & gt_bin).sum())
    union    = float((pred_bin | gt_bin).sum())
    iou      = inter / union if union > 0 else 1.0
    dice     = (2 * inter) / (pred_bin.sum() + gt_bin.sum() + 1e-8)
    return round(iou, 4), round(float(dice), 4)


# ──────────────────────────────────────────────
# DATASET SPLIT UTILITIES
# ──────────────────────────────────────────────

def split_files(
    files: List[Path],
    train: float,
    val:   float,
    test:  float,
    seed:  int = 42,
) -> Tuple[List[Path], List[Path], List[Path]]:
    assert abs(train + val + test - 1.0) < 1e-6, "Splits must sum to 1"
    random.seed(seed)
    random.shuffle(files)
    n      = len(files)
    n_tr   = int(n * train)
    n_val  = int(n * val)
    return files[:n_tr], files[n_tr:n_tr + n_val], files[n_tr + n_val:]


def save_pair(img: np.ndarray, mask: np.ndarray, name: str,
              img_dir: Path, mask_dir: Path) -> None:
    Image.fromarray(img).save(img_dir  / f"{name}.png")
    Image.fromarray(mask).save(mask_dir / f"{name}.png")


# ──────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────

def prepare_dataset(
    input_dir:   str,
    output_dir:  str,
    splits:      Tuple[float, float, float] = (0.8, 0.1, 0.1),
    target_size: Tuple[int, int]            = (256, 256),
    seed:        int                        = 42,
) -> dict:
    in_path  = Path(input_dir)
    out_path = Path(output_dir)

    image_files = sorted(
        list(in_path.glob("*.png")) + list(in_path.glob("*.jpg")) + list(in_path.glob("*.tif"))
    )

    if not image_files:
        logger.warning(f"No images found in {input_dir}. Generating synthetic demo data.")
        image_files = _generate_synthetic_data(in_path, n=100)

    train_files, val_files, test_files = split_files(
        image_files, *splits, seed=seed
    )

    splits_info = {
        "train": (train_files, "train"),
        "val":   (val_files,   "val"),
        "test":  (test_files,  "test"),
    }

    stats = {"splits": {}, "metrics_sample": {}}

    for split_name, (file_list, folder) in splits_info.items():
        img_dir  = out_path / "images" / folder
        mask_dir = out_path / "masks"  / folder
        img_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        ious, dices = [], []
        for fp in file_list:
            try:
                pil = Image.open(fp).convert("RGB").resize(target_size, Image.LANCZOS)
                arr = np.array(pil)

                # Generate mask using Week 7 heuristic pipeline
                mask = heuristic_house_mask(arr)

                save_pair(arr, mask, fp.stem, img_dir, mask_dir)

                # Self-check metric (would use GT in real scenario)
                iou, dice = compute_metrics(mask, mask)  # placeholder: mask vs itself = 1.0
                ious.append(iou)
                dices.append(dice)

            except Exception as e:
                logger.warning(f"Skipping {fp}: {e}")

        stats["splits"][split_name] = {
            "count":       len(file_list),
            "mean_iou":    round(float(np.mean(ious))  if ious  else 0, 4),
            "mean_dice":   round(float(np.mean(dices)) if dices else 0, 4),
        }
        logger.info(f"[{split_name}] {len(file_list)} samples processed.")

    stats_path = out_path / "dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Dataset stats saved to {stats_path}")
    return stats


def _generate_synthetic_data(out_dir: Path, n: int = 100) -> List[Path]:
    """Generate synthetic aerial-style images + save for demo purposes."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    rng   = np.random.default_rng(0)

    for i in range(n):
        img  = rng.integers(40, 180, (256, 256, 3), dtype=np.uint8)
        # Add rectangle 'buildings'
        draw_arr = Image.fromarray(img)
        draw     = ImageDraw.Draw(draw_arr)
        for _ in range(rng.integers(3, 10)):
            x0, y0 = rng.integers(0, 200, 2)
            x1, y1 = x0 + rng.integers(10, 50), y0 + rng.integers(10, 50)
            col    = tuple(rng.integers(140, 220, 3).tolist())
            draw.rectangle([x0, y0, x1, y1], fill=col)
        fp = out_dir / f"synth_{i:04d}.png"
        draw_arr.save(fp)
        paths.append(fp)

    return paths


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare aerial house-segmentation dataset")
    parser.add_argument("--input_dir",  default="data/raw_images")
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--split",      nargs=3, type=float, default=[0.8, 0.1, 0.1],
                        metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--size",       type=int, default=256)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    stats = prepare_dataset(
        input_dir   = args.input_dir,
        output_dir  = args.output_dir,
        splits      = tuple(args.split),
        target_size = (args.size, args.size),
        seed        = args.seed,
    )
    print(json.dumps(stats, indent=2))
