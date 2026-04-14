"""
CEG4195 – Lab 2
Week 7 Pixel Mask Generation  (adapted for aerial house segmentation)
======================================================================
Uses the EXACT Week 7 Mask R-CNN approach to generate pixel masks from
aerial imagery, then saves paired (image, mask) files for UNet training.

Usage
-----
# Generate masks from a folder of aerial images
python scripts/pixel_mask_generation.py \
    --input_dir  data/raw_images \
    --output_dir data/processed \
    --confidence 0.5 \
    --split 0.8 0.1 0.1

Dependencies
------------
pip install torch torchvision pillow numpy opencv-python-headless matplotlib
"""

import os
import json
import argparse
import random
import logging
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image, ImageDraw

import torch
import torchvision
from torchvision import transforms

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ──────────────────────────────────────────────────────────────
# COCO class names (from Week 7 code)
# ──────────────────────────────────────────────────────────────
COCO_CLASSES = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
    11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep',
    21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe',
    27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase',
    34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball',
    38: 'kite', 39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard',
    42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 46: 'wine glass',
    47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl',
    52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli',
    57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake',
    62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table',
    70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote',
    76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
    80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
    86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier',
    90: 'toothbrush'
}

# ──────────────────────────────────────────────────────────────
# WEEK 7 FUNCTIONS  (kept identical, comments added)
# ──────────────────────────────────────────────────────────────

def load_maskrcnn_model():
    """Load pre-trained Mask R-CNN (Week 7 approach)."""
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    device = torch.device('cuda') if torch.cuda.is_available() \
             else torch.device('cpu')
    model = model.to(device)
    logger.info(f"Mask R-CNN loaded on {device}")
    return model, device


def load_and_preprocess_image(image_path):
    """Load image and convert to tensor  (Week 7)."""
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),   # converts to [0,1] range
    ])
    image_tensor = transform(image)
    return image, image_tensor


def generate_pixel_masks(image_tensor, model, device,
                         confidence_threshold=0.7):
    """Generate pixel masks for all detected objects  (Week 7)."""
    with torch.no_grad():
        image_batch = image_tensor.unsqueeze(0).to(device)
        predictions = model(image_batch)

    pred = predictions[0]
    high_conf_mask    = pred['scores'] > confidence_threshold
    filtered_masks    = pred['masks'][high_conf_mask]   # [N,1,H,W]
    filtered_labels   = pred['labels'][high_conf_mask]
    filtered_scores   = pred['scores'][high_conf_mask]

    return filtered_masks, filtered_labels, filtered_scores


def visualize_masks(image, masks, labels, scores, class_names,
                    save_path=None):
    """Overlay masks on the original image  (Week 7)."""
    image_np  = np.array(image)
    num_masks = len(masks)
    colors    = plt.cm.tab20(np.linspace(0, 1, max(num_masks, 1)))

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image_np)
    overlay = np.zeros_like(image_np, dtype=np.float32)

    for i, (mask, label, score) in enumerate(zip(masks, labels, scores)):
        mask_np     = mask[0].cpu().numpy()
        mask_binary = mask_np > 0.5

        if mask_binary.sum() > 100:
            color = colors[i % len(colors)]
            overlay[mask_binary] = color[:3]

            coords       = np.argwhere(mask_binary)
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)

            ax.add_patch(plt.Rectangle(
                (x_min, y_min), x_max - x_min, y_max - y_min,
                fill=False, edgecolor=color, linewidth=2
            ))
            label_name = class_names.get(label.item(), f'Class {label.item()}')
            ax.text(x_min, y_min - 5,
                    f'{label_name}: {score:.2f}',
                    fontsize=10, color='white',
                    bbox=dict(facecolor=color, alpha=0.7, edgecolor='none'))

    blended = 0.4 * overlay + 0.6 * image_np / 255.0
    ax.imshow(blended, alpha=0.7)
    ax.axis('off')
    ax.set_title('Instance Segmentation with Pixel Masks')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Visualisation saved → {save_path}")
    else:
        plt.show()
    plt.close()


def save_masks_as_images(image, masks, output_dir='masks_output'):
    """Save each mask as a separate binary image  (Week 7)."""
    os.makedirs(output_dir, exist_ok=True)
    image_np = np.array(image)

    for i, mask in enumerate(masks):
        mask_np     = mask[0].cpu().numpy()
        mask_binary = (mask_np > 0.5).astype(np.uint8) * 255

        mask_path = os.path.join(output_dir, f'mask_{i:03d}.png')
        cv2.imwrite(mask_path, mask_binary)

        masked_object = image_np.copy()
        masked_object[mask_binary == 0] = 0
        object_path = os.path.join(output_dir, f'object_{i:03d}.png')
        Image.fromarray(masked_object).save(object_path)

        logger.debug(f"Saved: {mask_path} and {object_path}")


def get_pixel_mask_array(mask_tensor):
    """
    Return pixel mask as numpy array where each pixel has an object ID.
    (Week 7)
    """
    masks    = mask_tensor.cpu().numpy()
    num_masks = masks.shape[0]
    h, w     = masks.shape[2], masks.shape[3]

    combined_mask = np.zeros((h, w), dtype=np.int32)
    for i in range(num_masks):
        mask_binary = masks[i, 0] > 0.5
        combined_mask[mask_binary] = i + 1   # 0=background, 1+=object IDs

    return combined_mask


# ──────────────────────────────────────────────────────────────
# HOUSE-SPECIFIC ADAPTATION
# For aerial imagery Mask R-CNN may not detect "house" directly
# (COCO has no 'house' class).  We therefore:
#   1. Run Mask R-CNN to detect any large objects (car, truck, etc.)
#      as a warm-up / sanity check.
#   2. Build the HOUSE mask using a dedicated colour+edge heuristic
#      that is calibrated to aerial rooftop colours.
# This mirrors the Week 7 approach but targets the rooftop domain.
# ──────────────────────────────────────────────────────────────

def generate_house_mask_heuristic(image_np: np.ndarray) -> np.ndarray:
    """
    Generate a binary house/rooftop mask from an aerial RGB image.

    Pipeline  (mirrors Week 7 colour-segmentation approach):
      1. HSV colour segmentation for common rooftop hues
         (grey, red/terracotta, brown, blue-grey metal).
      2. Canny edge detection for sharp roof boundaries.
      3. Morphological closing to fill gaps inside roofs.
      4. Connected-component filtering to remove noise blobs.

    Returns
    -------
    mask : np.ndarray  uint8  (H, W), values 0 or 255
    """
    hsv     = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)

    # ── colour ranges ─────────────────────────────────────────────────────
    grey  = cv2.inRange(hsv, np.array([0,   0,  80]),
                             np.array([180, 50, 220]))
    red1  = cv2.inRange(hsv, np.array([0,  50,  50]),
                             np.array([15, 255, 200]))
    red2  = cv2.inRange(hsv, np.array([165, 50, 50]),
                             np.array([180, 255, 200]))
    brown = cv2.inRange(hsv, np.array([10, 40,  40]),
                             np.array([30, 200, 180]))
    blue  = cv2.inRange(hsv, np.array([100, 10, 80]),
                             np.array([130, 80, 220]))

    colour = grey | red1 | red2 | brown | blue

    # ── edges ─────────────────────────────────────────────────────────────
    gray_img = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    edges    = cv2.Canny(gray_img, 50, 150)

    # ── combine + morphology ──────────────────────────────────────────────
    combined = cv2.bitwise_or(colour, edges)
    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed   = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel,
                                iterations=3)
    dilated  = cv2.dilate(closed, kernel, iterations=1)

    # ── remove small blobs (< 300 px) ────────────────────────────────────
    num, labels, stats, _ = cv2.connectedComponentsWithStats(dilated)
    final = np.zeros_like(dilated)
    for lbl in range(1, num):
        if stats[lbl, cv2.CC_STAT_AREA] >= 300:
            final[labels == lbl] = 255

    return final


def merge_maskrcnn_and_heuristic(image, image_tensor, model, device,
                                 confidence=0.5) -> np.ndarray:
    """
    Combine Week 7 Mask R-CNN detections with the rooftop heuristic mask.
    Mask R-CNN finds structured objects; heuristic catches roof colours.
    Returns a single combined binary mask (H, W, uint8).
    """
    image_np = np.array(image.convert('RGB'))

    # 1. Heuristic rooftop mask
    heur_mask = generate_house_mask_heuristic(image_np)

    # 2. Mask R-CNN for any large objects (structural guidance)
    masks_t, labels, scores = generate_pixel_masks(
        image_tensor, model, device, confidence_threshold=confidence)

    mrcnn_mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
    if len(masks_t) > 0:
        combined_ids = get_pixel_mask_array(masks_t)
        mrcnn_mask[combined_ids > 0] = 255

    # 3. Union of both masks
    final_mask = cv2.bitwise_or(heur_mask, mrcnn_mask)

    # 4. Final morphological cleanup
    kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel,
                                  iterations=2)
    return final_mask


# ──────────────────────────────────────────────────────────────
# DATASET PREPARATION PIPELINE
# ──────────────────────────────────────────────────────────────

def prepare_dataset(input_dir: str, output_dir: str,
                    splits=(0.8, 0.1, 0.1),
                    target_size=(256, 256),
                    confidence: float = 0.5,
                    seed: int = 42,
                    use_maskrcnn: bool = True) -> dict:
    """
    Full pipeline:
      1. Discover / generate aerial images.
      2. Generate pixel masks (Week 7 Mask R-CNN + heuristic).
      3. Resize to target_size.
      4. Split into train / val / test.
      5. Save image–mask pairs.
      6. Return dataset statistics.
    """
    in_path  = Path(input_dir)
    out_path = Path(output_dir)

    # ── find images ───────────────────────────────────────────────────────
    exts   = ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"]
    files  = []
    for ext in exts:
        files += list(in_path.glob(ext))
    files = sorted(files)

    if not files:
        logger.warning("No real images found — generating synthetic demo set.")
        files = _generate_synthetic_aerial(in_path, n=100)

    # ── load model once ───────────────────────────────────────────────────
    mrcnn_model, device = (load_maskrcnn_model(), None)[0:2] \
        if use_maskrcnn else (None, None)
    if use_maskrcnn:
        mrcnn_model, device = load_maskrcnn_model()

    # ── shuffle + split ───────────────────────────────────────────────────
    random.seed(seed)
    random.shuffle(files)
    n      = len(files)
    n_tr   = int(n * splits[0])
    n_val  = int(n * splits[1])
    splits_map = {
        "train": files[:n_tr],
        "val":   files[n_tr:n_tr + n_val],
        "test":  files[n_tr + n_val:],
    }

    stats = {"splits": {}}

    for split_name, split_files in splits_map.items():
        img_dir  = out_path / "images" / split_name
        mask_dir = out_path / "masks"  / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"[{split_name}] Processing {len(split_files)} images …")

        for fp in split_files:
            try:
                pil_orig = Image.open(fp).convert("RGB")
                pil_img  = pil_orig.resize(target_size, Image.LANCZOS)
                arr      = np.array(pil_img)

                if use_maskrcnn:
                    _, img_tensor = load_and_preprocess_image(fp)
                    mask = merge_maskrcnn_and_heuristic(
                        pil_orig, img_tensor, mrcnn_model, device,
                        confidence=confidence)
                    mask = cv2.resize(mask, target_size,
                                      interpolation=cv2.INTER_NEAREST)
                else:
                    mask = generate_house_mask_heuristic(arr)

                Image.fromarray(arr).save(img_dir  / f"{fp.stem}.png")
                Image.fromarray(mask).save(mask_dir / f"{fp.stem}.png")

            except Exception as e:
                logger.warning(f"  Skipping {fp.name}: {e}")

        stats["splits"][split_name] = {"count": len(split_files)}
        logger.info(f"  → {split_name} done.")

    # ── save stats ────────────────────────────────────────────────────────
    stats_path = out_path / "dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Stats saved → {stats_path}")
    return stats


# ──────────────────────────────────────────────────────────────
# SYNTHETIC DATA GENERATOR  (used only when no real images found)
# ──────────────────────────────────────────────────────────────

def _generate_synthetic_aerial(out_dir: Path, n: int = 100) -> list:
    """
    Create synthetic bird's-eye-view images that look vaguely like
    aerial imagery (green background + grey/red rectangles as rooftops).
    Used only for local testing when no real dataset is available.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rng   = np.random.default_rng(0)
    paths = []

    for i in range(n):
        # Green/brown ground
        base  = rng.integers(60, 130, (256, 256, 3), dtype=np.uint8)
        base[:, :, 1] = np.clip(base[:, :, 1] + 20, 0, 255)   # greenish

        img  = Image.fromarray(base)
        draw = ImageDraw.Draw(img)

        # Draw rectangular 'rooftops'
        for _ in range(rng.integers(4, 12)):
            x0, y0 = int(rng.integers(5, 200)), int(rng.integers(5, 200))
            x1     = x0 + int(rng.integers(15, 45))
            y1     = y0 + int(rng.integers(15, 45))
            # Roof colour: grey or terracotta
            if rng.random() > 0.5:
                col = tuple(rng.integers(150, 200, 3).tolist())   # grey
            else:
                col = (int(rng.integers(180, 220)),
                       int(rng.integers(80, 120)),
                       int(rng.integers(60, 100)))                 # terracotta
            draw.rectangle([x0, y0, x1, y1], fill=col)

        fp = out_dir / f"synth_{i:04d}.png"
        img.save(fp)
        paths.append(fp)

    logger.info(f"Generated {n} synthetic aerial images in {out_dir}")
    return paths


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Week 7 pixel-mask generation for house segmentation dataset")
    parser.add_argument("--input_dir",    default="data/raw_images")
    parser.add_argument("--output_dir",   default="data/processed")
    parser.add_argument("--split",        nargs=3, type=float,
                        default=[0.8, 0.1, 0.1],
                        metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--size",         type=int, default=256)
    parser.add_argument("--confidence",   type=float, default=0.5)
    parser.add_argument("--no_maskrcnn",  action="store_true",
                        help="Use only heuristic (faster, no GPU needed)")
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args()

    assert abs(sum(args.split) - 1.0) < 1e-6, "Splits must sum to 1"

    stats = prepare_dataset(
        input_dir    = args.input_dir,
        output_dir   = args.output_dir,
        splits       = tuple(args.split),
        target_size  = (args.size, args.size),
        confidence   = args.confidence,
        seed         = args.seed,
        use_maskrcnn = not args.no_maskrcnn,
    )
    print(json.dumps(stats, indent=2))
