"""
CEG4195 – Lab 2
UNet Training Script for House Segmentation
============================================
Trains the UNet defined in app.py on the dataset prepared by
scripts/pixel_mask_generation.py.

Outputs
-------
models/unet_house_seg.pth      ← best weights (highest val IoU)
models/training_history.json   ← loss + metrics per epoch
models/test_results.json       ← final test-set IoU & Dice
                                  (read by app.py /metrics endpoint)

Usage
-----
python scripts/train.py \
    --data_dir data/processed \
    --epochs   50 \
    --batch    8  \
    --lr       1e-4 \
    --output   models/unet_house_seg.pth
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Make sure app.py UNet is importable from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)

# ──────────────────────────────────────────────────────────────
# MODEL  (duplicate here so script works standalone)
# ──────────────────────────────────────────────────────────────

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1,
                 features=(64, 128, 256, 512)):
        super().__init__()
        self.downs      = nn.ModuleList()
        self.ups        = nn.ModuleList()
        self.pool       = nn.MaxPool2d(2)

        ch = in_channels
        for f in features:
            self.downs.append(DoubleConv(ch, f))
            ch = f

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        for f in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(f * 2, f))

        self.final = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x     = self.bottleneck(x)
        skips = skips[::-1]

        for i in range(0, len(self.ups), 2):
            x    = self.ups[i](x)
            skip = skips[i // 2]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            x = torch.cat([skip, x], dim=1)
            x = self.ups[i + 1](x)

        return torch.sigmoid(self.final(x))


# ──────────────────────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────────────────────

class HouseSegDataset(Dataset):
    """Paired aerial image + binary mask dataset."""

    IMG_TF = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    MASK_TF = transforms.Compose([
        transforms.Resize((256, 256),
                          interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])

    def __init__(self, root: str, split: str):
        self.img_dir  = Path(root) / "images" / split
        self.msk_dir  = Path(root) / "masks"  / split
        self.files    = sorted(self.img_dir.glob("*.png"))
        if not self.files:
            raise FileNotFoundError(
                f"No images in {self.img_dir}. "
                "Run pixel_mask_generation.py first.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fp      = self.files[idx]
        mp      = self.msk_dir / fp.name
        img     = Image.open(fp).convert("RGB")
        mask    = Image.open(mp).convert("L") if mp.exists() \
                  else Image.new("L", img.size, 0)
        img     = self.IMG_TF(img)
        mask    = self.MASK_TF(mask)
        mask    = (mask > 0.5).float()
        return img, mask


# ──────────────────────────────────────────────────────────────
# LOSS
# ──────────────────────────────────────────────────────────────

class DiceBCELoss(nn.Module):
    """Dice + Binary Cross-Entropy combined loss."""

    def forward(self, pred, target):
        bce   = F.binary_cross_entropy(pred, target, reduction="mean")
        inter = (pred * target).sum(dim=(1, 2, 3))
        dice  = 1 - (2 * inter + 1) / (
            pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1)
        return bce + dice.mean()


# ──────────────────────────────────────────────────────────────
# METRICS
# ──────────────────────────────────────────────────────────────

def batch_iou(pred: torch.Tensor, target: torch.Tensor,
              thr: float = 0.5) -> float:
    p = (pred   > thr).float()
    g = (target > thr).float()
    inter = (p * g).sum()
    union = p.sum() + g.sum() - inter
    return float(inter / (union + 1e-8))


def batch_dice(pred: torch.Tensor, target: torch.Tensor,
               thr: float = 0.5) -> float:
    p = (pred   > thr).float()
    g = (target > thr).float()
    inter = (p * g).sum()
    return float(2 * inter / (p.sum() + g.sum() + 1e-8))


# ──────────────────────────────────────────────────────────────
# TRAINING / EVAL LOOPS
# ──────────────────────────────────────────────────────────────

def run_epoch(model, loader, device, criterion,
              optimizer=None) -> tuple:
    """Single epoch.  optimizer=None → eval mode."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    losses, ious, dices = [], [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds  = model(imgs)
            loss   = criterion(preds, masks)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            losses.append(loss.item())
            ious.append(batch_iou(preds,  masks))
            dices.append(batch_dice(preds, masks))

    return float(np.mean(losses)), float(np.mean(ious)), float(np.mean(dices))


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main(args):
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── data ──────────────────────────────────────────────────────────────
    train_ds = HouseSegDataset(args.data_dir, "train")
    val_ds   = HouseSegDataset(args.data_dir, "val")
    kw       = dict(num_workers=2, pin_memory=True)
    train_dl = DataLoader(train_ds, batch_size=args.batch,
                          shuffle=True,  **kw)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch,
                          shuffle=False, **kw)
    logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ── model + optimiser ─────────────────────────────────────────────────
    model     = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                  lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    criterion = DiceBCELoss()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── training loop ─────────────────────────────────────────────────────
    history  = {k: [] for k in
                ["train_loss","val_loss","train_iou","val_iou",
                 "train_dice","val_dice"]}
    best_iou = 0.0

    logger.info("=" * 60)
    logger.info(f"Training UNet for {args.epochs} epochs")
    logger.info("=" * 60)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tr_loss, tr_iou, tr_dice = run_epoch(
            model, train_dl, device, criterion, optimizer)
        va_loss, va_iou, va_dice = run_epoch(
            model, val_dl,   device, criterion)
        scheduler.step()

        for k, v in [("train_loss", tr_loss), ("val_loss",   va_loss),
                     ("train_iou",  tr_iou),  ("val_iou",    va_iou),
                     ("train_dice", tr_dice), ("val_dice",   va_dice)]:
            history[k].append(round(v, 6))

        logger.info(
            f"Ep {epoch:03d}/{args.epochs}  "
            f"train loss={tr_loss:.4f} IoU={tr_iou:.4f} Dice={tr_dice:.4f}  │  "
            f"val   loss={va_loss:.4f} IoU={va_iou:.4f} Dice={va_dice:.4f}  "
            f"[{time.time()-t0:.1f}s]"
        )

        if va_iou > best_iou:
            best_iou = va_iou
            torch.save(model.state_dict(), out_path)
            logger.info(f"  ✓ Best model saved  (val IoU={best_iou:.4f})")

    # ── save history ──────────────────────────────────────────────────────
    hist_path = out_path.parent / "training_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"History → {hist_path}")

    # ── test-set evaluation ───────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("Test-set evaluation")
    logger.info("=" * 60)
    try:
        test_ds = HouseSegDataset(args.data_dir, "test")
        test_dl = DataLoader(test_ds, batch_size=args.batch,
                             shuffle=False, **kw)
        model.load_state_dict(torch.load(out_path, map_location=device))
        te_loss, te_iou, te_dice = run_epoch(
            model, test_dl, device, criterion)

        logger.info(
            f"TEST  loss={te_loss:.4f}  IoU={te_iou:.4f}  Dice={te_dice:.4f}")

        results = {
            "test_loss":  round(te_loss, 6),
            "test_iou":   round(te_iou,  4),
            "test_dice":  round(te_dice, 4),
            "val_iou":    round(best_iou, 4),
            "epochs":     args.epochs,
            "model":      "UNet",
            "dataset":    args.data_dir,
        }
        results_path = out_path.parent / "test_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Test results → {results_path}")
        logger.info(f"\n  IoU  = {te_iou:.4f}")
        logger.info(f"  Dice = {te_dice:.4f}")

    except FileNotFoundError as e:
        logger.warning(f"Test evaluation skipped: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train UNet for house segmentation")
    parser.add_argument("--data_dir", default="data/processed",
                        help="Root dir with images/{train,val,test} and masks/")
    parser.add_argument("--epochs",   type=int,   default=50)
    parser.add_argument("--batch",    type=int,   default=8)
    parser.add_argument("--lr",       type=float, default=1e-4)
    parser.add_argument("--output",   default="models/unet_house_seg.pth",
                        help="Path to save best weights")
    main(parser.parse_args())
