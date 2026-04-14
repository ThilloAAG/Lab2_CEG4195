"""
CEG4195 – Lab 2: House Segmentation API
Author : Thillo Aïssata Ameth Gaye
Student: 300287192

Flask REST API with:
  - Secrets injection via python-dotenv
  - UNet house-segmentation model
  - IoU / Dice metric reporting
"""

import os
import io
import base64
import logging
from pathlib import Path

import numpy as np
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from PIL import Image

# ──────────────────────────────────────────────────────────────
# 1.  SECRETS INJECTION  (python-dotenv loads .env file)
# ──────────────────────────────────────────────────────────────
load_dotenv()                                    # reads .env if present

API_KEY        = os.getenv("API_KEY", "")        # optional API key gate
MODEL_PATH     = os.getenv("MODEL_PATH", "models/unet_house_seg.pth")
LOG_LEVEL      = os.getenv("LOG_LEVEL", "INFO")
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "1024"))

logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ──────────────────────────────────────────────────────────────
# 2.  LAZY TORCH IMPORT  (keeps container start fast if no GPU)
# ──────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available – running in demo mode.")

# ──────────────────────────────────────────────────────────────
# 3.  UNet ARCHITECTURE
# ──────────────────────────────────────────────────────────────

if TORCH_AVAILABLE:
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
        """
        Lightweight UNet for binary house segmentation.
        Input : (B, 3, 256, 256) normalised RGB aerial image
        Output: (B, 1, 256, 256) sigmoid probability mask
        """
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
# 4.  LOAD MODEL WEIGHTS
# ──────────────────────────────────────────────────────────────
model        = None
model_loaded = False
DEVICE       = None

if TORCH_AVAILABLE:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {DEVICE}")
    model = UNet().to(DEVICE)

    if Path(MODEL_PATH).exists():
        try:
            state = torch.load(MODEL_PATH, map_location=DEVICE)
            model.load_state_dict(state)
            model.eval()
            model_loaded = True
            logger.info(f"Model loaded from {MODEL_PATH}")
        except Exception as e:
            logger.warning(f"Could not load weights: {e}  →  demo mode")
    else:
        logger.warning(f"No weights at {MODEL_PATH}  →  demo mode")

IMG_TRANSFORM = None
if TORCH_AVAILABLE:
    IMG_TRANSFORM = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

# ──────────────────────────────────────────────────────────────
# 5.  NUMPY METRIC HELPERS  (importable by tests)
# ──────────────────────────────────────────────────────────────

def iou_score_np(pred: np.ndarray, gt: np.ndarray,
                 threshold: int = 127) -> float:
    p = pred > threshold
    g = gt   > threshold
    inter = float((p & g).sum())
    union = float((p | g).sum())
    return inter / union if union > 0 else 1.0


def dice_score_np(pred: np.ndarray, gt: np.ndarray,
                  threshold: int = 127) -> float:
    p     = pred > threshold
    g     = gt   > threshold
    inter = float((p & g).sum())
    denom = float(p.sum() + g.sum())
    return (2 * inter) / denom if denom > 0 else 1.0


def compute_metrics_from_masks(pred: np.ndarray, gt: np.ndarray):
    return iou_score_np(pred, gt), dice_score_np(pred, gt)

# ──────────────────────────────────────────────────────────────
# 6.  AUTH MIDDLEWARE
# ──────────────────────────────────────────────────────────────
import functools

def require_api_key(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if API_KEY:
            provided = request.headers.get("X-API-Key", "")
            if provided != API_KEY:
                return jsonify(
                    {"error": "Unauthorized – invalid or missing X-API-Key"}
                ), 401
        return f(*args, **kwargs)
    return decorated

# ──────────────────────────────────────────────────────────────
# 7.  ROUTES
# ──────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message":  "House Segmentation API – CEG4195 Lab 2",
        "status":   "running",
        "model":    "UNet (house segmentation)",
        "device":   str(DEVICE),
        "endpoints": {
            "/":        "GET  – Service info",
            "/health":  "GET  – Liveness probe",
            "/predict": "POST – Segment houses in aerial image",
            "/metrics": "GET  – Model evaluation metrics",
        },
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":       "healthy",
        "model_loaded": model_loaded,
        "device":       str(DEVICE),
    })


@app.route("/predict", methods=["POST"])
@require_api_key
def predict():
    """
    Body (JSON):
        image     : base64-encoded PNG/JPEG aerial image  (required)
        threshold : float 0–1, binarisation threshold     (default 0.5)

    Response (JSON):
        mask      : base64-encoded binary PNG mask
        iou       : per-image IoU  (model mode only)
        dice      : per-image Dice (model mode only)
        threshold : value used
        mode      : "model" | "demo"
    """
    try:
        data = request.get_json(force=True)
        if not data or "image" not in data:
            return jsonify({"error": "Field 'image' (base64 string) required"}), 400

        threshold = float(data.get("threshold", 0.5))

        # ── decode image ──────────────────────────────────────────────────
        img_bytes = base64.b64decode(data["image"])
        pil_img   = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        if max(pil_img.size) > MAX_IMAGE_SIZE:
            pil_img = pil_img.resize((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE),
                                     Image.LANCZOS)

        # ── inference ─────────────────────────────────────────────────────
        if model_loaded and TORCH_AVAILABLE:
            tensor = IMG_TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                prob_map = model(tensor)[0, 0].cpu().numpy()   # (256,256)
            mode = "model"
        else:
            # Demo: centre-weighted synthetic mask
            h, w   = 256, 256
            Y, X   = np.ogrid[:h, :w]
            dist   = np.sqrt((X - w//2)**2 + (Y - h//2)**2)
            prob_map = np.clip(1 - dist / (max(h, w) * 0.6), 0, 1)
            prob_map = np.clip(
                prob_map + np.random.normal(0, 0.15, prob_map.shape),
                0, 1).astype(np.float32)
            mode = "demo"

        # ── threshold → binary mask PNG ───────────────────────────────────
        binary = (prob_map >= threshold).astype(np.uint8) * 255
        buf    = io.BytesIO()
        Image.fromarray(binary).save(buf, format="PNG")
        mask_b64 = base64.b64encode(buf.getvalue()).decode()

        # ── metrics (from last eval run, stored in JSON) ──────────────────
        iou_val  = None
        dice_val = None
        if mode == "model":
            metrics_path = Path(MODEL_PATH).parent / "test_results.json"
            if metrics_path.exists():
                import json
                with open(metrics_path) as f:
                    m = json.load(f)
                iou_val  = m.get("test_iou")
                dice_val = m.get("test_dice")

        return jsonify({
            "mask":      mask_b64,
            "iou":       iou_val,
            "dice":      dice_val,
            "threshold": threshold,
            "mode":      mode,
        })

    except Exception as exc:
        logger.exception("Prediction failed")
        return jsonify({"error": str(exc)}), 500


@app.route("/metrics", methods=["GET"])
def metrics():
    """Return the evaluation metrics saved after training."""
    import json
    metrics_path = Path(MODEL_PATH).parent / "test_results.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            data = json.load(f)
        data["source"] = "test_results.json (real)"
        return jsonify(data)
    # fallback if model not yet trained
    return jsonify({
        "warning": "Model not trained yet – metrics unavailable",
        "source":  "none",
    }), 404


# ──────────────────────────────────────────────────────────────
# 8.  ENTRY POINT
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
