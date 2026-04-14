"""
CEG4195 – Lab 2  |  Unit Tests
================================
Run:  pytest tests/ -v
"""

import io
import base64
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

# ── project root on path ──────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Ensure no real secrets are required during tests
os.environ.setdefault("API_KEY",    "")
os.environ.setdefault("MODEL_PATH", "models/unet_house_seg.pth")
os.environ.setdefault("LOG_LEVEL",  "WARNING")

from app import app, iou_score_np, dice_score_np, compute_metrics_from_masks


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def make_b64_image(size=(64, 64)) -> str:
    """Create a tiny base64-encoded RGB PNG for test payloads."""
    arr = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ──────────────────────────────────────────────────────────────
# Flask endpoint tests
# ──────────────────────────────────────────────────────────────

class TestHomeEndpoint(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_returns_200(self):
        self.assertEqual(self.client.get("/").status_code, 200)

    def test_has_required_keys(self):
        data = json.loads(self.client.get("/").data)
        for key in ("message", "status", "endpoints"):
            self.assertIn(key, data)


class TestHealthEndpoint(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_returns_200(self):
        self.assertEqual(self.client.get("/health").status_code, 200)

    def test_has_model_loaded_field(self):
        data = json.loads(self.client.get("/health").data)
        self.assertIn("model_loaded", data)
        self.assertIn("device",       data)
        self.assertIn("status",       data)


class TestPredictEndpoint(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def _post(self, payload, headers=None):
        return self.client.post(
            "/predict",
            json=payload,
            content_type="application/json",
            headers=headers or {},
        )

    # ── validation ────────────────────────────────────────────
    def test_empty_body_returns_400(self):
        self.assertEqual(self._post({}).status_code, 400)

    def test_wrong_field_returns_400(self):
        self.assertEqual(self._post({"text": "oops"}).status_code, 400)

    def test_invalid_base64_returns_500(self):
        self.assertEqual(
            self._post({"image": "not!!base64"}).status_code, 500)

    # ── happy path ────────────────────────────────────────────
    def test_valid_image_returns_200(self):
        self.assertEqual(
            self._post({"image": make_b64_image()}).status_code, 200)

    def test_response_contains_mask_key(self):
        data = json.loads(self._post({"image": make_b64_image()}).data)
        self.assertIn("mask", data)

    def test_mask_is_valid_base64_png(self):
        data      = json.loads(self._post({"image": make_b64_image()}).data)
        raw       = base64.b64decode(data["mask"])
        img       = Image.open(io.BytesIO(raw))
        self.assertIn(img.mode, ("L", "RGB", "P"))

    def test_mode_field_present(self):
        data = json.loads(self._post({"image": make_b64_image()}).data)
        self.assertIn("mode", data)
        self.assertIn(data["mode"], ("model", "demo"))

    def test_custom_threshold_echoed(self):
        data = json.loads(
            self._post({"image": make_b64_image(), "threshold": 0.3}).data)
        self.assertAlmostEqual(data["threshold"], 0.3, places=5)

    # ── authentication ────────────────────────────────────────
    def test_api_key_required_when_set(self):
        with patch("app.API_KEY", "supersecret"):
            resp = self._post({"image": make_b64_image()})
            self.assertEqual(resp.status_code, 401)

    def test_correct_api_key_accepted(self):
        with patch("app.API_KEY", "supersecret"):
            resp = self._post(
                {"image": make_b64_image()},
                headers={"X-API-Key": "supersecret"},
            )
            self.assertEqual(resp.status_code, 200)

    def test_wrong_api_key_rejected(self):
        with patch("app.API_KEY", "supersecret"):
            resp = self._post(
                {"image": make_b64_image()},
                headers={"X-API-Key": "wrong"},
            )
            self.assertEqual(resp.status_code, 401)


class TestMetricsEndpoint(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_returns_200_or_404(self):
        code = self.client.get("/metrics").status_code
        self.assertIn(code, (200, 404))   # 404 if model not yet trained

    def test_404_has_warning_key(self):
        with patch("app.MODEL_PATH", "nonexistent/path.pth"):
            data = json.loads(self.client.get("/metrics").data)
            # either real metrics or a warning
            self.assertTrue("warning" in data or "test_iou" in data)


# ──────────────────────────────────────────────────────────────
# Metric function tests
# ──────────────────────────────────────────────────────────────

class TestIoUMetric(unittest.TestCase):

    def test_perfect_overlap_is_one(self):
        mask = np.ones((256, 256), dtype=np.uint8) * 255
        self.assertAlmostEqual(iou_score_np(mask, mask), 1.0, places=5)

    def test_no_overlap_is_zero(self):
        pred = np.zeros((256, 256), dtype=np.uint8)
        gt   = np.ones( (256, 256), dtype=np.uint8) * 255
        self.assertAlmostEqual(iou_score_np(pred, gt), 0.0, places=5)

    def test_value_in_range(self):
        rng  = np.random.default_rng(0)
        pred = rng.integers(0, 2, (256, 256), dtype=np.uint8) * 255
        gt   = rng.integers(0, 2, (256, 256), dtype=np.uint8) * 255
        iou  = iou_score_np(pred, gt)
        self.assertGreaterEqual(iou, 0.0)
        self.assertLessEqual(   iou, 1.0)

    def test_partial_overlap(self):
        pred = np.zeros((4, 4), dtype=np.uint8)
        gt   = np.zeros((4, 4), dtype=np.uint8)
        pred[0:2, 0:2] = 255
        gt[1:3, 1:3]   = 255
        # intersection = 1 pixel, union = 7 pixels → IoU = 1/7
        self.assertAlmostEqual(iou_score_np(pred, gt), 1/7, places=5)


class TestDiceMetric(unittest.TestCase):

    def test_perfect_overlap_is_one(self):
        mask = np.ones((128, 128), dtype=np.uint8) * 255
        self.assertAlmostEqual(dice_score_np(mask, mask), 1.0, places=5)

    def test_no_overlap_is_zero(self):
        pred = np.zeros((128, 128), dtype=np.uint8)
        gt   = np.ones( (128, 128), dtype=np.uint8) * 255
        self.assertAlmostEqual(dice_score_np(pred, gt), 0.0, places=5)

    def test_partial_overlap(self):
        pred = np.zeros((4, 4), dtype=np.uint8)
        gt   = np.zeros((4, 4), dtype=np.uint8)
        pred[0:2, 0:2] = 255
        gt[1:3, 1:3]   = 255
        # 2*1 / (4+4) = 0.25
        self.assertAlmostEqual(dice_score_np(pred, gt), 0.25, places=5)


class TestComputeMetrics(unittest.TestCase):

    def test_returns_two_floats(self):
        mask   = np.ones((64, 64), dtype=np.uint8) * 255
        result = compute_metrics_from_masks(mask, mask)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        for v in result:
            self.assertIsInstance(v, float)

    def test_perfect_masks_both_one(self):
        mask   = np.ones((64, 64), dtype=np.uint8) * 255
        iou, dice = compute_metrics_from_masks(mask, mask)
        self.assertAlmostEqual(iou,  1.0, places=5)
        self.assertAlmostEqual(dice, 1.0, places=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
