#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────
# CEG4195 Lab 2 – API Smoke Tests
# Usage: chmod +x test_api.sh && ./test_api.sh [PORT]
# ────────────────────────────────────────────────────────────

PORT=${1:-5000}
BASE="http://localhost:${PORT}"
PASS=0; FAIL=0

check() {
    local desc="$1" code="$2" expected="$3"
    if [ "$code" -eq "$expected" ]; then
        echo "  ✅  PASS – $desc  (HTTP $code)"
        ((PASS++))
    else
        echo "  ❌  FAIL – $desc  (expected HTTP $expected, got $code)"
        ((FAIL++))
    fi
}

# Generate a tiny 64×64 PNG encoded as base64
B64=$(python3 - <<'EOF'
import base64, io
from PIL import Image
import numpy as np
arr = np.random.randint(60, 200, (64, 64, 3), dtype=np.uint8)
buf = io.BytesIO()
Image.fromarray(arr).save(buf, format="PNG")
print(base64.b64encode(buf.getvalue()).decode())
EOF
)

echo ""
echo "════════════════════════════════════════════════════════"
echo "  CEG4195 Lab 2 – House Segmentation API – Test Suite"
echo "  Target : $BASE"
echo "════════════════════════════════════════════════════════"
echo ""

echo "── 1. Home endpoint (GET /) ─────────────────────────────"
CODE=$(curl -s -o /dev/null -w "%{http_code}" "$BASE/")
check "GET /" "$CODE" 200
echo ""

echo "── 2. Health check (GET /health) ───────────────────────"
RESP=$(curl -s "$BASE/health")
CODE=$(curl -s -o /dev/null -w "%{http_code}" "$BASE/health")
check "GET /health" "$CODE" 200
echo "     $RESP"
echo ""

echo "── 3. Predict – valid image ─────────────────────────────"
RESP=$(curl -s -w "\nHTTP_CODE:%{http_code}" \
    -X POST "$BASE/predict" \
    -H "Content-Type: application/json" \
    -d "{\"image\": \"$B64\"}")
CODE=$(echo "$RESP" | grep HTTP_CODE | cut -d: -f2)
check "POST /predict (valid image)" "$CODE" 200
echo ""

echo "── 4. Predict – missing 'image' field → 400 ────────────"
CODE=$(curl -s -o /dev/null -w "%{http_code}" \
    -X POST "$BASE/predict" \
    -H "Content-Type: application/json" \
    -d '{}')
check "POST /predict (missing field)" "$CODE" 400
echo ""

echo "── 5. Predict – custom threshold ───────────────────────"
CODE=$(curl -s -o /dev/null -w "%{http_code}" \
    -X POST "$BASE/predict" \
    -H "Content-Type: application/json" \
    -d "{\"image\": \"$B64\", \"threshold\": 0.3}")
check "POST /predict (threshold=0.3)" "$CODE" 200
echo ""

echo "── 6. Metrics endpoint (GET /metrics) ──────────────────"
CODE=$(curl -s -o /dev/null -w "%{http_code}" "$BASE/metrics")
# 200 if model trained, 404 if not — both are valid responses
if [ "$CODE" -eq 200 ] || [ "$CODE" -eq 404 ]; then
    echo "  ✅  PASS – GET /metrics  (HTTP $CODE)"
    ((PASS++))
else
    echo "  ❌  FAIL – GET /metrics  (unexpected HTTP $CODE)"
    ((FAIL++))
fi
echo ""

echo "════════════════════════════════════════════════════════"
echo "  Results : $PASS passed  |  $FAIL failed"
echo "════════════════════════════════════════════════════════"
echo ""
[ "$FAIL" -eq 0 ] && exit 0 || exit 1
