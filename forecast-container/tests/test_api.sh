#!/usr/bin/env bash
# test_api.sh — Smoke test script for forecast-container HTTP API

set -euo pipefail

BASE_URL="${1:-http://localhost:8000}"
DATA_DIR="${2:-./data}"

echo "=========================================================="
echo "SMOKE TESTING FORECAST API AT: $BASE_URL"
echo "=========================================================="

echo "1. Checking GET /health..."
curl -sSf "$BASE_URL/health"
echo -e "\n[OK] Healthcheck passed!"

echo -e "\n2. Testing 409 error on GET /predict before any POST..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/predict?article=char_article_1" || true)
if [ "$HTTP_CODE" -eq 409 ]; then
  echo "[OK] Correctly returned 409 Conflict prior to first POST"
else
  echo "[WARNING] Expected 409, got $HTTP_CODE"
fi

echo -e "\n3. Testing POST /predict with sample data..."
if [ -f "$DATA_DIR/sample_sales.csv" ] && [ -f "$DATA_DIR/sample_stock.csv" ]; then
  # Note: sample data has 100 rows, should trigger 422 history depth validation
  HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE_URL/predict" \
    -F "sales=@$DATA_DIR/sample_sales.csv" \
    -F "stock=@$DATA_DIR/sample_stock.csv" || true)
  if [ "$HTTP_CODE" -eq 422 ]; then
    echo "[OK] Successfully triggered 422 VALIDATION_ERROR on < 180 days sample"
  else
    echo "POST returned HTTP status: $HTTP_CODE"
  fi
fi

echo -e "\n4. Testing bad query params on GET /predict (no params)..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/predict" || true)
if [ "$HTTP_CODE" -eq 400 ]; then
  echo "[OK] Correctly returned 400 Bad Request when no params provided"
else
  echo "[WARNING] Expected 400, got $HTTP_CODE"
fi

echo -e "\n5. Testing bad format on GET /predict/all?format=xml..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/predict/all?format=xml" || true)
if [ "$HTTP_CODE" -eq 400 ]; then
  echo "[OK] Correctly returned 400 Bad Request for unsupported format"
else
  echo "[WARNING] Expected 400, got $HTTP_CODE"
fi

echo -e "\n=========================================================="
echo "ALL SMOKE CHECKS COMPLETED!"
echo "=========================================================="
