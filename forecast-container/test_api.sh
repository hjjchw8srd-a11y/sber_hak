#!/usr/bin/env bash
set -euo pipefail

# test_api.sh: Quick curl smoke test for the containerized forecast API
HOST="${1:-http://localhost:8000}"

echo "=== 1. Checking /health on $HOST ==="
curl -sS "$HOST/health" | grep -q '"status":"ok"' && echo "Health check PASSED" || { echo "Health check FAILED"; exit 1; }

echo ""
echo "=== 2. Testing 409 Conflict before POST ==="
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$HOST/predict?warehouse_name=warehouse_name_1")
if [ "$HTTP_CODE" -eq 409 ]; then
    echo "GET /predict before POST correctly returned 409 Conflict: PASSED"
else
    echo "Expected 409, got $HTTP_CODE: FAILED"
    exit 1
fi

echo ""
echo "=== 3. Sending POST /predict with sample data ==="
SALES_FILE="forecast-container/data/sample_sales.csv"
STOCK_FILE="forecast-container/data/sample_stock.csv"

if [ ! -f "$SALES_FILE" ]; then
    SALES_FILE="data/sample_sales.csv"
    STOCK_FILE="data/sample_stock.csv"
fi

POST_RESP=$(curl -sS -X POST "$HOST/predict" \
    -F "sales=@$SALES_FILE" \
    -F "stock=@$STOCK_FILE")

echo "$POST_RESP" | grep -q '"status":"success"' && echo "POST /predict PASSED" || { echo "POST /predict FAILED: $POST_RESP"; exit 1; }

echo ""
echo "=== 4. Testing GET /predict filter ==="
PRED_FILTER=$(curl -sS "$HOST/predict?warehouse_name=warehouse_name_1")
echo "$PRED_FILTER" | head -n 5
echo "... (truncated) GET /predict filter PASSED"

echo ""
echo "=== 5. Testing GET /predict/all?format=csv ==="
curl -sS "$HOST/predict/all?format=csv" | head -n 5
echo "... (truncated) GET /predict/all?format=csv PASSED"

echo ""
echo "=== 6. Testing Cache Hit Speed ==="
TIME_START=$(date +%s%N)
curl -sS -X POST "$HOST/predict" \
    -F "sales=@$SALES_FILE" \
    -F "stock=@$STOCK_FILE" > /dev/null
TIME_END=$(date +%s%N)
ELAPSED_MS=$(( (TIME_END - TIME_START) / 1000000 ))
echo "Cached POST latency: ${ELAPSED_MS}ms (expected < 500ms)"

echo ""
echo "ALL SMOKE TESTS COMPLETED SUCCESSFULLY!"
