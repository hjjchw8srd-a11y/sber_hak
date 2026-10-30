#!/usr/bin/env bash
set -euo pipefail

# test_api.sh: Smoke test for the demand forecast API conforming to Hackathon Section 10
HOST="${1:-http://127.0.0.1:8000}"

TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

TMP_HEALTH="$TMP_DIR/health.json"
TMP_POST="$TMP_DIR/post_pred.csv"
TMP_WH="$TMP_DIR/wh_pred.csv"
TMP_PAIR="$TMP_DIR/pair_pred.csv"
TMP_ALL_CSV="$TMP_DIR/all_pred.csv"
TMP_ALL_JSON="$TMP_DIR/all_pred.json"

echo "========================================================================"
echo "RUNNING API SMOKE TEST AGAINST $HOST"
echo "========================================================================"

# --- 1. Healthcheck ---
echo ""
echo "=== 1. Checking GET /health ==="
HTTP_CODE=$(curl -s -o "$TMP_HEALTH" -w "%{http_code}" "$HOST/health")
if [ "$HTTP_CODE" -ne 200 ]; then
    echo "ERROR: GET /health returned HTTP $HTTP_CODE, expected 200"
    exit 1
fi

if ! grep -q '"status":"healthy"' "$TMP_HEALTH"; then
    echo "ERROR: GET /health response does not contain '\"status\":\"healthy\"':"
    cat "$TMP_HEALTH"
    exit 1
fi
echo "Healthcheck PASSED (HTTP 200, status=healthy)"

# --- 2. GET /predict before POST (Expect 409 Conflict) ---
echo ""
echo "=== 2. Checking GET /predict before any POST (Expect 409 Conflict) ==="
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$HOST/predict?warehouse=warehouse_name_1")
if [ "$HTTP_CODE" -ne 409 ]; then
    echo "ERROR: GET /predict?warehouse=warehouse_name_1 before POST returned HTTP $HTTP_CODE, expected 409"
    exit 1
fi
echo "Initial state check PASSED (HTTP 409 Conflict prior to POST)"

# --- 3. POST /predict with demo data ---
echo ""
echo "=== 3. Sending POST /predict with demo CSV files ==="
SALES_FILE="forecast-container/data/demo_sales.csv"
STOCK_FILE="forecast-container/data/demo_stock.csv"

if [ ! -f "$SALES_FILE" ]; then
    SALES_FILE="data/demo_sales.csv"
    STOCK_FILE="data/demo_stock.csv"
fi

if [ ! -f "$SALES_FILE" ] || [ ! -f "$STOCK_FILE" ]; then
    echo "ERROR: Demo files not found at $SALES_FILE or $STOCK_FILE"
    exit 1
fi

HTTP_CODE=$(curl -s -o "$TMP_POST" -w "%{http_code}" -X POST "$HOST/predict" \
    -F "sales=@$SALES_FILE" \
    -F "stock=@$STOCK_FILE")

if [ "$HTTP_CODE" -ne 200 ]; then
    echo "ERROR: POST /predict returned HTTP $HTTP_CODE, expected 200"
    cat "$TMP_POST"
    exit 1
fi

# Validate CSV Header
FIRST_LINE=$(head -n 1 "$TMP_POST" | tr -d '\r')
EXPECTED_HEADER="week,warehouse_name,article,quantity"
if [ "$FIRST_LINE" != "$EXPECTED_HEADER" ]; then
    echo "ERROR: Invalid CSV header. Expected '$EXPECTED_HEADER', got '$FIRST_LINE'"
    exit 1
fi

# Validate total row count (1 header + 79101 records = 79102 lines)
TOTAL_LINES=$(wc -l < "$TMP_POST" | tr -d ' ')
if [ "$TOTAL_LINES" -ne 79102 ]; then
    echo "ERROR: Expected 79102 lines in POST response, got $TOTAL_LINES"
    exit 1
fi
echo "POST /predict PASSED: HTTP 200, header '$EXPECTED_HEADER', 79101 data rows"

# --- 4. GET /predict?warehouse=warehouse_name_1 ---
echo ""
echo "=== 4. Checking GET /predict?warehouse=warehouse_name_1 ==="
HTTP_CODE=$(curl -s -o "$TMP_WH" -w "%{http_code}" "$HOST/predict?warehouse=warehouse_name_1")
if [ "$HTTP_CODE" -ne 200 ]; then
    echo "ERROR: GET /predict?warehouse=warehouse_name_1 returned HTTP $HTTP_CODE, expected 200"
    exit 1
fi

# Expected: 1 header + (9 weeks * 799 articles = 7191 rows) = 7192 lines
WH_LINES=$(wc -l < "$TMP_WH" | tr -d ' ')
if [ "$WH_LINES" -ne 7192 ]; then
    echo "ERROR: Expected 7192 lines for warehouse filter, got $WH_LINES"
    exit 1
fi

# Verify non-negativity and integers in quantity column (column 4)
INVALID_QTY=$(awk -F',' 'NR>1 { if ($4 < 0 || $4 ~ /\./) print $0 }' "$TMP_WH" | head -n 1)
if [ -n "$INVALID_QTY" ]; then
    echo "ERROR: Found invalid quantity in warehouse response: $INVALID_QTY"
    exit 1
fi
echo "GET /predict?warehouse=warehouse_name_1 PASSED: HTTP 200, 7191 data rows, all quantity >= 0"

# --- 5. GET /predict?warehouse=warehouse_name_1&article=char_article_1 ---
echo ""
echo "=== 5. Checking GET /predict?warehouse=warehouse_name_1&article=char_article_1 ==="
HTTP_CODE=$(curl -s -o "$TMP_PAIR" -w "%{http_code}" "$HOST/predict?warehouse=warehouse_name_1&article=char_article_1")
if [ "$HTTP_CODE" -ne 200 ]; then
    echo "ERROR: GET /predict pair filter returned HTTP $HTTP_CODE, expected 200"
    exit 1
fi

# Expected: 1 header + 9 weeks = 10 lines
PAIR_LINES=$(wc -l < "$TMP_PAIR" | tr -d ' ')
if [ "$PAIR_LINES" -ne 10 ]; then
    echo "ERROR: Expected 10 lines for pair filter, got $PAIR_LINES"
    exit 1
fi
echo "GET /predict pair filter PASSED: HTTP 200, exactly 9 horizon weeks"

# --- 6. GET /predict/all ---
echo ""
echo "=== 6. Checking GET /predict/all (CSV and JSON) ==="
HTTP_CODE=$(curl -s -o "$TMP_ALL_CSV" -w "%{http_code}" "$HOST/predict/all?format=csv")
if [ "$HTTP_CODE" -ne 200 ]; then
    echo "ERROR: GET /predict/all?format=csv returned HTTP $HTTP_CODE, expected 200"
    exit 1
fi
ALL_LINES=$(wc -l < "$TMP_ALL_CSV" | tr -d ' ')
if [ "$ALL_LINES" -ne 79102 ]; then
    echo "ERROR: Expected 79102 lines in GET /predict/all CSV, got $ALL_LINES"
    exit 1
fi

HTTP_CODE=$(curl -s -o "$TMP_ALL_JSON" -w "%{http_code}" "$HOST/predict/all?format=json")
if [ "$HTTP_CODE" -ne 200 ]; then
    echo "ERROR: GET /predict/all?format=json returned HTTP $HTTP_CODE, expected 200"
    exit 1
fi
echo "GET /predict/all PASSED: both CSV and JSON formats return 79101 predictions"

# --- 7. Cache hit response latency check ---
echo ""
echo "=== 7. Checking Cache Hit Speed ==="
TIME_START=$(date +%s%N 2>/dev/null || python -c "import time; print(int(time.time()*1e9))")
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$HOST/predict" \
    -F "sales=@$SALES_FILE" \
    -F "stock=@$STOCK_FILE")
TIME_END=$(date +%s%N 2>/dev/null || python -c "import time; print(int(time.time()*1e9))")

if [ "$HTTP_CODE" -ne 200 ]; then
    echo "ERROR: Cached POST returned HTTP $HTTP_CODE"
    exit 1
fi

ELAPSED_MS=$(( (TIME_END - TIME_START) / 1000000 ))
echo "Cached POST latency: ${ELAPSED_MS}ms (limit: < 500ms)"
if [ "$ELAPSED_MS" -gt 500 ]; then
    echo "WARNING: Cache hit took longer than 500ms (${ELAPSED_MS}ms)"
fi

echo ""
echo "========================================================================"
echo "ALL SMOKE TESTS COMPLETED SUCCESSFULLY WITH EXIT CODE 0!"
echo "========================================================================"
