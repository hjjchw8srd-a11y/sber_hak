#!/usr/bin/env python3
"""
smoke_test.py: Cross-platform smoke test against running Uvicorn / Docker service.
Requires zero external dependencies (uses standard library urllib).
"""

from __future__ import annotations

import csv
import io
import json
import mimetypes
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
HOST = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"


def make_multipart(fields: dict[str, Path]) -> tuple[bytes, str]:
    boundary = f"----WebKitFormBoundary{int(time.time()*1000)}"
    body = bytearray()
    for field_name, file_path in fields.items():
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(
            f'Content-Disposition: form-data; name="{field_name}"; filename="{file_path.name}"\r\n'.encode("utf-8")
        )
        body.extend(b"Content-Type: text/csv\r\n\r\n")
        body.extend(file_path.read_bytes())
        body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode("utf-8"))
    content_type = f"multipart/form-data; boundary={boundary}"
    return bytes(body), content_type


def main():
    print("=" * 72)
    print(f"RUNNING PYTHON API SMOKE TEST AGAINST {HOST}")
    print("=" * 72)

    # 1. GET /health
    print("\n=== 1. Checking GET /health ===")
    req = urllib.request.Request(f"{HOST}/health")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            assert resp.status == 200, f"Expected HTTP 200, got {resp.status}"
            data = json.loads(resp.read().decode("utf-8"))
            assert data.get("status") == "healthy", f"Expected status healthy, got {data}"
            print(f"Healthcheck PASSED: status={data.get('status')}, bundle={data.get('bundle_digest', '')[:12]}...")
    except Exception as e:
        print(f"ERROR on GET /health: {e}")
        sys.exit(1)

    # 2. GET /predict before POST (Expect 409 Conflict)
    print("\n=== 2. Checking GET /predict before any POST (Expect 409) ===")
    req = urllib.request.Request(f"{HOST}/predict?warehouse=warehouse_name_1")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            print(f"ERROR: Expected 409 Conflict, got HTTP {resp.status}")
            sys.exit(1)
    except urllib.error.HTTPError as e:
        if e.code == 409:
            print("Initial state check PASSED (HTTP 409 Conflict prior to POST)")
        else:
            print(f"ERROR: Expected 409, got HTTP {e.code}: {e.read().decode('utf-8')}")
            sys.exit(1)
    except Exception as e:
        print(f"ERROR on initial GET /predict: {e}")
        sys.exit(1)

    # 3. POST /predict with demo CSV files
    print("\n=== 3. Sending POST /predict with demo data ===")
    sales_path = BASE_DIR / "data" / "demo_sales.csv"
    stock_path = BASE_DIR / "data" / "demo_stock.csv"
    assert sales_path.exists(), f"Missing {sales_path}"
    assert stock_path.exists(), f"Missing {stock_path}"

    body_bytes, content_type = make_multipart({"sales": sales_path, "stock": stock_path})
    req = urllib.request.Request(
        f"{HOST}/predict",
        data=body_bytes,
        headers={"Content-Type": content_type},
        method="POST",
    )

    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            assert resp.status == 200, f"Expected 200, got {resp.status}"
            csv_text = resp.read().decode("utf-8")
            elapsed = time.time() - t0
            lines = csv_text.strip().splitlines()
            assert len(lines) == 79102, f"Expected 79102 lines, got {len(lines)}"
            assert lines[0] == "week,warehouse_name,article,quantity", f"Bad header: {lines[0]}"
            print(f"POST /predict PASSED: HTTP 200 in {elapsed:.2f}s, 79101 predictions returned")
    except Exception as e:
        print(f"ERROR on POST /predict: {e}")
        sys.exit(1)

    # 4. GET /predict?warehouse=warehouse_name_1
    print("\n=== 4. Checking GET /predict?warehouse=warehouse_name_1 ===")
    req = urllib.request.Request(f"{HOST}/predict?warehouse=warehouse_name_1")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            assert resp.status == 200, f"Expected 200, got {resp.status}"
            csv_text = resp.read().decode("utf-8")
            reader = csv.DictReader(io.StringIO(csv_text))
            rows = list(reader)
            assert len(rows) == 7191, f"Expected 7191 rows, got {len(rows)}"
            for r in rows:
                assert int(r["quantity"]) >= 0, f"Negative quantity found: {r}"
                assert r["warehouse_name"] == "warehouse_name_1", f"Wrong warehouse: {r}"
            print(f"Warehouse filter PASSED: HTTP 200, exactly 7191 records, all non-negative integers")
    except Exception as e:
        print(f"ERROR on warehouse filter: {e}")
        sys.exit(1)

    # 5. GET /predict?warehouse=warehouse_name_1&article=char_article_1
    print("\n=== 5. Checking GET /predict with warehouse and article ===")
    req = urllib.request.Request(f"{HOST}/predict?warehouse=warehouse_name_1&article=char_article_1")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            assert resp.status == 200, f"Expected 200, got {resp.status}"
            reader = csv.DictReader(io.StringIO(resp.read().decode("utf-8")))
            rows = list(reader)
            assert len(rows) == 9, f"Expected 9 horizon rows, got {len(rows)}"
            weeks = [r["week"] for r in rows]
            assert len(set(weeks)) == 9, f"Duplicate weeks found: {weeks}"
            print(f"Pair filter PASSED: HTTP 200, exactly 9 weekly forecasts from {weeks[0]} to {weeks[-1]}")
    except Exception as e:
        print(f"ERROR on pair filter: {e}")
        sys.exit(1)

    # 6. GET /predict/all (CSV and JSON)
    print("\n=== 6. Checking GET /predict/all ===")
    req_csv = urllib.request.Request(f"{HOST}/predict/all?format=csv")
    with urllib.request.urlopen(req_csv, timeout=10) as resp:
        assert resp.status == 200
        assert len(resp.read().decode("utf-8").strip().splitlines()) == 79102

    req_json = urllib.request.Request(f"{HOST}/predict/all?format=json")
    with urllib.request.urlopen(req_json, timeout=10) as resp:
        assert resp.status == 200
        items = json.loads(resp.read().decode("utf-8"))
        assert len(items) == 79101, f"Expected 79101 items, got {len(items)}"
    print("GET /predict/all PASSED: Both CSV (79102 lines) and JSON (79101 items) verified")

    # 7. Cache hit response speed
    print("\n=== 7. Checking Cache Hit Speed ===")
    req_cached = urllib.request.Request(
        f"{HOST}/predict",
        data=body_bytes,
        headers={"Content-Type": content_type},
        method="POST",
    )
    t_start = time.perf_counter()
    with urllib.request.urlopen(req_cached, timeout=10) as resp:
        assert resp.status == 200
    elapsed_ms = (time.perf_counter() - t_start) * 1000
    print(f"Cached POST latency: {elapsed_ms:.2f}ms (expected < 500ms)")
    assert elapsed_ms < 500, f"Cache latency too high: {elapsed_ms:.2f}ms"

    print("\n" + "=" * 72)
    print("ALL SMOKE TESTS COMPLETED SUCCESSFULLY WITH EXIT CODE 0!")
    print("=" * 72)


if __name__ == "__main__":
    main()
