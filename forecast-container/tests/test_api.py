"""
test_api.py

Automated test suite for the FastAPI service conforming to Hackathon Section 10.
Executes all validation checks, endpoint contracts, error conditions,
and deterministic behavior.
"""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.main import app, startup_event
from app.predictor import predict_from_history

client = TestClient(app)


def init_server():
    startup_event()


def create_mock_sales_csv(
    days: int = 180,
    end_date: str = "2026-07-26",
    negative_quantity: bool = False,
    missing_columns: bool = False,
    empty: bool = False,
) -> io.BytesIO:
    if empty:
        return io.BytesIO(b"")

    end_dt = pd.Timestamp(end_date)
    start_dt = end_dt - pd.Timedelta(days=days - 1)
    dates = pd.date_range(start_dt, end_dt, freq="D")

    rows = []
    for d in dates:
        rows.append({
            "date": d.strftime("%Y-%m-%d"),
            "composite_division": "div_1",
            "cities_name": "city_1",
            "warehouse_name": "warehouse_name_1",
            "category_lvl_1": "category_lvl_1_0",
            "category_lvl_2": "category_lvl_2_0",
            "category_lvl_3": "category_lvl_3_0",
            "article": "char_article_1",
            "characteristic": "char_1",
            "quantity": -5.0 if negative_quantity and d == dates[-1] else 10.0,
            "sales_sum": 2.5,
            "discount_sum": 1.2 if d.day % 2 == 0 else np.nan,
        })

    df = pd.DataFrame(rows)
    if missing_columns:
        df = df.drop(columns=["quantity"])

    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf


def create_mock_stock_csv(
    days: int = 180,
    end_date: str = "2026-07-26",
    negative_stock: bool = True,
    missing_columns: bool = False,
    empty: bool = False,
) -> io.BytesIO:
    if empty:
        return io.BytesIO(b"")

    end_dt = pd.Timestamp(end_date)
    start_dt = end_dt - pd.Timedelta(days=days - 1)
    dates = pd.date_range(start_dt, end_dt, freq="D")

    rows = []
    for d in dates:
        rows.append({
            "date": d.strftime("%Y-%m-%d"),
            "warehouse_name": "warehouse_name_1",
            "char_article": "char_article_1",
            "stock": -15.0 if negative_stock and d == dates[-1] else 50.0,
        })

    df = pd.DataFrame(rows)
    if missing_columns:
        df = df.drop(columns=["stock"])

    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf


def test_01_healthcheck():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "candidate_lgbm52" in data["model_bundle"]


def test_02_get_predict_before_post_returns_409():
    resp = client.get("/predict?article=char_article_1")
    assert resp.status_code == 409
    assert resp.json()["code"] == "NO_PREDICTION_AVAILABLE"


def test_03_get_predict_all_before_post_returns_409():
    resp = client.get("/predict/all?format=csv")
    assert resp.status_code == 409
    assert resp.json()["code"] == "NO_PREDICTION_AVAILABLE"


def test_04_error_missing_both_query_params():
    resp = client.get("/predict")
    assert resp.status_code == 400
    assert resp.json()["code"] == "BAD_REQUEST"


def test_05_error_invalid_format_predict_all():
    resp = client.get("/predict/all?format=xml")
    assert resp.status_code == 400
    assert resp.json()["code"] == "BAD_REQUEST"


def test_06_error_empty_files():
    empty_sales = io.BytesIO(b"")
    empty_stock = io.BytesIO(b"")
    resp = client.post(
        "/predict",
        files={"sales": ("sales.csv", empty_sales, "text/csv"), "stock": ("stock.csv", empty_stock, "text/csv")},
    )
    assert resp.status_code == 400
    assert resp.json()["code"] == "BAD_REQUEST"


def test_07_error_missing_column():
    bad_sales = create_mock_sales_csv(missing_columns=True)
    good_stock = create_mock_stock_csv()
    resp = client.post(
        "/predict",
        files={"sales": ("sales.csv", bad_sales, "text/csv"), "stock": ("stock.csv", good_stock, "text/csv")},
    )
    assert resp.status_code == 400
    assert resp.json()["code"] == "BAD_REQUEST"


def test_08_error_negative_quantity():
    neg_sales = create_mock_sales_csv(negative_quantity=True)
    good_stock = create_mock_stock_csv()
    resp = client.post(
        "/predict",
        files={"sales": ("sales.csv", neg_sales, "text/csv"), "stock": ("stock.csv", good_stock, "text/csv")},
    )
    assert resp.status_code == 422
    assert resp.json()["code"] == "VALIDATION_ERROR"
    assert "negative" in resp.json()["detail"].lower()


def test_09_error_history_under_180_days():
    short_sales = create_mock_sales_csv(days=179)
    good_stock = create_mock_stock_csv(days=180)
    resp = client.post(
        "/predict",
        files={"sales": ("sales.csv", short_sales, "text/csv"), "stock": ("stock.csv", good_stock, "text/csv")},
    )
    assert resp.status_code == 422
    assert resp.json()["code"] == "VALIDATION_ERROR"
    assert "179" in resp.json()["detail"]


def test_10_dynamic_date_rules():
    """
    Test real predict_from_history across 3 different cutoff dates:
    1. Cutoff Sunday 2026-07-26 -> forecast starts Monday 2026-07-27
    2. Cutoff Friday 2026-07-31 -> forecast starts Monday 2026-08-03
    3. Cutoff Monday 2026-08-03 -> forecast starts Monday 2026-08-10 (historical Monday -> +7 days)
    """
    sales_path = PROJECT_ROOT.parent / "sales_dataset_hakaton_train.csv"
    stock_path = PROJECT_ROOT.parent / "stock_dataset_hakaton_train.csv"
    s_full = pd.read_csv(sales_path, parse_dates=["date"])
    st_full = pd.read_csv(stock_path, parse_dates=["date"])

    # 1. Sunday Cutoff: 2026-07-26
    cutoff_sun = pd.Timestamp("2026-07-26")
    start_sun = cutoff_sun - pd.Timedelta(days=179)
    s_sun = s_full[(s_full["date"] >= start_sun) & (s_full["date"] <= cutoff_sun)].copy()
    st_sun = st_full[(st_full["date"] >= start_sun) & (st_full["date"] <= cutoff_sun)].copy()

    print("\n  [Dynamic Dates 1/3] Sunday 2026-07-26 cutoff -> calling predict_from_history...")
    res_sun = predict_from_history(s_sun, st_sun)
    assert len(res_sun) == 79101
    weeks_sun = sorted(res_sun["week"].unique())
    assert len(weeks_sun) == 9
    assert weeks_sun[0] == "2026-07-27"
    assert (pd.to_datetime(res_sun["week"]).dt.weekday == 0).all()

    # 2. Friday Cutoff: 2026-07-31 (shift dates by +5 days)
    print("  [Dynamic Dates 2/3] Friday 2026-07-31 cutoff -> calling predict_from_history...")
    s_fri = s_sun.copy()
    s_fri["date"] = s_fri["date"] + pd.Timedelta(days=5)
    st_fri = st_sun.copy()
    st_fri["date"] = st_fri["date"] + pd.Timedelta(days=5)

    res_fri = predict_from_history(s_fri, st_fri)
    assert len(res_fri) == 79101
    weeks_fri = sorted(res_fri["week"].unique())
    assert len(weeks_fri) == 9
    assert weeks_fri[0] == "2026-08-03"
    assert (pd.to_datetime(res_fri["week"]).dt.weekday == 0).all()

    # 3. Monday Cutoff: 2026-08-03 (shift dates by +8 days)
    print("  [Dynamic Dates 3/3] Monday 2026-08-03 cutoff -> calling predict_from_history...")
    s_mon = s_sun.copy()
    s_mon["date"] = s_mon["date"] + pd.Timedelta(days=8)
    st_mon = st_sun.copy()
    st_mon["date"] = st_mon["date"] + pd.Timedelta(days=8)

    res_mon = predict_from_history(s_mon, st_mon)
    assert len(res_mon) == 79101
    weeks_mon = sorted(res_mon["week"].unique())
    assert len(weeks_mon) == 9
    assert weeks_mon[0] == "2026-08-10"
    assert (pd.to_datetime(res_mon["week"]).dt.weekday == 0).all()


def test_11_full_post_and_get_contract():
    # Use real 180-day slice from raw datasets
    sales_path = PROJECT_ROOT.parent / "sales_dataset_hakaton_train.csv"
    stock_path = PROJECT_ROOT.parent / "stock_dataset_hakaton_train.csv"
    assert sales_path.exists() and stock_path.exists()

    cutoff = pd.Timestamp("2026-07-26")
    start_date = cutoff - pd.Timedelta(days=179)

    print("\nReading 180-day slice for full POST test...")
    # Load 180-day slice
    s_full = pd.read_csv(sales_path, parse_dates=["date"])
    s_180 = s_full[(s_full["date"] >= start_date) & (s_full["date"] <= cutoff)].copy()

    st_full = pd.read_csv(stock_path, parse_dates=["date"])
    st_180 = st_full[(st_full["date"] >= start_date) & (st_full["date"] <= cutoff)].copy()

    s_buf = io.BytesIO()
    s_180.to_csv(s_buf, index=False)
    s_buf.seek(0)

    st_buf = io.BytesIO()
    st_180.to_csv(st_buf, index=False)
    st_buf.seek(0)

    print("Sending POST /predict...")
    resp_post = client.post(
        "/predict",
        files={"sales": ("sales.csv", s_buf, "text/csv"), "stock": ("stock.csv", st_buf, "text/csv")},
    )
    assert resp_post.status_code == 200
    assert resp_post.headers["content-type"].startswith("text/csv")

    post_df = pd.read_csv(io.StringIO(resp_post.text))
    assert len(post_df) == 79101
    assert set(post_df.columns) == {"week", "warehouse_name", "article", "quantity"}
    assert (post_df["quantity"] >= 0).all()
    assert (post_df["quantity"] % 1 == 0).all()
    print(f"POST returned {len(post_df)} rows, total qty: {post_df['quantity'].sum()}")

    # GET /predict with article filter
    resp_art = client.get("/predict?article=char_article_1")
    assert resp_art.status_code == 200
    df_art = pd.read_csv(io.StringIO(resp_art.text))
    assert len(df_art) == 99  # 9 weeks * 11 warehouses
    assert (df_art["article"] == "char_article_1").all()

    # GET /predict with warehouse filter
    resp_wh = client.get("/predict?warehouse=warehouse_name_1")
    assert resp_wh.status_code == 200
    df_wh = pd.read_csv(io.StringIO(resp_wh.text))
    assert len(df_wh) == 7191  # 9 weeks * 799 articles
    assert (df_wh["warehouse_name"] == "warehouse_name_1").all()

    # GET /predict with both filters (intersection)
    resp_pair = client.get("/predict?article=char_article_1&warehouse=warehouse_name_1")
    assert resp_pair.status_code == 200
    df_pair = pd.read_csv(io.StringIO(resp_pair.text))
    assert len(df_pair) == 9  # exactly 9 weeks
    assert (df_pair["article"] == "char_article_1").all()
    assert (df_pair["warehouse_name"] == "warehouse_name_1").all()

    # GET /predict/all?format=csv
    resp_all_csv = client.get("/predict/all?format=csv")
    assert resp_all_csv.status_code == 200
    df_all_csv = pd.read_csv(io.StringIO(resp_all_csv.text))
    assert len(df_all_csv) == 79101
    assert (df_all_csv["quantity"] == post_df["quantity"]).all()

    # GET /predict/all?format=json
    resp_all_json = client.get("/predict/all?format=json")
    assert resp_all_json.status_code == 200
    data_json = resp_all_json.json()
    assert len(data_json) == 79101
    assert data_json[0]["article"] == post_df.iloc[0]["article"]
    assert data_json[0]["quantity"] == int(post_df.iloc[0]["quantity"])

    print("POST and GET integration checks PASSED!")


def test_12_cache_behavior():
    # Build clean 180-day inputs to measure HTTP cache response time
    dates = pd.date_range("2026-01-28", "2026-07-26", freq="D")
    s_light = pd.DataFrame([{
        "date": d.strftime("%Y-%m-%d"),
        "composite_division": "div_0",
        "cities_name": "city_0",
        "warehouse_name": "warehouse_name_1",
        "category_lvl_1": "category_lvl_1_0",
        "category_lvl_2": "category_lvl_2_0",
        "category_lvl_3": "category_lvl_3_0",
        "article": "char_article_1",
        "characteristic": "char_0",
        "quantity": 5.0,
        "sales_sum": 10.0,
        "discount_sum": 1.0,
    } for d in dates])
    st_light = pd.DataFrame([{
        "date": d.strftime("%Y-%m-%d"),
        "warehouse_name": "warehouse_name_1",
        "article": "char_article_1",
        "stock": 50.0,
    } for d in dates])

    s_bytes = s_light.to_csv(index=False).encode("utf-8")
    st_bytes = st_light.to_csv(index=False).encode("utf-8")

    h0 = client.get("/health").json()
    calls_start = h0["inference_call_count"]

    # First request with no_cache=true: forces fresh inference and populates cache
    print("\nSending initial request with no_cache=true (expecting inference execution)...")
    t0 = time.perf_counter()
    resp1 = client.post(
        "/predict?no_cache=true",
        files={"sales": ("sales.csv", io.BytesIO(s_bytes), "text/csv"), "stock": ("stock.csv", io.BytesIO(st_bytes), "text/csv")},
    )
    t1 = time.perf_counter() - t0
    assert resp1.status_code == 200
    h1 = client.get("/health").json()
    assert h1["inference_call_count"] == calls_start + 1

    # Second request with identical bytes: MUST hit cache
    print("Sending second identical request (expecting cache hit)...")
    t2_start = time.perf_counter()
    resp2 = client.post(
        "/predict",
        files={"sales": ("sales.csv", io.BytesIO(s_bytes), "text/csv"), "stock": ("stock.csv", io.BytesIO(st_bytes), "text/csv")},
    )
    t2 = time.perf_counter() - t2_start
    assert resp2.status_code == 200
    h2 = client.get("/health").json()
    assert h2["inference_call_count"] == calls_start + 1  # Did NOT increase!
    assert t2 < 0.5, f"Cache hit took {t2:.3f}s, expected < 0.5s"
    assert resp1.text == resp2.text
    print(f"Cache test PASSED: initial inference {t1:.2f}s, cache hit {t2:.4f}s, call count held at {calls_start + 1}!")


def test_13_reproducibility_with_bypass_and_perturbation():
    dates = pd.date_range("2026-01-28", "2026-07-26", freq="D")
    s_light = pd.DataFrame([{
        "date": d.strftime("%Y-%m-%d"),
        "composite_division": "div_0",
        "cities_name": "city_0",
        "warehouse_name": "warehouse_name_1",
        "category_lvl_1": "category_lvl_1_0",
        "category_lvl_2": "category_lvl_2_0",
        "category_lvl_3": "category_lvl_3_0",
        "article": "char_article_1",
        "characteristic": "char_0",
        "quantity": 5.0,
        "sales_sum": 10.0,
        "discount_sum": 1.0,
    } for d in dates])
    st_light = pd.DataFrame([{
        "date": d.strftime("%Y-%m-%d"),
        "warehouse_name": "warehouse_name_1",
        "article": "char_article_1",
        "stock": 50.0,
    } for d in dates])

    s_bytes = s_light.to_csv(index=False).encode("utf-8")
    st_bytes = st_light.to_csv(index=False).encode("utf-8")

    h0 = client.get("/health").json()
    calls_start = h0["inference_call_count"]

    # Run 1 with cache bypass: MUST trigger real inference
    print("\n[Run 1 with bypass] Sending POST /predict?no_cache=true...")
    resp1 = client.post(
        "/predict?no_cache=true",
        files={"sales": ("sales.csv", io.BytesIO(s_bytes), "text/csv"), "stock": ("stock.csv", io.BytesIO(st_bytes), "text/csv")},
    )
    assert resp1.status_code == 200
    h1 = client.get("/health").json()
    assert h1["inference_call_count"] == calls_start + 1
    df1 = pd.read_csv(io.StringIO(resp1.text))

    # Run 2 with cache bypass: MUST trigger real inference again!
    print("[Run 2 with bypass] Sending POST /predict with X-Bypass-Cache: 1...")
    resp2 = client.post(
        "/predict",
        headers={"X-Bypass-Cache": "1"},
        files={"sales": ("sales.csv", io.BytesIO(s_bytes), "text/csv"), "stock": ("stock.csv", io.BytesIO(st_bytes), "text/csv")},
    )
    assert resp2.status_code == 200
    h2 = client.get("/health").json()
    assert h2["inference_call_count"] == calls_start + 2  # Inference ran again!
    df2 = pd.read_csv(io.StringIO(resp2.text))

    # Assert exact deterministic output
    diff = np.abs(df1["quantity"] - df2["quantity"]).sum()
    assert diff == 0, f"Reproducibility failed across independent model runs: diff={diff}"
    print("Exact reproducibility verified: both independent runs executed models and produced identical outputs!")

    # Run 3 with perturbed sales: MUST change output
    print("[Run 3 with perturbation] Sending perturbed sales (+50% sales on warehouse_name_1)...")
    s_perturbed = s_light.copy()
    mask = s_perturbed["warehouse_name"] == "warehouse_name_1"
    s_perturbed.loc[mask, "quantity"] = s_perturbed.loc[mask, "quantity"] * 1.5
    s_perturbed.loc[mask, "sales_sum"] = s_perturbed.loc[mask, "sales_sum"] + np.log(1.5)
    s_pert_bytes = s_perturbed.to_csv(index=False).encode("utf-8")

    resp3 = client.post(
        "/predict?no_cache=true",
        files={"sales": ("sales.csv", io.BytesIO(s_pert_bytes), "text/csv"), "stock": ("stock.csv", io.BytesIO(st_bytes), "text/csv")},
    )
    assert resp3.status_code == 200
    h3 = client.get("/health").json()
    assert h3["inference_call_count"] == calls_start + 3
    df3 = pd.read_csv(io.StringIO(resp3.text))

    diff_pert = np.abs(df3["quantity"] - df1["quantity"]).sum()
    rows_changed = int((df3["quantity"] != df1["quantity"]).sum())
    print(f"Perturbation changed {diff_pert} units across {rows_changed} rows.")
    assert diff_pert > 0, "Perturbation test failed: models did not respond to input data perturbation!"
    print("Perturbation test PASSED: Real ML models actively respond to input data changes!")


if __name__ == "__main__":
    init_server()
    tests = [
        test_01_healthcheck,
        test_02_get_predict_before_post_returns_409,
        test_03_get_predict_all_before_post_returns_409,
        test_04_error_missing_both_query_params,
        test_05_error_invalid_format_predict_all,
        test_06_error_empty_files,
        test_07_error_missing_column,
        test_08_error_negative_quantity,
        test_09_error_history_under_180_days,
        test_10_dynamic_date_rules,
        test_11_full_post_and_get_contract,
        test_12_cache_behavior,
        test_13_reproducibility_with_bypass_and_perturbation,
    ]
    passed = 0
    for t in tests:
        print(f"Running {t.__name__}...", end=" ")
        t()
        print("PASSED")
        passed += 1
    print(f"\nAll {passed} unit, integration, reproducibility, and sensitivity tests PASSED successfully!")
