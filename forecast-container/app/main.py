"""
main.py

FastAPI HTTP service for Section 10 of the hackathon specification.
Endpoints:
- POST /predict: Full forecast from uploaded sales and stock CSV files
- GET /predict: Filtered forecast by article and/or warehouse
- GET /predict/all: Full forecast from last successful POST (CSV or JSON)
- GET /health: Healthcheck and model readiness
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .predictor import Predictor
from .validators import (
    APIValidationError,
    validate_and_load_sales,
    validate_and_load_stock,
)

app = FastAPI(
    title="Demand Forecast API - Candidate LGBM52",
    description="Production-ready multi-model demand forecast service conforming to Hackathon Section 10.",
    version="1.0.0",
)

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"

# Server state
predictor: Optional[Predictor] = None
bundle_digest: Optional[str] = None
inference_call_count: int = 0
predict_lock = asyncio.Lock()

# Atomic prediction cache
last_prediction_df: Optional[pd.DataFrame] = None
last_prediction_csv: Optional[bytes] = None
last_cache_key: Optional[str] = None


@app.on_event("startup")
def startup_event():
    global predictor, bundle_digest
    print(f"Loading model bundle from {MODEL_DIR}...")
    try:
        predictor = Predictor.load(MODEL_DIR)
        manifest_files = predictor.manifest.get("files", {})
        digest_bytes = json.dumps(manifest_files, sort_keys=True).encode("utf-8")
        bundle_digest = hashlib.sha256(digest_bytes).hexdigest()
        print(f"Model bundle loaded successfully! Bundle digest: {bundle_digest[:16]}...")
    except Exception as e:
        print(f"FATAL: Failed to load model bundle: {e}")
        predictor = None
        bundle_digest = None


@app.exception_handler(APIValidationError)
async def api_validation_exception_handler(request: Request, exc: APIValidationError):
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.detail,
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    detail = exc.detail
    if isinstance(detail, dict) and "code" in detail and "detail" in detail:
        content = detail
    else:
        code = "BAD_REQUEST" if exc.status_code == 400 else "ERROR"
        content = {"detail": str(detail), "code": code}
    return JSONResponse(status_code=exc.status_code, content=content)


@app.get("/health")
def healthcheck():
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail={"detail": "Model bundle is not loaded or initialization failed", "code": "SERVICE_UNAVAILABLE"},
        )
    return {
        "status": "healthy",
        "model_bundle": predictor.manifest.get("bundle_name", "candidate_lgbm52_bundle"),
        "bundle_digest": bundle_digest,
        "components": list(predictor.postprocessing.get("components", {}).keys()),
        "has_cached_forecast": last_prediction_df is not None,
        "inference_call_count": inference_call_count,
    }


@app.post("/predict")
async def post_predict(
    request: Request,
    sales: UploadFile = File(..., description="Sales CSV file (last ~180+ days)"),
    stock: UploadFile = File(..., description="Stock CSV file (last ~180+ days)"),
    no_cache: bool = Query(False, description="Explicitly bypass cache to force fresh model inference"),
):
    global last_prediction_df, last_cache_key, inference_call_count

    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail={"detail": "Model predictor not ready", "code": "SERVICE_UNAVAILABLE"},
        )

    # Check header or query for bypass
    bypass_cache = no_cache or request.headers.get("x-bypass-cache", "").lower() in ("1", "true")

    # Read uploaded bytes
    sales_bytes = await sales.read()
    stock_bytes = await stock.read()

    if len(sales_bytes) == 0:
        raise APIValidationError(400, "Uploaded sales file is empty", code="BAD_REQUEST")
    if len(stock_bytes) == 0:
        raise APIValidationError(400, "Uploaded stock file is empty", code="BAD_REQUEST")

    # Cache key: hash of both files + full bundle digest
    hasher = hashlib.sha256()
    hasher.update(sales_bytes)
    hasher.update(b"::")
    hasher.update(stock_bytes)
    hasher.update(b"::")
    hasher.update((bundle_digest or "").encode("utf-8"))
    cache_key = hasher.hexdigest()

    global last_prediction_df, last_prediction_csv, last_cache_key, inference_call_count

    async with predict_lock:
        if not bypass_cache and last_cache_key == cache_key and last_prediction_csv is not None:
            return Response(content=last_prediction_csv, media_type="text/csv")

        # Validate inputs only when actually running inference
        sales_df = validate_and_load_sales(sales_bytes)
        cutoff_date = sales_df["date"].max()
        stock_df = validate_and_load_stock(stock_bytes, cutoff_date=cutoff_date)

        # Run inference in executor thread to keep event loop responsive
        loop = asyncio.get_running_loop()
        result_df = await loop.run_in_executor(None, predictor.predict, sales_df, stock_df)
        inference_call_count += 1

        csv_buffer = io.StringIO()
        result_df.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode("utf-8")

        # Atomic cache update
        last_prediction_df = result_df.copy()
        last_prediction_csv = csv_bytes
        last_cache_key = cache_key

    return Response(content=csv_bytes, media_type="text/csv")


@app.get("/predict")
def get_predict_filter(
    article: Optional[str] = Query(None, description="Article identifier"),
    warehouse: Optional[str] = Query(None, description="Warehouse name"),
):
    if article is None and warehouse is None:
        raise HTTPException(
            status_code=400,
            detail={"detail": "At least one query parameter ('article' or 'warehouse') must be provided", "code": "BAD_REQUEST"},
        )

    if last_prediction_df is None:
        raise HTTPException(
            status_code=409,
            detail={"detail": "No predictions calculated yet. Call POST /predict first.", "code": "NO_PREDICTION_AVAILABLE"},
        )

    df = last_prediction_df
    if article is not None:
        df = df[df["article"] == article]
    if warehouse is not None:
        df = df[df["warehouse_name"] == warehouse]

    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    return Response(content=csv_buffer.getvalue().encode("utf-8"), media_type="text/csv")


@app.get("/predict/all")
def get_predict_all(
    format: str = Query("csv", description="Output format: 'csv' or 'json'"),
):
    if format not in ("csv", "json"):
        raise HTTPException(
            status_code=400,
            detail={"detail": f"Unsupported format '{format}'. Supported formats: 'csv', 'json'", "code": "BAD_REQUEST"},
        )

    if last_prediction_df is None:
        raise HTTPException(
            status_code=409,
            detail={"detail": "No predictions calculated yet. Call POST /predict first.", "code": "NO_PREDICTION_AVAILABLE"},
        )

    if format == "csv":
        if last_prediction_csv is not None:
            return Response(content=last_prediction_csv, media_type="text/csv")
        csv_buffer = io.StringIO()
        last_prediction_df.to_csv(csv_buffer, index=False)
        return Response(content=csv_buffer.getvalue().encode("utf-8"), media_type="text/csv")
    else:
        # JSON format
        records = last_prediction_df.to_dict(orient="records")
        return JSONResponse(content=records)
