import numpy as np
import pandas as pd
from typing import Dict, List, Optional

METRIC_WEIGHTS: Dict[str, float] = {
    "smape": 0.50,
    "wape":  0.20,
    "rmse":  0.15,
    "bias":  0.10,
    "rmsle": 0.05,
}

def compute_all_metrics_fast(df_true: pd.DataFrame, df_pred: pd.DataFrame) -> Dict[str, float]:
    """
    Быстрый и 100% математически точный расчет метрик хакатона.
    Работает за < 1 секунды вместо нескольких минут.
    """
    df = df_true[["date", "warehouse_name", "article", "quantity"]].merge(
        df_pred[["date", "warehouse_name", "article", "quantity"]],
        on=["date", "warehouse_name", "article"],
        how="left",
        suffixes=("_true", "_pred")
    )
    df["quantity_pred"] = df["quantity_pred"].fillna(0.0).clip(lower=0.0)
    
    y_true = df["quantity_true"].values.astype(float)
    y_pred = df["quantity_pred"].values.astype(float)
    
    # 1. WAPE (глобально)
    denom = np.sum(np.abs(y_true))
    wape_val = float(np.sum(np.abs(y_true - y_pred)) / denom * 100.0) if denom > 0 else 0.0
    
    # 2. RMSE (глобально)
    rmse_val = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    
    # 3. RMSLE (глобально)
    rmsle_val = float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))
    
    # Для метрик по сериям (SMAPE и Bias)
    df["series_id"] = df["warehouse_name"].astype(str) + "|" + df["article"].astype(str)
    
    # SMAPE per element: |y - y_hat| / ((|y| + |y_hat|) / 2)
    # If both 0 -> 0
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    with np.errstate(divide='ignore', invalid='ignore'):
        elem_smape = np.where(denominator > 0, numerator / denominator * 100.0, 0.0)
    
    df["elem_smape"] = elem_smape
    df["smape_valid"] = (denominator > 0).astype(np.int8)
    
    # Серийный SMAPE: среднее по каждой серии, затем среднее по всем сериям
    series_group = df.groupby("series_id")
    # Официальный score.py исключает точки 0/0 из среднего внутри серии.
    smape_sum = series_group["elem_smape"].sum()
    smape_count = series_group["smape_valid"].sum()
    smape_series = pd.Series(
        np.divide(
            smape_sum.to_numpy(),
            smape_count.to_numpy(),
            out=np.zeros(len(smape_sum), dtype=float),
            where=smape_count.to_numpy() > 0,
        ),
        index=smape_sum.index,
    )
    smape_val = float(smape_series.mean())
    
    # Bias: | mean(pred) - mean(true) | / mean(true) * 100.0
    # Исключаются серии, где mean(true) == 0
    mean_true = series_group["quantity_true"].mean()
    mean_pred = series_group["quantity_pred"].mean()
    
    valid_series = mean_true > 0
    if valid_series.any():
        bias_series = (np.abs(mean_pred[valid_series] - mean_true[valid_series]) / mean_true[valid_series]) * 100.0
        bias_val = float(bias_series.mean())
    else:
        bias_val = 0.0
        
    return {
        "smape": smape_val,
        "wape":  wape_val,
        "rmse":  rmse_val,
        "bias":  bias_val,
        "rmsle": rmsle_val,
    }

if __name__ == "__main__":
    df_truth = pd.read_csv("val_ground_truth.csv")
    for name in ["baseline_zero", "baseline_naive_last", "baseline_dow_mean"]:
        df_sub = pd.read_csv(f"{name}.csv")
        m = compute_all_metrics_fast(df_truth, df_sub)
        print(f"\n=== {name} ===")
        for k, v in m.items():
            print(f"  {k:6s}: {v:10.4f}")
