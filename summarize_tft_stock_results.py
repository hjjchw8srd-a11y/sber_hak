"""Сводное сравнение базового TFT и TFT с полным блоком остатков."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    base_path = Path("experiments/tft/tft_postprocessing_cross_fitted.csv")
    stock_path = Path("experiments/tft_features_final/tft_feature_results.csv")
    output_dir = stock_path.parent

    base = pd.read_csv(base_path).rename(
        columns={"held_out_fold": "fold", "held_out_score": "base_score"}
    )[["fold", "base_score"]]
    stock = pd.read_csv(stock_path)[["fold", "normalized_score", "run_seconds"]].rename(
        columns={"normalized_score": "stock_score", "run_seconds": "stock_run_seconds"}
    )
    comparison = base.merge(stock, on="fold", validate="one_to_one")
    comparison["delta_stock_minus_base"] = comparison["stock_score"] - comparison["base_score"]
    comparison.to_csv(output_dir / "tft_stock_vs_base.csv", index=False)

    mean_base = float(comparison["base_score"].mean())
    mean_stock = float(comparison["stock_score"].mean())
    mean_delta = mean_stock - mean_base
    base_submission = pd.read_csv("experiments/tft/tft_final_submission.csv")
    stock_submission = pd.read_csv("experiments/tft/tft_stock_final_submission.csv")
    base_values_final = base_submission["quantity"].to_numpy(dtype=float)
    stock_values_final = stock_submission["quantity"].to_numpy(dtype=float)
    changed_rows = int(np.sum(base_values_final != stock_values_final))
    changed_rate = changed_rows / len(base_values_final)
    base_sum = float(base_values_final.sum())
    stock_sum = float(stock_values_final.sum())
    rows = "\n".join(
        f"| {int(row.fold)} | {row.base_score:.6f} | {row.stock_score:.6f} | "
        f"{row.delta_stock_minus_base:+.6f} |"
        for row in comparison.itertuples()
    )
    report = f"""# TFT: влияние истории остатков

Обе модели обучены по 300 шагов и проверены на одинаковых трёх окнах. Для обеих
использована одинаковая постобработка: прогнозы меньше 0.75 обнуляются, затем
значения округляются. Отрицательный delta означает улучшение stock-модели.

| fold | базовый TFT | TFT + stock_full | delta |
| --- | --- | --- | --- |
{rows}
| среднее | {mean_base:.6f} | {mean_stock:.6f} | {mean_delta:+.6f} |

Полный блок остатков улучшил два более свежих окна, но ухудшил самое раннее.
Среднее изменение равно **{mean_delta:+.6f}**, то есть практически нулю. Поэтому
остатки нельзя назвать бесполезными, но заменять ими текущий базовый TFT только
по этой проверке пока не стоит. Stock-модель сохраняется как кандидат для
ансамбля и для повторной проверки на дополнительных временных окнах.

После проверки обе версии обучены на всей доступной истории. В финальном
9-недельном прогнозе stock-модель изменила {changed_rows:,} из
{len(base_values_final):,} строк ({changed_rate:.1%}). Суммарный прогноз вырос
с {base_sum:,.0f} до {stock_sum:,.0f} единиц.
"""
    (output_dir / "TFT_STOCK_COMPARISON.md").write_text(report, encoding="utf-8")

    labels = [f"fold {int(value)}" for value in comparison["fold"]] + ["среднее"]
    base_values = comparison["base_score"].tolist() + [mean_base]
    stock_values = comparison["stock_score"].tolist() + [mean_stock]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, base_values, width, label="базовый TFT", color="#4C78A8")
    ax.bar(x + width / 2, stock_values, width, label="TFT + stock_full", color="#F58518")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Normalized score (меньше — лучше)")
    ax.set_title("Влияние полного блока остатков на TFT")
    ax.set_ylim(min(base_values + stock_values) - 0.004, max(base_values + stock_values) + 0.004)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "tft_stock_vs_base.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print({"base_mean": mean_base, "stock_mean": mean_stock, "delta": mean_delta})


if __name__ == "__main__":
    main()
