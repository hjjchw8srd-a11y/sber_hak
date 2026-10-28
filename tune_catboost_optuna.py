"""Optuna-подбор двухэтапного CatBoost на rolling validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd

from fast_score import compute_all_metrics_fast
from run_weekly_feature_experiments import (
    HORIZON,
    WeeklyMatrices,
    columns_for_experiment,
    make_truth,
    prediction_frame,
    train_and_predict,
)
from score import normalized_score


EXPERIMENT = "base_calendar_network"
FIXED_SCORE = 0.7432842543011406


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("weekly_data/weekly_dataset.csv.gz"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/optuna"))
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--max-train-rows", type=int, default=350_000)
    return parser.parse_args()


def prepare_folds(
    matrices: WeeklyMatrices, folds: int, max_train_rows: int
) -> list[dict[str, object]]:
    prepared: list[dict[str, object]] = []
    latest_validation_origin = len(matrices.weeks) - HORIZON
    for fold in range(folds):
        validation_origin = latest_validation_origin - fold * HORIZON
        latest_train_origin = validation_origin - HORIZON
        earliest_train_origin = max(26, latest_train_origin - 17)
        train_origins = np.arange(earliest_train_origin, latest_train_origin + 1)
        train, target, _ = matrices.make_rows(
            train_origins, max_train_rows, seed=42 + fold
        )
        validation, _, _ = matrices.make_rows(np.array([validation_origin]), None)
        truth = make_truth(matrices, validation_origin)
        baseline = prediction_frame(
            truth, validation["quantity_mean_13"].to_numpy()
        )
        prepared.append(
            {
                "fold": fold,
                "train": train,
                "target": target,
                "validation": validation,
                "truth": truth,
                "baseline_metrics": compute_all_metrics_fast(truth, baseline),
                "validation_start": str(matrices.weeks[validation_origin].date()),
                "validation_end": str(
                    matrices.weeks[validation_origin + HORIZON - 1].date()
                ),
            }
        )
    return prepared


def save_results(study: optuna.Study, output_dir: Path) -> None:
    trials = study.trials_dataframe(
        attrs=("number", "value", "state", "params", "user_attrs")
    )
    trials.to_csv(output_dir / "optuna_trials.csv", index=False)

    complete = trials[trials["state"] == "COMPLETE"].copy()
    fig, ax = plt.subplots(figsize=(11, 5.5))
    if not complete.empty:
        ax.plot(
            complete["number"],
            complete["value"],
            marker="o",
            linewidth=1.5,
            label="Средний score попытки",
        )
        running_best = complete["value"].cummin()
        ax.plot(
            complete["number"],
            running_best,
            linewidth=2.5,
            color="#2F855A",
            label="Лучший найденный score",
        )
    ax.axhline(
        FIXED_SCORE,
        color="#C53030",
        linestyle="--",
        label=f"Фиксированные параметры: {FIXED_SCORE:.4f}",
    )
    ax.set_xlabel("Номер попытки")
    ax.set_ylabel("Средний normalized score")
    ax.set_title("Optuna: подбор CatBoost на трёх rolling-окнах", fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "optuna_trials.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    best = {
        "experiment": EXPERIMENT,
        "best_trial": study.best_trial.number,
        "best_mean_score": study.best_value,
        "fixed_parameters_score": FIXED_SCORE,
        "improvement": FIXED_SCORE - study.best_value,
        "best_params": study.best_params,
        "fold_scores": [
            study.best_trial.user_attrs.get(f"fold_{fold}_score")
            for fold in range(3)
        ],
    }
    (output_dir / "best_params.json").write_text(
        json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = f"""# Optuna-подбор CatBoost

- Набор признаков: `{EXPERIMENT}`.
- Цель: средний normalized score по трём rolling-окнам.
- Порог классификатора: 0.5, не подбирался вместе с CatBoost.
- Score фиксированных параметров: **{FIXED_SCORE:.6f}**.
- Лучший score Optuna: **{study.best_value:.6f}**.
- Улучшение: **{FIXED_SCORE - study.best_value:.6f}**.
- Лучшая попытка: {study.best_trial.number}.

## Лучшие параметры

```json
{json.dumps(study.best_params, ensure_ascii=False, indent=2)}
```

## Score по окнам

{json.dumps(best['fold_scores'], ensure_ascii=False)}
"""
    (output_dir / "OPTUNA_REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.data, parse_dates=["week"])
    matrices = WeeklyMatrices(data)
    folds = prepare_folds(matrices, args.folds, args.max_train_rows)
    columns = columns_for_experiment(EXPERIMENT)

    def objective(trial: optuna.Trial) -> float:
        iterations = trial.suggest_int("iterations", 120, 360, step=40)
        params: dict[str, object] = {
            "depth": trial.suggest_int("depth", 5, 9),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.025, 0.15, log=True
            ),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
            "random_strength": trial.suggest_float(
                "random_strength", 0.1, 5.0, log=True
            ),
            "bagging_temperature": trial.suggest_float(
                "bagging_temperature", 0.0, 1.5
            ),
        }
        scores: list[float] = []
        for fold_data in folds:
            prediction, _ = train_and_predict(
                fold_data["train"],
                fold_data["target"],
                fold_data["validation"],
                columns,
                iterations=iterations,
                model_params=params,
            )
            metrics = compute_all_metrics_fast(
                fold_data["truth"],
                prediction_frame(fold_data["truth"], prediction),
            )
            score = normalized_score(metrics, fold_data["baseline_metrics"])
            fold_number = int(fold_data["fold"])
            scores.append(score)
            trial.set_user_attr(f"fold_{fold_number}_score", score)
            trial.report(float(np.mean(scores)), step=fold_number)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    storage = f"sqlite:///{(args.output_dir / 'catboost_optuna.db').as_posix()}"
    study = optuna.create_study(
        study_name="catboost_weekly",
        storage=storage,
        direction="minimize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    remaining = max(0, args.trials - len(study.trials))
    if remaining:
        study.optimize(objective, n_trials=remaining, gc_after_trial=True)
    save_results(study, args.output_dir)
    print(
        json.dumps(
            {
                "best_value": study.best_value,
                "best_params": study.best_params,
                "trials": len(study.trials),
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
