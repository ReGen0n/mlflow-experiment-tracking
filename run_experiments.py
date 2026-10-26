"""Runs all 6 variants. Each list = arguments for train.py."""

from __future__ import annotations

import train

VARIANTS: list[list[str]] = [
    ["--model", "logreg", "--C", "1.0", "--run-name", "logreg_C1_baseline", "--baseline"],
    ["--model", "logreg", "--C", "0.1", "--run-name", "logreg_C0.1"],
    ["--model", "random_forest", "--n-estimators", "200", "--max-depth", "6", "--run-name", "rf_200_d6"],
    ["--model", "random_forest", "--n-estimators", "500", "--max-depth", "12", "--run-name", "rf_500_d12"],
    ["--model", "gradient_boosting", "--n-estimators", "150", "--learning-rate", "0.1", "--max-depth", "3",
     "--run-name", "gb_150_lr0.1"],
    ["--model", "knn", "--n-neighbors", "7", "--weights", "distance", "--run-name", "knn_7_distance"],
]

if __name__ == "__main__":
    for argv in VARIANTS:
        train.main(argv)
