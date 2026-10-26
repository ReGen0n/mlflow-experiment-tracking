"""Pulls all runs from the experiment and saves a comparison to reports/
(csv with everything, short markdown table and a bar chart)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from common import DEFAULT_EXPERIMENT, configure_tracking  # noqa: E402

OUT = Path(__file__).resolve().parent / "reports"

COLS = {
    "tags.mlflow.runName": "run",
    "run_id": "run_id",
    "params.model": "model",
    "metrics.val_f1": "val_f1",
    "metrics.val_recall": "val_recall",
    "metrics.val_precision": "val_precision",
    "metrics.val_roc_auc": "val_auc",
    "metrics.cv_f1_mean": "cv_f1",
    "metrics.cv_f1_std": "cv_f1_std",
    "metrics.train_f1": "train_f1",
    "metrics.overfit_gap_f1": "gap",
}


def load_runs(experiment: str, include_repro: bool = False):
    df = mlflow.search_runs(experiment_names=[experiment], order_by=["metrics.val_f1 DESC"])
    if df.empty:
        raise SystemExit(f"No runs in experiment '{experiment}'. Run run_experiments.py first.")
    # skip reruns made with --config-from-run
    if not include_repro and "tags.reproduced_from_run" in df:
        df = df[df["tags.reproduced_from_run"].isna()]
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    args = p.parse_args()
    configure_tracking()

    df = load_runs(args.experiment)
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "runs_comparison.csv", index=False)

    table = df[list(COLS)].rename(columns=COLS).copy()
    table["run_id"] = table["run_id"].str[:8]
    num = table.select_dtypes("number").columns
    table[num] = table[num].round(4)
    md = ["| " + " | ".join(table.columns) + " |", "|" + "---|" * len(table.columns)]
    md += ["| " + " | ".join(str(v) for v in row) + " |" for row in table.itertuples(index=False)]
    (OUT / "runs_comparison.md").write_text("\n".join(md) + "\n")
    print(table.to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 4.6))
    t = table.iloc[::-1]
    y = range(len(t))
    ax.barh([i + 0.2 for i in y], t["val_f1"], height=0.4, label="validation F1", color="#2b6cb0")
    ax.barh([i - 0.2 for i in y], t["cv_f1"], height=0.4, xerr=t["cv_f1_std"],
            label="5-fold CV F1 (train)", color="#a0aec0")
    ax.set_yticks(list(y), t["run"])
    ax.set_xlim(0.85, 1.0)
    ax.set_xlabel("F1 (positive class = malignant)")
    ax.set_title(f"Experiment '{args.experiment}' - run comparison")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "val_f1_comparison.png", dpi=130)
    print(f"\nSaved comparison to {OUT}/")


if __name__ == "__main__":
    main()
