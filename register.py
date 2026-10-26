"""Picks the best run, checks it on the test set and registers it in the Model Registry.

Best = highest val_f1 (if equal: higher cv_f1, then simpler model).
Gets alias "champion" only if:
  - val_f1 >= val_f1 of the baseline run
  - val_recall >= 0.95 (missing a cancer is the worst mistake here)
  - test_f1 >= 0.93
otherwise "challenger".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from common import DEFAULT_EXPERIMENT, configure_tracking, load_splits  # noqa: E402
from compare import load_runs  # noqa: E402

MIN_VAL_RECALL = 0.95
MIN_TEST_F1 = 0.93
# for tie-break, lower = simpler
COMPLEXITY = {"logreg": 0, "knn": 1, "gradient_boosting": 2, "random_forest": 3}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    p.add_argument("--model-name", default="wdbc-malignancy-clf")
    args = p.parse_args()
    configure_tracking()
    client = MlflowClient()

    runs = load_runs(args.experiment).copy()
    runs["complexity"] = runs["params.model"].map(COMPLEXITY)
    runs = runs.sort_values(
        ["metrics.val_f1", "metrics.cv_f1_mean", "complexity"], ascending=[False, False, True]
    )
    best = runs.iloc[0]
    run_id = best["run_id"]

    baseline = runs[runs.get("tags.role") == "baseline"] if "tags.role" in runs else runs.iloc[0:0]
    baseline_f1 = float(baseline["metrics.val_f1"].iloc[0]) if len(baseline) else 0.0

    # test set, only for the chosen model
    params = best.filter(like="params.")
    splits = load_splits(int(params["params.seed"]), float(params["params.val_size"]),
                         float(params["params.test_size"]))
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    pred = model.predict(splits.X_test)
    proba = model.predict_proba(splits.X_test)[:, 1]
    test = {
        "test_accuracy": accuracy_score(splits.y_test, pred),
        "test_precision": precision_score(splits.y_test, pred),
        "test_recall": recall_score(splits.y_test, pred),
        "test_f1": f1_score(splits.y_test, pred),
        "test_roc_auc": roc_auc_score(splits.y_test, proba),
    }
    for k, v in test.items():
        client.log_metric(run_id, k, v)

    checks = {
        f"val_f1 >= baseline ({baseline_f1:.4f})": best["metrics.val_f1"] >= baseline_f1,
        f"val_recall >= {MIN_VAL_RECALL}": best["metrics.val_recall"] >= MIN_VAL_RECALL,
        f"test_f1 >= {MIN_TEST_F1}": test["test_f1"] >= MIN_TEST_F1,
    }
    alias = "champion" if all(checks.values()) else "challenger"

    mv = mlflow.register_model(f"runs:/{run_id}/model", args.model_name)
    client.set_registered_model_alias(args.model_name, alias, mv.version)
    client.update_registered_model(
        args.model_name,
        description="malignant (1) vs benign (0), UCI WDBC dataset. "
        "Promotion rule is in the README.",
    )
    client.update_model_version(
        args.model_name, mv.version,
        description=f"Run {best['tags.mlflow.runName']} ({run_id}); "
        f"val_f1={best['metrics.val_f1']:.4f}, test_f1={test['test_f1']:.4f}",
    )
    for k, v in {
        "source_run_name": best["tags.mlflow.runName"],
        "model_family": best["params.model"],
        "val_f1": f"{best['metrics.val_f1']:.4f}",
        "test_f1": f"{test['test_f1']:.4f}",
        "git_commit": best.get("tags.git_commit", "unknown"),
        "dataset_sha256": best.get("tags.dataset_sha256", "unknown"),
        "promotion_checks": json.dumps({k: bool(v) for k, v in checks.items()}),
    }.items():
        client.set_model_version_tag(args.model_name, mv.version, k, str(v))

    print(f"Best run   : {best['tags.mlflow.runName']}  run_id={run_id}")
    print(f"Validation : f1={best['metrics.val_f1']:.4f} recall={best['metrics.val_recall']:.4f} "
          f"cv_f1={best['metrics.cv_f1_mean']:.4f}")
    print("Test       : " + "  ".join(f"{k[5:]}={v:.4f}" for k, v in test.items()))
    for name, ok in checks.items():
        print(f"  [{'x' if ok else ' '}] {name}")
    print(f"Registered : {args.model_name} v{mv.version}  alias=@{alias}")
    print(f"Load with  : mlflow.pyfunc.load_model('models:/{args.model_name}@{alias}')")

    out = Path(__file__).resolve().parent / "reports"
    out.mkdir(exist_ok=True)
    (out / "registry_result.json").write_text(json.dumps({
        "model_name": args.model_name, "version": mv.version, "alias": alias,
        "run_id": run_id, "run_name": best["tags.mlflow.runName"],
        "val_f1": best["metrics.val_f1"], "baseline_val_f1": baseline_f1,
        **test, "checks": {k: bool(v) for k, v in checks.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
