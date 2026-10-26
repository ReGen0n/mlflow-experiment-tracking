"""Trains one model and logs it to MLflow.

usage:
  python train.py --model logreg --C 1.0
  python train.py --model random_forest --n-estimators 300 --max-depth 8
  python train.py --config-from-run <RUN_ID>   # retrain with params from an old run
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from common import (  # noqa: E402
    DATASET_DOI,
    DATASET_LICENSE,
    DATASET_NAME,
    DATASET_URL,
    DEFAULT_EXPERIMENT,
    MODEL_HPARAMS,
    build_model,
    configure_tracking,
    git_info,
    load_splits,
    resolve_hparams,
)

HP_FLAGS = {
    "C": "C",
    "max_iter": "max_iter",
    "n_estimators": "n_estimators",
    "max_depth": "max_depth",
    "learning_rate": "learning_rate",
    "n_neighbors": "n_neighbors",
    "weights": "weights",
}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MLflow-instrumented training run")
    p.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    p.add_argument("--model", choices=list(MODEL_HPARAMS), default="logreg")
    p.add_argument("--run-name", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-size", type=float, default=0.2)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--baseline", action="store_true", help="mark run as baseline")
    p.add_argument(
        "--config-from-run",
        metavar="RUN_ID",
        help="take params from this run instead of flags",
    )
    # if not given -> default from common.py
    p.add_argument("--C", type=float)
    p.add_argument("--max-iter", dest="max_iter", type=int)
    p.add_argument("--n-estimators", dest="n_estimators", type=int)
    p.add_argument("--max-depth", dest="max_depth", type=int)
    p.add_argument("--learning-rate", dest="learning_rate", type=float)
    p.add_argument("--n-neighbors", dest="n_neighbors", type=int)
    p.add_argument("--weights", choices=["uniform", "distance"])
    return p.parse_args(argv)


def metrics_for(y_true, y_pred, y_proba, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}_precision": precision_score(y_true, y_pred, zero_division=0),
        f"{prefix}_recall": recall_score(y_true, y_pred),
        f"{prefix}_f1": f1_score(y_true, y_pred),
        f"{prefix}_roc_auc": roc_auc_score(y_true, y_proba),
    }


def config_from_run(run_id: str) -> dict:
    run = MlflowClient().get_run(run_id)
    p = run.data.params
    model = p["model"]
    hparams = {k: p[k] for k in MODEL_HPARAMS[model] if k in p}
    return {
        "model": model,
        "hparams": resolve_hparams(model, hparams),
        "seed": int(p["seed"]),
        "val_size": float(p["val_size"]),
        "test_size": float(p["test_size"]),
        "experiment_id": run.info.experiment_id,
        "source_metrics": run.data.metrics,
        "source_tags": run.data.tags,
    }


def train(cfg: dict, experiment: str, run_name: str | None, extra_tags: dict) -> tuple[str, dict]:
    splits = load_splits(cfg["seed"], cfg["val_size"], cfg["test_size"])
    model = build_model(cfg["model"], cfg["hparams"], cfg["seed"])

    mlflow.set_experiment(experiment)
    run_name = run_name or cfg["model"]
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(
            {
                "model": cfg["model"],
                "seed": cfg["seed"],
                "val_size": cfg["val_size"],
                "test_size": cfg["test_size"],
                "preprocessing": "StandardScaler",
                "n_train": len(splits.y_train),
                "n_val": len(splits.y_val),
                **cfg["hparams"],
            }
        )
        mlflow.set_tags(
            {
                **git_info(),
                "dataset_name": DATASET_NAME,
                "dataset_url": DATASET_URL,
                "dataset_license": DATASET_LICENSE,
                "dataset_doi": DATASET_DOI,
                "dataset_sha256": splits.dataset_sha256,
                "positive_class": "malignant",
                **extra_tags,
            }
        )
        # dataset shows up in the UI then. mlflow prints some databricks warning here, ignore it
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ds = mlflow.data.from_numpy(
                    splits.X_train, targets=splits.y_train, name="wdbc-train",
                    source=DATASET_URL, digest=splits.dataset_sha256[:16],
                )
                mlflow.log_input(ds, context="training")
        except Exception:
            pass

        # 5-fold CV on train just to see if the score is stable
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg["seed"])
        cv_f1 = cross_val_score(model, splits.X_train, splits.y_train, cv=cv, scoring="f1")
        model.fit(splits.X_train, splits.y_train)

        tr_pred = model.predict(splits.X_train)
        tr_proba = model.predict_proba(splits.X_train)[:, 1]
        va_pred = model.predict(splits.X_val)
        va_proba = model.predict_proba(splits.X_val)[:, 1]

        metrics = {
            **metrics_for(splits.y_train, tr_pred, tr_proba, "train"),
            **metrics_for(splits.y_val, va_pred, va_proba, "val"),
            "cv_f1_mean": float(cv_f1.mean()),
            "cv_f1_std": float(cv_f1.std()),
        }
        metrics["overfit_gap_f1"] = metrics["train_f1"] - metrics["val_f1"]
        mlflow.log_metrics(metrics)

        # plots and files -> artifacts/evaluation
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)

            fig, ax = plt.subplots(figsize=(4.5, 4))
            ConfusionMatrixDisplay.from_predictions(
                splits.y_val, va_pred, display_labels=["benign", "malignant"],
                cmap="Blues", ax=ax, colorbar=False,
            )
            ax.set_title(f"{run_name} - validation")
            fig.tight_layout()
            fig.savefig(tmp / "confusion_matrix_val.png", dpi=120)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(4.5, 4))
            RocCurveDisplay.from_predictions(splits.y_val, va_proba, ax=ax, name=run_name)
            ax.set_title("ROC - validation")
            fig.tight_layout()
            fig.savefig(tmp / "roc_curve_val.png", dpi=120)
            plt.close(fig)

            pd.DataFrame(
                {"y_true": splits.y_val, "y_pred": va_pred, "p_malignant": np.round(va_proba, 4)}
            ).to_csv(tmp / "val_predictions.csv", index=False)

            (tmp / "classification_report_val.txt").write_text(
                classification_report(splits.y_val, va_pred, target_names=["benign", "malignant"], digits=4)
            )
            (tmp / "run_config.json").write_text(
                json.dumps({k: cfg[k] for k in ("model", "hparams", "seed", "val_size", "test_size")}, indent=2)
            )
            mlflow.log_artifacts(str(tmp), artifact_path="evaluation")

        signature = infer_signature(splits.X_val, model.predict(splits.X_val))
        mlflow.sklearn.log_model(
            model, artifact_path="model", signature=signature, input_example=splits.X_val[:3]
        )

        print(
            f"[{run_name}] run_id={run.info.run_id}  "
            f"val_f1={metrics['val_f1']:.4f}  val_recall={metrics['val_recall']:.4f}  "
            f"val_auc={metrics['val_roc_auc']:.4f}  cv_f1={metrics['cv_f1_mean']:.4f}±{metrics['cv_f1_std']:.4f}"
        )
        return run.info.run_id, metrics


def main(argv=None) -> None:
    args = parse_args(argv)
    uri = configure_tracking()
    print(f"MLflow tracking URI: {uri}")

    extra_tags: dict[str, str] = {}
    if args.config_from_run:
        src = config_from_run(args.config_from_run)
        cfg = {k: src[k] for k in ("model", "hparams", "seed", "val_size", "test_size")}
        extra_tags["reproduced_from_run"] = args.config_from_run
        run_name = args.run_name or f"repro_{args.config_from_run[:8]}"
        print(f"Re-training config from run {args.config_from_run}: {json.dumps(cfg)}")
    else:
        overrides = {hp: getattr(args, flag) for flag, hp in HP_FLAGS.items()}
        cfg = {
            "model": args.model,
            "hparams": resolve_hparams(args.model, overrides),
            "seed": args.seed,
            "val_size": args.val_size,
            "test_size": args.test_size,
        }
        run_name = args.run_name
        src = None
    if args.baseline:
        extra_tags["role"] = "baseline"

    _, metrics = train(cfg, args.experiment, run_name, extra_tags)

    # compare with the original run
    if src is not None:
        print("\nReproducibility check (original vs re-run):")
        ok = True
        for k in ("val_f1", "val_roc_auc", "cv_f1_mean"):
            a, b = src["source_metrics"].get(k), metrics[k]
            same = a is not None and abs(a - b) < 1e-9
            ok &= same
            print(f"  {k:<12} {a:.6f} vs {b:.6f}  {'OK' if same else 'DIFFERENT'}")
        print("  -> identical" if ok else "  -> metrics differ (check library versions / data)")


if __name__ == "__main__":
    main()
