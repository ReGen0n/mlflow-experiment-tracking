"""Common stuff used by all scripts: data split, models, git info.
Kept in one place so every run uses the same split and preprocessing."""

from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# dataset info, goes into tags of each run
DATASET_NAME = "Breast Cancer Wisconsin (Diagnostic)"
DATASET_URL = "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic"
DATASET_LICENSE = "CC BY 4.0"
DATASET_DOI = "10.24432/C5DW2B"

DEFAULT_EXPERIMENT = "thesis-mlflow-demo"
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"


@dataclass
class DataSplits:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    dataset_sha256: str


def configure_tracking() -> str:
    # env variable has priority, otherwise local sqlite
    import mlflow

    uri = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    mlflow.set_tracking_uri(uri)
    return uri


def load_splits(seed: int = 42, val_size: float = 0.2, test_size: float = 0.2) -> DataSplits:
    # train/val/test, stratified, fixed seed
    # test is used only once in register.py, not for choosing the model
    data = load_breast_cancer()
    X = data.data
    # in sklearn malignant = 0, I flip it so that 1 = malignant (positive class)
    y = (data.target == 0).astype(int)

    sha = hashlib.sha256()
    sha.update(np.ascontiguousarray(X, dtype=np.float64).tobytes())
    sha.update(np.ascontiguousarray(y, dtype=np.int64).tobytes())

    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    rel_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=rel_val, random_state=seed, stratify=y_tmp
    )
    return DataSplits(
        X_train, X_val, X_test, y_train, y_val, y_test,
        feature_names=list(data.feature_names),
        dataset_sha256=sha.hexdigest(),
    )


# default hyperparams, can be overridden from CLI
MODEL_HPARAMS: dict[str, dict] = {
    "logreg": {"C": 1.0, "max_iter": 1000},
    "random_forest": {"n_estimators": 200, "max_depth": 6},
    "gradient_boosting": {"n_estimators": 150, "learning_rate": 0.1, "max_depth": 3},
    "knn": {"n_neighbors": 7, "weights": "distance"},
}


def _cast(value, default):
    # mlflow returns params as strings, convert back
    if isinstance(value, str):
        if isinstance(default, bool):
            return value.lower() in {"1", "true", "yes"}
        if isinstance(default, int):
            return int(float(value))
        if isinstance(default, float):
            return float(value)
        if value == "None":
            return None
    return value


def resolve_hparams(model: str, overrides: dict | None = None) -> dict:
    if model not in MODEL_HPARAMS:
        raise ValueError(f"Unknown model '{model}'. Choose from {list(MODEL_HPARAMS)}")
    hp = dict(MODEL_HPARAMS[model])
    for k, v in (overrides or {}).items():
        if k in hp and v is not None:
            hp[k] = _cast(v, hp[k])
    return hp


def build_model(model: str, hparams: dict, seed: int) -> Pipeline:
    if model == "logreg":
        est = LogisticRegression(random_state=seed, **hparams)
    elif model == "random_forest":
        est = RandomForestClassifier(random_state=seed, n_jobs=1, **hparams)
    elif model == "gradient_boosting":
        est = GradientBoostingClassifier(random_state=seed, **hparams)
    elif model == "knn":
        est = KNeighborsClassifier(**hparams)
    else:
        raise ValueError(model)
    # same scaler for all models
    return Pipeline([("scaler", StandardScaler()), ("clf", est)])


def git_info() -> dict[str, str]:
    def _git(*args: str) -> str:
        return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL, text=True).strip()

    try:
        commit = _git("rev-parse", "HEAD")
        dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
        branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    except Exception:
        return {"git_commit": "unknown", "git_dirty": "unknown", "git_branch": "unknown"}
    return {"git_commit": commit, "git_dirty": str(dirty).lower(), "git_branch": branch}
