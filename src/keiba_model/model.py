from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline

from .features import build_preprocessor


@dataclass
class ModelResult:
    pipeline: Pipeline
    metrics: dict


def train_win_model(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    numerical: List[str],
    categorical: List[str],
    seed: int,
) -> ModelResult:
    features = numerical + categorical
    preprocessor = build_preprocessor(numerical=numerical, categorical=categorical)

    base = LogisticRegression(max_iter=500, random_state=seed)
    calibrated = CalibratedClassifierCV(base, method="sigmoid", cv=3)

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", calibrated),
        ]
    )

    pipeline.fit(train_df[features], train_df["is_win"])
    prob = pipeline.predict_proba(valid_df[features])[:, 1]

    is_win_valid = valid_df["is_win"]
    auc = float("nan")
    if is_win_valid.nunique(dropna=True) >= 2:
        auc = float(roc_auc_score(is_win_valid, prob))

    metrics = {
        "logloss": float(log_loss(is_win_valid, prob, labels=[0, 1])),
        "brier": float(brier_score_loss(is_win_valid, prob)),
        "auc": auc,
        "avg_prob": float(np.mean(prob)),
    }
    return ModelResult(pipeline=pipeline, metrics=metrics)


def save_model(pipeline: Pipeline, out_dir: str, file_name: str = "win_model.joblib") -> Path:
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    model_path = path / file_name
    joblib.dump(pipeline, model_path)
    return model_path
