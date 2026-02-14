from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class FeatureArtifacts:
    feature_columns: List[str]
    target_column: str


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["horse_id", "race_date"]).copy()
    out["prev_finish_mean_5"] = (
        out.groupby("horse_id")["finish_position"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )
    out["prev_win_rate_10"] = (
        out.groupby("horse_id")["is_win"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    )
    return out


def build_preprocessor(numerical: List[str], categorical: List[str]) -> ColumnTransformer:
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numerical),
            ("cat", cat_pipe, categorical),
        ]
    )
