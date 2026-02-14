from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


BASE_REQUIRED_COLUMNS = {
    "race_id",
    "race_date",
    "horse_id",
    "horse_name",
    "finish_position",
    "track",
    "surface",
    "weather",
    "distance",
}

ODDS_COLUMNS = {"win_odds", "payout_per_100"}


@dataclass
class Dataset:
    frame: pd.DataFrame


def load_raw_dataset(csv_path: str) -> Dataset:
    path = Path(csv_path)
    df = pd.read_csv(path)

    missing = BASE_REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"入力CSVに必要な列が不足: {sorted(missing)}")

    if not (set(df.columns) & ODDS_COLUMNS):
        raise ValueError("入力CSVには `win_odds` または `payout_per_100` の少なくとも1つが必要です。")

    df["race_date"] = pd.to_datetime(df["race_date"])
    df["is_win"] = (df["finish_position"] == 1).astype(int)
    return Dataset(frame=df)


def split_by_date(df: pd.DataFrame, train_end_date: str, valid_end_date: str, test_start_date: str):
    train = df[df["race_date"] <= pd.Timestamp(train_end_date)].copy()
    valid = df[(df["race_date"] > pd.Timestamp(train_end_date)) & (df["race_date"] <= pd.Timestamp(valid_end_date))].copy()
    test = df[df["race_date"] >= pd.Timestamp(test_start_date)].copy()
    return train, valid, test
