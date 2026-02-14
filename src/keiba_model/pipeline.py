from __future__ import annotations

import argparse
import json

import pandas as pd

from .backtest import run_backtest
from .config import load_config
from .data import load_raw_dataset, split_by_date
from .features import add_history_features
from .model import save_model, train_win_model
from .strategy import attach_payouts, generate_exotic_tickets, generate_win_tickets


def _run_win_strategy(cfg: dict, scored_df: pd.DataFrame) -> pd.DataFrame:
    return generate_win_tickets(
        scored_df=scored_df,
        min_edge=cfg["min_edge"],
        min_model_prob=cfg["min_model_prob"],
        bankroll=cfg["bankroll"],
        unit_bet=cfg["unit_bet"],
        max_race_exposure=cfg["max_race_exposure"],
        kelly_fraction=cfg["kelly_fraction"],
    )


def _run_exotic_strategy(cfg: dict, scored_df: pd.DataFrame, payout_csv: str) -> pd.DataFrame:
    tickets = generate_exotic_tickets(
        scored_df=scored_df,
        bet_type=cfg["bet_type"],
        top_k_per_race=int(cfg.get("top_k_per_race", 6)),
        min_joint_prob=float(cfg.get("min_joint_prob", 0.001)),
        unit_bet=int(cfg["unit_bet"]),
    )
    payout_df = pd.read_csv(payout_csv)
    return attach_payouts(tickets, payout_df)


def main() -> None:
    parser = argparse.ArgumentParser(description="JRA予想モデル実行")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw = load_raw_dataset(cfg["data"]["raw_csv"]).frame
    feat = add_history_features(raw)

    numerical = cfg["features"]["numerical"] + ["prev_finish_mean_5", "prev_win_rate_10"]
    categorical = cfg["features"]["categorical"]

    train_df, valid_df, test_df = split_by_date(
        feat,
        cfg["training"]["train_end_date"],
        cfg["training"]["valid_end_date"],
        cfg["training"]["test_start_date"],
    )

    result = train_win_model(train_df, valid_df, numerical, categorical, seed=cfg.seed)
    model_path = save_model(result.pipeline, cfg["data"]["model_dir"])

    features = numerical + categorical
    test_df = test_df.copy()
    test_df["pred_prob"] = result.pipeline.predict_proba(test_df[features])[:, 1]

    strategy_cfg = cfg["strategy"]
    bet_type = strategy_cfg.get("bet_type", "win")

    if bet_type == "win":
        tickets = _run_win_strategy(strategy_cfg, test_df)
    elif bet_type in {"trio", "trifecta"}:
        payout_csv = cfg["data"].get("payout_csv")
        if not payout_csv:
            raise ValueError("trio/trifecta では config.data.payout_csv の指定が必要です。")
        tickets = _run_exotic_strategy(strategy_cfg, test_df, payout_csv)
    else:
        raise ValueError(f"未対応の bet_type: {bet_type}")

    summary = run_backtest(tickets)

    print(json.dumps({"validation_metrics": result.metrics, "model_path": str(model_path)}, ensure_ascii=False, indent=2))
    print(json.dumps(summary.__dict__, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
