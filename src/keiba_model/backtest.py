from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class BacktestSummary:
    total_bet: float
    total_return: float
    recovery_rate: float
    profit_roi: float
    ticket_hit_rate: float
    race_hit_rate: float
    tickets: int
    races: int
    payout_mode: str
    race_mode: str


def run_backtest(tickets_df: pd.DataFrame) -> BacktestSummary:
    """買い目DataFrameから回収指標を計算する。

    必須列:
      - stake: 賭け金（円）
      - is_win: 的中フラグ（0/1想定。欠損・異常値は0/1へ正規化）

    払戻の優先順位:
      1) payout_per_100: JRAの100円あたり払戻金（元返し込み）
      2) win_odds: 倍率データ（fallback）
    """

    if tickets_df.empty:
        return BacktestSummary(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, "empty", "empty")

    frame = tickets_df.copy()
    required = {"stake", "is_win"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"バックテスト入力に必要な列が不足: {sorted(missing)}")

    # CSV起因の文字列/欠損混入に備えて数値化。
    frame["stake"] = pd.to_numeric(frame["stake"], errors="coerce").fillna(0.0).astype(float)

    # is_winは0/1に正規化（NaN, 2, -1, 文字列混在への防御）。
    is_win_numeric = pd.to_numeric(frame["is_win"], errors="coerce").fillna(0.0)
    win_mask = is_win_numeric.clip(lower=0, upper=1).astype(int)

    if "payout_per_100" in frame.columns:
        frame["payout_per_100"] = pd.to_numeric(frame["payout_per_100"], errors="coerce").fillna(0.0).astype(float)
        # JRA払い戻し金（100円あたり◯円）を使うのが最も安全。
        frame["payoff"] = (frame["stake"] / 100.0) * frame["payout_per_100"] * win_mask
        payout_mode = "payout_per_100"
    elif "win_odds" in frame.columns:
        frame["win_odds"] = pd.to_numeric(frame["win_odds"], errors="coerce").fillna(0.0).astype(float)
        # fallback: 倍率オッズ
        frame["payoff"] = frame["stake"] * frame["win_odds"] * win_mask
        payout_mode = "win_odds_multiplier"
    else:
        raise ValueError("`payout_per_100` または `win_odds` のいずれかが必要です。")

    total_bet = float(frame["stake"].sum())
    total_return = float(frame["payoff"].sum())

    recovery_rate = (total_return / total_bet) if total_bet > 0 else 0.0
    profit_roi = ((total_return - total_bet) / total_bet) if total_bet > 0 else 0.0

    ticket_hit_rate = float((win_mask == 1).mean())

    if "race_id" in frame.columns:
        race_hit = win_mask.groupby(frame["race_id"], observed=True).max()
        race_hit_rate = float(race_hit.mean())
        races = int(race_hit.shape[0])
        race_mode = "race_level"
    else:
        race_hit_rate = ticket_hit_rate
        races = 0
        race_mode = "ticket_level"

    return BacktestSummary(
        total_bet=total_bet,
        total_return=total_return,
        recovery_rate=recovery_rate,
        profit_roi=profit_roi,
        ticket_hit_rate=ticket_hit_rate,
        race_hit_rate=race_hit_rate,
        tickets=int(len(frame)),
        races=races,
        payout_mode=payout_mode,
        race_mode=race_mode,
    )
