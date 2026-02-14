from __future__ import annotations

from itertools import combinations, permutations
from typing import Iterable, List

import pandas as pd


def _normalize_probs(values: pd.Series) -> pd.Series:
    total = float(values.sum())
    if total <= 0:
        return pd.Series([0.0] * len(values), index=values.index)
    return values / total


def _make_ticket_key(horse_ids: Iterable[str], bet_type: str) -> str:
    ids = [str(h) for h in horse_ids]
    if bet_type == "trio":
        # 可能なら数値順でソートし、不可なら文字列順で安定化。
        try:
            ids = [str(h) for h in sorted((int(x) for x in ids))]
        except ValueError:
            ids = sorted(ids)
    return "-".join(ids)


def generate_exotic_tickets(
    scored_df: pd.DataFrame,
    bet_type: str,
    top_k_per_race: int,
    min_joint_prob: float,
    unit_bet: int,
) -> pd.DataFrame:
    """3連系（3連複/3連単）の候補チケットを作る。

    Parameters
    ----------
    scored_df: race_id, horse_id, pred_prob を含むDataFrame
    bet_type: "trio"（3連複） or "trifecta"（3連単）
    """

    if bet_type not in {"trio", "trifecta"}:
        raise ValueError("bet_type は 'trio' または 'trifecta' を指定してください。")

    required = {"race_id", "horse_id", "pred_prob"}
    missing = required - set(scored_df.columns)
    if missing:
        raise ValueError(f"3連系チケット生成に必要な列が不足: {sorted(missing)}")

    rows: List[dict] = []
    for race_id, race_df in scored_df.groupby("race_id", observed=True):
        cand = race_df.sort_values("pred_prob", ascending=False).head(top_k_per_race).copy()
        cand["pred_prob"] = pd.to_numeric(cand["pred_prob"], errors="coerce").fillna(0.0)
        cand["norm_prob"] = _normalize_probs(cand["pred_prob"])

        if bet_type == "trio":
            patterns = combinations(cand["horse_id"].tolist(), 3)
        else:
            patterns = permutations(cand["horse_id"].tolist(), 3)

        prob_map = cand.set_index("horse_id")["norm_prob"].to_dict()
        for ticket_horses in patterns:
            joint_prob = 1.0
            for hid in ticket_horses:
                joint_prob *= float(prob_map.get(hid, 0.0))
            if joint_prob < min_joint_prob:
                continue

            rows.append(
                {
                    "race_id": race_id,
                    "bet_type": bet_type,
                    "ticket_key": _make_ticket_key(ticket_horses, bet_type),
                    "joint_prob": joint_prob,
                    "stake": float(unit_bet),
                }
            )

    if not rows:
        return pd.DataFrame(columns=["race_id", "bet_type", "ticket_key", "joint_prob", "stake"])
    return pd.DataFrame(rows)


def attach_payouts(
    tickets_df: pd.DataFrame,
    payout_df: pd.DataFrame,
    *,
    payout_col: str = "payout_per_100",
) -> pd.DataFrame:
    """チケットと払戻テーブルを結合し、is_win/payout_per_100を付与する。"""

    required_tickets = {"race_id", "bet_type", "ticket_key", "stake"}
    required_payouts = {"race_id", "bet_type", "ticket_key", payout_col}
    mt = required_tickets - set(tickets_df.columns)
    mp = required_payouts - set(payout_df.columns)
    if mt:
        raise ValueError(f"tickets_df に必要な列が不足: {sorted(mt)}")
    if mp:
        raise ValueError(f"payout_df に必要な列が不足: {sorted(mp)}")

    key_cols = ["race_id", "bet_type", "ticket_key"]
    dup_mask = payout_df.duplicated(subset=key_cols, keep=False)
    if dup_mask.any():
        dups = payout_df.loc[dup_mask, key_cols].drop_duplicates().head(5).to_dict(orient="records")
        raise ValueError(f"payout_df に重複キーがあります: {dups}")

    merged = tickets_df.merge(
        payout_df[["race_id", "bet_type", "ticket_key", payout_col]],
        on=["race_id", "bet_type", "ticket_key"],
        how="left",
    )
    merged["payout_per_100"] = pd.to_numeric(merged[payout_col], errors="coerce").fillna(0.0)
    merged["is_win"] = (merged["payout_per_100"] > 0).astype(int)
    return merged


def generate_win_tickets(
    scored_df: pd.DataFrame,
    min_edge: float,
    min_model_prob: float,
    bankroll: float,
    unit_bet: int,
    max_race_exposure: float,
    kelly_fraction: float,
) -> pd.DataFrame:
    """期待値条件とケリー基準で単勝買い目を生成する。"""

    df = scored_df.copy()

    if "payout_per_100" in df.columns:
        odds_multiplier = df["payout_per_100"] / 100.0
    elif "win_odds" in df.columns:
        odds_multiplier = df["win_odds"]
    else:
        raise ValueError("買い目生成には `payout_per_100` または `win_odds` が必要です。")

    df["edge"] = df["pred_prob"] * odds_multiplier - 1.0
    df = df[(df["pred_prob"] >= min_model_prob) & (df["edge"] >= min_edge)].copy()
    if df.empty:
        return df

    b = (odds_multiplier.loc[df.index] - 1.0).clip(lower=0.01)
    q = 1.0 - df["pred_prob"]
    kelly = ((b * df["pred_prob"] - q) / b).clip(lower=0.0)
    df["stake_raw"] = bankroll * kelly_fraction * kelly

    race_cap = bankroll * max_race_exposure
    df["stake"] = (df["stake_raw"].clip(upper=race_cap) / unit_bet).round() * unit_bet
    df = df[df["stake"] >= unit_bet].copy()
    return df
