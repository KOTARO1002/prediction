# 競馬予想モデル設計図（Python 3.8）

## 1. 目的
JRA過去データを使って、以下を一気通貫で実装する。
1. 確率モデル（各馬の1着確率）
2. 買い目自動生成（単勝 / 3連複 / 3連単）
3. バックテスト（ROI・的中率）

## 2. 入力データ仕様（最低限）
### 2.1 馬単位CSV（`data.raw_csv`）
- race_id, race_date, horse_id, horse_name
- finish_position, payout_per_100（推奨）, win_odds（fallback）
- track, surface, weather, distance
- horse_age, carried_weight, horse_weight, popularity
- jockey, trainer, last3f_time, days_from_last_race, draw, frame_no

### 2.2 払戻CSV（`data.payout_csv` / 3連系で必須）
- race_id
- bet_type（`trio` or `trifecta`）
- ticket_key
  - 3連複: 昇順で `horse_id` を `-` 連結（例: `4-7-12`）
  - 3連単: 着順通りに `horse_id` を `-` 連結（例: `12-4-7`）
- payout_per_100（100円あたり払戻金）

## 3. 前処理ルール
- race_dateをdatetime化
- is_win=(finish_position==1)
- horse_idごとに時系列ソートしてリークなしの履歴特徴量を生成
  - prev_finish_mean_5
  - prev_win_rate_10

## 4. 分割戦略
- train: train_end_date以前
- valid: (train_end_date, valid_end_date]
- test: test_start_date以降

## 5. モデル
- Base: LogisticRegression
- 校正: CalibratedClassifierCV(sigmoid)
- 入力: 数値 + カテゴリ（OHE）
- 指標: logloss / brier / auc

## 6. 買い目生成
### 6.1 単勝（`bet_type: win`）
- 単勝期待値 edge = pred_prob * odds_multiplier - 1
  - odds_multiplier = payout_per_100 / 100（推奨）
  - payout_per_100 が無い場合のみ win_odds を使用
- 条件: pred_prob >= min_model_prob かつ edge >= min_edge
- 賭け金:
  - kelly f* = (bp-q)/b
  - stake = bankroll * kelly_fraction * f*
  - 1レース上限=max_race_exposure*bankroll
  - unit_bet刻みに丸める

### 6.2 3連複/3連単（`bet_type: trio` / `trifecta`）
- 各レースで pred_prob 上位 `top_k_per_race` を候補化
- 3連複: 組合せ（順不同）
- 3連単: 順列（順序あり）
- joint_prob = 候補馬の正規化確率の積
- `min_joint_prob` 以上を採用し、まずは定額 `unit_bet` でチケット化
- `ticket_key` で払戻CSVとJOINして `payout_per_100` / `is_win` を付与

## 7. バックテスト
- 的中時 payoff（推奨） = (stake / 100) * payout_per_100
- fallback: payoff = stake * win_odds
- total_bet, total_return に加え、
  - recovery_rate = total_return / total_bet（回収率）
  - profit_roi = (total_return - total_bet) / total_bet（利益率）
  - ticket_hit_rate（券的中率）
  - race_hit_rate（レース的中率）
  - race_mode（race_id有無による集計モード）
  を集計

## 8. 実行コマンド
```bash
python -m keiba_model.pipeline --config config.yaml
```

## 9. 今後の拡張
- 3連系での賭け金最適化（確率温度調整 + 分散制約付きKelly）
- Optunaで閾値と特徴量選択最適化
- 時系列CV（月次ローリング）
- ドローダウン/破産確率の追加
