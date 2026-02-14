# JRA予想モデル実装テンプレート（Python 3.8）

ユーザー要件の「確率モデル→買い目自動生成→バックテスト」を、そのまま実装できる最小構成で用意しています。

## ディレクトリ
- `docs/IMPLEMENTATION_TEMPLATE.md`: 設計図
- `config.yaml`: 実験設定
- `src/keiba_model/`: 実装コード

## セットアップ
```bash
python3.8 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## 実行
```bash
python -m keiba_model.pipeline --config config.yaml
```

## 入力CSV
- `data.raw_csv`: 馬単位特徴量データ（`win_odds` または `payout_per_100` の少なくとも一方が必要）。
- `data.payout_csv`: 3連複/3連単の払戻テーブル（`bet_type`, `ticket_key`, `payout_per_100` を含む）。

## bet_type
- `win`: 単勝（edge + Kelly）
- `trio`: 3連複チケット生成（上位馬組合せ） + 払戻JOIN
- `trifecta`: 3連単チケット生成（上位馬順列） + 払戻JOIN

## 払戻データの推奨形式
- 推奨: `payout_per_100`（100円あたり払戻金）
- fallback: `win_odds`（倍率）

バックテストは `payout_per_100` を優先使用し、`recovery_rate`（回収率）と `profit_roi`（利益率）を分けて出力します。
- `race_id` が無い入力では `race_hit_rate` は `ticket_hit_rate` と同値になり、`race_mode="ticket_level"` を返します。
