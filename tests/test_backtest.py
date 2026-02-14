import unittest

import pandas as pd

from keiba_model.backtest import run_backtest


class BacktestTests(unittest.TestCase):
    def test_payout_per_100_priority_and_rates(self):
        df = pd.DataFrame(
            {
                "race_id": [1, 1, 2],
                "stake": [100, 200, 100],
                "is_win": [1, 0, 0],
                "payout_per_100": [12340, 0, 0],
                "win_odds": [2.0, 3.0, 4.0],
            }
        )

        summary = run_backtest(df)
        self.assertEqual(summary.total_bet, 400.0)
        self.assertEqual(summary.total_return, 12340.0)
        self.assertAlmostEqual(summary.recovery_rate, 12340.0 / 400.0)
        self.assertAlmostEqual(summary.profit_roi, (12340.0 - 400.0) / 400.0)
        self.assertAlmostEqual(summary.ticket_hit_rate, 1 / 3)
        self.assertAlmostEqual(summary.race_hit_rate, 0.5)
        self.assertEqual(summary.payout_mode, "payout_per_100")
        self.assertEqual(summary.race_mode, "race_level")

    def test_win_odds_fallback(self):
        df = pd.DataFrame(
            {
                "race_id": [1, 2],
                "stake": [100, 100],
                "is_win": [1, 0],
                "win_odds": [2.5, 10.0],
            }
        )
        summary = run_backtest(df)
        self.assertEqual(summary.total_return, 250.0)
        self.assertEqual(summary.payout_mode, "win_odds_multiplier")

    def test_safe_numeric_conversion_and_is_win_sanitize(self):
        df = pd.DataFrame(
            {
                "stake": ["100", "bad", None, 100],
                "is_win": ["1", 2, -1, None],
                "win_odds": ["2.0", "3.0", "bad", None],
            }
        )
        summary = run_backtest(df)

        # stake: [100,0,0,100], win_mask: [1,1,0,0], win_odds:[2,3,0,0]
        self.assertEqual(summary.total_bet, 200.0)
        self.assertEqual(summary.total_return, 200.0)
        self.assertAlmostEqual(summary.ticket_hit_rate, 0.5)
        self.assertAlmostEqual(summary.race_hit_rate, 0.5)
        self.assertEqual(summary.race_mode, "ticket_level")


if __name__ == "__main__":
    unittest.main()
