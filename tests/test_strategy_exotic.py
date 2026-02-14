import unittest

import pandas as pd

from keiba_model.strategy import attach_payouts, generate_exotic_tickets


class ExoticStrategyTests(unittest.TestCase):
    def test_trio_ticket_key_is_numeric_sorted(self):
        scored = pd.DataFrame(
            {
                "race_id": [1, 1, 1],
                "horse_id": [12, 4, 7],
                "pred_prob": [0.5, 0.3, 0.2],
            }
        )
        tickets = generate_exotic_tickets(scored, bet_type="trio", top_k_per_race=3, min_joint_prob=0.0, unit_bet=100)
        self.assertEqual(tickets.iloc[0]["ticket_key"], "4-7-12")

    def test_attach_payouts_duplicate_keys_raise(self):
        tickets = pd.DataFrame(
            {
                "race_id": [1],
                "bet_type": ["trio"],
                "ticket_key": ["1-2-3"],
                "joint_prob": [0.1],
                "stake": [100],
            }
        )
        payouts = pd.DataFrame(
            {
                "race_id": [1, 1],
                "bet_type": ["trio", "trio"],
                "ticket_key": ["1-2-3", "1-2-3"],
                "payout_per_100": [12340, 12000],
            }
        )
        with self.assertRaises(ValueError):
            attach_payouts(tickets, payouts)

    def test_generate_trio_tickets(self):
        scored = pd.DataFrame(
            {
                "race_id": [1, 1, 1, 1],
                "horse_id": [1, 2, 3, 4],
                "pred_prob": [0.4, 0.3, 0.2, 0.1],
            }
        )
        tickets = generate_exotic_tickets(scored, bet_type="trio", top_k_per_race=4, min_joint_prob=0.0, unit_bet=100)
        self.assertIn("ticket_key", tickets.columns)
        self.assertTrue((tickets["stake"] == 100).all())

    def test_attach_payouts_sets_is_win(self):
        tickets = pd.DataFrame(
            {
                "race_id": [1, 1],
                "bet_type": ["trio", "trio"],
                "ticket_key": ["1-2-3", "1-2-4"],
                "joint_prob": [0.1, 0.05],
                "stake": [100, 100],
            }
        )
        payouts = pd.DataFrame(
            {
                "race_id": [1],
                "bet_type": ["trio"],
                "ticket_key": ["1-2-3"],
                "payout_per_100": [12340],
            }
        )
        merged = attach_payouts(tickets, payouts)
        self.assertListEqual(merged["is_win"].tolist(), [1, 0])
        self.assertListEqual(merged["payout_per_100"].tolist(), [12340.0, 0.0])


if __name__ == "__main__":
    unittest.main()
