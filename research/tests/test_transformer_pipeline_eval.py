"""Checks for research/experiments/transformer_pipeline_eval.py."""

import unittest

from research.experiments.transformer_pipeline import DELAYS, LR, SECONDARY_LR
from research.experiments.transformer_pipeline_eval import evaluate


def rows_for(stale_excess, perm_excess, practical, secondary_stale):
    rows = []
    for s in (0, 1):
        for lr, base in ((LR, 1.8), (SECONDARY_LR, 1.7)):
            rows.append({"method": "fresh", "delay": 0, "lr": lr, "seed": s, "final_val_loss": base})
        rows.append({"method": "basis_rotation_fresh", "delay": 0, "lr": LR, "seed": s, "final_val_loss": 1.7})
        for d in DELAYS:
            rows.append({"method": "stale", "delay": d, "lr": LR, "seed": s, "final_val_loss": 1.8 + stale_excess[d]})
            rows.append({"method": "permuted", "delay": d, "lr": LR, "seed": s, "final_val_loss": 1.8 + perm_excess[d]})
            for m, f in practical.items():
                loss = 1.8 + stale_excess[d] - f * (stale_excess[d] - perm_excess[d])
                rows.append({"method": m, "delay": d, "lr": LR, "seed": s, "final_val_loss": loss})
            for m in ("stale", "permuted"):
                ex = secondary_stale[d] if m == "stale" else 0.0
                rows.append({"method": m, "delay": d, "lr": SECONDARY_LR, "seed": s, "final_val_loss": 1.7 + ex})
    return rows


class EvalTests(unittest.TestCase):
    def test_verdicts(self):
        stale = {3: 0.0, 6: 0.01, 12: 0.1, 24: 0.4, 48: 0.7}
        perm = {d: 0.0 for d in DELAYS}
        secondary = {3: 0.0, 6: 0.2, 12: 0.5, 24: 0.8, 48: 0.9}
        out = evaluate(rows_for(stale, perm, {"pipemare_lr": 0.3, "weight_predict": -0.3, "basis_rotation": 0.4},
                                secondary))
        self.assertEqual(out["harmful_delays"], [12, 24, 48])
        self.assertEqual(out["floors"], {str(LR): 6, str(SECONDARY_LR): 3})
        self.assertEqual(out["verdicts"]["P1"], "held")
        self.assertEqual(out["verdicts"]["P2"], "held")
        self.assertEqual(out["verdicts"]["P3"], "held")
        self.assertAlmostEqual(out["recovery"]["48"]["basis_rotation"], 0.4)
        out = evaluate(rows_for(stale, perm, {"pipemare_lr": 0.6, "weight_predict": 0, "basis_rotation": 0}, stale))
        self.assertEqual(out["verdicts"]["P2"], "killed")
        self.assertEqual(out["verdicts"]["P3"], "killed")


if __name__ == "__main__":
    unittest.main()
