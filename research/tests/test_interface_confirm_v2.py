"""Harness checks for research/reports/interface_confirm_v2_protocol.md."""

import unittest

from research.experiments import interface_confirm_v2 as v2


def row(family, fit, **kw):
    return {"family": family, "fit_epoch": fit, **kw}


class InterfaceConfirmV2Tests(unittest.TestCase):
    def test_power_check_picks_largest_passing_delay(self):
        rows = []
        for task in ("adder", "automaton"):
            for d, n_fit in [(2, 5), (4, 4), (8, 3 if task == "adder" else 5)]:
                rows += [row("CAL1", 10 if k < n_fit else None, task=task, delay=d) for k in range(5)]
        for opt, passing in [("adam", 16), ("sgd", 2), ("muon", None)]:
            for d in v2.H2_CAL_DELAYS:
                n_fit = 10 if passing is not None and d <= passing else 3
                rows += [row("CAL2", 10 if k < n_fit else None, optimizer=opt, delay=d) for k in range(10)]
        chosen = v2.choose_delays(rows)
        self.assertEqual(chosen["H1_delay"], 4)
        self.assertEqual(chosen["H2_delays"], {"adam": 16, "sgd": 2, "muon": None})

    def test_calibration_runs_only_the_stronger_or_weaker_arm_alone(self):
        jobs = v2.calibration_jobs()
        self.assertTrue(all(j[4] == "stale" for j in jobs))
        self.assertEqual({j[2] for j in jobs if j[0] == "CAL1"}, {"G_cont_w64"})
        self.assertTrue(all(j[3] in v2.CALIBRATION_SEEDS for j in jobs))


if __name__ == "__main__":
    unittest.main()
