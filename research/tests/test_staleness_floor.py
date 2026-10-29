"""Harness checks for research/experiments/staleness_floor.py."""

import unittest

import research.experiments.staleness_floor as sf


class StalenessFloorTests(unittest.TestCase):
    def test_mechanism_twins_share_norm(self):
        saved = sf.MECHANISM_EPOCHS
        sf.MECHANISM_EPOCHS = 1
        try:
            rows = sf.mechanism_job(("adam", 1 / 300, 8, "A_cont_w1", 0))["records"]
        finally:
            sf.MECHANISM_EPOCHS = saved
        self.assertTrue(rows)
        for r in rows:
            self.assertAlmostEqual(r["real"]["norm"], r["permuted"]["norm"], places=6)

    def test_surface_job_reports_fit_or_none(self):
        saved = sf.FLOOR_EPOCHS
        sf.FLOOR_EPOCHS = 1
        try:
            row = sf.surface_job(("adam", 1 / 300, "A_cont_w1", 0, "stale", 2))
        finally:
            sf.FLOOR_EPOCHS = saved
        self.assertIn(row["fit_epoch"], (None, 1))


if __name__ == "__main__":
    unittest.main()
