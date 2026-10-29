"""Checks for research/experiments/calibrated_floor.py."""

import unittest

from research.experiments.calibrated_floor import (FROZEN_RATIO, calibrate, calibration_data, floor_from_ratios,
                                                   lr_scaling, within_2x)


class CalibratedFloorTests(unittest.TestCase):
    def test_floor_stops_at_first_failure_and_nan_fails(self):
        ratios = {"1": 1.0, "2": 3.0, "4": 50.0, "8": 1.0, "16": 1.0, "32": 1.0, "64": 1.0}
        self.assertEqual(floor_from_ratios(ratios, 32), 2)
        self.assertEqual(floor_from_ratios({**ratios, "2": float("nan")}, 32), 1)
        self.assertEqual(floor_from_ratios(None, 32), 0)

    def test_within_2x(self):
        self.assertTrue(within_2x(8, 4))
        self.assertFalse(within_2x(16, 4))
        self.assertTrue(within_2x(0, 0))
        self.assertFalse(within_2x(0, 4))

    def test_calibration_reproduces_frozen_ratio(self):
        ratio, _ = calibrate(*calibration_data())
        self.assertEqual(ratio, FROZEN_RATIO)

    def test_lr_scaling_uses_nearest_nonzero_setting(self):
        cal = {("sgd0", 1.0): 8, ("sgd0", 3.0): 0}
        self.assertEqual(lr_scaling("sgd0", 2.0, cal), 4.0)
        self.assertEqual(lr_scaling("sgd0", 0.5, cal), 16.0)


if __name__ == "__main__":
    unittest.main()
