"""Checks for research/experiments/sharpness_floor.py."""

import unittest


from research.experiments.sharpness_floor import boundary, floor_job, make, predicted_floor, top_eigenvalue


class SharpnessFloorTests(unittest.TestCase):
    def test_boundary_matches_known_values(self):
        self.assertAlmostEqual(boundary(1, 0.0), 1.0, places=3)
        self.assertAlmostEqual(boundary(2, 0.0), 0.618, places=3)  # golden-ratio boundary
        self.assertLess(boundary(16, 0.9), boundary(16, 0.0))

    def test_predicted_floor_is_monotone(self):
        self.assertEqual(predicted_floor(0.5, 0.0), 2)
        self.assertEqual(predicted_floor(5.0, 0.0), 0)
        self.assertEqual(predicted_floor(1e-6, 0.0), 64)

    def test_power_iteration_on_a_known_quadratic(self):
        trainer = make("sgd0", 1.0, 0)
        # replace the loss with a quadratic whose top eigenvalue is 3
        params = list(trainer.net.parameters())
        trainer.full_loss = lambda views, x, y: 1.5 * (params[0] ** 2).sum() + 0.5 * sum((p ** 2).sum() for p in params[1:])
        value, _ = top_eigenvalue(trainer)
        self.assertAlmostEqual(value, 3.0, places=3)

    def test_sgd0_has_no_momentum_in_floor_jobs(self):
        trainer = make("sgd0", 1.0, 0)
        self.assertEqual(trainer.opt.param_groups[0]["momentum"], 0.0)
        import research.experiments.staleness_floor as sf
        sf_epochs, sf.FLOOR_EPOCHS = sf.FLOOR_EPOCHS, 1
        try:
            row = floor_job(("sgd0", 1.0, "A_cont_w1", 0, "stale", 2))
        finally:
            sf.FLOOR_EPOCHS = sf_epochs
        self.assertEqual(row["optimizer"], "sgd0")


if __name__ == "__main__":
    unittest.main()
