import unittest

import torch

from research.lfm.initialize_teacher_embedding import align_embedding_geometry


class EmbeddingInitializationTests(unittest.TestCase):
    def test_alignment_has_expected_shape_and_scale(self):
        torch.manual_seed(43)
        teacher = torch.randn(64, 12)
        student = torch.randn(64, 6) * 0.2
        combined, metrics = align_embedding_geometry(teacher, student, seed=7, blend=1.0)
        self.assertEqual(combined.shape, student.shape)
        self.assertAlmostEqual(float(combined.std()), float(student.std()), places=5)
        self.assertEqual(metrics["projection_shape"], [12, 6])

    def test_zero_blend_preserves_student_exactly(self):
        torch.manual_seed(47)
        teacher = torch.randn(32, 10)
        student = torch.randn(32, 4)
        combined, _ = align_embedding_geometry(teacher, student, seed=11, blend=0.0)
        self.assertTrue(torch.equal(combined, student))


if __name__ == "__main__":
    unittest.main()
