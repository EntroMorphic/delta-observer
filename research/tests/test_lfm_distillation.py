import unittest

import torch

from research.lfm.distillation import (
    TrajectoryAlignment,
    conditional_assignment_loss,
    conditional_contrast_loss,
    sparse_topk_kl,
)


class LFMDistillationTests(unittest.TestCase):
    def test_sparse_kl_is_zero_for_identical_distribution(self):
        torch.manual_seed(29)
        logits = torch.randn(2, 5, 31)
        values, indices = torch.topk(logits, 7, dim=-1)
        normalizer = torch.logsumexp(logits, dim=-1)
        loss = sparse_topk_kl(logits, indices, values, normalizer)
        self.assertLess(abs(float(loss)), 1e-6)

    def test_sparse_kl_detects_different_distribution(self):
        torch.manual_seed(31)
        teacher = torch.randn(2, 4, 23)
        student = torch.randn(2, 4, 23)
        values, indices = torch.topk(teacher, 5, dim=-1)
        loss = sparse_topk_kl(student, indices, values, torch.logsumexp(teacher, dim=-1))
        self.assertGreater(float(loss.detach()), 0.0)

    def test_sparse_kl_mask_excludes_tokens(self):
        torch.manual_seed(33)
        teacher = torch.randn(1, 3, 19)
        student = teacher.clone()
        student[:, 2] = torch.randn(1, 19) * 5
        values, indices = torch.topk(teacher, 5, dim=-1)
        mask = torch.tensor([[1.0, 1.0, 0.0]])
        loss = sparse_topk_kl(
            student, indices, values, torch.logsumexp(teacher, dim=-1), mask=mask
        )
        self.assertLess(abs(float(loss)), 1e-6)

    def test_conditional_contrast_penalizes_collapsed_instruction_rows(self):
        student = torch.zeros(2, 4, 11, requires_grad=True)
        teacher = torch.zeros(2, 4, 11)
        teacher[0, 2, 3] = 8.0
        teacher[1, 1, 7] = 8.0
        values, indices = torch.topk(teacher, 4, dim=-1)
        mask = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]], dtype=torch.float32)
        loss, metrics = conditional_contrast_loss(
            student,
            indices,
            values,
            torch.logsumexp(teacher, dim=-1),
            mask,
        )
        self.assertGreater(float(loss.detach()), 0.0)
        self.assertEqual(metrics["pairs"], 1)
        self.assertGreater(metrics["teacher_js"], metrics["student_js"])
        loss.backward()
        self.assertIsNotNone(student.grad)

    def test_conditional_contrast_ignores_unmasked_raw_rows(self):
        logits = torch.randn(2, 3, 13)
        values, indices = torch.topk(logits, 4, dim=-1)
        loss, metrics = conditional_contrast_loss(
            logits,
            indices,
            values,
            torch.logsumexp(logits, dim=-1),
            torch.ones(2, 3),
        )
        self.assertEqual(float(loss), 0.0)
        self.assertEqual(metrics["pairs"], 0)

    def test_assignment_contrast_rewards_correct_teacher_pairing(self):
        teacher = torch.zeros(2, 4, 11)
        teacher[0, 2, 3] = 8.0
        teacher[1, 1, 7] = 8.0
        values, indices = torch.topk(teacher, 4, dim=-1)
        mask = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]], dtype=torch.float32)
        collapsed = torch.zeros(2, 4, 11, requires_grad=True)
        separated = teacher.clone().requires_grad_(True)
        collapsed_loss, collapsed_metrics = conditional_assignment_loss(
            collapsed, indices, values, torch.logsumexp(teacher, dim=-1), mask
        )
        separated_loss, separated_metrics = conditional_assignment_loss(
            separated, indices, values, torch.logsumexp(teacher, dim=-1), mask
        )
        self.assertLess(float(separated_loss.detach()), float(collapsed_loss.detach()))
        self.assertGreater(
            separated_metrics["assignment_accuracy"],
            collapsed_metrics["assignment_accuracy"],
        )
        collapsed_loss.backward()
        self.assertIsNotNone(collapsed.grad)

    def test_trajectory_alignment_shapes_and_gradients(self):
        torch.manual_seed(37)
        states = tuple(torch.randn(2, 6, 12, requires_grad=True) for _ in range(4))
        teacher = torch.randn(2, 7, 6, 8)
        alignment = TrajectoryAlignment(12, 8)
        losses = alignment(states, teacher)
        total = losses["state"] + losses["delta"]
        total.backward()
        self.assertTrue(all(state.grad is not None for state in states))
        self.assertTrue(torch.isfinite(total))

    def test_trajectory_mask_shapes_and_gradients(self):
        torch.manual_seed(39)
        states = tuple(torch.randn(2, 5, 12, requires_grad=True) for _ in range(4))
        teacher = torch.randn(2, 7, 5, 8)
        mask = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 0, 0]], dtype=torch.float32)
        losses = TrajectoryAlignment(12, 8)(states, teacher, mask=mask)
        (losses["state"] + losses["delta"]).backward()
        self.assertTrue(all(state.grad is not None for state in states))


if __name__ == "__main__":
    unittest.main()
