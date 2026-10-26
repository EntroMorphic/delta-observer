"""Observer.D full-language behavior and trajectory objectives."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def sparse_topk_kl(
    student_logits: torch.Tensor,
    teacher_top_indices: torch.Tensor,
    teacher_top_values: torch.Tensor,
    teacher_logsumexp: torch.Tensor,
    mask: torch.Tensor | None = None,
    epsilon: float = 1e-7,
) -> torch.Tensor:
    """KL over teacher top-k tokens plus one aggregated tail bucket.

    Unlike renormalizing only the saved tokens, the tail bucket preserves the
    teacher's recorded top-k probability mass and penalizes overconfidence.
    """
    student_log_probs = F.log_softmax(student_logits.float(), dim=-1)
    student_top_log_probs = student_log_probs.gather(-1, teacher_top_indices.long())
    teacher_top_log_probs = teacher_top_values.float() - teacher_logsumexp.float().unsqueeze(-1)
    teacher_top_probs = teacher_top_log_probs.exp()

    teacher_tail = (1.0 - teacher_top_probs.sum(dim=-1)).clamp_min(epsilon)
    student_top_probs = student_top_log_probs.exp()
    student_tail = (1.0 - student_top_probs.sum(dim=-1)).clamp_min(epsilon)
    top_kl = teacher_top_probs * (teacher_top_log_probs - student_top_log_probs)
    tail_kl = teacher_tail * (teacher_tail.log() - student_tail.log())
    return masked_mean(top_kl.sum(dim=-1) + tail_kl, mask)


def jensen_shannon(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Jensen-Shannon divergence for normalized distributions."""
    midpoint = 0.5 * (first + second)
    return 0.5 * (
        torch.special.xlogy(first, first / midpoint.clamp_min(1e-30)).sum()
        + torch.special.xlogy(second, second / midpoint.clamp_min(1e-30)).sum()
    )


def sparse_teacher_pair_js(
    first_indices: torch.Tensor,
    first_values: torch.Tensor,
    first_logsumexp: torch.Tensor,
    second_indices: torch.Tensor,
    second_values: torch.Tensor,
    second_logsumexp: torch.Tensor,
) -> torch.Tensor:
    """Coarse teacher JS over the union of saved tokens plus one tail bucket."""
    union = torch.unique(torch.cat((first_indices.long(), second_indices.long())))
    first_top = torch.exp(first_values.float() - first_logsumexp.float())
    second_top = torch.exp(second_values.float() - second_logsumexp.float())
    first_union = (
        (union[:, None] == first_indices.long()[None, :]).float() * first_top[None, :]
    ).sum(dim=-1)
    second_union = (
        (union[:, None] == second_indices.long()[None, :]).float() * second_top[None, :]
    ).sum(dim=-1)
    first_distribution = torch.cat((
        first_union,
        (1.0 - first_top.sum()).clamp_min(1e-7).unsqueeze(0),
    ))
    second_distribution = torch.cat((
        second_union,
        (1.0 - second_top.sum()).clamp_min(1e-7).unsqueeze(0),
    ))
    return jensen_shannon(first_distribution, second_distribution)


def conditional_contrast_loss(
    student_logits: torch.Tensor,
    teacher_top_indices: torch.Tensor,
    teacher_top_values: torch.Tensor,
    teacher_logsumexp: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    """Match pairwise response-start separation on masked instruction rows."""
    instruction_rows = mask.sum(dim=-1) < mask.shape[-1]
    selected_rows = torch.nonzero(instruction_rows, as_tuple=False).flatten()
    if selected_rows.numel() < 2:
        zero = student_logits.new_zeros(())
        return zero, {"pairs": 0, "student_js": 0.0, "teacher_js": 0.0}
    first_positions = mask[selected_rows].float().argmax(dim=-1)
    selected_logits = student_logits[selected_rows, first_positions].float()
    selected_probabilities = F.softmax(selected_logits, dim=-1)
    losses = []
    student_values = []
    teacher_values = []
    for first in range(selected_rows.numel()):
        for second in range(first + 1, selected_rows.numel()):
            first_row = selected_rows[first]
            second_row = selected_rows[second]
            first_position = first_positions[first]
            second_position = first_positions[second]
            student_js = jensen_shannon(
                selected_probabilities[first], selected_probabilities[second]
            )
            teacher_js = sparse_teacher_pair_js(
                teacher_top_indices[first_row, first_position],
                teacher_top_values[first_row, first_position],
                teacher_logsumexp[first_row, first_position],
                teacher_top_indices[second_row, second_position],
                teacher_top_values[second_row, second_position],
                teacher_logsumexp[second_row, second_position],
            )
            losses.append((student_js - teacher_js.detach()).square())
            student_values.append(student_js.detach())
            teacher_values.append(teacher_js.detach())
    return torch.stack(losses).mean(), {
        "pairs": len(losses),
        "student_js": float(torch.stack(student_values).mean().item()),
        "teacher_js": float(torch.stack(teacher_values).mean().item()),
    }


def conditional_assignment_loss(
    student_logits: torch.Tensor,
    teacher_top_indices: torch.Tensor,
    teacher_top_values: torch.Tensor,
    teacher_logsumexp: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """Identify each response-start distribution among its batch peers."""
    instruction_rows = mask.sum(dim=-1) < mask.shape[-1]
    selected_rows = torch.nonzero(instruction_rows, as_tuple=False).flatten()
    if selected_rows.numel() < 2:
        zero = student_logits.new_zeros(())
        return zero, {"pairs": 0, "assignment_accuracy": 0.0, "margin": 0.0}
    first_positions = mask[selected_rows].float().argmax(dim=-1)
    selected_logits = student_logits[selected_rows, first_positions].float()
    student_log_probs = F.log_softmax(selected_logits, dim=-1)
    costs = []
    for teacher_index, teacher_row in enumerate(selected_rows):
        teacher_position = first_positions[teacher_index]
        indices = teacher_top_indices[teacher_row, teacher_position].long()
        teacher_log_probs = (
            teacher_top_values[teacher_row, teacher_position].float()
            - teacher_logsumexp[teacher_row, teacher_position].float()
        )
        teacher_probs = teacher_log_probs.exp()
        teacher_tail = (1.0 - teacher_probs.sum()).clamp_min(1e-7)
        student_top_log_probs = student_log_probs[:, indices]
        student_top_probs = student_top_log_probs.exp()
        student_tail = (1.0 - student_top_probs.sum(dim=-1)).clamp_min(1e-7)
        cross_entropy = -(
            (teacher_probs[None, :] * student_top_log_probs).sum(dim=-1)
            + teacher_tail * student_tail.log()
        )
        costs.append(cross_entropy)
    cost_matrix = torch.stack(costs)
    labels = torch.arange(selected_rows.numel(), device=student_logits.device)
    row_loss = F.cross_entropy(-cost_matrix / temperature, labels)
    column_loss = F.cross_entropy(-cost_matrix.T / temperature, labels)
    diagonal = cost_matrix.diag()
    off_diagonal = cost_matrix[~torch.eye(
        cost_matrix.shape[0], dtype=torch.bool, device=cost_matrix.device
    )]
    return 0.5 * (row_loss + column_loss), {
        "pairs": int(selected_rows.numel() * (selected_rows.numel() - 1) // 2),
        "assignment_accuracy": float(
            ((cost_matrix.argmin(dim=-1) == labels).float().mean()).item()
        ),
        "margin": float((off_diagonal.mean() - diagonal.mean()).item()),
    }


def masked_mean(values: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Average token values over an optional binary supervision mask."""
    if mask is None:
        return values.mean()
    weights = mask.to(device=values.device, dtype=values.dtype)
    if weights.shape != values.shape:
        raise ValueError(f"mask shape {weights.shape} does not match values {values.shape}")
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


class TrajectoryAlignment(nn.Module):
    """Align repeated Observer.D method states with an LFM layer trajectory."""

    def __init__(self, student_width: int, projection_dim: int):
        super().__init__()
        self.student_projection = nn.Linear(student_width, projection_dim, bias=False)

    @staticmethod
    def _resample_depth(teacher: torch.Tensor, target_depth: int) -> torch.Tensor:
        # [batch, teacher_depth, sequence, projection] -> interpolate depth.
        batch, depth, sequence, projection = teacher.shape
        values = teacher.permute(0, 2, 3, 1).reshape(batch * sequence, projection, depth)
        values = F.interpolate(values, size=target_depth, mode="linear", align_corners=True)
        return values.reshape(batch, sequence, projection, target_depth).permute(0, 3, 1, 2)

    def forward(
        self,
        student_states: tuple[torch.Tensor, ...],
        teacher_projected_states: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if not student_states:
            raise ValueError("Observer.D must return method states for trajectory alignment")
        student = torch.stack(student_states, dim=1)
        student = self.student_projection(student.float())
        teacher = self._resample_depth(teacher_projected_states.float(), student.shape[1])

        student_normalized = F.normalize(student, dim=-1)
        teacher_normalized = F.normalize(teacher, dim=-1)
        state_error = (student_normalized - teacher_normalized).square().mean(dim=(1, 3))
        state_loss = masked_mean(state_error, mask)

        student_delta = student[:, 1:] - student[:, :-1]
        teacher_delta = teacher[:, 1:] - teacher[:, :-1]
        delta_error = (
            F.normalize(student_delta, dim=-1)
            - F.normalize(teacher_delta, dim=-1)
        ).square().mean(dim=(1, 3))
        delta_loss = masked_mean(delta_error, mask)
        return {"state": state_loss, "delta": delta_loss}
