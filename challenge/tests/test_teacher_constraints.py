import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Deep_Object_Pose" / "common"))

from teacher_constraints import (  # noqa: E402
    final_belief_distillation_loss,
    mask_extent_per_frame,
    teacher_peak_retention_loss,
    teacher_peak_retention_per_frame,
    top_fraction_cvar,
)


def test_exact_teacher_has_zero_distillation_and_peak_retention():
    teacher = torch.zeros(2, 3, 4, 5, dtype=torch.float64)
    teacher[0, 0, 1, 2] = 0.9
    teacher[0, 1, 2, 3] = 0.8
    teacher[1, 0, 3, 4] = 0.7
    student = teacher.clone().requires_grad_()
    channel_valid = torch.tensor([[1, 1, 0], [1, 0, 0]])

    distill = final_belief_distillation_loss(
        student, teacher, channel_mask=channel_valid)
    peak = teacher_peak_retention_loss(
        student, teacher, channel_validity=channel_valid,
        teacher_peak_threshold=0.5, margin=0.05)
    assert distill.item() == 0.0
    assert peak.item() == 0.0
    (distill + peak).backward()
    assert torch.isfinite(student.grad).all()
    assert student.grad.abs().sum().item() == 0.0
    assert teacher.requires_grad is False


def test_weakened_teacher_peak_is_positive_and_gradient_restores_cell():
    teacher = torch.zeros(1, 2, 3, 4)
    teacher[0, 0, 1, 2] = 1.0
    teacher[0, 1, 2, 1] = 0.2  # Excluded by threshold.
    student = teacher.clone()
    student[0, 0, 1, 2] = 0.5
    student.requires_grad_()

    per_frame, valid = teacher_peak_retention_per_frame(
        student, teacher, teacher_peak_threshold=0.5, margin=0.1)
    assert valid.tolist() == [True]
    torch.testing.assert_close(per_frame, torch.tensor([0.4]))
    per_frame.mean().backward()
    assert student.grad[0, 0, 1, 2].item() < 0.0
    assert student.grad[0, 1].abs().sum().item() == 0.0
    before = student.detach()[0, 0, 1, 2].item()
    after = (student.detach() - 0.2 * student.grad)[0, 0, 1, 2].item()
    assert after > before
    assert torch.isfinite(student.grad).all()


def test_top_fraction_cvar_excludes_nonhard_and_invalid_frames():
    per_frame = torch.tensor(
        [0.1, 4.0, 2.0, 100.0], dtype=torch.float64, requires_grad=True)
    loss, info = top_fraction_cvar(
        per_frame, torch.tensor([1, 1, 1, 0]), top_fraction=0.34)

    # ceil(3 * .34) = 2: only 4 and 2 are selected; invalid 100 is ignored.
    assert loss.item() == 3.0
    assert info == {
        "valid_count": 3,
        "selected_count": 2,
        "selected_fraction": 2 / 3,
        "top_fraction": 0.34,
    }
    loss.backward()
    torch.testing.assert_close(
        per_frame.grad, torch.tensor([0.0, 0.5, 0.5, 0.0],
                                     dtype=torch.float64))


def test_top_fraction_cvar_can_rank_by_frozen_teacher_extent():
    student_loss = torch.tensor([10.0, 1.0, 2.0], requires_grad=True)
    teacher_extent = torch.tensor([0.1, 0.9, 0.8])
    loss, info = top_fraction_cvar(
        student_loss, torch.ones(3, dtype=torch.bool), top_fraction=1 / 3,
        rank_by=teacher_extent)
    assert loss.item() == 1.0
    assert info["selected_count"] == 1
    loss.backward()
    # Teacher frame 1 ranks worst even though its student loss is smallest.
    torch.testing.assert_close(
        student_loss.grad, torch.tensor([0.0, 1.0, 0.0]))


def test_top_fraction_cvar_no_valid_frame_is_finite_graph_zero():
    per_frame = torch.tensor(
        [float("nan"), 2.0], dtype=torch.float64, requires_grad=True)
    loss, info = top_fraction_cvar(
        per_frame, torch.tensor([0, 0]), top_fraction=0.25)
    assert loss.item() == 0.0
    assert info["valid_count"] == 0
    assert info["selected_count"] == 0
    loss.backward()
    assert torch.isfinite(per_frame.grad).all()
    assert per_frame.grad.abs().sum().item() == 0.0


def test_mask_extent_per_frame_matches_four_sided_cell_gaps():
    belief = torch.zeros(2, 8, 10, 10)
    # Eight peaks span x=[3,6], y=[2,7].
    coordinates = [
        (3, 2), (6, 2), (6, 7), (3, 7),
        (4, 3), (5, 3), (5, 6), (4, 6),
    ]
    for channel, (x, y) in enumerate(coordinates):
        belief[:, channel, y, x] = 1.0
    mask = torch.zeros(2, 1, 10, 10)
    mask[0, 0, 1:9, 2:8] = 1.0  # bbox x=[2,7], y=[1,8].

    per_frame, valid, gap_cells = mask_extent_per_frame(
        belief, mask, torch.tensor([1, 1]), radius=0,
        temperature=0.1, tolerance=0.0)
    assert valid.tolist() == [True, False]
    torch.testing.assert_close(gap_cells[0], torch.tensor(1.0))
    torch.testing.assert_close(per_frame[0], torch.tensor(0.1))
    assert torch.isfinite(per_frame).all()


def test_masked_distillation_none_returns_graph_zero_for_invalid_frame():
    teacher = torch.ones(2, 2, 2, 2)
    student = torch.zeros_like(teacher, requires_grad=True)
    per_frame = final_belief_distillation_loss(
        student, teacher, channel_mask=torch.tensor([[1, 0], [0, 0]]),
        reduction="none")
    torch.testing.assert_close(per_frame, torch.tensor([1.0, 0.0]))
    per_frame.sum().backward()
    assert student.grad[0, 0].abs().sum().item() > 0.0
    assert student.grad[0, 1].abs().sum().item() == 0.0
    assert student.grad[1].abs().sum().item() == 0.0
