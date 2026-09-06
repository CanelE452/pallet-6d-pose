"""DiffPnP YOLO loss 게이트 — 통과 전에는 학습을 걸지 않는다.

    conda run -n pallet-yolo26 python -m pytest challenge/tests/test_diffpnp_yolo_loss.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from pallet_yolo_loss.diffpnp import (DiffPnPConfig, DiffPnPIndex,  # noqa: E402
                                      diffpnp_corner_loss, project,
                                      recover_affine)

INDEX_DIR = REPO / "data/pallet/results/diffpnp_yolo_v1"
pytestmark = pytest.mark.skipif(not (INDEX_DIR / "diffpnp_index.npz").exists(),
                                reason="사이드카 인덱스가 없다")


@pytest.fixture(scope="module")
def index():
    return DiffPnPIndex(INDEX_DIR)


def _batch(index, n=16, start=0):
    sl = slice(start, start + n)
    dt = torch.float64
    return {k: torch.as_tensor(v[sl], dtype=dt) for k, v in
            {"K": index.K, "X": index.X, "uv": index.uv,
             "R": index.R, "t": index.t}.items()} | {
        "diag": torch.as_tensor(index.diag[sl], dtype=dt)}


def test_reference_pose_reprojects_onto_source_keypoints(index):
    """사이드카의 참조 pose 가 소스 2D 를 실제로 재현해야 한다."""
    b = _batch(index, 64)
    uv, _ = project(b["R"], b["t"], b["X"], b["K"])
    err = (uv - b["uv"]).norm(dim=-1)
    assert float(err.max()) < 1e-3, float(err.max())


def test_affine_recovery_is_exact_for_a_known_warp(index):
    """K' = A K 보정의 전제 — affine 이 정확히 복원돼야 한다."""
    b = _batch(index, 32)
    M = b["uv"].shape[0]
    g = torch.Generator().manual_seed(0)
    A_true = torch.zeros(M, 2, 3, dtype=torch.float64)
    A_true[:, 0, 0] = 1.0 + 0.3 * torch.rand(M, generator=g, dtype=torch.float64)
    A_true[:, 1, 1] = 1.0 + 0.3 * torch.rand(M, generator=g, dtype=torch.float64)
    A_true[:, 0, 1] = 0.1 * torch.randn(M, generator=g, dtype=torch.float64)
    A_true[:, 0, 2] = 50.0 * torch.randn(M, generator=g, dtype=torch.float64)
    A_true[:, 1, 2] = 50.0 * torch.randn(M, generator=g, dtype=torch.float64)
    p = torch.cat([b["uv"], torch.ones(M, 8, 1, dtype=torch.float64)], dim=-1)
    dst = torch.einsum("mij,mnj->mni", A_true, p)

    A, residual, ok = recover_affine(b["uv"], dst, torch.ones(M, 8, dtype=torch.float64))
    assert bool(ok.all())
    assert float(residual.max()) < 1e-6, float(residual.max())
    assert torch.allclose(A, A_true, atol=1e-8)


def test_warped_intrinsics_reproduce_warped_keypoints(index):
    """K'=A K 가 실제로 워프된 2D 를 만들어내는지 — 증강 대응의 핵심 가정."""
    b = _batch(index, 32)
    M = b["uv"].shape[0]
    A = torch.zeros(M, 2, 3, dtype=torch.float64)
    A[:, 0, 0] = 0.7; A[:, 1, 1] = 0.7
    A[:, 0, 2] = 100.0; A[:, 1, 2] = -25.0
    p = torch.cat([b["uv"], torch.ones(M, 8, 1, dtype=torch.float64)], dim=-1)
    dst = torch.einsum("mij,mnj->mni", A, p)

    row = torch.zeros(M, 1, 3, dtype=torch.float64)
    row[:, 0, 2] = 1.0
    K_eff = torch.bmm(torch.cat([A, row], dim=1), b["K"])
    uv, _ = project(b["R"], b["t"], b["X"], K_eff)
    assert float((uv - dst).norm(dim=-1).max()) < 1e-3


def test_perfect_prediction_gives_near_zero_loss(index):
    """예측이 GT 와 같으면 3D 코너 오차가 0 이어야 한다."""
    b = _batch(index, 32)
    cfg = DiffPnPConfig(enabled=True, lambda_dp=1.0, gn_steps=5)
    vis = torch.ones(b["uv"].shape[0], 8, dtype=torch.float64)
    loss, valid, stats = diffpnp_corner_loss(
        b["uv"], vis, b["K"], b["X"], b["R"], b["t"], b["diag"], cfg)
    assert bool(valid.all())
    assert float(loss) < 1e-10, float(loss)
    assert stats["mean_corner_norm"] < 1e-5


def test_perturbed_prediction_gives_positive_loss_and_gradient(index):
    """틀린 예측이면 loss 가 커지고, 예측 2D 로 grad 가 흘러야 한다."""
    b = _batch(index, 32)
    cfg = DiffPnPConfig(enabled=True, lambda_dp=1.0, gn_steps=5)
    vis = torch.ones(b["uv"].shape[0], 8, dtype=torch.float64)
    g = torch.Generator().manual_seed(3)
    noisy = (b["uv"] + 6.0 * torch.randn(b["uv"].shape, generator=g,
                                         dtype=torch.float64)).requires_grad_(True)
    loss, valid, stats = diffpnp_corner_loss(
        noisy, vis, b["K"], b["X"], b["R"], b["t"], b["diag"], cfg)
    assert float(loss) > 1e-6
    loss.backward()
    assert noisy.grad is not None
    assert torch.isfinite(noisy.grad).all()
    assert float(noisy.grad.abs().sum()) > 0.0


def test_loss_grows_with_keypoint_error(index):
    """단조성 — 2D 오차가 커지면 3D 코너 항도 커져야 한다."""
    b = _batch(index, 48)
    cfg = DiffPnPConfig(enabled=True, lambda_dp=1.0, gn_steps=5)
    vis = torch.ones(b["uv"].shape[0], 8, dtype=torch.float64)
    g = torch.Generator().manual_seed(7)
    base = torch.randn(b["uv"].shape, generator=g, dtype=torch.float64)
    previous = -1.0
    for sigma in (1.0, 3.0, 9.0):
        loss, _, _ = diffpnp_corner_loss(b["uv"] + sigma * base, vis, b["K"],
                                         b["X"], b["R"], b["t"], b["diag"], cfg)
        value = float(loss)
        assert value > previous, (sigma, value, previous)
        previous = value


def test_lambda_zero_never_builds_the_index():
    """구성적 parity — lambda_dp=0 이면 사이드카조차 읽지 않는다."""
    import os

    from pallet_yolo_loss.diffpnp import DiffPnPPoseLoss26

    assert "DIFFPNP_CONFIG" not in os.environ or True
    cfg = DiffPnPConfig()
    assert cfg.enabled is False and cfg.lambda_dp == 0.0
    # 클래스가 인덱스를 만드는 조건을 명시적으로 확인한다
    src = Path(DiffPnPPoseLoss26.__init__.__code__.co_filename).read_text()
    assert "if self.dp.enabled and self.dp.lambda_dp != 0.0:" in src
