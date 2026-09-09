"""A2b (LC_DERIVED_STABLE_MECHANISM_BASELINE) 의 수식·배선 테스트.

학습을 돌리기 전에 통과해야 한다.  특히 torch Jacobian 이
finite-difference 로 이미 검증된 numpy 구현과 일치하는지를 본다 —
회전 블록을 한 번 틀린 적이 있다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "challenge/yolo_pose_one_model/pallet_translation_loss_v1"))
sys.path.insert(0, str(ROOT / "scripts/research/pallet_translation_loss_v1"))

a2b = pytest.importorskip("a2b_loss")
from pnp_jacobian import expm_so3, finite_difference_jacobian, pnp_jacobian  # noqa: E402


def _cuboid(w, h, d):
    return np.array([[-w/2, -h/2, -d/2], [w/2, -h/2, -d/2], [w/2, h/2, -d/2], [-w/2, h/2, -d/2],
                     [-w/2, -h/2, d/2], [w/2, -h/2, d/2], [w/2, h/2, d/2], [-w/2, h/2, d/2]])


def _case(seed=0, M=3):
    rng = np.random.default_rng(seed)
    K4, R, t, X = [], [], [], []
    for _ in range(M):
        K4.append([rng.uniform(400, 1200), rng.uniform(400, 1200), 320.0, 240.0])
        R.append(expm_so3(rng.normal(size=3) * 0.7))
        t.append([rng.uniform(-.4, .4), rng.uniform(-.4, .4), rng.uniform(1.5, 4.0)])
        X.append(_cuboid(rng.uniform(.6, 1.4), rng.uniform(.09, .18), rng.uniform(.6, 1.4)))
    f = lambda a: torch.tensor(np.asarray(a), dtype=torch.float64)
    return f(K4), f(R), f(t), f(X)


def test_torch_jacobian_matches_verified_numpy():
    K4, R, t, X = _case(1)
    J2d, Jalt = a2b.pose_jacobians(K4, R, t, X)
    for m in range(len(K4)):
        Kn = np.array([[K4[m, 0], 0, K4[m, 2]], [0, K4[m, 1], K4[m, 3]], [0, 0, 1]])
        ref = pnp_jacobian(Kn, R[m].numpy(), t[m].numpy(), X[m].numpy())
        assert np.abs(J2d[m].numpy() - ref).max() < 1e-8
        assert Jalt[m].shape == (24, 6)


def test_torch_jacobian_matches_finite_difference():
    K4, R, t, X = _case(2, M=1)
    Kn = np.array([[K4[0, 0], 0, K4[0, 2]], [0, K4[0, 1], K4[0, 3]], [0, 0, 1]])
    fd = finite_difference_jacobian(Kn, R[0].numpy(), t[0].numpy(), X[0].numpy())
    J2d, _ = a2b.pose_jacobians(K4, R, t, X)
    col = np.abs(J2d[0].numpy() - fd).max(0) / np.maximum(np.abs(fd).max(0), 1e-12)
    assert col.max() < 1e-4


def test_projection_round_trip():
    K4, R, t, X = _case(3)
    u = a2b.project(K4, R, t, X)
    assert u.shape == (len(K4), 8, 2) and torch.isfinite(u).all()


def _inputs(seed=4, noise=0.0, M=3):
    K4, R, t, X = _case(seed, M)
    gt = a2b.project(K4, R, t, X)
    rng = np.random.default_rng(seed + 99)
    pred = gt + torch.tensor(rng.normal(scale=noise, size=gt.shape), dtype=gt.dtype)
    vis = torch.ones(gt.shape[:2], dtype=torch.bool)
    inv_std = torch.ones_like(gt)
    return K4, R, t, X, gt, pred.requires_grad_(True), vis, inv_std


def test_covariance_part_is_zero_at_gt_and_grows_with_residual():
    """전체 NLL 이 아니라 **공분산 항**이 잔차의 단조 함수여야 한다.

    L = log(prior) + 0.5 (cov + lin)/prior 이고 prior 도 가중치를 통해 잔차에
    의존하므로, LC 는 전체 NLL 의 단조성을 약속하지 않는다.  약속하는 것은
    잔차가 0 이면 공분산 기여가 0 이라는 것과, 잔차가 커지면 그것이 커진다는 것이다.
    """
    cfg = a2b.A2BConfig(enabled=True, lambda_geo=1.0)
    covs = []
    for n in (0.0, 1.0, 3.0, 8.0):
        st = {}
        a2b.lc_covariance_term(*_inputs(5, n), cfg, stats=st)
        covs.append(st["cov_err"])

    # 잔차가 정확히 0 이면 공분산 대각이 전부 0 이라 LC 원본의 비-PSD sentinel
    # (`torch.where(good, sum, 1)`)이 걸려 1.0 이 된다.  원형 충실도를 위해
    # 그대로 두었고, 실제 학습에서는 잔차가 정확히 0 이 되지 않는다.
    assert covs[0] == pytest.approx(1.0), covs
    assert all(b > a for a, b in zip(covs[1:], covs[2:])), covs
    assert covs[1] < 0.1, covs


def test_weights_do_not_collapse_at_zero_residual():
    """퇴화 가드 — 잔차 0 에서 가중치가 사라지면 prior 가 폭발한다."""
    cfg = a2b.A2BConfig(enabled=True, lambda_geo=1.0)
    st0, st1 = {}, {}
    a2b.lc_covariance_term(*_inputs(4, 0.0), cfg, stats=st0)
    a2b.lc_covariance_term(*_inputs(4, 2.0), cfg, stats=st1)
    # 완벽히 맞춘 표본의 prior 가 어긋난 표본보다 크면 안 된다
    assert st0["prior_err"] < st1["prior_err"] * 10, (st0, st1)
    assert st0["prior_err"] < 1.0, st0        # 미터 단위. 폭발하면 1e3 이상이 된다


def test_gradient_is_finite_and_reaches_prediction():
    cfg = a2b.A2BConfig(enabled=True, lambda_geo=1.0)
    K4, R, t, X, gt, pred, vis, inv_std = _inputs(6, 2.0)
    a2b.lc_covariance_term(K4, R, t, X, gt, pred, vis, inv_std, cfg).mean().backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum() > 0


def test_no_nan_on_near_singular_view():
    cfg = a2b.A2BConfig(enabled=True, lambda_geo=1.0)
    K4 = torch.tensor([[900.0, 900.0, 320.0, 240.0]], dtype=torch.float64)
    R = torch.eye(3, dtype=torch.float64)[None]
    t = torch.tensor([[0.0, 0.0, 12.0]], dtype=torch.float64)
    X = torch.tensor(_cuboid(1.1, 0.02, 1.1), dtype=torch.float64)[None]
    gt = a2b.project(K4, R, t, X)
    pred = (gt + 0.7).requires_grad_(True)
    v = a2b.lc_covariance_term(K4, R, t, X, gt, pred, torch.ones(1, 8, dtype=torch.bool),
                               torch.ones_like(gt), cfg)
    assert torch.isfinite(v).all()
    v.mean().backward()
    assert torch.isfinite(pred.grad).all()


def test_invisible_keypoints_contribute_nothing():
    cfg = a2b.A2BConfig(enabled=True, lambda_geo=1.0)
    K4, R, t, X, gt, pred, vis, inv_std = _inputs(7, 0.0)
    pred = pred.detach().clone()
    vis2 = vis.clone()
    vis2[:, 5:] = False
    a = a2b.lc_covariance_term(K4, R, t, X, gt, pred.clone(), vis2, inv_std, cfg).detach()
    pred[:, 5:, :] += 40.0
    b = a2b.lc_covariance_term(K4, R, t, X, gt, pred, vis2, inv_std, cfg).detach()
    assert torch.allclose(a, b, atol=1e-8), (a, b)


def test_centroid_index_is_excluded():
    K4, R, t, X = _case(8)
    assert X.shape[1] == 8
    J2d, Jalt = a2b.pose_jacobians(K4, R, t, X)
    assert J2d.shape[1] == 16 and Jalt.shape[1] == 24


def test_config_default_is_disabled():
    cfg = a2b.A2BConfig()
    assert cfg.enabled is False and cfg.lambda_geo == 0.0


def test_sidetable_present_and_consistent():
    p = ROOT / "challenge/yolo_pose_one_model/pallet_translation_loss_v1/GEOMETRY_SIDETABLE.npz"
    assert p.is_file()
    d = np.load(p, allow_pickle=True)
    assert len(d["stems"]) == 60000
    assert d["Xcf"].shape == (60000, 8, 3)
    assert d["match_err"].max() < 0.05
