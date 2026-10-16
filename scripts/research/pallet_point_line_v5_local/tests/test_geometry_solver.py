"""Generated-fixture gate: analytic geometry, Jacobians, symmetry, recovery."""
import sys, math
from pathlib import Path
import pytest, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plpose_local import geometry as G, solver as S

torch.manual_seed(0)


def scene(B=3, dtype=torch.float64):
    K = torch.tensor([[500., 0., 320.], [0., 500., 240.], [0., 0., 1.]], dtype=dtype).expand(B, 3, 3).contiguous()
    dims = torch.tensor([[1.2, .8, .15], [1.0, 1.0, .14], [1.4, .9, .16]], dtype=dtype)[:B]
    r = torch.tensor([[.2, .5, -.1], [-.3, .9, .05], [.1, -.6, .2]], dtype=dtype)[:B]
    R = G.exp_so3(r)
    t = torch.tensor([[.05, .02, 3.0], [-.1, .05, 2.4], [.2, -.03, 3.6]], dtype=dtype)[:B]
    return K, R, t, dims


def test_corner_extents_match_dims():
    K, R, t, dims = scene()
    c = G.corners(dims)[:, :8]
    ext = c.max(1).values - c.min(1).values
    assert torch.allclose(ext, torch.stack((dims[:, 0], dims[:, 2], dims[:, 1]), -1), atol=1e-12)
    assert torch.allclose(G.corners(dims)[:, 8], torch.zeros(3, dtype=dims.dtype))


def test_projection_jacobian_matches_finite_difference():
    K, R, t, dims = scene()
    X = G.corners(dims)
    J = G.project_jacobian(K, R, t, X)
    eps = 1e-7
    for k in range(6):
        d = torch.zeros(K.shape[0], 6, dtype=X.dtype); d[:, k] = eps
        up = G.project(K, G.exp_so3(d[:, :3]) @ R, t + d[:, 3:], X)[0]
        dn = G.project(K, G.exp_so3(-d[:, :3]) @ R, t - d[:, 3:], X)[0]
        fd = (up - dn) / (2 * eps)
        assert torch.allclose(J[..., k], fd, atol=1e-5), f'dof {k}'


def test_rotation_jacobian_rejects_the_RX_plus_t_mistake():
    """-skew(RX) is correct; -skew(RX+t) must not reproduce the finite difference."""
    K, R, t, dims = scene()
    X = G.corners(dims)
    RX = X @ R.transpose(-1, -2)
    good = G.skew(RX)
    bad = G.skew(RX + t[:, None])
    assert not torch.allclose(good, bad, atol=1e-6)


def test_c2_symmetry_permutes_corners_for_any_footprint():
    K, R, t, dims = scene()
    c = G.corners(dims)
    perm = G.symmetry_permutations(2)[1]
    rot = c @ G.yaw(2, dtype=c.dtype).T
    assert torch.allclose(c[:, perm], rot, atol=1e-12)


def test_c4_permutation_only_valid_when_square():
    square = torch.tensor([[1.0, 1.0, .15]], dtype=torch.float64)
    rect = torch.tensor([[1.2, .8, .15]], dtype=torch.float64)
    perm = G.symmetry_permutations(4)[1]
    for dims, ok in ((square, True), (rect, False)):
        c = G.corners(dims)
        rot = c @ G.yaw(1, dtype=c.dtype).T
        assert torch.allclose(c[:, perm], rot, atol=1e-12) is ok


def test_reframe_rejects_reflection_and_scale():
    R = G.exp_so3(torch.tensor([[.1, .2, .3]], dtype=torch.float64))
    with pytest.raises(ValueError):
        G.reframe_pose(R, torch.diag(torch.tensor([1., 1., -1.], dtype=torch.float64)))
    with pytest.raises(ValueError):
        G.reframe_pose(R, torch.eye(3, dtype=torch.float64) * 2)
    assert torch.allclose(G.reframe_pose(R, torch.eye(3, dtype=torch.float64)), R)


def _observations(K, R, t, dims):
    uv, _ = G.project(K, R, t, G.corners(dims))
    diag = torch.full((K.shape[0],), math.hypot(640, 480), dtype=uv.dtype)
    u = torch.tensor([e[0] for e in G.EDGES]); v = torch.tensor([e[1] for e in G.EDGES])
    a, b = uv[:, u], uv[:, v]
    d = b - a
    n = torch.stack((-d[..., 1], d[..., 0]), -1)
    n = n / torch.linalg.vector_norm(n, dim=-1, keepdim=True)
    c = -(n * a).sum(-1)
    lines = torch.cat((n, c[..., None]), -1)[:, :, None]
    mass = torch.ones(K.shape[0], 12, 1, dtype=uv.dtype)
    mask = torch.ones(K.shape[0], 12, dtype=torch.bool)
    return uv[:, :9], diag, lines, mass, mask


def test_point_only_recovers_pose_from_a_perturbed_start():
    K, R, t, dims = scene()
    uv, diag, *_ = _observations(K, R, t, dims)
    pm = torch.ones(K.shape[0], 9, dtype=torch.bool)
    R0 = G.exp_so3(torch.full((K.shape[0], 3), .05, dtype=uv.dtype)) @ R
    t0 = t + .05
    Rr, tr = S.refine(K, R0, t0, dims, uv, pm, None, None, None, diag, iters=25)
    assert torch.allclose(tr, t, atol=1e-4)
    assert torch.allclose(Rr, R, atol=1e-4)


def test_point_plus_line_recovers_pose_and_lines_lower_the_energy():
    K, R, t, dims = scene()
    uv, diag, lines, mass, mask = _observations(K, R, t, dims)
    pm = torch.ones(K.shape[0], 9, dtype=torch.bool)
    R0 = G.exp_so3(torch.full((K.shape[0], 3), .05, dtype=uv.dtype)) @ R
    t0 = t + .05
    e0, ok0 = S.energy(K, R0, t0, dims, uv, pm, lines, mass, mask, diag)
    Rr, tr = S.refine(K, R0, t0, dims, uv, pm, lines, mass, mask, diag, iters=25)
    e1, ok1 = S.energy(K, Rr, tr, dims, uv, pm, lines, mass, mask, diag)
    assert ok0.all() and ok1.all() and (e1 < e0).all()
    assert torch.allclose(tr, t, atol=1e-4)


def test_endpoints_of_an_edge_share_one_mode_responsibility():
    """A decoy mode fitting only one endpoint must not drive the energy to zero."""
    K, R, t, dims = scene(B=1)
    uv, diag, lines, mass, mask = _observations(K, R, t, dims)
    decoy = lines.clone(); decoy[..., 2] = decoy[..., 2] + 30.
    two = torch.cat((lines, decoy), dim=2)
    m2 = torch.ones(1, 12, 2, dtype=uv.dtype)
    pm = torch.ones(1, 9, dtype=torch.bool)
    e_true, _ = S.energy(K, R, t, dims, uv, pm, lines, mass, mask, diag)
    e_both, ok = S.energy(K, R, t, dims, uv, pm, two, m2, mask, diag)
    assert ok.all() and torch.isfinite(e_both).all() and (e_both >= e_true - 1e-9).all()


def test_selection_uses_energy_only_and_flags_invalid():
    K, R, t, dims = scene()
    uv, diag, lines, mass, mask = _observations(K, R, t, dims)
    pm = torch.ones(K.shape[0], 9, dtype=torch.bool)
    behind = t.clone(); behind[:, 2] = -3.
    Rs = torch.stack((R, R), 1); ts = torch.stack((behind, t), 1)
    Rb, tb, e, best, valid = S.select(K, Rs, ts, dims, uv, pm, lines, mass, mask, diag)
    assert (best == 1).all() and valid.all()
    assert torch.allclose(tb, t)


def test_energy_reports_validity_instead_of_a_huge_constant():
    """A behind-camera candidate must be flagged, not given a finite penalty.

    A large constant would dominate any softmax over candidates; that defect
    made the ranking term reach 2e8 with a 1e10 gradient before it was fixed.
    """
    K, R, t, dims = scene(B=1)
    uv, diag, lines, mass, mask = _observations(K, R, t, dims)
    pm = torch.ones(1, 9, dtype=torch.bool)
    behind = t.clone(); behind[:, 2] = -3.
    good, ok_good = S.energy(K, R, t, dims, uv, pm, lines, mass, mask, diag)
    bad, ok_bad = S.energy(K, R, behind, dims, uv, pm, lines, mass, mask, diag)
    assert bool(ok_good.all()) and not bool(ok_bad.any())
    assert torch.isfinite(bad).all()
