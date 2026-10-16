"""Geometric counterexamples independent of learned pallet predictions."""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plpose_local import geometry as G, solver as S
from plpose_local.heads import decode_modes, lattice_lines, PoseNet
from plpose_local.losses import line_targets


def test_quantization_across_pi_preserves_line_location():
    # Near vertical image lines have normals near 0/pi. Wrong rho sign moves
    # an x=-.5 line to x=+.5, nearly the entire ROI width away.
    for deg in (1., 179., 181., 359.):
        theta = math.radians(deg)
        n = torch.tensor([math.cos(theta), math.sin(theta)], dtype=torch.float64)
        d = torch.stack((-n[1], n[0]))
        pts = torch.zeros(1, 9, 2, dtype=torch.float64)
        pts[0, 0], pts[0, 1] = .5*n-.2*d, .5*n+.2*d
        tgt, ok = line_targets(pts, torch.tensor([[-1., -1., 1., 1.]]), 36, 65)
        line = lattice_lines(36, 65, dtype=torch.float64)[tgt[0, 0]]
        assert ok[0, 0]
        assert ((pts[0, :2] @ line[:2] + line[2]).abs() < .03).all()


def test_mode_crosses_seam_with_flipped_rho_and_aligned_sign():
    T, R = 36, 65
    lines = lattice_lines(T, R, dtype=torch.float64)
    logits = torch.full((1, 1, T*R), -80., dtype=torch.float64)
    logits[0, 0, 44], logits[0, 0, 35*R+20] = 0., 0.
    out, mass = decode_modes(logits, lines, T, R, modes=1)
    assert mass[0, 0, 0] > .999  # both equivalent-side peaks collected
    l = out[0, 0, 0]
    assert torch.allclose(torch.linalg.vector_norm(l[:2]), torch.tensor(1., dtype=l.dtype))
    assert abs(float(l[2])) > .5  # no cancellation into line through origin
    assert abs(float(l[1])) > .02  # both peaks actually contributed


def test_distinct_parallel_lines_do_not_average_into_empty_space():
    T, R = 36, 65
    lines = lattice_lines(T, R)
    logits = torch.full((1, 1, T*R), -80.)
    logits[0, 0, 12*R+12], logits[0, 0, 12*R+52] = 0., -.01
    out, _ = decode_modes(logits, lines, T, R, modes=2)
    assert (out[0, 0, :, 2].abs() > .8).all()


def scene():
    dtype = torch.float64
    K = torch.tensor([[[500., 0., 320.], [0., 500., 240.], [0., 0., 1.]]], dtype=dtype)
    R = G.exp_so3(torch.tensor([[.2, .5, -.1]], dtype=dtype))
    t = torch.tensor([[.05, .02, 3.]], dtype=dtype)
    dims = torch.tensor([[1.2, .8, .15]], dtype=dtype)
    uv = G.project(K, R, t, G.corners(dims))[0]
    u, v = torch.tensor(G.EDGES).T
    a, b = uv[:, u], uv[:, v]
    d = b-a
    n = torch.stack((-d[..., 1], d[..., 0]), -1)
    n = n/torch.linalg.vector_norm(n, dim=-1, keepdim=True)
    lines = torch.cat((n, -(n*a).sum(-1, keepdim=True)), -1)[:, :, None]
    return K, R, t, dims, uv, lines


def test_one_pixel_point_and_line_energy_units_are_comparable():
    K, R, t, dims, uv, lines = scene()
    pm = torch.tensor([[True]*8+[False]])
    lm = torch.ones(1, 12, dtype=torch.bool)
    mass = torch.ones(1, 12, 1, dtype=uv.dtype)
    diag = torch.tensor([800.], dtype=uv.dtype)
    obs = uv.clone(); obs[:, :8, 0] += 1.
    point, _ = S.energy(K, R, t, dims, obs, pm, None, None, None, diag)
    shifted = lines.clone(); shifted[..., 2] += 1.
    joint, _ = S.energy(K, R, t, dims, uv, pm, shifted, mass, lm, diag)
    # 8 point x residuals; 24 endpoint normal residuals with 1/3 incidence.
    assert torch.allclose(point, joint, rtol=1e-9, atol=1e-14)


def test_final_pose_loss_backpropagates_to_lines_matches_finite_difference():
    K, R, t, dims, uv, lines = scene()
    pm = torch.tensor([[True]*8+[False]])
    lm = torch.ones(1, 12, dtype=torch.bool)
    mass = torch.ones(1, 12, 1, dtype=uv.dtype)
    diag = torch.tensor([800.], dtype=uv.dtype)
    R0 = G.exp_so3(torch.tensor([[.01, -.02, .01]], dtype=uv.dtype))@R
    t0 = t+.02
    def objective(offset):
        l = lines + torch.stack((torch.zeros_like(offset),torch.zeros_like(offset),offset)).reshape(1,1,1,3)
        rr, tt = S.refine(K, R0, t0, dims, uv, pm, l, mass, lm, diag, iters=5)
        p = G.project(K, rr, tt, G.corners(dims))[0]
        return (p-uv).square().mean()
    x = torch.tensor(1.3, dtype=uv.dtype, requires_grad=True)
    grad = torch.autograd.grad(objective(x), x)[0]
    eps = 1e-4
    fd = (objective(x.detach()+eps)-objective(x.detach()-eps))/(2*eps)
    assert torch.isfinite(grad) and grad.abs() > 1e-5
    assert torch.allclose(grad, fd, rtol=2e-3, atol=1e-6)


def test_final_pose_loss_alone_reaches_hough_parameters_and_features():
    torch.manual_seed(91)
    K, R, t, dims, uv, _ = scene()
    model = PoseNet('hough').float()
    features = torch.randn(1, 192, 24, 24, requires_grad=True)
    obs = dict(features=features, content=torch.ones(1,1,24,24,dtype=torch.bool),
               K=K.float(), dims=dims.float(), box=torch.tensor([[180.,160.,460.,340.]]),
               image_hw=torch.tensor([[480.,640.]]))
    out = model(obs)
    # Known nearby initialisation isolates the differentiable observation path.
    rr, tt = S.refine(obs['K'], R.float(), t.float()+.01, obs['dims'], uv.float(),
        torch.tensor([[True]*8+[False]]), out['lines'], out['mass'],
        torch.ones(1,12,dtype=torch.bool), torch.tensor([800.]), iters=2)
    pred = G.project(obs['K'], rr, tt, G.corners(obs['dims']))[0]
    loss = (pred-uv.float()).square().mean()  # no auxiliary point/line loss
    loss.backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()
    assert features.grad.abs().sum() > 0
    for name in ('line.reduce.weight', 'line.out.weight'):
        grad = dict(model.named_parameters())[name].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
