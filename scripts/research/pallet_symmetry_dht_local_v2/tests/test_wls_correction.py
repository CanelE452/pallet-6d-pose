"""Independent absolute-coordinate least-squares reference and defect regressions."""
import numpy as np
import pytest
import torch

from scripts.research.pallet_symmetry_dht_local_v2.symdht_local_v2 import geometry as G


def numpy_reference(base, valid, sigma, lines, weights, line_sigma, hw, edges=None):
    edges = G.EDGES if edges is None else edges
    result = base.copy()
    for b in range(len(base)):
        for k in range(8):
            if not valid[b, k]:
                continue
            # Solve for absolute q using stacked weighted observations, not
            # the implementation's increment/normal-equation assembly.
            design = [np.eye(2)[j] / sigma[b, k] for j in range(2)]
            target = [base[b, k, j] / sigma[b, k] for j in range(2)]
            for e, endpoints in enumerate(edges):
                if k in endpoints and weights[b, e] > 0:
                    scale = np.sqrt(weights[b, e]) / line_sigma[b]
                    design.append(lines[b, e, :2] * scale)
                    target.append(-lines[b, e, 2] * scale)
            q = np.linalg.lstsq(np.asarray(design), target, rcond=None)[0]
            delta = q - base[b, k]
            cap = .01 * np.linalg.norm(hw[b])
            delta *= min(1., cap / max(np.linalg.norm(delta), 1e-12))
            result[b, k] += delta
    return result


def fixture(ax=2., bx=0.):
    base = torch.zeros(1, 9, 2, dtype=torch.float64)
    base[0, 0] = torch.tensor([ax, 0.]); base[0, 1] = torch.tensor([bx, 10.])
    valid = torch.ones(1, 9, dtype=torch.bool)
    sigma = torch.ones(1, 9, dtype=torch.float64)
    lines = torch.zeros(1, 12, 3, dtype=torch.float64); lines[..., 0] = 1
    weights = torch.zeros(1, 12, dtype=torch.float64); weights[0, 0] = 1
    return base, valid, sigma, lines, weights, torch.ones(1, dtype=torch.float64), torch.tensor([[800., 800.]], dtype=torch.float64)


@pytest.mark.parametrize('ax,bx', [(2., 0.), (0., 2.), (2., -2.)])
def test_asymmetric_endpoints_match_reference_and_utility(ax, bx, monkeypatch):
    args = fixture(ax, bx)
    actual, _ = G.single_mode_wls(*args)
    expected = numpy_reference(*(t.numpy() for t in args))
    np.testing.assert_allclose(actual.numpy(), expected, atol=1e-12, rtol=0)
    if bx == 0: assert torch.equal(actual[0, 1], args[0][0, 1])
    target = args[0].clone(); target[0, :2, 0] = 0
    label, gain, _ = G.candidate_utility(args[0], args[1], args[2], args[3], target, args[1], args[5], args[6])
    measured = (torch.linalg.vector_norm(args[0][0, :2]-target[0, :2], dim=-1) - torch.linalg.vector_norm(actual[0, :2]-target[0, :2], dim=-1)).mean()
    torch.testing.assert_close(gain[0, 0], measured)
    assert label[0, 0] == float(measured > .25)
    monkeypatch.setattr(G, 'EDGES', tuple((b, a) for a, b in G.EDGES))
    reversed_result, _ = G.single_mode_wls(*args)
    assert torch.equal(actual, reversed_result)


def test_zero_utility_noop_and_center_exact():
    args = list(fixture()); args[4].zero_()
    out, delta = G.single_mode_wls(*args)
    assert torch.equal(out, args[0]) and torch.count_nonzero(delta) == 0


def test_random_multiedge_matches_independent_lstsq():
    rng = np.random.default_rng(817)
    base = rng.normal(size=(4, 9, 2)) * 50
    valid = rng.random((4, 9)) > .2
    sigma = rng.uniform(.5, 10, (4, 9))
    theta = rng.uniform(-np.pi, np.pi, (4, 12))
    lines = np.stack((np.cos(theta), np.sin(theta), rng.normal(size=(4, 12))*30), -1)
    weights = rng.uniform(0, 1, (4, 12)); ls = rng.uniform(.5, 5, 4)
    hw = np.full((4, 2), 800.)
    arrays = (base, valid, sigma, lines, weights, ls, hw)
    out, delta = G.single_mode_wls(*(torch.from_numpy(x) for x in arrays))
    np.testing.assert_allclose(out.numpy(), numpy_reference(*arrays), rtol=0, atol=1e-10)
    assert torch.equal(out[:, 8], torch.from_numpy(base[:, 8]))
    assert float(delta.norm(dim=-1).max()) <= .01*np.linalg.norm(hw[0])+1e-10


def test_gradient_matches_independent_torch_absolute_solve():
    args = list(fixture(2., -2.))
    args[3] = args[3].clone().requires_grad_()
    args[4] = args[4].clone().requires_grad_()
    q, _ = G.single_mode_wls(*args)
    loss = q[0, :2, 0].square().sum()
    got = torch.autograd.grad(loss, (args[3], args[4]))
    normal, offset = args[3][0, 0, :2], args[3][0, 0, 2]
    weight = args[4][0, 0]
    A = torch.eye(2, dtype=torch.float64) + weight*torch.outer(normal, normal)
    reference = torch.stack([torch.linalg.solve(A, args[0][0, k]-weight*offset*normal) for k in (0, 1)])
    want = torch.autograd.grad(reference[:, 0].square().sum(), (args[3], args[4]))
    torch.testing.assert_close(got[0][0, 0], want[0][0, 0])
    torch.testing.assert_close(got[1][0, 0], want[1][0, 0])
