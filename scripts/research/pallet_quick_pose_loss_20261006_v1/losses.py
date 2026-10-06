"""Two fixed pose-cost losses; the inherited scores determine action support."""
import numpy as np
import torch


def derive_targets(cost, available, action_valid, scale, tau):
    """Preserve native indices; failed actions remain in the model denominator."""
    cost = np.asarray(cost, dtype=np.float64)
    available = np.asarray(available, dtype=bool)
    action_valid = np.asarray(action_valid, dtype=bool)
    assert cost.shape == available.shape == action_valid.shape
    assert np.isfinite(scale) and scale > 0 and np.isfinite(tau) and tau > 0
    success = action_valid & available & np.isfinite(cost)
    assert np.isposinf(cost[action_valid & ~available]).all()
    eligible = success.any(-1)
    minimum = np.where(success, cost, np.inf).min(-1)
    minimum = np.where(eligible, minimum, 0.)
    relative = np.where(success, cost, minimum[:, None]) - minimum[:, None]
    relative /= scale
    maximum = relative.max(-1)
    effective = np.where(success, relative, maximum[:, None] + 1.)
    effective = np.where(action_valid & eligible[:, None], effective, 0.)
    unnormalized = np.where(success, np.exp(-relative / tau), 0.)
    probability = unnormalized / np.maximum(unnormalized.sum(-1, keepdims=True), 1e-300)
    return probability.astype(np.float32), effective.astype(np.float32), eligible


def pose_loss(scores, targets, eligible, method):
    """Return eligible-frame mean and counts, with no GT action-mask changes."""
    method = method.upper()
    if method not in ('SOFT6D', 'EXPECT6D'):
        raise ValueError('method must be SOFT6D or EXPECT6D')
    targets = targets.to(device=scores.device, dtype=scores.dtype)
    eligible = eligible.to(device=scores.device, dtype=torch.bool)
    assert scores.ndim == 2 and targets.shape == scores.shape
    assert eligible.shape == (len(scores),)
    assert not torch.isnan(scores).any() and not torch.isposinf(scores).any()
    assert torch.isfinite(targets).all() and (targets >= 0).all()
    counts = dict(eligible_frames=int(eligible.sum()), excluded_frames=int((~eligible).sum()))
    if not eligible.any():
        return torch.where(torch.isfinite(scores), scores, torch.zeros_like(scores)).sum() * 0, counts
    selected, target = scores[eligible], targets[eligible]
    support = torch.isfinite(selected)
    assert support.any(-1).all()
    if method == 'SOFT6D':
        assert torch.allclose(target.sum(-1), torch.ones(len(target), device=target.device, dtype=target.dtype), atol=2e-6, rtol=0)
        assert (target.masked_select(~support) == 0).all(), 'SOFT6D target mass inaccessible under inherited decoder'
        logp = selected.log_softmax(-1)
        safe_logp = torch.where(support, logp, torch.zeros_like(logp))
        value = -(target * safe_logp).sum(-1).mean()
    else:
        # Failed-F actions have finite effective cost and remain supported.
        value = (selected.softmax(-1) * target).sum(-1).mean()
    assert torch.isfinite(value)
    return value, counts


def analytic_gradient_checks():
    """Analytic score gradients, invariance and failure/padding toy contracts."""
    records = []
    backward_calls = 0
    for method in ('SOFT6D', 'EXPECT6D'):
        scores = torch.tensor([[.2, -.3, .7, -torch.inf], [.8, -.1, .3, -torch.inf],
                               [0., -torch.inf, -torch.inf, -torch.inf]], dtype=torch.float64, requires_grad=True)
        eligible = torch.tensor([True, False, True])
        target = torch.tensor([[.6, .4, 0., 0.], [0., 0., 0., 0.], [1., 0., 0., 0.]], dtype=torch.float64)
        if method == 'EXPECT6D':
            target = torch.tensor([[0., 1., 2., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.]], dtype=torch.float64)
        loss, counts = pose_loss(scores, target, eligible, method)
        loss.backward(); backward_calls += 1
        p = scores.detach().softmax(-1)
        expected = (p - target) / 2 if method == 'SOFT6D' else p * (target - (p * target).sum(-1, keepdim=True)) / 2
        expected[~eligible] = 0
        assert torch.allclose(scores.grad, expected, atol=1e-13, rtol=1e-12)
        assert torch.isfinite(scores.grad).all() and (scores.grad[:, 3] == 0).all()
        assert scores.grad[0, 0] < 0 and scores.grad[0, 2] > 0
        stepped = scores.detach() - .05 * scores.grad
        assert stepped.softmax(-1)[0, 0] > p[0, 0]
        stepped_value, _ = pose_loss(stepped, target, eligible, method)
        assert stepped_value < loss.detach()
        records.append(dict(check='analytic_gradient_' + method, passed=True, counts=counts))
    c = np.array([[4., 5., np.inf, np.inf], [3., 3., np.inf, np.inf], [np.inf] * 4])
    action = np.array([[1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 1, 0]], bool)
    valid = np.isfinite(c)
    y, r, eligible = derive_targets(c, valid, action, .5, .8)
    y2, r2, eligible2 = derive_targets(c + 100., valid, action, .5, .8)
    assert np.array_equal(y, y2) and np.array_equal(r, r2) and np.array_equal(eligible, eligible2)
    assert r[0, 2] == 3 and y[0, 2] == 0 and y[0, 3] == r[0, 3] == 0
    assert np.array_equal(y[1, :2], [.5, .5]) and not eligible[2]
    records.append(dict(check='constant_shift_failure_padding_equal_cost_all_failure', passed=True))
    for method, before, shifted in [('SOFT6D', y, y2), ('EXPECT6D', r, r2)]:
        gradients = []
        for value in (before, shifted):
            scores = torch.tensor([[.2, -.3, .7, -torch.inf], [.8, -.1, -torch.inf, -torch.inf],
                                   [0., .1, .2, -torch.inf]], dtype=torch.float64, requires_grad=True)
            loss, _ = pose_loss(scores, torch.from_numpy(value), torch.from_numpy(eligible), method)
            gradients.append(torch.autograd.grad(loss, scores)[0]); backward_calls += 1
        assert torch.equal(gradients[0], gradients[1])
        records.append(dict(check='constant_cost_shift_exact_gradient_' + method, passed=True))
    scores = torch.zeros((1, 2), dtype=torch.float64, requires_grad=True)
    value, counts = pose_loss(scores, torch.zeros_like(scores), torch.tensor([True]), 'EXPECT6D')
    value.backward(); backward_calls += 1
    assert value == 0 and torch.equal(scores.grad, torch.zeros_like(scores))
    records.append(dict(check='equal_cost_EXPECT6D_zero_gradient', passed=True))
    for method in ('SOFT6D', 'EXPECT6D'):
        scores = torch.tensor([[.1, -torch.inf]], dtype=torch.float64, requires_grad=True)
        value, counts = pose_loss(scores, torch.zeros_like(scores), torch.tensor([False]), method)
        value.backward(); backward_calls += 1
        assert value == 0 and torch.equal(scores.grad, torch.zeros_like(scores)) and counts['excluded_frames'] == 1
    records.append(dict(check='all_excluded_differentiable_zero', passed=True))
    scores = torch.tensor([[0., -torch.inf]], requires_grad=True)
    try:
        pose_loss(scores, torch.tensor([[.5, .5]]), torch.tensor([True]), 'SOFT6D')
    except AssertionError:
        records.append(dict(check='unsupported_positive_target_mass_rejected', passed=True))
    else:
        raise AssertionError('Unrepresentable SOFT6D target was accepted')
    return dict(status='PASS', checks=records, passed=len(records), toy_backward_calls=backward_calls,
                formal_model_forwards=0, final_F_calls=0, optimizer_updates=0)
