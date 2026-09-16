"""Exact discrete joint minimization of linear costs plus two global RLE clamps."""
import itertools
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint


def objective(cost, branch):
    selected = cost[np.arange(len(branch)), branch]
    return float(selected[:, 0].sum() + np.maximum(selected[:, 1:].sum(0), 0).sum())


def joint_minimum(cost, valid):
    """cost[N,G,3]: position+visibility, weighted raw RLE head1, raw RLE head2.

    A separable linear lower-bound certificate often proves the solution immediately.
    Otherwise solve the exact one-hot MILP, never substitute coordinate descent.
    Selection is detached FP64; selected loss is recomputed by the stock FP32 path.
    """
    cost = np.asarray(cost, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    assert cost.shape[:2] == valid.shape and cost.shape[2] == 3
    assert np.isfinite(cost).all() and valid.any(1).all()
    n, groups = valid.shape
    if n == 0:
        return np.zeros(0, dtype=np.int64), dict(method='empty', value=0., lower_bound=0.)
    # min_g(c + u*r) is a valid lower bound because max(0,r) >= u*r.
    for signs in [(1, 1), (0, 1), (1, 0), (0, 0)]:
        linear = cost[:, :, 0] + cost[:, :, 1:] @ np.array(signs)
        linear = np.where(valid, linear, np.inf)
        branch = linear.argmin(1)
        raw = cost[np.arange(n), branch, 1:].sum(0)
        if all((r >= 0 if s else r <= 0) for r, s in zip(raw, signs)):
            value = objective(cost, branch)
            lower = float(linear[np.arange(n), branch].sum())
            assert abs(value-lower) <= 1e-10 * max(1, abs(value))
            return branch, dict(method='certified_linear_bound', signs=list(signs), value=value, lower_bound=lower)
    count = n*groups
    c = np.r_[cost[:, :, 0].ravel(), 1., 1.]
    integrality = np.r_[np.ones(count), 0., 0.]
    lower = np.zeros(count+2); upper = np.r_[valid.ravel().astype(float), np.inf, np.inf]
    matrix = np.zeros((n+2, count+2))
    for i in range(n): matrix[i, i*groups:(i+1)*groups] = 1
    for h in range(2):
        matrix[n+h, :count] = -cost[:, :, h+1].ravel()
        matrix[n+h, count+h] = 1
    constraints = LinearConstraint(matrix, np.r_[np.ones(n), 0., 0.], np.r_[np.ones(n), np.inf, np.inf])
    result = milp(c, integrality=integrality, bounds=Bounds(lower, upper), constraints=constraints,
                  options=dict(mip_rel_gap=0., presolve=True, time_limit=30.))
    if not result.success or result.mip_gap > 1e-10:
        raise RuntimeError(f'Joint optimum not certified; stop rather than approximate: {result.message}')
    branch = result.x[:count].reshape(n, groups).argmax(1)
    assert valid[np.arange(n), branch].all()
    value = objective(cost, branch)
    assert abs(value-result.fun) <= 2e-6 * max(1, abs(value))
    return branch, dict(method='certified_MILP', value=value, lower_bound=float(result.mip_dual_bound),
                        gap=float(result.mip_gap), nodes=int(result.mip_node_count))
