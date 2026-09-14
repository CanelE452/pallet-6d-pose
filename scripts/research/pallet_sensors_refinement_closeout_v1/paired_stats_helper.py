"""Paired seed-mean pooled medians; a single baseline is never n independent models."""
from __future__ import annotations
import numpy as np


def _prepare(rows, frame_group):
    values = np.concatenate([np.asarray(row, dtype=float) for row in rows])
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Require nonempty finite supervised errors; audit missing outputs separately')
    group = np.concatenate([np.full(len(rows[i]), frame_group[i], dtype=int) for i in range(len(rows))])
    order = np.argsort(values, kind='stable')
    return values[order], group[order]


def _median(prepared, counts):
    values, groups = prepared
    weights = counts[groups]
    cumulative = np.cumsum(weights, dtype=np.int64)
    n = int(cumulative[-1])
    if n == 0:
        return np.nan
    # Exact np.median of an integer-multiplicity expanded multiset, including even n.
    left, right = (n - 1) // 2, n // 2
    il = np.searchsorted(cumulative, left, side='right')
    ir = np.searchsorted(cumulative, right, side='right')
    return float((values[il] + values[ir]) * .5)


def paired_pooled_median(baseline_rows, candidate_seed_rows, sessions,
                         resamples=10000, seed=20260914, level='session'):
    n = len(sessions)
    if n == 0 or len(baseline_rows) != n or not candidate_seed_rows:
        raise ValueError('Empty or inconsistent panel')
    if level not in ('session', 'frame'):
        raise ValueError(level)
    for rows in candidate_seed_rows:
        if len(rows) != n or any(len(rows[i]) != len(baseline_rows[i]) for i in range(n)):
            raise ValueError('Membership/support mismatch; do not silently intersect predictions')
    names = sorted(set(sessions)) if level == 'session' else list(range(n))
    lookup = {name: i for i, name in enumerate(names)}
    group = np.asarray([lookup[s] for s in sessions]) if level == 'session' else np.arange(n)
    base = _prepare(baseline_rows, group)
    candidates = [_prepare(rows, group) for rows in candidate_seed_rows]
    units = len(names)
    ones = np.ones(units, dtype=np.int64)
    base_stat = _median(base, ones)
    per_seed = [_median(item, ones) - base_stat for item in candidates]
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=float)
    for r in range(resamples):
        counts = rng.multinomial(units, np.full(units, 1 / units))
        b = _median(base, counts)
        draws[r] = np.mean([_median(item, counts) for item in candidates]) - b
    if not np.isfinite(draws).all():
        raise ValueError('Empty resampled supervision; record and review, do not discard draws')
    return dict(estimand='mean_seed(pooled median candidate) - pooled median single baseline',
                lower_is_better=True,level=level,units=units,frames=n,seeds=len(candidates),
                baseline=base_stat,per_seed_delta=per_seed,delta=float(np.mean(per_seed)),
                low=float(np.quantile(draws,.025)),high=float(np.quantile(draws,.975)),
                resamples=resamples,random_seed=seed,
                scope='conditional on supplied support, observed sessions and trained seed panel; not a novelty or acceptance test')
