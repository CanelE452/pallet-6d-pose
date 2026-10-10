"""Paired frame/session summaries from sealed numeric prediction rows only."""
from collections import Counter
import hashlib
import math
import statistics as standard_statistics

import numpy as np

from . import verdict as V

METHODS = ('BASE', 'N3_DIM_SYM', 'SUBPIX', 'N3_THEN_SUBPIX')
POSE_METRICS = {'T_cm': 'translation_cm', 'R_deg': 'rotation_deg',
                'ADDsym_m': 'ADDsym_m', 'IoU3D': 'IoU3D'}
SEEDS = (1, 2, 3)


class Draws:
    """Exactly the legacy default_rng multinomial algorithm in 100-draw batches."""
    def __init__(self, labels, level):
        _, self.group = np.unique(np.asarray(labels), return_inverse=True)
        self.n = int(self.group.max() + 1)
        self.counts = np.bincount(self.group, minlength=self.n).astype(float)
        rng = np.random.default_rng(V.BOOTSTRAP_SEED)
        raw = np.concatenate([rng.multinomial(self.n, np.full(self.n, 1 / self.n), size=100)
                              for _ in range(V.BOOTSTRAP_DRAWS // 100)])
        packed = raw.astype('<u2')
        self.sha256 = hashlib.sha256(packed.tobytes()).hexdigest()
        self.weights = raw.astype(float)
        self.denominator = self.weights @ self.counts
        self.level = level
        self.master_frames = len(labels)

    def subset(self, master_labels, selected_indices):
        """Reuse the exact master draw; absent scope units have zero count."""
        selected_indices = np.asarray(selected_indices, int)
        sub = object.__new__(Draws)
        sub.group = self.group[selected_indices]
        sub.n, sub.weights, sub.sha256, sub.level = self.n, self.weights, self.sha256, self.level
        sub.master_frames = self.master_frames
        sub.counts = np.bincount(sub.group, minlength=self.n).astype(float)
        sub.denominator = sub.weights @ sub.counts
        return sub

    def metadata(self):
        return dict(level=self.level, units=self.n, resamples=V.BOOTSTRAP_DRAWS,
                    seed=V.BOOTSTRAP_SEED, draws_sha256_uint16_le=self.sha256,
                    algorithm='default_rng.multinomial; 100 batches of100; uniformly sample units',
                    interval='numpy linear quantile at0.025 and0.975', multiplicity_adjusted=False,
                    scope_frames=int(self.counts.sum()), master_frames=self.master_frames,
                    nonempty_resamples=int((self.denominator > 0).sum()),
                    scope_rule='same full-population draws; absent units contribute zero count')

    def samples(self, values):
        values = np.asarray(values, float)
        totals = np.bincount(self.group, weights=values, minlength=self.n)
        return np.divide(self.weights @ totals, self.denominator,
                         out=np.full(len(self.denominator), np.nan), where=self.denominator > 0)

    def interval(self, values):
        samples = self.samples(values)
        valid = samples[np.isfinite(samples)]
        return np.quantile(valid, [0.025, 0.975]).tolist() if len(valid) else None

    def conditional_interval(self, values):
        values = np.asarray(values, float)
        valid = np.isfinite(values)
        if not valid.any():
            return None
        if valid.all():
            return self.interval(values)
        totals = np.bincount(self.group, weights=np.where(valid, values, 0), minlength=self.n)
        counts = np.bincount(self.group, weights=valid.astype(float), minlength=self.n)
        denominator = self.weights @ counts
        samples = np.divide(self.weights @ totals, denominator,
                            out=np.full(len(denominator), np.nan), where=denominator > 0)
        return np.quantile(samples[np.isfinite(samples)], [0.025, 0.975]).tolist()


def describe(values, draws=None):
    values = np.asarray(values, float)
    x = values[np.isfinite(values)].tolist()
    if not x:
        return dict(n=0, mean=None, variance=None, std=None, median=None, P90=None, max=None, CI95=None)
    variance = standard_statistics.variance(x) if len(x) > 1 else None
    return dict(n=len(x), mean=standard_statistics.fmean(x), variance=variance,
        std=math.sqrt(variance) if variance is not None else None,
        median=standard_statistics.median(x), P90=float(np.quantile(x, .9)), max=max(x),
        CI95=draws.conditional_interval(values) if draws is not None else None,
        variance_ddof=1, CI95_target='mean', distribution='per-frame')


def _method(row):
    return row['method'].removesuffix('_VIS')


def _pose_value(row, key):
    p = row['pose']
    return float(p[key]) if p.get('available') and p.get(key) is not None else float('nan')


def _vectors(rows):
    indicators = [V.indicators(r['pose']) for r in rows]
    result = {name: np.asarray([_pose_value(r, key) for r in rows]) for name, key in POSE_METRICS.items()}
    result.update({name: np.asarray([v[name] for v in indicators])
                   for name in ('confusion_rate', 'success_rate')})
    result['available'] = np.asarray([v['available'] for v in indicators], bool)
    return result


def _corner(rows):
    corners = [r['corner'] for r in rows if r.get('corner', {}).get('evaluable', False)]
    errors = [v for c in corners for v in c['errors']]
    observed = [v for c in corners for v in c['observed_errors']]
    summary = dict(evaluable_frames=len(corners), reference_corners=len(errors), observed_corners=len(observed),
        errors=describe(errors), observed_errors=describe(observed),
        PCK={str(t): float(np.mean(np.asarray(errors) <= t)) if errors else None for t in (5, 10, 20)},
        gross20=float(np.mean(np.asarray(errors) > 20)) if errors else None,
        coordinate_change_due_to_VIS=False)
    if corners:
        summary['E_sym'] = standard_statistics.fmean(c['E_sym'] for c in corners)
    return summary


def _summary(rows, vectors, draws, *, is_seed_mean=False):
    rates = {}
    for name in ('confusion_rate', 'success_rate'):
        x = vectors[name]
        rates[name] = dict(rate=float(x.mean()), positive_count_sum=float(x.sum()), denominator=len(x),
                           CI95=draws.interval(x), binary_before_seed_mean=True,
                           missing_pose_indicator_false_descriptive_only=True)
    return dict(frames=len(rows), available=int(vectors['available'].sum()),
        unavailable=int((~vectors['available']).sum()), coverage=float(vectors['available'].mean()),
        metrics={name: describe(vectors[name], draws) for name in POSE_METRICS},
        rates=rates, corner=_corner(rows),
        seed_mean_distribution='per-ID arithmetic mean of3 seed errors; no ensemble pose' if is_seed_mean else None)


def _contrast(a, b, draws, secondary=None):
    result = dict(coverage_complete=bool(a['available'].all() and b['available'].all()),
                  bootstrap=draws.metadata(), metrics={})
    for metric in (*POSE_METRICS, 'confusion_rate', 'success_rate'):
        # Input shape3xN. A primary binary difference is averaged before draws.
        delta = a[metric] - b[metric]
        complete = np.isfinite(delta).all(axis=0)
        mean_delta = np.mean(delta, axis=0)
        per_seed = [float(x[np.isfinite(x)].mean()) if np.isfinite(x).any() else None for x in delta]
        ci = draws.conditional_interval(mean_delta)
        lower_better = metric != 'success_rate' and metric != 'IoU3D'
        r = dict(delta=float(mean_delta[complete].mean()) if complete.any() else None, CI95=ci,
            per_seed_delta=per_seed, improved_seeds=sum(x is not None and (x < 0 if lower_better else x > 0)
                                                     for x in per_seed),
            paired_frames=int(complete.sum()), lower_is_better=lower_better,
            coverage_complete=result['coverage_complete'], multiplicity_adjusted=False)
        if secondary is not None:
            r['scenario_cluster_secondary'] = dict(delta=r['delta'], CI95=secondary.conditional_interval(mean_delta),
                                                   bootstrap=secondary.metadata())
        if metric in ('confusion_rate', 'success_rate'):
            result[metric] = r
        else:
            result['metrics'][metric] = r
    return result


def _damage(base, changed):
    damage, recovery, transitions, unavailable = [], [], [], []
    for b, v in zip(base, changed):
        assert b['id'] == v['id']
        bi, vi = V.indicators(b['pose']), V.indicators(v['pose'])
        if bi['success_rate'] and not vi['success_rate']:
            damage.append(b['id'])
        if not bi['success_rate'] and vi['success_rate']:
            recovery.append(b['id'])
        if b.get('final_hypothesis') != v.get('final_hypothesis'):
            transitions.append(dict(id=b['id'], ALL=b.get('final_hypothesis'), VIS=v.get('final_hypothesis'),
                                    delta_T_cm=_pose_value(v, 'translation_cm')-_pose_value(b, 'translation_cm'),
                                    delta_R_deg=_pose_value(v, 'rotation_deg')-_pose_value(b, 'rotation_deg')))
        if not vi['available']:
            unavailable.append(v['id'])
    return dict(success_to_failure_ids=damage, failure_to_success_ids=recovery,
                success_to_failure=len(damage), failure_to_success=len(recovery),
                hypothesis_changes=len(transitions), hypothesis_transition_records=transitions,
                unavailable_VIS_ids=unavailable)


def aggregate(rows_all, rows_vis, *, population, bootstrap_level='frame', scopes=None):
    """Shared A1/A2 schema; never infer a new population or discard a failure."""
    assert bootstrap_level in ('frame', 'cluster')
    all_index = {(int(r['seed']), _method(r), r['id']): r for r in rows_all}
    vis_index = {(int(r['seed']), _method(r), r['id']): r for r in rows_vis}
    assert len(all_index) == len(rows_all) and len(vis_index) == len(rows_vis)
    assert set(all_index) == set(vis_index)
    ids = [r['id'] for r in rows_all if int(r['seed']) == 1 and _method(r) == 'BASE']
    expected = {(s, m, i) for s in SEEDS for m in METHODS for i in ids}
    assert len(ids) == len(set(ids)) and set(all_index) == expected
    for key in expected:
        b, v = all_index[key], vis_index[key]
        assert b['corner'] == v['corner'], ('VIS corner metrics changed', key)
    if scopes is None:
        scopes = {'ALL': ids}
        if all('grade' in all_index[(1, 'BASE', i)] for i in ids):
            for grade in ('clean', 'moderate', 'severe'):
                scopes[grade] = [i for i in ids if all_index[(1, 'BASE', i)]['grade'] == grade]
    assert scopes.get('ALL') == ids
    metrics = dict(schema='pallet_vispnp_metrics_v1', status='COMPLETE', population=population,
        unique_frames=len(ids), seeds=list(SEEDS), methods=[*METHODS, *(m+'_VIS' for m in METHODS)],
        scope_frame_ids=scopes,
        by_seed={str(s): {} for s in SEEDS}, seed_mean={},
        definitions=dict(primary_binary=V.rules(), numeric_variance_ddof=1,
            primary_full_denominator=True, numeric_pose_statistics='available-only; n reported',
            seed_mean_error_distribution='per-ID arithmetic mean3 errors, requiring all3 available',
            seed_mean_rates='arithmetic mean of3 binary indicators, no thresholding averaged errors',
            corner='existing canonical proper symmetry branch and full GT-valid corner denominator'),
        diagnostics=dict(hidden_count_by_seed={str(s): {} for s in SEEDS},
                         hidden_count_grouping='each seed/method own sealed VIS hidden_count; ALL and VIS same IDs'))
    paired = dict(schema='pallet_vispnp_paired_v1', population=population,
                  by_seed={str(s): {} for s in SEEDS}, seed_mean={}, bootstrap={})
    failures = dict(schema='pallet_vispnp_failures_v1', population=population,
                    per_seed={str(s): {} for s in SEEDS}, counts_are_repeated_executions_not_independent_frames=True)
    master_clusters = [all_index[(1, 'BASE', i)]['session'] for i in ids]
    master = Draws(np.arange(len(ids)) if bootstrap_level == 'frame' else master_clusters, bootstrap_level)
    secondary_master = Draws(master_clusters, 'scenario_cluster') if bootstrap_level == 'frame' else None
    id_index = {fid:i for i,fid in enumerate(ids)}
    for scope, scope_ids in scopes.items():
        if not scope_ids:
            continue
        assert set(scope_ids) <= set(ids)
        indices = [id_index[fid] for fid in scope_ids]
        draws = master.subset(master_clusters, indices)
        secondary = secondary_master.subset(master_clusters, indices) if secondary_master is not None else None
        paired['bootstrap'][scope] = dict(primary=draws.metadata(),
            scenario_cluster_secondary=secondary.metadata() if secondary is not None else None)
        vectors, grouped = {}, {}
        for s in SEEDS:
            metrics['by_seed'][str(s)][scope] = {}
            paired['by_seed'][str(s)][scope] = {}
            for m in METHODS:
                for index, suffix in ((all_index, ''), (vis_index, '_VIS')):
                    name = m + suffix
                    rows = [index[(s, m, i)] for i in scope_ids]
                    grouped[(s, name)] = rows
                    vector = _vectors(rows)
                    vectors[(s, name)] = vector
                    metrics['by_seed'][str(s)][scope][name] = _summary(rows, vector, draws)
                a = {k: x[None] for k, x in vectors[(s, m+'_VIS')].items()}
                b = {k: x[None] for k, x in vectors[(s, m)].items()}
                paired['by_seed'][str(s)][scope][m+'_VIS__minus__'+m] = _contrast(a, b, draws, secondary)
        metrics['seed_mean'][scope], paired['seed_mean'][scope] = {}, {}
        for m in METHODS:
            for suffix in ('', '_VIS'):
                name = m + suffix
                full = {k: np.stack([vectors[(s, name)][k] for s in SEEDS])
                        for k in vectors[(1, name)]}
                mean = {k: full[k].all(0) if k == 'available' else np.mean(full[k], axis=0) for k in full}
                r = _summary(grouped[(1, name)], mean, draws, is_seed_mean=True)
                r['corner'] = dict(arithmetic_mean_seed_PCK={str(t): float(np.mean([
                    metrics['by_seed'][str(s)][scope][name]['corner']['PCK'][str(t)] for s in SEEDS]))
                    for t in (5, 10, 20)},
                    arithmetic_mean_seed_gross20=float(np.mean([
                    metrics['by_seed'][str(s)][scope][name]['corner']['gross20'] for s in SEEDS])),
                    invariant_VIS_coordinates=True,
                    note='corner rates average seed rates; no thresholding averaged corner errors')
                metrics['seed_mean'][scope][name] = r
            a = {k: np.stack([vectors[(s, m+'_VIS')][k] for s in SEEDS]) for k in vectors[(1, m)]}
            b = {k: np.stack([vectors[(s, m)][k] for s in SEEDS]) for k in vectors[(1, m)]}
            paired['seed_mean'][scope][m+'_VIS__minus__'+m] = _contrast(a, b, draws, secondary)
        if scope == 'ALL':
            for s in SEEDS:
                for m in METHODS:
                    changed = grouped[(s, m+'_VIS')]
                    d = _damage(grouped[(s, m)], changed)
                    fallback = [r['id'] for r in changed if r.get('visibility', {}).get('fallback', False)]
                    d.update(fallback_lt4_ids=fallback, fallback_lt4=len(fallback),
                        hidden_count_distribution=dict(Counter(str(r.get('visibility', {}).get('hidden_count', 0))
                                                              for r in changed)))
                    failures['per_seed'][str(s)][m] = d
                    metrics['diagnostics']['hidden_count_by_seed'][str(s)][m] = {}
                    counts = [r['visibility']['hidden_count'] for r in changed]
                    for count in sorted(set(counts)):
                        local_indices = [i for i,c in enumerate(counts) if c == count]
                        scoped_draws = master.subset(master_clusters, local_indices)
                        scoped_base = [grouped[(s,m)][i] for i in local_indices]
                        scoped_vis = [changed[i] for i in local_indices]
                        base_vectors = {k:x[local_indices] for k,x in vectors[(s,m)].items()}
                        vis_vectors = {k:x[local_indices] for k,x in vectors[(s,m+'_VIS')].items()}
                        metrics['diagnostics']['hidden_count_by_seed'][str(s)][m][str(count)] = dict(
                            frames=len(local_indices), frame_ids=[r['id'] for r in scoped_vis],
                            ALL=_summary(scoped_base, base_vectors, scoped_draws),
                            VIS=_summary(scoped_vis, vis_vectors, scoped_draws),
                            damage=_damage(scoped_base, scoped_vis), bootstrap=scoped_draws.metadata())
        print('STATISTICS_SCOPE', population, scope, len(scope_ids), flush=True)
    paired['primary'] = paired['seed_mean']['ALL'][V.PRIMARY_CONTRAST]
    return metrics, paired, failures
