"""Six-method public saved-row statistics; no inference, fitting or pose solver.

The fixed 13-session draws and pooling definitions follow the October 8
statistics.py.  Seed-mean distributions average errors for each original
ID/canonical corner, retaining 319 image identities rather than 957 images.
"""
from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

METHODS = ('BASE', 'N3_DIM_SYM', 'SUBPIX', 'N3_THEN_SUBPIX', 'JOINT_FIXED_ISOTROPIC', 'FG_JOINT_POSTERIOR')
SEEDS = (1, 2, 3)
SETS = ('ALL', 'clean', 'moderate', 'severe')
METRICS = ('corner_px', 'translation_cm', 'rotation_deg', 'ADDsym_cm')
COMPARATORS = ('N3_DIM_SYM', 'SUBPIX', 'N3_THEN_SUBPIX', 'JOINT_FIXED_ISOTROPIC')
JOINT_METHODS = ('JOINT_FIXED_ISOTROPIC', 'FG_JOINT_POSTERIOR')
POSE_FIELDS = {'translation_cm': ('translation_cm', 1.),
               'rotation_deg': ('rotation_deg', 1.), 'ADDsym_cm': ('ADDsym_m', 100.)}
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DOC = ROOT / '_docs/experiments/pallet_feature_gradient_joint_20261010'


def read_rows(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                   allow_nan=False) + '\n', encoding='utf-8')


def distribution(values, unit=None):
    a = np.asarray(list(values), dtype=np.float64)
    assert a.ndim == 1 and np.isfinite(a).all(), 'Nonfinite eligible error'
    n = len(a)
    return dict(n=n, mean=float(a.mean()) if n else None,
                sample_variance=float(a.var(ddof=1)) if n > 1 else None,
                sample_std=float(a.std(ddof=1)) if n > 1 else None,
                median=float(np.median(a)) if n else None,
                P90=float(np.quantile(a, .9)) if n else None,
                min=float(a.min()) if n else None, max=float(a.max()) if n else None,
                ddof=1, unit=unit)


def observed_mask(row):
    c = row['corner']
    valid = np.asarray(c.get('canonical_valid', [False] * 8), dtype=bool)
    obs = np.asarray(row['canonical_observed'], dtype=bool)
    assert obs.shape == valid.shape == (8,) and not (obs & ~valid).any()
    if not c['evaluable']:
        assert not obs.any()
        return obs
    e = np.asarray(c['canonical_errors'], dtype=float)
    assert np.isfinite(e[valid]).all() and (e[valid] >= 0).all()
    assert np.array_equal(np.sort(e[obs]), np.sort(c['observed_errors']))
    return obs


def group_rows(rows):
    grouped = {str(s): {m: [] for m in METHODS} for s in SEEDS}
    for r in rows:
        assert r['seed'] in SEEDS and r['method'] in METHODS
        grouped[str(r['seed'])][r['method']].append(r)
    ids = [r['id'] for r in grouped['1']['BASE']]
    assert len(ids) == len(set(ids)) == 319
    assert len(rows) == 18 * 319
    for seed, methods in grouped.items():
        for name, rr in methods.items():
            lookup = {r['id']: r for r in rr}
            assert len(rr) == len(lookup) == 319 and set(lookup) == set(ids)
            methods[name] = [lookup[i] for i in ids]
    master = grouped['1']['BASE']
    assert len(set(r['session'] for r in master)) == 13
    assert Counter(r['grade'] for r in master) == {'clean': 153, 'moderate': 92, 'severe': 74}
    for methods in grouped.values():
        for rr in methods.values():
            assert [(r['id'], r['session'], r['grade']) for r in rr] == [
                (r['id'], r['session'], r['grade']) for r in master]
            for r in rr:
                observed_mask(r)
    return grouped, master


class SharedBootstrap:
    """Exact legacy draws, shared across every seed, metric and grade subset."""
    def __init__(self, master):
        self.ids = [r['id'] for r in master]
        self.units, self.inverse = np.unique([r['session'] for r in master], return_inverse=True)
        assert len(self.units) == 13
        raw = np.random.default_rng(20260917).multinomial(
            13, np.full(13, 1 / 13), size=10000).astype('uint16')
        self.draw_sha256 = hashlib.sha256(raw.tobytes()).hexdigest()
        self.weights = raw.astype(float)
        self.indices = dict(zip(self.ids, self.inverse.tolist()))
        self.denominators = {}

    def pooled(self, rows, values, unit):
        count = np.zeros(13)
        total = np.zeros(13)
        for row, val in zip(rows, values):
            j = self.indices[row['id']]
            count[j] += len(val)
            total[j] += float(np.sum(val, dtype=np.float64))
        key = tuple(count)
        if key not in self.denominators:
            self.denominators[key] = np.sum(self.weights * count, axis=1)
        denominator = self.denominators[key]
        numerator = np.sum(self.weights * total, axis=1)
        keep = denominator > 0
        draws = numerator[keep] / denominator[keep]
        pooled = np.concatenate(values) if values else np.asarray([], dtype=float)
        eligible = int(np.count_nonzero(count))
        return dict(mean_paired_difference=float(pooled.mean()) if len(pooled) else None,
                    median_paired_difference=float(np.median(pooled)) if len(pooled) else None,
                    CI95=np.quantile(draws, [.025, .975]).tolist() if len(draws) and eligible > 1 else None,
                    common_eligible_observations=len(pooled),
                    common_eligible_frames=int(sum(len(v) > 0 for v in values)),
                    excluded_frames=int(sum(len(v) == 0 for v in values)),
                    full_master_frames=len(rows), eligible_sessions=eligible,
                    valid_resamples=int(keep.sum()), empty_resamples=int((~keep).sum()),
                    unit=unit, draw_sha256=self.draw_sha256, seed=20260917, resamples=10000,
                    observation_weighting='original observed canonical corners' if unit == 'px'
                    else 'original common successful frames')

    def meta(self, master):
        counts = Counter(r['session'] for r in master)
        return dict(level='original_session', units=13, full_master_frames=319,
                    seed=20260917, resamples=10000, draw_sha256=self.draw_sha256,
                    group_sizes=dict(sorted(counts.items())), shared_across_seeds_methods_grades=True,
                    weighting='uniform original-session draws; pooled original observations',
                    multiplicity_adjusted=False)


def metric_values(rows, metric):
    values = []
    for r in rows:
        if metric == 'corner_px':
            c = r['corner']
            a = np.asarray(c.get('canonical_errors', [None] * 8), float)
            values.append(a[observed_mask(r)])
        else:
            field, factor = POSE_FIELDS[metric]
            values.append(np.asarray([float(r['pose'][field]) * factor]
                          if r['pose']['available'] else [], dtype=float))
    return values


def summarize(rows, bootstrap):
    corners = [r['corner'] for r in rows if r['corner']['evaluable']]
    full = [e for c in corners for e in c['errors']]
    summaries = {}
    for metric in METRICS:
        unit = 'px' if metric == 'corner_px' else 'degree' if metric == 'rotation_deg' else 'cm'
        values = metric_values(rows, metric)
        stats = distribution(np.concatenate(values) if values else [], unit)
        ci = bootstrap.pooled(rows, values, unit)
        stats.update(CI95=ci['CI95'], CI95_method='13-session paired shared cluster bootstrap 10000 draws',
                     eligible_sessions=ci['eligible_sessions'], valid_resamples=ci['valid_resamples'],
                     draw_sha256=bootstrap.draw_sha256)
        summaries[metric] = stats
    success = [r for r in rows if r['pose']['available']]
    meters = distribution([r['pose']['ADDsym_m'] for r in success], 'm')
    conversion = {}
    for field, factor in (('mean', 100.), ('sample_std', 100.), ('sample_variance', 10000.)):
        cm = summaries['ADDsym_cm'][field]
        passed = cm is None and meters[field] is None or bool(np.isclose(cm, meters[field] * factor, rtol=1e-12, atol=1e-12))
        assert passed, (field, cm, meters[field])
        conversion[field] = dict(factor=factor, passed=passed)
    a = np.asarray(full, float)
    statuses, fallbacks = Counter(), Counter()
    fallback_ids = []
    algorithm_calls = 0
    summary_only = bool(rows and rows[0].get('summary_only'))
    if not summary_only:
        for row in rows:
            diagnostics = row.get('correction', {}).get('diagnostics', {})
            statuses.update(diagnostics.get('status_counts', {}))
            fallbacks.update(diagnostics.get('fallback_counts', {}))
            algorithm_calls += diagnostics.get('algorithm_corner_calls', 0)
            if any(diagnostics.get('fallback_counts', {}).values()):
                fallback_ids.append(row['id'])
    correction_diagnostics = dict(summary_only=summary_only, status_counts=dict(statuses),
        fallback_counts=dict(fallbacks), fallback_corners=sum(fallbacks.values()) if not summary_only else None,
        fallback_frames=len(fallback_ids) if not summary_only else None, fallback_ids=fallback_ids,
        algorithm_corner_calls=algorithm_calls if not summary_only else None,
        definition='explicit local-refinement fallback corners retain the method starting point before final Base cap; corner fallback differs from no_pose; frozen baseline rows are reused historical results, not extra current executions'
        if not summary_only else 'averaged error summary has no correction execution; use per-seed diagnostics')
    from scripts.paper.pose_metric_closure_v1.symmetry_aware_pose_metrics import (
        pose_auc, AUC_MAX_FRACTION, AUC_INTEGRATION_POINTS)
    normalized = [float(r['pose']['ADDsym_normalized']) if r['pose']['available'] else np.inf for r in rows]
    valid_normalized = [v for v in normalized if np.isfinite(v)]
    auc = dict(status='EXISTING_CONTRACT', full=pose_auc(normalized, 1.) if rows else None,
        conditional=pose_auc(valid_normalized, 1.) if valid_normalized else None,
        full_frames=len(rows), conditional_frames=len(valid_normalized),
        max_normalized_fraction=AUC_MAX_FRACTION, integration_points=AUC_INTEGRATION_POINTS,
        source='scripts/paper/pose_metric_closure_v1/symmetry_aware_pose_metrics.py:pose_auc',
        definition='existing ADDsym_normalized; accuracy <=threshold over0..0.1/1001 grid; trapezoid integral divided by0.1; failed poses infinity')
    return dict(total_frames=len(rows), metrics=summaries,
                corner=dict(reference_corners=len(full), observed_corners=summaries['corner_px']['n'],
                            evaluable_frames=len(corners), matched_frames=sum(bool(r['corner']['matched']) for r in rows),
                            detected_frames=sum(bool(r['corner']['detected']) for r in rows),
                            PCK={str(t): float(np.mean(a <= t)) if len(a) else None for t in (5, 10, 20)},
                            gross20_count=int(np.sum(a > 20)), gross20_rate=float(np.mean(a > 20)) if len(a) else None,
                            full_reference_error_px=distribution(full, 'px')),
                pose=dict(available=len(success), failures=len(rows) - len(success),
                          coverage=len(success) / len(rows) if rows else None,
                          failure_rate=(len(rows) - len(success)) / len(rows) if rows else None,
                          successful_ids=[r['id'] for r in success],
                          failure_ids=[r['id'] for r in rows if not r['pose']['available']]),
                ADDsym_m_to_cm_check=conversion,
                ADDsym_AUC=auc,
                success_5cm_5deg=dict(status='NOT_CONFIRMED',reason='no frozen5cm/5degree operational success contract in the reused baseline; no new definition invented'),
                correction_diagnostics=correction_diagnostics,
                final_hypothesis_counts=dict(Counter(r.get('final_hypothesis') or 'UNAVAILABLE' for r in rows)),
                largest_errors={metric: sorted([
                    dict(id=r['id'], session=r['session'], grade=r['grade'], error=float(v.mean()))
                    for r, v in zip(rows, metric_values(rows, metric)) if len(v)],
                    key=lambda x: x['error'], reverse=True)[:10] for metric in METRICS})


def mean_seed_rows(grouped, method):
    """Average scalar reference errors per ID; never average R,t or rerun F."""
    result = []
    for triple in zip(*(grouped[str(s)][method] for s in SEEDS)):
        r = copy.deepcopy(triple[0])
        r['seed'] = 'seed_mean'
        r['summary_only'] = 'arithmetic mean of each per-ID/canonical error over all3 seeds; not a new pose prediction'
        for key in ('canonical_valid', 'matched', 'detected', 'evaluable'):
            assert all(x['corner'].get(key) == r['corner'].get(key) for x in triple), (r['id'], key)
        assert all(x['canonical_observed'] == r['canonical_observed'] for x in triple)
        c = r['corner']
        if c['evaluable']:
            valid = np.asarray(c['canonical_valid'], bool)
            arrays = np.asarray([x['corner']['canonical_errors'] for x in triple], float)
            av = arrays.mean(axis=0)
            c['canonical_errors'] = [float(v) if flag else None for v, flag in zip(av, valid)]
            c['errors'] = av[valid].tolist()
            c['observed_errors'] = av[np.asarray(r['canonical_observed'], bool)].tolist()
            c['frame_mean_px'] = float(np.mean([x['corner']['frame_mean_px'] for x in triple]))
        r['pose']['available'] = all(x['pose']['available'] for x in triple)
        if r['pose']['available']:
            for field in ('translation_cm', 'rotation_deg', 'ADDsym_m', 'ADDsym_normalized'):
                r['pose'][field] = float(np.mean([x['pose'][field] for x in triple]))
        hypotheses = [x.get('final_hypothesis') for x in triple]
        r['final_hypothesis'] = hypotheses[0] if len(set(hypotheses)) == 1 else 'SEED_HETEROGENEOUS'
        result.append(r)
    return result


def damage(before, after):
    a, b = [], []
    frames = []
    harmed = []
    recovered = []
    for old, new in zip(before, after):
        assert old['id'] == new['id']
        if not old['corner']['evaluable']:
            continue
        assert old['corner']['canonical_valid'] == new['corner']['canonical_valid']
        for k, valid in enumerate(old['corner']['canonical_valid']):
            if valid:
                x, y = old['corner']['canonical_errors'][k], new['corner']['canonical_errors'][k]
                a.append(x); b.append(y)
                if x < 5 and y > 10:
                    harmed.append(dict(id=old['id'], canonical_corner=k, before_px=x, after_px=y))
                if x > 20 and y <= 10:
                    recovered.append(dict(id=old['id'], canonical_corner=k, before_px=x, after_px=y))
        frames.append(new['corner']['frame_mean_px'] - old['corner']['frame_mean_px'])
    a, b, delta = np.asarray(a), np.asarray(b), np.asarray(frames)
    return dict(corners=len(a), before_good5=int(np.sum(a < 5)), before_bad20=int(np.sum(a > 20)),
                good5_to_bad10=len(harmed), bad20_to_good10=len(recovered),
                improved_frames=int(np.sum(delta < -1e-9)), harmed_frames=int(np.sum(delta > 1e-9)),
                unchanged_frames=int(np.sum(np.abs(delta) <= 1e-9)), harmed_corner_records=harmed,
                recovered_corner_records=recovered, canonical_GT_identity_aligned=True,
                definition='before<5px and after>10px; before>20px and after<=10px; full canonical reference')


def paired_values(new, base, metric):
    values = []
    for a, b in zip(new, base):
        assert a['id'] == b['id']
        if metric == 'corner_px':
            assert a['corner']['canonical_valid'] == b['corner']['canonical_valid']
            mask = observed_mask(a) & observed_mask(b)
            av = np.asarray(a['corner'].get('canonical_errors', [None] * 8), float)
            bv = np.asarray(b['corner'].get('canonical_errors', [None] * 8), float)
            values.append(av[mask] - bv[mask])
        else:
            field, factor = POSE_FIELDS[metric]
            values.append(np.asarray([factor * (a['pose'][field] - b['pose'][field])]
                if a['pose']['available'] and b['pose']['available'] else [], dtype=float))
    return values


def compare(new, base, bootstrap, sn, sb):
    paired = {metric: bootstrap.pooled(new, paired_values(new, base, metric),
              'px' if metric == 'corner_px' else 'degree' if metric == 'rotation_deg' else 'cm')
              for metric in METRICS}
    return dict(statistics=paired,
                difference_of_marginal_statistics={metric: {key: sn['metrics'][metric][key] - sb['metrics'][metric][key]
                    if sn['metrics'][metric][key] is not None and sb['metrics'][metric][key] is not None else None
                    for key in ('mean', 'median', 'P90', 'sample_variance', 'sample_std')} for metric in METRICS},
                coverage=dict(full_frames=len(new), common_success=sum(a['pose']['available'] and b['pose']['available'] for a,b in zip(new,base)),
                    new_failures=sum(not r['pose']['available'] for r in new),
                    base_failures=sum(not r['pose']['available'] for r in base),
                    new_only_success=sum(a['pose']['available'] and not b['pose']['available'] for a,b in zip(new,base)),
                    base_only_success=sum(not a['pose']['available'] and b['pose']['available'] for a,b in zip(new,base)),
                    both_failed=sum(not a['pose']['available'] and not b['pose']['available'] for a,b in zip(new,base))),
                canonical_damage=damage(base,new),
                uncertainty='reused DEV; original13-session paired cluster bootstrap; CI crossing zero does not establish a difference')


def hypothesis(new, base):
    switches, stable, unavailable = [], [], []
    quadrants = Counter(); switch_records = []
    for a, b in zip(new, base):
        if not (a['pose']['available'] and b['pose']['available'] and a.get('final_hypothesis') and b.get('final_hypothesis')):
            unavailable.append(a['id']); quadrants['UNAVAILABLE'] += 1
            continue
        (switches if a['final_hypothesis'] != b['final_hypothesis'] else stable).append(a['id'])
        dt = a['pose']['translation_cm'] - b['pose']['translation_cm']
        dr = a['pose']['rotation_deg'] - b['pose']['rotation_deg']
        if a['final_hypothesis'] != b['final_hypothesis']:
            switch_records.append(dict(id=a['id'],session=a['session'],grade=a['grade'],
                before_hypothesis=b['final_hypothesis'],after_hypothesis=a['final_hypothesis'],
                translation_delta_cm=dt,rotation_delta_deg=dr))
        quadrants['BOTH_DECREASE' if dt < 0 and dr < 0 else 'BOTH_INCREASE' if dt > 0 and dr > 0
                  else 'T_DECREASE_R_INCREASE' if dt < 0 and dr > 0 else 'T_INCREASE_R_DECREASE'
                  if dt > 0 and dr < 0 else 'ONE_OR_BOTH_EXACTLY_UNCHANGED'] += 1
    return dict(switch_frames=len(switches), no_switch_frames=len(stable), unavailable_frames=len(unavailable),
                switch_ids=switches, no_switch_ids=stable, unavailable_ids=unavailable,
                pose_quadrants=dict(quadrants), switch_records=switch_records,inference_rule_changed=False,
                definition='prediction-only W/D hypothesis; reference-conditioned post-hoc explanation is not inference')


def motion(methods):
    additional, total, pre_cap = [], [], []
    caps, capped_ids = [], []
    status, fallback = Counter(), Counter()
    calls = same_n3 = same_sub = changed_native = changed_final = 0
    for row, sub in zip(methods['N3_THEN_SUBPIX'], methods['SUBPIX']):
        q0, qn, qs, qf = [np.asarray(row[k], float) for k in ('q0','qN','qS','qFinal')]
        mask = np.asarray(row['prediction_support'], bool)[:8] & np.isfinite(q0[:8]).all(axis=1) & ~(q0[:8] == -1).all(axis=1)
        native = np.linalg.norm(qs[:8] - q0[:8], axis=1)
        final = np.linalg.norm(qf[:8] - q0[:8], axis=1)
        cap = float(row['correction']['cap_px'])
        assert np.all(final[mask] <= cap + 1e-10)
        active = mask & (native > cap)
        for k in np.flatnonzero(active):
            caps.append(dict(id=row['id'], corner=int(k), before_px=float(native[k]), cap_px=cap))
        if active.any():
            capped_ids.append(row['id'])
        additional.extend(np.linalg.norm(qs[:8] - qn[:8], axis=1)[mask].tolist())
        total.extend(final[mask].tolist()); pre_cap.extend(native[mask].tolist())
        sqs = np.asarray(sub['qS'], float); sqf = np.asarray(sub['qFinal'], float)
        changed_native += sum(not np.array_equal(qs[k], sqs[k], equal_nan=True) for k in np.flatnonzero(mask))
        changed_final += sum(not np.array_equal(qf[k], sqf[k], equal_nan=True) for k in np.flatnonzero(mask))
        same_n3 += int(np.array_equal(qf,qn,equal_nan=True)); same_sub += int(np.array_equal(qf,sqf,equal_nan=True))
        d = row['correction']['diagnostics']
        calls += int(d['algorithm_corner_calls'])
        status.update(d['status_counts']); fallback.update(d['fallback_counts'])
    return dict(eligible_corners=len(total), SUBPIX_additional_move_from_N3_px=distribution(additional,'px'),
                total_move_from_BASE_px=distribution(total,'px'), before_cap_total_move_from_BASE_px=distribution(pre_cap,'px'),
                cap_active_corners=len(caps), cap_active_frames=len(capped_ids), cap_active_ids=capped_ids,
                cap_corner_records=caps, native_SUBPIX_result_different_from_BASE_start_corners=int(changed_native),
                final_result_different_from_SUBPIX_corners=int(changed_final), final_exactly_N3_frames=same_n3,
                final_exactly_SUBPIX_frames=same_sub, OpenCV_corner_calls=calls,
                status_counts=dict(status), fallback_counts=dict(fallback),
                definition='original supported corners0..7; additional qS-qN; final qFinal-q0; exact equality')


def joint_diagnostics(rows):
    """Post-hoc numerical records only; covariance is an uncertainty proxy."""
    movements, totals, pre_cap = [], [], []
    capped_ids, caps, unchanged = [], 0, 0
    matrix_values = {k: dict(minimum=[],maximum=[],trace=[],missing=0) for k in ('C','Sigma','A')}
    strengths = []
    statuses, fallbacks = Counter(), Counter()
    for row in rows:
        q0,qn,qu,qf = [np.asarray(row[k],float) for k in ('q0','qN','qS','qFinal')]
        valid = np.asarray(row['prediction_support'],bool)[:8] & np.isfinite(q0[:8]).all(1) & ~(q0[:8]==-1).all(1)
        before = np.linalg.norm(qu[:8]-q0[:8],axis=1)
        total = np.linalg.norm(qf[:8]-q0[:8],axis=1)
        active = valid & (before > row['correction']['cap_px'])
        assert np.all(total[valid] <= row['correction']['cap_px'] + 1e-10)
        movements.extend(np.linalg.norm(qu[:8]-qn[:8],axis=1)[valid].tolist())
        totals.extend(total[valid].tolist()); pre_cap.extend(before[valid].tolist())
        caps += int(active.sum())
        if active.any():
            capped_ids.append(row['id'])
        unchanged += int(np.array_equal(qf,qn,equal_nan=True))
        detail = row['correction']['diagnostics']
        statuses.update(detail['status_counts']); fallbacks.update(detail['fallback_counts'])
        for record in detail['corner_records']:
            for key in matrix_values:
                matrix = record.get(key)
                if matrix is None:
                    matrix_values[key]['missing'] += 1
                    continue
                a = np.asarray(matrix,float)
                assert a.shape == (2,2) and np.isfinite(a).all(),(row['id'],key)
                eigen = np.linalg.eigvalsh(a)
                matrix_values[key]['minimum'].append(float(eigen[0]))
                matrix_values[key]['maximum'].append(float(eigen[1]))
                matrix_values[key]['trace'].append(float(np.trace(a)))
            if record.get('S') is not None:
                strengths.append(float(record['S']))
    return dict(eligible_corners=len(totals), unconstrained_move_from_N3_px=distribution(movements,'px'),
                total_move_from_BASE_px=distribution(totals,'px'), before_cap_total_move_from_BASE_px=distribution(pre_cap,'px'),
                cap_active_corners=caps,cap_active_frames=len(capped_ids),cap_active_ids=capped_ids,
                final_exactly_N3_frames=unchanged,status_counts=dict(statuses),fallback_counts=dict(fallbacks),
                posterior_C_eigen_px2={k:distribution(matrix_values['C'][k],'px^2') for k in ('minimum','maximum','trace')},
                stabilized_Sigma_eigen_px2={k:distribution(matrix_values['Sigma'][k],'px^2') for k in ('minimum','maximum','trace')},
                gradient_A_eigen={k:distribution(matrix_values['A'][k],'dimensionless') for k in ('minimum','maximum','trace')},
                matrix_missing_records={k:v['missing'] for k,v in matrix_values.items()},
                gradient_strength_S=distribution(strengths,'fixed gray/Sobel scale'),
                definition='qS means joint unconstrained solution, never cornerSubPix; C weighted222-candidate native variance is an uncapped proxy, not calibrated confidence')


def build(rows):
    grouped, master = group_rows(rows)
    bootstrap = SharedBootstrap(master)
    metrics = dict(schema='pallet_feature_gradient_joint_statistics_v1', status='COMPLETE', by_seed={}, seed_mean={},
                   bootstrap=bootstrap.meta(master),
                   population=dict(images=319,sessions=13,seeds=[1,2,3],rows=5742,
                                   matched_frames=311,reference_corners=2499,observed_corners=2445,
                                   grade_frames=dict(Counter(r['grade'] for r in master))),
                   seed_mean_definition='per original ID arithmetic mean of scalar errors across seeds1/2/3; canonical corner identities averaged; common successful poses only; 319 images, never957 independent images',
                   interpretation=dict(std='sample error dispersion, ddof1; not CI or training-seed dispersion',
                       corner='observed canonical corner pooling; PCK and gross20 full reference denominator',
                       pose='conditional successful frames; full319 operational failures separately retained',
                       mean_CI='all methods share exact original13-session draws including grade subsets',
                       limitations='reused REAL_DEV; geometric reference; three seeds; multiple comparisons unadjusted'),
                   diagnostics=dict(by_seed={},overall={}))
    paired = dict(schema='pallet_feature_gradient_joint_paired_v1', by_seed={}, seed_mean={},
                  bootstrap=bootstrap.meta(master), sign='new minus reference; negative is lower error',
                  seed_mean_definition=metrics['seed_mean_definition'],
                  primary_comparison='FG_JOINT_POSTERIOR_minus_N3_THEN_SUBPIX')
    mean_methods = {m: mean_seed_rows(grouped,m) for m in METHODS}
    for seed, methods in [*grouped.items(), ('seed_mean',mean_methods)]:
        dest = metrics['seed_mean'] if seed == 'seed_mean' else metrics['by_seed'].setdefault(seed,{})
        for subset in SETS:
            dest[subset] = {m: summarize([r for r in rr if subset == 'ALL' or r['grade'] == subset],bootstrap)
                            for m,rr in methods.items()}
        for summary in dest['ALL'].values():
            assert summary['corner']['reference_corners'] == 2499
            assert summary['corner']['observed_corners'] == 2445
            assert summary['corner']['matched_frames'] == 311
        if seed == 'seed_mean':
            for subset in SETS:
                for method in METHODS:
                    corner = dest[subset][method]['corner']
                    corner['threshold_of_seed_mean_errors'] = dict(PCK=corner['PCK'],
                        gross20_count=corner['gross20_count'], gross20_rate=corner['gross20_rate'])
                    seed_corners = [metrics['by_seed'][str(s)][subset][method]['corner'] for s in SEEDS]
                    corner['PCK'] = {str(t): float(np.mean([c['PCK'][str(t)] for c in seed_corners])) for t in (5,10,20)}
                    corner['gross20_count'] = float(np.mean([c['gross20_count'] for c in seed_corners]))
                    corner['gross20_rate'] = float(np.mean([c['gross20_rate'] for c in seed_corners]))
                    corner['threshold_definition'] = 'headline PCK/gross20 arithmetic mean of3 per-seed rates/counts; threshold_of_seed_mean_errors separately thresholds scalar averaged errors; not an ensemble inference'
                    auc = dest[subset][method]['ADDsym_AUC']
                    auc['AUC_of_seed_mean_errors'] = dict(full=auc['full'],conditional=auc['conditional'])
                    for key in ('full','conditional'):
                        vals = [metrics['by_seed'][str(s)][subset][method]['ADDsym_AUC'][key] for s in SEEDS]
                        auc[key] = float(np.mean(vals)) if all(v is not None for v in vals) else None
                    auc['seed_mean_definition'] = 'headline arithmetic mean of3 seed AUCs; AUC_of_seed_mean_errors thresholds averaged per-ID normalized errors separately'
        pdest = paired['seed_mean'] if seed == 'seed_mean' else paired['by_seed'].setdefault(seed,{})
        for subset in SETS:
            pdest[subset] = {}
            selected = {m:[r for r in rr if subset=='ALL' or r['grade']==subset] for m,rr in methods.items()}
            for base in COMPARATORS:
                pdest[subset]['FG_JOINT_POSTERIOR_minus_' + base] = compare(
                    selected['FG_JOINT_POSTERIOR'],selected[base],bootstrap,
                    dest[subset]['FG_JOINT_POSTERIOR'],dest[subset][base])
        if seed != 'seed_mean':
            metrics['diagnostics']['by_seed'][seed] = dict(sequential_motion=motion(methods),
                joint={m:joint_diagnostics(methods[m]) for m in JOINT_METHODS},
                correction_by_method={m:dest['ALL'][m]['correction_diagnostics'] for m in METHODS},
                damage_BASE={m:damage(methods['BASE'],rr) for m,rr in methods.items()},
                damage_N3={m:damage(methods['N3_DIM_SYM'],rr) for m,rr in methods.items()},
                damage_SEQUENTIAL={m:damage(methods['N3_THEN_SUBPIX'],rr) for m,rr in methods.items()},
                hypothesis={m:hypothesis(methods['FG_JOINT_POSTERIOR'],methods[m]) for m in COMPARATORS})
    diag = metrics['diagnostics']
    counts = ('eligible_corners','cap_active_corners','cap_active_frames','OpenCV_corner_calls',
              'final_exactly_N3_frames','final_exactly_SUBPIX_frames')
    diag['overall'] = dict(repeated_seed_frame_executions=957, unique_images=319,
        explanation='counts sum repeated executions on the same319 images; not957 independent observations',
        sequential_motion={k:sum(diag['by_seed'][str(s)]['sequential_motion'][k] for s in SEEDS) for k in counts},
        joint={m:{k:sum(diag['by_seed'][str(s)]['joint'][m][k] for s in SEEDS)
                  for k in ('eligible_corners','cap_active_corners','cap_active_frames','final_exactly_N3_frames')} for m in JOINT_METHODS},
        distinct_joint_cap_active_images={m:len(set(i for s in SEEDS for i in diag['by_seed'][str(s)]['joint'][m]['cap_active_ids'])) for m in JOINT_METHODS},
        damage_BASE={m:{k:sum(diag['by_seed'][str(s)]['damage_BASE'][m][k] for s in SEEDS)
                       for k in ('corners','before_good5','before_bad20','good5_to_bad10','bad20_to_good10')}
                       for m in METHODS},
        hypothesis={m:{k:sum(diag['by_seed'][str(s)]['hypothesis'][m][k] for s in SEEDS)
                          for k in ('switch_frames','no_switch_frames','unavailable_frames')}
                          for m in COMPARATORS})
    metrics['arithmetic_mean_of_seed_metrics'] = {subset:{m:{metric:{key:float(np.mean([
        metrics['by_seed'][str(s)][subset][m]['metrics'][metric][key] for s in SEEDS]))
        if all(metrics['by_seed'][str(s)][subset][m]['metrics'][metric][key] is not None for s in SEEDS) else None
        for key in ('mean','sample_variance','sample_std','median','P90','max')}
        for metric in METRICS} for m in METHODS} for subset in SETS}
    metrics['arithmetic_mean_of_seed_metrics_definition'] = 'auxiliary average of three independently summarized seed metrics; differs from per-ID seed_mean variance/median/P90'
    return metrics, paired


def failures(rows,metrics):
    grouped,_=group_rows(rows)
    output=dict(schema='pallet_feature_gradient_joint_failures_v1',by_seed={},
        definition='pose failures, explicit correction fallback reasons, canonical corner damage and large finite errors are distinct; no favorable exclusion or finite failure substitute')
    for seed,methods in grouped.items():
        output['by_seed'][seed]={}
        for method,rr in methods.items():
            summary=metrics['by_seed'][seed]['ALL'][method]
            output['by_seed'][seed][method]=dict(no_pose_ids=summary['pose']['failure_ids'],
                total_frames=319,available_poses=summary['pose']['available'],
                correction_fallback=summary['correction_diagnostics'],
                fallback_records=[dict(id=r['id'],corner=d['corner'],reason=d.get('fallback_reason',d.get('status')))
                    for r in rr for d in r.get('correction',{}).get('diagnostics',{}).get('corner_records',[])
                    if d.get('fallback_reason')],
                damage_vs_BASE=metrics['diagnostics']['by_seed'][seed]['damage_BASE'][method],
                damage_vs_N3=metrics['diagnostics']['by_seed'][seed]['damage_N3'][method],
                damage_vs_SEQUENTIAL=metrics['diagnostics']['by_seed'][seed]['damage_SEQUENTIAL'][method],
                largest_finite_errors=summary['largest_errors'])
    return output


def write_csv(path, metrics):
    fields = ('seed','method','set','metric','unit','n','mean','variance','std','median','P90','max','CI95_low','CI95_high')
    with Path(path).open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n'); writer.writeheader()
        for seed, groups in [*metrics['by_seed'].items(),('seed_mean',metrics['seed_mean'])]:
            for subset in SETS:
                for method in METHODS:
                    for metric in METRICS:
                        s=groups[subset][method]['metrics'][metric]
                        writer.writerow(dict(seed=seed,method=method,set=subset,metric=metric,unit=s['unit'],n=s['n'],
                            mean=s['mean'],variance=s['sample_variance'],std=s['sample_std'],median=s['median'],P90=s['P90'],max=s['max'],
                            CI95_low=s['CI95'][0] if s['CI95'] else None,CI95_high=s['CI95'][1] if s['CI95'] else None))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--doc',type=Path,default=DEFAULT_DOC)
    args=parser.parse_args()
    rows_path=args.doc/'PREDICTIONS.jsonl.gz'
    metrics,paired=build(read_rows(rows_path))
    metrics['predictions_sha256']=paired['predictions_sha256']=sha(rows_path)
    write(args.doc/'METRICS.json',metrics)
    write(args.doc/'PAIRED.json',paired)
    write(args.doc/'FAILURES.json',failures(read_rows(rows_path),metrics))
    write_csv(args.doc/'METRICS.csv',metrics)
    print(json.dumps(dict(status='COMPLETE',rows=5742,images=319,csv_rows=384,
                          draws_sha256=metrics['bootstrap']['draw_sha256'])))


if __name__ == '__main__':
    main()
