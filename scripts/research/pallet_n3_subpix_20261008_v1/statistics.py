"""Statistics and post-hoc diagnostics from saved rows; no inference or F calls."""
from __future__ import annotations

from collections import Counter

import numpy as np

from scripts.research.pallet_pose_target_6d_20261006_v1.reporting import SharedBootstrap
from scripts.paper.pose_metric_closure_v1.symmetry_aware_pose_metrics import pose_auc

ARMS = ('BASE', 'N3', 'SUBPIX', 'N3_SUBPIX')
METRICS = ('corner_px', 'translation_cm', 'rotation_deg', 'ADDsym_cm')
POSE_FIELDS = {'translation_cm': ('translation_cm', 1.),
               'rotation_deg': ('rotation_deg', 1.), 'ADDsym_cm': ('ADDsym_m', 100.)}
COMPARISONS = (('N3_SUBPIX', 'N3'), ('N3_SUBPIX', 'SUBPIX'),
               ('N3_SUBPIX', 'BASE'), ('N3', 'BASE'), ('SUBPIX', 'BASE'))
CATEGORIES = ('DIRECT_VISIBLE', 'SELF_OCCLUDED', 'EXTERNAL_OCCLUDED',
              'OUT_OF_FRAME', 'UNKNOWN', 'UNANNOTATED')


def distribution(values, unit=None):
    """Keep every finite eligible error; never replace failures or trim tails."""
    values = np.asarray(list(values), dtype=np.float64)
    assert values.ndim == 1 and np.isfinite(values).all(), 'Nonfinite eligible error'
    n = int(values.size)
    variance = float(np.var(values, ddof=1)) if n >= 2 else None
    std = float(np.std(values, ddof=1)) if n >= 2 else None
    result = dict(n=n, mean=float(np.mean(values)) if n else None,
                  sample_variance=variance, sample_std=std,
                  median=float(np.median(values)) if n else None,
                  P90=float(np.quantile(values, .9)) if n else None,
                  min=float(np.min(values)) if n else None,
                  max=float(np.max(values)) if n else None, ddof=1, unit=unit)
    if n >= 2:
        assert np.isclose(std * std, variance, rtol=1e-12, atol=1e-12)
    return result


def observed_mask(row):
    corner = row['corner']
    if not corner['evaluable']:
        return np.zeros(8, dtype=bool)
    valid = np.asarray(corner['canonical_valid'], dtype=bool)
    assert valid.shape == (8,)
    explicit = row.get('canonical_observed', corner.get('canonical_observed'))
    observed = valid & bool(corner['matched']) if explicit is None else np.asarray(explicit, dtype=bool)
    assert observed.shape == (8,) and not (observed & ~valid).any()
    errors = np.asarray(corner['canonical_errors'], dtype=np.float64)
    assert np.isfinite(errors[observed]).all()
    assert np.array_equal(np.sort(errors[observed]),
                          np.sort(np.asarray(corner['observed_errors'], dtype=np.float64)))
    return observed


def summarize(rows):
    corners = [r['corner'] for r in rows if r['corner']['evaluable']]
    full = np.asarray([e for c in corners for e in c['errors']], dtype=np.float64)
    observed = [e for c in corners for e in c['observed_errors']]
    assert np.isfinite(full).all() and (full >= 0).all()
    for row in rows:
        observed_mask(row)
    available = [r['pose'] for r in rows if r['pose']['available']]
    metrics = {'corner_px': distribution(observed, 'px')}
    for metric, (field, factor) in POSE_FIELDS.items():
        metrics[metric] = distribution([float(r[field]) * factor for r in available],
                                       'degree' if metric == 'rotation_deg' else 'cm')
    meters = distribution([r['ADDsym_m'] for r in available], 'm')
    cm = metrics['ADDsym_cm']
    conversion = {}
    for field, factor in (('mean', 100.), ('sample_std', 100.), ('sample_variance', 10000.)):
        passed = (cm[field] is None and meters[field] is None) or bool(
            np.isclose(cm[field], meters[field] * factor, rtol=1e-12, atol=1e-12))
        assert passed, ('ADDsym conversion', field)
        conversion[field] = dict(factor=factor, passed=passed)
    normalized = np.asarray([r['pose']['ADDsym_normalized'] if r['pose']['available']
                             else np.inf for r in rows], dtype=np.float64)
    assert not np.isnan(normalized).any() and (normalized >= 0).all()
    selected = Counter(str(r.get('fixed_metadata', {}).get('selected_index', r.get('selected_index')))
                       for r in rows)
    return dict(total_frames=len(rows), metrics=metrics,
                corner=dict(reference_corners=int(len(full)), observed_corners=len(observed),
                            evaluable_frames=len(corners),
                            matched_frames=sum(bool(r['corner']['matched']) for r in rows),
                            detected_frames=sum(bool(r['corner']['detected']) for r in rows),
                            PCK={str(t): float(np.mean(full <= t)) if len(full) else None
                                 for t in (5, 10, 20)},
                            gross20_rate=float(np.mean(full > 20)) if len(full) else None,
                            gross20_count=int(np.sum(full > 20)),
                            full_reference_error_px=distribution(full, 'px')),
                pose=dict(available=len(available), failures=len(rows) - len(available),
                          failure_rate=(len(rows) - len(available)) / len(rows) if rows else None,
                          coverage=len(available) / len(rows) if rows else None),
                ADDsym_AUC=dict(full=pose_auc(normalized, 1.) if rows else None,
                                conditional=pose_auc(normalized[np.isfinite(normalized)], 1.)
                                if available else None, full_frames=len(rows),
                                definition='existing ADDsym_normalized; 0..0.1/1001 thresholds; failures inf'),
                ADDsym_m_to_cm_check=conversion, selected_index_counts=dict(selected),
                selected_index_source='fixed_metadata selected detection; old scorer action-index field is not detection selection',
                final_hypothesis_counts=dict(Counter(r.get('final_hypothesis') or 'UNAVAILABLE'
                                                    for r in rows)))


def damage(before, after):
    """Existing canonical identity, full-reference good/bad thresholds."""
    assert [r['id'] for r in before] == [r['id'] for r in after]
    a, b, frame_delta = [], [], []
    for old, new in zip(before, after):
        oc, nc = old['corner'], new['corner']
        assert oc['evaluable'] == nc['evaluable']
        if not oc['evaluable']:
            continue
        assert oc['canonical_valid'] == nc['canonical_valid']
        valid = np.asarray(oc['canonical_valid'], dtype=bool)
        a.extend(np.asarray(oc['canonical_errors'], dtype=np.float64)[valid].tolist())
        b.extend(np.asarray(nc['canonical_errors'], dtype=np.float64)[valid].tolist())
        frame_delta.append(nc['frame_mean_px'] - oc['frame_mean_px'])
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    d = np.asarray(frame_delta, dtype=np.float64)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    return dict(corners=len(a), before_good5=int(np.sum(a < 5)), before_bad20=int(np.sum(a > 20)),
                good5_to_bad10=int(np.sum((a < 5) & (b > 10))),
                bad20_to_good10=int(np.sum((a > 20) & (b < 10))),
                improved_frames=int(np.sum(d < -1e-9)), harmed_frames=int(np.sum(d > 1e-9)),
                unchanged_frames=int(np.sum(np.abs(d) <= 1e-9)),
                canonical_GT_identity_aligned=True,
                definition='before<5px and after>10px; before>20px and after<10px; full canonical reference')


def paired_values(new, base, metric):
    assert [r['id'] for r in new] == [r['id'] for r in base]
    result = []
    for a, b in zip(new, base):
        if metric == 'corner_px':
            assert a['corner']['canonical_valid'] == b['corner']['canonical_valid']
            mask = observed_mask(a) & observed_mask(b)
            av = np.asarray(a['corner'].get('canonical_errors', [None] * 8), dtype=np.float64)
            bv = np.asarray(b['corner'].get('canonical_errors', [None] * 8), dtype=np.float64)
            delta = av[mask] - bv[mask]
        else:
            field, factor = POSE_FIELDS[metric]
            delta = [factor * (a['pose'][field] - b['pose'][field])] if (
                a['pose']['available'] and b['pose']['available']) else []
        values = np.asarray(delta, dtype=np.float64)
        assert np.isfinite(values).all()
        result.append(values)
    return result


def pooled_contrast(values, bootstrap, unit):
    """Apply the shared original-session draws to original corner/frame counts."""
    counts = np.asarray([len(v) for v in values], dtype=np.float64)
    totals = np.asarray([np.sum(v, dtype=np.float64) for v in values], dtype=np.float64)
    session_counts = np.bincount(bootstrap.inverse, weights=counts, minlength=len(bootstrap.units))
    session_totals = np.bincount(bootstrap.inverse, weights=totals, minlength=len(bootstrap.units))
    denominator = bootstrap.weights @ session_counts
    numerator = bootstrap.weights @ session_totals
    keep = denominator > 0
    draws = numerator[keep] / denominator[keep]
    pooled = np.concatenate(values) if values else np.asarray([], dtype=np.float64)
    eligible_sessions = int(np.count_nonzero(session_counts))
    session_means = session_totals[session_counts > 0] / session_counts[session_counts > 0]
    return dict(mean_paired_difference=float(np.mean(pooled)) if len(pooled) else None,
                median_paired_difference=float(np.median(pooled)) if len(pooled) else None,
                CI95=np.quantile(draws, [.025, .975]).tolist()
                if len(draws) and eligible_sessions > 1 else None,
                common_eligible_observations=len(pooled),
                common_eligible_frames=int(np.count_nonzero(counts)),
                excluded_frames=int(np.sum(counts == 0)), full_master_frames=len(values),
                eligible_sessions=eligible_sessions, valid_resamples=int(np.sum(keep)),
                empty_resamples=int(np.sum(~keep)), unit=unit,
                observation_weighting='original observed canonical corners' if unit == 'px'
                else 'original common successful frames',
                auxiliary_session_equal_weight_mean=float(np.mean(session_means))
                if eligible_sessions else None,
                draw_sha256=bootstrap.draw_sha256, seed=20260917, resamples=10000)


def pose_quadrants(new, base):
    hist = Counter(); common = 0
    for a, b in zip(new, base):
        if not (a['pose']['available'] and b['pose']['available']):
            hist['UNAVAILABLE'] += 1
            continue
        common += 1
        dt = a['pose']['translation_cm'] - b['pose']['translation_cm']
        dr = a['pose']['rotation_deg'] - b['pose']['rotation_deg']
        if dt < 0 and dr < 0:
            state = 'BOTH_DECREASE'
        elif dt > 0 and dr > 0:
            state = 'BOTH_INCREASE'
        elif dt < 0 and dr > 0:
            state = 'T_DECREASE_R_INCREASE'
        elif dt > 0 and dr < 0:
            state = 'T_INCREASE_R_DECREASE'
        else:
            state = 'ONE_OR_BOTH_EXACTLY_UNCHANGED'
        hist[state] += 1
    return dict(full_frames=len(new), common_pose_frames=common,
                counts={k:hist[k] for k in ('BOTH_DECREASE', 'BOTH_INCREASE',
                    'T_DECREASE_R_INCREASE', 'T_INCREASE_R_DECREASE',
                    'ONE_OR_BOTH_EXACTLY_UNCHANGED', 'UNAVAILABLE')})


def compare(new, base, bootstrap):
    a = np.asarray([r['pose']['available'] for r in new], dtype=bool)
    b = np.asarray([r['pose']['available'] for r in base], dtype=bool)
    paired = {metric: pooled_contrast(paired_values(new, base, metric), bootstrap,
              'px' if metric == 'corner_px' else 'degree' if metric == 'rotation_deg' else 'cm')
              for metric in METRICS}
    paired['corner_px']['excluded_reference_corners'] = (
        sum(r['corner'].get('corners', 0) for r in base) - paired['corner_px']['common_eligible_observations'])
    sa, sb = summarize(new), summarize(base)
    return dict(statistics=paired,
                difference_of_marginal_statistics={metric: {
                    key: sa['metrics'][metric][key] - sb['metrics'][metric][key]
                    if sa['metrics'][metric][key] is not None and sb['metrics'][metric][key] is not None else None
                    for key in ('mean', 'median', 'P90', 'sample_variance', 'sample_std')}
                    for metric in METRICS},
                coverage=dict(full_frames=len(new), common_success=int(np.sum(a & b)),
                              new_failures=int(np.sum(~a)), base_failures=int(np.sum(~b)),
                              new_only_success=int(np.sum(a & ~b)), base_only_success=int(np.sum(b & ~a)),
                              both_failed=int(np.sum(~a & ~b))),
                canonical_damage=damage(base, new), pose_quadrants=pose_quadrants(new, base),
                uncertainty='13-session repeated-use DEV exploratory paired cluster bootstrap; CI including zero is uncertainty, not equivalence')


def hypothesis_layers(new, base):
    groups = {state: [] for state in ('SWITCH', 'NO_SWITCH', 'UNAVAILABLE')}
    for i, (a, b) in enumerate(zip(new, base)):
        state = 'UNAVAILABLE'
        if a['pose']['available'] and b['pose']['available'] and a.get('final_hypothesis') and b.get('final_hypothesis'):
            state = 'SWITCH' if a['final_hypothesis'] != b['final_hypothesis'] else 'NO_SWITCH'
        groups[state].append(i)
    result = {}
    for state, indices in groups.items():
        aa, bb = [new[i] for i in indices], [base[i] for i in indices]
        result[state] = dict(frames=len(indices), ids=[r['id'] for r in aa],
                            mean_error_change={metric: distribution(
                                np.concatenate(paired_values(aa, bb, metric)) if indices else [],
                                'px' if metric == 'corner_px' else 'degree' if metric == 'rotation_deg' else 'cm')
                                for metric in METRICS})
    return dict(groups=result, inference_rule_changed=False,
                definition='actual final prediction-only width/depth hypothesis; post-hoc subdivision only')


def visibility_layers(methods, labels):
    output = {}
    for name, rows in methods.items():
        groups = {state: [] for state in CATEGORIES}
        for row, raw, n3 in zip(rows, methods['BASE'], methods['N3']):
            c = row['corner']
            if not c['evaluable']:
                continue
            obs = observed_mask(row)
            for k, valid in enumerate(c['canonical_valid']):
                if valid:
                    state = labels.get(('DEV319', row['id'], k), 'UNANNOTATED')
                    assert state in groups
                    groups[state].append((row['id'], c['canonical_errors'][k], bool(obs[k]),
                                          raw['corner']['canonical_errors'][k],
                                          n3['corner']['canonical_errors'][k]))
        output[name] = {}
        for state, values in groups.items():
            errors = np.asarray([v[1] for v in values], dtype=np.float64)
            raw = np.asarray([v[3] for v in values], dtype=np.float64)
            n3 = np.asarray([v[4] for v in values], dtype=np.float64)
            output[name][state] = dict(reference_corners=len(values),
                observed_corners=sum(v[2] for v in values), frames=len({v[0] for v in values}),
                observed_error_px=distribution([v[1] for v in values if v[2]], 'px'),
                full_PCK10=float(np.mean(errors <= 10)) if len(values) else None,
                full_gross20=float(np.mean(errors > 20)) if len(values) else None,
                BASE_good5_to_bad10=int(np.sum((raw < 5) & (errors > 10))),
                BASE_bad20_to_good10=int(np.sum((raw > 20) & (errors < 10))),
                N3_good5_to_bad10=int(np.sum((n3 < 5) & (errors > 10))),
                N3_bad20_to_good10=int(np.sum((n3 > 20) & (errors < 10))))
    return dict(methods=output, GT_used_in_inference=False,
                definition='existing canonical human visibility used after predictions; no stratum-specific symmetry branch or inference gate')


def motion_diagnostics(methods):
    combined, sub = methods['N3_SUBPIX'], methods['SUBPIX']
    additional, total, pre_cap = [], [], []
    cap_corners = cap_frames = changed_native = changed_final = identical_n3 = identical_sub = 0
    statuses = Counter(); fallbacks = Counter(); calls = 0
    for row, sr in zip(combined, sub):
        q0, qn, qs, qf = (np.asarray(row[k], dtype=np.float64) for k in ('q0', 'qN', 'qS', 'qFinal'))
        sqs = np.asarray(sr.get('qS', sr['native_points']), dtype=np.float64)
        sqf = np.asarray(sr.get('qFinal', sr['native_points']), dtype=np.float64)
        support = np.asarray(row['prediction_support'], dtype=bool)
        usable = support[:8] & np.isfinite(q0[:8]).all(-1) & ~np.all(q0[:8] == -1, axis=-1)
        final_move = np.linalg.norm(qf[:8] - q0[:8], axis=-1)
        before_move = np.linalg.norm(qs[:8] - q0[:8], axis=-1)
        add_move = np.linalg.norm(qs[:8] - qn[:8], axis=-1)
        cap = float(row['correction']['cap_px'])
        active = usable & (before_move > cap)
        assert np.all(final_move[usable] <= cap + 1e-10)
        additional.extend(add_move[usable].tolist()); total.extend(final_move[usable].tolist())
        pre_cap.extend(before_move[usable].tolist())
        cap_corners += int(np.sum(active)); cap_frames += int(np.any(active))
        changed_native += int(sum(not np.array_equal(qs[k], sqs[k], equal_nan=True)
                                  for k in np.flatnonzero(usable)))
        changed_final += int(sum(not np.array_equal(qf[k], sqf[k], equal_nan=True)
                                 for k in np.flatnonzero(usable)))
        identical_n3 += int(np.array_equal(qf, qn, equal_nan=True))
        identical_sub += int(np.array_equal(qf, sqf, equal_nan=True))
        d = row['correction']['diagnostics']
        calls += int(d['algorithm_corner_calls'])
        statuses.update(d['status_counts']); fallbacks.update(d['fallback_counts'])
    return dict(eligible_corners=len(total), SUBPIX_additional_move_from_N3_px=distribution(additional, 'px'),
                total_move_from_BASE_px=distribution(total, 'px'),
                before_cap_total_move_from_BASE_px=distribution(pre_cap, 'px'),
                cap_active_corners=cap_corners, cap_active_frames=cap_frames,
                native_SUBPIX_result_different_from_BASE_start_corners=changed_native,
                final_result_different_from_SUBPIX_corners=changed_final,
                final_exactly_N3_frames=identical_n3, final_exactly_SUBPIX_frames=identical_sub,
                OpenCV_corner_calls=calls, status_counts=dict(statuses), fallback_counts=dict(fallbacks),
                definition='eligible original prediction-supported corners0..7; additional move is qS-qN; total cap anchored at q0; equality exact')


def aggregate(methods, labels=None):
    assert tuple(methods) == ARMS
    ids = [r['id'] for r in methods['BASE']]
    assert len(ids) == len(set(ids)) == 319
    for rows in methods.values():
        assert [r['id'] for r in rows] == ids
        assert [r['session'] for r in rows] == [r['session'] for r in methods['BASE']]
    sessions = [r['session'] for r in methods['BASE']]
    assert len(set(sessions)) == 13
    bootstrap = SharedBootstrap(ids, sessions)
    summaries = {name: summarize(rows) for name, rows in methods.items()}
    for s in summaries.values():
        assert s['corner']['reference_corners'] == 2499
        assert s['corner']['observed_corners'] == 2445
        assert s['corner']['matched_frames'] == 311
    comparisons = {f'{new}_minus_{base}': compare(methods[new], methods[base], bootstrap)
                   for new, base in COMPARISONS}
    return dict(status='DONE', full_frames=319, sessions=13, seed=1,
                summaries=summaries, paired_comparisons=comparisons, bootstrap=bootstrap.meta(),
                RAW_canonical_damage={name: damage(methods['BASE'], rows) for name, rows in methods.items()},
                N3_canonical_damage={name: damage(methods['N3'], rows) for name, rows in methods.items()},
                motion=motion_diagnostics(methods),
                hypothesis={f'N3_SUBPIX_minus_{name}': hypothesis_layers(methods['N3_SUBPIX'], methods[name])
                            for name in ('N3', 'BASE')},
                visibility=visibility_layers(methods, labels) if labels is not None else dict(status='MISSING_VISIBILITY'),
                interpretation=dict(std='sample error dispersion across eligible frames/corners; ddof1; not CI, standard error, temporal jitter or variation between training seeds',
                                    corner='conditional observed canonical corner pooling; original corner weights',
                                    pose='nonnegative reference errors; conditional successful-pose original frame weights',
                                    failures='full319 retained separately; no zero or arbitrary finite error replacement',
                                    bootstrap='10000 same original13-session multinomial draws seed20260917 for all metrics/comparisons; original observation pooling; no equivalence inference'),
                execution=dict(new_model_forwards=0, new_final_F=0, new_optimizer_updates=0))
