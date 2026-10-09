"""Saved-row statistics only: no image correction, model inference, or pose solver."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

EXISTING_ARMS = ('BASE', 'N3', 'SUBPIX', 'N3_SUBPIX')
NEW_ARMS = tuple(f'{start}_{method}_{limit}' for method in ('WIDE', 'BOUNDARY')
                 for start in ('BASE', 'N3') for limit in ('NATIVE', 'CAP1'))
ARMS = (*EXISTING_ARMS, *NEW_ARMS)
METRICS = ('corner_px', 'translation_cm', 'rotation_deg', 'ADDsym_cm')
POSE_FIELDS = {'translation_cm': ('translation_cm', 1.),
               'rotation_deg': ('rotation_deg', 1.), 'ADDsym_cm': ('ADDsym_m', 100.)}
CATEGORIES = ('DIRECT_VISIBLE', 'SELF_OCCLUDED', 'EXTERNAL_OCCLUDED',
              'OUT_OF_FRAME', 'UNKNOWN', 'UNANNOTATED')
SEED = 20260917
RESAMPLES = 10000
VISIBILITY_RELATIVE = Path('_docs/experiments/pallet_combined_closeout_20261003_v1/'
    'closeout_20261006_v1/visibility_square/STATIC_VISIBILITY_MERGE_AUDIT.json')
TARGETS_RELATIVE = Path('data/pallet/results/pallet_posefix_replay_diagnosis_v1/TARGETS.json')
PRIOR_SUMMARY_RELATIVE = Path('_docs/experiments/pallet_n3_subpix_20261008_v1/SUMMARY.json')


def distribution(values, unit='px'):
    values = np.asarray(list(values), dtype=np.float64)
    assert values.ndim == 1 and np.isfinite(values).all(), 'Nonfinite eligible error'
    n = len(values)
    return dict(n=n, mean=float(values.mean()) if n else None,
                sample_variance=float(values.var(ddof=1)) if n > 1 else None,
                sample_std=float(values.std(ddof=1)) if n > 1 else None,
                median=float(np.median(values)) if n else None,
                P90=float(np.quantile(values, .9)) if n else None,
                min=float(values.min()) if n else None,
                max=float(values.max()) if n else None, ddof=1, unit=unit)


class SharedBootstrap:
    """Exact existing multinomial session draws, reused with every observation mask."""
    def __init__(self, ids, sessions, resamples=RESAMPLES):
        self.ids = list(ids)
        assert len(self.ids) == len(set(self.ids)) > 0
        assert len(sessions) == len(self.ids)
        self.units, self.inverse = np.unique(sessions, return_inverse=True)
        self.counts = np.bincount(self.inverse, minlength=len(self.units))
        n = len(self.units)
        draws = np.random.default_rng(SEED).multinomial(n, np.full(n, 1 / n),
            size=resamples).astype('uint16')
        self.draw_sha256 = hashlib.sha256(draws.tobytes()).hexdigest()
        self.weights = draws.astype(np.float64)
        self.resamples = resamples

    def meta(self):
        return dict(seed=SEED, resamples=self.resamples, units=len(self.units),
                    level='original_scenario_or_session', full_master_frames=len(self.ids),
                    draw_sha256=self.draw_sha256,
                    group_sizes=dict(zip(self.units.tolist(), self.counts.tolist())),
                    weighting='uniform original-session draws; retain every original corner/frame observation; same draws for all methods and masks',
                    multiplicity_adjusted=False, N3_training_seed=1,
                    sample_std_is='sample spread across observed errors; not a CI or variation across training seeds')


def pooled_contrast(values, bootstrap, unit):
    """Pool individual observations within each sampled session, using their counts."""
    assert len(values) == len(bootstrap.ids)
    arrays = [np.asarray(v, dtype=np.float64) for v in values]
    assert all(a.ndim == 1 and np.isfinite(a).all() for a in arrays)
    counts = np.asarray([len(v) for v in arrays], dtype=np.float64)
    totals = np.asarray([v.sum() for v in arrays], dtype=np.float64)
    session_counts = np.bincount(bootstrap.inverse, weights=counts, minlength=len(bootstrap.units))
    session_totals = np.bincount(bootstrap.inverse, weights=totals, minlength=len(bootstrap.units))
    denominator = bootstrap.weights @ session_counts
    keep = denominator > 0
    draws = (bootstrap.weights @ session_totals)[keep] / denominator[keep]
    pooled = np.concatenate(arrays)
    eligible_sessions = int(np.count_nonzero(session_counts))
    return dict(mean_paired_difference=float(pooled.mean()) if len(pooled) else None,
                median_paired_difference=float(np.median(pooled)) if len(pooled) else None,
                CI95=np.quantile(draws, [.025, .975]).tolist()
                if len(draws) and eligible_sessions > 1 else None,
                common_eligible_observations=len(pooled),
                common_eligible_frames=int(np.count_nonzero(counts)),
                excluded_frames=int(np.sum(counts == 0)), full_master_frames=len(values),
                eligible_sessions=eligible_sessions, valid_resamples=int(keep.sum()),
                empty_resamples=int((~keep).sum()), unit=unit,
                observation_weighting='original observed canonical corners' if unit == 'px'
                else 'original common successful frames',
                auxiliary_session_equal_weight_mean=float(np.mean(
                    session_totals[session_counts > 0] / session_counts[session_counts > 0]))
                if eligible_sessions else None,
                draw_sha256=bootstrap.draw_sha256, seed=SEED, resamples=bootstrap.resamples)


def observed_mask(row):
    c = row['corner']
    if not c['evaluable']:
        return np.zeros(8, dtype=bool)
    valid = np.asarray(c['canonical_valid'], dtype=bool)
    observed = np.asarray(row['canonical_observed'], dtype=bool)
    assert valid.shape == observed.shape == (8,) and not (observed & ~valid).any()
    error = np.asarray(c['canonical_errors'], dtype=np.float64)
    assert np.isfinite(error[valid]).all()
    assert np.array_equal(np.sort(error[observed]), np.sort(c['observed_errors']))
    return observed


def pose_auc(rows):
    values = np.asarray([r['pose']['ADDsym_normalized'] if r['pose']['available']
                         else np.inf for r in rows], dtype=np.float64)
    assert not np.isnan(values).any() and (values >= 0).all()
    thresholds = np.linspace(0., .1, 1001)
    curve = np.asarray([(values <= t).mean() for t in thresholds])
    integrate = getattr(np, 'trapezoid', None) or np.trapz
    return float(integrate(curve, thresholds) / .1)


def summarize(rows):
    full = [e for r in rows if r['corner']['evaluable'] for e in r['corner']['errors']]
    observed = [e for r in rows if r['corner']['evaluable'] for e in r['corner']['observed_errors']]
    for row in rows:
        observed_mask(row)
    available = [r['pose'] for r in rows if r['pose']['available']]
    metrics = {'corner_px': distribution(observed)}
    for metric, (field, factor) in POSE_FIELDS.items():
        metrics[metric] = distribution([p[field] * factor for p in available],
            'degree' if metric == 'rotation_deg' else 'cm')
    meters = distribution([p['ADDsym_m'] for p in available], 'm')
    conversion = {}
    for key, factor in (('mean', 100.), ('sample_std', 100.), ('sample_variance', 10000.)):
        a, b = metrics['ADDsym_cm'][key], meters[key]
        passed = (a is None and b is None) or bool(np.isclose(a, factor * b, rtol=1e-12, atol=1e-12))
        assert passed
        conversion[key] = dict(factor=factor, passed=passed)
    errors = np.asarray(full)
    return dict(total_frames=len(rows), metrics=metrics,
                corner=dict(reference_corners=len(full), observed_corners=len(observed),
                    evaluable_frames=sum(r['corner']['evaluable'] for r in rows),
                    matched_frames=sum(r['corner']['matched'] for r in rows),
                    detected_frames=sum(r['corner']['detected'] for r in rows),
                    PCK={str(t): float((errors <= t).mean()) if len(errors) else None for t in (5, 10, 20)},
                    gross20_count=int((errors > 20).sum()),
                    gross20_rate=float((errors > 20).mean()) if len(errors) else None,
                    full_reference_error_px=distribution(full)),
                pose=dict(available=len(available), failures=len(rows) - len(available),
                    failure_rate=(len(rows) - len(available)) / len(rows)),
                ADDsym_AUC=dict(full=pose_auc(rows),
                    conditional=pose_auc([r for r in rows if r['pose']['available']]) if available else None,
                    full_frames=len(rows), definition='existing normalized ADDsym; 0..0.1/1001 thresholds; failures inf'),
                ADDsym_m_to_cm_check=conversion,
                final_hypothesis_counts=dict(Counter(r.get('final_hypothesis') or 'UNAVAILABLE' for r in rows)))


def corner_damage(before, after):
    a, b = np.asarray(before, dtype=float), np.asarray(after, dtype=float)
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    delta = b - a
    return dict(corners=len(a), before_good5=int((a < 5).sum()),
                before_bad20=int((a > 20).sum()),
                good5_to_bad10=int(((a < 5) & (b > 10)).sum()),
                worsened_by_over_1px=int((delta > 1).sum()),
                good5_worsened_by_over_1px=int(((a < 5) & (delta > 1)).sum()),
                bad20_to_good5=int(((a > 20) & (b <= 5)).sum()),
                bad20_to_good10=int(((a > 20) & (b <= 10)).sum()),
                improved_corners=int((delta < -1e-9).sum()),
                harmed_corners=int((delta > 1e-9).sum()),
                unchanged_corners=int((np.abs(delta) <= 1e-9).sum()),
                definition='canonical GT identity; full-reference errors retain existing missing penalty; good<5→bad>10; worsening>1; bad>20→good<=5/10')


def paired_values(new, base, metric, eligible_corners=None):
    result = []
    assert [r['id'] for r in new] == [r['id'] for r in base]
    for a, b in zip(new, base):
        if metric == 'corner_px':
            assert a['corner']['canonical_valid'] == b['corner']['canonical_valid']
            mask = observed_mask(a) & observed_mask(b)
            if eligible_corners is not None:
                mask &= np.asarray([(a['id'], k) in eligible_corners for k in range(8)], dtype=bool)
            value = np.asarray(a['corner']['canonical_errors'], float)[mask] - np.asarray(b['corner']['canonical_errors'], float)[mask]
        else:
            field, factor = POSE_FIELDS[metric]
            value = [factor * (a['pose'][field] - b['pose'][field])] if a['pose']['available'] and b['pose']['available'] else []
        result.append(np.asarray(value, dtype=np.float64))
    return result


def compare(new, base, bootstrap, summaries, corner_groups=None):
    av, bv = [np.asarray([r['pose']['available'] for r in arm], dtype=bool) for arm in (new, base)]
    statistics = {metric: pooled_contrast(paired_values(new, base, metric), bootstrap,
        'px' if metric == 'corner_px' else 'degree' if metric == 'rotation_deg' else 'cm') for metric in METRICS}
    statistics['corner_px']['excluded_reference_corners'] = sum(r['corner'].get('corners', 0) for r in base) - statistics['corner_px']['common_eligible_observations']
    for name, eligible in (corner_groups or {}).items():
        stat = pooled_contrast(paired_values(new, base, 'corner_px', eligible), bootstrap, 'px')
        stat.update(reference_corners=len(eligible),
            excluded_reference_corners=len(eligible) - stat['common_eligible_observations'],
            group=name, definition='same original 13-session draws; original common observed canonical GT corners in the fixed post-hoc label group; no subgroup-specific symmetry branch')
        statistics[f'{name}_corner_px'] = stat
    before, after = [], []
    frame_delta = []
    quadrants = Counter()
    for old, newrow in zip(base, new):
        c, nc = old['corner'], newrow['corner']
        if c['evaluable']:
            valid = np.asarray(c['canonical_valid'], bool)
            before.extend(np.asarray(c['canonical_errors'], float)[valid])
            after.extend(np.asarray(nc['canonical_errors'], float)[valid])
            frame_delta.append(nc['frame_mean_px'] - c['frame_mean_px'])
        if old['pose']['available'] and newrow['pose']['available']:
            dt = newrow['pose']['translation_cm'] - old['pose']['translation_cm']
            dr = newrow['pose']['rotation_deg'] - old['pose']['rotation_deg']
            key = 'BOTH_DECREASE' if dt < 0 and dr < 0 else 'BOTH_INCREASE' if dt > 0 and dr > 0 else 'T_DECREASE_R_INCREASE' if dt < 0 and dr > 0 else 'T_INCREASE_R_DECREASE' if dt > 0 and dr < 0 else 'ONE_OR_BOTH_EXACTLY_UNCHANGED'
        else:
            key = 'UNAVAILABLE'
        quadrants[key] += 1
    damage = corner_damage(before, after)
    fd = np.asarray(frame_delta)
    damage.update(improved_frames=int((fd < -1e-9).sum()), harmed_frames=int((fd > 1e-9).sum()), unchanged_frames=int((np.abs(fd) <= 1e-9).sum()))
    marginal = {}
    for metric in METRICS:
        aa, bb = [s['metrics'][metric] for s in summaries]
        marginal[metric] = {key: aa[key] - bb[key] if aa[key] is not None and bb[key] is not None else None
            for key in ('mean', 'sample_variance', 'sample_std', 'median', 'P90')}
    return dict(statistics=statistics, difference_of_marginal_statistics=marginal,
                coverage=dict(full_frames=len(new), common_success=int((av & bv).sum()),
                    new_failures=int((~av).sum()), base_failures=int((~bv).sum()),
                    new_only_success=int((av & ~bv).sum()), base_only_success=int((bv & ~av).sum()), both_failed=int((~av & ~bv).sum())),
                canonical_damage=damage, pose_quadrants=dict(quadrants),
                hypothesis_changes=sum(a.get('final_hypothesis') != b.get('final_hypothesis') for a, b in zip(new, base)),
                signs='new minus reference; negative improves',
                uncertainty='exploratory repeated-use DEV paired session bootstrap; CI including zero is uncertainty, not equivalence; no multiplicity adjustment')


def point_summary(records, method):
    error = np.asarray([x['errors'][method] for x in records], dtype=float)
    observed = np.asarray([x['observed'][method] for x in records], dtype=bool)
    return dict(reference_corners=len(records), observed_corners=int(observed.sum()),
                unobserved_corners=int((~observed).sum()), frames=len({r['id'] for r in records}),
                full_reference_error_px=distribution(error), observed_error_px=distribution(error[observed]),
                full_PCK={str(t): float((error <= t).mean()) if len(error) else None for t in (5, 10, 20)},
                observed_PCK={str(t): float((error[observed] <= t).mean()) if observed.any() else None for t in (5, 10, 20)},
                full_gross20_count=int((error > 20).sum()),
                full_gross20_rate=float((error > 20).mean()) if len(error) else None,
                observed_gross20_count=int((error[observed] > 20).sum()),
                observed_gross20_rate=float((error[observed] > 20).mean()) if observed.any() else None,
                damage_from={base: corner_damage([r['errors'][base] for r in records], error) for base in ('BASE', 'N3')})


def reference_records(methods, labels, targets=None):
    records = []
    unknown = Counter()
    checks = Counter()
    for index, base in enumerate(methods['BASE']):
        if not base['corner']['evaluable']:
            continue
        valid = np.asarray(base['corner']['canonical_valid'], bool)
        target = targets[base['id']] if targets is not None else None
        for method, rows in methods.items():
            row = rows[index]
            assert row['corner']['canonical_valid'] == valid.tolist()
            observed = observed_mask(row)
            if target is not None:
                gt = np.asarray(target['gt'], float)
                assert np.array_equal(valid, np.asarray(target['valid'], bool)[:8])
                permutation = np.asarray(target['permutations'][row['corner']['branch']], int)
                points = np.asarray(row['native_points'], float)
                pv = np.isfinite(points).all(-1) & ~np.all(points == -1, axis=-1) & bool(row['corner']['matched'])
                canonical_points = np.empty_like(points)
                canonical_points[permutation] = points
                canonical_support = np.zeros(9, bool)
                canonical_support[permutation] = pv
                expected = np.where(canonical_support, np.linalg.norm(canonical_points - gt, axis=-1), np.hypot(*row['raw_hw']))
                assert np.allclose(expected[:8][valid], np.asarray(row['corner']['canonical_errors'], float)[valid], atol=1e-7, rtol=0)
                assert np.array_equal(observed, canonical_support[:8] & valid)
                checks['native_to_saved_whole_object_branch_verified_rows'] += 1
        for k in np.flatnonzero(valid):
            label = labels.get((base['id'], int(k)))
            category = 'UNANNOTATED' if label is None else label['category']
            assert category in CATEGORIES
            source = (label or {}).get('label_evidence', {}).get('annotation_coordinate_source') or 'UNRECORDED'
            reference_source = (label or {}).get('reference_source') or 'UNRECORDED'
            if target is not None and label is not None:
                assert np.allclose(label['reference_xy'], target['gt'][k], atol=1e-7, rtol=0)
                checks['visibility_reference_coordinate_equality_verified'] += 1
            unknown[category] += 1
            records.append(dict(id=base['id'], corner_id=int(k), category=category,
                annotation_coordinate_source=source, reference_source=reference_source,
                errors={m: rows[index]['corner']['canonical_errors'][k] for m, rows in methods.items()},
                observed={m: bool(observed_mask(rows[index])[k]) for m, rows in methods.items()}))
    return records, dict(checks), dict(unknown)


def aggregate(methods, labels, targets=None, *, allow_existing_only=False):
    required = EXISTING_ARMS if allow_existing_only else ARMS
    assert set(methods) == set(required), ('Unexpected/missing arms', set(methods), set(required))
    methods = {m: methods[m] for m in required}
    ids = [r['id'] for r in methods['BASE']]
    sessions = [r['session'] for r in methods['BASE']]
    assert len(ids) == len(set(ids)) == 319 and len(set(sessions)) == 13
    for arm, rows in methods.items():
        assert [r['id'] for r in rows] == ids and [r['session'] for r in rows] == sessions, arm
    bootstrap = SharedBootstrap(ids, sessions)
    assert bootstrap.draw_sha256 == '63e288a51d7b0612616beefac28fcc625e76e8c5d7b5a0ecc5e8b85c73048fa5'
    summaries = {m: summarize(rows) for m, rows in methods.items()}
    records, checks, counts = reference_records(methods, labels, targets)
    filters = {state: [r for r in records if r['category'] == state] for state in CATEGORIES}
    filters.update(ALL=records,
        INVISIBLE_IN_FRAME=[r for r in records if r['category'] in ('SELF_OCCLUDED', 'EXTERNAL_OCCLUDED')],
        CANONICAL_0_3=[r for r in records if r['corner_id'] in (0, 3)],
        DIRECT_VISIBLE_0_3=[r for r in records if r['corner_id'] in (0, 3) and r['category'] == 'DIRECT_VISIBLE'])
    paired_corner_groups = {name: {(r['id'], r['corner_id']) for r in filters[name]}
        for name in ('DIRECT_VISIBLE', 'DIRECT_VISIBLE_0_3')}
    groups = {name: {m: point_summary(rr, m) for m in methods} for name, rr in filters.items()}
    sources = {source: {m: point_summary([r for r in records if r['annotation_coordinate_source'] == source], m) for m in methods}
        for source in sorted({r['annotation_coordinate_source'] for r in records})}
    comparisons = {f'{new}_minus_{base}': compare(methods[new], methods[base], bootstrap,
        (summaries[new], summaries[base]), paired_corner_groups)
        for new in NEW_ARMS if new in methods for base in EXISTING_ARMS}
    native_cap = {}
    for start in ('BASE', 'N3'):
        for kind in ('WIDE', 'BOUNDARY'):
            native, cap = f'{start}_{kind}_NATIVE', f'{start}_{kind}_CAP1'
            if native not in methods:
                continue
            changed = []
            for a, b in zip(methods[native], methods[cap]):
                aa, bb = np.asarray(a['native_points'], float)[:8], np.asarray(b['native_points'], float)[:8]
                supported = np.isfinite(aa).all(-1) & ~np.all(aa == -1, axis=-1) & np.isfinite(bb).all(-1) & ~np.all(bb == -1, axis=-1)
                moved = np.linalg.norm(aa[supported] - bb[supported], axis=-1)
                changed.append(dict(id=a['id'], changed_corners=int((moved > 1e-7).sum())))
            native_cap[f'{start}_{kind}'] = dict(native=native, strict_original_Base_cap=cap,
                changed_frames=sum(r['changed_corners'] > 0 for r in changed),
                changed_corners=sum(r['changed_corners'] for r in changed),
                definition='paired final native/cap1 coordinates; original Base limit is enforced by producer; Native results never merged with capped results')
    return dict(schema='pallet_visible_boundary_statistics_20261009_v1', status='DONE',
                stage='EXISTING_ONLY_SMOKE' if allow_existing_only else 'FINAL_12_PATHS',
                methods=list(methods), frames=319, sessions=13, N3_seed=1,
                summaries=summaries, point_groups=groups, reference_sources=sources,
                reference_source_counts=dict(Counter(r['reference_source'] for r in records)),
                annotation_coordinate_source_counts=dict(Counter(r['annotation_coordinate_source'] for r in records)),
                comparisons=comparisons, bootstrap=bootstrap.meta(), native_vs_cap=native_cap,
                validation=dict(**checks, reference_corners=len(records), visibility_category_counts=counts,
                    UNKNOWN=counts.get('UNKNOWN', 0), UNANNOTATED=counts.get('UNANNOTATED', 0),
                    canonical_0_3_definition='original GT corner IDs 0 and 3 after each saved whole-object evaluation branch; no subgroup-specific branch'),
                contracts=dict(visibility_used='post-hoc only; never gates inference',
                    corner_mean='pooled original observed canonical corners; full missing-penalty statistics separately named',
                    missing='unchanged existing diagonal penalty for full-reference PCK/damage; omitted only from observed-error statistics',
                    reference_geometry='existing frozen coordinates/6D reference; not newly independently measured physical GT',
                    coordinate_source='exact recorded annotation_coordinate_source; UNRECORDED is not assigned manual/projection provenance',
                    pose='each saved final-coordinate actual pose result; failures separate; no point-visibility-specific pose population invented',
                    cap='Native and strict 1%-of-original-Base-diagonal CAP1 arms reported separately',
                    selection='no GT-optimal branch mixing or best-per-frame/point method selection',
                    execution='statistics only; model, correction, and F calls=0'))


def load_methods(path):
    methods = {m: [] for m in ARMS}
    opener = gzip.open if Path(path).suffix == '.gz' else open
    with opener(path, 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            assert row['method'] in methods, row['method']
            methods[row['method']].append(row)
    methods = {m: rows for m, rows in methods.items() if rows}
    base_ids = [r['id'] for r in methods['BASE']]
    for method, rows in methods.items():
        by_id = {r['id']: r for r in rows}
        assert len(by_id) == len(rows) and set(by_id) == set(base_ids), method
        methods[method] = [by_id[i] for i in base_ids]
    return methods


def load_visibility(path):
    packet = json.loads(Path(path).read_text())
    labels = {}
    for row in packet['rows']:
        if row['population'] != 'DEV319':
            continue
        key = row['frame_id'], row['corner_id']
        assert key not in labels
        labels[key] = row
    return labels


def binding(path, source_root):
    path = Path(path)
    return dict(path=str(path.relative_to(source_root)) if path.is_relative_to(source_root) else path.name,
                bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.pending')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def check_cached_baselines(result, prior):
    """Check frozen cached arms against their original published distributions."""
    fields = ('n', 'mean', 'sample_variance', 'sample_std', 'median', 'P90')
    verified = 0
    for arm in EXISTING_ARMS:
        for metric in METRICS:
            for key in fields:
                a = result['summaries'][arm]['metrics'][metric][key]
                b = prior['summaries'][arm]['metrics'][metric][key]
                passed = (a is None and b is None) or (a is not None and b is not None and
                    bool(np.isclose(a, b, rtol=1e-12, atol=1e-12)))
                assert passed, ('Cached baseline changed', arm, metric, key, a, b)
                verified += 1
    assert verified == 96
    return dict(status='PASS', description='frozen four cached arms independently reaggregated against their published prior SUMMARY',
                arms=list(EXISTING_ARMS), metrics=list(METRICS), fields=list(fields),
                verified_numeric_fields=verified, tolerance=dict(rtol=1e-12, atol=1e-12))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--visibility', type=Path)
    parser.add_argument('--targets', type=Path)
    parser.add_argument('--prior-summary', type=Path)
    parser.add_argument('--smoke-existing', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    visibility = args.visibility or args.source_root / VISIBILITY_RELATIVE
    targets = args.targets or args.source_root / TARGETS_RELATIVE
    prior_summary = args.prior_summary or args.source_root / PRIOR_SUMMARY_RELATIVE
    outputs = [args.output / name for name in ('SUMMARY.json', 'PAIRED_COMPARISON.json', 'CHECKS_STATISTICS.json')]
    assert args.overwrite or not any(p.exists() for p in outputs), 'Preserve completed statistics; explicit --overwrite needed'
    result = aggregate(load_methods(args.predictions), load_visibility(visibility),
        json.loads(targets.read_text()), allow_existing_only=args.smoke_existing)
    baseline_check = check_cached_baselines(result, json.loads(prior_summary.read_text()))
    result['source_bindings'] = {name: binding(path, args.source_root) for name, path in
        [('predictions', args.predictions), ('visibility', visibility), ('targets', targets),
         ('prior_summary', prior_summary)]}
    result['statistics_code_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(outputs[0], result)
    write_json(outputs[1], dict(schema=result['schema'], stage=result['stage'],
        bootstrap=result['bootstrap'], comparisons=result['comparisons'],
        source_bindings=result['source_bindings'], statistics_code_sha256=result['statistics_code_sha256']))
    write_json(outputs[2], dict(status='PASS', stage=result['stage'],
        description='statistics checks on completed saved rows; not an additional evaluation or F call',
        frozen_cached_baseline_reaggregation=baseline_check,
        current_saved_row_canonical_alignment=result['validation'],
        paired_common_corner_counts={name: {metric: dict(
            common_eligible_observations=v['common_eligible_observations'],
            common_eligible_frames=v['common_eligible_frames'],
            excluded_reference_corners=v.get('excluded_reference_corners'),
            eligible_sessions=v['eligible_sessions'], empty_resamples=v['empty_resamples'],
            draw_sha256=v['draw_sha256']) for metric, v in comparison['statistics'].items()
            if v['unit'] == 'px'} for name, comparison in result['comparisons'].items()},
        source_bindings=result['source_bindings'], statistics_code_sha256=result['statistics_code_sha256']))
    print(json.dumps(dict(status='DONE', stage=result['stage'], methods=len(result['methods']),
        frames=result['frames'], comparisons=len(result['comparisons']), output=str(args.output),
        draw_sha256=result['bootstrap']['draw_sha256'])))


if __name__ == '__main__':
    main()
