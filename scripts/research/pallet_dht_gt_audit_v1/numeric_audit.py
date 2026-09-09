"""Independent replay of saved DEV319 point errors; no model or image inference.

Only this audit's output directory is written. Official identities, GT coordinates,
visibility masks and box-match rules remain unchanged. Subsets and GT-assisted
assignment are descriptive diagnostics, never alternate official scores.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[3]
ARMS = ('baseline', 'point_only', 'line_fusion')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


class Inputs:
    def __init__(self):
        self.hashes = {}

    def bind(self, path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        if expected is not None:
            assert digest == expected, f'Frozen input changed: {path}'
        self.hashes[str(path)] = digest
        return path

    def read(self, path, expected=None):
        return json.loads(self.bind(path, expected).read_text())

    def verify(self):
        for p, s in self.hashes.items():
            assert sha(p) == s, f'Input changed during audit: {p}'


def stats(values):
    a = np.asarray(values, dtype=np.float64)
    assert a.ndim == 1 and np.isfinite(a).all()
    return dict(n=len(a), mean_px=float(a.mean()) if len(a) else None,
                median_px=float(np.median(a)) if len(a) else None,
                p90_px=float(np.quantile(a, .9, method='linear')) if len(a) else None,
                max_px=float(a.max()) if len(a) else None,
                pck10_observed=float(np.mean(a <= 10)) if len(a) else None,
                above20_count=int(np.sum(a > 20)), above50_count=int(np.sum(a > 50)),
                above100_count=int(np.sum(a > 100)))


def summarize(rows):
    n_gt = sum(r['gt_supervised'] for r in rows)
    obs = [r for r in rows if r['observed']]
    out = dict(n_frames=len({r['id'] for r in rows}), n_point_slots=len(rows),
               n_gt_points=n_gt, n_observed_points=len(obs),
               n_frames_with_observations=len({r['id'] for r in obs}),
               n_missing_or_unmatched_points=n_gt-len(obs),
               point_coverage=len(obs)/n_gt if n_gt else None, arms={})
    for arm in ARMS:
        ss = stats([r[f'error_{arm}_px'] for r in obs])
        ss['pck10_all_supervised'] = sum(r[f'error_{arm}_px'] <= 10 for r in obs)/n_gt if n_gt else None
        ss['mean_delta_vs_baseline_px'] = float(np.mean([r[f'error_{arm}_px']-r['error_baseline_px'] for r in obs])) if obs else None
        ss['p90_delta_vs_baseline_px'] = ss['p90_px']-out['arms']['baseline']['p90_px'] if arm != 'baseline' and obs else 0. if obs else None
        out['arms'][arm] = ss
    return out


def groups(rows, key):
    values = sorted({str(r[key]) for r in rows})
    return {v: summarize([r for r in rows if str(r[key]) == v]) for v in values}


def box_iou(first, second):
    # Independent copy of paper_real_eval._box_iou, with its arithmetic order.
    left, top = max(float(first[0]), float(second[0])), max(float(first[1]), float(second[1]))
    right, bottom = min(float(first[2]), float(second[2])), min(float(first[3]), float(second[3]))
    intersection = max(0., right-left)*max(0., bottom-top)
    area1 = max(0., float(first[2]-first[0]))*max(0., float(first[3]-first[1]))
    area2 = max(0., float(second[2]-second[0]))*max(0., float(second[3]-second[1]))
    union = area1+area2-intersection
    return intersection/union if union > 0. else 0.


def points(payload):
    p = np.asarray([[np.nan, np.nan] if v is None else v for v in payload['points']], dtype=float)
    assert p.shape == (9, 2)
    valid = np.asarray(payload['point_valid'], bool) & np.isfinite(p).all(1)
    return p, valid


def row_excerpt(r):
    keys = ('id', 'session_id', 'image', 'gt_path', 'point_id', 'point_kind', 'visibility',
            'annotation_source', 'reason', 'gt_x', 'gt_y', 'manual_x', 'manual_y',
            'manual_vs_scored_distance_px', 'baseline_x', 'baseline_y', 'error_baseline_px',
            'error_point_only_px', 'error_line_fusion_px', 'prior_review')
    return {k: r[k] for k in keys}


def run(run_dir):
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    source = ROOT/'data/pallet/results/pallet_dht_decoder_probe_v1'
    review_root = ROOT/'data/pallet/results/accuracy_root_cause_v1'
    inputs = Inputs()
    inputs.bind(__file__)
    for p in (run_dir/'PURPOSE.md', run_dir/'PROTOCOL.json'):
        if p.exists():
            inputs.bind(p)
    evaluation = inputs.read(source/'EVALUATION_COMPLETION.json')
    assert evaluation['complete'] and evaluation['PASS']
    protocol = inputs.read(source/'TRAIN_PROTOCOL.json', evaluation['input_sha256'][str(source/'TRAIN_PROTOCOL.json')])
    original = inputs.read(source/'RESULTS.json', evaluation['output_sha256']['RESULTS.json'])
    saved_metrics = inputs.read(source/'FRAME_METRICS.json', evaluation['output_sha256']['FRAME_METRICS.json'])
    predictions = inputs.read(source/'PREDICTIONS.json', evaluation['output_sha256']['PREDICTIONS.json'])
    assert original['complete'] and original['PASS'] and saved_metrics['complete'] and predictions['complete'] and predictions['PASS']
    manifest = inputs.read(protocol['data']['real_manifest'], protocol['data']['real_manifest_sha256'])
    inputs.bind(protocol['backbone']['checkpoint'], protocol['backbone']['sha256'])
    inputs.bind(ROOT/'scripts/research/pallet_dht_decoder_probe_v1/evaluate.py')
    inputs.bind(ROOT/'challenge/evaluation_v2/paper_real_eval.py')
    reviews = inputs.read(review_root/'REVIEW_LOCALSTORAGE.json')
    review_summary = inputs.read(review_root/'GT_REVIEW_PHASE23.json')
    phase1 = inputs.read(review_root/'GT_REVIEW_RESULTS.json')
    quarantine = inputs.read(ROOT/'challenge/real_gt_v2/INVALID_GT_QUARANTINE.json')
    pred = {r['id']: r for r in predictions['records'] if r['population'] == 'real_dev'}
    frames = {r['id']: r for r in saved_metrics['records'] if r['population'] == 'real_dev'}
    items = {r['frame_id']: r for r in manifest['items']}
    assert manifest['role'] == 'DEV' and manifest['expected_count'] == 319
    assert len(items) == len(pred) == len(frames) == 319 and set(items) == set(pred) == set(frames)
    labels = {'ok': 'ok', 'some': 'partly', 'bad': 'off', 'cannot_tell': 'cannot_tell'}
    review_frames = {k.replace('__', ':', 1): v for k, v in reviews['gtreview_v1'].items()}
    assert dict(Counter(v['extrap'] for v in reviews['gtreview_v1'].values())) == review_summary['extrap']
    hypothesis_wrong = {r['fid'].replace('__', ':', 1) for r in phase1 if r['pick'] == 'both_wrong'}
    rows, gt_lineage, assignment_rows = [], [], []
    coordinate_maxdiff = 0.
    arithmetic_maxdiff = 0.
    for fid, item in items.items():
        rec, saved = pred[fid], frames[fid]
        assert Path(rec['image']).resolve() == (ROOT/item['image_path']).resolve()
        gt_path = (ROOT/item['gt_v2_path']).resolve()
        gt = inputs.read(gt_path, evaluation['input_sha256'][str(gt_path)])
        assert len(gt['objects']) == 1
        obj = gt['objects'][0]
        ann = obj['keypoint_annotations']
        xy = np.asarray([a['xy'] for a in ann], float)
        vis = np.asarray([a['visibility'] for a in ann], int)
        assert xy.shape == (9, 2) and np.isfinite(xy).all()
        assert np.array_equal(xy, saved['gt_points']) and np.array_equal(vis > 0, saved['gt_supervised'])
        raw_manual = obj.get('manual_kps')
        manual = np.asarray([[np.nan, np.nan] if p is None else p for p in raw_manual], float) if raw_manual is not None else None
        assert manual is None or manual.shape == (9, 2)
        extra = obj.get('extrapolated_mask')
        assert extra is None or len(extra) == 9
        migration = gt.get('real_gt_v2_migration') or {}
        legacy_path = (ROOT/migration['source_label']).resolve() if migration.get('source_label') else None
        legacy_actual_sha = None
        if legacy_path is not None and legacy_path.exists():
            inputs.bind(legacy_path)
            legacy_actual_sha = inputs.hashes[str(legacy_path)]
        gt_lineage.append(dict(id=fid, gt_path=str(gt_path), gt_sha256=inputs.hashes[str(gt_path)],
            schema_version=gt.get('schema_version'), keypoint_frame=obj.get('keypoint_frame'),
            manual_kps_present=manual is not None, extrapolated_mask_present=extra is not None,
            migration_source_path=str(legacy_path) if legacy_path else None,
            migration_source_declared_sha256=migration.get('source_sha256'),
            migration_source_current_sha256=legacy_actual_sha,
            migration_source_current_matches_declared=(legacy_actual_sha == migration.get('source_sha256')) if legacy_actual_sha else None,
            migration_status=migration.get('status'), object_migration_status=obj.get('migration_status'),
            annotation_sources=dict(Counter(a.get('source', 'unknown') for a in ann))))
        target_box = np.r_[xy[:8].min(0), xy[:8].max(0)]
        candidate_box = rec['baseline']['box_xyxy']
        iou = box_iou(np.asarray(candidate_box), target_box) if candidate_box is not None else None
        matched = iou is not None and iou >= .5
        assert matched == saved['baseline_match_iou50']
        arrays = {a: points(rec['baseline'] if a == 'baseline' else rec['arms'][a]) for a in ARMS}
        base, valid = arrays['baseline']
        observed = (vis > 0) & valid & matched
        for arm, (pp, vv) in arrays.items():
            assert np.array_equal(vv, valid)
            assert np.array_equal(pp[8], base[8], equal_nan=True)
            coordinate_maxdiff = max(coordinate_maxdiff, float(np.max(np.abs(pp-np.asarray(saved['arms'][arm]['points'])))))
            err = np.linalg.norm(pp-xy, axis=1)
            for j in range(9):
                old = saved['arms'][arm]['errors_px'][j]
                assert (old is not None) == bool(observed[j])
                if old is not None:
                    arithmetic_maxdiff = max(arithmetic_maxdiff, abs(float(err[j])-old))
        prior_raw = review_frames.get(fid, {}).get('extrap')
        prior = labels[prior_raw] if prior_raw is not None else 'not_reviewed'
        local = []
        for j in range(9):
            mxy = manual[j] if manual is not None and np.isfinite(manual[j]).all() else None
            row = dict(id=fid, session_id=item['session_id'], image=str(Path(rec['image']).resolve()),
                gt_path=str(gt_path), gt_sha256=inputs.hashes[str(gt_path)],
                point_id=j, point_kind='corner' if j < 8 else 'centroid',
                visibility=int(vis[j]), reason=ann[j].get('reason', 'unknown'),
                annotation_source=ann[j].get('source', 'unknown'), in_frame=bool(ann[j]['in_frame']),
                gt_x=float(xy[j, 0]), gt_y=float(xy[j, 1]), gt_supervised=bool(vis[j] > 0),
                predicted_valid=bool(valid[j]), matched=matched, match_iou=iou, observed=bool(observed[j]),
                width=rec['width'], height=rec['height'], image_resolution=f"{rec['width']}x{rec['height']}",
                manual_kps_present=manual is not None, manual_x=float(mxy[0]) if mxy is not None else None,
                manual_y=float(mxy[1]) if mxy is not None else None,
                manual_vs_scored_distance_px=float(np.linalg.norm(mxy-xy[j])) if mxy is not None else None,
                explicit_extrapolated_mask=bool(extra[j]) if extra is not None else 'absent',
                prior_review=prior, prior_review_raw=prior_raw, prior_hypothesis_both_wrong=fid in hypothesis_wrong,
                migration_source_path=str(legacy_path) if legacy_path else None,
                migration_source_declared_sha256=migration.get('source_sha256'))
            for arm, (pp, _) in arrays.items():
                row[f'{arm}_x'], row[f'{arm}_y'] = float(pp[j, 0]), float(pp[j, 1])
                row[f'error_{arm}_px'] = float(np.linalg.norm(pp-xy, axis=1)[j]) if observed[j] else None
            rows.append(row)
            local.append(row)
        # GT-assisted free one-to-one assignment: diagnostic, NOT a valid symmetry metric.
        for arm, (pp, vv) in arrays.items():
            target_ids = np.flatnonzero(observed[:8])
            predicted_ids = np.flatnonzero(vv[:8])
            mapping, vals = [], []
            if len(target_ids):
                distances = np.linalg.norm(xy[target_ids, None]-pp[None, predicted_ids], axis=2)
                target_pos, pred_pos = linear_sum_assignment(distances)
                assert len(target_pos) == len(target_ids)
                for a, b in zip(target_pos, pred_pos):
                    mapping.append([int(target_ids[a]), int(predicted_ids[b])])
                    vals.append(float(distances[a, b]))
            with_center = vals + ([local[8][f'error_{arm}_px']] if observed[8] else [])
            assignment_rows.append(dict(id=fid, arm=arm, target_to_predicted_ids=mapping,
                n_nonidentity=sum(a != b for a, b in mapping), errors_corner=vals,
                errors_with_fixed_centroid=with_center))
    assert coordinate_maxdiff == 0. and arithmetic_maxdiff == 0.
    all_stats = summarize(rows)
    assert (len(rows), all_stats['n_gt_points'], all_stats['n_observed_points'], all_stats['n_frames_with_observations']) == (2871, 2818, 2738, 309)
    reconstruction = {}
    for arm in ARMS:
        ref = next(r for r in original['summaries'] if r['population'] == 'real_dev' and r['arm'] == arm)
        ss = all_stats['arms'][arm]
        for key in ('mean_px', 'median_px', 'p90_px', 'pck10_observed'):
            assert abs(ss[key]-ref[key]) < 1e-12, (arm, key, ss[key], ref[key])
        for key in ('n_gt_points', 'n_observed_points', 'n_frames_with_observations', 'n_missing_or_unmatched_points'):
            assert all_stats[key] == ref[key]
        reconstruction[arm] = dict(metric_source_max_abs_tolerance=1e-12, metric_source_parity_PASS=True,
                                   median_px=ss['median_px'], p90_px=ss['p90_px'], mean_px=ss['mean_px'])
    observed_rows = [r for r in rows if r['observed']]
    sorted_rows = sorted(observed_rows, key=lambda r: (r['error_baseline_px'], r['id'], r['point_id']))
    rank = (len(sorted_rows)-1)*.9
    lo, hi = int(np.floor(rank)), int(np.ceil(rank))
    boundary = dict(zero_based_fractional_index=rank, lower_rank_one_based=lo+1,
        upper_rank_one_based=hi+1, upper_weight=rank-lo,
        lower=row_excerpt(sorted_rows[lo]), upper=row_excerpt(sorted_rows[hi]),
        p90_reconstructed=(1-(rank-lo))*sorted_rows[lo]['error_baseline_px']+(rank-lo)*sorted_rows[hi]['error_baseline_px'],
        nearest_12_order_statistics=[dict(rank_one_based=i+1, **row_excerpt(sorted_rows[i])) for i in range(lo-5, hi+6)])
    p90 = all_stats['arms']['baseline']['p90_px']
    for r in rows:
        r['baseline_above_p90'] = r['observed'] and r['error_baseline_px'] > p90
    tail = [r for r in rows if r['baseline_above_p90']]
    tail_counts = {}
    for key in ('session_id', 'visibility', 'annotation_source', 'point_kind', 'prior_review'):
        counter = Counter(str(r[key]) for r in tail)
        tail_counts[key] = {k: dict(n_points=n, fraction_of_tail=n/len(tail)) for k, n in sorted(counter.items())}
    manual_d = [r['manual_vs_scored_distance_px'] for r in rows if r['manual_vs_scored_distance_px'] is not None]
    manual_changed = [r for r in rows if r['manual_vs_scored_distance_px'] is not None and r['manual_vs_scored_distance_px'] != 0.]
    official = [e for e in quarantine['entries'] if e['classification'] != 'STALE_DUPLICATE_INVALID']
    stale = [e for e in quarantine['entries'] if e['classification'] == 'STALE_DUPLICATE_INVALID']
    official_ids = {e['frame_id'] for e in official}
    stale_ids = {e['frame_id'] for e in stale}
    forbidden_sha = {e['source_sha256'] for e in quarantine['entries']}
    official_overlap = sorted(fid for fid in items if fid.split(':', 1)[1] in official_ids)
    stale_id_overlap = sorted(fid for fid in items if fid.split(':', 1)[1] in stale_ids)
    forbidden_gt_sha_overlap = [r['id'] for r in gt_lineage if r['gt_sha256'] in forbidden_sha]
    forbidden_migration_overlap = [r['id'] for r in gt_lineage if r['migration_source_declared_sha256'] in forbidden_sha]
    assert not (official_overlap or forbidden_gt_sha_overlap or forbidden_migration_overlap)
    prior_intersection = sorted(set(items) & set(review_frames))
    prior_ok_ids = sorted(fid for fid in prior_intersection if review_frames[fid].get('extrap') == 'ok')
    assignment = {}
    for arm in ARMS:
        ar = [r for r in assignment_rows if r['arm'] == arm]
        assignment[arm] = dict(corners=stats([x for r in ar for x in r['errors_corner']]),
            including_fixed_centroid=stats([x for r in ar for x in r['errors_with_fixed_centroid']]),
            n_frames_with_nonidentity_assignment=sum(r['n_nonidentity'] > 0 for r in ar),
            n_nonidentity_corner_assignments=sum(r['n_nonidentity'] for r in ar))
    epsilon = []
    for eps in (2., 5., 10.):
        intervals = {a: [max(0., all_stats['arms'][a]['p90_px']-eps), all_stats['arms'][a]['p90_px']+eps] for a in ARMS}
        delta = all_stats['arms']['line_fusion']['p90_px']-p90
        epsilon.append(dict(euclidean_GT_error_bound_px=eps, p90_guaranteed_intervals_px=intervals,
            line_minus_baseline_p90_conservative_interval_px=[delta-2*eps, delta+2*eps],
            current_baseline_points_guaranteed_over20=sum(r['error_baseline_px'] > 20+eps for r in observed_rows),
            current_baseline_points_guaranteed_over30=sum(r['error_baseline_px'] > 30+eps for r in observed_rows)))
    out = dict(schema='pallet_dht_gt_numeric_audit_v1', complete=True, audit_integrity_PASS=True,
        generated_at=datetime.now(timezone.utc).isoformat(), input_sha256=inputs.hashes,
        baseline_checkpoint=protocol['backbone'], population=dict(id=manifest['population_id'], role=manifest['role'],
            n_frames=319, n_sessions=13, n_detected_frames=sum(r['baseline']['detected'] for r in pred.values()),
            no_new_sealed_FINAL=True),
        metric_contract=dict(coordinates='Original image pixels', target='objects[0].keypoint_annotations[0:9].xy',
            GT_mask='visibility > 0 (visible=2 and occluded/unknown=1); exclude visibility=0',
            predicted_mask='baseline point_valid and finite; preserved by both trained arms',
            association='Highest-confidence preselected baseline candidate; IoU >= .5 to unclipped box of ALL eight GT corners',
            identity='Same semantic point ID; eight corners PLUS centroid8; no symmetry permutation',
            p90='numpy.quantile(errors, .9, method=linear) pooled over observed points',
            missing='80 supervised points without matched observations excluded from error quantile, retained in coverage denominator',
            three_arms_same_mask=True),
        metric_reconstruction=dict(summary=all_stats, by_arm=reconstruction,
            saved_point_coordinate_max_abs_delta_px=coordinate_maxdiff,
            saved_error_max_abs_delta_px=arithmetic_maxdiff, p90_boundary=boundary),
        by_visibility=groups(rows, 'visibility'), by_annotation_source=groups(rows, 'annotation_source'),
        by_reason=groups(rows, 'reason'), by_corner_centroid=groups(rows, 'point_kind'),
        by_session=groups(rows, 'session_id'), by_point_id=groups(rows, 'point_id'),
        by_image_resolution=groups(rows, 'image_resolution'),
        by_explicit_extrapolated_mask=groups(rows, 'explicit_extrapolated_mask'),
        manual_visible=summarize([r for r in rows if r['annotation_source'] == 'manual_click' and r['visibility'] == 2]),
        visible_corners=summarize([r for r in rows if r['point_kind'] == 'corner' and r['visibility'] == 2]),
        visibility1_explicit_source=groups([r for r in rows if r['visibility'] == 1], 'annotation_source'),
        visibility2_explicit_source=groups([r for r in rows if r['visibility'] == 2], 'annotation_source'),
        tail_composition=dict(definition='baseline error strictly greater than full-population baseline P90; same fixed subset for all arms',
            summary=summarize(tail), shares=tail_counts,
            largest_frame_counts=[dict(id=k, n_tail_points=n) for k, n in Counter(r['id'] for r in tail).most_common()],
            worst_20_points=[row_excerpt(r) for r in sorted_rows[-20:][::-1]]),
        manual_vs_scored_xy=dict(n_manual_slots=len(manual_d), n_nonidentical_points=len(manual_changed),
            n_nonidentical_frames=len({r['id'] for r in manual_changed}), distance_summary=stats(manual_d),
            changed_points=[row_excerpt(r) for r in manual_changed],
            limitation='manual_kps includes projected/extrapolated and unknown-source points; field name and integer coordinates do NOT prove direct clicks. Explicit annotation source=manual_click is separate metadata evidence, not visual accuracy certification.'),
        prior_human_review=dict(raw_total_frames=len(review_frames), raw_counts=review_summary['extrap'],
            current_intersection_frames=len(prior_intersection), intersection_ids=prior_intersection,
            current_prior_ok_frames=len(prior_ok_ids), prior_ok_ids=prior_ok_ids,
            by_frame_verdict=groups(rows, 'prior_review'),
            prior_ok=summarize([r for r in rows if r['prior_review'] == 'ok']),
            rest=summarize([r for r in rows if r['prior_review'] != 'ok']),
            pose_hypothesis_both_wrong_original_ids=sorted(hypothesis_wrong),
            pose_hypothesis_both_wrong_current_ids=sorted(hypothesis_wrong & set(items)),
            pose_hypothesis_both_wrong_summary=summarize([r for r in rows if r['prior_hypothesis_both_wrong']]),
            limitation='Historical frame verdict rates extrapolated-GT plausibility, not independently repeated pixel clicks. This is an exact frame-ID join, not a verified byte-identical coordinate history from review day. This small, previously reviewed, selected subset cannot certify the 319 labels. Phase1 both_wrong rejects A/B pose hypotheses; it is not a confirmed 2D-GT error label.'),
        quarantine=dict(registry_entries=len(quarantine['entries']), official_excluded_entries=len(official),
            official_frame_id_overlap=official_overlap, forbidden_GT_bytes_overlap=forbidden_gt_sha_overlap,
            forbidden_migration_source_bytes_overlap=forbidden_migration_overlap,
            stale_duplicate_frame_id_overlap=stale_id_overlap,
            interpretation='Official excluded frame IDs and forbidden annotation bytes absent. A stale duplicate ID alone is not forbidden because a different valid canonical annotation may remain.'),
        pointset_assignment_diagnostic=dict(uses_GT=True, changes_official_metric=False,
            rule='Minimize summed Euclidean corner error using free one-to-one assignment of observed GT corners to available eight predicted corners; fixed centroid untouched.',
            physical_symmetry_metric=False, by_arm=assignment,
            limitation='Oracle relaxation ignores semantic IDs and can pair distinct non-equivalent corners. Reduction can reflect label-ID confusion, geometry error, or coincidental neighboring predictions; it neither identifies GT error nor yields deployable corrected predictions.',
            per_frame=assignment_rows),
        annotation_epsilon_bounds=dict(assumption='Each scored GT location differs from the unknown correct location by at most epsilon in Euclidean pixels; predicted coordinates and existing match/visibility/availability masks stay fixed.',
            derivation='Reverse triangle inequality gives |e_i_new-e_i_old| <= epsilon. Coordinatewise monotonicity and translation equivariance of the linear empirical quantile give max(0,Q90-epsilon) <= Q90_new <= Q90+epsilon.',
            bounds=epsilon, not_measured_repeatability=True,
            limitations='Does not cover semantic ID errors, changed visibility, changed matches after rebuilding GT boxes, or unbounded occluded-point uncertainty. A per-coordinate epsilon assumption would imply sqrt(2)*epsilon Euclidean bound. Small between-arm P90 differences are not robustly ordered by these conservative bounds.'),
        gt_lineage=gt_lineage,
        limitations=['No raw-image visual judgment in this numerical script.',
            'Subsets change the score denominator and are not corrected whole-population accuracy.',
            'All arms share the frozen joint-backbone seed1. point_only names a new point/P4 decoder module, not a different historical YOLO model.',
            'One decoder-training seed and reused DEV319; no new training, neural inference, GT edits, or model/threshold selection.',
            'visibility=2/1 denotes point visibility, not annotation schema v2/v1.',
            'Numeric self-consistency, projected geometry and a prediction disagreement do not establish visual GT correctness.'])
    inputs.verify()
    csv_path = run_dir/'NUMERIC_AUDIT.csv'
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    out['csv'] = dict(path=str(csv_path), sha256=sha(csv_path), rows=len(rows))
    path = run_dir/'NUMERIC_AUDIT.json'
    tmp = path.with_suffix('.pending.json')
    tmp.write_text(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)
    print(json.dumps(dict(path=str(path), sha256=sha(path), audit_integrity_PASS=True,
        reconstruction=reconstruction, manual_differences=len(manual_changed),
        prior_ok_frames=len(prior_ok_ids), prior_ok_p90=out['prior_human_review']['prior_ok']['arms']['baseline']['p90_px'],
        boundary=boundary), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', default=str(ROOT/'data/pallet/results/pallet_dht_gt_audit_v1'))
    run(parser.parse_args().run_dir)
