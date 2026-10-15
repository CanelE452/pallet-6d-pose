"""Frozen lower-boundary alternatives; CAL before reused TEST, postseal 2D only."""
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path
import time

import cv2
import numpy as np

from . import common as C
from . import evaluation_interval_v4_2b as E
from . import statewise_evaluation_v4 as V
from . import statewise_uncertainty_calibration_v4 as UC
from . import visible_bi_correction_v2 as B

BASE = V.FIX / 'boundary_recovery_v1'
OUT = BASE / 'nonsam_boundary_v1'
METHODS = ('RGB_GRADIENT', 'TINYUNET', 'MOBILESAM')
TARGET = 'eval_pallet09:1778653661653195264'


def put(path, value):
    if path.exists():
        raise FileExistsError('Preserve completed comparison: ' + str(path))
    C.write(path, value)


def candidates(row):
    """Same frozen pose/state eligibility before any method or mask is accessed."""
    rt = row['runtime']; g = rt['initial_geometry']
    P = np.asarray(g['initial_projection_xy'], float)
    nonself = [k for k in range(8) if not g['initial_self_hidden'][k]]
    hull = [nonself[int(i)] for i in cv2.convexHull(P[nonself].astype(np.float32), returnPoints=False).ravel()]
    result = []; audit = []
    for k in range(8):
        rec = rt['state_observation']['records'][k]
        entry = dict(native_corner=k, state=rt['states'][k], eligible=False)
        audit.append(entry)
        if rt['states'][k] not in ('B', 'U') or g['initial_self_hidden'][k]:
            entry['reason'] = 'non_boundary_or_initial_S'; continue
        if rec['initial_role'] != 'B' or not rec['rgb_evaluated'] or rec['raw_query_out_of_frame'] is not False:
            entry['reason'] = 'no_initial_in_canvas_boundary_role'; continue
        if k not in hull:
            entry['reason'] = 'not_on_initial_hull'; continue
        index = hull.index(k)
        mates = sorted([hull[(index-1) % len(hull)], hull[(index+1) % len(hull)]], key=lambda j: P[j, 0])
        if P[k, 1] <= max(P[mates, 1]):
            entry['reason'] = 'not_lower_hull_apex'; continue
        if any(rt['states'][m] != 'B' or g['initial_self_hidden'][m] or g['initial_roles'][m] != 'B' for m in mates):
            entry['reason'] = 'two_nonS_B_neighbors_required'; continue
        guides = P[mates]-P[k]
        lengths = np.linalg.norm(guides, axis=1)
        if np.any(lengths < 10.):
            entry['reason'] = 'short_model_guide'; continue
        h, w = g['image_hw']
        if any(not (0 <= P[j, 0] < w and 0 <= P[j, 1] < h) for j in [k, *mates]):
            entry['reason'] = 'projected_guide_endpoint_out_of_canvas'; continue
        entry.update(eligible=True, reason='frozen_lower_B_U_with_two_B_neighbors', neighbor_ids=mates,
                     direction_sine=abs(float(np.linalg.det(guides/lengths[:, None]))))
        result.append(dict(native_corner=k, state=rt['states'][k], neighbor_ids=mates,
                           direction_sine=entry['direction_sine']))
    return result, audit


def compact(payload):
    result = deepcopy(payload)
    if 'split_profile' in result:
        keys = ('split_contour_index', 'split_arc_px', 'split_contour_xy', 'robust_Huber_cost',
                'both_original_line_guards_pass', 'finite_intersection', 'intersection_xy',
                'intersection_sine', 'normal_matrix_condition_number', 'finite_observed_support_pass',
                'observed_support_audit')
        result['split_profile'] = [{k: p.get(k) for k in keys} for p in result['split_profile']]
    return C.finite(result)


def run_frame(row):
    from .rgb_lower_boundary_v1 import fit_lower_boundary_rgb
    from .mask_lower_boundary_candidates_v1 import fit_lower_boundary_mask
    cv2.setNumThreads(1)
    eligible, eligibility = candidates(row)
    results = []
    bindings = dict(image_sha256=row['image_sha256'])
    if eligible:
        image_path = C.ROOT/row['image']
        if C.sha(image_path) != row['image_sha256']:
            raise ValueError('Actual image changed')
        image = cv2.imread(str(image_path))
        stem = row['id'].replace(':', '__')
        mask_path = BASE/'image_evidence_v2'/(stem+'.npz')
        meta = C.read(mask_path.with_suffix('.json'))
        if C.sha(mask_path) != meta['mask_file_sha256'] or meta['image_sha256'] != row['image_sha256']:
            raise ValueError('Frozen mask/image binding changed')
        with np.load(mask_path) as archive:
            masks = {'TINYUNET': archive['state_pallet_mask'], 'MOBILESAM': archive['pallet_mask']}
        bindings['mask_file_sha256'] = C.sha(mask_path)
        geometry = deepcopy(row['runtime']['initial_geometry'])
        cfg = deepcopy(row['runtime']['visible_correction']['configuration'])
        for candidate in eligible:
            item = dict(candidate, methods={})
            for method in METHODS:
                started = time.perf_counter()
                if method == 'RGB_GRADIENT':
                    value = fit_lower_boundary_rgb(image, geometry, candidate['native_corner'], candidate['neighbor_ids'], configuration=cfg)
                else:
                    value = fit_lower_boundary_mask(image, masks[method], geometry, candidate['native_corner'],
                        candidate['neighbor_ids'], configuration=cfg, mask_provider=method,
                        allow_endpoint_association_warning=True)
                value = compact(value)
                value['fit_only_elapsed_ms'] = 1000*(time.perf_counter()-started)
                fit = value.get('fit_payload') or {}
                selected = fit.get('selected_representative')
                profile = fit.get('split_profile') or []
                value['coverage'] = dict(
                    has_contour=bool(len(fit.get('contour') or [])),
                    RGB_supported_samples=int(np.sum(fit.get('RGB_eligible') or [])),
                    searched_splits=len(profile),
                    two_line_guard_pass=sum(bool(p.get('both_original_line_guards_pass')) for p in profile),
                    finite_supported_pairs=sum(bool(p.get('both_original_line_guards_pass') and p.get('finite_intersection')
                        and p.get('finite_observed_support_pass')) for p in profile),
                    selected=selected is not None)
                item['methods'][method] = value
            results.append(item)
    return dict(id=row['id'], image=row['image'], partition=row['partition'], results=results,
                eligibility=eligibility, bindings=bindings, GT_inputs=False, states_changed=False,
                original_BI_corrected_coordinates_read=False, raw_corner_coordinate_inputs=False, new_pose_fits=0)


def source_bindings():
    from . import rgb_lower_boundary_v1 as RGB
    from . import mask_lower_boundary_candidates_v1 as MASK
    from . import mask_lower_boundary_v1 as STRICT_MASK
    from . import boundary_refine as F
    from . import visible_bi_boundary_recovery_v4_3b as SUPPORT
    files = [__file__, RGB.__file__, MASK.__file__, STRICT_MASK.__file__, F.__file__, B.__file__, SUPPORT.__file__,
             BASE/'kp3_bottom_piecewise_v1/diagnose_piecewise_lower_contour_v1.py']
    return {str(Path(p).relative_to(C.ROOT)): C.sha(p) for p in files}


def predict(partition, workers=3):
    rows, _, _ = V._verify_prediction_seal(E.OUTPUT)
    rows = [r for r in rows if r['cohort'] == 'PRIMARY_NONSEVERE' and r['partition'] == partition]
    assert len(rows) == {'CAL': 46, 'TEST': 49}[partition]
    if partition == 'CAL':
        if OUT.exists():
            raise FileExistsError('Preserve completed comparison')
        OUT.mkdir(parents=True)
        protected = C.read(E.OUTPUT/'PROTOCOL.json')['protected_user_sha256']
        protocol = dict(schema='LOWER_BOUNDARY_ALTERNATIVES_V1', methods=list(METHODS), source_sha256=source_bindings(),
            protected_user_sha256=protected, original_prediction_sha256=C.sha(E.OUTPUT/'PREDICTIONS.jsonl.gz'),
            method_config=B.DEFAULT_CONFIG, upstream_states_pose_backbone_unchanged=True,
            RGB_mask_free_scope='boundary fitting only; frozen upstream state decisions retained',
            candidate_scope='lower hull B/U with initial B role, nonS in canvas and two nonS B neighbors',
            all_methods_same_geometry_only_endpoints=True, existing_SAM_corrected_endpoints_used=False,
            raw_junction_or_painted_point_used=False,
            mask_endpoint_association_warning='12px normal search + 7px context = 19px retained as warning; fit-only candidates continue',
            allow_endpoint_association_warning=True, fit_candidates_are_not_accepted_native_points=True,
            RGB_search_radius_px=12., RGB_new_network_weights_bytes=0,
            mask_inputs='sealed actual MobileSAM and TinyUNet outputs; no mask forward or retraining',
            line_fit_guard='existing RGB guards, minimum branch arc36px, Huber1.25px, finite support11px',
            added_lower_shift_px=0., uncertainty_refitted=False, native_keypoint_adopted=False, states_changed=False,
            CAL_batch_before_TEST_batch=True, known_example_RGB_debugged_before_batch=True,
            TEST_reused_exploratory=True, severe_count=0,
            scoring='postseal DIRECT/native-compatible 2D intersection diagnostic; coverage and matched subsets',
            new_full_cohort_pose_fits=0, target_GT_known_from_previous_posthoc_but_not_method_input=True)
        V._verify_sources(protected)
        put(OUT/'PROTOCOL.json', protocol)
    else:
        C.read(OUT/'CAL_SCORE.json')
        protocol = C.read(OUT/'PROTOCOL.json')
    V._verify_sources(protocol['source_sha256'])
    V._verify_sources(protocol['protected_user_sha256'])
    if C.sha(E.OUTPUT/'PREDICTIONS.jsonl.gz') != protocol['original_prediction_sha256']:
        raise ValueError('Upstream predictions changed')
    predictions = []; started = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, value in enumerate(pool.map(run_frame, rows, chunksize=1)):
            predictions.append(value)
            print(f'{partition} {i+1}/{len(rows)} lower candidates={len(value["results"])}', flush=True)
    C.save_rows(OUT/(partition+'_PREDICTIONS.jsonl.gz'), predictions)
    V._verify_sources(protocol['source_sha256'])
    V._verify_sources(protocol['protected_user_sha256'])
    put(OUT/(partition+'_SEAL.json'), dict(predictions_sha256=C.sha(OUT/(partition+'_PREDICTIONS.jsonl.gz')),
        protocol_sha256=C.sha(OUT/'PROTOCOL.json'), ids=[r['id'] for r in rows], frames=len(rows),
        candidates=sum(len(r['results']) for r in predictions), before_new_partition_annotation_score=True,
        elapsed_seconds=time.monotonic()-started, new_pose_fits=0, mask_forwards=0, GT_inputs=False))


def distribution(values):
    a = np.asarray(values, float)
    return dict(n=len(a), mean_px=float(a.mean()) if len(a) else None,
                median_px=float(np.median(a)) if len(a) else None,
                P90_px=float(np.quantile(a, .9)) if len(a) else None,
                max_px=float(a.max()) if len(a) else None)


def score(partition):
    seal = C.read(OUT/(partition+'_SEAL.json')); protocol = C.read(OUT/'PROTOCOL.json')
    assert seal['before_new_partition_annotation_score']
    assert C.sha(OUT/(partition+'_PREDICTIONS.jsonl.gz')) == seal['predictions_sha256']
    assert C.sha(OUT/'PROTOCOL.json') == seal['protocol_sha256']
    V._verify_sources(protocol['source_sha256'])
    label_path = UC.BASE/('OOF_pose_v1/ISOLATED_CAL_SCORING.json' if partition == 'CAL' else 'test_reused_v1/ISOLATED_TEST_SCORING.json')
    labels = C.read(label_path)['rows']
    originals = {r['id']: r for r in C.rows(E.OUTPUT/'PREDICTIONS.jsonl.gz')}
    predictions = list(C.rows(OUT/(partition+'_PREDICTIONS.jsonl.gz')))
    assert [r['id'] for r in predictions] == seal['ids']
    points = []; coverage = {m: dict(eligible=0, contour_found=0, RGB_supported=0,
                                     two_line_qualified=0, selected=0, endpoint_association_warnings=0,
                                     selected_with_endpoint_warning=0, failure_reasons=Counter()) for m in METHODS}
    for row in predictions:
        label = labels[row['id']]
        alignment = UC._reference_alignment(originals[row['id']], label)
        for candidate in row['results']:
            k = candidate['native_corner']; rid = int(alignment['mapping'][k])
            valid = bool(label['target']['valid'][rid])
            human = label['metadata']['human_states_native'][k]
            target = np.asarray(label['target']['gt'][rid], float) if valid else None
            entry = dict(id=row['id'], native_corner=k, state=candidate['state'], human_state=human,
                         valid_annotation=valid, native_compatible=alignment['compatible'],
                         annotation_xy=target.tolist() if valid else None, error_px={}, accepted={}, intersection_xy={},
                         model_projection_xy=originals[row['id']]['runtime']['initial_geometry']['initial_projection_xy'][k])
            if valid:
                entry['model_projection_error_px'] = float(np.linalg.norm(np.asarray(entry['model_projection_xy'])-target))
            for method, value in candidate['methods'].items():
                cov = value['coverage']; c = coverage[method]
                c['eligible'] += 1; c['contour_found'] += int(cov['has_contour'])
                c['RGB_supported'] += int(cov['RGB_supported_samples'] >= B.DEFAULT_CONFIG['minimum_inlier_count'])
                c['two_line_qualified'] += int(cov['two_line_guard_pass'] > 0)
                c['selected'] += int(cov['selected'])
                warning = value['audit'].get('endpoint_guard_pass') is False
                c['endpoint_association_warnings'] += int(warning)
                c['selected_with_endpoint_warning'] += int(warning and cov['selected'])
                if not cov['selected']:
                    c['failure_reasons'][value['status']] += 1
                selected = (value.get('fit_payload') or {}).get('selected_representative')
                entry['accepted'][method] = selected is not None
                entry['intersection_xy'][method] = selected['intersection_xy'] if selected else None
                entry['error_px'][method] = float(np.linalg.norm(np.asarray(selected['intersection_xy'])-target)) if selected and valid else None
            points.append(entry)
    strata = dict(DIRECT_COMPATIBLE=[p for p in points if p['valid_annotation'] and p['native_compatible'] and p['human_state'] == 'DIRECT_VISIBLE'],
                  U_DIRECT_COMPATIBLE=[p for p in points if p['valid_annotation'] and p['native_compatible'] and p['human_state'] == 'DIRECT_VISIBLE' and p['state'] == 'U'],
                  DIRECT_INCOMPATIBLE_DIAGNOSTIC=[p for p in points if p['valid_annotation'] and not p['native_compatible'] and p['human_state'] == 'DIRECT_VISIBLE'],
                  OTHER_VALID_ANNOTATIONS=[p for p in points if p['valid_annotation'] and p['human_state'] != 'DIRECT_VISIBLE'])
    summaries = {}
    for name, pp in strata.items():
        methods = {}
        for method in METHODS:
            accepted = [p for p in pp if p['accepted'][method]]
            methods[method] = dict(eligible=len(pp), accepted=len(accepted),
                accepted_error=distribution([p['error_px'][method] for p in accepted]),
                initial_model_on_same_accepted=distribution([p['model_projection_error_px'] for p in accepted]),
                better_than_initial_model=sum(p['error_px'][method] < p['model_projection_error_px'] for p in accepted))
        matched = [p for p in pp if all(p['accepted'][m] for m in METHODS)]
        pairs = {}
        for alternative in METHODS[:2]:
            common = [p for p in pp if p['accepted'][alternative] and p['accepted']['MOBILESAM']]
            pairs[alternative+'_vs_MOBILESAM'] = dict(n=len(common),
                alternative_error=distribution([p['error_px'][alternative] for p in common]),
                SAM_error=distribution([p['error_px']['MOBILESAM'] for p in common]),
                alternative_better=sum(p['error_px'][alternative] < p['error_px']['MOBILESAM'] for p in common))
        summaries[name] = dict(methods=methods, all_three_matched_n=len(matched),
            all_three_matched_error={m: distribution([p['error_px'][m] for p in matched]) for m in METHODS}, pairs=pairs)
    put(OUT/(partition+'_SCORE.json'), dict(partition=partition, frames=len(predictions), coverage=C.finite(coverage),
        strata=summaries, points=points, annotation_sha256=C.sha(label_path), prediction_sha256=seal['predictions_sha256'],
        new_pose_fits=0, adopted_native_keypoints=0, GT_inputs_to_predictions=False,
        conditional_mean_error_is_not_full_coverage_superiority=True, TEST_reused_exploratory=partition == 'TEST'))
    print(C.finite(dict(partition=partition, coverage=coverage, DIRECT_COMPATIBLE=summaries['DIRECT_COMPATIBLE'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('predict', 'score'))
    parser.add_argument('partition', choices=('CAL', 'TEST'))
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    if args.stage == 'predict':
        predict(args.partition, args.workers)
    else:
        score(args.partition)


if __name__ == '__main__':
    main()
