"""Score the complete four-arm sealed population with unchanged proxy metrics.

This module does not call inference, correction, selection, PnP or an optimizer.
Existing references are opened only after the new complete geometry seal passes.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import time

from . import common as C


def v3_control_parity(args, geometry):
    """Compare the unchanged control with old GT-free geometry before references.

    The published control is an audit input only, never a new pose input.
    Every nonzero numeric difference is retained; tolerance is not bit identity.
    """
    import numpy as np
    reference = C.REPO / '_docs/experiments/pallet_cornerwise_refiner_20261010_v3/GEOMETRY_SEALED.jsonl.gz'
    frozen = C.read(args.protocol)['inputs']
    C.require(C.binding(reference) in frozen.values(), 'published v3 geometry is not protocol bound')
    previous = {r['id']: r for r in C.rows(reference) if r['method'] == 'N3_CORNERWISE_ROLE'}
    current = {r['id']: r for r in geometry if r['method'] == 'N3_CORNERWISE_ROLE'}
    C.require(set(previous) == set(current) and len(current) == 245, 'old v3 parity cohort differs')
    rows, failures = [], []
    def numeric(a, b, label, diffs, exact):
        if a is None or b is None:
            if a != b: failures.append(dict(id=fid, check=label, reason='missing_value_differs'))
            return
        x, y = np.asarray(a, float), np.asarray(b, float)
        if x.shape != y.shape or not np.array_equal(np.isfinite(x), np.isfinite(y)):
            failures.append(dict(id=fid, check=label, reason='shape_or_finite_pattern_differs'))
            return
        same = np.array_equal(x, y, equal_nan=True)
        exact[label] = bool(same)
        valid = np.isfinite(x) & np.isfinite(y)
        difference = float(np.max(np.abs(x[valid] - y[valid]))) if valid.any() else 0.
        diffs[label] = difference
        if not np.allclose(x, y, atol=1e-7, rtol=0, equal_nan=True):
            failures.append(dict(id=fid, check=label, reason='numeric_difference_over_fixed_tolerance', max_abs=difference))
    for fid, a in current.items():
        b = previous[fid]; diffs = {}; exact = {}
        for key in ('input_points','native_points'):
            numeric(a.get(key), b.get(key), key, diffs, exact)
        for container in ('initial_pose','actual_pose','solver'):
            pa, pb = a.get(container) or {}, b.get(container) or {}
            for key in ('R_cf','R_physical','centroid','cf_extents','projected','reprojection_px'):
                numeric(pa.get(key), pb.get(key), container+'.'+key, diffs, exact)
            for key in ('available','selected_hypothesis'):
                if pa.get(key) != pb.get(key): failures.append(dict(id=fid, check=container+'.'+key, reason='categorical_difference'))
        for key in ('output_status','new_pose_estimated','fallback_used','pose_available','no_pose',
                    'hidden_initial','hidden_after','hidden_set_changed','reprojected_ids','selected_corner_ids',
                    'hidden_reprojected','output_coordinate_sources'):
            if a.get(key) != b.get(key): failures.append(dict(id=fid, check=key, reason='categorical_or_ID_difference'))
        for key in ('state','reason','fit_input_ids','final_inliers','generator_ids','prior_used'):
            if a['solver'].get(key) != b['solver'].get(key): failures.append(dict(id=fid, check='solver.'+key, reason='categorical_or_ID_difference'))
        ca = (a.get('observation_contract') or {}).get('cornerwise_records', [])
        cb = (b.get('observation_contract') or {}).get('cornerwise_records', [])
        if len(ca) != len(cb): failures.append(dict(id=fid, check='corner_records', reason='length_difference'))
        else:
            for ra, rb in zip(ca, cb):
                label = 'corner.'+str(ra['id'])
                for key in ('id','reason','accepted','heldout_fit_requested_excluded_ids','heldout_pose_calls','fit_input_ids','final_inlier_ids'):
                    if ra.get(key) != rb.get(key): failures.append(dict(id=fid, check=label+'.'+key, reason='categorical_or_ID_difference'))
                for key in ('candidate_xy','native_N3_xy','candidate_LOO_residual_px','native_N3_LOO_residual_px'):
                    numeric(ra.get(key), rb.get(key), label+'.'+key, diffs, exact)
        rows.append(dict(id=fid, numeric_max_abs=diffs, numeric_value_exact_equal=exact,
                         all_checked_numeric_values_exact=all(exact.values()),
                         nonzero_numeric_differences={k:v for k,v in diffs.items() if v != 0}))
    result = dict(schema='unchanged_v3_control_geometry_parity_v4', passed=not failures,
        frames=len(rows), atol=1e-7, rtol=0, reference_geometry=C.binding(reference),
        new_geometry=C.binding(Path(args.output)/'GEOMETRY_SEALED.jsonl.gz'),
        checked_before_GT_reference_reads=True, reference_used_for_pose_or_selection=False,
        numerically_exact_frames=sum(r['all_checked_numeric_values_exact'] for r in rows),
        within_tolerance_is_not_bit_exact=True, rows=rows, failures=failures,
        new_model_PnP_optimizer_training_RGB_calls=0)
    C.write_new(C.output_path(args,'V3_CONTROL_PARITY.json'), C.finite(result))
    C.require(result['passed'], 'fresh old-v3 control differs; preserve parity failure before scoring')
    return result


def sealed(args):
    C.verify_protocol(args)
    C.protect(args)
    folder = Path(args.output)
    seal = C.read(folder / 'GEOMETRY_SEAL.json')
    inference = C.read(folder / 'INFERENCE_RECEIPT.json')
    C.require(inference['complete'] is True and inference['cleanup_error'] is None and
              inference['actual_complete_frames'] == 245 and inference['method_rows'] == 980 and
              inference['fixed_rows'] == 490,
              'completed inference and successful cleanup receipt required before truth')
    C.require(seal['complete'] and seal['GT_read_allowed'] is False and
              seal['frames'] == 245 and seal['rows'] == 980 and seal['fixed_rows'] == 490,
              'complete four-arm pre-truth seal absent')
    C.require(seal['protocol'] == C.binding(args.protocol), 'seal protocol differs')
    C.require(tuple(seal['methods']) == tuple(C.METHODS), 'sealed methods differ')
    for key, name in (('geometry', 'GEOMETRY_SEALED.jsonl.gz'),
                      ('fixed_geometry', 'FIXED_GEOMETRY_SEALED.jsonl.gz'),
                      ('observations', 'OBSERVATIONS.jsonl.gz'),
                      ('parity', 'BASE_N3_PARITY.json')):
        C.bound(folder / name, seal[key], 'sealed ' + key)
    ids = {f['id'] for f in C.cohort_frames(args)}
    C.require(len(ids) == 245, 'selected cohort differs')
    geometry = list(C.rows(folder / 'GEOMETRY_SEALED.jsonl.gz'))
    fixed = list(C.rows(folder / 'FIXED_GEOMETRY_SEALED.jsonl.gz'))
    observations = list(C.rows(folder / 'OBSERVATIONS.jsonl.gz'))
    for values, methods in ((geometry, C.METHODS), (fixed, ('BASE', 'N3_SUBPIX'))):
        C.require(len(values) == 245 * len(methods) and
                  Counter(r['method'] for r in values) == {m: 245 for m in methods} and
                  len({(r['method'], r['id']) for r in values}) == len(values) and
                  all({r['id'] for r in values if r['method'] == m} == ids for m in methods),
                  'sealed method/ID population differs')
        C.require(not any(k in row for row in values for k in
                          ('pose', 'corner', 'mask_audit', 'baseline_pose', 'baseline_corner',
                           'evaluation_reference')),
                  'pre-truth geometry contains posthoc scores')
    C.require(len(observations) == len({r['id'] for r in observations}) == 245 and
              {r['id'] for r in observations} == ids and
              all(r.get('GT_input') is False for r in observations),
              'complete GT-free observation population differs')
    C.require(C.read(folder / 'BASE_N3_PARITY.json')['passed'] is True,
              'fresh Base/N3 parity did not pass')
    for row in geometry:
        if row['method'] == 'N3_CORNERWISE_ROLE':
            C.require(row['solver']['prior_used'] is True, 'old-v3 control lost its unchanged prior')
        else:
            C.require(row['solver']['prior_used'] is False and
                      row['solver']['initial_pose_used'] is False and
                      row['solver']['initial_projection_used'] is False and
                      row['solver']['initial_dimension_prior_used'] is False,
                      'independent geometry used an initial numeric pose prior')
    return seal, geometry, fixed


def run(args):
    import numpy as np
    seal, geometry, fixed = sealed(args)
    for name in ('PREDICTIONS.jsonl.gz', 'FIXED_PREDICTIONS.jsonl.gz',
                 'SCORING_RECEIPT.json', 'SCORING_PARITY.json', 'SCORING_STARTED.json', 'V3_CONTROL_PARITY.json'):
        C.output_path(args, name)
    C.write_new(C.output_path(args, 'SCORING_STARTED.json'), dict(
        protocol=C.binding(args.protocol),
        geometry_seal=C.binding(Path(args.output) / 'GEOMETRY_SEAL.json'),
        inference_receipt=C.binding(Path(args.output) / 'INFERENCE_RECEIPT.json'),
        complete_geometry_before_reference_reads=True, configured_rows=1470,
        configured_method_rows=980, configured_fixed_rows=490, new_pose_fits=0))
    start = time.monotonic()
    old_v3_parity = v3_control_parity(args, geometry)
    authority = {f['id']: f for f in C.read(C.OLD / 'INPUTS.json')['frames']}
    ids = [f['id'] for f in C.cohort_frames(args)]
    scope = dict(ids=ids, authority=authority, count=len(ids))
    from scripts.research.pallet_boundary_corner_refiner_20261010_v2.evaluate import forbid_fits
    with C.legacy_context(args) as (old, _, _), forbid_fits():
        from scripts.research.pallet_kp_corrected_supervision_20261010_v1.scoring_resume import references
        from scripts.research.pallet_observation_refiner_20261009_v1.evaluate import score
        E, frames, targets, reference = references(old, scope)
        historical = {(r['method'], r['id']): r for r in C.rows(C.OLD / 'FIXED_CONTROLS.jsonl.gz')
                      if r['method'] in ('BASE', 'N3_SUBPIX') and r['id'] in set(ids)}
        C.require(len(historical) == 490, 'historical fixed parity population differs')
        scored_fixed = [score(E, frames[r['id']], targets[r['id']], r) for r in fixed]
        controls = {(r['method'], r['id']): r for r in scored_fixed}
        parity = []
        for row in scored_fixed:
            previous = historical[(row['method'], row['id'])]
            C.require(row['pose']['available'] == previous['pose']['available'],
                      'fixed operational availability differs')
            differences = {k: abs(float(row['pose'][k]) - float(previous['pose'][k]))
                           for k in ('translation_cm', 'rotation_deg', 'ADDsym_m')
                           if row['pose']['available']}
            C.require(all(v <= 1e-7 for v in differences.values()), 'fresh fixed pose score differs')
            C.require(row['corner']['branch'] == previous['corner']['branch'],
                      'fresh fixed corner phase differs')
            a = np.asarray(row['corner']['observed_errors'], float)
            b = np.asarray(previous['corner']['observed_errors'], float)
            C.require(a.shape == b.shape and np.allclose(a, b, rtol=0, atol=1e-7),
                      'fresh fixed corner score differs')
            parity.append(dict(id=row['id'], method=row['method'], pose_max_abs=differences,
                               corner_max_abs=float(np.max(np.abs(a - b))) if a.size else 0.))
        label_path = Path(args.source_root) / (
            '_docs/experiments/pallet_combined_closeout_20261003_v1/closeout_20261006_v1/'
            'visibility_square/STATIC_VISIBILITY_MERGE_AUDIT.json')
        labels = {(r['frame_id'], r['corner_id']): r['category'] for r in old.read(label_path)['rows']
                  if r['population'] == 'DEV319' and r['frame_id'] in set(ids)}
        scored = [score(E, frames[r['id']], targets[r['id']], r) for r in geometry]
        for row in scored + scored_fixed:
            fid = row['id']
            baseline = controls[('N3_SUBPIX', fid)]
            target = targets[fid]
            perm = target['permutations'][baseline['corner'].get('branch', 0)][:8]
            states = [labels.get((fid, k), 'UNANNOTATED') for k in perm]
            H = set(row['hidden_initial'])
            humanH = {k for k, state in enumerate(states) if state == 'SELF_OCCLUDED'}
            known = {k for k, state in enumerate(states) if state != 'UNANNOTATED'}
            solver = row.get('solver') or {}
            eligible = set(solver.get('eligible', range(8)))
            used = set(solver.get('used', []))
            mask_applied = row['method'] in C.METHODS and row['method'] != 'N3_INDEPENDENT_ROBUST_NO_MASK'
            row['mask_audit'] = dict(
                human_states_native=states, oracle_phase_only_for_audit=True,
                reference_phase_control='N3_SUBPIX', native_reference_permutation=perm,
                known_ids=sorted(known), mask_applied=mask_applied,
                false_excluded_visible=sorted(H & {k for k, state in enumerate(states)
                                                  if state == 'DIRECT_VISIBLE'}),
                false_retained_self=sorted((humanH - H) & eligible),
                mask_wrong_on_known=bool((H ^ humanH) & known) if mask_applied else None,
                excluded_set_differs_from_human_self_on_known=bool((H ^ humanH) & known),
                remaining_human_direct_visible=len(used & {k for k, state in enumerate(states)
                                                         if state == 'DIRECT_VISIBLE'}),
                classification_difference_is_pose_failure=False,
                references_used_for_observation_selection=False)
            row['baseline_pose'] = baseline['pose']
            row['baseline_corner'] = baseline['corner']
            gt = np.asarray(target['gt'], float)[perm]
            valid = np.asarray(target['valid'], bool)[perm] & np.isfinite(gt).all(-1)
            row['evaluation_reference'] = dict(
                reference='existing geometric proxy; physical truth not independently validated',
                phase_control='N3_SUBPIX', native_points_px=gt.tolist(),
                valid_native_ids=np.flatnonzero(valid).tolist(), matched=bool(target['matched']),
                initial_N3_native_points=baseline['native_points'],
                initial_BASE_native_points=controls[('BASE', fid)]['native_points'],
                human_states_native=states)
            if row['new_pose_estimated']:
                C.require(set(row['hidden_initial']).isdisjoint(solver['fit_input_ids']),
                          'hidden observation entered final fit')
    C.save_rows(C.output_path(args, 'PREDICTIONS.jsonl.gz'), scored)
    C.save_rows(C.output_path(args, 'FIXED_PREDICTIONS.jsonl.gz'), scored_fixed)
    C.write_new(C.output_path(args, 'SCORING_PARITY.json'),
                dict(passed=True, rows=parity, atol=1e-7, rtol=0))
    C.verify_protocol(args)
    preserved = C.protect(args)
    C.write_new(C.output_path(args, 'SCORING_RECEIPT.json'), dict(
        schema='observation_only_cornerwise_posthoc_scoring_v4', complete=True, frames=245,
        scored_method_rows=980, fixed_control_rows=490, methods=list(C.METHODS),
        geometry_seal=C.binding(Path(args.output) / 'GEOMETRY_SEAL.json'),
        inference_receipt=C.binding(Path(args.output) / 'INFERENCE_RECEIPT.json'),
        geometry=C.binding(Path(args.output) / 'GEOMETRY_SEALED.jsonl.gz'),
        predictions=C.binding(Path(args.output) / 'PREDICTIONS.jsonl.gz'),
        fixed_predictions=C.binding(Path(args.output) / 'FIXED_PREDICTIONS.jsonl.gz'),
        old_v3_control_parity=C.binding(Path(args.output) / 'V3_CONTROL_PARITY.json'),
        old_v3_control_parity_before_reference_reads=True,
        reference=C.finite(reference),
        reference_image_ID_mapping=C.binding(C.CORRECTED / 'REFERENCE_ID_MAPPING.json'),
        human_visibility=old.binding(label_path), frozen_reference_phase='fresh fixed N3_SUBPIX',
        GT_access_only_after_complete_seal=True, new_detector_forwards=0, new_head_forwards=0,
        new_pose_fits=0, new_rays=0, scoring_fitting_entries_forbidden=True,
        elapsed_seconds=time.monotonic() - start,
        status_counts={method: dict(Counter(r['output_status'] for r in scored
                                           if r['method'] == method)) for method in C.METHODS},
        unchanged_protocol_inputs_verified_after_scoring=True,
        prior_tracked_and_user_checkout_protection=preserved,
        completed_local_audits_protected=True))
    print('CORNERWISE_REFINER_SCORED', len(scored), len(scored_fixed), flush=True)


def main():
    parser = C.parser(__doc__, ('preflight', 'score'))
    args = parser.parse_args()
    if args.stage == 'preflight':
        sealed(args)
        print('CORNERWISE_SCORING_PREFLIGHT_PASS', flush=True)
    else:
        run(args)


if __name__ == '__main__':
    main()
