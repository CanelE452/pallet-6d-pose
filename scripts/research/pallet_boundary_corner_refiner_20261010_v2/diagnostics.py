"""Posthoc correspondence and damage audit from sealed, already scored rows.

Only recorded proxy references are read. No image, detector, checkpoint, pose
solver, mesh, human-annotation loader, or additional GT loader is imported.
The common reference phase is the fresh fixed N3+SubPix phase in evaluate.py.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import time

import numpy as np

from . import common as C

CORRECT_PX = 8.0
POSE_KEYS = ('translation_cm', 'rotation_deg', 'ADDsym_m')
VISIBILITY_SETS = ('DIRECT_VISIBLE', 'SELF_OCCLUDED', 'ACTUALLY_REPROJECTED_H')


def describe(values):
    a = np.asarray(values, dtype=float)
    C.require(a.ndim == 1 and np.isfinite(a).all(), 'nonfinite diagnostic statistic')
    n = len(a)
    return dict(n=n, mean=float(a.mean()) if n else None,
                sample_variance=float(a.var(ddof=1)) if n > 1 else None,
                sample_std=float(a.std(ddof=1)) if n > 1 else None,
                median=float(np.median(a)) if n else None,
                P90=float(np.quantile(a, .9)) if n else None,
                maximum=float(a.max()) if n else None,
                ddof=1, quantile_method='numpy linear')


def points(value, label):
    a = np.asarray(value, dtype=float)
    C.require(a.shape in ((8, 2), (9, 2)), label + ' shape differs')
    return a[:8].copy()


def point_valid(a):
    return np.isfinite(a).all(-1) & ~np.all(a == -1, axis=-1)


def errors(a, reference, known):
    available = known & point_valid(a)
    e = np.full(8, np.nan)
    e[available] = np.linalg.norm(a[available] - reference[available], axis=-1)
    return e


def ids(value, label):
    answer = [int(k) for k in value]
    C.require(len(answer) == len(set(answer)) and all(0 <= k < 8 for k in answer),
              label + ' duplicate/out-of-range corner')
    return sorted(answer)


def layout(a, selected_ids):
    """Centered observation layout, not a new PnP/Jacobian rank check."""
    selected_ids = ids(selected_ids, 'layout')
    chosen = np.asarray(a, float)[selected_ids]
    finite = np.isfinite(chosen).all(-1)
    chosen = chosen[finite]
    dimension = np.asarray(a).shape[-1]
    if len(chosen) < 2:
        singular = np.zeros(dimension)
    else:
        singular = np.linalg.svd(chosen - chosen.mean(0), compute_uv=False)
        singular = np.pad(singular, (0, dimension - len(singular)))
    relative = singular / singular[0] if singular[0] > 0 else np.zeros(dimension)
    return dict(ids=selected_ids, finite_ids=[k for k, f in zip(selected_ids, finite) if f],
                n=len(chosen), singular_values=singular.tolist(), relative_singular_values=relative.tolist(),
                centered_numerical_rank=int(np.sum(relative > 1e-10)),
                rank_threshold_relative=1e-10,
                rms_radius=float(np.sqrt(np.mean(np.sum((chosen - chosen.mean(0)) ** 2, axis=1))))
                if len(chosen) else None)


def cuboid(dimensions):
    x, y, z = np.asarray(dimensions, float) / 2
    return np.asarray([[-x,-y,-z],[x,-y,-z],[x,y,-z],[-x,y,-z],
                       [-x,-y,z],[x,-y,z],[x,y,z],[-x,y,z]])


def corner_group(selected_ids, initial, incoming, outgoing):
    selected_ids = ids(selected_ids, 'corner group')
    comparable = [k for k in selected_ids if np.isfinite(initial[k]) and np.isfinite(outgoing[k])]
    input_comparable = [k for k in selected_ids if np.isfinite(incoming[k]) and np.isfinite(outgoing[k])]
    damaged = [k for k in comparable if initial[k] <= 5. and outgoing[k] > 10.]
    return dict(ids=selected_ids, reference_count=len(selected_ids),
                initial_N3_available_ids=[k for k in selected_ids if np.isfinite(initial[k])],
                input_available_ids=[k for k in selected_ids if np.isfinite(incoming[k])],
                output_available_ids=[k for k in selected_ids if np.isfinite(outgoing[k])],
                N3_to_output_comparable_ids=comparable, input_to_output_comparable_ids=input_comparable,
                damage_N3_le5_to_output_gt10_ids=damaged,
                improved_vs_N3_ids=[k for k in comparable if outgoing[k] < initial[k] - 1e-9],
                worsened_vs_N3_ids=[k for k in comparable if outgoing[k] > initial[k] + 1e-9])


def annotate(row, observation, fixed, label):
    fid = row['id']
    reference = row['evaluation_reference']
    C.require(reference['phase_control'] == 'N3_SUBPIX', 'diagnostic reference phase changed')
    gt = points(reference['native_points_px'], 'reference')
    gt_ids = ids(reference['valid_native_ids'], 'reference')
    known = np.zeros(8, bool)
    known[gt_ids] = True
    known &= point_valid(gt) & bool(reference['matched'])
    qn3 = points(fixed['native_points'], 'fixed N3')
    C.require(np.array_equal(qn3, points(reference['initial_N3_native_points'], 'stored N3'), equal_nan=True),
              'recorded initial N3 reference differs from fixed row')
    incoming = points(row['input_points'], 'input')
    outgoing = points(row['native_points'], 'output')
    e0, ei, eo = (errors(a, gt, known) for a in (qn3, incoming, outgoing))
    correct = set(np.flatnonzero(np.isfinite(ei) & (ei <= CORRECT_PX)).tolist())
    incorrect = set(np.flatnonzero(np.isfinite(ei) & (ei > CORRECT_PX)).tolist())
    unknown = set(range(8)) - correct - incorrect
    solver = row.get('solver') or {}
    eligible = ids(solver.get('eligible', []), 'eligible')
    used = ids(solver.get('used', []), 'used')
    fitted = ids(solver.get('fit_input_ids', []), 'fit')
    inliers = ids(solver.get('final_inliers', []), 'final inlier')
    H = ids(row['hidden_initial'], 'hidden')
    reprojected = ids(row['reprojected_ids'], 'reprojected')
    C.require(not row['new_pose_estimated'] or set(H).isdisjoint(fitted), 'hidden point entered fit')
    C.require(reprojected == (H if row['new_pose_estimated'] else []), 'actual reprojection ID/status mismatch')
    C.require(set(used) <= set(eligible) and set(inliers) <= set(used) and set(fitted) <= set(used),
              'solver observation ID provenance differs')
    if not row['new_pose_estimated']:
        C.require(np.array_equal(outgoing, qn3, equal_nan=True), 'fallback output differs from fixed N3')
    states = reference['human_states_native']
    C.require(len(states) == 8 and states == row['mask_audit']['human_states_native'], 'human state phase differs')
    direct = {k for k,s in enumerate(states) if s == 'DIRECT_VISIBLE'}
    self_hidden = {k for k,s in enumerate(states) if s == 'SELF_OCCLUDED'}
    known_mask = {k for k,s in enumerate(states) if s != 'UNANNOTATED'}
    wrong = bool((set(H) ^ self_hidden) & known_mask)
    C.require(wrong == row['mask_audit']['mask_wrong_on_known'], 'recorded known-mask predicate differs')
    comparable = bool(row['pose']['available'] and fixed['pose']['available'])
    delta = {k:float(row['pose'][k] - fixed['pose'][k]) for k in POSE_KEYS} if comparable else {}
    both_better = comparable and delta['translation_cm'] < -1e-9 and delta['rotation_deg'] < -1e-9
    both_worse = comparable and delta['translation_cm'] > 1e-9 and delta['rotation_deg'] > 1e-9
    outcome = ('NO_COMMON_OPERATIONAL_POSE' if not comparable else
               'BOTH_TRANSLATION_ROTATION_IMPROVED' if both_better else
               'BOTH_TRANSLATION_ROTATION_WORSENED' if both_worse else 'MIXED_OR_EQUAL')
    dimensions = solver.get('cf_extents') or row['initial_pose'].get('cf_extents')
    model = cuboid(dimensions) if dimensions is not None else None
    pools = {}
    for name, pool in (('eligible', eligible), ('used', used), ('fit_input', fitted), ('final_inlier', inliers),
                       ('accepted_new_pose_final_inlier', inliers if row['new_pose_estimated'] else [])):
        good = sorted(set(pool) & correct)
        pools[name] = dict(ids=pool, correct_ids=good, incorrect_ids=sorted(set(pool) & incorrect),
            unknown_ids=sorted(set(pool) & unknown), human_direct_ids=sorted(set(pool) & direct),
            correct_human_direct_ids=sorted(set(pool) & correct & direct),
            image_layout=layout(incoming, pool), correct_image_layout=layout(incoming, good),
            cuboid_ID_layout=layout(model, pool) if model is not None else None,
            correct_cuboid_ID_layout=layout(model, good) if model is not None else None)
    contract = row.get('observation_contract') or {}
    checks = contract.get('final_line_consensus_checks', [])
    admission = contract.get('corner_admission', [])
    computed = observation.get('corners', [])
    admitted_ids = set(contract.get('validated_boundary_corner_ids', []))
    replaced_ids = set(contract.get('hybrid_boundary_corner_ids', []))
    boundary_evidence = []
    for corner in computed:
        k = int(corner['id'])
        xy = np.asarray(corner['xy'], float)
        candidate_error = float(np.linalg.norm(xy-gt[k])) if known[k] and point_valid(xy[None])[0] else None
        n3_error = float(e0[k]) if np.isfinite(e0[k]) else None
        boundary_evidence.append(dict(id=k, xy=xy.tolist(), edges=corner['edges'], radius_px=corner['radius_px'],
            reference_available=candidate_error is not None, error_px=candidate_error,
            correct_within8px=candidate_error <= CORRECT_PX if candidate_error is not None else None,
            N3_error_px=n3_error, error_delta_candidate_minus_N3_px=candidate_error-n3_error
            if candidate_error is not None and n3_error is not None else None,
            admitted=k in admitted_ids, selected_for_hybrid=k in replaced_ids,
            ID_in_solver_pool=k in used, ID_is_final_inlier=k in inliers,
            boundary_coordinate_in_solver_pool=bool(k in used and
                (row['method']=='VALIDATED_ROLE_ONLY' and k in admitted_ids or
                 row['method'].startswith('N3_VALIDATED_ROLE') and k in replaced_ids)),
            displayed_as_boundary_observation=bool(row['new_pose_estimated'] and k not in H and
                (row['method']=='VALIDATED_ROLE_ONLY' and k in admitted_ids or
                 row['method'].startswith('N3_VALIDATED_ROLE') and k in replaced_ids)),
            no_hybrid_coordinate_substitution_on_fallback=True))
    groups = {name:corner_group(sorted(which & set(np.flatnonzero(known))), e0, ei, eo)
              for name, which in (('DIRECT_VISIBLE',direct), ('SELF_OCCLUDED',self_hidden),
                                  ('ACTUALLY_REPROJECTED_H',set(reprojected)))}
    return dict(id=fid, session=row['session'], method=row['method'], difficulty_label=label,
        output_status=row['output_status'], new_pose_estimated=bool(row['new_pose_estimated']),
        fallback_used=bool(row['fallback_used']), pose_available=bool(row['pose']['available']),
        solver_state=solver.get('state'), solver_reason=solver.get('reason'),
        pose_jacobian=solver.get('geometry', {}).get('jacobian'),
        reference_phase='fresh fixed N3_SUBPIX', reference=row['evaluation_reference']['reference'],
        reference_valid_ids=sorted(np.flatnonzero(known).tolist()), reference_native_points_px=gt.tolist(),
        input_native_points_px=incoming.tolist(), output_native_points_px=outgoing.tolist(),
        fixed_N3_native_points_px=qn3.tolist(), human_states_native=states,
        input_error_px=ei.tolist(), output_error_px=eo.tolist(), fixed_N3_error_px=e0.tolist(),
        input_correct_ids=sorted(correct), input_incorrect_ids=sorted(incorrect), input_unknown_ids=sorted(unknown),
        input_correct_threshold_px=CORRECT_PX, pools=pools,
        final_inlier_scope='accepted_new_pose' if row['new_pose_estimated'] else 'rejected_candidate_or_no_new_pose',
        hidden_initial_ids=H, human_self_hidden_ids=sorted(self_hidden), human_direct_visible_ids=sorted(direct),
        human_direct_reference_missing_ids=sorted(direct-set(np.flatnonzero(known))),
        human_self_reference_missing_ids=sorted(self_hidden-set(np.flatnonzero(known))), known_mask_ids=sorted(known_mask),
        mask_fully_annotated=len(known_mask)==8, mask_wrong_on_known=wrong,
        false_excluded_direct_ids=sorted(set(H)&direct), false_retained_self_ids=sorted((self_hidden-set(H))&set(eligible)),
        hidden_set_changed=bool(row['hidden_set_changed']), mask_difference_is_pose_failure=False,
        actual_reprojected_ids=reprojected, actual_reprojected_reference_missing_ids=sorted(set(reprojected)-set(np.flatnonzero(known))),
        comparable_to_N3=comparable, delta_vs_N3=delta, pose_outcome_vs_N3=outcome,
        wrong_known_mask_both_pose_improved=bool(wrong and both_better),
        matching_known_mask_both_pose_worsened=bool(not wrong and both_worse),
        matching_known_mask_does_not_imply_unknown_labels_are_correct=True,
        corner_groups=groups,
        observations=dict(selected_query_count=observation.get('selected_queries'),
            decoded_line_count=len(observation.get('lines', [])), computed_corner_count=len(computed),
            computed_corner_ids=[int(c['id']) for c in computed], computed_corner_radii_px=[c['radius_px'] for c in computed],
            admitted_corner_ids=contract.get('validated_boundary_corner_ids', []),
            hybrid_replaced_corner_ids=contract.get('hybrid_boundary_corner_ids', []),
            hybrid_rejected_corner_ids=contract.get('hybrid_rejected_boundary_corner_ids', []),
            final_line_consensus_checks=checks, invalid_final_line_edges=contract.get('invalid_final_line_edges', []),
            corner_admission=admission, per_validated_corner=contract.get('per_validated_corner', []),
            boundary_corner_evidence=boundary_evidence,
            line_veto_count=sum(not c['accepted'] for c in checks),
            corner_admission_reasons=dict(Counter(c['reason'] for c in admission)),
            line_and_corner_observations_are_correlated=True, partial_lines_are_pose_inputs=False))


def summarize(values):
    answer = dict(n=len(values), status_counts=dict(Counter(r['output_status'] for r in values)),
        new_pose=sum(r['new_pose_estimated'] for r in values), fallback=sum(r['fallback_used'] for r in values),
        no_pose=sum(not r['pose_available'] for r in values),
        wrong_known_mask_both_pose_improved_ids=[r['id'] for r in values if r['wrong_known_mask_both_pose_improved']],
        matching_known_mask_both_pose_worsened_ids=[r['id'] for r in values if r['matching_known_mask_both_pose_worsened']],
        actual_hidden_reprojection_frames=sum(bool(r['actual_reprojected_ids']) for r in values),
        actual_hidden_reprojection_reference_missing=sum(len(r['actual_reprojected_reference_missing_ids']) for r in values),
        hidden_set_changed=sum(r['hidden_set_changed'] for r in values))
    answer['mask_pose_groups'] = {}
    for wrong in (False, True):
        for outcome in ('BOTH_TRANSLATION_ROTATION_IMPROVED', 'BOTH_TRANSLATION_ROTATION_WORSENED',
                        'MIXED_OR_EQUAL', 'NO_COMMON_OPERATIONAL_POSE'):
            selected = [r for r in values if r['mask_wrong_on_known']==wrong and r['pose_outcome_vs_N3']==outcome]
            key = ('wrong_on_known' if wrong else 'matching_on_known') + '__' + outcome
            answer['mask_pose_groups'][key] = dict(n=len(selected), ids=[r['id'] for r in selected],
                fully_annotated_mask_count=sum(r['mask_fully_annotated'] for r in selected),
                new_pose=sum(r['new_pose_estimated'] for r in selected), fallback=sum(r['fallback_used'] for r in selected),
                delta_vs_N3={k:describe([r['delta_vs_N3'][k] for r in selected if r['comparable_to_N3']]) for k in POSE_KEYS})
    answer['correspondences'] = {pool:dict(
        correct_count=describe([len(r['pools'][pool]['correct_ids']) for r in values]),
        incorrect_count=describe([len(r['pools'][pool]['incorrect_ids']) for r in values]),
        unknown_count=describe([len(r['pools'][pool]['unknown_ids']) for r in values]),
        correct_image_rank_counts=dict(Counter(r['pools'][pool]['correct_image_layout']['centered_numerical_rank'] for r in values)))
        for pool in ('eligible','used','fit_input','final_inlier','accepted_new_pose_final_inlier')}
    answer['corner_errors'] = {}
    for name in VISIBILITY_SETS:
        comparable = [(r,k) for r in values for k in r['corner_groups'][name]['N3_to_output_comparable_ids']]
        selected = [(r,k) for r in values for k in r['corner_groups'][name]['ids']]
        answer['corner_errors'][name] = dict(reference_count=len(selected), comparable_N3_output=len(comparable),
            N3_error_px=describe([r['fixed_N3_error_px'][k] for r,k in comparable]),
            output_error_same_N3_set_px=describe([r['output_error_px'][k] for r,k in comparable]),
            delta_output_minus_N3_px=describe([r['output_error_px'][k]-r['fixed_N3_error_px'][k] for r,k in comparable]),
            input_error_available_px=describe([r['input_error_px'][k] for r,k in selected if np.isfinite(r['input_error_px'][k])]),
            output_error_available_px=describe([r['output_error_px'][k] for r,k in selected if np.isfinite(r['output_error_px'][k])]),
            damage_N3_le5_to_output_gt10=sum(len(r['corner_groups'][name]['damage_N3_le5_to_output_gt10_ids']) for r in values),
            damaged_frame_ids=[r['id'] for r in values if r['corner_groups'][name]['damage_N3_le5_to_output_gt10_ids']])
    answer['observation_counts'] = {key:describe([r['observations'][key] for r in values])
        for key in ('decoded_line_count','computed_corner_count','line_veto_count')}
    answer['observation_counts'].update({key:describe([len(r['observations'][key]) for r in values])
        for key in ('admitted_corner_ids','hybrid_replaced_corner_ids','hybrid_rejected_corner_ids')})
    answer['computed_corner_radii_px'] = describe([x for r in values for x in r['observations']['computed_corner_radii_px']])
    answer['boundary_corner_quality'] = {}
    for scope in ('computed','admitted','selected_for_hybrid','displayed_as_boundary_observation'):
        candidates = [c for r in values for c in r['observations']['boundary_corner_evidence']
                      if scope=='computed' or c[scope]]
        known = [c for c in candidates if c['reference_available']]
        paired = [c for c in known if c['N3_error_px'] is not None]
        answer['boundary_corner_quality'][scope] = dict(total_count=len(candidates), known_count=len(known),
            unknown_count=len(candidates)-len(known), correct_within8px=sum(c['correct_within8px'] for c in known),
            error_px=describe([c['error_px'] for c in known]),
            N3_error_same_paired_set_px=describe([c['N3_error_px'] for c in paired]),
            candidate_error_same_paired_set_px=describe([c['error_px'] for c in paired]),
            candidate_minus_N3_error_px=describe([c['error_delta_candidate_minus_N3_px'] for c in paired]),
            improves_vs_N3=sum(c['error_delta_candidate_minus_N3_px'] < -1e-9 for c in paired),
            harms_vs_N3=sum(c['error_delta_candidate_minus_N3_px'] > 1e-9 for c in paired))
    return answer


def run(args):
    start = time.monotonic()
    root = Path(args.input_root)
    inputs = dict(protocol=Path(args.protocol), statistics_protocol=C.DOC/'STATISTICS_PROTOCOL.json',
        cohort=Path(args.cohort), geometry_seal=root/'GEOMETRY_SEAL.json', scoring_receipt=root/'SCORING_RECEIPT.json',
        predictions=root/'PREDICTIONS.jsonl.gz', fixed_predictions=root/'FIXED_PREDICTIONS.jsonl.gz',
        geometry=root/'GEOMETRY_SEALED.jsonl.gz', fixed_geometry=root/'FIXED_GEOMETRY_SEALED.jsonl.gz',
        observations=root/'OBSERVATIONS.jsonl.gz', code=Path(__file__))
    before = {name:C.binding(path) for name,path in inputs.items()}
    for name in ('POSTHOC_ROWS.jsonl.gz', 'DIAGNOSTICS.json'):
        C.output_path(args, name)
    seal, receipt = C.read(inputs['geometry_seal']), C.read(inputs['scoring_receipt'])
    C.require(seal['complete'] and seal['GT_read_allowed'] is False and receipt['complete'] and
              receipt['new_detector_forwards']==receipt['new_head_forwards']==receipt['new_pose_fits']==0,
              'complete GT-free seal/scoring receipt required')
    C.require(seal['protocol']==before['protocol'] and receipt['geometry_seal']==before['geometry_seal'],
              'sealed protocol/reference binding differs')
    for name in ('geometry','fixed_geometry','observations'):
        C.require(seal[name]==before[name], 'seal '+name+' binding differs')
    for name in ('predictions','fixed_predictions'):
        C.require(receipt[name]==before[name], 'scoring '+name+' binding differs')
    cohort = C.read(args.cohort)
    expected = cohort['ids']
    C.require(len(expected)==len(set(expected))==245, 'frozen selected245 population differs')
    labels = {r['id']:r['label'] for r in cohort['frames']}
    C.require(Counter(labels.values())==dict(clean=153,moderate=92), 'easy/middle population differs')
    predictions = list(C.rows(inputs['predictions']))
    fixed = list(C.rows(inputs['fixed_predictions']))
    observations = list(C.rows(inputs['observations']))
    C.require(len(predictions)==1225 and Counter(r['method'] for r in predictions)=={m:245 for m in C.METHODS},
              'scored method count differs')
    C.require(len(fixed)==490 and Counter(r['method'] for r in fixed)==dict(BASE=245,N3_SUBPIX=245), 'fixed count differs')
    for values, methods in ((predictions,C.METHODS),(fixed,('BASE','N3_SUBPIX'))):
        C.require(len({(r['id'],r['method']) for r in values})==len(values) and
                  all({r['id'] for r in values if r['method']==m}==set(expected) for m in methods), 'row IDs differ')
    C.require(len(observations)==245 and len({r['id'] for r in observations})==245 and
              {r['id'] for r in observations}==set(expected), 'observation IDs differ')
    by_observation = {r['id']:r for r in observations}
    by_fixed = {r['id']:r for r in fixed if r['method']=='N3_SUBPIX'}
    annotations = [annotate(r, by_observation[r['id']], by_fixed[r['id']], labels[r['id']]) for r in predictions]
    summary = {scope:{m:summarize([r for r in annotations if r['method']==m and
        (label is None or r['difficulty_label']==label)]) for m in C.METHODS}
        for scope,label in (('combined',None),('easy','clean'),('medium','moderate'))}
    C.require({name:C.binding(path) for name,path in inputs.items()}==before, 'diagnostic inputs changed')
    C.save_rows(C.output_path(args,'POSTHOC_ROWS.jsonl.gz'), annotations)
    C.write_new(C.output_path(args,'DIAGNOSTICS.json'), dict(schema='boundary_refiner_posthoc_diagnostics_v2',
        complete=True, frames=245, rows=1225, reference_phase='fresh fixed N3_SUBPIX',
        strata=summary, input_bindings=before, rows_binding=C.binding(Path(args.output)/'POSTHOC_ROWS.jsonl.gz'),
        correspondence_correctness='known matched proxy reference, finite supplied point, Euclidean error<=8px',
        unknown_inputs_are_neither_correct_nor_incorrect=True,
        layout_rank='centered observation SVD; not PnP/Jacobian observability; 3D is native cuboid ID layout',
        uncertainty_is_source_calibrated_diagnostic_not_real_coverage_guarantee=True,
        visibility_reference='existing geometric proxy/manual states; no independently validated physical GT claim',
        per_point_deltas_use_same_available_N3_output_set=True,
        no_additional_GT_reads=True, new_detector_forwards=0, new_head_forwards=0,
        new_pose_fits=0, new_rays=0, new_training_updates=0, elapsed_seconds=time.monotonic()-start))
    print('BOUNDARY_POSTHOC_DIAGNOSTICS', len(annotations), flush=True)


def main():
    p = C.parser(__doc__)
    p.set_defaults(output=str(C.DOC))
    p.add_argument('--input-root', default=str(C.PRIVATE/'accuracy'))
    run(p.parse_args())


if __name__ == '__main__':
    main()
