"""Score complete sealed geometry using the unchanged proxy-reference metric.

This stage opens only the selected reference frames after the GT-free geometry
seal exists. It never runs a detector, head, PnP, pose optimizer or ray caster.
"""
from __future__ import annotations

from contextlib import contextmanager
from collections import Counter
from pathlib import Path
import time

from . import common as C


@contextmanager
def forbid_fits():
    import cv2
    import scipy.optimize
    from . import pose
    entries = [(cv2, n) for n in ('solvePnP', 'solvePnPGeneric', 'solvePnPRefineLM')]
    entries += [(scipy.optimize, 'least_squares'), (pose, 'refine')]
    saved = [(obj, name, getattr(obj, name)) for obj, name in entries]
    def forbidden(*_, **__):
        raise RuntimeError('BOUNDARY_SCORING_GUARD: pose fitting during posthoc scoring')
    for obj, name, _ in saved:
        setattr(obj, name, forbidden)
    try:
        yield
    finally:
        for obj, name, fn in saved:
            setattr(obj, name, fn)


def sealed(args):
    C.verify_protocol(args)
    seal = C.read(Path(args.output) / 'GEOMETRY_SEAL.json')
    C.require(seal['complete'] and seal['GT_read_allowed'] is False and seal['frames'] == 245 and
              seal['rows'] == 1225 and seal['fixed_rows'] == 490, 'complete pre-truth seal absent')
    C.require(seal['protocol'] == C.binding(args.protocol), 'seal protocol differs')
    for key, name in (('geometry', 'GEOMETRY_SEALED.jsonl.gz'),
                      ('fixed_geometry', 'FIXED_GEOMETRY_SEALED.jsonl.gz'),
                      ('observations', 'OBSERVATIONS.jsonl.gz'), ('parity', 'BASE_N3_PARITY.json')):
        C.bound(Path(args.output) / name, seal[key], 'sealed ' + key)
    ids = {f['id'] for f in C.cohort_frames(args)}
    geometry = list(C.rows(Path(args.output) / 'GEOMETRY_SEALED.jsonl.gz'))
    fixed = list(C.rows(Path(args.output) / 'FIXED_GEOMETRY_SEALED.jsonl.gz'))
    for values, methods in ((geometry, C.METHODS), (fixed, ('BASE', 'N3_SUBPIX'))):
        C.require(len(values) == 245 * len(methods) and
                  Counter(r['method'] for r in values) == {m:245 for m in methods} and
                  len({(r['method'], r['id']) for r in values}) == len(values) and
                  all({r['id'] for r in values if r['method'] == m} == ids for m in methods),
                  'sealed method/ID population differs')
        C.require(not any(k in row for row in values for k in
                          ('pose', 'corner', 'mask_audit', 'baseline_pose', 'evaluation_reference')),
                  'pre-truth geometry contains posthoc scores')
    return seal, geometry, fixed


def run(args):
    import numpy as np
    seal, geometry, fixed = sealed(args)
    prior = C.prior_snapshot(args.prior_bindings)
    for name in ('PREDICTIONS.jsonl.gz', 'FIXED_PREDICTIONS.jsonl.gz', 'SCORING_RECEIPT.json', 'SCORING_PARITY.json'):
        C.output_path(args, name)
    C.write_new(C.output_path(args, 'SCORING_STARTED.json'), dict(
        protocol=C.binding(args.protocol), geometry_seal=C.binding(Path(args.output) / 'GEOMETRY_SEAL.json'),
        complete_geometry_before_reference_reads=True, configured_rows=1715, new_pose_fits=0))
    start = time.monotonic()
    authority = {f['id']: f for f in C.read(C.OLD / 'INPUTS.json')['frames']}
    ids = [f['id'] for f in C.cohort_frames(args)]
    scope = dict(ids=ids, authority=authority, count=len(ids))
    with C.legacy_context(args) as (old, _, _), forbid_fits():
        from scripts.research.pallet_kp_corrected_supervision_20261010_v1.scoring_resume import references
        from scripts.research.pallet_observation_refiner_20261009_v1.evaluate import score
        E, frames, targets, reference = references(old, scope)
        historical = {(r['method'], r['id']):r for r in C.rows(C.OLD / 'FIXED_CONTROLS.jsonl.gz')
                      if r['method'] in ('BASE', 'N3_SUBPIX') and r['id'] in set(ids)}
        scored_fixed = [score(E, frames[r['id']], targets[r['id']], r) for r in fixed]
        controls = {(r['method'],r['id']):r for r in scored_fixed}
        parity = []
        for row in scored_fixed:
            previous = historical[(row['method'],row['id'])]
            C.require(row['pose']['available'] == previous['pose']['available'], 'fixed operational availability differs')
            differences = {k:abs(float(row['pose'][k]) - float(previous['pose'][k]))
                           for k in ('translation_cm', 'rotation_deg', 'ADDsym_m') if row['pose']['available']}
            C.require(all(v <= 1e-7 for v in differences.values()), 'fresh fixed pose score differs')
            C.require(row['corner']['branch'] == previous['corner']['branch'], 'fresh fixed corner phase differs')
            a,b = np.asarray(row['corner']['observed_errors'],float),np.asarray(previous['corner']['observed_errors'],float)
            C.require(a.shape == b.shape and np.allclose(a,b,rtol=0,atol=1e-7), 'fresh fixed corner score differs')
            parity.append(dict(id=row['id'],method=row['method'],pose_max_abs=differences,
                               corner_max_abs=float(np.max(np.abs(a-b))) if a.size else 0.))
        label_path = Path(args.source_root) / '_docs/experiments/pallet_combined_closeout_20261003_v1/closeout_20261006_v1/visibility_square/STATIC_VISIBILITY_MERGE_AUDIT.json'
        labels = {(r['frame_id'],r['corner_id']):r['category'] for r in old.read(label_path)['rows']
                  if r['population'] == 'DEV319' and r['frame_id'] in set(ids)}
        scored = [score(E, frames[r['id']], targets[r['id']], r) for r in geometry]
        for row in scored + scored_fixed:
            fid = row['id']; baseline = controls[('N3_SUBPIX',fid)]; target = targets[fid]
            perm = target['permutations'][baseline['corner'].get('branch',0)][:8]
            states = [labels.get((fid,k),'UNANNOTATED') for k in perm]
            H = set(row['hidden_initial']); humanH = {k for k,s in enumerate(states) if s=='SELF_OCCLUDED'}
            known = {k for k,s in enumerate(states) if s!='UNANNOTATED'}
            solver = row.get('solver') or {}
            eligible = set(solver.get('eligible',range(8))); used = set(solver.get('used',[]))
            row['mask_audit'] = dict(human_states_native=states, oracle_phase_only_for_audit=True,
                reference_phase_control='N3_SUBPIX', native_reference_permutation=perm,
                known_ids=sorted(known), false_excluded_visible=sorted(H & {k for k,s in enumerate(states) if s=='DIRECT_VISIBLE'}),
                false_retained_self=sorted((humanH-H)&eligible), mask_wrong_on_known=bool((H^humanH)&known),
                remaining_human_direct_visible=len(used & {k for k,s in enumerate(states) if s=='DIRECT_VISIBLE'}),
                classification_difference_is_pose_failure=False, references_used_for_observation_selection=False)
            row['baseline_pose'] = baseline['pose']; row['baseline_corner'] = baseline['corner']
            gt = np.asarray(target['gt'],float)[perm]
            valid = np.asarray(target['valid'],bool)[perm] & np.isfinite(gt).all(-1)
            row['evaluation_reference'] = dict(reference='existing geometric proxy; physical truth not independently validated',
                phase_control='N3_SUBPIX', native_points_px=gt.tolist(), valid_native_ids=np.flatnonzero(valid).tolist(),
                matched=bool(target['matched']), initial_N3_native_points=baseline['native_points'],
                initial_BASE_native_points=controls[('BASE',fid)]['native_points'], human_states_native=states)
            if row['new_pose_estimated']:
                C.require(set(row['hidden_initial']).isdisjoint(solver['fit_input_ids']), 'hidden observation entered final fit')
    C.save_rows(C.output_path(args,'PREDICTIONS.jsonl.gz'), scored)
    C.save_rows(C.output_path(args,'FIXED_PREDICTIONS.jsonl.gz'), scored_fixed)
    C.write_new(C.output_path(args,'SCORING_PARITY.json'), dict(passed=True,rows=parity,atol=1e-7,rtol=0))
    C.require(C.prior_snapshot(args.prior_bindings) == prior, 'prior files changed during scoring')
    C.write_new(C.output_path(args,'SCORING_RECEIPT.json'), dict(
        schema='validated_boundary_posthoc_scoring_v2',complete=True,frames=245,scored_method_rows=1225,fixed_control_rows=490,
        geometry_seal=C.binding(Path(args.output)/'GEOMETRY_SEAL.json'),
        geometry=C.binding(Path(args.output)/'GEOMETRY_SEALED.jsonl.gz'),
        predictions=C.binding(Path(args.output)/'PREDICTIONS.jsonl.gz'),
        fixed_predictions=C.binding(Path(args.output)/'FIXED_PREDICTIONS.jsonl.gz'),
        reference=C.finite(reference), reference_image_ID_mapping=C.binding(C.CORRECTED/'REFERENCE_ID_MAPPING.json'),
        human_visibility=old.binding(label_path), frozen_reference_phase='fresh fixed N3_SUBPIX',
        GT_access_only_after_complete_seal=True,new_detector_forwards=0,new_head_forwards=0,new_pose_fits=0,new_rays=0,
        scoring_fitting_entries_forbidden=True, elapsed_seconds=time.monotonic()-start,
        status_counts={m:dict(Counter(r['output_status'] for r in scored if r['method']==m)) for m in C.METHODS},
        preserved_materialized_files=len(prior)))
    print('BOUNDARY_REFINER_SCORED',len(scored),len(scored_fixed),flush=True)


def main():
    parser = C.parser(__doc__, ('preflight','score'))
    args = parser.parse_args()
    if args.stage == 'preflight':
        sealed(args)
        print('BOUNDARY_SCORING_PREFLIGHT_PASS',flush=True)
    else:
        run(args)


if __name__ == '__main__':
    main()
