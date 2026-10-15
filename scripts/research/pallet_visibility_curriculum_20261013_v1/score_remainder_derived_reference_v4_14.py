"""Score sealed remaining143 poses against the stored derived reference.

The selected camera-facing PnP pose supplies the nominal centroid.  An
unconfirmed signed canonical pair supplies rotation only after exact centroid
copy and C2 equivalence checks.  Candidate zero is fixed before predictions;
alternative W/D references never select the headline metric.  No annotations
are decoded by ``prepare`` or before the complete143 prediction seal.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import shutil

import numpy as np

from . import common as C
from .scoring import distribution, pose_metric
from scripts.annotate import pallet_geometry as PG
from scripts.annotate import pallet_symmetry as SYM

REMAINDER = C.DOC / 'nonsevere_v1/remainder_after_test49_v1'
INVENTORY = REMAINDER / 'inventory_v2/REMAINING_CLEAN_MODERATE143.json'
PREDICTIONS = REMAINDER / 'pose_evaluation_v4_14'
OUTPUT = REMAINDER / 'derived_reference_score_v4_14'
SYMMETRY = C.ROOT / 'challenge/real_gt_v2/SYMMETRY_CONTRACT.json'
ARM = 'FROZEN_WD_FACE_AWARE_PARTIAL_BI'
POLICY = dict(nominal_centroid='objects[0].camera_facing_pnp.pose_transform[:3,3]',
    canonical_rotation='provided confirmed canonical pose, else fixed candidate[0] only if exact t-copy and C2 pair',
    unresolved_rotation='unset; retain the translation row',
    alternative_CF_hypotheses='separate stored transforms only; no reconstruction from reprojection summaries',
    prediction_error_reference_selection=False, independent_physical_metrology_claimed=False,
    reference_axis_unconfirmed_and_intrinsics_UNKNOWN_disclosed=True)


def _verify(bindings):
    for relative, expected in bindings.items():
        if C.sha(C.ROOT / relative) != expected:
            raise ValueError('Binding changed: ' + relative)


def _transform(value):
    T = np.asarray(value, dtype=float)
    if T.shape != (4, 4) or not np.isfinite(T).all() or not np.array_equal(T[3], [0., 0., 0., 1.]):
        raise ValueError('Finite proper centroid-origin reference transform required')
    PG.validate_proper_rotation(T[:3, :3])
    return T


def _pair_check(records, nominal):
    """Validate the source conversion, exact translation copy and exact C2 pair."""
    errors = []
    transforms = []
    if len(records) != 2:
        return None, dict(eligible=False, reasons=['canonical signed pair count is not two'])
    for index, record in enumerate(records):
        try:
            T = _transform(record['pose_transform'])
            A = PG.canonical_to_camera_facing_transform(PG.AxisAssignment(record['axis_assignment']))
            if not np.array_equal(T[:3, 3], nominal[:3, 3]):
                errors.append(f'candidate {index} centroid is not an exact camera-facing copy')
            if not np.allclose(T[:3, :3], nominal[:3, :3] @ A, rtol=0, atol=1e-6):
                errors.append(f'candidate {index} rotation differs from its signed conversion')
            transforms.append(T)
        except (ValueError, KeyError, TypeError) as error:
            errors.append(f'candidate {index}: {error}')
    if len(transforms) == 2:
        yaw180 = PG.canonical_to_camera_facing_transform(PG.AxisAssignment.YAW_180)
        first = PG.AxisAssignment(records[0]['axis_assignment'])
        second = PG.AxisAssignment(records[1]['axis_assignment'])
        if (second.yaw_degrees - first.yaw_degrees) % 360 != 180:
            errors.append('signed assignments are not separated by 180 degrees')
        if not np.array_equal(transforms[1][:3, :3], transforms[0][:3, :3] @ yaw180):
            errors.append('candidate rotations are not an exact C2 equivalence class')
    return (transforms[0][:3, :3] if not errors else None,
            dict(eligible=not errors, reasons=errors, fixed_representative_index=0,
                 exact_centroid_copy=not any('centroid' in s for s in errors),
                 canonical_pair_is_evaluation_equivalence_not_signed_axis_confirmation=True))


def reference_from_document(document):
    """Pure reader; the caller must enforce the prediction-seal gate first."""
    if document.get('schema_version') != 'real_pallet_gt_v2' or len(document.get('objects', [])) != 1:
        raise ValueError('One explicit real_pallet_gt_v2 object required')
    obj = document['objects'][0]
    cf = obj['camera_facing_pnp']
    nominal = _transform(cf['pose_transform'])
    dimensions = [obj['physical_dimensions_m'][axis] for axis in ('x', 'y', 'z')]
    dimensions = np.asarray(dimensions, dtype=float)
    if dimensions.shape != (3,) or not np.isfinite(dimensions).all() or np.any(dimensions <= 0):
        raise ValueError('Explicit positive canonical physical dimensions required')
    R, pair_audit = _pair_check(obj.get('canonical_pose_candidates', []), nominal)
    confirmed = obj.get('canonical_pose')
    if confirmed is not None:
        signed = _transform(confirmed['pose_transform'])
        A = PG.canonical_to_camera_facing_transform(PG.AxisAssignment(confirmed['axis_assignment']))
        eligible = (cf.get('axis_assignment_confirmed') is True
                    and np.array_equal(signed[:3, 3], nominal[:3, 3])
                    and np.allclose(signed[:3, :3], nominal[:3, :3] @ A, rtol=0, atol=1e-6))
        R = signed[:3, :3] if eligible else None
        pair_audit['confirmed_canonical_used'] = eligible
        pair_audit['eligible'] = eligible
        if not eligible:
            pair_audit['reasons'].append('provided canonical pose lacks a confirmed exact centroid conversion')
    alternatives = []
    for index, hypothesis in enumerate(cf.get('hypotheses') or []):
        record = dict(index=index, name=hypothesis.get('camera_facing_hypothesis', hypothesis.get('hypothesis')))
        # The normal annotation summary does not persist the competing R/t.
        # Do not infer a reference from dimensions or its reported reprojection error.
        if hypothesis.get('pose_transform') is None:
            record.update(centroid_available=False, reason='stored hypothesis has summary only, no pose_transform')
        else:
            T = _transform(hypothesis['pose_transform'])
            record.update(centroid_available=True, centroid=T[:3, 3].tolist(),
                          nominal_centroid_distance_cm=float(np.linalg.norm(T[:3, 3]-nominal[:3, 3])*100))
        alternatives.append(record)
    return dict(t=nominal[:3, 3].tolist(), R=None if R is None else R.tolist(),
        xyz=dimensions.tolist(), order=2, audit=pair_audit, alternatives=alternatives,
        metadata=dict(schema_version=document['schema_version'], population_role=document.get('population_role'),
            gt_source=obj.get('gt_source', document.get('gt_source')),
            pose_status=obj.get('pose_status'), migration_status=obj.get('migration_status'),
            axis_assignment_confirmed=cf.get('axis_assignment_confirmed', False),
            canonical_pose_present=confirmed is not None,
            canonical_pose_candidate_count=len(obj.get('canonical_pose_candidates', [])),
            intrinsics_quality=document.get('intrinsics_quality'),
            selected_camera_facing_hypothesis=cf.get('selected_hypothesis'),
            reference_is_derived_annotation_PnP=True, independent_physical_pose_confirmed=False))


def metric_for_reference(prediction, reference):
    if reference['R'] is not None:
        result = pose_metric(prediction, reference)
    else:
        t = np.asarray(prediction['centroid'], dtype=float)
        if t.shape != (3,) or not np.isfinite(t).all():
            raise ValueError('Every fallback must expose its finite reported centroid')
        result = dict(translation_cm=float(np.linalg.norm(t-reference['t'])*100),
                      rotation_deg=None, ADDsym_cm=None, success_5cm_5deg=None)
    alternatives = []
    for alternative in reference['alternatives']:
        record = deepcopy(alternative)
        if record['centroid_available']:
            record['translation_cm'] = float(np.linalg.norm(np.asarray(prediction['centroid'])-record['centroid'])*100)
        alternatives.append(record)
    values = [record['translation_cm'] for record in alternatives if record['centroid_available']]
    result['alternative_CF_centroid_metrics'] = alternatives
    result['alternative_CF_translation_range_cm'] = [min(values), max(values)] if values else None
    result['alternative_range_does_not_choose_headline_reference'] = True
    return result


def prepare(output=OUTPUT, predictions=PREDICTIONS):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Preserve every earlier reference-scoring preparation')
    inventory = C.read(INVENTORY)['rows']  # metadata-only inventory; no annotation decode
    if len(inventory) != 143 or len({r['id'] for r in inventory}) != 143:
        raise ValueError('Exact original143 inventory required')
    observation_protocol = C.read(REMAINDER/'fresh_observations_v4_14/PROTOCOL.json')
    if [r['id'] for r in inventory] != observation_protocol['ids']:
        raise ValueError('Reference membership must match the actual observation population')
    protected = observation_protocol['protected_user_sha256']; _verify(protected)
    sources = {str(Path(path).relative_to(C.ROOT)): C.sha(path) for path in
               (__file__, C.__file__, pose_metric.__code__.co_filename, PG.__file__, SYM.__file__)}
    inputs = {str(INVENTORY.relative_to(C.ROOT)): C.sha(INVENTORY),
              str(SYMMETRY.relative_to(C.ROOT)): C.sha(SYMMETRY)}
    annotations = []
    for row in inventory:
        if row['severity'] not in ('CLEAN', 'MODERATE_OCCLUSION'):
            raise ValueError('Severe data may not enter the declared143')
        path = C.ROOT / row['annotation']
        digest = C.sha(path)  # bytes hashed only; no reference coordinates decoded
        inputs[row['annotation']] = digest
        annotations.append(dict(id=row['id'], annotation=row['annotation'], annotation_sha256=digest,
                                session=row['session'], severity=row['severity']))
    SYM.load_symmetry_contract(SYMMETRY)
    output.mkdir(parents=True)
    C.write(output/'PLAN.json', dict(schema='DERIVED_FINAL143_REFERENCE_SCORE_PLAN_V4_14',
        rows=143, severe=0, ids=[r['id'] for r in inventory], annotations=annotations,
        predictions=str(Path(predictions).resolve().relative_to(C.ROOT)), arm=ARM, reference_policy=POLICY,
        source_sha256=sources, input_sha256=inputs, protected_user_sha256=protected,
        annotation_coordinate_or_pose_values_decoded_at_preparation=False,
        no_reference_used_to_select_prediction_family_or_pose=True,
        all143_translation_rows_even_if_rotation_unresolved=True))
    for relative, digest in sources.items():
        target=output/'code_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(C.ROOT/relative,target)
        if C.sha(target) != digest:raise ValueError('Source snapshot changed')
    C.write(output/'PREPARATION_SEAL.json', dict(plan_sha256=C.sha(output/'PLAN.json'),
        source_sha256=sources, input_sha256=inputs, before_annotation_numeric_decode=True))


def score(output=OUTPUT):
    output = Path(output).resolve(); plan = C.read(output/'PLAN.json')
    preparation = C.read(output/'PREPARATION_SEAL.json')
    if C.sha(output/'PLAN.json') != preparation['plan_sha256']:
        raise ValueError('Declared reference policy changed')
    for key in ('source_sha256','input_sha256','protected_user_sha256'):_verify(plan[key])
    folder = C.ROOT/plan['predictions']; protocol = C.read(folder/'PROTOCOL.json')
    seal = C.read(folder/'PREDICTION_SEAL.json'); journal = folder/'PREDICTIONS.jsonl.gz'
    if (seal['rows'] != 143 or seal['severe'] != 0 or not seal['before_remaining_GT_or_metric_read']
        or C.sha(journal) != seal['predictions_sha256'] or C.sha(folder/'PROTOCOL.json') != seal['protocol_sha256']
        or seal['ordered_ids'] != plan['ids'] or protocol['ids'] != plan['ids']):
        raise ValueError('Complete original143 pose prediction seal required BEFORE annotation decode')
    for key in ('source_sha256','input_sha256','protected_user_sha256'):_verify(protocol[key])
    predictions = list(C.rows(journal))
    if [r['id'] for r in predictions] != plan['ids']:raise ValueError('Keep every original143 reported pose')
    # First and only annotation JSON decode happens below the full prediction gate.
    rows=[]
    for prediction, annotation in zip(predictions, plan['annotations']):
        reference=reference_from_document(C.read(C.ROOT/annotation['annotation']))
        arm=prediction['arms'][plan['arm']]
        metric=metric_for_reference(arm['actual_pose'], reference)
        rows.append(dict(id=prediction['id'],session=annotation['session'],severity=annotation['severity'],
            metric=metric,accepted=arm['accepted'],frozen_family=prediction['frozen_family'],
            reference_audit=reference['audit'],reference_metadata=reference['metadata'],
            target_annotation_binding=dict(path=annotation['annotation'],sha256=annotation['annotation_sha256'])))
    for key in ('source_sha256','input_sha256','protected_user_sha256'):_verify(plan[key])
    eligible=[r for r in rows if r['metric']['rotation_deg'] is not None]
    translations=[r['metric']['translation_cm'] for r in rows]
    summary=dict(rows=143,rotation_eligible_rows=len(eligible),rotation_unresolved_rows=143-len(eligible),
        translation_cm=distribution(translations),rotation_deg=distribution([r['metric']['rotation_deg'] for r in eligible]),
        translation_le5=sum(x <= 5 for x in translations),
        successes_5cm_5deg_on_rotation_eligible=sum(r['metric']['success_5cm_5deg'] for r in eligible),
        fallback_count=sum(not r['accepted'] for r in rows),
        max_translation_id=rows[int(np.argmax(translations))]['id'],
        translation_over5_ids=[r['id'] for r in rows if r['metric']['translation_cm'] > 5],
        exact_candidate_tcopy_and_C2_eligible=sum(r['reference_audit']['eligible'] for r in rows))
    C.write(output/'POSE_REFERENCE_SCORE.json',dict(schema='DERIVED_FINAL143_POSE_REFERENCE_SCORE_V4_14',
        summary=summary,rows=rows,reference_policy=POLICY,prediction_seal_sha256=C.sha(folder/'PREDICTION_SEAL.json'),
        all143_kept=True,independent_physical_metrology_claimed=False,
        caption_KO='저장된 수동 주석 PnP 참조에 대한 오차입니다. 축 부호 미확정·카메라 품질 UNKNOWN이며 독립 실측 GT가 아닙니다.'))
    C.write(output/'SCORE_SEAL.json',dict(prediction_seal_sha256=C.sha(folder/'PREDICTION_SEAL.json'),
        preparation_seal_sha256=C.sha(output/'PREPARATION_SEAL.json'),
        score_sha256=C.sha(output/'POSE_REFERENCE_SCORE.json'),rows=143,all_labels_read_after_complete_pose_prediction_seal=True))
    return summary


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','score'));parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--predictions',type=Path,default=PREDICTIONS);args=parser.parse_args()
    if args.stage=='prepare':prepare(args.output,args.predictions)
    else:print(score(args.output),flush=True)
