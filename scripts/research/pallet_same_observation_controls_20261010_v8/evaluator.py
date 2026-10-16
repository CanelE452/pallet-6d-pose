"""Score complete V8 same-observation controls with the unchanged V7 scorer.

Truth is first opened after the complete GT-free population, cleanup receipt,
seal and separate independent geometry validation all pass. No inference or
pose fitting is performed here. Parent scored rows are postseal parity inputs.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from pathlib import Path
import time

from . import common as C


OUTPUTS = ('SCORING_STARTED.json', 'SCORING_RECEIPT.json', 'SCORING_PARITY.json',
    'SCORING_FAILURE.json', 'PREDICTIONS.jsonl.gz', 'FIXED_PREDICTIONS.jsonl.gz',
    'PENDING_PREDICTIONS.jsonl.gz', 'PENDING_FIXED_PREDICTIONS.jsonl.gz',
    'INTERRUPTED_PREDICTIONS.jsonl.gz', 'INTERRUPTED_FIXED_PREDICTIONS.jsonl.gz')


def same_byte_binding(path, expected, label):
    C.bound(path, expected, label)


def validation_gate(args, seal):
    folder = Path(args.output)
    protocol_path = folder / 'VALIDATION_PROTOCOL.json'
    receipt_path = folder / 'VALIDATION_CHECKS.json'
    validation, receipt = C.read(protocol_path), C.read(receipt_path)
    C.require(validation['schema'] == 'supplemental_same_sparse_ROLE_scalar_protocol_v8' and
        receipt['schema'] == 'supplemental_same_sparse_ROLE_scalar_checks_v8' and
        receipt['complete'] is True and receipt['passed'] is True,
        'independent postgeometry scalar validation PASS required before truth')
    same_byte_binding(protocol_path, receipt['protocol'], 'scalar protocol')
    C.require(receipt['inputs'] == validation['inputs'], 'scalar receipt/protocol inputs differ')
    for name, path in (
            ('PROTOCOL.json', Path(args.protocol)),
            ('GEOMETRY_SEAL.json', folder / 'GEOMETRY_SEAL.json'),
            ('GEOMETRY_SEALED.jsonl.gz', folder / 'GEOMETRY_SEALED.jsonl.gz'),
            ('CONTROL_LEDGERS.jsonl.gz', folder / 'CONTROL_LEDGERS.jsonl.gz'),
            ('CONTROL_RECEIPT.json', folder / 'CONTROL_RECEIPT.json'),
            ('PARENT_POPULATION_CHECKS.json', folder / 'PARENT_POPULATION_CHECKS.json'),
            ('checker_code', C.CODE / 'validation_checks.py')):
        same_byte_binding(path, validation['inputs'][name], 'scalar input ' + name)
    same_byte_binding(C.CODE / 'validation_checks.py', receipt['checker'], 'successful scalar checker')
    C.require(C.same_binding(validation['inputs']['GEOMETRY_SEALED.jsonl.gz'], seal['geometry']) and
        C.same_binding(validation['inputs']['CONTROL_LEDGERS.jsonl.gz'], seal['ledgers']),
        'validated geometry/ledger is not the scored seal')
    return dict(protocol=C.binding(protocol_path), receipt=C.binding(receipt_path),
                checker=C.binding(C.CODE / 'validation_checks.py'))


def sealed(args):
    protocol = C.verify_protocol(args)
    C.protect(args)
    folder = Path(args.output)
    seal, receipt = C.read(folder / 'GEOMETRY_SEAL.json'), C.read(folder / 'CONTROL_RECEIPT.json')
    C.require(seal['complete'] is True and seal['GT_read_allowed'] is False and
        seal['cleanup_completed_before_seal'] is True and seal['no_scored_or_human_inputs'] is True and
        seal['frames'] == 245 and seal['rows'] == 735 and seal['ledger_rows'] == 245 and
        seal['methods'] == list(C.METHODS) and seal['primary'] == C.PRIMARY,
        'complete three-route pretruth seal required')
    C.require(receipt['complete'] is True and receipt['error'] is None and receipt['cleanup_error'] is None and
        receipt['actual_complete_frames'] == 245 and receipt['GT_access_during_controls'] is False and
        receipt['environment_and_monkeypatch_cleanup_completed'] is True and
        not receipt['source_asset_or_GT_attempted_reads'], 'successful control cleanup required before truth')
    C.require(receipt['method_rows'] == {method: 245 for method in C.METHODS} and
        receipt['actual_counts']['geometry_rows'] == 735 and receipt['actual_counts']['ledger_rows'] == 245,
        'complete output counters required')
    for key, path in (('protocol', Path(args.protocol)),
            ('geometry', folder / 'GEOMETRY_SEALED.jsonl.gz'),
            ('ledgers', folder / 'CONTROL_LEDGERS.jsonl.gz'),
            ('control_receipt', folder / 'CONTROL_RECEIPT.json'),
            ('parent_population_checks', folder / 'PARENT_POPULATION_CHECKS.json')):
        same_byte_binding(path, seal[key], 'sealed ' + key)
    same_byte_binding(Path(args.protocol), receipt['protocol'], 'control receipt protocol')
    for name, expected in (('geometry', 735), ('ledgers', 245)):
        stream = receipt['serialization']['streams'][name]
        C.require(stream['rows'] == expected and stream['published'] is True and
            not stream['close_errors'] and not stream['interruption_preservation_errors'],
            'control stream did not close cleanly: ' + name)
    ids = C.cohort_ids(args)
    cohort = {row['id']: row for row in C.read(args.cohort)['frames']}
    counts, keys = Counter(), set()
    for row in C.rows(folder / 'GEOMETRY_SEALED.jsonl.gz'):
        key = (row['method'], row['id'])
        C.require(row['method'] in C.METHODS and row['id'] in cohort and key not in keys,
                  'unknown/duplicate sealed identity')
        C.require(row['session'] == cohort[row['id']]['session'], 'sealed session differs')
        C.require(not C.TRUTH_FIELDS.intersection(row) and 'evaluation_reference' not in row,
                  'geometry already contains scored/human fields')
        C.require(row['solver']['prior_used'] is False and
            row['solver']['known_dimension_constraint_used'] is False and
            row['reprojections_reused_as_observations'] is False and
            row['missing_sparse_filled_for_numeric_fit'] is False,
            'postseal numeric contract changed')
        if row['new_pose_estimated']:
            C.require(set(row['hidden_initial']).isdisjoint(row['solver']['fit_input_ids']),
                      'masked H entered accepted fit')
        keys.add(key)
        counts[row['method']] += 1
    C.require(counts == {method: 245 for method in C.METHODS} and len(keys) == 735,
              'complete same245 geometry required')
    ledgers = list(C.rows(folder / 'CONTROL_LEDGERS.jsonl.gz'))
    C.require(len(ledgers) == 245 and [row['id'] for row in ledgers] == ids and
        all(row['parent_replay_parity']['passed'] and row['parent_unchanged'] for row in ledgers),
        'complete ordered control ledgers required')
    # Fixed controls are immutable parent geometry, re-scored only after this gate.
    parent = Path(args.parent)
    for name in ('FIXED_GEOMETRY_SEALED.jsonl.gz', 'FIXED_PREDICTIONS.jsonl.gz', 'PREDICTIONS.jsonl.gz'):
        same_byte_binding(parent / name, protocol['inputs']['parent:' + name], 'frozen parent ' + name)
    fixed = list(C.rows(parent / 'FIXED_GEOMETRY_SEALED.jsonl.gz'))
    fixed_keys = {(row['method'], row['id']) for row in fixed}
    C.require(len(fixed) == len(fixed_keys) == 490 and
        fixed_keys == {(method, fid) for method in ('BASE', 'N3_SUBPIX') for fid in ids} and
        not any(C.TRUTH_FIELDS.intersection(row) for row in fixed), 'complete parent fixed geometry required')
    validated = validation_gate(args, seal)
    return seal, receipt, validated, fixed


@contextmanager
def forbid_models_and_fits():
    from ..pallet_boundary_corner_refiner_20261010_v2.evaluate import forbid_fits
    from ..pallet_three_head_observation_20261010_v7.pipeline import Pipeline
    import torch.nn as nn
    from ..pallet_observation_refiner_20261009_v1.model import CorrespondenceHead
    calls = Counter()
    entries = [(Pipeline, '__init__'), (Pipeline, 'capture'), (Pipeline, 'infer_frame'),
               (CorrespondenceHead, '__init__'), (CorrespondenceHead, 'forward'), (nn.Module, '_call_impl')]
    existing = []
    def blocked(*args, **kwargs):
        calls['forbidden_model_or_capture_entries'] += 1
        raise RuntimeError('SAME_OBSERVATION_SCORING_GUARD: model/inference during scoring')
    for owner, name in entries:
        if hasattr(owner, name):
            existing.append((owner, name, getattr(owner, name)))
    try:
        for owner, name, _ in existing:
            setattr(owner, name, blocked)
        with forbid_fits():
            yield calls
    finally:
        for owner, name, function in existing:
            setattr(owner, name, function)


def parity(current, previous, kind):
    import numpy as np
    C.require(current['pose']['available'] == previous['pose']['available'], 'parent pose availability differs')
    differences = {key: abs(float(current['pose'][key]) - float(previous['pose'][key]))
        for key in ('translation_cm', 'rotation_deg', 'ADDsym_m') if current['pose']['available']}
    C.require(all(value <= 1e-7 for value in differences.values()), 'unchanged scorer pose parity differs')
    C.require(current['corner']['branch'] == previous['corner']['branch'], 'parent N3-phase corner branch differs')
    a, b = (np.asarray(item['corner']['observed_errors'], float) for item in (current, previous))
    C.require(a.shape == b.shape and np.allclose(a, b, atol=1e-7, rtol=0), 'unchanged corner score parity differs')
    return dict(id=current['id'], method=current['method'], kind=kind, pose_max_abs=differences,
                corner_max_abs=float(np.max(np.abs(a-b))) if a.size else 0.,
                score_method_label_only_renamed=kind == 'parent_primary')


def annotate(row, controls, target, labels):
    import numpy as np
    fid = row['id']
    baseline = controls[('N3_SUBPIX', fid)]
    perm = target['permutations'][baseline['corner'].get('branch', 0)][:8]
    states = [labels.get((fid, k), 'UNANNOTATED') for k in perm]
    H = set(row.get('hidden_initial', []))
    humanH = {k for k, value in enumerate(states) if value == 'SELF_OCCLUDED'}
    known = {k for k, value in enumerate(states) if value != 'UNANNOTATED'}
    solver = row.get('solver') or {}
    eligible, used = set(solver.get('eligible', range(8))), set(solver.get('used', []))
    applied = row['method'] in C.METHODS and row['method'] != 'ROLE_BOUNDARY_NO_MASK_ROBUST'
    row['mask_audit'] = dict(human_states_native=states, oracle_phase_only_for_audit=True,
        reference_phase_control='N3_SUBPIX', native_reference_permutation=perm, known_ids=sorted(known),
        mask_applied=applied,
        false_excluded_visible=sorted(H & {k for k, value in enumerate(states) if value == 'DIRECT_VISIBLE'}),
        false_retained_self=sorted((humanH-H) & eligible),
        mask_wrong_on_known=bool((H ^ humanH) & known) if applied else None,
        excluded_set_differs_from_human_self_on_known=bool((H ^ humanH) & known),
        remaining_human_direct_visible=len(used & {k for k, value in enumerate(states) if value == 'DIRECT_VISIBLE'}),
        classification_difference_is_pose_failure=False, references_used_for_observation_selection=False)
    gt = np.asarray(target['gt'], float)[perm]
    valid = np.asarray(target['valid'], bool)[perm] & np.isfinite(gt).all(-1) & bool(target['matched'])
    row['baseline_pose'], row['baseline_corner'] = baseline['pose'], baseline['corner']
    row['evaluation_reference'] = dict(reference='existing geometric proxy; physical truth not independently validated',
        phase_control='N3_SUBPIX', native_points_px=gt.tolist(), valid_native_ids=np.flatnonzero(valid).tolist(),
        matched=bool(target['matched']), initial_N3_native_points=baseline['native_points'],
        initial_BASE_native_points=controls[('BASE', fid)]['native_points'], human_states_native=states)
    if row['new_pose_estimated']:
        C.require(set(H).isdisjoint(solver['fit_input_ids']), 'hidden observations entered scored final fit')
    return row


def run(args):
    from ..pallet_three_head_observation_20261010_v7.run import RowWriter
    from ..pallet_three_head_observation_20261010_v7 import common as V7
    seal, receipt, validation, fixed = sealed(args)
    for name in OUTPUTS:
        C.output_path(args, name)
    folder, parent = Path(args.output), Path(args.parent)
    C.write_new(C.output_path(args, 'SCORING_STARTED.json'), dict(protocol=C.binding(args.protocol),
        geometry_seal=C.binding(folder/'GEOMETRY_SEAL.json'), control_receipt=C.binding(folder/'CONTROL_RECEIPT.json'),
        independent_validation=validation, complete_geometry_and_cleanup_before_reference_reads=True,
        configured_method_rows=735, configured_fixed_rows=490, new_model_PnP_GT_selection_calls=0))
    started = time.monotonic()
    writers, count, status_counts, parity_rows = {}, 0, {m: Counter() for m in C.METHODS}, []
    try:
        for name in ('PREDICTIONS.jsonl.gz', 'FIXED_PREDICTIONS.jsonl.gz'):
            writers[name] = RowWriter(C.output_path(args, name),
                interrupted_path=C.output_path(args, 'INTERRUPTED_' + name))
        authority = {row['id']: row for row in C.read(V7.OLD/'INPUTS.json')['frames']}
        ids = C.cohort_ids(args)
        scope = dict(ids=ids, authority=authority, count=245)
        with V7.legacy_context(args) as (old, _, _), forbid_models_and_fits() as forbidden:
            from ..pallet_kp_corrected_supervision_20261010_v1.scoring_resume import references
            from ..pallet_observation_refiner_20261009_v1.evaluate import score
            E, frames, targets, reference = references(old, scope)
            previous_fixed = {(row['method'], row['id']): row for row in C.rows(parent/'FIXED_PREDICTIONS.jsonl.gz')}
            C.require(len(previous_fixed) == 490, 'parent fixed score population differs')
            scored_fixed = [score(E, frames[row['id']], targets[row['id']], row) for row in fixed]
            controls = {(row['method'], row['id']): row for row in scored_fixed}
            for row in scored_fixed:
                parity_rows.append(parity(row, previous_fixed[(row['method'],row['id'])], 'fixed_control'))
            label_path = Path(args.source_root)/('_docs/experiments/pallet_combined_closeout_20261003_v1/'
                'closeout_20261006_v1/visibility_square/STATIC_VISIBILITY_MERGE_AUDIT.json')
            labels = {(row['frame_id'],row['corner_id']):row['category'] for row in old.read(label_path)['rows']
                if row['population'] == 'DEV319' and row['frame_id'] in set(ids)}
            # Only small old score/corner fields are retained; old truth never enters a fit.
            previous_primary = {row['id']: dict(pose=row['pose'], corner=row['corner'])
                for row in C.rows(parent/'PREDICTIONS.jsonl.gz') if row['method'] == C.PARENT_PRIMARY}
            C.require(set(previous_primary) == set(ids), 'parent primary score population differs')
            for row in scored_fixed:
                writers['FIXED_PREDICTIONS.jsonl.gz'].write(annotate(row,controls,targets[row['id']],labels))
            for saved in C.rows(folder/'GEOMETRY_SEALED.jsonl.gz'):
                row = score(E,frames[saved['id']],targets[saved['id']],saved)
                if row['method'] == C.PRIMARY:
                    parity_rows.append(parity(row,previous_primary[row['id']], 'parent_primary'))
                writers['PREDICTIONS.jsonl.gz'].write(annotate(row,controls,targets[row['id']],labels))
                status_counts[row['method']][row['output_status']] += 1
                count += 1
            C.require(not forbidden, 'forbidden inference entries attempted during scoring')
        C.require(count == 735 and len(scored_fixed) == 490 and len(parity_rows) == 735,
                  'postseal score/parity population incomplete')
        C.verify_protocol(args)
        preserved = C.protect(args)
        for writer in writers.values():
            writer.close()
        C.require(not any(writer.close_errors for writer in writers.values()), 'scoring row close/fsync failed')
        for writer in writers.values():
            writer.promote()
        C.write_new(C.output_path(args,'SCORING_PARITY.json'),dict(passed=True,atol=1e-7,rtol=0,
            fixed_rows=490,parent_primary_rows=245,rows=parity_rows,
            old_scores_only_used_after_complete_seal_and_validation=True,old_scores_used_for_geometry=False))
        C.write_new(C.output_path(args,'SCORING_RECEIPT.json'),dict(
            schema='unchanged_proxy_same_observation_scoring_receipt_v8',complete=True,frames=245,
            methods=list(C.METHODS),scored_method_rows=735,fixed_control_rows=490,
            protocol=C.binding(args.protocol),geometry_seal=C.binding(folder/'GEOMETRY_SEAL.json'),
            control_receipt=C.binding(folder/'CONTROL_RECEIPT.json'),independent_geometry_validation=validation,
            predictions=C.binding(folder/'PREDICTIONS.jsonl.gz'),
            fixed_predictions=C.binding(folder/'FIXED_PREDICTIONS.jsonl.gz'),
            parity=C.binding(folder/'SCORING_PARITY.json'),reference=C.finite(reference),
            human_visibility=C.binding(label_path),scorer=C.binding(Path(score.__code__.co_filename)),
            frozen_reference_phase='same V7 fixed N3_SUBPIX phase; existing GEOMETRIC_PROXY',
            GT_access_only_after_complete_seal_cleanup_and_validation=True,
            new_detector_forwards=0,new_N3_forwards=0,new_head_forwards=0,new_pose_fits=0,new_rays=0,
            scoring_fitting_and_model_entries_forbidden=True,forbidden_entries_attempted=0,
            full_original_candidate_witnesses_preserved=True,
            status_counts={method:dict(value) for method,value in status_counts.items()},
            elapsed_seconds=time.monotonic()-started,elapsed_is_deployment_latency=False,
            prior_tracked_and_user_checkout_protection=preserved,automatic_retry=False))
        print('SAME_OBSERVATION_SCORED',count,len(scored_fixed),flush=True)
    except BaseException as error:
        for writer in writers.values():
            writer.preserve_interrupted()
        C.write_new(C.output_path(args,'SCORING_FAILURE.json'),dict(complete=False,
            error=dict(type=type(error).__name__,message=str(error)),scored_prefix_rows=count,
            preserved_partial_outputs=[C.binding(writer.interrupted_path) for writer in writers.values()
                if writer.interrupted_path and writer.interrupted_path.is_file()],
            stream_errors={name:writer.close_errors+writer.preservation_errors for name,writer in writers.items()},
            new_model_PnP_optimizer_training_RGB_calls=0,automatic_retry=False))
        raise


def main():
    args = C.parser(__doc__,('preflight','score')).parse_args()
    if args.stage == 'preflight':
        sealed(args)
        print('SAME_OBSERVATION_SCORING_PREFLIGHT_PASS',flush=True)
    else:
        run(args)


if __name__ == '__main__':
    main()
