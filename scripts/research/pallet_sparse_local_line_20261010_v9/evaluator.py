"""Postseal scoring of one local C2 arm with the unchanged N3-phase scorer.

The local geometry and cleanup are complete and independently validated
before source truth is opened. Scoring executes no model, point/line solver,
local optimizer, initialization, calibration or new observation construction.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from pathlib import Path
import time

from . import common as C
from ..pallet_same_observation_controls_20261010_v8 import evaluator as V8

STREAMS = ('PREDICTIONS.jsonl.gz','COMPARATOR_PREDICTIONS.jsonl.gz','FIXED_PREDICTIONS.jsonl.gz')
OUTPUTS = ('SCORING_STARTED.json','SCORING_RECEIPT.json','SCORING_PARITY.json','SCORING_FAILURE.json',
    *STREAMS,*('PENDING_'+n for n in STREAMS),*('INTERRUPTED_'+n for n in STREAMS))


def validation_gate(args,seal):
    folder = Path(args.output)
    path,receipt_path = folder/'VALIDATION_PROTOCOL.json',folder/'VALIDATION_CHECKS.json'
    own,receipt = C.read(path),C.read(receipt_path)
    C.require(own['schema'] == 'supplemental_sparse_local_point_line_scalar_protocol_v9' and
        receipt['schema'] == 'supplemental_sparse_local_point_line_scalar_checks_v9' and
        receipt['complete'] is True and receipt['passed'] is True,
        'independent local actual-factor scalar validation PASS required')
    C.bound(path,receipt['own_protocol'],'local own validation protocol')
    C.require(receipt['inputs'] == own['inputs'],'local validation receipt inputs differ')
    paths = dict(geometry=folder/'GEOMETRY_SEALED.jsonl.gz',ledgers=folder/'CONTROL_LEDGERS.jsonl.gz',
        accuracy_protocol=Path(args.protocol),geometry_seal=folder/'GEOMETRY_SEAL.json',
        control_receipt=folder/'CONTROL_RECEIPT.json',parent_population_checks=folder/'PARENT_POPULATION_CHECKS.json',
        checker=C.CODE/'validation_checks.py')
    for name,source in paths.items():
        C.bound(source,own['inputs'][name],'validated local input '+name)
    C.bound(paths['checker'],own['checker'],'local validation frozen checker')
    C.bound(paths['checker'],receipt['checker'],'local validation successful checker')
    C.require(C.same_binding(own['inputs']['geometry'],seal['geometry']) and
        C.same_binding(own['inputs']['ledgers'],seal['ledgers']),'validation refers to another local population')
    return dict(protocol=C.binding(path),receipt=C.binding(receipt_path),checker=C.binding(paths['checker']))


def sealed(args):
    protocol = C.verify_protocol(args)
    C.protect(args)
    folder = Path(args.output)
    seal,receipt = C.read(folder/'GEOMETRY_SEAL.json'),C.read(folder/'CONTROL_RECEIPT.json')
    C.require(seal['schema'] == 'fixed_sparse_local_line_geometry_seal_v9' and seal['complete'] is True and
        seal['GT_read_allowed'] is False and seal['cleanup_completed_before_seal'] is True and
        seal['no_scored_or_human_inputs'] is True and seal['frames'] == 245 and seal['rows'] == 245 and
        seal['ledger_rows'] == 245 and seal['methods'] == list(C.METHODS) and seal['primary'] == C.PRIMARY,
        'complete GT-free local geometry and cleanup seal required')
    C.require(receipt['schema'] == 'fixed_sparse_local_line_control_receipt_v9' and receipt['complete'] is True and
        receipt['error'] is None and receipt['cleanup_error'] is None and receipt['actual_complete_frames'] == 245 and
        receipt['GT_access_during_controls'] is False and receipt['environment_and_monkeypatch_cleanup_completed'] is True and
        not receipt['source_asset_or_GT_attempted_reads'],'successful local cleanup receipt required')
    C.require(receipt['method_rows'] == {C.PRIMARY:245} and receipt['actual_counts']['geometry_rows'] == 245 and
        receipt['actual_counts']['ledger_rows'] == 245,'complete local row counters required')
    for key,path in (('protocol',Path(args.protocol)),('geometry',folder/'GEOMETRY_SEALED.jsonl.gz'),
        ('ledgers',folder/'CONTROL_LEDGERS.jsonl.gz'),('control_receipt',folder/'CONTROL_RECEIPT.json'),
        ('parent_population_checks',folder/'PARENT_POPULATION_CHECKS.json')):
        C.bound(path,seal[key],'sealed local '+key)
    C.bound(Path(args.protocol),receipt['protocol'],'local receipt protocol')
    for key in ('geometry','ledgers'):
        stream = receipt['serialization']['streams'][key]
        C.require(stream['rows'] == 245 and stream['published'] is True and not stream['close_errors'] and
            not stream['interruption_preservation_errors'],'local stream cleanup '+key)
    ids = C.cohort_ids(args)
    cohort = {row['id']:row for row in C.read(args.cohort)['frames']}
    seen = []
    for row in C.rows(folder/'GEOMETRY_SEALED.jsonl.gz'):
        C.require(row['id'] in cohort and row['session'] == cohort[row['id']]['session'] and
            row['method'] == C.PRIMARY,'local sealed identity/session differs')
        C.require(not {'pose','corner','evaluation_reference','mask_audit','GT','ground_truth'}.intersection(row),
            'local geometry already contains scored or human fields')
        solver = row['solver']
        C.require(solver['prior_used'] is False and solver['initial_pose_residual_prior'] is False and
            solver['initial_dimension_prior_used'] is False and solver['known_dimension_constraint_used'] is False and
            solver['global_uniqueness_proven'] is False and row['independent_PnP_or_PnL'] is False and
            row['reprojections_reused_as_observations'] is False and row['missing_sparse_filled_for_numeric_fit'] is False and
            row['native_N3_is_numeric_final_pose_observation'] is False,'local observation/start contract differs')
        if row['new_pose_estimated']:
            C.require(solver['available'] is True and solver['state'] == 'NEW_POSE' and
                solver['local_refinement_estimated'] is True and solver['initial_pose_start_used'] is True and
                solver['inlier_geometry_is_acceptance_gate'] is False and
                set(row['hidden_initial']).isdisjoint(solver['fit_input_ids']), 'accepted numerical local output contract')
            geometry = solver['point_line_geometry']
            C.require(geometry['observed_normal_joint']['numerical_rank'] == 6 and
                geometry['modeled_normal_joint']['numerical_rank'] == 6,'accepted whole actual-factor local rank')
            # Inlier support weakness is a separate recorded diagnosis. Do not
            # retroactively apply a four-point/inlier-rank acceptance condition.
        seen.append(row['id'])
    C.require(seen == ids,'complete ordered245 local geometry required')
    ledgers = list(C.rows(folder/'CONTROL_LEDGERS.jsonl.gz'))
    C.require(len(ledgers) == 245 and [r['id'] for r in ledgers] == ids and
        all(r['parent_unchanged'] and r['residual_prior_used'] is False and
            r['native_N3_coordinates_used_for_numeric_fit'] is False for r in ledgers), 'complete local ledgers required')
    bindings = protocol['inputs']
    for name in ('GEOMETRY_SEALED.jsonl.gz','GEOMETRY_SEAL.json','CONTROL_RECEIPT.json',
        'VALIDATION_PROTOCOL.json','VALIDATION_CHECKS.json','SCORING_RECEIPT.json','SCORING_PARITY.json',
        'PREDICTIONS.jsonl.gz','FIXED_PREDICTIONS.jsonl.gz'):
        C.bound(Path(args.point_parent)/name,bindings['point:'+name],'unchanged point parent '+name)
    for name in ('FIXED_GEOMETRY_SEALED.jsonl.gz','FIXED_PREDICTIONS.jsonl.gz','PREDICTIONS.jsonl.gz'):
        C.bound(Path(args.parent)/name,bindings['parent:'+name],'unchanged V7 parent '+name)
    point = [r for r in C.rows(Path(args.point_parent)/'GEOMETRY_SEALED.jsonl.gz') if r['method'] == C.COMPARATOR]
    fixed = list(C.rows(Path(args.parent)/'FIXED_GEOMETRY_SEALED.jsonl.gz'))
    C.require(len(point) == 245 and [r['id'] for r in point] == ids and
        all(r['head_arm'] == 'IMAGE_ROLE' and r['observation_supply'] == 'BOUNDARY_ONLY' for r in point),
        'same sparse point comparator population')
    C.require(len(fixed) == 490 and len({(r['method'],r['id']) for r in fixed}) == 490 and
        Counter(r['method'] for r in fixed) == {'BASE':245,'N3_SUBPIX':245} and
        all(r['id'] in cohort and r['session'] == cohort[r['id']]['session'] for r in fixed),
        'complete fixed parent geometry')
    validation = validation_gate(args,seal)
    return seal,receipt,validation,point,fixed


@contextmanager
def forbid_models_fits_and_local_optimizers():
    from . import pipeline,solver
    calls = Counter()
    entries = ((pipeline,'solve_local_packet'),(solver,'least_squares'),
        (solver.LocalPointLineBank,'__init__'),(solver.LocalPointLineBank,'solve'),
        (solver.LocalPointLineBank,'_geometry'))
    saved = [(owner,name,getattr(owner,name)) for owner,name in entries]
    def blocked(*a,**kw):
        calls['forbidden_local_entries'] += 1
        raise RuntimeError('LOCAL_POSTSEAL_SCORING_GUARD: local fitting during scoring')
    try:
        for owner,name,_ in saved:
            setattr(owner,name,blocked)
        with V8.forbid_models_and_fits() as models:
            yield calls,models
    finally:
        for owner,name,fn in saved:
            setattr(owner,name,fn)


def run(args):
    from ..pallet_three_head_observation_20261010_v7.run import RowWriter
    from ..pallet_three_head_observation_20261010_v7 import common as V7
    seal,receipt,validation,point,fixed = sealed(args)
    for name in OUTPUTS:
        C.output_path(args,name)
    folder,parent,point_parent = Path(args.output),Path(args.parent),Path(args.point_parent)
    C.write_new(C.output_path(args,'SCORING_STARTED.json'),dict(protocol=C.binding(args.protocol),
        geometry_seal=C.binding(folder/'GEOMETRY_SEAL.json'),control_receipt=C.binding(folder/'CONTROL_RECEIPT.json'),
        independent_validation=validation,complete_geometry_and_cleanup_before_reference_reads=True,
        configured_local_rows=245,configured_comparator_rows=245,configured_fixed_rows=490,
        new_model_PnP_GT_selection_calls=0))
    started = time.monotonic()
    writers,count,parity_rows = {},0,[]
    statuses = {m:Counter() for m in (*C.METHODS,C.COMPARATOR)}
    try:
        for name in STREAMS:
            writers[name] = RowWriter(C.output_path(args,name),interrupted_path=C.output_path(args,'INTERRUPTED_'+name))
        authority = {r['id']:r for r in C.read(V7.OLD/'INPUTS.json')['frames']}
        ids = C.cohort_ids(args)
        scope = dict(ids=ids,authority=authority,count=245)
        with C.legacy_context(args) as (old,_,_),forbid_models_fits_and_local_optimizers() as (forbidden,models):
            from ..pallet_kp_corrected_supervision_20261010_v1.scoring_resume import references
            from ..pallet_observation_refiner_20261009_v1.evaluate import score
            E,frames,targets,reference = references(old,scope)
            previous_fixed = {(r['method'],r['id']):r for r in C.rows(parent/'FIXED_PREDICTIONS.jsonl.gz')}
            previous_point = {r['id']:r for r in C.rows(point_parent/'PREDICTIONS.jsonl.gz') if r['method'] == C.COMPARATOR}
            C.require(len(previous_fixed) == 490 and set(previous_point) == set(ids),'parent saved score populations')
            scored_fixed = [score(E,frames[r['id']],targets[r['id']],r) for r in fixed]
            controls = {(r['method'],r['id']):r for r in scored_fixed}
            for row in scored_fixed:
                parity_rows.append(V8.parity(row,previous_fixed[(row['method'],row['id'])],'fixed_control'))
            label_path = Path(args.source_root)/('_docs/experiments/pallet_combined_closeout_20261003_v1/'
                'closeout_20261006_v1/visibility_square/STATIC_VISIBILITY_MERGE_AUDIT.json')
            labels = {(r['frame_id'],r['corner_id']):r['category'] for r in old.read(label_path)['rows']
                if r['population'] == 'DEV319' and r['frame_id'] in set(ids)}
            for row in scored_fixed:
                writers['FIXED_PREDICTIONS.jsonl.gz'].write(V8.annotate(row,controls,targets[row['id']],labels))
            for saved in point:
                row = score(E,frames[saved['id']],targets[saved['id']],saved)
                parity_rows.append(V8.parity(row,previous_point[row['id']],'same_sparse_point_control'))
                writers['COMPARATOR_PREDICTIONS.jsonl.gz'].write(V8.annotate(row,controls,targets[row['id']],labels))
                statuses[C.COMPARATOR][row['output_status']] += 1
            for saved in C.rows(folder/'GEOMETRY_SEALED.jsonl.gz'):
                row = score(E,frames[saved['id']],targets[saved['id']],saved)
                row = V8.annotate(row,controls,targets[row['id']],labels)
                row['evaluation_numeric_LOCAL_NEW_is_not_pose_accuracy_success'] = True
                row['evaluation_local_global_uniqueness_proven'] = False
                writers['PREDICTIONS.jsonl.gz'].write(row)
                statuses[C.PRIMARY][row['output_status']] += 1
                count += 1
            C.require(not forbidden and not models,'forbidden model/fit/optimizer entry attempted in scoring')
        C.require(count == 245 and len(point) == 245 and len(scored_fixed) == 490 and len(parity_rows) == 735,
            'complete980 score rows and735 existing score parities')
        C.verify_protocol(args)
        protection = C.protect(args)
        for writer in writers.values():
            writer.close()
        C.require(not any(w.close_errors for w in writers.values()),'scoring close/fsync failed')
        for writer in writers.values():
            writer.promote()
        C.write_new(C.output_path(args,'SCORING_PARITY.json'),dict(passed=True,atol=1e-7,rtol=0,
            fixed_rows=490,point_control_rows=245,total_rows=735,rows=parity_rows,
            fixed_authority='V7 FIXED_PREDICTIONS',point_authority='V8 ROLE_BOUNDARY_H_ROBUST PREDICTIONS',
            saved_scores_only_read_after_complete_seal_cleanup_validation=True,old_scores_used_for_geometry=False))
        C.write_new(C.output_path(args,'SCORING_RECEIPT.json'),dict(
            schema='unchanged_proxy_sparse_local_line_scoring_receipt_v9',complete=True,frames=245,
            methods=list(C.METHODS),comparator=C.COMPARATOR,scored_method_rows=245,comparator_rows=245,fixed_control_rows=490,
            actual_total_scored_rows=980,protocol=C.binding(args.protocol),geometry_seal=C.binding(folder/'GEOMETRY_SEAL.json'),
            control_receipt=C.binding(folder/'CONTROL_RECEIPT.json'),independent_geometry_validation=validation,
            predictions=C.binding(folder/'PREDICTIONS.jsonl.gz'),comparator_predictions=C.binding(folder/'COMPARATOR_PREDICTIONS.jsonl.gz'),
            fixed_predictions=C.binding(folder/'FIXED_PREDICTIONS.jsonl.gz'),parity=C.binding(folder/'SCORING_PARITY.json'),
            point_geometry=C.binding(point_parent/'GEOMETRY_SEALED.jsonl.gz'),point_saved_scores=C.binding(point_parent/'PREDICTIONS.jsonl.gz'),
            fixed_saved_scores=C.binding(parent/'FIXED_PREDICTIONS.jsonl.gz'),reference=C.finite(reference),
            human_visibility=C.binding(label_path),scorer=C.binding(Path(score.__code__.co_filename)),
            frozen_reference_phase='same V7/V8 fixed N3_SUBPIX phase; existing GEOMETRIC_PROXY',
            GT_access_only_after_complete_seal_cleanup_and_validation=True,
            new_detector_forwards=0,new_N3_forwards=0,new_head_forwards=0,new_pose_fits=0,new_local_optimizers=0,new_rays=0,
            scoring_fitting_and_model_entries_forbidden=True,local_optimizer_entries_forbidden=True,forbidden_entries_attempted=0,
            full_original_local_candidate_witnesses_preserved=True,numeric_LOCAL_NEW_is_not_accuracy_success=True,
            local_global_uniqueness_proven=False,status_counts={k:dict(v) for k,v in statuses.items()},
            elapsed_seconds=time.monotonic()-started,elapsed_is_deployment_latency=False,
            prior_tracked_and_user_checkout_protection=protection,automatic_retry=False))
        print('SPARSE_LOCAL_SCORED',count,len(point),len(scored_fixed),flush=True)
    except BaseException as error:
        for writer in writers.values():
            writer.preserve_interrupted()
        C.write_new(C.output_path(args,'SCORING_FAILURE.json'),dict(complete=False,
            error=dict(type=type(error).__name__,message=str(error)),scored_local_prefix_rows=count,
            preserved_partial_outputs=[C.binding(w.interrupted_path) for w in writers.values()
                if w.interrupted_path and w.interrupted_path.is_file()],
            stream_errors={name:w.close_errors+w.preservation_errors for name,w in writers.items()},
            new_model_PnP_local_optimizer_training_RGB_calls=0,automatic_retry=False))
        raise


def main():
    args = C.parser(__doc__,('preflight','score')).parse_args()
    if args.stage == 'preflight':
        sealed(args)
        print('SPARSE_LOCAL_SCORING_PREFLIGHT_PASS',flush=True)
    else:
        run(args)


if __name__ == '__main__':
    main()
