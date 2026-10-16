"""Resume only posthoc scoring of existing selected geometry after ID repair.

No detector, head, PnP, optimizer, ray or regenerated geometry is executed.
Original Data.real_frames uses cache keys session__stem, while public IDs use
session:stem. Join them by the frozen exact image path and session.
"""
import argparse
import ast
from collections import Counter
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import textwrap
import time
from . import subset_downstream as S

sys.dont_write_bytecode=True


def fixed(args):
    output=Path(args.output)
    return dict(code=S.binding(__file__),original_subset_driver=S.binding(S.__file__),
                original_scope_protocol=S.binding(args.scope_protocol),cohort=S.binding(args.cohort),
                geometry=S.binding(output/'LEARNED_GEOMETRY_SEALED.jsonl.gz'),
                observations=S.binding(output/'LEARNED_OBSERVATIONS.jsonl.gz'),
                started=S.binding(output/'SUBSET_EVALUATE_STARTED.json'),
                point_line_checks=S.binding(output/'POINT_LINE_CHECKS.json'),
                inference_receipt=S.binding(output/'SUBSET_INFER_ADAPTER_RECEIPT.json'),
                original_scoring=S.binding(S.CODE/'learned_evaluate.py'),
                original_metric=S.binding(S.CODE/'evaluate.py'),
                axis_manifest=S.binding(Path(args.source_root)/'data/pallet/results/paper_pose_metric_closure_v1/AXIS_REVIEW_MANIFEST.json'),
                reference_mapping=S.binding(S.DOC/'REFERENCE_ID_MAPPING.json'))


def fit_counts(raw):
    counts=Counter();attempts=Counter();optimizers=0;nfev=0
    for row in raw:
        if row['method']!='IMAGE_ROLE_POINT_LINE':counts.update(row['solver']['operation_counts'])
        else:
            for attempt in row['solver'].get('attempts',[]):
                attempts[attempt.get('reason','unspecified')]+=1
                if 'nfev' in attempt:optimizers+=1;nfev+=attempt['nfev']
                elif attempt.get('reason') in ('ValueError','LinAlgError'):optimizers+=1
    return dict(point_solver_ledger=dict(counts),point_line_attempt_reasons=dict(attempts),
                point_line_starts_checked=sum(attempts.values()),point_line_optimizer_calls=optimizers,
                point_line_scipy_reported_nfev=nfev,optimizer_residual_callback_calls=None,
                optimizer_residual_callback_count_status='NA: in-process counter lost on guarded posthoc failure',
                unit_check_optimizer_calls=2,
                unit_check_count_basis='Passed unchanged point_line.tests invokes solve with two identical starts once; derived from preserved source and PASS output, not a saved primitive counter',
                count_basis='Sum sealed per-solve operation_counts, never cumulative bank snapshots; original real point-line attempts distinguish insufficient-dimension precheck from actual optimizer.')


def prepare(args):
    args.stage='preflight';scope=S.preflight(args);raw=list(S.rows(scope['output']/'LEARNED_GEOMETRY_SEALED.jsonl.gz'))
    S.require(len(raw)==6*scope['count'] and Counter(r['method'] for r in raw)=={m:scope['count'] for m in S.METHODS},'complete selected geometry absent')
    S.require(not (scope['output']/'LEARNED_PREDICTIONS.jsonl.gz').exists(),'preserve scored rows')
    axis_path=Path(args.source_root)/'data/pallet/results/paper_pose_metric_closure_v1/AXIS_REVIEW_MANIFEST.json'
    axis=S.read(axis_path)['frames_list'];by_image={f['image']:fid for fid,f in scope['authority'].items()};mapping=[]
    S.require(len(axis)==len(by_image)==319,'original axis/image authority population differs')
    for row in axis:
        S.require(row['image'] in by_image,'axis image outside frozen authority');fid=by_image[row['image']];frame=scope['authority'][fid]
        S.require(row['session_id']==frame['session'] and row['frame_id']==frame['session']+'__'+Path(frame['image']).stem,'axis session/cache alias differs')
        mapping.append(dict(source_cache_id=row['frame_id'],public_id=fid,session=frame['session'],image=frame['image'],image_sha256=frame['image_sha256'],selected=fid in set(scope['ids'])))
    S.require(len({r['public_id'] for r in mapping})==319 and sum(r['selected'] for r in mapping)==scope['count'],'source/public mapping not bijective')
    S.write_new(S.DOC/'REFERENCE_ID_MAPPING.json',dict(schema='selected_reference_id_mapping_v1',authority_frames=319,selected_frames=scope['count'],rows=mapping,axis_manifest=S.binding(axis_path),frozen_inputs=S.binding(S.OLD/'INPUTS.json'),selection_uses_reference_values=False))
    plan=dict(schema='selected_posthoc_scoring_resume_protocol_v1',status='PREPARED_SCORING_ONLY',fixed_inputs=fixed(args),
              frames=scope['count'],rows=6*scope['count'],methods=list(S.METHODS),
              repair='axis cache IDs session__stem mapped to public IDs by exact frozen image path plus session; selected axis filtered before cached-frame loop',
              original_model_decoder_solver_unchanged=True,new_detector_forwards=0,new_head_forwards=0,
              new_PnP_calls=0,new_optimizer_calls=0,new_geometry_rows=0,
              scoring='unchanged original score and native Base-phase mask audit, all1470 sealed rows',
              source_reference='GEOMETRIC_PROXY; not independently measured physical truth')
    S.write_new(args.resume_protocol,plan)
    S.write_new(scope['output']/'SUBSET_EVALUATE_FAILURE.json',dict(
        schema='selected_geometry_posthoc_guard_failure_v1',error='SUBSET_GUARD: original319 reference authority differs',
        cause='axis.frame_id=session__stem; frozen public id=session:stem; no reference cache frame opened and no scoring completed before rejection',
        geometry_complete=True,geometry_rows=len(raw),scored_rows=0,detector_retry=0,geometry_retry=0,
        original_attempt_scope_protocol=S.binding(args.scope_protocol),geometry=S.binding(scope['output']/'LEARNED_GEOMETRY_SEALED.jsonl.gz'),
        saved_fit_counts=fit_counts(raw),failed_fit_wall_seconds=None,wall_status='NA: monotonic duration was not persisted before guarded exception'))
    S.write_new(S.DOC/'EVALUATION_INTERRUPTION.json',dict(
        schema='selected_evaluation_interruption_v1',error='SUBSET_GUARD: original319 reference authority differs',
        traceback_logical_frames=['subset_downstream.run -> derived original learned_evaluate.run after C.save_rows(geometry)',
                                 'subset_downstream.reference_loader -> derived original Data.real_frames',
                                 'subset_downstream.select_axis -> require(axis.frame_id equals public IDs)'],
        geometry_complete=True,geometry_rows=len(raw),scored_rows_before_failure=0,reference_cached_frames_opened_before_failure=0,
        preserved_attempt=S.binding(scope['output']/'SUBSET_EVALUATE_FAILURE.json'),
        original_subset_driver=S.binding(S.__file__),original_scope_protocol=S.binding(args.scope_protocol),
        resume_protocol=S.binding(args.resume_protocol),new_detector_forwards=0,new_head_forwards=0,new_fit_calls=0))
    print('SCORING_RESUME_FROZEN',S.sha(args.resume_protocol),flush=True)


def references(C,scope):
    from scripts.research.pallet_training_free_compare_20261007_v1 import common as real
    E,baseline_root=real.legacy();selected=set(scope['ids'])
    by_image={f['image']:fid for fid,f in scope['authority'].items()}
    S.require(len(by_image)==319,'frozen image identity is not one-to-one')
    tree=ast.parse(textwrap.dedent(inspect.getsource(E.Data.real_frames)));changes=[]
    class Filter(ast.NodeTransformer):
        def visit_For(self,node):
            node=self.generic_visit(node)
            if isinstance(node.iter,ast.Call) and isinstance(node.iter.func,ast.Name) and node.iter.func.id=='sorted':
                node.iter=ast.Call(ast.Name('_selected_axis',ast.Load()),[node.iter],[]);changes.append('filter selected exact image/session aliases before cached-frame loop')
            return node
    derived=ast.fix_missing_locations(Filter().visit(tree));S.require(len(changes)==1,'immutable frame method shape differs')
    aliases={}
    def selected_axis(axis):
        S.require(len(axis)==319,'original axis population differs')
        for row in axis:
            S.require(row['image'] in by_image,'axis image outside frozen authority')
            fid=by_image[row['image']];frame=scope['authority'][fid]
            S.require(row['session_id']==frame['session'],'axis session differs')
            S.require(row['frame_id']==frame['session']+'__'+Path(frame['image']).stem,'axis cache alias is not exact session__stem')
            S.require(fid not in aliases,'axis public identity duplicated');aliases[fid]=row['frame_id']
        S.require(set(aliases)==set(scope['authority']),'original319 axis/public identity mismatch')
        return [row for row in axis if by_image[row['image']] in selected]
    namespace=dict(E.Data.real_frames.__globals__,_selected_axis=selected_axis)
    exec(compile(derived,str(Path(__file__))+'::selected_original_reference_math','exec'),namespace)
    data=object.__new__(E.Data);data.root=C.ROOT;data.dim=C.ROOT/'data/pallet/results/pallet_dim_conditioned_p_v1'
    frames=namespace['real_frames'](data)
    S.require(len(frames)==scope['count'] and {f['id'] for f in frames}==selected,'selected cached reference IDs differ')
    target_path=C.ROOT/'data/pallet/results/pallet_posefix_replay_diagnosis_v1/TARGETS.json';targets=real.read(target_path)
    baseline_path=baseline_root/'_docs/experiments/pallet_joint_action_handoff_20261006_v1/results/A_REAL_DEV_BASELINES.json';baselines=real.read(baseline_path)
    authority=set(scope['authority']);S.require(baselines['target_sha256']==S.sha(target_path),'targets authority differs')
    S.require(all(len(rr)==319 and {r['id'] for r in rr}==authority for rr in baselines['rows'].values()),'319 baseline authority differs')
    S.require(sum(sum(targets[i]['valid'][:8]) for i in authority)==2499 and sum(targets[i]['matched'] for i in authority)==311,'319 target population differs')
    S.require(sum(len(r['corner']['observed_errors']) for r in baselines['rows']['RAW'])==2445,'319 corner baseline population differs')
    return E,{f['id']:f for f in frames},{fid:targets[fid] for fid in scope['ids']},dict(
        axis_aliases=aliases,authority_metadata_frames=319,actual_cached_reference_frames=scope['count'],excluded_cached_reference_frames=0,
        mapping=S.binding(S.DOC/'REFERENCE_ID_MAPPING.json'),
        actual_scored_frames=scope['count'],targets=S.binding(target_path),baselines=S.binding(baseline_path),
        data_frame_math=S.binding(inspect.getsourcefile(E.Data.real_frames)),
        derived_frame_AST_sha256=hashlib.sha256(ast.dump(derived,include_attributes=False).encode()).hexdigest(),changes=changes)


def run(args):
    args.stage='preflight';scope=S.preflight(args);before=S.prior_snapshot();plan=S.read(args.resume_protocol)
    S.require(plan['fixed_inputs']==fixed(args),'frozen scoring resume inputs changed')
    output=scope['output'];S.require(not (output/'LEARNED_PREDICTIONS.jsonl.gz').exists() and not (output/'SCORING_RESUME_STARTED.json').exists(),'preserve completed/interrupted scoring resume')
    raw=list(S.rows(output/'LEARNED_GEOMETRY_SEALED.jsonl.gz'));selected=set(scope['ids']);count=scope['count']
    S.require(len(raw)==6*count and {r['id'] for r in raw}==selected,'selected geometry population differs')
    S.require(not any(k in r for r in raw for k in ('pose','corner','mask_audit','baseline_pose','baseline_corner')),'geometry seal includes scored truth')
    S.write_new(output/'SCORING_RESUME_STARTED.json',dict(protocol=S.binding(args.resume_protocol),new_fit_calls=0,configured_scoring_rows=len(raw)))
    os.environ['PALLET_SOURCE_ROOT']=str(Path(args.source_root).resolve());os.environ['PALLET_BASELINE_ROOT']=str(Path(args.baseline_root).resolve())
    from scripts.research.pallet_observation_refiner_20261009_v1 import common as C
    C.source_modules();start=time.monotonic()
    E,frames,targets,reference=references(C,scope)
    from scripts.research.pallet_observation_refiner_20261009_v1.evaluate import score
    old={r['id']:r for r in S.rows(S.OLD/'FIXED_CONTROLS.jsonl.gz') if r['method']=='BASE' and r['id'] in selected}
    label_path=C.ROOT/'_docs/experiments/pallet_combined_closeout_20261003_v1/closeout_20261006_v1/visibility_square/STATIC_VISIBILITY_MERGE_AUDIT.json'
    labels={(r['frame_id'],r['corner_id']):r['category'] for r in C.read(label_path)['rows'] if r['population']=='DEV319' and r['frame_id'] in selected}
    scored=[]
    for r in raw:
        fid=r['id'];row=score(E,frames[fid],targets[fid],r);prior=old[fid]
        perm=targets[fid]['permutations'][prior['corner'].get('branch',0)][:8]
        states=[labels.get((fid,k),'UNANNOTATED') for k in perm];H=set(row['hidden_initial']);known={k for k,s in enumerate(states) if s!='UNANNOTATED'}
        humanH={k for k,s in enumerate(states) if s=='SELF_OCCLUDED'}
        row['mask_audit']=dict(human_states_native=states,oracle_phase_only_for_audit=True,
            known_ids=sorted(known),false_excluded_visible=sorted(H&{k for k,s in enumerate(states) if s=='DIRECT_VISIBLE'}),
            false_retained_self=sorted((humanH-H)&set(row['solver']['eligible'])),mask_wrong_on_known=bool((H^humanH)&known),
            remaining_human_direct_visible=len(set(row['solver']['used'])&{k for k,s in enumerate(states) if s=='DIRECT_VISIBLE'}))
        row['baseline_pose']=prior['pose'];row['baseline_corner']=prior['corner'];scored.append(row)
    C.save_rows(output/'LEARNED_PREDICTIONS.jsonl.gz',scored)
    counts=fit_counts(raw)
    S.write_new(output/'LEARNED_POSE_EXECUTION.json',dict(complete=True,rows=len(scored),models=3,point_solver_paths=5*count,point_line_paths=count,
        counts=dict(counts['point_solver_ledger'],point_line_paths=count,point_line_optimizer_starts=counts['point_line_starts_checked'],point_line_nfev=counts['point_line_scipy_reported_nfev']),
        seconds=None,seconds_status='Original full geometry+score wall not persisted on guarded failure; scoring_resume_wall_seconds measured separately',
        scoring_resume_wall_seconds=time.monotonic()-start,initial_Base_pose_reused_from_same_fixed_numeric_inputs=True,GT_canary=True,
        raw_observations=S.binding(output/'LEARNED_OBSERVATIONS.jsonl.gz'),final_sealed=S.binding(output/'LEARNED_GEOMETRY_SEALED.jsonl.gz'),
        partial_original_evaluation_resumed_scoring_only=True,new_geometry_on_resume=0,fit_counts=counts))
    S.require(S.prior_snapshot()==before,'protected prior331 files changed')
    S.write_new(output/'SUBSET_EVALUATE_ADAPTER_RECEIPT.json',dict(
        schema='subset_original_path_supervision_repair_adapter_v1',complete=True,stage='evaluate',
        protocol=S.binding(args.protocol),training_protocol=S.binding(args.protocol),cohort=S.binding(args.cohort),
        completed_training=S.binding(scope['completion_path']),scope_protocol=S.binding(args.scope_protocol),
        driver=S.binding(S.__file__),scoring_resume_code=S.binding(__file__),scoring_resume_protocol=S.binding(args.resume_protocol),
        frames=count,sessions=len({f['session'] for f in scope['cohort']['frames']}),models=list(S.ARMS),methods=list(S.METHODS),
        original319_authority_verified=True,excluded_frames_executed=0,protected_prior_files_preserved=331,
        geometry=S.binding(output/'LEARNED_GEOMETRY_SEALED.jsonl.gz'),predictions=S.binding(output/'LEARNED_PREDICTIONS.jsonl.gz'),
        pose_execution=S.binding(output/'LEARNED_POSE_EXECUTION.json'),reference_scope=reference,
        actual_scoring_rows=len(scored),new_detector_forwards=0,new_head_forwards=0,new_PnP_calls=0,new_optimizer_calls=0,
        primitive_counts_reconstructed=counts,wall_seconds=time.monotonic()-start,runtime_benchmark=False,
        original_guard_failure_preserved=S.binding(output/'SUBSET_EVALUATE_FAILURE.json'),
        geometry_recomputed=False,reference='GEOMETRIC_PROXY; not independent physical truth'))
    print('SUBSET_POSTHOC_SCORING_COMPLETE',len(scored),'new_fits=0',flush=True)


if __name__=='__main__':
    p=S.parser();p._actions[1].choices=('prepare','resume')
    p.add_argument('--resume-protocol',default=str(S.DOC/'SCORING_RESUME_PROTOCOL.json'))
    args=p.parse_args()
    if args.stage=='prepare':prepare(args)
    else:run(args)
