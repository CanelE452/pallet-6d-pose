"""Freeze one cornerwise policy; capture245 once, seal geometry before scoring."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import time
from . import common as C

def quiet():
    from ..pallet_corner_mechanism_audit_20261010_v1.deployment_smoke import quiet as probe
    value=probe()
    C.require(value['quiet'] and value['temperature_under_80'],'pending competing workload/thermal guard')
    return value

def freeze(args):
    from ..pallet_boundary_corner_refiner_20261010_v2 import protection
    from .selection import POLICY
    from ..pallet_cornerwise_independent_20261010_v4.pose import POLICY as POSE_POLICY
    from ..pallet_partial_line_independent_20261010_v5.solver import POLICY as LINE_POLICY
    from .solver import POLICY as ENDPOINT_POLICY
    C.cohort_frames(args)
    checks=C.read(C.DOC/'ENDPOINT_VALIDATION_CHECKS.json')
    C.require(checks['complete'] and checks['passed'],'synthetic selection checks required')
    for b in checks['code'].values():C.bound(C.REPO/b['path'],b,'tested code')
    extra=[]
    for name in ('pallet_boundary_bootstrap_audit_20261010_v1','pallet_source_neck_contract_20261010_v1'):
        for area in ('scripts/research','_docs/experiments'):
            for path in sorted((C.REPO/area/name).rglob('*')):
                if path.is_file():extra.append(C.binding(path))
    C.write_new(args.prior_bindings,dict(tracked=protection.snapshot(args.source_root),additional_completed_audits=extra))
    C.write_new(args.protocol,dict(schema='one_shot_native_endpoint_heldout_partial_line_protocol_v6',
        methods=list(C.METHODS),primary=C.PRIMARY,frames=245,method_rows=980,fixed_rows=490,
        new_training_updates=0,new_RGB=0,new_seeds=0,GT_tuning=False,
        checkpoint_selection='unchanged corrected last IMAGE_ROLE step3000',selection_policy=POLICY,pose_policy=POSE_POLICY,partial_line_policy=LINE_POLICY,endpoint_validation_policy=ENDPOINT_POLICY,
        self_hidden='initial H excluded from final fit; no initial pose priors in all four numeric arms; new valid R,t projects H once; never refit projections',
        inputs=C.inputs(args),CPU_checks=C.binding(C.DOC/'ENDPOINT_VALIDATION_CHECKS.json'),
        CPU_actual_attempts=checks.get('actual_attempts',1),
        evaluation='one preregistered both-endpoint-heldout validation structural ablation of C2 partial lines on unchanged IMAGE_ROLE observations; no numerical initial pose prior; fresh245 once; no score-based threshold/seed/weight changes; existing geometric proxy DEV',
        timing='separate quiet600 complete fresh API calls:4arms x (20warmup+26x5)',
        performance_success='primary operational245 mean T and R both below fixed seed1 N3_SUBPIX',
        missing_insufficient_ambiguous='record distinct statuses and N3 fallback; no mask-error frame veto',
        hypothesis_reuse='same-coordinate four-subset bank reused across masks; refit calls counted separately',
        limitations=['New numeric validation/final poses exclude H/k influence through initialpose priors; H and Base-derived proposals remain inherited estimator outputs; no independent physical truth.',
                    'Severe74 excluded from new runs by direct cohort instruction; historical319 retained.',
                    'Source controlled variants and independent role-feature replay are not completed by this run.']))
    C.verify_protocol(args);C.protect(args)
    print('CORNERWISE_FROZEN',C.sha(args.protocol),flush=True)

def infer(args):
    import cv2
    import numpy as np
    from .pipeline import Pipeline,preserve_prediction
    C.verify_protocol(args);C.protect(args)
    frames=C.cohort_frames(args)
    names=('INFERENCE_STARTED.json','OBSERVATIONS.jsonl.gz','GEOMETRY_SEALED.jsonl.gz',
           'FIXED_GEOMETRY_SEALED.jsonl.gz','GEOMETRY_SEAL.json','BASE_N3_PARITY.json','INFERENCE_RECEIPT.json')
    for name in names:C.output_path(args,name)
    probes=[]
    def guard(phase,completed=0):
        from ..pallet_corner_mechanism_audit_20261010_v1.deployment_smoke import quiet as probe
        value=dict(phase=phase,completed_frames=completed,**probe())
        C.write_new(C.output_path(args,'RESOURCE_%03d.json'%len(probes)),value)
        probes.append(value)
        C.require(value['quiet'] and value['temperature_under_80'],'pending competing workload/thermal guard; exact probe retained')
    guard('before_models')
    C.write_new(C.output_path(args,names[0]),dict(protocol=C.binding(args.protocol),
        configured_frames=245,configured_rows=980,configured_fixed_rows=490,no_automatic_retry=True))
    geometry=[];controls=[];observations=[];parity=[];banks=[];calls={};model_calls={};cv_calls={}
    start=time.monotonic();pipeline=None;complete=False;primitive=None;head_hook=None;head_actual=Counter();cleanup_error=None
    try:
        with C.inference_canary():pipeline=Pipeline(args)
        head_hook=pipeline.head.register_forward_pre_hook(lambda *_:head_actual.update(ROLE=1))
        with C.primitive_counter() as primitive:
            for frame in frames:
                path=Path(args.source_root)/frame['image']
                C.require(C.sha(path)==frame['image_sha256'],'original RGB SHA changed')
                image=cv2.imread(str(path),cv2.IMREAD_COLOR)
                C.require(image is not None and list(image.shape[:2])==frame['raw_hw'],'RGB shape/decode differs')
                meta={key:frame[key] for key in ('id','session','object_type')}
                with C.inference_canary(),pipeline.torch.no_grad():
                    captured=pipeline.capture(image,frame['K'],frame['xyz'],meta,need_base_pose=True)
                    preserve_prediction(captured['raw'],captured['prediction'])
                    result,ledger=pipeline.outcomes(captured)
                    fixed=[pipeline.fixed_result(captured,arm) for arm in ('BASE','N3_SUBPIX')]
                # Stored prediction parity is diagnostic after fresh inference;
                # no stored coordinates or pose are handed to selection/PnP.
                idx=captured['raw']['selected_index']
                metadata=None if idx is None else {k:v for k,v in captured['raw']['candidates'][idx].items() if k!='keypoints_xy'}
                C.require(idx==frame['selected_index'] and C.finite(metadata)==C.finite(frame['candidate_metadata']),'detector metadata differs')
                base=pipeline.runtime._point_parity(captured['original_base_points'],frame['points']['BASE'],1e-7,frame['id'])
                n3=pipeline.runtime._point_parity(captured['native_N3_points'],frame['points']['N3_SUBPIX'],1e-7,frame['id'])
                C.require(np.array_equal(captured['native_N3_points'][8],captured['original_base_points'][8],equal_nan=True),'center changed')
                parity.append(dict(id=frame['id'],BASE=base,N3_SUBPIX=n3,metadata_preserved=True,
                    stored_parity_not_prediction_input=True,center_preserved=True))
                geometry.extend(result);controls.extend(fixed);banks.append(dict(id=frame['id'],**ledger))
                observations.append(dict(id=frame['id'],session=frame['session'],GT_input=False,
                    original_base_points=captured['original_base_points'],native_N3_points=captured['native_N3_points'],
                    initial_N3_pose=captured['initial_pose'],predicted_N3_hidden=captured['hidden'],**captured['observation']))
                if len(observations)%32==0:
                    guard('after_frame',len(observations))
                    print('CORNERWISE_GEOMETRY',len(observations),245,flush=True)
            cv_calls=dict(primitive)
        calls=dict(pipeline.counts)
        model_calls=dict(detector=pipeline.models.detector_forwards,N3=pipeline.models.n3_forwards,ROLE=head_actual['ROLE'])
        C.require(len(observations)==245 and len(geometry)==980 and len(controls)==490,'incomplete population')
        C.require(calls['detector_calls']==calls['N3_route_calls']==calls['initial_pose_calls']==245 and
            calls['final_pose_paths']==980 and calls['base_control_pose_calls']==245,'actual path counts differ')
        C.require(model_calls['N3']==245 and model_calls['detector']==246,'actual frozen model counts differ')
        guard('after_geometry',len(observations))
        C.verify_protocol(args);preserved=C.protect(args)
        C.save_rows(C.output_path(args,'OBSERVATIONS.jsonl.gz'),observations)
        C.save_rows(C.output_path(args,'GEOMETRY_SEALED.jsonl.gz'),geometry)
        C.save_rows(C.output_path(args,'FIXED_GEOMETRY_SEALED.jsonl.gz'),controls)
        C.write_new(C.output_path(args,'BASE_N3_PARITY.json'),dict(passed=True,rows=parity,rtol=0,atol=1e-7))
        seal=dict(schema='native_endpoint_heldout_partial_line_geometry_seal_v6',complete=True,GT_read_allowed=False,
            frames=245,rows=980,fixed_rows=490,methods=list(C.METHODS),protocol=C.binding(args.protocol),
            observations=C.binding(Path(args.output)/'OBSERVATIONS.jsonl.gz'),
            geometry=C.binding(Path(args.output)/'GEOMETRY_SEALED.jsonl.gz'),
            fixed_geometry=C.binding(Path(args.output)/'FIXED_GEOMETRY_SEALED.jsonl.gz'),
            parity=C.binding(Path(args.output)/'BASE_N3_PARITY.json'),actual_calls=calls,
            actual_model_forwards=model_calls,actual_OpenCV_entry_calls=cv_calls,banks=banks,
            model_bindings=pipeline.bindings,resource_snapshots=probes,protection=preserved,
            wall_seconds=time.monotonic()-start,runtime_benchmark=False)
        C.write_new(C.output_path(args,'GEOMETRY_SEAL.json'),seal);complete=True
    finally:
        if primitive is not None:cv_calls=dict(primitive)
        if pipeline is not None:
            calls=dict(pipeline.counts)
            model_calls=dict(detector=pipeline.models.detector_forwards,N3=pipeline.models.n3_forwards,ROLE=head_actual['ROLE'])
            if head_hook is not None:head_hook.remove()
            try:pipeline.close()
            except BaseException as error:
                cleanup_error=dict(type=type(error).__name__,message=str(error));complete=False
        if not complete:
            C.save_rows(C.output_path(args,'INTERRUPTED_GEOMETRY.jsonl.gz'),geometry)
            C.save_rows(C.output_path(args,'INTERRUPTED_FIXED_GEOMETRY.jsonl.gz'),controls)
            C.save_rows(C.output_path(args,'INTERRUPTED_OBSERVATIONS.jsonl.gz'),observations)
        C.write_new(C.output_path(args,'INFERENCE_RECEIPT.json'),dict(complete=complete,
            actual_complete_frames=len(observations),method_rows=len(geometry),fixed_rows=len(controls),
            actual_calls=calls,actual_model_forwards=model_calls,actual_OpenCV_entry_calls=cv_calls,
            cleanup_error=cleanup_error,
            new_training_updates=0,new_RGB=0,GT_access_during_inference=False,no_automatic_retry=True,
            wall_seconds=time.monotonic()-start))
    C.require(complete,'inference did not complete; preserved prefix receipt')
    print('CORNERWISE_SEALED',len(geometry),len(controls),'before_GT',flush=True)

def main():
    p=C.parser(__doc__,('freeze','preflight','infer'))
    args=p.parse_args()
    if args.stage=='freeze':freeze(args)
    elif args.stage=='preflight':C.verify_protocol(args);C.protect(args);quiet();print('CORNERWISE_PREFLIGHT_PASS',flush=True)
    else:infer(args)

if __name__=='__main__':main()
