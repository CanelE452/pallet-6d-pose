"""Fixed corrected heads and two point-only observation supplies, before scoring."""
from __future__ import annotations
import builtins
from contextlib import contextmanager
import io
from pathlib import Path
from ..pallet_boundary_corner_refiner_20261010_v2 import common as V

REPO=V.REPO
CODE=Path(__file__).resolve().parent
DOC=REPO/'_docs/experiments/pallet_three_head_observation_20261010_v7'
PRIVATE=Path('/tmp/pallet-three-head-observation-private-20261010-v7')
OLD=V.OLD
CORRECTED=V.CORRECTED
V4_DOC=REPO/'_docs/experiments/pallet_cornerwise_independent_20261010_v4'
ARMS=('GEOMETRY_ONLY','IMAGE_NO_ROLE','IMAGE_ROLE')
CAL_ROOT=DOC/'source_calibration'
PRIMARY='IMAGE_ROLE_BOUNDARY_ONLY'
METHODS=(PRIMARY,'GEOMETRY_ONLY_BOUNDARY_ONLY','IMAGE_NO_ROLE_BOUNDARY_ONLY',
         'GEOMETRY_ONLY_CORNERWISE_HYBRID','IMAGE_NO_ROLE_CORNERWISE_HYBRID',
         'IMAGE_ROLE_CORNERWISE_HYBRID','N3_INDEPENDENT_ROBUST_H','N3_INDEPENDENT_ROBUST_NO_MASK')
METHOD_SPECS={arm+'_'+supply:dict(arm=arm,supply=supply)
              for arm in ARMS for supply in ('BOUNDARY_ONLY','CORNERWISE_HYBRID')}
CONTRASTS=tuple([(method,'N3_SUBPIX') for method in METHOD_SPECS]+
    [(arm+'_CORNERWISE_HYBRID',arm+'_BOUNDARY_ONLY') for arm in ARMS]+
    [(a+'_'+supply,b+'_'+supply) for supply in ('BOUNDARY_ONLY','CORNERWISE_HYBRID')
     for a,b in [('IMAGE_NO_ROLE','GEOMETRY_ONLY'),('IMAGE_ROLE','IMAGE_NO_ROLE')]]+
    [('N3_INDEPENDENT_ROBUST_H','N3_INDEPENDENT_ROBUST_NO_MASK'),
     (PRIMARY,'N3_INDEPENDENT_ROBUST_H'),(PRIMARY,'N3_INDEPENDENT_ROBUST_NO_MASK')])
read,rows,sha,binding,bound,finite=V.read,V.rows,V.sha,V.binding,V.bound,V.finite
write_new,save_rows,require=V.write_new,V.save_rows,V.require
legacy_context,cohort_frames,primitive_counter=V.legacy_context,V.cohort_frames,V.primitive_counter

def validate_calibrations(args,head_arms=ARMS):
    from .calibration import validate_deployment_inputs
    return validate_deployment_inputs(args,head_arms)

def parser(description,stages=None):
    p=V.parser(description,stages)
    p.set_defaults(output=str(PRIVATE/'accuracy'),protocol=str(DOC/'PROTOCOL.json'),
                   prior_bindings=str(DOC/'PROTECTION_BEFORE.json'))
    p.add_argument('--calibration-root',default=str(CAL_ROOT))
    return p

def output_path(args,name,allow_existing=False):
    require(name==Path(name).name and name not in ('','.','..'),'output basename required')
    unexpanded=Path(args.output)
    require(not unexpanded.is_symlink(),'symlink output')
    output=unexpanded.resolve()
    require(output==DOC.resolve() or output.is_relative_to(PRIVATE.resolve()),'output must be exact new DOC or new private subtree')
    require(args.source_root and args.baseline_root,'explicit source and baseline roots required')
    for value in (args.source_root,args.baseline_root,args.fits,
                  '/dev/shm/pallet-observation-private-20261009',
                  '/dev/shm/pallet-boundary-corner-private-20261010-v2'):
        root=Path(value).resolve()
        require(not output.is_relative_to(root) and not root.is_relative_to(output),'output overlaps prior inputs')
    target=output/name
    require(not target.is_symlink() and (allow_existing or not target.exists()),'preserve '+name)
    return target

def inputs(args):
    paths={name:CODE/name for name in ('common.py','pipeline.py','run.py','evaluate.py','runtime.py','statistics.py','verify.py','test_pipeline.py','test_calibration.py','calibration.py','source_calibration_check.py','render.py','public_review.py','reprojection_check.py','archive_evidence.py','restore_archives.py')}
    paths.update({'v4_'+name:REPO/'scripts/research/pallet_cornerwise_independent_20261010_v4'/name for name in ('common.py','selection.py','pose.py')})
    paths.update({'v2_'+name:V.CODE/name for name in ('common.py','pipeline.py','deployment.py','pose.py','observations.py')})
    paths.update(cohort=args.cohort,calibration=args.calibration,original_inputs=OLD/'INPUTS.json',
        runtime_panel=CORRECTED/'RUNTIME_PANEL.json',
        training_completion=Path(args.fits)/'TRAINING_COMPLETION.json',
        checkpoint_metadata=CORRECTED/'CHECKPOINT_METADATA.json',
        source_calibration_protocol=Path(args.calibration_root)/'PROTOCOL.json',
        source_calibration_completion=Path(args.calibration_root)/'COMPLETION.json',
        source_calibration_contract_checks=DOC/'SOURCE_CALIBRATION_CONTRACT_CHECKS.json',
        source_calibration_independent_checks=DOC/'SOURCE_CALIBRATION_CHECKS.json',
        pipeline_checks=DOC/'PIPELINE_CHECKS.json',
        ROLE_calibration_reuse=Path(args.calibration_root)/'IMAGE_ROLE'/'REUSE_RECEIPT.json',
        Base=args.base_weights or Path(args.source_root)/V.BASE_WEIGHT_REL,
        N3=args.N3_weights or Path(args.source_root)/V.N3_WEIGHT_REL,
        original_model=REPO/'scripts/research/pallet_observation_refiner_20261009_v1/model.py',
        original_solver=REPO/'scripts/research/pallet_observation_refiner_20261009_v1/solver.py',
        metric=REPO/'scripts/research/pallet_observation_refiner_20261009_v1/evaluate.py',
        reference_adapter=REPO/'scripts/research/pallet_kp_corrected_supervision_20261010_v1/scoring_resume.py',
        statistics_protocol=V.DOC/'STATISTICS_PROTOCOL.json',visual_protocol=V.DOC/'VISUAL_CASE_PROTOCOL.json',
        bootstrap_draws=REPO/'_docs/experiments/pallet_kp_difficulty_20261010_v1/BOOTSTRAP_SESSION_DRAWS.json.gz',
        borrowed_statistics=REPO/'scripts/research/pallet_kp_corrected_supervision_20261010_v1/statistics.py',
        original_solver_checks=REPO/'scripts/research/pallet_observation_refiner_20261009_v1/test_solver.py',
        scorer_fit_canary=V.CODE/'evaluate.py',
        original_context=REPO/'scripts/research/pallet_kp_repair_runtime_20261010_v1/adapter.py',
        original_common=REPO/'scripts/research/pallet_observation_refiner_20261009_v1/common.py',
        original_inference=REPO/'scripts/research/pallet_observation_refiner_20261009_v1/inference.py',
        original_learned=REPO/'scripts/research/pallet_observation_refiner_20261009_v1/learned_infer.py',
        evaluation_contract=DOC/'EVALUATION_CONTRACT_KO.md',protection=args.prior_bindings,
        v4_control_geometry=V4_DOC/'GEOMETRY_SEALED.jsonl.gz',
        v4_control_protocol=V4_DOC/'PROTOCOL.json')
    for arm in ARMS:
        paths['checkpoint:'+arm]=Path(args.fits)/(arm+'.pt')
        paths['calibration:'+arm]=Path(args.calibration_root)/arm/'CALIBRATION.json'
        paths['calibration_execution:'+arm]=Path(args.calibration_root)/arm/'CALIBRATION_EXECUTION.json'
    source=Path(args.source_root)
    for relative in ('scripts/research/pallet_n3_subpix_20261008_v1/runtime.py',
        'scripts/research/pallet_n3_subpix_20261008_v1/common.py',
        'scripts/research/pallet_training_free_compare_20261007_v1/methods.py',
        'scripts/research/pallet_pose_target_6d_20261006_v1/baseline.py',
        '_docs/experiments/pallet_dim_conditioned_p_v1/CALIBRATION_AND_SELECTION.json',
        '_docs/experiments/pallet_dim_conditioned_p_v1/DIM_NORMALIZATION_LOCK.json'):
        paths['source:'+relative]=source/relative
    baseline=Path(args.baseline_root)
    for relative in ('scripts/research/pallet_n3_completion_v3/square_yolo.py',
        'scripts/research/pallet_dim_conditioned_p_v1/inference.py',
        'scripts/research/pallet_dim_conditioned_p_v1/pose.py',
        'scripts/research/pallet_dim_conditioned_p_v1/refiner.py',
        'scripts/research/pallet_final_ml_contribution_test_v1/generic_point_refiner.py',
        'scripts/research/pallet_line_pose_v1/features.py'):
        paths['baseline:'+relative]=baseline/relative
    return {name:binding(path) for name,path in paths.items()}

def verify_protocol(args):
    p=read(args.protocol)
    require(p['inputs']==inputs(args),'fixed code/input drift')
    require(p['methods']==list(METHODS) and p['primary']==PRIMARY and p['frames']==245,'fixed evaluation scope drift')
    require(p['contrasts']==[list(pair) for pair in CONTRASTS],'fixed head/supply comparisons drift')
    require(p['new_training_updates']==0 and p['new_RGB']==0 and p['GT_tuning'] is False,'scope drift')
    from .pipeline import POLICY
    require(p['supply_policy']==POLICY,'head/supply policy changed')
    from ..pallet_cornerwise_independent_20261010_v4.pose import POLICY as POSE_POLICY
    require(p['pose_policy']==POSE_POLICY,'independent pose policy changed')
    return p

def protect(args):
    from ..pallet_boundary_corner_refiner_20261010_v2.protection import verify
    snapshot=read(args.prior_bindings)
    result=verify(snapshot['tracked'],args.source_root)
    require(result['passed'],'prior tracked/source bytes changed')
    for b in snapshot['additional_completed_audits']:
        bound(REPO/b['path'],b,'completed local audit')
    return result

@contextmanager
def inference_canary():
    """Block scored data/annotations; file hashes are checked before this scope."""
    forbidden=('PREDICTIONS.jsonl','FIXED_PREDICTIONS.jsonl','POSTHOC','METRICS.json',
               'REAL_CORRESPONDENCE','STATIC_VISIBILITY','COCO','annotation',
               'target_cache','target_arrays','READY_PREPARED_TARGETS','GROUND_TRUTH',
               'OBSERVATIONS.jsonl','GEOMETRY_SEALED.jsonl','FIXED_GEOMETRY_SEALED.jsonl')
    saved=[(builtins,'open',builtins.open),(io,'open',io.open)]
    def guarded(fn):
        def call(path,*a,**kw):
            if isinstance(path,(str,Path)):
                require(not any(token.lower() in str(path).lower() for token in forbidden),
                        'inference attempted scored/ground-truth read: '+str(path))
            return fn(path,*a,**kw)
        return call
    for obj,name,fn in saved:setattr(obj,name,guarded(fn))
    try:yield
    finally:
        for obj,name,fn in saved:setattr(obj,name,fn)
