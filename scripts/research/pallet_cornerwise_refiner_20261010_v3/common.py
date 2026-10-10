"""Fixed inputs for the one-shot cornerwise correction and scoped evaluation."""
from __future__ import annotations
import builtins
from contextlib import contextmanager
import io
from pathlib import Path
from ..pallet_boundary_corner_refiner_20261010_v2 import common as V

REPO=V.REPO
CODE=Path(__file__).resolve().parent
DOC=REPO/'_docs/experiments/pallet_cornerwise_refiner_20261010_v3'
PRIVATE=Path('/dev/shm/pallet-cornerwise-private-20261010-v3')
OLD=V.OLD
CORRECTED=V.CORRECTED
PRIMARY='N3_CORNERWISE_ROLE'
METHODS=(PRIMARY,'N3_VALIDATED_ROLE','N3_BASIN_ROBUST','N3_BASIN_NO_MASK_ROBUST')
read,rows,sha,binding,bound,finite=V.read,V.rows,V.sha,V.binding,V.bound,V.finite
write_new,save_rows,require=V.write_new,V.save_rows,V.require
legacy_context,cohort_frames,primitive_counter=V.legacy_context,V.cohort_frames,V.primitive_counter

def parser(description,stages=None):
    p=V.parser(description,stages)
    p.set_defaults(output=str(PRIVATE/'accuracy'),protocol=str(DOC/'PROTOCOL.json'),
                   prior_bindings=str(DOC/'PROTECTION_BEFORE.json'))
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
    paths={name:CODE/name for name in ('common.py','selection.py','pipeline.py','run.py','evaluate.py','runtime.py','statistics.py','verify.py','test_selection.py','render.py')}
    paths.update({'v2_'+name:V.CODE/name for name in ('common.py','pipeline.py','deployment.py','pose.py','observations.py')})
    paths.update(cohort=args.cohort,calibration=args.calibration,original_inputs=OLD/'INPUTS.json',
        runtime_panel=CORRECTED/'RUNTIME_PANEL.json',
        training_completion=Path(args.fits)/'TRAINING_COMPLETION.json',ROLE=Path(args.fits)/'IMAGE_ROLE.pt',
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
        evaluation_contract=DOC/'EVALUATION_CONTRACT_KO.md',protection=args.prior_bindings)
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
    require(p['new_training_updates']==0 and p['new_RGB']==0 and p['GT_tuning'] is False,'scope drift')
    from .selection import POLICY
    require(p['selection_policy']==POLICY,'cornerwise policy changed')
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
