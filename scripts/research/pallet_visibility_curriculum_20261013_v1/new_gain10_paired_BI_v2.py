"""Actual paired RAW/N3 image observations, B/I recovery and common BI-only PnP.

This is a new paired diagnostic, not a V33 full-method replay. Each input's
own saved F pose defines its registered geometry/S gate. Both branches run
fresh state/masks/B/I on actual RGB. No old state, mask or E pose is reused.
The parent comparison F poses are inputs only; O/T/U/S never enter BI PnP as
2D correspondences. No reference file or scoring API exists in this module.
"""
from __future__ import annotations
import os
for _name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_name]='1'
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path
import shutil
import time
import traceback

import numpy as np

from . import common as C
from . import statewise_evaluation_v4 as E
from . import statewise_joint_experiment_v2 as EXP
from . import statewise_joint_api_v4 as API
from . import statewise_masks_v2 as MASK
from . import visible_bi_boundary_recovery_v4_3b as RECOVERY
from . import joint_ot_pnp as JOINT
from . import uncertain_u_regions_v4_5 as UREGION
from . import evaluate_u_regions_v4_5 as U
from .state_visibility_v2 import load_state_model
from .truncation_interval_v4_2 import load_truncation_interval_model

OUTPUT=U.BOUNDARY/'new_gain10_paired_BI_v2'
INPUT=U.BOUNDARY/'raw_ns_BI_gain10_input_v1'
INPUTS_SHA256='da7888b529939ea64dd20a78dcd4fc0493943f64cea46ca9f0a77a5f92891212'
INPUT_SEAL_SHA256='785a1238cfe4bbd95c78f1d9f3d06910b03101694e2a88abba26f61b8317e2ac'
BRANCHES=('RAW_YOLO','N3_SUBPIX')
ARMS=('RAW_YOLO_F','N3_SUBPIX_F','RAW_YOLO_BI_ONLY_FRESH_PNP','N3_SUBPIX_BI_ONLY_FRESH_PNP')
SOLVER_CONFIG={**JOINT._DEFAULTS,'analytic_jacobian':True}


def relative(path):return str(Path(path).resolve().relative_to(C.ROOT))


def _require_GO(value):
    if value is not True:raise PermissionError('ROOT actual paired B/I GO before IO')


class CapturingProvider:
    def __init__(self,provider):self.provider=provider;self.last=None
    def __call__(self,*args,**kwargs):
        self.last=self.provider(*args,**kwargs)
        return self.last


def common_BI_pose(observation,input_arm,K,raw_hw):
    """Same original fresh root/least-squares/constrained solver, no OT terms."""
    source=dict(states=list(observation['states']),
        frozen_visible_measurements=deepcopy(observation['frozen_visible_measurements']))
    xyz=np.asarray(observation['initial_geometry']['native_xyz'],float)
    expected=C.cuboid(*input_arm['coarse_pose']['cf_extents'])
    if not np.array_equal(xyz,expected):raise ValueError('Own registered native W/D model changed')
    ids=[m['native_corner'] for m in source['frozen_visible_measurements']]
    if len(ids)!=len(set(ids)) or any(source['states'][k] not in ('B','I') for k in ids):
        raise ValueError('Only unique B/I image observations may enter PnP')
    result=EXP._bi_only(source,dict(initial_pose=input_arm['coarse_pose'],K=K,raw_hw=raw_hw),deepcopy(SOLVER_CONFIG))
    result.update(method='BI_ONLY_FRESH_PNP',input_native_ids=ids,
        observed_states_preserved=list(observation['states']),solver_configuration=deepcopy(SOLVER_CONFIG),
        O_T_U_are_not_PnP_points=True,S_image_terms=0,OT_U_region_pose_constraints=0,
        E_or_cached_pose_bank_starts=0,own_input_F_coarse_start=True,
        full_current_V33_method=False,measurement_coordinates_are_not_final_pose_projection=True)
    result['accepted']=bool(result['solution']['accepted'])
    result['fallback']=not result['accepted']
    return C.finite(result)


def observe_branch(image,frame,branch,context):
    head,provider,region_model,device=context
    source=deepcopy(frame['arms'][branch]);pose=source['coarse_pose']
    provider.last=None
    runtime=API.StatewiseJointV4(head,ot_model=None,mask_provider=provider,
        config=dict(prediction_source=branch,record_id=frame['id'],device=device))
    observed=runtime.observe(image,source['native_points'],frame['K'],
        C.cuboid(*pose['cf_extents']),pose)
    base=deepcopy(observed['visible_correction'])
    masks=provider.last
    if masks is None:
        masks={'pallet_mask':np.zeros(image.shape[:2],np.float32),
               'state_pallet_mask':np.zeros(image.shape[:2],np.float32),
               'provenance':{'reason':'no_gated_RGB_queries_no_provider_invocation'}}
    recovered=RECOVERY.recover_visible_boundary(image,base['raw_points'],frame['K'],pose,
        observed['states'],pallet_mask=masks['pallet_mask'],base_correction=base,
        native_xyz=observed['initial_geometry']['native_xyz'])
    observed['visible_correction']=recovered
    observed['frozen_visible_measurements']=deepcopy(recovered['visible_measurements'])
    regions=UREGION.build_u_regions(observed,region_model)
    observed['u_region_display']=regions
    hidden=observed['initial_geometry']['initial_self_hidden']
    assert all(not r.get('pixel_queried') for k,r in enumerate(recovered['records']) if hidden[k] or observed['states'][k] in ('S','O','T','U'))
    assert all(not r.get('rgb_evaluated') for k,r in enumerate(observed['state_observation']['records']) if hidden[k])
    assert regions['S_coordinate_reads']==0 and regions['image_reads']==0
    result=dict(input=source,observation=observed,base_visible_correction=base,
        masks_provenance=deepcopy(masks.get('provenance',{})),
        state_and_S_recomputed_for_own_input=True,same_frozen_state_model_and_threshold=True,
        raw_state_calibration_coverage_not_validated=True,new_N3_or_YOLO_forwards=0,
        independent_occluder_certificate=False,target_mask_complement_not_an_occluder=True,
        observation_coordinates_not_final_model_projection=True)
    return C.finite(result),masks


def _protocol(device):
    if OUTPUT.exists():raise FileExistsError('Preserve every paired run attempt')
    if C.sha(INPUT/'INPUTS.json')!=INPUTS_SHA256 or C.sha(INPUT/'INPUT_SEAL.json')!=INPUT_SEAL_SHA256:
        raise ValueError('Exact predictor-only paired inputs required')
    data=C.read(INPUT/'INPUTS.json')
    if data['arms']!=list(BRANCHES) or len(data['frames'])!=10 or len(set(data['ordered_ids']))!=10:
        raise ValueError('All fixed ten paired inputs required')
    config,paths,bindings=E._runtime_models()
    bindings[relative(INPUT/'INPUTS.json')]=INPUTS_SHA256
    bindings[relative(INPUT/'INPUT_SEAL.json')]=INPUT_SEAL_SHA256
    bindings[relative(U.REGION_MODEL)]=C.sha(U.REGION_MODEL)
    sources={relative(p):C.sha(p) for p in (Path(__file__),Path(C.__file__),Path(API.__file__),Path(MASK.__file__),Path(E.__file__),Path(EXP.__file__),Path(RECOVERY.__file__),Path(RECOVERY.ORIGINAL.__file__),Path(RECOVERY.LEGACY.__file__),Path(JOINT.__file__),Path(UREGION.__file__))}
    sources.update(EXP.source_bindings())
    for name in ('state_visibility_v4.py','joint_ot_jacobian_v2.py'):
        path=Path(__file__).with_name(name);sources[relative(path)]=C.sha(path)
    for f in data['frames']:
        if C.sha(C.ROOT/f['image'])!=f['image_sha256']:raise ValueError('Original source RGB changed')
        bindings[f['image']]=f['image_sha256']
    OUTPUT.mkdir(parents=True);(OUTPUT/'frames').mkdir();(OUTPUT/'masks').mkdir()
    protocol=dict(schema='ACTUAL_GAIN10_RAW_NS_COMMON_BI_FRESH_PNP_PROTOCOL_V1',
        UTC=datetime.now(timezone.utc).isoformat(),ordered_ids=data['ordered_ids'],rows=10,branches=list(BRANCHES),arms=list(ARMS),
        inputs_sha256=INPUTS_SHA256,input_seal_sha256=INPUT_SEAL_SHA256,source_sha256=sources,input_sha256=bindings,
        device=device,state_masks_each_branch_fresh=True,independent_own_geometry_S_gate=True,
        OT_model=None,OT_model_not_relabelled=True,source_prediction_tags=list(BRANCHES),
        state_model_cross_source_coverage_validated=False,
        recovery='same frozen visible_bi_correction_v2 then visible_bi_boundary_recovery_v4_3b',
        solver='same statewise_joint_experiment_v2._bi_only and joint_ot_pnp defaults with analytic_jacobian=True',
        solver_configuration=SOLVER_CONFIG,solver_fit_only='B/I recovered measurements; S/O/T/U never 2D PnP correspondences',
        native_model='each input own stored F native CF W/D and Q; no channel permutation',
        U='display approximate borrowed CAL region only; no local image queries or pose point',
        no_runtime_pose_or_reference_based_routing=True,case_selection='fixed posthoc gain10 from original sealed TEST49 comparison',reference_or_GT_inputs=False,
        E_or_old_pose_bank_inputs=False,full_V33_method=False,default_method_changed=False)
    C.write(OUTPUT/'PROTOCOL.json',protocol)
    C.write(OUTPUT/'PLAN_SEAL.json',dict(protocol_sha256=C.sha(OUTPUT/'PROTOCOL.json')))
    return data,paths


def observe(*,root_actual_BI_GO=False,device='cuda'):
    _require_GO(root_actual_BI_GO);data,paths=_protocol(device)
    import torch,cv2
    cv2.setNumThreads(1);torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    head=load_state_model(str(paths['state_model']),device)
    provider=CapturingProvider(MASK.StatewiseMaskProvider.from_configs(C.read(paths['mask_descriptor']),paths['boundary_mask_config'],device))
    region_model=load_truncation_interval_model(U.REGION_MODEL)
    start=time.monotonic();records=[]
    for i,f in enumerate(data['frames']):
        image=cv2.imread(str(C.ROOT/f['image']))
        if image is None or list(image.shape[:2])!=f['raw_hw']:raise ValueError('Current image camera mismatch')
        row={k:deepcopy(f[k]) for k in ('id','session','image','image_sha256','raw_hw','K','xyz')};row['branches']={}
        for branch in BRANCHES:
            before=time.monotonic()
            result,masks=observe_branch(image,f,branch,(head,provider,region_model,device))
            path=OUTPUT/'masks'/f'{i:02d}_{branch}.npz'
            np.savez_compressed(path,pallet_mask=masks['pallet_mask'],state_pallet_mask=masks['state_pallet_mask'])
            result.update(mask_binding=dict(path=relative(path),sha256=C.sha(path)),elapsed_seconds=time.monotonic()-before)
            row['branches'][branch]=result
            C.write(OUTPUT/'frames'/f'{i:02d}_{branch}_OBSERVATION.json',dict(id=f['id'],branch=branch,result=result))
            print('OBS',i+1,branch,f['id'],''.join(result['observation']['states']),flush=True)
        C.write(OUTPUT/'frames'/f'{i:02d}_OBSERVATIONS.json',row);records.append(row)
    C.save_rows(OUTPUT/'OBSERVATIONS.jsonl.gz',records)
    protocol=C.read(OUTPUT/'PROTOCOL.json')
    for path,sha in protocol['source_sha256'].items():
        if C.sha(C.ROOT/path)!=sha:raise ValueError('Source changed during actual observation')
    C.write(OUTPUT/'OBSERVATION_SEAL.json',dict(schema='FRESH_PAIRED_GAIN10_BI_OBSERVATION_SEAL_V1',rows=10,branches=20,
        ordered_ids=data['ordered_ids'],observations_sha256=C.sha(OUTPUT/'OBSERVATIONS.jsonl.gz'),
        protocol_sha256=C.sha(OUTPUT/'PROTOCOL.json'),GT_inputs=False,new_YOLO_N3_forwards=0,
        actual_state_and_mask_BI_measurement_runs=20,elapsed_seconds=time.monotonic()-start))
    return OUTPUT


def _solve_row(row):
    for branch in BRANCHES:
        record=row['branches'][branch];start=time.monotonic()
        record['pose_result']=common_BI_pose(record['observation'],record['input'],row['K'],row['raw_hw'])
        record['pose_elapsed_seconds']=time.monotonic()-start
    row['predictions']={}
    for branch in BRANCHES:
        record=row['branches'][branch];result=record['pose_result']
        row['predictions'][branch+'_F']=dict(actual_pose=deepcopy(record['input']['coarse_pose']),source='unchanged own historical input F pose',accepted=True,fallback=False)
        row['predictions'][branch+'_BI_ONLY_FRESH_PNP']=dict(actual_pose=deepcopy(result['actual_pose']),accepted=result['accepted'],fallback=result['fallback'],source='fresh paired B/I-only solver')
    return C.finite(row)


def solve(*,root_actual_BI_GO=False,workers=3):
    _require_GO(root_actual_BI_GO)
    if workers not in (1,2,3):raise ValueError('At most three CPU workers')
    if (OUTPUT/'PREDICTIONS.jsonl.gz').exists():raise FileExistsError('Preserve completed/partial predictions')
    seal=C.read(OUTPUT/'OBSERVATION_SEAL.json');protocol=C.read(OUTPUT/'PROTOCOL.json')
    if C.sha(OUTPUT/'OBSERVATIONS.jsonl.gz')!=seal['observations_sha256'] or C.sha(OUTPUT/'PROTOCOL.json')!=seal['protocol_sha256']:
        raise ValueError('Completed paired actual observations required')
    for path,sha in protocol['source_sha256'].items():
        if C.sha(C.ROOT/path)!=sha:raise ValueError('Actual observation source changed before solver')
    rows=list(C.rows(OUTPUT/'OBSERVATIONS.jsonl.gz'));byid={};start=time.monotonic()
    with ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn')) as pool:
        pending={pool.submit(_solve_row,row):i for i,row in enumerate(rows)}
        for future in as_completed(pending):
            index=pending[future]
            try:row=future.result()
            except Exception:
                (OUTPUT/f'FAILED_SOLVE_{index:02d}.txt').write_text(traceback.format_exc());raise
            byid[row['id']]=row;C.write(OUTPUT/'frames'/f'{index:02d}_PREDICTION.json',row)
            print('PNP',len(byid),'/10',row['id'],{b:row['branches'][b]['pose_result']['accepted'] for b in BRANCHES},flush=True)
    C.save_rows(OUTPUT/'PREDICTIONS.jsonl.gz',[byid[i] for i in protocol['ordered_ids']])
    C.write(OUTPUT/'PREDICTION_SEAL.json',dict(schema='PAIRED_GAIN10_COMMON_BI_FRESH_PNP_PREDICTION_SEAL_V1',
        rows=10,ordered_ids=protocol['ordered_ids'],arms=list(ARMS),predictions_sha256=C.sha(OUTPUT/'PREDICTIONS.jsonl.gz'),
        protocol_sha256=C.sha(OUTPUT/'PROTOCOL.json'),input_seal_sha256=INPUT_SEAL_SHA256,inputs_sha256=INPUTS_SHA256,
        observation_seal_sha256=C.sha(OUTPUT/'OBSERVATION_SEAL.json'),before_scoring=True,GT_inputs=False,
        no_population_drop=True,failed_or_fallback_rows_retained=True,full_V33_method=False,
        elapsed_solver_seconds=time.monotonic()-start,source_sha256=protocol['source_sha256']))
    print('PREDICTION_SEAL',C.sha(OUTPUT/'PREDICTION_SEAL.json'),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('observe','solve'))
    parser.add_argument('--root-actual-BI-GO',action='store_true');parser.add_argument('--device',default='cuda');parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    if args.stage=='observe':observe(root_actual_BI_GO=args.root_actual_BI_GO,device=args.device)
    else:solve(root_actual_BI_GO=args.root_actual_BI_GO,workers=args.workers)
