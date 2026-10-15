"""Foundation instance-mask CAL pilot; seal predictions before annotation use."""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from .foundation import build_prompts,choose_mask,fixed_support_appearance

ROOT=Path(__file__).resolve().parents[3]
DOC=ROOT/'_docs/experiments/pallet_visibility_curriculum_20261013_v1'
SCRATCH=Path('/dev/shm/pallet-visibility-curriculum-20261013-v1')
OUT=DOC/'foundation_mask_v1'
sys.path.insert(0,str(OUT))
from code_snapshot import common as C
from code_snapshot import appearance as A
from code_snapshot import segmented_appearance as S
from code_snapshot.boundary_refine import infer_rgb_boundary_points,DEFAULT_CONFIG
from code_snapshot.curriculum import solve_curriculum
C.ROOT=ROOT;C.DOC=DOC;C.SCRATCH=SCRATCH


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def inference_only():
    import builtins,io
    originals=builtins.open,io.open,Path.open
    forbidden=('SCORING_METADATA','ORACLE_CAL_INPUTS','TARGETS.json','/annotations/',
               'GEOMETRY_RESOLVED_POSE_GT','AXIS_REVIEW_MANIFEST','/test_v1/','TEST_INPUTS','FINAL')
    def wrap(function):
        def guarded(path,*args,**kwargs):
            if any(k in str(path) for k in forbidden):
                raise AssertionError('Annotation/TEST/FINAL access during inference: '+str(path))
            return function(path,*args,**kwargs)
        return guarded
    builtins.open,io.open,Path.open=map(wrap,originals)
    try:yield
    finally:builtins.open,io.open,Path.open=originals


def prepare():
    assert not (OUT/'PROTOCOL.json').exists()
    vendor=OUT/'vendor/MobileSAM';checkpoint=vendor/'weights/mobile_sam.pt'
    sparse=dict(minimum_material_inside=.60,maximum_material_outside=.40,minimum_transition=.30,
        maximum_context_outside=.50,minimum_context_inside=.45,transition_probe_px=2.,context_probe_px=5.,
        minimum_inlier_count=5,minimum_inlier_fraction=.40,minimum_inlier_span_fraction=.50,
        maximum_line_rms_px=1.5,maximum_angle_change_deg=15.,minimum_edge_length_px=8.)
    strict=dict(DEFAULT_CONFIG,minimum_visibility=.65,normal_search_radius_px=12,maximum_corner_shift_px=6)
    sparse=dict(DEFAULT_CONFIG,**sparse,minimum_visibility=.65,normal_search_radius_px=12,maximum_corner_shift_px=12)
    source_files=list((OUT/'code_snapshot').glob('*.py'))+list(Path(__file__).parent.glob('*.py'))
    inputs=[SCRATCH/'CAL_INPUTS.json',SCRATCH/'cal_segmentation_v2/PREDICTIONS.jsonl.gz']
    protocol=dict(schema='CAL_FROZEN_MOBILESAM_PILOT_V1',CAL_only=True,TEST_accessed=False,FINAL_accessed=False,
        declared_before_foundation_inference_and_new_scoring=True,model_type='vit_t',checkpoint=str(checkpoint),
        checkpoint_sha256=sha(checkpoint),checkpoint_bytes=checkpoint.stat().st_size,
        checkpoint_source='https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/f706ad9c4eb7f219c00d9050e46328518ffb65d2/weights/mobile_sam.pt',
        official_repository='https://github.com/ChaoningZhang/MobileSAM',commit='f706ad9c4eb7f219c00d9050e46328518ffb65d2',
        license='Apache-2.0 root LICENSE; no separate checkpoint license found',license_sha256=sha(vendor/'LICENSE'),
        published_training_prior='Paper describes TinyViT distillation on 1% of SA-1B for 8 epochs with original SAM decoder; not a checkpoint-specific training manifest',
        no_new_training=True,images_uploaded=False,inference_location='local GPU only',
        prompt_policies=['BOX_ONLY','BOX_RGB_ANCHORS'],prompt_box='axis-aligned bounds of original finite independent q8, clipped to frame; center q9 excluded',
        anchor_policy='At most3, original q RGB p>=.85 and initial nonself visible-face membership; .7*q+.3*observed_face_centroid; raw material 5x5 support>=.8; >=2r separation',
        negative_policy='Up to4 box-side centers at2r outside initial box, only inside original frame',
        mask_selection='three SAM alternatives; highest uncalibrated predicted IoU; exact ties smallest original index; no semantic or annotation score',
        full_mask_threshold='frozen SAM model.mask_threshold=0 logits; official predictor boolean output',
        appearance_policy='frozen raw RGB head applied only to original raw descriptor; visibility requires raw p>=.35 AND SAM inward occupancy>=.5; score=min(body,occupancy)',
        head_domain_transfer_avoided=True,pose_observation_source='original q8 or independent mask-transition line intersections; never geometric projections',
        curriculum=dict(visible_threshold=.35),boundary_profiles={'STRICT_R12_S6':strict,'SPARSE_R12_S12':sparse},
        geometry='Supplied initial cf_extents, native channel phase, self_hidden and physical Q frozen for every arm',
        inference_inputs=str(inputs[0]),cached_raw_RGB_and_initial_predictions=str(inputs[1]),
        input_sha256={str(p):sha(p) for p in inputs},code_sha256={str(p):sha(p) for p in source_files},
        labels_opened_only_after_prediction_seal=True,
        evaluation='All CAL76; DIRECT vs EXTERNAL states, all DIRECT coordinate errors incl unchanged and outlier points, all-frame physical pose distributions; no new model selection',
        limitations=['Class-agnostic mask may include occluders or object parts','SAM quality is uncalibrated and does not identify a pallet',
                    'Initial box/H/face prompts may be wrong','Rounded/virtual cuboid corners need not lie on actual object outline',
                    'Historical CAL development is exploratory; no unseen validation claim'])
    C.write(OUT/'PROTOCOL.json',protocol)
    print('PREREGISTERED',OUT,flush=True)


def infer():
    import torch
    from mobile_sam import sam_model_registry,SamPredictor
    p=C.read(OUT/'PROTOCOL.json')
    assert not (OUT/'PREDICTIONS.jsonl.gz').exists()
    assert all(sha(f)==d for f,d in {**p['input_sha256'],**p['code_sha256']}.items())
    assert sha(p['checkpoint'])==p['checkpoint_sha256']
    inputs=C.read(p['inference_inputs']);frames=inputs['frames']
    assert len(frames)==76 and all(f['partition']=='CAL' for f in frames)
    cache={}
    for row in C.rows(p['cached_raw_RGB_and_initial_predictions']):
        if row['method'] in ('INITIAL','RGB_CURRICULUM','SEGMENTED_RGB_CURRICULUM'):
            cache[row['id'],row['method']]=row
    cv2.setNumThreads(1);torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=sam_model_registry['vit_t']()
    model.load_state_dict(torch.load(p['checkpoint'],map_location='cpu',weights_only=True))
    model=model.to(device).eval().requires_grad_(False);predictor=SamPredictor(model)
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    # Synthetic loading/forward check uses no pallet image or annotation.
    smoke=np.zeros((96,128,3),np.uint8);smoke[20:76,25:103]=[50,160,60]
    with torch.inference_mode():
        predictor.set_image(smoke)
        sm,quality,_=predictor.predict(box=np.array([25,20,103,76]),multimask_output=True)
    assert sm.shape==(3,96,128) and np.isfinite(quality).all()
    result=[];prompts=[];timings=[];mask_bindings=[];start=time.perf_counter()
    with inference_only(),torch.inference_mode():
        for index,frame in enumerate(frames):
            initial=frame['initial_poses']['N3_SUBPIX'];raw=cache[frame['id'],'INITIAL']
            image_path=Path(inputs['source_root'])/frame['image'];assert sha(image_path)==frame['image_sha256']
            image=cv2.imread(str(image_path));q=np.asarray(raw['original_points'],float);H=np.asarray(raw['self_hidden'],bool)
            xyz=C.cuboid(*initial['cf_extents']);K=np.asarray(frame['K']);R=np.asarray(initial['R_cf']);t=np.asarray(initial['centroid'])
            Q=R.T@np.asarray(initial['R_physical'])
            raw_mask=A.build_appearance_mask(image,q,H)
            query=build_prompts(image.shape[:2],q,H,raw['appearance'],xyz,R,t,raw_mask)
            assert len(query)==2
            if device=='cuda':torch.cuda.synchronize()
            tic=time.perf_counter();predictor.set_image(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
            if device=='cuda':torch.cuda.synchronize()
            encode=time.perf_counter()-tic
            for policy in query:
                pts=np.asarray(policy['points'],np.float32).reshape(-1,2)
                labels=np.asarray(policy['labels'],int)
                if device=='cuda':torch.cuda.synchronize()
                tic=time.perf_counter()
                alternatives,quality,_=predictor.predict(point_coords=pts if len(pts) else None,
                    point_labels=labels if len(pts) else None,box=np.asarray(policy['box']),multimask_output=True)
                if device=='cuda':torch.cuda.synchronize()
                decode=time.perf_counter()-tic
                selected,choice=choose_mask(alternatives,quality)
                mask_path=OUT/'masks'/policy['policy']/f'{index:03d}.png';mask_path.parent.mkdir(parents=True,exist_ok=True)
                assert cv2.imwrite(str(mask_path),selected.astype(np.uint8)*255)
                mask_bindings.append(dict(id=frame['id'],policy=policy['policy'],path=str(mask_path),sha256=sha(mask_path)))
                rgb_support=[float(np.count_nonzero(a&(raw_mask>0))/max(1,np.count_nonzero(a))) for a in alternatives]
                prompts.append(dict(id=frame['id'],**policy,predicted_mask_quality=quality.tolist(),selected_index=choice,
                    selected_area_px=int(selected.sum()),raw_RGB_material_precision_proxy=rgb_support,
                    selection_uses_proxy=False,mask_path=str(mask_path),model_quality_is_calibrated=False))
                tic=time.perf_counter()
                evidence=S.infer_segmented_appearance(image,q,H,selected.astype(np.float32))
                appearance=fixed_support_appearance(evidence,raw['appearance'])
                appearance_seconds=time.perf_counter()-tic
                methods={policy['policy']+'_RAW':(q,None)}
                for profile,config in p['boundary_profiles'].items():
                    correction=infer_rgb_boundary_points(image,q,H,appearance,config,
                        material_mask=selected.astype(np.uint8)*255,mask_source='FROZEN_OFFICIAL_MOBILESAM')
                    methods[policy['policy']+'_'+profile]=(np.asarray(correction['points'],float)[:8],correction)
                for method,(observed,correction) in methods.items():
                    tic=time.perf_counter();solution=solve_curriculum(observed,xyz,K,R,t,H,appearance,p['curriculum']);solve_seconds=time.perf_counter()-tic
                    assert not set(solution['used_ids'])&set(np.flatnonzero(H))
                    np.testing.assert_array_equal(observed[H],q[H])
                    pose=dict(initial,R_cf=solution['R'],R_physical=(np.asarray(solution['R'])@Q).tolist(),centroid=solution['t'])
                    result.append(dict(id=frame['id'],session=frame['session'],partition='CAL',method=method,
                        policy=policy['policy'],actual_pose=pose,solution=solution,self_hidden=H.tolist(),
                        appearance=appearance,original_points=q.tolist(),observed_points=observed.tolist(),correction=correction))
                    timings.append(dict(id=frame['id'],policy=policy['policy'],method=method,encode_seconds=encode,
                        decode_seconds=decode,appearance_seconds=appearance_seconds,solve_seconds=solve_seconds))
            for baseline_method in ('INITIAL','RGB_CURRICULUM','SEGMENTED_RGB_CURRICULUM'):
                row=deepcopy(cache[frame['id'],baseline_method]);row['method']='BASELINE_'+baseline_method
                row['observed_points']=row['original_points'];row['partition']='CAL';result.append(row)
            if (index+1)%10==0:print('MOBILESAM_CAL',index+1,'/76',round(time.perf_counter()-start,1),flush=True)
    C.save_rows(OUT/'PREDICTIONS.jsonl.gz',result);C.write(OUT/'PROMPTS.json',prompts)
    C.write(OUT/'MASK_BINDINGS.json',mask_bindings);C.save_rows(OUT/'TIMINGS.jsonl.gz',timings)
    C.write(OUT/'PREDICTION_SEAL.json',dict(complete=True,before_scoring=True,rows=len(result),frames=76,
        sha256=sha(OUT/'PREDICTIONS.jsonl.gz'),prompts_sha256=sha(OUT/'PROMPTS.json'),mask_bindings_sha256=sha(OUT/'MASK_BINDINGS.json'),
        protocol_sha256=sha(OUT/'PROTOCOL.json'),inference_annotation_access_canary='PASS',TEST_accessed=False,FINAL_accessed=False,
        runtime=dict(device=device,torch_version=torch.__version__,threads=1,total_seconds=time.perf_counter()-start,
                     gpu_name=torch.cuda.get_device_name() if device=='cuda' else None,
                     peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20 if device=='cuda' else None),
        model_training=False,ground_truth_prompts=False,image_uploads=False))
    print('MASK_POSE_SEALED',len(result),flush=True)


def score():
    from code_snapshot.scoring import pose_metric,distribution,appearance_metrics,paired
    seal=C.read(OUT/'PREDICTION_SEAL.json')
    assert seal['before_scoring'] and seal['sha256']==sha(OUT/'PREDICTIONS.jsonl.gz')
    # Only isolated CAL labels/geometry are read, after mask/pose sealing.
    oracle=C.read(SCRATCH/'ORACLE_CAL_INPUTS.json')['frames']
    assert len(oracle)==76 and all(f['partition']=='CAL' for f in oracle)
    oracle={f['id']:f for f in oracle};meta={k:f['scoring_metadata'] for k,f in oracle.items()}
    rows=list(C.rows(OUT/'PREDICTIONS.jsonl.gz'));groups=defaultdict(list);coordinates=defaultdict(list)
    for row in rows:
        row['metric']=pose_metric(row['actual_pose'],oracle[row['id']]['target']['truth'])
        groups[row['method']].append(row)
        if row.get('correction'):
            gt=np.asarray(oracle[row['id']]['target']['gt'],float)[meta[row['id']]['native_to_reference_corner_ids']]
            original=np.asarray(row['original_points']);observed=np.asarray(row['observed_points'])
            changed=set(row['correction'].get('observed_refined_ids',row['correction'].get('refined_ids',[])))
            for k,state in enumerate(meta[row['id']]['human_states_native']):
                if state!='DIRECT_VISIBLE' or not np.isfinite(gt[k]).all() or (gt[k]==-1).all():continue
                before=float(np.linalg.norm(original[k]-gt[k]));after=float(np.linalg.norm(observed[k]-gt[k]))
                coordinates[row['method']].append(dict(id=row['id'],corner=k,changed=k in changed,before_px=before,after_px=after))
    summary={};state={};coord={};comparison={}
    initial={r['id']:r for r in groups['BASELINE_INITIAL']};source={r['id']:r for r in groups['BASELINE_SEGMENTED_RGB_CURRICULUM']}
    for method,group in groups.items():
        summary[method]=dict(frames=len(group),accepted=sum(r['solution']['accepted'] for r in group),
            metrics={k:distribution([r['metric'][k] for r in group]) for k in ('translation_cm','rotation_deg','ADDsym_cm')},
            success_count=sum(r['metric']['success_5cm_5deg'] for r in group))
        state[method]=appearance_metrics(group,meta)
        if not method.startswith('BASELINE'):
            comparison[method]=dict(vs_initial=paired(group,initial),vs_source_CNN=paired(group,source))
        if method in coordinates:
            entries=coordinates[method];changed=[e for e in entries if e['changed']]
            coord[method]=dict(all_DIRECT_count=len(entries),changed_DIRECT_count=len(changed),
                changed_frames=len({e['id'] for e in changed}),all_before=distribution([e['before_px'] for e in entries]),
                all_after=distribution([e['after_px'] for e in entries]),
                changed_before=distribution([e['before_px'] for e in changed]),changed_after=distribution([e['after_px'] for e in changed]),
                improved=sum(e['after_px']<e['before_px'] for e in changed),worsened=sum(e['after_px']>e['before_px'] for e in changed))
    C.save_rows(OUT/'SCORED_ROWS.jsonl.gz',rows);C.write(OUT/'COORDINATE_AUDIT.json',coordinates)
    C.write(OUT/'RESULTS.json',dict(CAL_only=True,TEST_accessed=False,FINAL_accessed=False,
        sealed_predictions_sha256=seal['sha256'],summary=summary,appearance=state,coordinates=coord,paired_comparisons=comparison,
        limitations=C.read(OUT/'PROTOCOL.json')['limitations'],no_method_selected_on_these_results=True))
    for method,s in summary.items():
        print(method,'accepted',s['accepted'],'T',s['metrics']['translation_cm']['mean'],'R',s['metrics']['rotation_deg']['mean'],'success',s['success_count'],flush=True)
    for method,c in coord.items():print('COORD',method,c,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('prepare','infer','score'));args=parser.parse_args()
    # Source stays out of the environment's import paths and package registry.
    sys.path.insert(0,str(OUT/'vendor/MobileSAM'))
    {'prepare':prepare,'infer':infer,'score':score}[args.command]()
