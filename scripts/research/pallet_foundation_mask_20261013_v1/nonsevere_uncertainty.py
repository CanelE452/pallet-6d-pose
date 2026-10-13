"""Fresh CAL46 SAM outlines and fixed uncertainty/fusion hypotheses."""
from collections import Counter,defaultdict
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from .foundation import choose_mask
from .uncertainty import intersection_uncertainty,fuse_observations,uncertainty_accuracy

ROOT=Path(__file__).resolve().parents[3]
DOC=ROOT/'_docs/experiments/pallet_visibility_curriculum_20261013_v1'
COHORT=DOC/'nonsevere_v1';OUT=COHORT/'SAM_boundary_uncertainty_v1'
VENDOR=DOC/'foundation_mask_v1/vendor/MobileSAM'
sys.path.insert(0,str(OUT));sys.path.insert(0,str(VENDOR))
from code_snapshot import common as C
from code_snapshot.boundary_refine import infer_rgb_boundary_points,DEFAULT_CONFIG
from code_snapshot.curriculum import solve_curriculum
C.ROOT=ROOT;C.DOC=DOC


@contextmanager
def no_labels_or_old_data():
    import builtins,io
    originals=builtins.open,io.open,Path.open
    def wrap(f):
        def guard(path,*args,**kwargs):
            text=str(path)
            if any(s in text for s in ('SCORING','ORACLE','TARGETS','TRAINING_LABELS','LABEL_AUDIT','SCORED_ROWS','RESULTS.json','/test_reused','FINAL','/annotations/')):
                raise AssertionError('CAL46 inference read forbidden labels/TEST/FINAL: '+text)
            if '/foundation_mask_v1/' in text and '/vendor/MobileSAM/' not in text:
                raise AssertionError('Old study mask/observations reused: '+text)
            return f(path,*args,**kwargs)
        return guard
    builtins.open,io.open,Path.open=map(wrap,originals)
    try:yield
    finally:builtins.open,io.open,Path.open=originals


def prepare():
    assert not (OUT/'PROTOCOL.json').exists()
    frames=C.read(COHORT/'CAL_INPUTS.json')['frames'];assert len(frames)==46 and all(f['partition']=='CAL' for f in frames)
    baseline=COHORT/'OOF_pose_v1/PREDICTIONS.jsonl.gz';seal=C.read(baseline.parent/'PREDICTION_SEAL.json');assert seal['sha256']==C.sha(baseline)
    profile=dict(DEFAULT_CONFIG,minimum_material_inside=.6,maximum_material_outside=.4,minimum_transition=.3,
        maximum_context_outside=.5,minimum_context_inside=.45,transition_probe_px=2.,context_probe_px=5.,
        minimum_inlier_count=5,minimum_inlier_fraction=.4,minimum_inlier_span_fraction=.5,maximum_line_rms_px=1.5,
        maximum_angle_change_deg=15.,minimum_edge_length_px=8.,minimum_visibility=.65,
        normal_search_radius_px=12.,maximum_corner_shift_px=12.)
    controls=[dict(coordinates=c,staged=stage) for c in ('RAW','FULL_INTERSECTION','FUSED_SIGMA1','FUSED_SIGMA2') for stage in (True,False)]
    controls.append(dict(coordinates='FUSED_SIGMA2',staged=True,uncertainty_difficulty=True))
    files=[COHORT/'CAL_INPUTS.json',baseline,baseline.parent/'PREDICTION_SEAL.json',
        COHORT/'visibility_logistic_CAL46_v2.json',COHORT/'clean_source_segmentation_v1/best_source_CAL.pt',
        VENDOR/'weights/mobile_sam.pt']
    code=list((OUT/'code_snapshot').glob('*.py'))+[Path(__file__),Path(__file__).with_name('uncertainty.py')]
    C.write(OUT/'PROTOCOL.json',dict(schema='NONSEVERE_CAL46_SAM_BOUNDARY_UNCERTAINTY_V1',
        before_new_mask_coordinate_pose_inference_and_scoring=True,CAL46_only=True,severe_rows=0,TEST_rows=0,FINAL_rows=0,
        frame_ids=[f['id'] for f in frames],inputs_sha256={str(p):C.sha(p) for p in files},source_sha256={str(p):C.sha(p) for p in code},
        model='official frozen MobileSAM vit_t',model_checkpoint_sha256=C.sha(VENDOR/'weights/mobile_sam.pt'),model_training=False,
        mask_prompt='BOX_ONLY original finite native q8 bounds clipped to original frame; no anchors/labels/GT/projected points',
        mask_selection='Highest uncalibrated SAM predicted-quality among3; exact ties originalindex',fresh_mask_calls=46,old_masks_reused=False,
        baseline_appearance='Only sealed nonsevere FRESH_CAL_VISIBILITY_CONTROL: clean-source426/72 Tiny + fresh CAL46 visibility v2, unchanged gates/weights for coordinate controls',
        baseline_scope='FULLCAL46-trained visibility is in-sample; this is finite CAL development, not new validation',
        profile=profile,coordinate_controls=controls,bootstrap=dict(repetitions=64,moving_block_length=3,seed=20261013,minimum_successful=16,
            estimator='Robust weighted TLS on retained observed-line inliers; same line resamples shared across incident vertices'),
        covariance_floor=dict(normal_offset_std_px=1.,reason='Nonzero native-pixel mask-boundary floor even for exact raster lines; not calibrated model/semantic uncertainty'),
        fusion='alpha=min(.5,assumed_detector_sigma^2/(assumed_detector_sigma^2+largest_intersection_covariance_eigenvalue)); q_fused=q+alpha*(independent_intersection-q)',
        assumed_detector_sigma_px=[1.,2.],no_detector_variance_ground_truth_fit=True,
        difficulty_variant='FUSED_SIGMA2 staged, measured-point accuracy times(.5+.5*variance_reliability), preserve original .10 eligibility; semantic states/visibility unchanged',
        pose='Supplied model/native phase/self-H/Q/initial pose preserved; old solver defaults and original visibility threshold .2; staged versus together only; no W/D branch intervention',
        metrics='All46 including fallbacks; all DIRECT coordinates and regressions; covariance/line support/Jacobian-spread diagnostics; paired4-session bootstrap; labels only afterseal',
        limitations=['Line resampling covariance does not model SAM semantic or rounded/virtual-corner bias',
            'Shared-line corners are correlated; per-KP weighted pose fits do not remove that correlation',
            'Detector sigma and1px floor are prespecified assumptions, not calibrated probabilities',
            'Candidate family and initial H may be wrong; retained rather than repaired here']))
    print('UNCERTAINTY_PREREGISTERED',len(controls),'pose controls',flush=True)


def infer():
    import torch
    from mobile_sam import sam_model_registry,SamPredictor
    p=C.read(OUT/'PROTOCOL.json');assert not (OUT/'PREDICTIONS.jsonl.gz').exists()
    for path,digest in {**p['inputs_sha256'],**p['source_sha256']}.items():assert C.sha(path)==digest
    cv2.setNumThreads(1);torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    model=sam_model_registry['vit_t']();model.load_state_dict(torch.load(VENDOR/'weights/mobile_sam.pt',map_location='cpu',weights_only=True));model=model.to('cuda').eval().requires_grad_(False);predictor=SamPredictor(model)
    rows=[];observations=[];masks=[];start=time.perf_counter()
    with no_labels_or_old_data(),torch.inference_mode():
        frames=C.read(COHORT/'CAL_INPUTS.json')['frames'];assert len(frames)==46 and {f['id'] for f in frames}==set(p['frame_ids'])
        base={r['id']:r for r in C.rows(COHORT/'OOF_pose_v1/PREDICTIONS.jsonl.gz') if r['method']=='FRESH_CAL_VISIBILITY_CONTROL'}
        for index,frame in enumerate(frames):
            initial=frame['initial_poses']['N3_SUBPIX'];original=np.asarray(frame['points']['N3_SUBPIX'],float);q=original[:8];appearance=base[frame['id']]['appearance'];H=np.asarray(base[frame['id']]['self_hidden'],bool)
            xyz=C.cuboid(*initial['cf_extents']);K=np.asarray(frame['K']);R=np.asarray(initial['R_cf']);t=np.asarray(initial['centroid']);Q=R.T@np.asarray(initial['R_physical'])
            image_path=ROOT/frame['image'];assert C.sha(image_path)==frame['image_sha256'];image=cv2.imread(str(image_path));height,width=image.shape[:2]
            valid=np.isfinite(q).all(1)&~np.all(q==-1,axis=1);lo=np.maximum(q[valid].min(0),[0,0]);hi=np.minimum(q[valid].max(0),[width-1,height-1]);box=np.r_[lo,hi]
            predictor.set_image(cv2.cvtColor(image,cv2.COLOR_BGR2RGB));alternatives,quality,_=predictor.predict(box=box,multimask_output=True);selected,choice=choose_mask(alternatives,quality)
            mask_path=OUT/'masks'/f'{index:03d}.png';mask_path.parent.mkdir(exist_ok=True);cv2.imwrite(str(mask_path),selected.astype(np.uint8)*255)
            masks.append(dict(id=frame['id'],path=str(mask_path),sha256=C.sha(mask_path),box=box.tolist(),quality=quality.tolist(),selected_index=choice))
            correction=infer_rgb_boundary_points(image,original,H,appearance,p['profile'],material_mask=selected,mask_source='FRESH_CAL46_FROZEN_OFFICIAL_MOBILESAM_FULLFRAME')
            unc,lines=intersection_uncertainty(correction)
            full=np.asarray(correction['points'],float);f1,m1=fuse_observations(original,correction,unc,1.);f2,m2=fuse_observations(original,correction,unc,2.)
            variants={'RAW':original,'FULL_INTERSECTION':full,'FUSED_SIGMA1':f1,'FUSED_SIGMA2':f2}
            observations.append(dict(id=frame['id'],session=frame['session'],original_points=original.tolist(),correction=correction,
                intersection_uncertainty=unc,line_uncertainty=lines,fusion_SIGMA1=m1,fusion_SIGMA2=m2,
                easiest_measured_ids=sorted(unc,key=lambda k:unc[k]['largest_std_px']),labels_used=False))
            for control in p['coordinate_controls']:
                tag=control['coordinates'];measured=variants[tag];app=uncertainty_accuracy(appearance,unc) if control.get('uncertainty_difficulty') else appearance
                cfg=dict(visible_threshold=.2,staged_visible=control['staged'])
                solution=solve_curriculum(measured[:8],xyz,K,R,t,H,app,cfg)
                if tag=='RAW' and control['staged']:
                    np.testing.assert_array_equal(solution['R'],base[frame['id']]['solution']['R']);np.testing.assert_array_equal(solution['t'],base[frame['id']]['solution']['t'])
                for stage in solution['stages'][1:]:assert not(set(stage['used_ids'])&set(np.flatnonzero(H)))
                assert np.array_equal(measured[H],original[H]) if len(measured)==8 else np.array_equal(measured[:8][H],original[:8][H])
                np.testing.assert_array_equal(measured[8:],original[8:])
                name=tag+('_STAGED' if control['staged'] else '_TOGETHER')+('_UNCERTAINTY_DIFFICULTY' if control.get('uncertainty_difficulty') else '')
                rows.append(dict(id=frame['id'],session=frame['session'],method=name,
                    actual_pose=dict(initial,R_cf=solution['R'],R_physical=(np.asarray(solution['R'])@Q).tolist(),centroid=solution['t']),
                    solution=solution,self_hidden=H.tolist(),appearance=app,observed_points=measured.tolist(),original_points=original.tolist(),
                    geometry_changes=False,independent_mask_measurements_only=True))
            rows.append(dict(id=frame['id'],session=frame['session'],method='INITIAL',actual_pose=initial,solution=dict(accepted=False),original_points=original.tolist(),observed_points=original.tolist()))
            if (index+1)%10==0:print('SAM_UNCERTAINTY_CAL46',index+1,'seconds',round(time.perf_counter()-start,2),flush=True)
    C.save_rows(OUT/'PREDICTIONS.jsonl.gz',rows);C.write(OUT/'OBSERVED_LINE_UNCERTAINTY.json',observations);C.write(OUT/'MASK_BINDINGS.json',masks)
    C.write(OUT/'PREDICTION_SEAL.json',dict(complete=True,before_scoring=True,CAL46_only=True,rows=len(rows),fresh_masks=46,
        predictions_sha256=C.sha(OUT/'PREDICTIONS.jsonl.gz'),observations_sha256=C.sha(OUT/'OBSERVED_LINE_UNCERTAINTY.json'),
        masks_binding_sha256=C.sha(OUT/'MASK_BINDINGS.json'),protocol_sha256=C.sha(OUT/'PROTOCOL.json'),
        old_mask_head_reuse=False,severe_rows=0,TEST_rows=0,FINAL_rows=0,annotation_access_canary='PASS',
        initial_H_Q_model_phase_center_preserved=True,baseline_current_scope_reproduced_exactly=True,seconds=time.perf_counter()-start))
    print('SAM_UNCERTAINTY_SEALED',len(rows),flush=True)


def distribution(values):
    a=np.asarray(values,float);return dict(n=len(a),mean=float(a.mean()) if len(a) else None,median=float(np.median(a)) if len(a) else None,P90=float(np.quantile(a,.9)) if len(a) else None)


def score():
    from code_snapshot.scoring import pose_metric,paired
    seal=C.read(OUT/'PREDICTION_SEAL.json');assert seal['before_scoring'] and seal['predictions_sha256']==C.sha(OUT/'PREDICTIONS.jsonl.gz')
    labels=C.read(COHORT/'OOF_pose_v1/ISOLATED_CAL_SCORING.json')['rows'];assert len(labels)==46 and set(labels)==set(C.read(OUT/'PROTOCOL.json')['frame_ids'])
    rows=list(C.rows(OUT/'PREDICTIONS.jsonl.gz'));groups=defaultdict(list);coordinates=defaultdict(list)
    observations=C.read(OUT/'OBSERVED_LINE_UNCERTAINTY.json')
    for row in rows:
        record=labels[row['id']];meta=record['metadata'];target=record['target'];row['metric']=pose_metric(row['actual_pose'],target['truth']);groups[row['method']].append(row)
        gt=np.asarray(target['gt'],float)[meta['native_to_reference_corner_ids']];q=np.asarray(row['original_points'])[:8];measured=np.asarray(row['observed_points'])[:8]
        for k,state in enumerate(meta['human_states_native']):
            if state!='DIRECT_VISIBLE' or not np.isfinite(gt[k]).all() or np.all(gt[k]==-1):continue
            coordinates[row['method']].append(dict(id=row['id'],corner=k,original_error_px=float(np.linalg.norm(q[k]-gt[k])),
                measured_error_px=float(np.linalg.norm(measured[k]-gt[k])),changed=bool(np.linalg.norm(q[k]-measured[k])>1e-12)))
    summary={};coords={};comparisons={};base={r['id']:r for r in groups['RAW_STAGED']}
    for method,group in groups.items():
        summary[method]=dict(frames=46,accepted=sum(r['solution']['accepted'] for r in group),success=sum(r['metric']['success_5cm_5deg'] for r in group),
            metrics={k:distribution([r['metric'][k] for r in group]) for k in ('translation_cm','rotation_deg')},
            geometry_rejection_counts=dict(Counter(s['reason'] for r in group for s in r['solution'].get('stages',[]) if s.get('geometry',{}).get('usable') is False)))
        c=coordinates[method];changed=[r for r in c if r['changed']]
        coords[method]=dict(all_DIRECT=len(c),changed_DIRECT=len(changed),changed_frames=len({r['id'] for r in changed}),
            all_before=distribution([r['original_error_px'] for r in c]),all_after=distribution([r['measured_error_px'] for r in c]),
            changed_before=distribution([r['original_error_px'] for r in changed]),changed_after=distribution([r['measured_error_px'] for r in changed]),
            improved=sum(r['measured_error_px']<r['original_error_px'] for r in changed),worsened=sum(r['measured_error_px']>r['original_error_px'] for r in changed))
        if method!='RAW_STAGED':comparisons[method]=paired(group,base)
    cov=[d for row in observations for d in row['intersection_uncertainty'].values()]
    uncertainty=dict(measured_corners=len(cov),frames_with_measured_intersection=sum(bool(r['intersection_uncertainty']) for r in observations),
        supported_edges=sum(len(r['line_uncertainty']) for r in observations),largest_std_px=distribution([d['largest_std_px'] for d in cov]),
        resample_count=distribution([d['successful_pair_resamples'] for d in cov]),
        alpha_SIGMA1=distribution([d['alpha'] for r in observations for d in r['fusion_SIGMA1']]),
        alpha_SIGMA2=distribution([d['alpha'] for r in observations for d in r['fusion_SIGMA2']]),
        floor_normal_px=1.,semantic_or_systematic_mask_uncertainty_calibrated=False)
    C.save_rows(OUT/'SCORED_ROWS.jsonl.gz',rows);C.write(OUT/'COORDINATE_AUDIT.json',coordinates)
    C.write(OUT/'RESULTS.json',dict(CAL46_only=True,severe_rows=0,TEST_rows=0,FINAL_rows=0,summary=summary,coordinates=coords,
        paired_vs_RAW_STAGED=comparisons,line_intersection_uncertainty=uncertainty,method_selected=None,
        protocol_sha256=seal['protocol_sha256'],prediction_sha256=seal['predictions_sha256'],limitations=C.read(OUT/'PROTOCOL.json')['limitations']))
    for method,s in summary.items():print(method,s,'COORD',coords[method],flush=True)
    print('UNCERTAINTY',uncertainty,flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('prepare','infer','score'));args=parser.parse_args()
    {'prepare':prepare,'infer':infer,'score':score}[args.phase]()
