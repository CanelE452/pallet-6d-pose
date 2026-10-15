"""Fixed SAM outlines, frozen original appearance gates and unchanged solver."""
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import time

import cv2
import numpy as np

from .pilot import OUT as BASE,ROOT,SCRATCH,C,sha,inference_only,solve_curriculum,infer_rgb_boundary_points

OUT=BASE/'independent_boundary_v1'


def infer():
    assert not (OUT/'PREDICTIONS.jsonl.gz').exists()
    protocol=C.read(OUT/'PROTOCOL.json');parent=C.read(BASE/'PROTOCOL.json')
    assert protocol['declared_before_new_independent_boundary_inference_and_scoring']
    assert sha(BASE/'MASK_BINDINGS.json')==protocol['mask_bindings_sha256']
    assert sha(BASE/'PREDICTION_SEAL.json')==protocol['foundation_seal_sha256']
    assert sha(parent['cached_raw_RGB_and_initial_predictions'])==protocol['cached_primary_predictions_sha256']
    masks={(r['id'],r['policy']):r for r in C.read(BASE/'MASK_BINDINGS.json')}
    for r in masks.values():assert sha(r['path'])==r['sha256']
    C.write(OUT/'RUNNER_BINDING.json',dict(before_inference=True,runner_sha256=sha(__file__),protocol_sha256=sha(OUT/'PROTOCOL.json')))
    frames=C.read(parent['inference_inputs'])['frames'];assert len(frames)==76 and all(f['partition']=='CAL' for f in frames)
    cache={(r['id'],r['method']):r for r in C.rows(parent['cached_raw_RGB_and_initial_predictions'])
           if r['method'] in ('INITIAL','SEGMENTED_RGB_CURRICULUM')}
    rows=[];times=[];start=time.perf_counter();cv2.setNumThreads(1)
    with inference_only():
        for index,frame in enumerate(frames):
            original=cache[frame['id'],'SEGMENTED_RGB_CURRICULUM'];initial=frame['initial_poses']['N3_SUBPIX']
            image_path=ROOT/frame['image'];assert sha(image_path)==frame['image_sha256'];image=cv2.imread(str(image_path))
            q=np.asarray(original['original_points'],float);H=np.asarray(original['self_hidden'],bool);appearance=original['appearance']
            xyz=C.cuboid(*initial['cf_extents']);K=np.asarray(frame['K']);R=np.asarray(initial['R_cf']);t=np.asarray(initial['centroid']);Q=R.T@np.asarray(initial['R_physical'])
            check=solve_curriculum(q,xyz,K,R,t,H,appearance,protocol['pose_config'])
            np.testing.assert_array_equal(check['R'],original['solution']['R']);np.testing.assert_array_equal(check['t'],original['solution']['t'])
            for policy in protocol['prompt_policies']:
                mask=cv2.imread(masks[frame['id'],policy]['path'],cv2.IMREAD_GRAYSCALE)
                assert mask.shape==image.shape[:2] and np.isin(mask,[0,255]).all()
                for profile,config in protocol['profiles'].items():
                    tick=time.perf_counter()
                    corrected=infer_rgb_boundary_points(image,q,H,appearance,config,
                        material_mask=mask,mask_source='OBSERVED_RGB_MASK_ONLY_OFFICIAL_FROZEN_MOBILESAM')
                    observed=np.asarray(corrected['points'],float)[:8]
                    np.testing.assert_array_equal(observed[H],q[H])
                    solution=solve_curriculum(observed,xyz,K,R,t,H,appearance,protocol['pose_config'])
                    assert not(set(solution['used_ids'])&set(np.flatnonzero(H)))
                    pose=dict(initial,R_cf=solution['R'],R_physical=(np.asarray(solution['R'])@Q).tolist(),centroid=solution['t'])
                    rows.append(dict(id=frame['id'],session=frame['session'],partition='CAL',method=policy+'_'+profile,
                        actual_pose=pose,solution=solution,self_hidden=H.tolist(),appearance=deepcopy(appearance),
                        original_points=q.tolist(),observed_points=observed.tolist(),correction=corrected,
                        coordinate_source='OBSERVED_RGB_MASK_ONLY',mask_path=masks[frame['id'],policy]['path']))
                    times.append(dict(id=frame['id'],method=policy+'_'+profile,seconds=time.perf_counter()-tick))
            for method in ('INITIAL','SEGMENTED_RGB_CURRICULUM'):
                row=deepcopy(cache[frame['id'],method]);row['method']='BASELINE_'+method;row['observed_points']=row['original_points'];rows.append(row)
            if (index+1)%10==0:print('INDEPENDENT_SAM_BOUNDARY',index+1,'/76',round(time.perf_counter()-start,2),flush=True)
    C.save_rows(OUT/'PREDICTIONS.jsonl.gz',rows);C.save_rows(OUT/'TIMINGS.jsonl.gz',times)
    C.write(OUT/'PREDICTION_SEAL.json',dict(complete=True,before_scoring=True,rows=len(rows),frames=76,
        sha256=sha(OUT/'PREDICTIONS.jsonl.gz'),runner_binding_sha256=sha(OUT/'RUNNER_BINDING.json'),
        inference_annotation_access_canary='PASS',TEST_accessed=False,FINAL_accessed=False,
        GPU_inference_calls=0,new_mask_selection=False,unchanged_original_baseline_replay='EXACT',seconds=time.perf_counter()-start))
    print('INDEPENDENT_BOUNDARY_SEALED',len(rows),flush=True)


def score():
    from code_snapshot.scoring import pose_metric,distribution,paired
    seal=C.read(OUT/'PREDICTION_SEAL.json');assert seal['before_scoring'] and seal['sha256']==sha(OUT/'PREDICTIONS.jsonl.gz')
    oracle=C.read(SCRATCH/'ORACLE_CAL_INPUTS.json')['frames'];assert len(oracle)==76 and all(f['partition']=='CAL' for f in oracle)
    oracle={f['id']:f for f in oracle};rows=list(C.rows(OUT/'PREDICTIONS.jsonl.gz'));groups=defaultdict(list);coord=defaultdict(list);changes=defaultdict(list)
    for row in rows:
        row['metric']=pose_metric(row['actual_pose'],oracle[row['id']]['target']['truth']);groups[row['method']].append(row)
        if not row.get('correction'):continue
        gt=oracle[row['id']];meta=gt['scoring_metadata'];xy=np.asarray(gt['target']['gt'])[meta['native_to_reference_corner_ids']]
        q=np.asarray(row['original_points']);observed=np.asarray(row['observed_points']);refined=set(row['correction']['observed_refined_ids'])
        for k,state in enumerate(meta['human_states_native']):
            if not np.isfinite(xy[k]).all() or (xy[k]==-1).all():continue
            entry=dict(id=row['id'],corner=k,state=state,before_px=float(np.linalg.norm(q[k]-xy[k])),after_px=float(np.linalg.norm(observed[k]-xy[k])),changed=k in refined)
            if state=='DIRECT_VISIBLE':coord[row['method']].append(entry)
            if k in refined:changes[row['method']].append(entry)
    source={r['id']:r for r in groups['BASELINE_SEGMENTED_RGB_CURRICULUM']};summary={};coordinates={};comparisons={}
    for method,group in groups.items():
        summary[method]=dict(frames=len(group),accepted=sum(r['solution']['accepted'] for r in group),
            success_count=sum(r['metric']['success_5cm_5deg'] for r in group),
            metrics={key:distribution([r['metric'][key] for r in group]) for key in ('translation_cm','rotation_deg','ADDsym_cm')})
        if not method.startswith('BASELINE'):comparisons[method]=paired(group,source)
        if method in coord:
            all_direct=coord[method];changed=[r for r in changes[method] if r['state']=='DIRECT_VISIBLE']
            coordinates[method]=dict(all_DIRECT_count=len(all_direct),changed_DIRECT_count=len(changed),
                changed_nonDIRECT_count=sum(r['state']!='DIRECT_VISIBLE' for r in changes[method]),
                changed_frames=len({r['id'] for r in changes[method]}),
                all_before=distribution([r['before_px'] for r in all_direct]),all_after=distribution([r['after_px'] for r in all_direct]),
                changed_before=distribution([r['before_px'] for r in changed]),changed_after=distribution([r['after_px'] for r in changed]),
                improved=sum(r['after_px']<r['before_px'] for r in changed),worsened=sum(r['after_px']>r['before_px'] for r in changed))
    C.save_rows(OUT/'SCORED_ROWS.jsonl.gz',rows);C.write(OUT/'COORDINATE_AUDIT.json',dict(all_DIRECT=coord,all_changed=changes))
    C.write(OUT/'RESULTS.json',dict(CAL_only=True,TEST_accessed=False,FINAL_accessed=False,summary=summary,
        coordinates=coordinates,paired_vs_frozen_source=comparisons,selected_method=None,
        limitations=C.read(OUT/'PROTOCOL.json')['limits'],inference_seal_sha256=sha(OUT/'PREDICTION_SEAL.json')))
    for method,s in summary.items():print(method,'accepted',s['accepted'],'T',s['metrics']['translation_cm']['mean'],'R',s['metrics']['rotation_deg']['mean'],'success',s['success_count'],flush=True)
    for method,c in coordinates.items():print('COORD',method,c,flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('infer','score'));args=parser.parse_args()
    {'infer':infer,'score':score}[args.command]()
