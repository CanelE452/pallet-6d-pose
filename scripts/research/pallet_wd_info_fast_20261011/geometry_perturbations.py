"""Locked exact-renderer noise and cached floor/depth sensitivity diagnoses."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import math
from pathlib import Path
import time
import numpy as np

from . import geometry_analysis as A
from . import common as C


def hashed(ids, count):
    import hashlib
    return sorted(set(ids), key=lambda x:(hashlib.sha256(x.encode()).hexdigest(),x))[:count]


def geometry(source, ids):
    with np.load(source / A.GEOMETRY_REL, allow_pickle=False) as archive:
        g = {k:archive[k] for k in archive.files}
    ix={str(x):i for i,x in enumerate(g['stems'])}
    return {fid:{k:v[ix[fid]] for k,v in g.items() if k != 'stems'} for fid in ids}


def projected(X,R,t,K):
    cam=np.vstack([X,np.zeros((1,3))]) @ R.T + t
    q=cam@K.T
    assert np.all(q[:,2]>0)
    return q[:,:2]/q[:,2:]


def exact_truth(r,g):
    K=np.asarray(r['fixed_metadata']['K'],float)
    fx,fy,cx,cy=g['K'];pad=float(g['pad'])
    kg=np.array([[fx,0,cx-pad],[0,fy,cy-pad],[0,0,1.]])
    assert np.allclose(K,kg,rtol=0,atol=1e-8), 'Native / padded intrinsics mismatch'
    xyz=np.asarray(r['fixed_metadata']['dimensions_pnp_WH_D_m'],float)
    assert np.allclose(xyz,g['dims'],rtol=0,atol=1e-8)
    X=g['Xcf']; w=np.linalg.norm(X[1]-X[0]);d=np.linalg.norm(X[4]-X[0])
    assert np.allclose(sorted([w,d]),sorted([xyz[0],xyz[2]]),rtol=0,atol=1e-6)
    h=A.NAMES[0] if w>d else A.NAMES[1]
    assert h==r['hypOracle'] and abs(w-d)>1e-6
    truth=dict(R=g['R'],t=g['t'],xyz=xyz,body_R=g['R'],body_xyz=xyz,
               order=int(r['pose_symmetry_order']))
    return projected(X,g['R'],g['t'],K),truth,h,K


def a3():
    lock=C.lock_ready(); cfg=lock['A3']
    assert cfg['hash_sample_frames']==16 and cfg['draws_per_sigma']==8
    historical={r['id']:r for r in A.rows(C.PUBLIC/A.WD_REL/'STAGE1_ROWS_SYNTH.jsonl.gz')
                if r['seed']==1 and r['method']=='N3_THEN_SUBPIX'}
    ids=hashed(historical,16)
    assert A.digest(ids)=='a3f75e6521c5718d20b07a3de5ff7412fd29acba281804812cb06ff92de745d1'
    gg=geometry(C.SOURCE,ids);pose=A.pose_api(C.SOURCE)
    from scripts.research.pallet_wd_hypothesis_diag_20261010 import solver
    solver.configure(pose);rng=np.random.default_rng(20261011)
    records=[]; projection=[]; start=time.monotonic()
    with C.pnp_counter('A3',6016) as calls:
        for fid in ids:
            r=historical[fid];g=gg[fid];qref,truth,label,K=exact_truth(r,g)
            qpred=np.asarray(r['qFinal'],float);residual=qpred-qref
            assert np.isfinite(residual).all()
            for condition,values in [('correlated_residual',[(x,0) for x in (0,.25,.5,1)]),
                                     ('isotropic_Gaussian',[(s,j) for s in (0,.5,1,2,4) for j in range(8)])]:
                for scale,draw in values:
                    q=qref+scale*residual if condition=='correlated_residual' else qref+rng.normal(0,scale,qref.shape)
                    # centroid8 is kept in the same residual/noise model; no GT remapping.
                    selection=solver.production(q,K,truth['xyz'],True)
                    metric=pose.metric((fid,selection['actual_pose'],truth))
                    records.append(dict(id=fid,condition=condition,scale=scale,draw=draw,
                        selected=selection['hyp'],reference_parity=label,
                        parity_wrong=int(selection['hyp']!=label),available=int(metric.get('available',False)),
                        pose=metric,success=int(A.good(metric)), reference_corner_source='Exact renderer Xcf projection',
                        new_pixels_rendered=0))
            # Same physical R/t/K/H/footprint area; no RGB generation.
            area=float(truth['xyz'][0]*truth['xyz'][2])
            for name,ratio,order in [('plastic_ratio',1.3/1.1,2),('wood_ratio',.8/.59,2),('square',1.,4)]:
                xyz=np.array([math.sqrt(area/ratio),truth['xyz'][1],math.sqrt(area*ratio)])
                X=g['Xcf'].copy();X[:,0]*=xyz[0]/truth['xyz'][0];X[:,2]*=xyz[2]/truth['xyz'][2]
                q=projected(X,g['R'],g['t'],K)
                if name=='square':
                    actual=pose.infer(q,K,xyz,True);selection=dict(actual_pose=actual,hyp=actual.get('selected_hypothesis'))
                else:selection=solver.production(q,K,xyz,True)
                tr={**truth,'xyz':xyz,'body_xyz':xyz,'order':order}
                metric=pose.metric((fid,selection['actual_pose'],tr))
                width=np.linalg.norm(X[1]-X[0]);depth=np.linalg.norm(X[4]-X[0])
                ref=None if name=='square' else (A.NAMES[0] if width>depth else A.NAMES[1])
                projection.append(dict(id=fid,ratio_condition=name,ratio=ratio,selected=selection['hyp'],
                    reference_parity=ref,parity_wrong=None if ref is None else int(selection['hyp']!=ref),
                    square_C4_equivalent=name=='square',pose=metric,success=int(A.good(metric))))
    A.write_rows('A3_NOISE_ROWS.jsonl.gz',records)
    A.write_rows('A3_RATIO_PROJECTION_ROWS.jsonl.gz',projection)
    grouped=defaultdict(list)
    for r in records:grouped[(r['condition'],r['scale'])].append(r)
    summaries=[dict(condition=kind,scale=scale,frames=16,draws_per_frame=len(rr)//16,
        parity_error=float(np.mean([x['parity_wrong'] for x in rr])),success=float(np.mean([x['success'] for x in rr])),
        T_cm=A.basic(x['pose'].get('translation_cm') for x in rr),R_deg=A.basic(x['pose'].get('rotation_deg') for x in rr))
        for (kind,scale),rr in sorted(grouped.items())]
    A.write('A3_GEOMETRY_DIAGNOSTIC.json',dict(status='DONE',frames=16,frame_ids=ids,
        sample='SYNTH existing heldout ID SHA256 ascending, first16; fixed before solver',seed=20261011,
        predictions='Current N3_THEN_SUBPIX seed1; actual correlated residual retained without remapping',
        Gaussian_model='Independent isotropic auxiliary pixel noise; not actual detector error distribution',
        centroid_noise='All9 points (center included), same declared noise model',
        summaries=summaries,new_PnP_calls=dict(calls),total_new_PnP_calls=sum(calls.values()),
        projection_only_conditions=['plastic_ratio','wood_ratio','square'],
        original_K_R_t_H_and_footprint_area_fixed=True,new_RGB_generation=0,
        interpretation='Small geometry diagnosis; not a frozen-feature classifier upper bound',wall_seconds=time.monotonic()-start))


def rotation(axis,deg):
    import cv2
    vec=np.zeros(3);vec[axis]=np.radians(deg)
    return cv2.Rodrigues(vec)[0]


def g2():
    allowed=set(A.read(A.DOC/'INPUT_CONTRACT.json')['allowed_ids'])
    pose=A.pose_api(C.SOURCE)
    from scripts.research.pallet_wd_hypothesis_diag_20261010 import solver,s3
    solver.configure(pose)
    records=[];source_bindings=[]
    for pop in ('SYNTH','REAL'):
        stage={ (r['id'],r['seed']):r for r in A.rows(C.PUBLIC/A.WD_REL/f'STAGE1_ROWS_{pop}.jsonl.gz')
                if r['method']=='N3_THEN_SUBPIX' and (pop=='SYNTH' or r['id'] in allowed)}
        plans=[r for r in A.rows(C.PUBLIC/A.WD_REL/f'STAGE2_SELECTIONS_S3_{pop}.jsonl.gz')
               if r['method']=='N3_THEN_SUBPIX' and (pop=='SYNTH' or r['id'] in allowed)]
        ids=hashed([r['id'] for r in plans],64) if pop=='SYNTH' else sorted({r['id'] for r in plans})
        if pop=='SYNTH':
            gg=geometry(C.SOURCE,ids)
            truth={fid:exact_truth(stage[(fid,1)],gg[fid])[1] for fid in ids}
        else:truth=A.truth_real(C.SOURCE,{fid:stage[(fid,1)] for fid in ids})
        source_bindings.append(dict(population=pop,frames=len(ids),seeds=[1,2,3],sample='hash-first64' if pop=='SYNTH' else 'All61 cached eligible GT-peer frames'))
        for plan in plans:
            fid=plan['id']
            if fid not in set(ids):continue
            r=stage[(fid,plan['seed'])];normal=np.array(plan['plane_normal']);offset=float(plan['plane_offset_m'])
            metrics={h:pose.metric((fid,r['candidates'][h]['actual_pose'],truth[fid])) for h in A.NAMES}
            for kind,axis,delta in [('offset_cm',None,d) for d in (-5,-2,0,2,5)]+[('normal_deg',ax,d) for ax in range(3) for d in (-2,-1,0,1,2)]:
                n=normal if axis is None else rotation(axis,delta)@normal
                b=offset+delta/100 if axis is None else offset
                chosen=s3.select_from_candidates(r['candidates'],n,b,r['fixed_metadata']['K'])
                h=chosen['hyp'];metric=dict(available=False) if h is None else metrics[h]
                records.append(dict(population='SYNTH64' if pop=='SYNTH' else 'REAL61_GT_PEER_FEASIBILITY',
                    id=fid,session=r['session'],seed=r['seed'],perturbation=kind,camera_axis=axis,delta=delta,
                    selected=h,changed_from_exact_plane=h!=plan['selection']['hyp'],
                    axis_wrong=A.axis_wrong(h,r['hypOracle'],fid),wd_confusion=int(bool(metric.get('available') and metric['rotation_deg']>45 and abs(metric['yaw_deg'])>=60)),
                    success=int(A.good(metric)),pose=metric))
    A.write_rows('G2_PLANE_SENSITIVITY_ROWS.jsonl.gz',records)
    group=defaultdict(list)
    for r in records:group[(r['population'],r['perturbation'],r['camera_axis'],r['delta'])].append(r)
    result=[]
    for key,rr in group.items():
        result.append(dict(population=key[0],perturbation=key[1],camera_axis=key[2],delta=key[3],
            seed_frame_rows=len(rr),frame_ids=len({r['id'] for r in rr}),
            selection_changed=int(sum(r['changed_from_exact_plane'] for r in rr)),
            success_rate=float(np.mean([r['success'] for r in rr])),
            axis_error=A.basic(r['axis_wrong'] for r in rr),wd_confusion=float(np.mean([r['wd_confusion'] for r in rr])),
            T_cm=A.basic(r['pose'].get('translation_cm') for r in rr),R_deg=A.basic(r['pose'].get('rotation_deg') for r in rr)))
    A.write('G2_PLANE_SENSITIVITY.json',dict(status='DONE',interpretation='FEASIBILITY_ONLY; no independent runtime plane',
        source_populations=source_bindings,conditions=result,new_PnP_calls=0,
        score='Unchanged S3 mean absolute bottom4 signed plane residual, in meters; fixed ties long-face first',
        plane_equation='n^T X - offset=0 (b=-offset); normal rotates about camera origin with offset held fixed',
        one_factor_at_a_time=True,weights_tuned=False,positive_offset_direction='Plane moves along its normal',
        actual_sensor_accuracy_claim=False))


def g3():
    allow=set(A.read(A.DOC/'INPUT_CONTRACT.json')['allowed_ids'])
    pose=A.pose_api(C.SOURCE)
    stage={(r['id'],r['seed']):r for r in A.rows(C.PUBLIC/A.WD_REL/'STAGE1_ROWS_REAL.jsonl.gz')
           if r['method']=='N3_THEN_SUBPIX' and r['id'] in allow}
    joined=[]
    for depth in A.rows(C.PUBLIC/A.WD_REL/'DEPTH_ACCURACY_ROWS_REAL.jsonl.gz'):
        if depth['method']!='N3_THEN_SUBPIX' or depth['id'] not in allow:continue
        r=stage[(depth['id'],depth['seed'])];z={}
        for h in A.NAMES:
            p=r['candidates'][h]['actual_pose']
            if not p.get('available'):z[h]=None;continue
            model=pose.cuboid(*p['cf_extents']);cam=model@np.asarray(p['R_cf']).T+p['centroid']
            z[h]=float(cam[:4,2].mean())
        diff=None if any(v is None for v in z.values()) else z[A.NAMES[0]]-z[A.NAMES[1]]
        joined.append(dict(id=r['id'],session=r['session'],seed=r['seed'],available=depth['available'],
            signed_depth_relative_error=depth['relative_error'],absolute_depth_relative_error=depth['absolute_relative_error'],
            candidate_front_mean_depth_m=z,front_depth_long_minus_short_m=diff,
            measured_depth_error_m=None if not depth['available'] else depth['median_m']-depth['reference_front_mean_z_m'],
            S4_selector_evaluated=False))
    original=A.read(C.PUBLIC/A.WD_REL/'DEPTH_GATE.json');assert original['S4_executed'] is False
    A.write_rows('G3_DEPTH_BIAS_ROWS.jsonl.gz',joined)
    by=defaultdict(list)
    for r in joined:by[r['id']].append(r)
    abs_frame=[np.mean([r['absolute_depth_relative_error'] for r in rr]) for rr in by.values()]
    A.write('G3_DEPTH_GATE_REUSE.json',dict(status='REUSED_GATE_FAILURE_WITH_CACHE_ONLY_BIAS_JOIN',
        original_gate_source=dict(path=str(Path(A.WD_REL)/'DEPTH_GATE.json'),sha256=A.sha(C.PUBLIC/A.WD_REL/'DEPTH_GATE.json')),
        original_frames=original['frames'],original_abs_relative_median=original['absolute_relative_error_median'],
        original_status=original['status'],same_failed_settings_preserved=True,
        new_subset='Original depth231 intersect permitted REAL173; not a new gate or threshold selection',
        permitted_join_frames=len(by),seed_frame_rows=len(joined),subset_abs_relative_error=A.basic(abs_frame),
        signed_depth_error=A.basic(r['signed_depth_relative_error'] for r in joined),
        candidate_front_depth_difference_m=A.basic(r['front_depth_long_minus_short_m'] for r in joined),
        S4_selector_evaluated=False,new_depth_inference=0,new_models_downloaded=0,new_PnP_calls=0,
        interpretation='Specific Metric3D setup failed old5% gate; 17.2% versus dimension18% does not prove impossibility for all depth models'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--task',choices=['a3','g2','g3'],required=True)
    args=parser.parse_args();start=time.monotonic()
    {'a3':a3,'g2':g2,'g3':g3}[args.task]()
    print(args.task.upper(),'DONE','wall_seconds',round(time.monotonic()-start,5),flush=True)


if __name__=='__main__':main()
