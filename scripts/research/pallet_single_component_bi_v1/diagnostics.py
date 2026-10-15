"""Saved-stage and separate CAL state audit, after full prediction/scoring."""
from pathlib import Path
from collections import Counter
import argparse
import numpy as np
from scripts.research.pallet_minimal_bi_ablation_v1 import batch as B
from scripts.research.pallet_minimal_bi_ablation_v1 import runtime as RT
from scripts.research.pallet_minimal_bi_ablation_v1 import postseal_diagnostics as PREV
from .run import verify

BRANCHES=(('RAW_YOLO','R0','R1'),('N3_ONLY','N3_F','N3_BI'),
          ('SUBPIX_ONLY','SUBPIX_F','SUBPIX_BI'),('N3_SUBPIX','N0','N1'))

def run(folder,prior,source_root,output):
    folder,prior,source_root,output=map(Path,(folder,prior,source_root,output));verify(folder)
    if output.exists():raise FileExistsError('Keep every diagnostic attempt')
    seal=B.read(folder/'score/SCORE_SEAL.json')
    if seal['prediction_seal_sha256']!=RT.sha256(folder/'PREDICTION_SEAL.json'):raise ValueError('Scored predictions changed')
    for name,h in seal['artifacts'].items():
        if RT.sha256(folder/'score'/name)!=h:raise ValueError('Score asset changed')
    frames=B.read(folder/'inputs_v1/INPUTS.json')['frames'];scored=B.read(folder/'score/SCORED_ROWS.json')['rows']
    calbinding=seal['reference_bindings']['CAL'];labelpath=source_root/calbinding['path']
    if RT.sha256(labelpath)!=calbinding['sha256']:raise ValueError('Original CAL labels changed')
    cal=B.read(labelpath)['rows'];matrices={b:Counter() for b,_,_ in BRANCHES};human=[];stages=[];unavailable=[]
    for i,(f,s) in enumerate(zip(frames,scored)):
        if f['id']!=s['id']:raise ValueError('Paired identity changed')
        for branch,base,arm in BRANCHES:
            root=prior/'run_v2' if branch in ('RAW_YOLO','N3_SUBPIX') else folder/'run'
            o=B.read(root/'observations'/f'{i:03d}_{branch}.json');p=B.read(root/'poses'/f'{i:03d}_{branch}.json')
            if o['id']!=f['id'] or p['id']!=f['id'] or o['result_sha256']!=B.digest(o['result']) or p['result_sha256']!=B.digest(p['result']):raise ValueError('Saved branch changed')
            obs=o['result']['observation'];result=p['result'];summary=PREV.observation_summary(obs,f['arms'][branch]['native_points'],result)
            a=f['arms'][branch]['coarse_pose'];z=result['actual_pose']
            relative=np.asarray(z['R_physical'])@np.asarray(a['R_physical']).T
            rot=float(np.rad2deg(np.arccos(np.clip((np.trace(relative)-1)/2,-1,1))))
            stages.append(dict(id=f['id'],partition=f['partition'],branch=branch,
                own_initial_hypothesis=a.get('selected_hypothesis'),states=obs['states'],diagnostics=summary,
                initial_S_ids=np.flatnonzero(obs['initial_geometry']['initial_self_hidden']).tolist(),
                initial_roles=obs['initial_geometry']['initial_roles'],mask_binding=o['result']['mask_binding'],
                initial_CF_extents=a['cf_extents'],initial_Q=(np.asarray(a['R_cf']).T@np.asarray(a['R_physical'])).tolist(),
                final_CF_extents=z['cf_extents'],final_Q=(np.asarray(z['R_cf']).T@np.asarray(z['R_physical'])).tolist(),
                pose_translation_change_cm=float(100*np.linalg.norm(np.asarray(z['centroid'])-np.asarray(a['centroid']))),
                pose_rotation_change_deg=rot,pose_change_is_not_reference_error=True,
                metrics_before=s['metrics'][base],metrics_after=s['metrics'][arm],
                accepted=result['accepted'],fallback=result['fallback'],
                rank=result['solution'].get('rank'),ambiguity=result['solution'].get('ambiguity')))
            if f['partition']!='CAL':continue
            mapping,reason=PREV.physical_vertex_mapping(a,f['arms']['N3_SUBPIX']['coarse_pose'])
            if mapping is None:unavailable.append(dict(id=f['id'],branch=branch,reason=reason));continue
            labels=cal[f['id']]['metadata']['human_states_native']
            for k,j in enumerate(mapping):
                actual,predicted=labels[j],obs['states'][k];matrices[branch][actual,predicted]+=1
                human.append(dict(id=f['id'],branch=branch,native_corner=k,old_N3_native_corner=j,
                    actual_human_state=actual,predicted_state=predicted))
    B.write(output/'STAGE_DIAGNOSTICS.json',dict(rows=stages,branch_rows=952,frame_rows=238,
        no_new_forward_or_pose_or_fit=True,stored_stage_arithmetic_only=True))
    B.write(output/'CAL_STATE_DIAGNOSTICS.json',dict(full_CAL_frames=46,
        matrices={b:[dict(actual=a,predicted=p,count=n) for (a,p),n in sorted(c.items())] for b,c in matrices.items()},
        records=human,unavailable=unavailable,reference_binding=calbinding,
        existing_CAL_supervision_not_independent_validation=True,new_training=False,
        label_transport='same registered physical vertex bijection as prior; does not authenticate predicted semantic edge identity',
        human_B_vs_I_GT='NA; DIRECT is not a manual boundary/interior role label',human_T_metrology_GT='NA'))
    B.write(output/'DIAGNOSTIC_SEAL.json',dict(prediction_seal_sha256=RT.sha256(folder/'PREDICTION_SEAL.json'),
        score_seal_sha256=RT.sha256(folder/'score/SCORE_SEAL.json'),source_sha256={Path(__file__).name:RT.sha256(__file__)},
        assets={p.name:RT.sha256(p) for p in output.iterdir() if p.is_file()},new_NN_or_PnP_or_fit=0))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,required=True);p.add_argument('--prior',type=Path,required=True)
    p.add_argument('--source-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.folder,a.prior,a.source_root,a.output)
