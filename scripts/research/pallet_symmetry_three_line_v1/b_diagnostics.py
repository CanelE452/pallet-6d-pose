"""Post-hoc line/ambiguity/candidate diagnostics. Never selects an eta."""
import math,csv
from collections import Counter
import numpy as np
import torch
import env as E
from b_model import raw_lattice
from audit_math import conditional_line_distribution,validate_edges
from b_evaluate import csvout
from scripts.research.pallet_symmetry_dht_local_v1.symdht_local.constants import EDGES
from scripts.research.pallet_symmetry_dht_local_v1.symdht_local.hough import Lattice
from scripts.research.pallet_symmetry_dht_local_v2.symdht_local_v2.geometry import decode_single_mode
from scripts.research.pallet_symmetry_dht_local_v1.symdht_local.data import ObservationDataset

def stats(a):
    a=np.array(a,float)
    if not len(a):return dict(n=0)
    return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),P90=float(np.quantile(a,.9)))
def main():
    torch.set_num_threads(4);lattice=Lattice();edges=np.array(EDGES);incidence=validate_edges(edges)
    selection=E.read(E.DOC/'B/selection_lock.json');eta=selection['selected_eta']['B3_THREE_AMBIG']
    configs=E.read(E.C.B/'P_SELECTION.json');ds=ObservationDataset(E.EXPORT/'synth_val.json',targets=True)
    metrics=E.read(E.RAW/'B/SYNTH_FULL_METRICS.json');byid={r['frame_id']:r for r in metrics['B0_P']['1']}
    p0={s:torch.load(E.RAW/f'B/synth_val_predictions/B0_P_seed{s}.pt',map_location='cpu',weights_only=False)['points'] for s in [1,2,3]}
    p3={s:torch.load(E.RAW/f'B/synth_val_predictions/B3_THREE_AMBIG_seed{s}.pt',map_location='cpu',weights_only=False)['points'] for s in [1,2,3]}
    geom=dict(np.load(E.ROOT/'challenge/yolo_pose_one_model/pallet_translation_loss_v1/GEOMETRY_SIDETABLE.npz'));gi={str(v):i for i,v in enumerate(geom['stems'])}
    rows=[];corners=[];changed=Counter();available=0;evaluable=0;outside={s:[0,0] for s in [1,2,3]}
    for i,record in enumerate(ds.records):
        c=torch.load(E.RAW/f'B/cache/synth_val/{i:04d}.pt',map_location='cpu',weights_only=False);t=ds[i]
        mapping=c['P'][1]['alignment']['mapping'];logits=c['line_logits'][mapping];valid=c['line_valid'][mapping]
        q,a,r,active,mask=conditional_line_distribution(logits,valid)
        lines,sigma=raw_lattice(lattice.lines,c['box_raw'])
        decoded=decode_single_mode(logits[None],valid[None],lattice,c['box_raw'][None])['raw_line'][0].numpy()
        original=t['target_points'].numpy();vm=t['target_valid'].numpy();fid=c['frame_id'];g=byid[fid]['branch'] if fid in byid else 0
        p=np.array(c['perms'][g]);gt=original[p];v=vm[p]&np.isfinite(gt).all(-1)
        X=geom['Xcf'][gi[fid]]
        for e,(aa,bb) in enumerate(edges):
            av=bool(active[e]);ev=bool(v[aa] and v[bb] and np.linalg.norm(gt[bb]-gt[aa])>1e-8);available+=av;evaluable+=ev
            direction=int(np.argmax(abs(X[bb]-X[aa])));role=['width_axis_X','height_axis_Y','depth_axis_Z'][direction]
            d=dict(frame_id=fid,edge=e,a=int(aa),b=int(bb),role_3D=role,source=record['source'],asset=record['asset'],
              symmetry_order=record['symmetry_order'],available=av,GT_evaluable=ev,ambiguity=float(a[e]),
              ambiguity_bin=min(int(float(a[e])*4),3),concentration=float(r[e]),physical_visibility='UNKNOWN',
              GT_alignment='same whole-object branch as P1 2D metric, posthoc only',projected_length_px=None,
              length_bin=None,MAP_endpoint_distance_px=None,posterior_expected_endpoint_distance_px=None,
              MAP_angle_deg=None,R0_connected_endpoint_distance_px=None,P1_connected_endpoint_distance_px=None)
            if ev:
                ab=gt[[aa,bb]];length=float(np.linalg.norm(ab[1]-ab[0]));d['projected_length_px']=length
                d['length_bin']=int(np.searchsorted([8,16,32,64],length,side='right'))
                normal=np.array([-(ab[1]-ab[0])[1],(ab[1]-ab[0])[0]])/length
                if av:
                    line=decoded[e];d['MAP_endpoint_distance_px']=float(abs(ab@line[:2]+line[2]).mean())
                    d['MAP_angle_deg']=float(np.degrees(np.arccos(np.clip(abs(line[:2]@normal),0,1))))
                    distances=abs(ab@lines[:,:2].numpy().T+lines[:,2].numpy());d['posterior_expected_endpoint_distance_px']=float((distances@q[e].numpy()).mean())
                for tag,points in [('R0',c['base_raw'].numpy()),('P1',p0[1][i])]:
                    aa2,bb2=points[[aa,bb]];norm=np.linalg.norm(bb2-aa2)
                    if np.isfinite(norm) and norm>1e-8:
                        n=np.array([-(bb2-aa2)[1],(bb2-aa2)[0]])/norm;d[tag+'_connected_endpoint_distance_px']=float(abs((ab-aa2)@n).mean())
            rows.append(d)
        for seed in [1,2,3]:
            pp=c['P'][seed];out=pp['output'];T=configs['temperatures'][str(seed)]['temperature'];bias=pp['biases']['B3_THREE_AMBIG']
            old=out['logits'][0]/T;new=old+eta*bias;locations=(out['points_raw'][0,:8,None]+out['candidate_displacements'][0,None]-torch.tensor(c['offset']))/c['gain']-100
            height,width=c['raw_hw']
            for which,pts in enumerate([p0[seed][i],p3[seed][i]]):outside[seed][which]+=int(((pts[:8]<0).any(-1)|(pts[:8,0]>=width)|(pts[:8,1]>=height)).sum())
            for k in range(8):
                ids=np.flatnonzero(incidence[k]);modified=bool((eta*bias[k]).abs().max()>0);changed[seed]+=modified
                topold=int(old[k].argmax());topnew=int(new[k].argmax());distanceold=distancenew=None
                if v[k]:
                    distanceold=float(np.linalg.norm(locations[k,topold].numpy()-gt[k]));distancenew=float(np.linalg.norm(locations[k,topnew].numpy()-gt[k]))
                corners.append(dict(frame_id=fid,seed=seed,corner=k,available_incident=int(active[ids].sum()),
                  reliability_sum=float(r[ids].sum()),bias_changed=modified,coordinate_move_px=float(np.linalg.norm(p3[seed][i,k]-p0[seed][i,k])),
                  top_before=topold,top_after=topnew,top_GT_distance_before=distanceold,top_GT_distance_after=distancenew,
                  expectation_GT_distance_before=float(np.linalg.norm(p0[seed][i,k]-gt[k])) if v[k] else None,
                  expectation_GT_distance_after=float(np.linalg.norm(p3[seed][i,k]-gt[k])) if v[k] else None,
                  GT_rank_reference='P1 whole-object branch held fixed for diagnostics only'))
    csvout(E.DOC/'B/line_error_and_ambiguity.csv',rows);csvout(E.DOC/'B/corner_diagnostics.csv',corners)
    fields=['MAP_endpoint_distance_px','posterior_expected_endpoint_distance_px','MAP_angle_deg','R0_connected_endpoint_distance_px','P1_connected_endpoint_distance_px']
    global_stats={k:stats([r[k] for r in rows if r[k] is not None]) for k in fields}
    d=np.array([r['MAP_endpoint_distance_px'] for r in rows if r['MAP_endpoint_distance_px'] is not None]);angles=np.array([r['MAP_angle_deg'] for r in rows if r['MAP_angle_deg'] is not None])
    groups=[]
    for field,values in [('role_3D',['width_axis_X','height_axis_Y','depth_axis_Z']),('length_bin',range(5)),('ambiguity_bin',range(4)),('source',sorted({r['source'] for r in rows})),('symmetry_order',[1,2])]:
        for value in values:
            group=[r for r in rows if r[field]==value];groups.append(dict(field=field,value=value,n_expected=len(group),stats={k:stats([r[k] for r in group if r[k] is not None]) for k in fields}))
    E.write(E.DOC/'B/LINE_DIAGNOSTICS.json',dict(expected_edges=len(rows),available_edges=available,GT_evaluable_edges=evaluable,
      endpoint_error=global_stats,diagnostic_exceedance_px={str(t):float((d>t).mean()) for t in [1,2,5,10]},
      diagnostic_exceedance_angle_deg={str(t):float((angles>t).mean()) for t in [5,10]},groups=groups,
      modified_corner_scores=dict(changed),outside_raw_image_corner_counts=outside,
      available_incident_counts={str(j):sum(r['available_incident']==j for r in corners if r['seed']==1) for j in range(4)},
      confidence_is_not_visibility=True,MAP_lines='post-hoc description only; B used all 2340 bins',
      unverified_subgroups=['physical occlusion','nighttime','independent measured viewpoint'],gate_for_execution=False))
    print('LINE DIAGNOSTICS',global_stats,flush=True)
if __name__=='__main__':main()
