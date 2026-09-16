"""DEV-only post-hoc semantic line audit; no coefficient re-selection."""
import math
from collections import Counter
import numpy as np,torch
import env as E
from audit_math import conditional_line_distribution
from b_model import raw_lattice
from b_evaluate import csvout
from b_diagnostics import stats
from scripts.research.pallet_symmetry_dht_local_v1.symdht_local.hough import Lattice
from scripts.research.pallet_symmetry_dht_local_v2.symdht_local_v2.geometry import decode_single_mode
from scripts.research.pallet_symmetry_dht_local_v1.symdht_local.constants import EDGES

def main():
    torch.set_num_threads(4);pe=E.C.old('paper_evaluation');pop=pe.population()
    baseline=E.read(E.C.LINE/'baseline/FULL_CANDIDATES.json');keys=sorted(baseline['frames']);indices={key:i for i,key in enumerate(keys)}
    metrics=E.read(E.RAW/'B/DEV_FULL_METRICS.json');base={r['frame_id']:r for r in metrics['B0_P']['1']}
    preds={s:{a:E.read(E.RAW/f'B/DEV_predictions/{a}_seed{s}.json')['frames'] for a in ['B0_P','B3_THREE_AMBIG']} for s in [1,2,3]}
    lattice=Lattice();rows=[];outside={s:[0,0] for s in [1,2,3]};availability=Counter()
    for item in pop.positive.items:
        key=pe.canonical_key(item.image);i=indices[key];c=torch.load(E.RAW/f'B/cache/DEV/{i:04d}.pt',map_location='cpu',weights_only=False)
        fid=item.frame_id;t=pe.E._legacy_forbidden_target(item);target=np.array(base[fid]['target'],float);vm=np.array(base[fid]['gt_mask'],bool)
        if not c['P']:continue
        al=c['P'][1]['alignment']['mapping'];logits=c['line_logits'][al];valid=c['line_valid'][al]
        q,a,r,active,mask=conditional_line_distribution(logits,valid);lines,sigma=raw_lattice(lattice.lines,c['box_raw'])
        maplines=decode_single_mode(logits[None],valid[None],lattice,c['box_raw'][None])['raw_line'][0].numpy()
        point=np.array(max(preds[1]['B0_P'][key],key=lambda x:x['score'])['keypoints_xy'])
        for e,(aa,bb) in enumerate(EDGES):
            role='height' if e in [1,3,5,7] else 'CF_depth' if e>=8 else 'CF_across'
            validGT=bool(vm[aa] and vm[bb] and np.isfinite(target[[aa,bb]]).all() and np.linalg.norm(target[bb]-target[aa])>1e-8)
            row=dict(frame_id=fid,edge=e,role=role,object_type=t.object_type,group='C2',available=bool(active[e]),GT_evaluable=validGT,
              ambiguity=float(a[e]),ambiguity_bin=min(int(float(a[e])*4),3),concentration=float(r[e]),
              physical_visibility='UNKNOWN',projected_length_px=None,length_bin=None,MAP_distance_px=None,
              full_posterior_expected_distance_px=None,MAP_angle_deg=None,P_connected_distance_px=None)
            if validGT:
                ab=target[[aa,bb]];length=float(np.linalg.norm(ab[1]-ab[0]));row['projected_length_px']=length;row['length_bin']=int(np.searchsorted([8,16,32,64],length,side='right'))
                n=np.array([-(ab[1]-ab[0])[1],(ab[1]-ab[0])[0]])/length
                if active[e]:
                    lp=maplines[e];row['MAP_distance_px']=float(abs(ab@lp[:2]+lp[2]).mean());row['MAP_angle_deg']=float(np.degrees(np.arccos(np.clip(abs(lp[:2]@n),0,1))))
                    row['full_posterior_expected_distance_px']=float((abs(ab@lines[:,:2].numpy().T+lines[:,2].numpy())@q[e].numpy()).mean())
                aa2,bb2=point[[aa,bb]];size=np.linalg.norm(bb2-aa2)
                if size>1e-8:
                    pn=np.array([-(bb2-aa2)[1],(bb2-aa2)[0]])/size;row['P_connected_distance_px']=float(abs((ab-aa2)@pn).mean())
            rows.append(row)
        for k in range(8):availability[int(sum(active[e] for e,(aa,bb) in enumerate(EDGES) if k in [aa,bb]))]+=1
        for s in [1,2,3]:
            for z,arm in enumerate(['B0_P','B3_THREE_AMBIG']):
                p=np.array(max(preds[s][arm][key],key=lambda x:x['score'])['keypoints_xy'])[:8];h,w=c['raw_hw']
                outside[s][z]+=int(((p<0).any(-1)|(p[:,0]>=w)|(p[:,1]>=h)).sum())
    fields=['MAP_distance_px','full_posterior_expected_distance_px','MAP_angle_deg','P_connected_distance_px'];groups=[]
    for field,values in [('role',['height','CF_depth','CF_across']),('length_bin',range(5)),('ambiguity_bin',range(4)),('object_type',sorted({r['object_type'] for r in rows}))]:
        for value in values:
            subset=[r for r in rows if r[field]==value];groups.append(dict(group=field,value=value,expected_edges=len(subset),metrics={k:stats([r[k] for r in subset if r[k] is not None]) for k in fields}))
    csvout(E.DOC/'B/DEV_line_error_and_ambiguity.csv',rows)
    E.write(E.DOC/'B/DEV_LINE_DIAGNOSTICS.json',dict(expected_edges=319*12,observation_edge_rows=len(rows),available_edges=sum(r['available'] for r in rows),GT_evaluable_edges=sum(r['GT_evaluable'] for r in rows),
      metrics={k:stats([r[k] for r in rows if r[k] is not None]) for k in fields},groups=groups,
      available_incident_counts={str(i):availability[i] for i in range(4)},outside_raw_image_corner_counts=outside,
      physical_visibility_not_inferred=True,GT_object_type_for_reporting_only=True,parameters_changed=False))
    print('DEV LINE DIAGNOSTICS COMPLETE',len(rows),flush=True)
if __name__=='__main__':main()
