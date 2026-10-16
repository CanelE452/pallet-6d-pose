"""Seal learned observations before reference reads; supports shared-neck timing."""
import argparse
import hashlib
from itertools import combinations
import time
import cv2
import numpy as np
import torch
from . import common as C
from . import model as M
from .source_audit import EDGES

FITS=C.SCRATCH/'learned_fits'

def checkpoint_path(arm='IMAGE_ROLE'):
    return FITS/(arm+'.pt')

def load_head(arm='IMAGE_ROLE'):
    checkpoint=torch.load(checkpoint_path(arm),map_location='cpu',weights_only=False)
    assert checkpoint['arm']==arm and checkpoint['steps']==3000
    assert checkpoint['config']==M.CONFIG
    head=M.CorrespondenceHead().cuda().eval();head.load_state_dict(checkpoint['model']);head.requires_grad_(False)
    return head

def decode(query,logits):
    """Each query picks one bin or no-match. TLS never averages different basins."""
    raw=logits.detach().float().cpu().numpy();choice=raw.argmax(-1);lines=[];queries=[]
    for i,(e,a,b,u,length) in enumerate(query['identity']):
        valid=bool(query['valid'][i]);chosen=int(choice[i])
        queries.append(dict(query=i,edge=int(e),endpoints=[int(a),int(b)],fraction=u,center=query['center'][i].tolist(),
            normal=query['normal'][i].tolist(),candidate_logits=raw[i].tolist(),no_match=chosen==65 or not valid,
            chosen_candidate=chosen,selected_xy=query['candidate'][i,chosen].tolist() if chosen<65 and valid else None))
    for edge,(a,b) in enumerate(EDGES):
        ids=[i for i in range(edge*7,(edge+1)*7) if query['valid'][i] and choice[i]<65]
        if len(ids)<2:continue
        points=np.array([query['candidate'][i,choice[i]] for i in ids]);center=points.mean(0)
        _,singular,v=np.linalg.svd(points-center,full_matrices=False)
        if singular[0]<1e-6:continue
        tangent=v[0];normal=np.array([-tangent[1],tangent[0]]);offset=float(normal@center)
        support=(points-center)@tangent;span=float(support.max()-support.min())
        lines.append(dict(edge=edge,endpoints=[a,b],normal=normal.tolist(),offset=offset,
            support_length_px=span,queries=ids,support_points=points.tolist(),
            residual_rms_px=float(np.sqrt(np.mean((points@normal-offset)**2))),
            correlated_observation_source='one physical edge; sampled queries are not separate 3D corner IDs'))
    lookup={line['edge']:line for line in lines};corners=[]
    for corner in range(8):
        incident=[e for e,(a,b) in enumerate(EDGES) if corner in (a,b) and e in lookup]
        pairs=[]
        for a,b in combinations(incident,2):
            matrix=np.array([lookup[a]['normal'],lookup[b]['normal']]);det=float(np.linalg.det(matrix))
            if abs(det)>1e-6:pairs.append((-abs(det),a,b,matrix))
        if not pairs:continue
        _,a,b,matrix=min(pairs,key=lambda r:r[:3]);xy=np.linalg.solve(matrix,[lookup[a]['offset'],lookup[b]['offset']])
        if not np.isfinite(xy).all():continue
        corners.append(dict(id=corner,xy=xy.tolist(),edges=[a,b],source='two selected physical-edge support line intersection',
            extrapolation_possible=True,correlated_edges=[a,b]))
    return dict(corners=corners,lines=lines,queries=queries,selected_queries=sum(not q['no_match'] for q in queries),
        raw_logits_sha256=hashlib.sha256(raw.astype('<f4').tobytes()).hexdigest(),
        all_no_match=len(lines)==0,quantization='integer1px candidate; no sub-bin refinement',
        feature_initial_pose=query['initial_pose'],feature_hidden_initial=query['hidden_initial'],predicted_roles=query['role'])

@torch.no_grad()
def observe(image,captured,frame,head,arm='IMAGE_ROLE'):
    points=frame['points']['BASE'];x,query=M.inputs(image,captured,points,np.array(frame['K']),np.array(frame['xyz']))
    return decode(query,head(x[None],arm)[0])

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='LEARNED_OBSERVATIONS.jsonl.gz');args=p.parse_args()
    assert C.read(C.DOC/'TRAINING_COMPLETION.json')['complete']
    C.source_modules()
    from scripts.research.pallet_line_pose_v1.features import FrozenYoloFeatures
    detector=FrozenYoloFeatures(weights=C.ROOT/'challenge/yolo_pose_one_model/spatial_concat_scratch/runs/YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42/weights/best.pt')
    heads={arm:load_head(arm) for arm in M.ARMS};frames=C.read(C.DOC/'INPUTS.json')['frames'];rows=[];start=time.monotonic();calls=0;parity=[]
    cv2.setNumThreads(1);torch.set_num_threads(1)
    with C.no_truth_reads():
        for index,frame in enumerate(frames):
            image=cv2.imread(str(C.ROOT/frame['image']));captured=detector.predict(image);calls+=1;selected=captured['selected_index']
            if selected is None:
                for arm in M.ARMS:rows.append(dict(id=frame['id'],session=frame['session'],method=arm,corners=[],lines=[],queries=[],detector_available=False,GT_input=False))
                continue
            points=captured['candidates'][selected]['keypoints_xy'];error=float(np.max(np.abs(points-np.asarray(frame['points']['BASE']))));parity.append(error)
            assert error<=.001,('frozen detector parity',frame['id'],error)
            x,query=M.inputs(image,captured,points,np.array(frame['K']),np.array(frame['xyz']))
            with torch.no_grad():
                for arm,head in heads.items():
                    observation=decode(query,head(x[None],arm)[0]);rows.append(dict(id=frame['id'],session=frame['session'],method=arm,
                        original_base_points=points.tolist(),**observation,detector_available=True,GT_input=False))
            if (index+1)%32==0:print('SEALED_LEARNED_OBSERVATIONS',index+1,len(frames),round(time.monotonic()-start,1),flush=True)
    detector.close();C.save_rows(C.DOC/args.output,rows)
    C.write(C.DOC/'OBSERVATION_SEAL.json',dict(complete=True,frames=len(frames),models=list(M.ARMS),rows=len(rows),
        records=C.binding(C.DOC/args.output),checkpoints=[C.binding(checkpoint_path(a)) for a in M.ARMS],
        detector_forward=calls,head_forward=len(rows),initial_feature_pose_calls=len(frames),
        wall_seconds=time.monotonic()-start,GT_open_allowed=False,max_detector_coordinate_parity=max(parity or [0.])))
    print('LEARNED_OBSERVATIONS_COMPLETE',len(rows),time.monotonic()-start,flush=True)

if __name__=='__main__':main()
