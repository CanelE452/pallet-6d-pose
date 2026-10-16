"""Frozen final-checkpoint evaluation and point-input interventions, no selection."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from prepare import RUN,ROOT,EXPORT,OLD,Split,sha,write
from model import GuidedDHT,point_maps
from train import pack,V6,ARMS
from plpose_local.geometry import EDGES,symmetry_permutations


def read(p):return json.loads(p.read_text())


def score(rows,data):
    evaluated=[]
    for i,row in enumerate(rows):
        O,S,orders=data.batch([i],'cpu',targets=True)
        l=np.array(row['lines']);mass=np.array(row['mass']);n=l/np.linalg.norm(l[...,:2],axis=-1)[...,None]
        top=mass.argmax(-1);topline=n[np.arange(12),top]
        gt=S['points'][0,:8].numpy();u,v=np.array(EDGES).T
        cands=[]
        for g in range(int(orders[0])):
            perm=np.array(symmetry_permutations(int(orders[0]))[g])[:8]
            uv=gt[perm];ep=np.stack((uv[u],uv[v]),1)
            valid=np.linalg.norm(ep[:,1]-ep[:,0],axis=-1)>=1.
            homogeneous=np.concatenate((ep,np.ones((*ep.shape[:-1],1))),-1)
            cost=np.abs(np.einsum('epd,qmd->epqm',homogeneous,n)).mean(1)
            role=cost[np.arange(12),np.arange(12),top]
            cands.append((float(role[valid].mean()),role,ep,valid,cost,g))
        val,role,ep,valid,cost,g=min(cands,key=lambda c:c[0])
        # cost [GTedge, predictedrole, mode]. Best of modes is oracle diagnostic only.
        selected=cost[:,np.arange(12),top]
        hu,hv=linear_sum_assignment(selected[valid]);geom=float(selected[valid][hu,hv].mean())
        diag=np.linalg.norm(O['image_hw'][0].numpy())
        evaluated.append(dict(id=row['id'],role_mean_px=val,role_normalized_mean=float((role[valid]/diag).clip(0,1).mean()),
            unordered_mean_px=geom,horizontal_mean_px=float(role[:8][valid[:8]].mean()),height_mean_px=float(role[8:][valid[8:]].mean()),
            best3_role_mean_px=float(cost[np.arange(12),np.arange(12)].min(-1)[valid].mean()),
            gt_coverage10=float((selected[valid].min(-1)<=10).mean()),
            role_correct10=float((role[valid]<=10).mean()),valid_edges=valid.tolist(),gt_endpoints=ep.tolist(),
            per_edge_role_px=role.tolist(),scoring_symmetry=g,image_diagonal=float(diag)))
    keys=['role_mean_px','role_normalized_mean','unordered_mean_px','horizontal_mean_px','height_mean_px',
          'best3_role_mean_px','gt_coverage10','role_correct10']
    stats={k:dict(mean=float(np.mean([r[k] for r in evaluated])),median=float(np.median([r[k] for r in evaluated])),
                  p90=float(np.quantile([r[k] for r in evaluated],.9))) for k in keys}
    return dict(rows=evaluated,summary=stats,n=len(rows))


def main():
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert read(RUN/'TRAINING_COMPLETE.json')['complete']
    proto=RUN/'EVALUATION_PROTOCOL.json'
    if proto.exists():assert read(proto)['source_sha256']==sha(Path(__file__))
    else:write(proto,dict(source_sha256=sha(Path(__file__)),training_protocol_sha256=sha(RUN/'PROTOCOL.json'),
        prediction_scope='All network prediction files for both splits are saved before evaluation loads GT.',
        scoring='One permitted object-wide C1/C2 alignment per prediction; all methods use identical endpoint-normal error. '
                'Role-free metric uses one-to-one line assignment, not independent nearest-GT matching.',
        primary='synth_val',interventions='points-trained models evaluated with points/none/donor/box inputs on calibration',
        scope='Reused matched populations; no fresh holdout or real evaluation. No checkpoint or hyperparameter selection.'))
    (RUN/'predictions').mkdir(exist_ok=True);(RUN/'metrics').mkdir(exist_ok=True)
    started=time.monotonic();files=[]
    for split in ('synth_val','calibration'):
        loaded=[];orig=torch.load
        def traced(path,*args,**kwargs):loaded.append(str(path));return orig(path,*args,**kwargs)
        torch.load=traced
        try:
            data,O,maps,orders,target,points,_=pack(split,'cuda:0',False);assert target is None
            bb=O['box'];unit=bb.new_tensor([[0,0],[.5,0],[1,0],[1,.5],[1,1],[.5,1],[0,1],[0,.5]])
            maps['box']=point_maps(bb[:,None,:2]+unit[None]*(bb[:,None,2:]-bb[:,None,:2]),bb)
            for seed in (1,2,3):
                for arm in ARMS:
                    checkpoint=RUN/'models'/f'{arm}_seed{seed}'/'final.pt'
                    model=GuidedDHT(V6/f'cache_heads/hough_seed{seed}/checkpoint_final.pt',seed).to('cuda:0')
                    model.load_state_dict(torch.load(checkpoint,map_location='cuda:0',weights_only=False)['state_dict']);model.eval()
                    variants=['points','none','donor','box'] if split=='calibration' and arm=='points' else [arm]
                    for variant in variants:
                        name=f'{arm}_seed{seed}_{split}_{variant}';path=RUN/'predictions'/f'{name}.json'
                        if path.exists():raise SystemExit('prediction already exists: '+name)
                        rows=[]
                        with torch.inference_mode():
                            for first in range(0,len(data.records),16):
                                ii=list(range(first,min(first+16,len(data.records))));obs={k:v[ii] for k,v in O.items()}
                                out=model(obs,maps[variant][ii],decode=True)
                                for j,i in enumerate(ii):
                                    gate=out['gate'][j]
                                    row=dict(id=data.records[i]['id'],lines=out['lines'][j].cpu().tolist(),mass=out['mass'][j].cpu().tolist(),
                                        yolo_points=points[i].cpu().tolist(),gate_mean=float(gate.mean()),
                                        gate_mean_abs_change=float((gate-1).abs().mean()))
                                    if arm=='points':row['gate_map']=gate.mean(0).cpu().tolist()
                                    rows.append(row)
                        assert [r['id'] for r in rows]==[r['id'] for r in data.records]
                        write(path,dict(rows=rows,arm=arm,seed=seed,split=split,variant=variant,
                            checkpoint_sha256=sha(checkpoint),evaluation_protocol_sha256=sha(proto),
                            scope='Original YOLO points unchanged; gate is feature modulation, not a causal attribution map.'))
                        files.append(name);print('predicted',name,len(rows),flush=True)
                    del model
        finally:torch.load=orig
        assert not any('/supervision/' in p for p in loaded)
        write(RUN/f'INFERENCE_AUDIT_{split}.json',dict(supervision_loads=0,observed_tensor_load_calls=len(loaded),
            matched_frames=len(data.records),all_prediction_ids_complete=True))
        del O,maps,points
    # Independent scoring phase begins after predictions are fixed on disk.
    summaries={}
    for name in files:
        p=read(RUN/'predictions'/f'{name}.json');data=Split(EXPORT/f'{p["split"]}.json')
        result=score(p['rows'],data);result.update(prediction_sha256=sha(RUN/'predictions'/f'{name}.json'))
        write(RUN/'metrics'/f'{name}.json',result);summaries[name]=result['summary']
        print('scored',name,'role',round(result['summary']['role_mean_px']['mean'],3),'geometry',round(result['summary']['unordered_mean_px']['mean'],3),flush=True)
    gates={}
    for seed in (1,2,3):
        b=summaries[f'none_seed{seed}_synth_val_none'];p=summaries[f'points_seed{seed}_synth_val_points'];d=summaries[f'donor_seed{seed}_synth_val_donor']
        cp=summaries[f'points_seed{seed}_calibration_points'];cd=summaries[f'points_seed{seed}_calibration_donor']
        gates[str(seed)]=dict(role_mean_gain5=p['role_mean_px']['mean']<=.95*b['role_mean_px']['mean'],
            geometry_mean_gain5=p['unordered_mean_px']['mean']<=.95*b['unordered_mean_px']['mean'],
            role_p90_nonworse=p['role_mean_px']['p90']<=b['role_mean_px']['p90'],
            beats_trained_donor=p['role_mean_px']['mean']<d['role_mean_px']['mean'],
            correct_input_beats_donor_input=cp['role_mean_px']['mean']<cd['role_mean_px']['mean'])
    write(RUN/'SUMMARY.json',dict(complete=True,summaries=summaries,gates=gates,
        pilot_gate=all(all(g.values()) for g in gates.values()),elapsed_seconds=time.monotonic()-started,
        evaluated_cells=len(files),training_steps=18000,real_or_final_pose_improvement_established=False))
    print('EVALUATION COMPLETE',gates,flush=True)


if __name__=='__main__':main()
