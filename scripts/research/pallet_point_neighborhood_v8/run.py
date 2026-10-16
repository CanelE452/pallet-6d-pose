"""Point-neighborhood guidance for DHT candidates, without connecting points.

The soft prior depends on distance from a candidate infinite line to ANY one
YOLO point. Point order, point pairs, GT roles and GT pose are never inputs.
This is a frozen-score feasibility diagnostic, not trained attention.
"""
from pathlib import Path
import hashlib
import json
import sys
import time
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

KIT=Path(__file__).resolve().parent
ROOT=KIT.parents[2]
OLD=ROOT/'scripts/research/pallet_point_line_decisive_v6'
sys.path.insert(0,str(OLD))
from diagnostics import Split, load_model, choose_oracle_symmetry, G
from cli_train import sha
from plpose_local.heads import decode_modes, roi_lines_to_raw

RUN=ROOT/'data/pallet/results/pallet_point_neighborhood_v8'
V6=ROOT/'data/pallet/results/pallet_point_line_decisive_v6'
STOCK=ROOT/'data/pallet/results/pallet_point_line_causal_v7/STOCK_PREDICTIONS.json'
MANIFEST=ROOT/'data/pallet/results/pallet_point_line_v5_local/export/calibration.json'
CASES={'baseline':None,'points_001':('points',.01),'points_003':('points',.03),
       'points_010':('points',.10),'donor_003':('donor',.03),'box_003':('box',.03)}


def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def proximity_bias(lines,points,sigma,floor=.1):
    """lines [B,N,3], points [B,P,2], sigma [B]. No pair construction."""
    n=lines[...,:2];c=lines[...,2]
    dist=((n[:,:,None]*points[:,None]).sum(-1)+c[:,:,None]).abs()/torch.linalg.vector_norm(n,dim=-1).clamp_min(1e-8)[...,None]
    nearest=dist.min(-1).values
    prior=floor+(1-floor)*torch.exp(-.5*(nearest/sigma[:,None]).square())
    return prior.log()


def controls():
    # A single point supports all orientations through itself: no hidden line joining.
    l=torch.tensor([[[1.,0.,0.],[0.,1.,0.],[1.,0.,-10.]]])
    p=torch.tensor([[[0.,0.]]]);s=torch.tensor([1.])
    b=proximity_bias(l,p,s)
    assert b[0,0]==b[0,1]==0 and b[0,2]<-2
    p=torch.tensor([[[.4,.3],[2.,4.],[5.,1.]]],requires_grad=True)
    a=proximity_bias(l,p,s);assert torch.equal(a,proximity_bias(l,p[:,[2,0,1]],s))
    a.sum().backward();assert torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
    assert torch.isfinite(a).all() and a.min()>=np.log(.1)-1e-6 and a.max()<=0
    return dict(single_point_supports_multiple_orientations=True,point_order_invariant=True,
                finite_nonzero_point_gradient=True,soft_floor_bounded=True)


def decode(model,o,O,bias):
    logits=o['line_logits'] if bias is None else o['line_logits']+bias[:,None]
    l,m=decode_modes(logits,model.lattice,model.thetas,model.rhos,model.modes)
    return roi_lines_to_raw(l,O['box']),m


def stats(rows):
    keys=['role_mean_px','unordered_mean_px','role_best3_mean_px','horizontal_mean_px','height_mean_px']
    return {k:dict(mean=float(np.mean([r[k] for r in rows])),median=float(np.median([r[k] for r in rows])),
                   p90=float(np.quantile([r[k] for r in rows],.9))) for k in keys}


def main():
    torch.set_num_threads(2);RUN.mkdir(parents=True,exist_ok=True)
    test=controls();data=Split(MANIFEST)
    stockdoc=json.loads(STOCK.read_text());assert stockdoc['manifest_sha256']==sha(MANIFEST)
    stock=stockdoc['rows'];assert [r['id'] for r in stock]==[r['id'] for r in data.records]
    ids=[r['id'] for r in data.records]
    order=sorted(range(len(ids)),key=lambda i:hashlib.sha256(('neighborhood-v8:'+ids[i]).encode()).digest())
    donors={order[k]:order[(k+1)%len(order)] for k in range(len(order))}
    protocol=dict(source_sha256=sha(Path(__file__)),manifest_sha256=sha(MANIFEST),stock_sha256=sha(STOCK),
        checkpoints={str(s):sha(V6/f'cache_heads/hough_seed{s}/checkpoint_final.pt') for s in (1,2,3)},
        cases=CASES,floor=.1,main_case='points_003',donor_rule='next SHA-ordered frame; map donor points into current observed box',
        point_prior='log(.1+.9*exp(-min_point_distance_to_line^2/(2*sigma^2))); sigma=fraction*observed_box_diagonal',
        box_control='4 observed bounding-box corners and 4 side midpoints; no YOLO keypoints',
        acceptance='All 3 seeds: main unordered line error mean >=5% lower than baseline, '
                   'role-aware mean and P90 not worse, main unordered mean lower than donor control. '
                   'This gate only supports a frozen-score feasibility follow-up, never actual pose improvement.',
        gt_scope='Scoring only, after all predictions for a seed are saved. Same original point-selected C2 alignment for all cases. '
                 'No GT point anchors. Ignore projected GT edges shorter than 1px for all cases; report their counts.',
        hypothesis='Correct point neighborhoods can distinguish useful line candidates while leaving YOLO point coordinates unchanged.',
        alternatives=['One point leaves line orientation ambiguous.','Many clutter lines pass close to a corner.',
                      'Role ambiguity remains even if geometric line localization improves.',
                      'Hard local masking could remove useful long-line evidence; this diagnostic retains all candidates at >=.1 prior.'],
        scope='Reused calibration matched253/256. No training, no point pair lines, no self-training, no fresh holdout or real images.')
    lock=RUN/'PROTOCOL.json'
    if lock.exists():assert json.loads(lock.read_text())==json.loads(json.dumps(protocol))
    else:write(lock,protocol)
    write(RUN/'CONTROLS.json',test)
    boxes=[]
    for i in range(len(ids)):
        O,_,_=data.batch([i],'cpu',targets=False);boxes.append(O['box'][0])
    boxes=torch.stack(boxes);points=torch.tensor([r['points'] for r in stock],dtype=boxes.dtype)
    norms=(points-boxes[:,None,:2])/(boxes[:,None,2:]-boxes[:,None,:2])
    unitbox=torch.tensor([[0.,0.],[.5,0.],[1.,0.],[1.,.5],[1.,1.],[.5,1.],[0.,1.],[0.,.5]])
    started=time.monotonic();summaries={};allchecks={}
    with torch.inference_mode():
        for seed in (1,2,3):
            model,_=load_model(V6/f'cache_heads/hough_seed{seed}/checkpoint_final.pt','cpu')
            predpath=RUN/f'predictions_seed{seed}.json'
            if predpath.exists():raise SystemExit('refusing to overwrite predictions')
            predictions=[];loaded=[];orig=torch.load
            def traced(path,*args,**kwargs):
                loaded.append(str(path));return orig(path,*args,**kwargs)
            torch.load=traced
            try:
                for first in range(0,len(ids),8):
                    ii=list(range(first,min(first+8,len(ids))));O,_,_=data.batch(ii,'cpu',targets=False);o=model(O)
                    B=len(ii);bb=O['box'];wh=bb[:,2:]-bb[:,:2]
                    lattice=roi_lines_to_raw(model.lattice[None,None].expand(B,1,-1,-1),bb)[:,0]
                    anchors={'points':points[ii],
                        'donor':bb[:,None,:2]+norms[[donors[i] for i in ii]]*wh[:,None],
                        'box':bb[:,None,:2]+unitbox[None]*wh[:,None]}
                    results={}
                    for name,case in CASES.items():
                        bias=None if case is None else proximity_bias(lattice,anchors[case[0]],torch.linalg.vector_norm(wh,dim=-1)*case[1])
                        results[name]=decode(model,o,O,bias)
                    for j,i in enumerate(ii):
                        predictions.append(dict(id=ids[i],original_new_points=o['points'][j].tolist(),
                            yolo_points=points[i].tolist(),donor_id=ids[donors[i]],
                            cases={k:dict(lines=l[j].tolist(),mass=m[j].tolist()) for k,(l,m) in results.items()}))
            finally:torch.load=orig
            assert not any('/supervision/' in p for p in loaded)
            write(predpath,dict(rows=predictions,protocol_sha256=sha(lock),gt_opened_during_inference=0,
                                observed_tensor_load_calls=len(loaded)))
            # All network inference finished before opening any supervision tensor.
            scored={k:[] for k in CASES};valid_count=0;excluded=0
            for i,row in enumerate(predictions):
                O,S,orders=data.batch([i],'cpu',targets=True)
                gt,_,flip=choose_oracle_symmetry(O,S,orders,torch.tensor(row['original_new_points'])[None])
                uv=gt[0,:8].numpy();u,v=np.array(G.EDGES).T
                a,b=uv[u],uv[v];valid=np.linalg.norm(b-a,axis=-1)>=1.
                valid_count+=int(valid.sum());excluded+=int((~valid).sum())
                for k,case in row['cases'].items():
                    l=np.array(case['lines']);mass=np.array(case['mass']);n=l[...,:2];c=l[...,2]
                    err=(abs((a[:,None]*n).sum(-1)+c)+abs((b[:,None]*n).sum(-1)+c))/(2*np.linalg.norm(n,axis=-1))
                    top=mass.argmax(-1);topl=l[np.arange(12),top];role=err[np.arange(12),top]
                    # Same one-to-one matching for every method, geometry-only scoring diagnostic.
                    costs=(abs((a[:,None]*topl[None,:,:2]).sum(-1)+topl[None,:,2])+
                           abs((b[:,None]*topl[None,:,:2]).sum(-1)+topl[None,:,2]))/(2*np.linalg.norm(topl[:,:2],axis=-1)[None])
                    au,av=linear_sum_assignment(costs[valid]);unordered=costs[valid][au,av].mean()
                    scored[k].append(dict(id=ids[i],role_mean_px=float(role[valid].mean()),unordered_mean_px=float(unordered),
                        role_best3_mean_px=float(err.min(-1)[valid].mean()),
                        horizontal_mean_px=float(role[:8][valid[:8]].mean()),height_mean_px=float(role[8:][valid[8:]].mean()),
                        per_edge_role_px=role.tolist(),valid_edges=valid.tolist(),gt_endpoints=np.stack((a,b),1).tolist(),
                        scoring_c2_flip=bool(flip[0])))
            summary={k:stats(v) for k,v in scored.items()};summaries[str(seed)]=summary
            checks=dict(yolo_points_unchanged=all(np.array_equal(np.array(r['yolo_points']),points[i].numpy()) for i,r in enumerate(predictions)),
                        no_target_loads_during_inference=True,scored_frames=len(predictions),valid_edges=valid_count,excluded_degenerate_edges=excluded)
            assert checks['yolo_points_unchanged'];allchecks[str(seed)]=checks
            write(RUN/f'metrics_seed{seed}.json',dict(rows=scored,summary=summary,checks=checks,predictions_sha256=sha(predpath)))
            print('seed',seed,{k:{m:round(v[m]['mean'],3) for m in ('role_mean_px','unordered_mean_px')} for k,v in summary.items()},flush=True)
    gates={}
    for seed,s in summaries.items():
        b=s['baseline'];p=s['points_003'];d=s['donor_003']
        gates[seed]=dict(unordered_gain_5pct=p['unordered_mean_px']['mean']<=.95*b['unordered_mean_px']['mean'],
            role_mean_nonworse=p['role_mean_px']['mean']<=b['role_mean_px']['mean'],
            role_p90_nonworse=p['role_mean_px']['p90']<=b['role_mean_px']['p90'],
            correct_points_better_than_donor=p['unordered_mean_px']['mean']<d['unordered_mean_px']['mean'])
    write(RUN/'SUMMARY.json',dict(complete=True,summaries=summaries,gates=gates,checks=allchecks,
        pilot_gate=all(all(g.values()) for g in gates.values()),elapsed_seconds=time.monotonic()-started,
        scope=protocol['scope'],point_position_or_pose_improvement_tested=False))
    print('COMPLETE gates',gates,flush=True)


if __name__=='__main__':main()
