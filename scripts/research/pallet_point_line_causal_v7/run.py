"""Frozen-model interventions: locate why strong point accuracy is lost.

No training, model selection, real evaluation, or production modifications.
GT-assisted correspondence/mode interventions are privileged diagnostics.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

KIT = Path(__file__).resolve().parent
ROOT = KIT.parents[2]
OLD = ROOT / 'scripts/research/pallet_point_line_decisive_v6'
sys.path.insert(0, str(OLD))
from diagnostics import load_model, Split, G, run_solver, metrics, lines_from_points, choose_oracle_symmetry
from stock_geometry_baseline import R0, EXPECTED, set_error
from cli_train import sha

RUN = ROOT / 'data/pallet/results/pallet_point_line_causal_v7'
V6 = ROOT / 'data/pallet/results/pallet_point_line_decisive_v6'
MANIFEST = ROOT / 'data/pallet/results/pallet_point_line_v5_local/export/calibration.json'
CASES = [
    ('new','learned','none',5), ('new','learned','predicted',5),
    ('new','learned','none',25), ('new','learned','predicted',25),
    ('new','pnp','none',5), ('new','pnp','predicted',5),
    ('new','pnp','gt_best_mode',5), ('new','pnp','oracle',5),
    ('gt','learned','none',5), ('gt','learned','predicted',5),
    ('gt','pnp','none',5), ('gt','pnp','predicted',5),
    ('stock_oracle_order','pnp','none',5), ('stock_oracle_order','pnp','predicted',5),
    ('stock_oracle_order','pnp','gt_best_mode',5), ('stock_oracle_order','pnp','oracle',5),
    ('stock_oracle_order','learned','none',5), ('stock_oracle_order','learned','predicted',5),
]


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def pnp(O, points, fallback):
    """SQPnP from the supplied points only. No RANSAC or GT initialization."""
    X = G.corners(O['dims']).cpu().numpy()[:, :8].astype(np.float64)
    uv = points.detach().cpu().numpy()[:, :8].astype(np.float64)
    kk = O['K'].cpu().numpy().astype(np.float64)
    rr, tt, flags = [], [], []
    for j in range(len(X)):
        ok = False
        try:
            solved, rv, tv = cv2.solvePnP(X[j], uv[j], kk[j], None, flags=cv2.SOLVEPNP_SQPNP)
            r = cv2.Rodrigues(rv)[0] if solved else np.eye(3)
            t = tv.reshape(3) if solved else np.zeros(3)
            ok = bool(solved and np.isfinite(r).all() and np.isfinite(t).all()
                      and ((X[j] @ r.T + t)[:,2]>0).all())
        except cv2.error:
            pass
        if not ok:
            r = fallback['R0'][j,0].cpu().numpy()
            t = fallback['t0'][j,0].cpu().numpy()
        rr.append(r); tt.append(t); flags.append(ok)
    return (torch.as_tensor(np.stack(rr),device=points.device,dtype=points.dtype)[:,None],
            torch.as_tensor(np.stack(tt),device=points.device,dtype=points.dtype)[:,None],flags)


def get_stock(data, device):
    out = RUN/'STOCK_PREDICTIONS.json'
    if out.exists():
        j=json.loads(out.read_text()); assert j['checkpoint_sha256']==EXPECTED
        assert [r['id'] for r in j['rows']]==[r['id'] for r in data.records]
        return j['rows']
    from ultralytics import YOLO
    assert sha(R0)==EXPECTED
    model=YOLO(str(R0),task='pose'); rows=[]
    # No supervision is opened during this phase; cached box matching uses observations.
    for i,rec in enumerate(data.records):
        O,_,_=data.batch([i],'cpu',targets=False)
        result=model.predict(str(ROOT/rec['original_image']),imgsz=640,conf=.001,iou=.7,
                             max_det=300,device=device,verbose=False,half=False)[0]
        boxes=result.boxes.xyxy.cpu().numpy()-100.
        assert len(boxes)>0
        delta=np.max(np.abs(boxes-O['box'][0].numpy()),axis=-1); best=int(delta.argmin())
        assert delta[best]<.1, (rec['id'],float(delta[best]))
        points=result.keypoints.xy[best].cpu().numpy()[:8]-100.
        rows.append(dict(id=rec['id'],points=points.tolist(),box_delta=float(delta[best])))
        if (i+1)%50==0: print('stock predictions',i+1,flush=True)
    write(out,dict(checkpoint_sha256=EXPECTED,manifest_sha256=sha(MANIFEST),rows=rows,
                   scope='GT-free point inference; original detection-match population is GT-conditioned.'))
    del model
    return rows


def summarize(rows):
    valid=np.array([r['valid'] for r in rows]); n=np.array([r['normalized'] for r in rows])
    raw=np.array([r['raw_px'] for r in rows if r['valid']])
    geom=np.array([r['unordered_px'] for r in rows if r['valid']])
    return dict(n=len(rows),coverage=float(valid.mean()),normalized_mean=float(n.mean()),
                raw_mean=float(raw.mean()),raw_median=float(np.median(raw)),raw_p90=float(np.quantile(raw,.9)),
                unordered_mean=float(geom.mean()))


def main(args):
    torch.set_num_threads(2);cv2.setNumThreads(1)
    RUN.mkdir(parents=True,exist_ok=True)
    lock=RUN/'PROTOCOL.json'
    protocol=dict(source_sha256=sha(Path(__file__)),manifest_sha256=sha(MANIFEST),cases=CASES,
        checkpoints={str(s):sha(V6/f'cache_heads/hough_seed{s}/checkpoint_final.pt') for s in (1,2,3)},
        dependency_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(OLD.glob('plpose_local/*.py'))},
        hypotheses=[
            'H1 weak point observations: replace points only, keep learned starts and line evidence.',
            'H2 weak starts: replace learned starts by point-derived SQPnP, keep points and lines.',
            'H3 finite iterations: compare 5 and 25 with identical inputs and starts.',
            'H4 inaccurate lines harm strong points: compare line off/on with oracle-ordered stock points and same PnP start.',
            'H5 mode selection: replace mixture by GT-best existing mode versus exact GT line at same points/start.',
            'H6 implementation sanity: exact GT points plus point-derived PnP and no lines must reconstruct GT.'],
        scope='Reused synthetic calibration253/256 matched detections, 3 fixed Hough checkpoints. '
              'No training and no fresh holdout. Not a deployable fusion or improvement claim. '
              'All gt, stock_oracle_order, gt_best_mode, oracle cases use privileged target information.',
        failure_policy='Record invalid PnP, fall back to first learned start; keep every frame in metrics.',
        source='https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html')
    if lock.exists():
        assert json.loads(lock.read_text())==json.loads(json.dumps(protocol))
    else: write(lock,protocol)
    data=Split(MANIFEST); started=time.monotonic(); stocks=get_stock(data,args.device)
    all_summary={}
    with torch.inference_mode():
        for seed in (1,2,3):
            out=RUN/f'seed{seed}.json'
            if out.exists():
                all_summary[str(seed)]=json.loads(out.read_text())['summary']; continue
            model,_=load_model(V6/f'cache_heads/hough_seed{seed}/checkpoint_final.pt',args.device)
            cells={'/'.join(map(str,c)):[] for c in CASES}; observations=[]; pnp_flags=[]
            for first in range(0,len(data.records),8):
                ids=list(range(first,min(first+8,len(data.records))))
                O,_,orders=data.batch(ids,args.device,targets=False); o=model(O)
                # This practical intervention is fully computed before target access.
                npnp=pnp(O,o['points'],o)
                _,sup,_=data.batch(ids,args.device,targets=True)
                gt,_,_=choose_oracle_symmetry(O,sup,orders,o['points'])
                stock=torch.zeros_like(gt)
                assignments=[]
                for j,i in enumerate(ids):
                    pp=np.array(stocks[i]['points'])
                    u,v=linear_sum_assignment(np.linalg.norm(pp[:,None]-gt[j,:8].cpu().numpy()[None],axis=-1))
                    ordered=np.zeros((8,2));ordered[v]=pp[u]
                    stock[j,:8]=torch.as_tensor(ordered,device=args.device,dtype=gt.dtype)
                    assignments.append(v.tolist())
                stock[:,8]=stock[:,:8].mean(1) # centre is masked, never scored or fitted.
                gpnp=pnp(O,gt,o);spnp=pnp(O,stock,o)
                pts={'new':o['points'],'gt':gt,'stock_oracle_order':stock}
                pnps={'new':npnp,'gt':gpnp,'stock_oracle_order':spnp}
                oracle,_=lines_from_points(gt)
                u,v=torch.tensor(G.EDGES,device=args.device).T
                n=o['lines'][...,:2];cc=o['lines'][...,2]
                dist=(((gt[:,u,None]*n).sum(-1)+cc).abs()+((gt[:,v,None]*n).sum(-1)+cc).abs())/torch.linalg.vector_norm(n,dim=-1).clamp_min(1e-8)
                best=dist.argmin(-1)
                bestline=o['lines'].gather(2,best[...,None,None].expand(-1,-1,1,3))
                lines={'none':(None,None),'predicted':(o['lines'],o['mass']),
                       'gt_best_mode':(bestline,torch.ones_like(bestline[...,0])),
                       'oracle':(oracle,torch.ones_like(oracle[...,0]))}
                for j,i in enumerate(ids):
                    ob=dict(id=data.records[i]['id'],gt_points=sup['points'][j,:8].cpu().tolist(),
                            stock_assignment_to_aligned_gt=assignments[j])
                    for k,pp in pts.items():
                        e,_=set_error(pp[j,:8].cpu().numpy(),sup['points'][j,:8].cpu().numpy())
                        ob[k+'_unordered_px']=float(e.mean())
                    observations.append(ob)
                    pnp_flags.append(dict(id=data.records[i]['id'],**{k:bool(vv[2][j]) for k,vv in pnps.items()}))
                for point_source,init,source,iters in CASES:
                    key='/'.join(map(str,(point_source,init,source,iters)))
                    override=dict(o,points=pts[point_source]);ll,mm=lines[source]
                    rr,tt=(pnps[point_source][:2] if init=='pnp' else (o['R0'],o['t0']))
                    pred=run_solver(O,override,ll,mm,R0=rr,t0=tt,iters=iters)
                    met=metrics(O,sup,orders,pred)
                    uv=G.project(O['K'],pred['R'],pred['t'],G.corners(O['dims']))[0]
                    for j,i in enumerate(ids):
                        ok=bool(met['valid'][j]); geom,_=set_error(uv[j,:8].cpu().numpy(),sup['points'][j,:8].cpu().numpy())
                        cells[key].append(dict(id=data.records[i]['id'],valid=ok,
                            normalized=float(met['normalized'][j]),raw_px=float(met['raw_px'][j]) if ok else None,
                            unordered_px=float(geom.mean()) if ok else None,
                            projected_points=uv[j,:8].cpu().tolist(),R=pred['R'][j].cpu().tolist(),t=pred['t'][j].cpu().tolist()))
                if first%64==0:print('intervention seed',seed,'frames',first+len(ids),flush=True)
            summary={k:summarize(v) for k,v in cells.items()}
            # Exact GT control catches coordinate, correspondence and PnP/refinement implementation defects.
            control=summary['gt/pnp/none/5']
            assert control['coverage']==1 and control['raw_p90']<.01, control
            write(out,dict(seed=seed,complete=True,summary=summary,cells=cells,observations=observations,
                           pnp_flags=pnp_flags,protocol_sha256=sha(lock)))
            all_summary[str(seed)]=summary
            print('finished seed',seed, json.dumps({k:round(v['raw_mean'],3) for k,v in summary.items()}),flush=True)
            del model
    write(RUN/'SUMMARY.json',dict(complete=True,seeds=all_summary,elapsed_seconds=time.monotonic()-started,
                                protocol_sha256=sha(lock),scope=protocol['scope']))
    print('COMPLETE',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--device',default='cpu');main(ap.parse_args())
