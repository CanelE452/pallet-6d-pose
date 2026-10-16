"""Frozen predictions, identical starts; intervene on line evidence only.

All oracle outputs are diagnostic. Never export them as model performance.
"""
from pathlib import Path
import argparse
import json
import math
import sys
import numpy as np
import torch

KIT = Path(__file__).resolve().parent
ROOT = KIT.parents[2]
sys.path.insert(0,str(KIT))
from cli_train import Split, sha
from plpose_local import solver as S, geometry as G
from plpose_local.heads import PoseNet


def load_model(path,device):
    ck = torch.load(path,map_location='cpu',weights_only=False)
    c = ck['config']
    m = PoseNet(arm=ck['arm'],channels=c['channels'],width=c['width'],starts=c['starts'],
                thetas=c['thetas'],rhos=c['rhos'],modes=c['modes'])
    m.load_state_dict(ck['state_dict'],strict=True)
    return m.to(device).eval(),ck


def lines_from_points(uv):
    u,v = torch.tensor(G.EDGES,device=uv.device).T
    a,b = uv[:,u],uv[:,v]
    d = b-a
    n = torch.stack((-d[...,1],d[...,0]),-1)
    length = torch.linalg.vector_norm(n,dim=-1)
    n = n/length.clamp_min(1e-6)[...,None]
    return torch.cat((n,-(n*a).sum(-1,keepdim=True)),-1)[:,:,None],length>1e-6


def run_solver(O,o,lines,mass,*,R0=None,t0=None,point_mask=None,iters=5):
    rr = o['R0'] if R0 is None else R0
    tt = o['t0'] if t0 is None else t0
    pm = O['point_mask'] if point_mask is None else point_mask
    B,N = rr.shape[:2]
    rep = lambda x:x[:,None].expand(B,N,*x.shape[1:]).reshape(B*N,*x.shape[1:])
    flat = lambda x:x.reshape(B*N,*x.shape[2:])
    diag = torch.linalg.vector_norm(O['image_hw'],dim=-1)
    R,t = S.refine(rep(O['K']),flat(rr),flat(tt),rep(O['dims']),rep(o['points']),rep(pm),
        rep(lines) if lines is not None else None,rep(mass) if lines is not None else None,
        rep(O['line_mask']) if lines is not None else None,rep(diag),iters=iters)
    R,t = R.reshape(B,N,3,3),t.reshape(B,N,3)
    rb,tb,e,best,ok = S.select(O['K'],R,t,O['dims'],o['points'],pm,lines,mass,O['line_mask'],diag)
    return dict(R=rb,t=tb,valid=ok,Rs=R,ts=t,selected=best,energy=e)


def metrics(O,sup,orders,pred):
    B,N = pred['Rs'].shape[:2]
    X = G.corners(O['dims'])
    gtuv = G.project(O['K'],sup['R'],sup['t'],X)[0]
    diag = torch.linalg.vector_norm(O['image_hw'],dim=-1)
    values=[];px=[];valid=[]
    rep=lambda x:x[:,None].expand(B,N,*x.shape[1:]).reshape(B*N,*x.shape[1:])
    for g in (0,1):
        R=pred['Rs']@G.yaw(2*g,device=X.device,dtype=X.dtype).T
        uv,z=G.project(rep(O['K']),R.reshape(B*N,3,3),pred['ts'].reshape(B*N,3),rep(X))
        uv=uv.reshape(B,N,9,2);z=z.reshape(B,N,9)
        err=torch.linalg.vector_norm(uv[:,:,:8]-gtuv[:,None,:8],dim=-1)
        ok=(z>0).all(-1)&torch.isfinite(err).all(-1)
        norm=(err/diag[:,None,None]).clamp(max=1.).mean(-1)
        norm=torch.where(ok,norm,torch.ones_like(norm))
        if g: norm=torch.where((orders==2)[:,None],norm,torch.full_like(norm,2.))
        values.append(norm);px.append(err.mean(-1));valid.append(ok)
    vv=torch.stack(values,-1);gi=vv.argmin(-1)
    norm=vv.gather(-1,gi[...,None])[...,0]
    raw=torch.stack(px,-1).gather(-1,gi[...,None])[...,0]
    ok=torch.stack(valid,-1).gather(-1,gi[...,None])[...,0]
    idx=torch.arange(B,device=X.device);chosen=pred['selected']
    return dict(normalized=norm[idx,chosen],raw_px=raw[idx,chosen],
                valid=ok[idx,chosen]&pred['valid'],oracle_best_start=norm.min(-1).values)


def choose_oracle_symmetry(O,sup,orders,points):
    # A single object-wide C2 element. GT-assisted and labelled as such.
    original=sup['points'];perm=G.symmetry_permutations(2)[1]
    alternate=original[:,perm]
    err0=(points[:,:8]-original[:,:8]).square().sum((-1,-2))
    err1=(points[:,:8]-alternate[:,:8]).square().sum((-1,-2))
    flip=(orders==2)&(err1<err0)
    uv=torch.where(flip[:,None,None],alternate,original)
    R=torch.where(flip[:,None,None],sup['R']@G.yaw(2,device=points.device,dtype=points.dtype).T,sup['R'])
    return uv,R,flip


def stats(rows):
    norm=np.array([x['normalized'] for x in rows]);px=np.array([x['raw_px'] for x in rows if x['valid']])
    return dict(n=len(rows),primary=float(norm.mean()),coverage=float(np.mean([x['valid'] for x in rows])),
        raw_mean=float(px.mean()) if len(px) else None,raw_median=float(np.median(px)) if len(px) else None,
        raw_p90=float(np.quantile(px,.9)) if len(px) else None,
        oracle_best_start_primary=float(np.mean([x['oracle_best_start'] for x in rows])))


def main(args):
    torch.set_num_threads(2)
    run=Path(args.run);export=ROOT/'data/pallet/results/pallet_point_line_v5_local/export'
    data=Split(export/'calibration.json')
    output=run/'DIAGNOSTICS.json'
    if output.exists():raise SystemExit('diagnostics already exist')
    cells={};line_rows=[];view_rows={};bindings={}
    # Predict each arm before reading this batch's supervision; interventions
    # occur after network inference. Synthetic calibration only.
    with torch.inference_mode():
        for arm in ('direct','hough'):
            for seed in (1,2,3):
                path=run/'cache_heads'/f'{arm}_seed{seed}'/'checkpoint_final.pt'
                bindings[str(path)]=sha(path)
                model,ck=load_model(path,args.device)
                ckey=f'{arm}_seed{seed}';results={}
                for first in range(0,len(data.records),8):
                    indices=list(range(first,min(first+8,len(data.records))))
                    O,_,orders=data.batch(indices,args.device,targets=False)
                    o=model(O)
                    _,sup,_=data.batch(indices,args.device,targets=True)
                    uv,Rgt,flip=choose_oracle_symmetry(O,sup,orders,o['points'])
                    oracle,mask=lines_from_points(uv)
                    assert bool(mask.all())
                    mass=torch.ones_like(oracle[...,0])
                    nearR=(G.exp_so3(torch.tensor([[.03,-.04,.02]],device=args.device)).expand(len(indices),3,3)@Rgt)[:,None]
                    nearT=(sup['t']+torch.tensor([[.01,-.01,.02]],device=args.device))[:,None]
                    sources={'none':(None,None),'predicted':(o['lines'],o['mass']),
                             'shuffled':(o['lines'].roll(1,1),o['mass'].roll(1,1)),
                             'oracle':(oracle,mass)}
                    for init in ('actual','near_gt'):
                        for source,(lines,weights) in sources.items():
                            key=f'{init}/{source}'
                            p=run_solver(O,o,lines,weights,
                                R0=nearR if init=='near_gt' else None,
                                t0=nearT if init=='near_gt' else None)
                            m=metrics(O,sup,orders,p)
                            rowlist=results.setdefault(key,[])
                            for j,i in enumerate(indices):
                                rowlist.append(dict(id=data.records[i]['id'],normalized=float(m['normalized'][j]),
                                    raw_px=float(m['raw_px'][j]) if bool(m['valid'][j]) else None,
                                    valid=bool(m['valid'][j]),oracle_best_start=float(m['oracle_best_start'][j])))
                    # More iterations only diagnose finite-iteration/local basin failure.
                    p=run_solver(O,o,oracle,mass,iters=25);m=metrics(O,sup,orders,p)
                    for j,i in enumerate(indices):
                        results.setdefault('actual/oracle_25iters',[]).append(dict(id=data.records[i]['id'],
                            normalized=float(m['normalized'][j]),raw_px=float(m['raw_px'][j]) if bool(m['valid'][j]) else None,
                            valid=bool(m['valid'][j]),oracle_best_start=float(m['oracle_best_start'][j])))
                    u,v=torch.tensor(G.EDGES,device=args.device).T
                    n=o['lines'][...,:2];cc=o['lines'][...,2]
                    norm=torch.linalg.vector_norm(n,dim=-1).clamp_min(1e-8)
                    dist=(((uv[:,u,None]*n).sum(-1)+cc).abs()+((uv[:,v,None]*n).sum(-1)+cc).abs())/(2*norm)
                    top=o['mass'].argmax(-1);topdist=dist.gather(-1,top[...,None])[...,0]
                    lengths=torch.linalg.vector_norm(uv[:,u]-uv[:,v],dim=-1)
                    for j,i in enumerate(indices):
                        fid=data.records[i]['id']
                        cam=-(sup['R'][j].T@sup['t'][j]);az=math.degrees(math.atan2(abs(float(cam[0])),abs(float(cam[2]))))
                        oblique=min(az,90-az)
                        el=math.degrees(math.atan2(abs(float(cam[1])),float(torch.linalg.vector_norm(cam[[0,2]]))))
                        view_rows[fid]=dict(obliqueness_deg=oblique,elevation_deg=el,
                            view='near_front' if oblique<=10 else 'oblique',height='low' if el<=15 else 'high')
                        for edge in range(12):
                            line_rows.append(dict(cell=ckey,id=fid,edge=edge,kind='height' if edge>=8 else 'horizontal',
                                length_px=float(lengths[j,edge]),top1_normal_error_px=float(topdist[j,edge]),
                                oracle_best_mode_error_px=float(dist[j,edge].min()),oracle_global_c2=bool(flip[j])))
                cells[ckey]=dict(summary={k:stats(v) for k,v in results.items()},per_frame=results)
                print('diagnostic',ckey,{k:round(v['primary'],5) for k,v in cells[ckey]['summary'].items()},flush=True)
    out=dict(complete=True,split='calibration',n_frames=len(data.records),cells=cells,
        line_metrics=line_rows,views=view_rows,checkpoint_sha256=bindings,
        scope='Frozen-head causal diagnostics on reused synthetic calibration. Oracle lines, oracle '
              'symmetry choice, near-GT starts and best-start metrics are privileged diagnostics, not predictions. '
              'No training updates, no real images, no final-test data.',
        constants=dict(near_gt_rotation_increment=[.03,-.04,.02],near_gt_translation_increment_m=[.01,-.01,.02],
                       near_front_degrees=10,low_elevation_degrees=15,role_shuffle='cyclic shift by one edge',
                       normal_solver_iters=5,diagnostic_extended_iters=25))
    output.write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--device',default='cpu')
    main(ap.parse_args())
