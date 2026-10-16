from common import *
from model import Residual
import sys,time,importlib.util
import torch,numpy as np,cv2
sys.path.insert(0,str(KIT.parent/'pallet_dimension_symmetry_v1'))
spec=importlib.util.spec_from_file_location('dimsym_translation_diagnostic',KIT.parent/'pallet_dimension_symmetry_v1/evaluate.py')
prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior);old_pnp=prior.pnp

def iou(a,b,hw):
    a=np.array(a).copy();b=np.array(b).copy();h,w=hw
    for v in [a,b]:v[[0,2]]=v[[0,2]].clip(0,w);v[[1,3]]=v[[1,3]].clip(0,h)
    inter=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
    return float(inter/max(1e-9,np.maximum(0,a[2:]-a[:2]).prod()+np.maximum(0,b[2:]-b[:2]).prod()-inter))

def pose(xy,vis,r):
    return old_pnp(xy,vis,np.asarray(r['dims_xyz'])[[0,2,1]],r['K'])

def score(pred,vis,rows):
    output=[]
    for xy,v,r in zip(pred,vis,rows):
        e=np.linalg.norm(xy[:8]-np.asarray(r['gt']),axis=-1)
        diag=float(np.linalg.norm(r['raw_hw']))
        match=r['confidence']>=.25 and iou(r['pred_box'],r['box'],r['raw_hw'])>=.5
        mean=float(e.mean()) if match else diag
        pp=pose(xy,v,r) if match else None
        output.append(dict(id=r['id'],asset=r['asset'],seen=r['seen'],state=r['state'],group=r['group'],
            matched=match,normalized=min(1.,mean/diag),penalized_px=mean,matched_px=float(e.mean()) if match else None,
            inside_px=float(e[np.array(r['inside'])].mean()) if match and any(r['inside']) else None,
            outside_px=float(e[~np.array(r['inside'])].mean()) if match and not all(r['inside']) else None,
            translation_cm=float(np.linalg.norm(np.asarray(pp['t'])-r['t'])*100) if pp else None))
    return output

def summary(rows):
    matched=[r['matched_px'] for r in rows if r['matched']];trans=[r['translation_cm'] for r in rows if r['translation_cm'] is not None]
    return dict(n=len(rows),matched=len(matched),primary=float(np.mean([r['normalized'] for r in rows])),
        penalized_mean_px=float(np.mean([r['penalized_px'] for r in rows])),
        matched_mean_px=float(np.mean(matched)) if matched else None,
        matched_p90_px=float(np.quantile(matched,.9)) if matched else None,
        large_errors_gt20_including_misses=sum(r['penalized_px']>20 for r in rows),
        translation_n=len(trans),translation_median_cm=float(np.median(trans)) if trans else None)

def bootstrap(delta,groups):
    g=sorted(set(groups));x=np.array([np.mean(delta[np.array(groups)==v]) for v in g]);rng=np.random.default_rng(451)
    values=x[rng.integers(0,len(x),(10000,len(x)))].mean(1)
    return dict(mean=float(delta.mean()),low=float(np.quantile(values,.025)),high=float(np.quantile(values,.975)),
        clusters=len(x),unit='shared randomized scene group across views, states and assets; seed-average',replicates=10000)

def main():
    torch.set_num_threads(2);cv2.setNumThreads(1);cfg=register()
    assert (OUT/'TRAIN_COMPLETE.json').exists()
    for n,h in read(OUT/'TRAIN_LOCK.json')['sources'].items():assert sha(KIT/n)==h,n
    z=torch.load(OUT/'test.pt',weights_only=True);rows=read(OUT/'test_records.json')
    gt_pose_errors=[]
    for r in rows:
        p=pose(np.asarray(r['gt']),np.ones(8),r)
        gt_pose_errors.append(float(np.linalg.norm(np.asarray(p['t'])-r['t'])) if p else 1e6)
    write(OUT/'POSE_GT_AUDIT.json',dict(max_translation_m=max(gt_pose_errors),passed=max(gt_pose_errors)<.001,
         note='Translation only. Successful GT replay does not disambiguate semantic rotation.'))
    assert max(gt_pose_errors)<.001,max(gt_pose_errors)
    allrows={'stock':score(z['stock'].numpy(),z['visibility'].numpy(),rows)};predictions={'stock':z['stock'].numpy()};latency={}
    for seed in cfg['seeds']:
        for arm in ARMS:
            name=f'{arm}_s{seed}';m=Residual(arm,seed)
            m.load_state_dict(torch.load(OUT/'models'/name/'final.pt',weights_only=True));m.eval()
            variants=['correct','shuffle','noise5'] if arm!='A_points' else ['correct']
            for variant in variants:
                zz=dict(z)
                if variant=='shuffle':
                    # Roll among distinct scene groups; keep physical solver input in rows correct.
                    zz['dims']=z['dims'].roll(12,0)
                if variant=='noise5':
                    noise=np.random.default_rng(451).normal(0,.05,(len(rows),3))
                    zz['dims']=z['dims']*torch.tensor(np.exp(noise),dtype=torch.float32)
                with torch.no_grad():xy=m(zz).numpy()
                tag=name if variant=='correct' else name+'_'+variant
                allrows[tag]=score(xy,z['visibility'].numpy(),rows)
                if variant=='correct':predictions[tag]=xy
            small={k:v[:1] for k,v in z.items()}
            with torch.no_grad():
                for _ in range(10):m(small)
                ts=[]
                for _ in range(100):
                    t=time.perf_counter();m(small);ts.append((time.perf_counter()-t)*1000)
            latency[name]=dict(cpu_branch_batch1_p50_ms=float(np.median(ts)),cpu_branch_batch1_p95_ms=float(np.quantile(ts,.95)),cpu_threads=2)
    results={}
    for name,rr in allrows.items():
        results[name]={}
        for pop,select in [('seen',lambda r:r['seen']),('unseen',lambda r:not r['seen'])]:
            selected=[r for r in rr if select(r)]
            results[name][pop]=dict(overall=summary(selected),
                conditions={s:summary([r for r in selected if r['state']==s]) for s in cfg['conditions']},
                assets={a:summary([r for r in selected if r['asset']==a]) for a in sorted({r['asset'] for r in selected})})
    ix=np.array([not r['seen'] for r in rows]);groups=[r['group'] for r in rows if not r['seen']]
    def arr(arm):
        return np.array([[r['normalized'] for r in allrows[f'{arm}_s{s}']] for s in cfg['seeds']])[:,ix]
    C=arr('C_geometry');B=arr('B_concat');A=arr('A_points');D=arr('D_no_relations')
    stock=np.array([r['normalized'] for r in allrows['stock']])[ix]
    ci=bootstrap((C-B).mean(0),groups)
    noharm=True;cells=[]
    for seed in cfg['seeds']:
        cr=allrows[f'C_geometry_s{seed}'];br=allrows[f'B_concat_s{seed}']
        for asset in ASSETS[2:]:
            for state in cfg['conditions']:
                inds=[i for i,r in enumerate(rows) if r['asset']==asset and r['state']==state]
                c=np.mean([cr[i]['normalized'] for i in inds]);b=np.mean([br[i]['normalized'] for i in inds])
                noharm &= c<=b*1.02;cells.append(dict(seed=seed,asset=asset,state=state,gain=1-c/b))
    checks=dict(all_seed_C_over_B_2pct=bool(((1-C.mean(1)/B.mean(1))>=.02).all()),
        paired_cluster_CI_upper_below_zero=ci['high']<0,
        all_seed_C_beats_A_and_stock=bool(((C.mean(1)<A.mean(1))&(C.mean(1)<stock.mean())).all()),
        each_asset_condition_no_more_than_2pct_worse=bool(noharm))
    write(OUT/'PER_FRAME.json',allrows);np.savez_compressed(OUT/'predictions.npz',**predictions)
    write(OUT/'SUMMARY.json',dict(results=results,latency=latency,checks=checks,
        decision='SUPPORTS_BOUNDED_GEOMETRY_SCREEN' if all(checks.values()) else 'GEOMETRY_NOT_STABLY_SUPPORTED',
        C_over_B_gain_per_seed=(1-C.mean(1)/B.mean(1)).tolist(),C_minus_B_bootstrap=ci,
        C_minus_D_bootstrap=bootstrap((C-D).mean(0),groups),B_minus_A_bootstrap=bootstrap((B-A).mean(0),groups),
        per_asset_condition_gain_C_over_B=cells,
        pose_metric='GT-free two cuboid-correspondence PnP translation diagnostic only; semantic rotation not established',
        real_evaluation=False,new_unseen_mesh_count=2,trainable_YOLO_parameters=0))
    print(json.dumps({k:v for k,v in read(OUT/'SUMMARY.json').items() if k not in ['results','latency','per_asset_condition_gain_C_over_B']},indent=2))
if __name__=='__main__':main()
