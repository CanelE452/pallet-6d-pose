from common import *
from model import SelectorResidual,PERMS
import torch,numpy as np,cv2,sys,importlib.util,time

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
sys.path.insert(0,str(KIT.parent/'pallet_dimension_symmetry_v1'))
old_pnp=module('previous_translation_metric',KIT.parent/'pallet_dimension_symmetry_v1/evaluate.py').pnp
PreviousGeometry=module('previous_geometry_model',KIT.parent/'pallet_unseen_dimensions_v1/model.py').Residual
sys.path.insert(0,str(ROOT))
from scripts.annotate.convert_to_camera_facing_v4 import compute_perm_v4

def iou(a,b,hw):
    a=np.array(a).copy();b=np.array(b).copy();h,w=hw
    for v in [a,b]:v[[0,2]]=v[[0,2]].clip(0,w);v[[1,3]]=v[[1,3]].clip(0,h)
    inter=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
    return float(inter/max(1e-9,np.maximum(0,a[2:]-a[:2]).prod()+np.maximum(0,b[2:]-b[:2]).prod()-inter))
def score(xy,vis,rows):
    result=[]
    for p,v,r in zip(xy,vis,rows):
        error=np.linalg.norm(p[:8]-np.array(r['gt']),axis=-1);diag=float(np.linalg.norm(r['raw_hw']))
        matched=r['confidence']>=.25 and iou(r['pred_box'],r['box'],r['raw_hw'])>=.5
        pose=old_pnp(p,v,np.asarray(r['dims_xyz'])[[0,2,1]],r['K']) if matched else None
        mean=float(error.mean()) if matched else diag
        result.append(dict(id=r['id'],group=r['group'],seen=r['seen'],asset=r['asset'],state=r['state'],matched=matched,
            normalized=min(1.,mean/diag),penalized_px=mean,matched_px=float(error.mean()) if matched else None,
            translation_cm=float(np.linalg.norm(np.asarray(pose['t'])-r['t'])*100) if pose else None))
    return result
def stats(rows):
    p=[r['matched_px'] for r in rows if r['matched']];t=[r['translation_cm'] for r in rows if r['translation_cm'] is not None]
    return dict(n=len(rows),matched=len(p),primary=float(np.mean([r['normalized'] for r in rows])),
        matched_mean_px=float(np.mean(p)) if p else None,matched_median_px=float(np.median(p)) if p else None,
        matched_p90_px=float(np.quantile(p,.9)) if p else None,large_gt20_including_misses=sum(r['penalized_px']>20 for r in rows),
        translation_n=len(t),translation_median_cm=float(np.median(t)) if t else None)
def ci(delta,groups):
    groups=np.array(groups);unique=sorted(set(groups));x=np.array([delta[groups==g].mean() for g in unique])
    rng=np.random.default_rng(652);v=x[rng.integers(0,len(x),(10000,len(x)))].mean(1)
    return dict(mean=float(delta.mean()),low=float(np.quantile(v,.025)),high=float(np.quantile(v,.975)),clusters=len(x),
        unit='paired scene groups across all views/states/meshes; seed-average first; conditional on these meshes')

def main():
    cfg=register();assert read(OUT/'PREFLIGHT.json')['PASS'];assert (OUT/'TRAIN_COMPLETE.json').exists()
    for n,h in read(OUT/'TRAIN_LOCK.json')['sources'].items():assert sha(KIT/n)==h,n
    torch.set_num_threads(2);cv2.setNumThreads(1);overall={};allframes={};allpred={};allchoices={};selector_stats={};latencies={}
    for population,folder in [('reused_dev',PREVIOUS),('fresh',OUT)]:
        rows=read(folder/'test_records.json');z=torch.load(folder/'test.pt',weights_only=True)
        pred={'stock':z['stock'].numpy()};visibility={'stock':z['visibility'].numpy()};choices={};riskvalues={}
        # Same observation-only canonicalization as prior diagnosis, frozen, no threshold tuning.
        sign=np.array([[-1,1,1],[1,1,1],[1,-1,1],[-1,-1,1],[-1,1,-1],[1,1,-1],[1,-1,-1],[-1,-1,-1]])[:,[0,2,1]]
        canonical=pred['stock'].copy();vv=visibility['stock'].copy()
        for i,xy in enumerate(canonical):
            p=compute_perm_v4(sign,xy[:8])
            if p is not None:canonical[i]=xy[p];vv[i]=vv[i,p]
        pred['canonical']=canonical;visibility['canonical']=vv
        for seed in cfg['seeds']:
            previous=PreviousGeometry('C_geometry',seed)
            previous.load_state_dict(torch.load(PREVIOUS/'models'/f'C_geometry_s{seed}'/'final.pt',weights_only=True));previous.eval()
            with torch.no_grad():pred[f'previous_geometry_s{seed}']=previous(z).numpy()
            visibility[f'previous_geometry_s{seed}']=visibility['stock']
            for arm in ARMS:
                name=f'{arm}_s{seed}';m=SelectorResidual(arm,seed)
                m.load_state_dict(torch.load(OUT/'models'/name/'final.pt',weights_only=True));m.eval()
                variants=['normal','no_selector','no_residual','shuffled_dims'] if arm=='C_select_residual' else ['normal']
                for variant in variants:
                    zz=dict(z)
                    if variant=='shuffled_dims':zz['dims']=z['dims'].roll(12,0)
                    with torch.no_grad():output=m(zz,mode=variant)
                    tag=name if variant=='normal' else name+'_'+variant
                    pred[tag]=output['points'].numpy();visibility[tag]=output['visibility'].numpy()
                    choices[tag]=output['choice'].numpy();riskvalues[tag]=(output['risk']*output['scale'][:,None]).numpy()
                if population=='fresh':
                    small={k:v[:1] for k,v in z.items()};times=[]
                    with torch.no_grad():
                        for _ in range(10):m(small)
                        for _ in range(100):
                            start=time.perf_counter();m(small);times.append((time.perf_counter()-start)*1000)
                    latencies[name]=dict(cpu_branch_p50_ms=float(np.median(times)),cpu_branch_p95_ms=float(np.quantile(times,.95)),threads=2)
        scored={name:score(p,visibility[name],rows) for name,p in pred.items()}
        summary={}
        for name,rr in scored.items():
            summary[name]={}
            for pop,seen in [('seen',True),('unseen',False)]:
                selected=[r for r in rr if r['seen']==seen]
                summary[name][pop]=dict(overall=stats(selected),conditions={s:stats([r for r in selected if r['state']==s]) for s in cfg['conditions']})
        diagnostics={};target=np.array([r['gt'] for r in rows]);candidates=z['stock'].numpy()[:,np.array(PERMS),:][:,:,:8]
        errors=np.linalg.norm(candidates-target[:,None],axis=-1).mean(-1)
        for name,choice in choices.items():
            mask=np.array([r['matched'] and not r['seen'] for r in scored[name]])
            before=errors[:,0];after=errors[np.arange(len(rows)),choice];best=errors.min(-1)
            good=(before<=10)&mask;newerror=np.array([r['penalized_px'] for r in scored[name]])
            diagnostics[name]=dict(matched_unseen=int(mask.sum()),switches=int(((choice!=0)&mask).sum()),
                mean_selection_regret_px=float((after-best)[mask].mean()),role_oracle_mean_px=float(best[mask].mean()),
                mean_pre_residual_selected_px=float(after[mask].mean()),
                useful_switches=int(((choice!=0)&(after<before)&mask).sum()),harmful_switches=int(((choice!=0)&(after>before)&mask).sum()),
                new_gt20_from_stock_le10=int((good&(newerror>20)).sum()),
                baseline_large_fixable_frames=int(((before-best>20)&mask).sum()),
                large_fixable_improved_gt20=int(((before-best>20)&(before-after>20)&mask).sum()))
        overall[population]=summary;allframes[population]=scored;selector_stats[population]=diagnostics
        allpred[population]=pred;allchoices[population]=choices
        np.savez_compressed(OUT/f'{population}_predictions.npz',**pred)
        np.savez_compressed(OUT/f'{population}_selector.npz',**{k:choices[k] for k in choices},**{k+'_risk':v for k,v in riskvalues.items()})
        write(OUT/f'{population}_records.json',rows)
    rows=read(OUT/'fresh_records.json');scored=allframes['fresh'];ix=np.array([not r['seen'] for r in rows]);groups=[r['group'] for r in rows if not r['seen']]
    def array(arm):return np.array([[r['normalized'] for r in scored[f'{arm}_s{s}']] for s in cfg['seeds']])[:,ix]
    C=array('C_select_residual');A=array('A_direct');stock=np.array([r['normalized'] for r in scored['stock']])[ix]
    interval=ci(C.mean(0)-stock,groups);conditions=True
    for seed in cfg['seeds']:
        for pop in ['seen','unseen']:
            for state in cfg['conditions']:
                c=overall['fresh'][f'C_select_residual_s{seed}'][pop]['conditions'][state]['primary']
                b=overall['fresh']['stock'][pop]['conditions'][state]['primary'];conditions &= c<=b*1.02
    harms=[]
    for seed in cfg['seeds']:
        base=np.array([r['penalized_px'] for r in scored['stock']]);cp=np.array([r['penalized_px'] for r in scored[f'C_select_residual_s{seed}']])
        harms.append(int(((base<=10)&(cp>20)).sum()))
    checks=dict(C_vs_stock_each_seed_2pct=bool(((1-C.mean(1)/stock.mean())>=.02).all()),
        C_vs_A_each_seed_2pct=bool(((1-C.mean(1)/A.mean(1))>=.02).all()),
        C_minus_stock_bootstrap_upper_below_zero=interval['high']<0,
        each_seen_unseen_condition_no_more_than_2pct_worse=bool(conditions),no_new_large_errors_from_good_frames=not any(harms))
    contrasts={}
    for lhs,rhs in [('C_select_residual','B_select_only'),('C_select_residual','D_select_residual_no_dims'),('B_select_only','A_direct')]:
        contrasts[lhs+' minus '+rhs]=ci((array(lhs)-array(rhs)).mean(0),groups)
    write(OUT/'PER_FRAME.json',allframes)
    write(OUT/'SUMMARY.json',dict(complete=True,results=overall,selector=selector_stats,latency=latencies,checks=checks,
        decision='STABLE_BOUNDED_SCREEN_SUPPORT' if all(checks.values()) else 'NO_STABLE_SCREEN_SUPPORT',
        C_over_stock_gain_per_seed=(1-C.mean(1)/stock.mean()).tolist(),C_minus_stock_bootstrap=interval,
        C_new_large_error_frames_all_assets=harms,contrasts=contrasts,
        fresh_scene_evaluation=True,new_real_evaluation=False,semantic_rotation_evaluation=False,backbone_updates=0))
    print(json.dumps({k:v for k,v in read(OUT/'SUMMARY.json').items() if k not in ['results','selector','latency']},indent=2))
if __name__=='__main__':main()
