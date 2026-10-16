from common import *
from model import SelectorResidual,PERMS
from evaluate import score,stats,ci,compute_perm_v4
import torch,numpy as np

def main():
    folder=OUT/'mean_risk_followup';assert (folder/'TRAIN_COMPLETE.json').exists();torch.set_num_threads(2)
    rows=read(folder/'test_records.json');z=torch.load(folder/'test.pt',weights_only=True)
    previous_rows=read(PREVIOUS/'RENDER_COMPLETE.json')['rows']+read(OUT/'RENDER_COMPLETE.json')['rows']
    assert not {r['sha256'] for r in rows}&{r['sha256'] for r in previous_rows}
    predictions={'stock':z['stock'].numpy()};visibility={'stock':z['visibility'].numpy()};choices={}
    sign=np.array([[-1,1,1],[1,1,1],[1,-1,1],[-1,-1,1],[-1,1,-1],[1,1,-1],[1,-1,-1],[-1,-1,-1]])[:,[0,2,1]]
    canonical=predictions['stock'].copy();vis=visibility['stock'].copy()
    for i,xy in enumerate(canonical):
        p=compute_perm_v4(sign,xy)
        if p is not None:canonical[i]=xy[p];vis[i]=vis[i,p]
    predictions['canonical']=canonical;visibility['canonical']=vis
    frozen_errors=[]
    for seed in [1,2,3]:
        for kind in ['direct','robust','mean']:
            arm='A_direct' if kind=='direct' else 'C_select_residual';m=SelectorResidual(arm,seed)
            path=folder/'models'/f'seed{seed}.pt' if kind=='mean' else OUT/'models'/f'{arm}_s{seed}'/'final.pt'
            m.load_state_dict(torch.load(path,weights_only=True));m.eval()
            with torch.no_grad():o=m(z)
            tag=f'{kind}_s{seed}';predictions[tag]=o['points'].numpy();visibility[tag]=o['visibility'].numpy();choices[tag]=o['choice'].numpy()
            if kind=='mean':
                old=torch.load(OUT/'models'/f'C_select_residual_s{seed}'/'final.pt',weights_only=True)
                frozen_errors.append(max(float((v-old[k]).abs().max()) for k,v in m.state_dict().items() if not k.startswith('risk')))
    scored={k:score(v,visibility[k],rows) for k,v in predictions.items()};results={}
    for name,rr in scored.items():
        results[name]={}
        for pop,seen in [('seen',True),('unseen',False)]:
            sub=[r for r in rr if r['seen']==seen]
            results[name][pop]=dict(overall=stats(sub),conditions={state:stats([r for r in sub if r['state']==state]) for state in ['clean','panel','crop25']})
    mask=np.array([not r['seen'] for r in rows]);groups=[r['group'] for r in rows if not r['seen']]
    def a(name):return np.array([[r['normalized'] for r in scored[f'{name}_s{s}']] for s in [1,2,3]])[:,mask]
    stock=np.array([r['normalized'] for r in scored['stock']])[mask];mean=a('mean');robust=a('robust')
    improvement=1-mean.mean(1)/stock.mean();interval=ci(mean.mean(0)-stock,groups)
    base=np.array([r['penalized_px'] for r in scored['stock']]);harm=[];conditions=True
    for seed in [1,2,3]:
        p=np.array([r['penalized_px'] for r in scored[f'mean_s{seed}']]);harm.append(int(((base<=10)&(p>20)).sum()))
        for pop in ['seen','unseen']:
            for state in ['clean','panel','crop25']:
                conditions &= results[f'mean_s{seed}'][pop]['conditions'][state]['primary']<=results['stock'][pop]['conditions'][state]['primary']*1.02
    checks=dict(all_seed_gain2pct=bool((improvement>=.02).all()),CI_upper_below_zero=interval['high']<0,
        no_new_large_error_from_good=not any(harm),condition_nonharm=bool(conditions))
    GT=np.array([r['gt'] for r in rows]);candidate=z['stock'].numpy()[:,np.array(PERMS),:][:,:,:8]
    errors=np.linalg.norm(candidate-GT[:,None],axis=-1).mean(-1);diag={}
    for name,choice in choices.items():
        matched=np.array([r['matched'] and not r['seen'] for r in scored[name]])
        delta=errors[:,0]-errors[np.arange(len(rows)),choice]
        diag[name]=dict(switches=int(((choice!=0)&matched).sum()),useful_switches=int(((choice!=0)&(delta>0)&matched).sum()),
            harmful_switches=int(((choice!=0)&(delta<0)&matched).sum()),role_oracle_mean_px=float(errors.min(-1)[matched].mean()))
    assert max(frozen_errors)==0
    np.savez_compressed(folder/'predictions.npz',**predictions);np.savez_compressed(folder/'choices.npz',**choices)
    write(folder/'PER_FRAME.json',scored)
    write(folder/'SUMMARY.json',dict(results=results,selector=diag,checks=checks,decision='STABLE_MEAN_RISK_SUPPORT' if all(checks.values()) else 'MEAN_RISK_NOT_STABLY_SUPPORTED',
        mean_vs_stock_gain_per_seed=improvement.tolist(),mean_minus_stock_CI=interval,mean_minus_robust_CI=ci((mean-robust).mean(0),groups),
        new_large_errors_all_assets=harm,independent_scene_hashes=True,residual_weights_max_change=max(frozen_errors)))
    print(json.dumps({k:v for k,v in read(folder/'SUMMARY.json').items() if k not in ['results','selector']},indent=2))
if __name__=='__main__':main()
