from common import *
from model import SelectorResidual,PERMS
from evaluate import score,stats,ci,compute_perm_v4
import torch,numpy as np

def main():
    folder=OUT/'foreground_matched';assert (folder/'TRAIN_COMPLETE.json').exists();torch.set_num_threads(2)
    rows=read(folder/'test_records.json');z=torch.load(folder/'test.pt',weights_only=True)
    old=read(PREVIOUS/'RENDER_COMPLETE.json')['rows']+read(OUT/'RENDER_COMPLETE.json')['rows']+read(OUT/'mean_risk_followup'/'RENDER_COMPLETE.json')['rows']
    assert not {r['sha256'] for r in rows}&{r['sha256'] for r in old}
    pred={'stock':z['stock'].numpy()};vis={'stock':z['visibility'].numpy()};choice={}
    sign=np.array([[-1,1,1],[1,1,1],[1,-1,1],[-1,-1,1],[-1,1,-1],[1,1,-1],[1,-1,-1],[-1,-1,-1]])[:,[0,2,1]]
    canonical=pred['stock'].copy();vv=vis['stock'].copy()
    for i,p in enumerate(canonical):
        perm=compute_perm_v4(sign,p)
        if perm is not None:canonical[i]=p[perm];vv[i]=vv[i,perm]
    pred['canonical']=canonical;vis['canonical']=vv
    replay=[]
    for seed in [1,2,3]:
        for kind in ['unmatched_training','direct','selected']:
            m=SelectorResidual('A_direct' if kind=='direct' else 'C_select_residual',seed)
            path=OUT/'mean_risk_followup'/'models'/f'seed{seed}.pt' if kind=='unmatched_training' else folder/'models'/f'{kind}_s{seed}.pt'
            m.load_state_dict(torch.load(path,weights_only=True));m.eval()
            with torch.no_grad():o=m(z)
            tag=f'{kind}_s{seed}';pred[tag]=o['points'].numpy();vis[tag]=o['visibility'].numpy();choice[tag]=o['choice'].numpy()
            with torch.no_grad():replay.append(float(np.abs(m(z)['points'].numpy()-pred[tag]).max()))
    scored={k:score(v,vis[k],rows) for k,v in pred.items()};results={}
    for name,rr in scored.items():
        results[name]={}
        for pop,seen in [('seen',True),('unseen',False)]:
            sub=[r for r in rr if r['seen']==seen];results[name][pop]=dict(overall=stats(sub),conditions={s:stats([r for r in sub if r['state']==s]) for s in ['clean','panel','crop25']})
    mask=np.array([not r['seen'] for r in rows]);groups=[r['group'] for r in rows if not r['seen']]
    def a(kind):return np.array([[r['normalized'] for r in scored[f'{kind}_s{s}']] for s in [1,2,3]])[:,mask]
    stock=np.array([r['normalized'] for r in scored['stock']])[mask];selected=a('selected');direct=a('direct')
    interval=ci(selected.mean(0)-stock,groups);harm=[];conditions=True
    for seed in [1,2,3]:
        base=np.array([r['penalized_px'] for r in scored['stock']]);p=np.array([r['penalized_px'] for r in scored[f'selected_s{seed}']]);harm.append(int(((base<=10)&(p>20)).sum()))
        for pop in ['seen','unseen']:
            for state in ['clean','panel','crop25']:conditions &= results[f'selected_s{seed}'][pop]['conditions'][state]['primary']<=results['stock'][pop]['conditions'][state]['primary']*1.02
    checks=dict(all_seed_vs_stock2pct=bool(((1-selected.mean(1)/stock.mean())>=.02).all()),all_seed_vs_direct2pct=bool(((1-selected.mean(1)/direct.mean(1))>=.02).all()),
        CI_upper_below_zero=interval['high']<0,no_new_good_to_bad=not any(harm),condition_nonharm=bool(conditions))
    gt=np.array([r['gt'] for r in rows]);candidate=z['stock'].numpy()[:,np.array(PERMS),:][:,:,:8];e=np.linalg.norm(candidate-gt[:,None],axis=-1).mean(-1);diag={}
    for name,c in choice.items():
        matched=np.array([r['matched'] and not r['seen'] for r in scored[name]]);change=e[:,0]-e[np.arange(len(rows)),c]
        diag[name]=dict(switches=int(((c!=0)&matched).sum()),useful=int(((c!=0)&(change>0)&matched).sum()),harmful=int(((c!=0)&(change<0)&matched).sum()),
            oracle_mean_px=float(e.min(-1)[matched].mean()),large_fixable=int(((e[:,0]-e.min(-1)>20)&matched).sum()))
    np.savez_compressed(folder/'predictions.npz',**pred);np.savez_compressed(folder/'choices.npz',**choice);write(folder/'PER_FRAME.json',scored)
    write(folder/'SUMMARY.json',dict(results=results,selector=diag,checks=checks,decision='STABLE_MATCHED_SUPERVISION_SUPPORT' if all(checks.values()) else 'NO_STABLE_MATCHED_SUPERVISION_SUPPORT',
        selected_vs_stock_gain_per_seed=(1-selected.mean(1)/stock.mean()).tolist(),selected_vs_direct_gain_per_seed=(1-selected.mean(1)/direct.mean(1)).tolist(),
        selected_minus_stock_CI=interval,selected_minus_direct_CI=ci((selected-direct).mean(0),groups),
        selected_minus_unmatched_training_CI=ci((selected-a('unmatched_training')).mean(0),groups),
        new_good_to_bad_all_assets=harm,new_images_disjoint=True,model_replay_max_px=max(replay)))
    print(json.dumps({k:v for k,v in read(folder/'SUMMARY.json').items() if k not in ['results','selector']},indent=2))
if __name__=='__main__':main()
