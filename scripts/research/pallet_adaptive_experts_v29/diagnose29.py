"""Post-evaluation diagnosis only: frozen router risk versus actual expert error."""
from context29 import *
from model29 import Router
from train25 import dataset
from eval27 import population,conditions

def diagnose(candidates,risk,gt,valid,kind,diag,source_mask):
    err=np.linalg.norm(candidates-gt[:,:,None],axis=-1);rows={}
    for name,k in (('all',None),('intact_proxy',0),('hidden',1),('outside',2)):
        mask=valid&source_mask[:,None]
        if k is not None:mask&=kind==k
        if not mask.any():continue
        e=err[mask];r=risk[mask];best=e.argmin(-1);chosen=r.argmin(-1)
        rows[name]=dict(points=int(mask.sum()),actual_expert_px=e.mean(0).tolist(),predicted_expert_normalized_error=r.mean(0).tolist(),actual_expert_normalized_error=(err/diag[:,None,None])[mask].mean(0).tolist(),line_beats_best_point_fraction=float((e[:,2:].min(-1)<e[:,:2].min(-1)-1).mean()),line_chosen_fraction=float((chosen>=2).mean()),best_expert_accuracy=float((chosen==best).mean()),line_added_oracle_gain_px=float((e[:,:2].min(-1)-e.min(-1)).mean()))
    return rows

def main():
    setup();rows=[]
    for seed in SEEDS:
        O,r,gt,valid,ids,kind,_=dataset(seed);del r;O.pop('edge');plan=read(RUN/f'PLAN_seed{seed}.json');assert ids==plan['ids']
        cache=torch.load(RUN/'training_cache'/f'seed{seed}.pt',map_location='cpu',weights_only=True)
        model=Router(seed,'adaptive').cuda().eval();model.load_state_dict(torch.load(RUN/'models'/f'adaptive_seed{seed}'/'final.pt',map_location='cpu',weights_only=True)['state_dict']);risks=[]
        with torch.no_grad():
            for start in range(0,len(ids),32):
                sl=slice(start,start+32);obs={k:v[sl].cuda() for k,v in O.items()};risks.append(model(obs,cache['candidates'][sl].cuda(),cache['quality'][sl].cuda())['risk'].cpu())
        risk=torch.cat(risks).numpy();diag=(O['box'][:,2:]-O['box'][:,:2]).norm(dim=-1).numpy()
        for part,key in (('expert_train','expert_sources'),('router_train','router_sources')):
            allowed=set(plan[key]);mask=np.array([x.split('::')[0] in allowed for x in ids])
            rows.append(dict(seed=seed,population=part,condition='all_training_conditions',stats=diagnose(cache['candidates'].numpy(),risk,gt.numpy(),valid.numpy(),kind.numpy(),diag,mask)))
        del O,gt,valid,kind,cache,model
        for split in EVAL_SPLITS:
            for c in conditions(split):
                m,ts=population(split,c);b=np.load(RUN/'predictions'/split/c/f'bundle_seed{seed}.npz');p=np.load(RUN/'predictions'/split/c/f'adaptive_seed{seed}.npz')
                rows.append(dict(seed=seed,population=split,condition=c,stats=diagnose(b['candidates'],p['risk'],np.array([t['points'] for t in ts]),np.array([t['valid'] for t in ts]),C.P.target_type(ts).numpy(),b['diag'],b['detected'])))
    write(RUN/'RISK_DIAGNOSIS.json',dict(post_evaluation=True,new_training=False,rows=rows,limits='Paired synthetic/real populations differ; risk ordering reversal is descriptive evidence, not proof of one causal domain factor. State labels are stress proxies.'))
    print('RISK_DIAGNOSIS29 COMPLETE',flush=True)

if __name__=='__main__':main()
