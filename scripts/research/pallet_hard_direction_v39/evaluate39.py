from context39 import *
from sklearn.metrics import roc_auc_score,average_precision_score,precision_score,recall_score,f1_score
METHODS=('anchor','standard','balanced','hard_geometry','hard_appearance','point_gate','line_gate','GT_hard_geometry','GT_hard_appearance','GT_safe_hard_geometry','candidate_oracle')

def metric(E,R,M,groups):
    axes=tuple(range(1,E.ndim));count=M.sum(axis=axes);per=np.where(M,E,0).sum(axis=axes)/count;ref=np.where(M,R,0).sum(axis=axes)/count;names=np.unique(groups[M.any(axis=tuple([0]+list(range(2,M.ndim))))]);sums=[];counts=[]
    for g in names:
        mask=M.copy();mask[:,groups!=g]=False;sums.append(np.where(mask,E-R,0).sum(axis=axes));counts.append(mask.sum(axis=axes))
    ss=np.array(sums).T;cc=np.array(counts).T;rng=np.random.default_rng(39001);idx=rng.integers(0,len(names),(5000,len(names)));bs=ss[:,idx].sum(-1);bc=cc[:,idx].sum(-1);ok=(bc>0).all(0);samples=(bs[:,ok]/bc[:,ok]).mean(0)
    return dict(mean_px=float(per.mean()),per_seed_px=per.tolist(),per_seed_points=count.tolist(),gain_percent=((1-per/ref)*100).tolist(),difference=dict(mean=float((per-ref).mean()),ci95=np.quantile(samples,[.025,.975]).tolist(),bootstrap_draws=int(ok.sum()),groups=int((cc.sum(0)>0).sum())))

def masks(p,hard):
    part=partition(p)
    for split in ('fit','calibration','check'):
      for cond in ['all',*sorted(set(p['condition']))]:
        frame=part[split]&(True if cond=='all' else p['condition']==cond)
        for typ,mask in [('all_detected',np.ones_like(hard)),('hard',hard),('easy',~hard),('outside',np.broadcast_to(p['g']['outside'],hard.shape)),('hidden',np.broadcast_to(p['g']['hidden'],hard.shape))]:
            m=mask&frame[None,:,None]
            if (m.sum((1,2))>0).all():yield split+'/'+cond+'/'+typ,m

def main():
    assert read(RUN/'TRAINING_COMPLETE.json')['complete'];p=population();part=partition(p);g=p['g'];data={(arm,s):dict(np.load(RUN/'arrays'/f'{arm}_seed{s}.npz')) for arm in ARMS for s in SEEDS};hard=np.stack([data[('both',s)]['hard'] for s in SEEDS]);results={};comparisons={};direction={};hardness={};funnel={}
    for key,M in masks(p,hard):
        row={};comp={};dd={}
        for arm in ARMS:
            base=np.stack([np.linalg.norm(data[(arm,s)]['anchor']-g['gt'],axis=-1) for s in SEEDS]);row[arm]={}
            for mode in METHODS:
                E=np.stack([np.linalg.norm(data[(arm,s)]['anchor' if mode=='anchor' else 'points_'+mode]-g['gt'],axis=-1) for s in SEEDS]);r=metric(E,base,M,p['groups']);r['selected_points']=[0 if mode=='anchor' else int(((data[(arm,s)]['selected_'+mode]>=0)&M[s-1]).sum()) for s in SEEDS];r['selected_fraction']=[n/den for n,den in zip(r['selected_points'],r['per_seed_points'])];r['harmed_over1px']=[int(((E[i]-base[i]>1)&M[i]).sum()) for i in range(3)];row[arm][mode]=r
            ee=np.stack([np.linalg.norm(data[(arm,s)]['points_line_gate']-g['gt'],axis=-1) for s in SEEDS]);rr=np.stack([np.linalg.norm(data[(arm,s)]['points_point_gate']-g['gt'],axis=-1) for s in SEEDS]);comp[arm]=metric(ee,rr,M,p['groups'])
            if key.endswith('/hard') or key in ('fit/all/all_detected','check/all/all_detected'):
                dd[arm]={};target=np.stack([data[(arm,s)]['target'] for s in SEEDS]);scale=p['diag'][None,:,None,None];MM=np.broadcast_to(M[...,None],target.shape);standard=np.abs(np.stack([data[(arm,s)]['mu_standard'] for s in SEEDS])-target)*scale
                modes=['standard',*REGRESSORS]+(['oof_hard_geometry'] if key.startswith('fit/') else [])
                for mode in modes:
                    mu=np.stack([data[(arm,s)]['oof_hard_geometry' if mode=='oof_hard_geometry' else 'mu_'+mode] for s in SEEDS]);E=np.abs(mu-target)*scale;r=metric(E,standard,MM,p['groups']);active=MM&(np.abs(target)*scale>=1);r['sign_accuracy']=[float((np.sign(mu[i][active[i]])==np.sign(target[i][active[i]])).mean()) for i in range(3)];r['sign_samples']=[int(m.sum()) for m in active];r['zero_target_MAE_px']=[float(np.abs(target[i]*scale[0])[MM[i]].mean()) for i in range(3)];dd[arm][mode]=r
                eh=np.abs(np.stack([data[(arm,s)]['mu_hard_geometry'] for s in SEEDS])-target)*scale;ea=np.abs(np.stack([data[(arm,s)]['mu_hard_appearance'] for s in SEEDS])-target)*scale;dd[arm]['geometry_vs_appearance']=metric(eh,ea,MM,p['groups'])
        results[key]=dict(methods=row);comparisons[key]=comp
        if dd:direction[key]=dd
    for arm in ARMS:
      for seed in SEEDS:
        name=f'{arm}_seed{seed}';a=data[(arm,seed)];hardness[name]={};funnel[name]={};base=np.linalg.norm(a['anchor']-g['gt'],axis=-1);candidates=np.load(V37/'arrays'/f'v34_{arm}_seed{seed}.npz')['candidate'];ce=np.linalg.norm(candidates-g['gt'][:,:,None],axis=-1);useful=base[...,None]-ce>1
        for split in ('fit','calibration','check'):
            mask=np.broadcast_to(part[split][:,None],base.shape);gt=a['hard'][mask];hardness[name][split]={}
            for mode in GATES:
                score=a['hard_probability_'+mode][mask];predicted=a['hard_prediction_'+mode][mask];hardness[name][split][mode]=dict(points=int(mask.sum()),hard_points=int(gt.sum()),prevalence=float(gt.mean()),auroc=float(roc_auc_score(gt,score)),average_precision=float(average_precision_score(gt,score)),precision=float(precision_score(gt,predicted,zero_division=0)),recall=float(recall_score(gt,predicted,zero_division=0)),f1=float(f1_score(gt,predicted,zero_division=0)),predicted_hard=int(predicted.sum()))
            mm=mask&a['hard'];funnel[name][split]={}
            for mode in ('standard',*REGRESSORS):
                positive=a['gain_'+mode]>0;sel=a['selected_'+mode];err=np.linalg.norm(a['points_'+mode]-g['gt'],axis=-1);funnel[name][split][mode]=dict(hard_points=int(mm.sum()),possible_gain_over1px=int((mm&useful.any(-1)).sum()),useful_positive_predicted=int((mm&(useful&positive).any(-1)).sum()),selected_help_over1px=int((mm&(base-err>1)).sum()),selected_harm_over1px=int((mm&(err-base>1)).sum()),selected=int((mm&(sel>=0)).sum()))
    d=direction['check/all/hard']['both'];dgate=min(d['hard_geometry']['gain_percent'])>=10 and d['hard_geometry']['difference']['ci95'][1]<0;add=d['geometry_vs_appearance']['difference']['ci95'][1]<0;hgate=all(hardness[f'both_seed{s}']['check']['line']['precision']>=.5 and hardness[f'both_seed{s}']['check']['line']['recall']>=.5 for s in SEEDS);r=results['check/all/all_detected']['methods']['both']['line_gate'];gates=dict(hard_direction_improved=bool(dgate),line_direction_added=bool(add),hardness_recognized=bool(hgate),point_improvement=bool(min(r['gain_percent'])>=1 and r['difference']['ci95'][1]<0),all_condition_preservation=bool(all(min(v['methods']['both']['line_gate']['gain_percent'])>=-1 for k,v in results.items() if k.startswith('check/'))),nontrivial_selection=bool(min(r['selected_fraction'])>=.01),line_gate_beats_point_gate=bool(comparisons['check/all/all_detected']['both']['difference']['ci95'][1]<0));gates['accepted']=all(gates.values());write(RUN/'SUMMARY.json',dict(complete=True,new_hgb_models=48,boosting_rounds=2880,neural_updates=0,results=results,comparisons=comparisons,direction=direction,hardness=hardness,funnel=funnel,gates=gates,limits=read(RUN/'PROTOCOL.json')['limits']));write(RUN/'EVALUATION_COMPLETE.json',dict(complete=True,files={str((RUN/'SUMMARY.json').relative_to(ROOT)):sha(RUN/'SUMMARY.json')}));print('EVAL39',gates,flush=True)
if __name__=='__main__':main()
