from context42 import *
from sklearn.metrics import roc_auc_score,average_precision_score
METHODS=('anchor','fallback')+tuple(t+'_'+f+'_'+r for t in TARGETS for f in FEATURES for r in ROUTES)+tuple('GT_'+t+'_'+r for t in TARGETS for r in ROUTES)+('GT_candidate','GT_safe_primary')

def main():
    assert read(RUN/'TRAINING_COMPLETE.json')['complete'];p=population();part=partition(p);g=p['g'];v=[dict(np.load(RUN/'arrays'/f'seed{k}.npz')) for k in SEEDS];hard=np.stack([q['hard'] for q in v]);E={m:np.stack([np.linalg.norm(q[m if m in ('anchor','fallback') else 'points_'+m]-g['gt'],axis=-1) for q in v]) for m in METHODS};results={};selected={};classification={}
    for key,M in masks(p,hard):
        if key.startswith('fit/') or key.startswith('calibration/'):continue
        row={}
        for method in METHODS:
            r=metric(E[method],E['fallback'],M,p['groups']);r['vs_anchor']=metric(E[method],E['anchor'],M,p['groups']);r['harmed_over1px_vs_fallback']=[int(((E[method][i]-E['fallback'][i]>1)&M[i]).sum()) for i in range(3)];r['switches']=[int((M[i]&(np.linalg.norm(v[i][method if method in ('anchor','fallback') else 'points_'+method]-v[i]['fallback'],axis=-1)>1e-8)).sum()) for i in range(3)];row[method]=r
        results[key]=row
    for seed,q in zip(SEEDS,v):
        classification[str(seed)]={};selected[str(seed)]={}
        for target in TARGETS:
          for feature in FEATURES:
            name=target+'_'+feature;pr=q['prob_'+name];truth=q['label_'+target];sel=q['accepted_'+name];classification[str(seed)][name]={}
            for split in ('fit','calibration','check'):
                mask=part[split][:,None,None]&q['valid'];y=truth[mask];z=pr[mask];take=sel[mask];tp=int((take&y).sum());total=int(take.sum());classification[str(seed)][name][split]=dict(slots=int(mask.sum()),positive=int(y.sum()),prevalence=float(y.mean()),auroc=float(roc_auc_score(y,z)) if len(set(y))>1 else None,average_precision=float(average_precision_score(y,z)),accepted=total,precision=float(tp/total) if total else None,recall=float(tp/y.sum()) if y.sum() else None)
            for route in ROUTES:
                method=name+'_'+route;k=q['selected_'+method];mask=part['check'][:,None]&(k>=0);index=np.maximum(k,0);labels={t:q['label_'+t][np.arange(p['N'])[:,None],np.arange(8)[None],index] for t in TARGETS};sw=np.linalg.norm(q['points_'+method]-q['fallback'],axis=-1)>1e-8;be=E['fallback'][seed-1];err=E[method][seed-1];selected[str(seed)][method]=dict(selected_points=int(mask.sum()),switches=int((mask&sw).sum()),constraint_precision=float(labels['constraint'][mask].mean()) if mask.any() else None,label_precision={t:float(y[mask].mean()) if mask.any() else None for t,y in labels.items()},harmed_over1px=int(((err-be>1)&mask).sum()),harmed_invalid_constraint=int(((err-be>1)&mask&~labels['constraint']).sum()),true_constraint_accepted=int((mask&labels['constraint']).sum()))
    primary='constraint_visual_pre_all';r=results['check/all/all_detected'][primary];gates=dict(selected_constraint_precision=all(selected[str(seed)][primary]['constraint_precision'] is not None and selected[str(seed)][primary]['constraint_precision']>=.9 for seed in SEEDS),incremental_point_improvement=min(r['gain_percent'])>=1 and r['difference']['ci95'][1]<0,condition_preservation=all(min(row[primary]['gain_percent'])>=-1 for row in results.values()),nontrivial_switches=min(r['switches'])>=.01*1040);gates['accepted']=all(gates.values())
    write(RUN/'SUMMARY.json',dict(results=results,classification=classification,selected=selected,gates=gates,scope='Reused synthetic diagnostic,pointmetric baseline=frozenV41direct10.410 rather thananchor10.463. vs_anchor explicit. GTpolicies neverdeployable.'))
    write(RUN/'EVALUATION_COMPLETE.json',dict(complete=True,files={str(q.relative_to(ROOT)):sha(q) for q in (RUN/'SUMMARY.json',KIT/'evaluate42.py')}));print('EVAL42',gates,flush=True);print('POINT42',{k:round(r['mean_px'],3) for k,r in results['check/all/all_detected'].items()},flush=True);print('SELECTED42',{k:v[primary] for k,v in selected.items()},flush=True)
if __name__=='__main__':main()
