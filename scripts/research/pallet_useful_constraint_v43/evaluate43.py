from context43 import *
from sklearn.metrics import roc_auc_score,average_precision_score
METHODS=('anchor','fallback','V42')+tuple(a+'_'+r for a in ARMS for r in ROUTES)+('GT_useful_best','GT_useful_residual')

def diagnose(q,p,part,method,split):
    frame=part[split];valid=q['valid'];useful=q['label_useful']&valid;M=frame[:,None,None]&valid;accepted=q['accepted_V42' if method=='V42' else 'accepted_'+method];positive=useful.any(-1)&frame[:,None];covered=(useful&accepted).any(-1)&frame[:,None];hard=q['hard'];arm='V42' if method=='V42' else method.rsplit('_',1)[0];pr=q['prob_'+arm];u=useful[M];score=pr[M];bins=[]
    for k in range(10):
        mm=M&(pr>=k/10)&((pr<(k+1)/10) if k<9 else pr<=1);n=int(mm.sum());bins.append(dict(lo=k/10,hi=(k+1)/10,slots=n,score_mean=float(pr[mm].mean()) if n else None,valid_rate=float(q['label_valid'][mm].mean()) if n else None,useful_rate=float(useful[mm].mean()) if n else None))
    selected=None
    if method!='V42':
        sel=q['selected_'+method];idx=np.maximum(sel,0);used=frame[:,None]&(sel>=0);truthsel=q['label_valid'][np.arange(p['N'])[:,None],np.arange(8)[None],idx];be=np.linalg.norm(q['fallback']-p['g']['gt'],axis=-1);err=np.linalg.norm(q['points_'+method]-p['g']['gt'],axis=-1);sw=np.linalg.norm(q['points_'+method]-q['fallback'],axis=-1)>1e-8
        selected=dict(points=int(used.sum()),constraint_precision=float(truthsel[used].mean()) if used.any() else None,switches=int((used&sw).sum()),help_over1px=int(((be-err>1)&used).sum()),harm_over1px=int(((err-be>1)&used).sum()),hard_help_over1px=int(((be-err>1)&used&hard).sum()),hard_harm_over1px=int(((err-be>1)&used&hard).sum()))
    return dict(slots=int(M.sum()),useful_slots=int(u.sum()),prevalence=float(u.mean()),useful_auroc=float(roc_auc_score(u,score)) if len(set(u))>1 else None,useful_AP=float(average_precision_score(u,score)) if u.any() else None,useful_probability_median=float(np.median(score[u])) if u.any() else None,nonuseful_probability_median=float(np.median(score[~u])) if (~u).any() else None,useful_slot_recall=float((useful&accepted&M).sum()/max(u.sum(),1)),opportunity_points=int(positive.sum()),covered_points=int(covered.sum()),hard_opportunity_points=int((positive&hard).sum()),hard_covered_points=int((covered&hard).sum()),selected=selected,bins=bins)

def main():
    p=population();part=partition(p);data=[dict(np.load(RUN/'arrays'/f'seed{s}.npz')) for s in SEEDS];hard=np.stack([q['hard'] for q in data]);E={m:np.stack([np.linalg.norm(q[m if m in ('anchor','fallback') else 'points_'+m]-p['g']['gt'],axis=-1) for q in data]) for m in METHODS};results={};comparisons={}
    for key,M in masks(p,hard):
        if not key.startswith('check/'):continue
        row={}
        for m in METHODS:
            z=metric(E[m],E['fallback'],M,p['groups']);z['switches']=[int((M[i]&(np.linalg.norm(data[i][m if m in ('anchor','fallback') else 'points_'+m]-data[i]['fallback'],axis=-1)>1e-8)).sum()) for i in range(3)];z['harm_over1px']=[int((M[i]&(E[m][i]-E['fallback'][i]>1)).sum()) for i in range(3)];row[m]=z
        results[key]=row
        comparisons[key]={m:metric(E[PRIMARY],E[m],M,p['groups']) for m in ('V42','valid_natural_visual_probability','useful_natural_visual_probability','useful_weighted_geometry_probability')}
    diag={str(s):{m:{split:diagnose(q,p,part,m,split) for split in ('fit','calibration','check')} for m in ('V42',*tuple(a+'_'+r for a in ARMS for r in ROUTES))} for s,q in zip(SEEDS,data)}
    r=results['check/all/all_detected'][PRIMARY];chosen=[diag[str(s)][PRIMARY]['check']['selected'] for s in SEEDS]
    gates=dict(point_gain=min(r['gain_percent'])>=1 and r['difference']['ci95'][1]<0,condition_preservation=all(min(row[PRIMARY]['gain_percent'])>=-1 for row in results.values()),nontrivial_switches=min(r['switches'])>=.01*1040,constraint_precision=all(x['constraint_precision'] is not None and x['constraint_precision']>=.9 for x in chosen),opportunity_coverage_doubled=all(diag[str(s)][PRIMARY]['check']['covered_points']>=2*diag[str(s)]['V42']['check']['covered_points'] for s in SEEDS));gates['accepted']=all(gates.values())
    write(RUN/'SUMMARY.json',dict(results=results,comparisons=comparisons,diagnostics=diag,gates=gates,primary=PRIMARY,scope=read(RUN/'PROTOCOL.json')['scope']))
    write(RUN/'EVALUATION_COMPLETE.json',dict(complete=True,files={str(q.relative_to(ROOT)):sha(q) for q in (RUN/'SUMMARY.json',KIT/'evaluate43.py')}));print('EVAL43',gates,flush=True);print('POINT43',{k:round(v['mean_px'],4) for k,v in results['check/all/all_detected'].items()},flush=True)
if __name__=='__main__':main()
