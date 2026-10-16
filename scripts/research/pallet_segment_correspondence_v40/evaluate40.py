from context40 import *
from observations40 import canonical
METHODS=('anchor','standard','hard_gate','GT_hard','candidate_oracle')+tuple(s+'_'+f for s in SOURCES for f in ('all','hard_gate','GT_hard'))

def main():
    assert read(RUN/'TRAINING_COMPLETE.json')['complete'];p=population();part=partition(p);g=p['g'];v=[dict(np.load(RUN/'arrays'/f'seed{k}.npz')) for k in SEEDS];hard=np.stack([q['hard'] for q in v]);E={m:np.stack([np.linalg.norm(q['anchor' if m=='anchor' else 'points_'+m]-g['gt'],axis=-1) for q in v]) for m in METHODS};results={};direction={};comparisons={};coverage={}
    for key,M in masks(p,hard):
        result={}
        for method in METHODS:
            row=metric(E[method],E['anchor'],M,p['groups']);row['harmed_over1px']=[int(((E[method][i]-E['anchor'][i]>1)&M[i]).sum()) for i in range(3)];row['moved_points']=[int(((E[method][i]!=E['anchor'][i])&M[i]).sum()) for i in range(3)];result[method]=row
        results[key]=result
        if key in ('fit/all/hard','check/all/hard','fit/all/all_detected','check/all/all_detected'):
            MM=np.broadcast_to(M[...,None],(3,p['N'],8,24));target=np.stack([q['target'] for q in v]);scale=p['diag'][None,:,None,None];dd={};compare={}
            for family in FAMILIES:
                errs={source:np.abs(np.stack([q['mu_'+source+'_'+family] for q in v])-target)*scale for source in ('baseline',*SOURCES)};dd[family]={}
                for source in ('baseline',*SOURCES):
                    mu=np.stack([q['mu_'+source+'_'+family] for q in v]);r=metric(errs[source],errs['baseline'],MM,p['groups']);sel=MM&(np.abs(target)*scale>=1);r['sign_accuracy']=[float((np.sign(mu[i][sel[i]])==np.sign(target[i][sel[i]])).mean()) for i in range(3)];r['zero_MAE_px']=[float(np.abs(target[i]*scale[0])[MM[i]].mean()) for i in range(3)];dd[family][source]=r
                compare[family]={a+'_vs_'+b:metric(errs[a],errs[b],MM,p['groups']) for a,b in [('GT_segment','GT_line'),('GT_segment','GT_shuffled_segment'),('GT_match_segment','pred_segment'),('GT_segment','GT_match_segment')]}
            direction[key]=dd;comparisons[key]=compare
    for split in ('fit','calibration','check'):
        fr=part[split];items=[]
        for seed in SEEDS:
            a=dict(np.load(RUN/'arrays'/f'matching_seed{seed}.npz'));row={}
            for name,validkey in [('pred_segment','pred_valid'),('GT_match_segment','GT_match_valid')]:
                seg=a[name];valid=a[validkey];n=canonical(seg)[1];visible=g['length']>0;dist=np.abs((g['support']*n[:,:,None]).sum(-1)-(n*seg[:,:,0]).sum(-1)[...,None]).mean(-1);mask=fr[:,None]&valid&visible;angle=np.degrees(np.arccos(np.clip(np.abs((n*g['normal']).sum(-1)),0,1)));row[name]=dict(assigned_roles=int((valid&fr[:,None]).sum()),visible_roles=int((visible&fr[:,None]).sum()),assigned_visible_roles=int(mask.sum()),assigned_invisible_roles=int((valid&~visible&fr[:,None]).sum()),physical_support_line_distance_px=float(dist[mask].mean()),angle_degrees=float(angle[mask].mean()),within3px10deg=int((mask&(dist<=3)&(angle<=10)).sum()))
            items.append(row)
        coverage[split]=items
    def improves(r,percent):return min(r['gain_percent'])>=percent and r['difference']['ci95'][1]<0
    d=direction['check/all/hard']['all'];comp=comparisons['check/all/hard']['all'];r=results['check/all/all_detected'];gates=dict(predicted_direction=improves(d['pred_segment'],10),predicted_points=improves(r['pred_segment_all'],1),predicted_condition_preservation=all(min(value['pred_segment_all']['gain_percent'])>=-1 for k,value in results.items() if k.startswith('check/')),GT_extent_adds=improves(comp['GT_segment_vs_GT_line'],10),GT_correct_role_adds=improves(comp['GT_segment_vs_GT_shuffled_segment'],10),GT_matching_adds=improves(comp['GT_match_segment_vs_pred_segment'],10),GT_support_adds=improves(comp['GT_segment_vs_GT_match_segment'],10));gates['accepted']=all(gates[k] for k in ('predicted_direction','predicted_points','predicted_condition_preservation'))
    write(RUN/'SUMMARY.json',dict(results=results,direction=direction,direction_comparisons=comparisons,coverage=coverage,gates=gates,scope='Reused synthetic diagnostic,GTvariants privileged,not deployable. Conditions/seed-specific hard masks preserved.'))
    write(RUN/'EVALUATION_COMPLETE.json',dict(complete=True,files={str(q.relative_to(ROOT)):sha(q) for q in (RUN/'SUMMARY.json',KIT/'evaluate40.py')}));print('EVAL40',gates,flush=True);print('ALLPOINT40',{k:round(x['mean_px'],3) for k,x in r.items()},flush=True);print('DIRECTION40',{k:round(x['mean_px'],3) for k,x in d.items()},flush=True)
if __name__=='__main__':main()
