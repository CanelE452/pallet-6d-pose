from context38 import *
from experiment38 import wquantile
from scipy.stats import spearmanr
METHODS=('mean_only',*[s+'_'+k for s in SCALES for k in ('marginal','point')])

def masks(p):
    part=partition(p)
    for split in ('fit','calibration','check'):
      for cond in ['all',*sorted(set(p['condition']))]:
        frame=part[split]&(True if cond=='all' else p['condition']==cond)
        for typ,m in [('all_detected',np.ones((p['N'],8),bool)),('supported',p['supported']),('outside',p['g']['outside']),('hidden',p['g']['hidden'])]:
            mask=frame[:,None]&m
            if mask.any():yield split+'/'+cond+'/'+typ,mask

def uqstats(a,method,mask,p):
    width=a['width_'+method];error=np.abs(a['target']-a['mu']);sel=a['selected_'+method];used=(sel>=0)&mask;index=np.maximum(sel,0);chosenerror=error[np.arange(p['N'])[:,None],np.arange(8)[None],index];chosenwidth=width[np.arange(p['N'])[:,None],np.arange(8)[None],index];m=method.rsplit('_',1)[0];scale=a['scale_'+m];constant=scale[mask].std()<1e-12
    return dict(candidate_coverage=float((error[mask]<=width[mask]).mean()),simultaneous_point_coverage=float((error[mask]<=width[mask]).all(-1).mean()),mean_halfwidth_px=float((width*p['diag'][:,None,None])[mask].mean()),median_halfwidth_px=float(np.median((width*p['diag'][:,None,None])[mask])),normal_target_MAE_px=float((error*p['diag'][:,None,None])[mask].mean()),zero_normal_target_MAE_px=float((np.abs(a['target'])*p['diag'][:,None,None])[mask].mean()),scale_spearman=None if constant else float(spearmanr(scale[mask].ravel(),error[mask].ravel()).statistic),selected_points=int(used.sum()),selected_coverage=float((chosenerror[used]<=chosenwidth[used]).mean()) if used.any() else None)

def main():
    assert read(RUN/'TRAINING_COMPLETE.json')['complete'];p=population();g=p['g'];part=partition(p);data={(arm,s):dict(np.load(RUN/'arrays'/f'{arm}_seed{s}.npz')) for arm in ARMS for s in SEEDS};results={};uncertainty={};comparisons={};coverage={}
    for key,mask in masks(p):
        row={};uq={};comp={}
        for arm in ARMS:
            base=np.stack([np.linalg.norm(data[(arm,s)]['anchor']-g['gt'],axis=-1) for s in SEEDS]);row[arm]={};uq[arm]={}
            for method in ('anchor',*METHODS):
                errors=np.stack([np.linalg.norm(data[(arm,s)]['anchor' if method=='anchor' else 'points_'+method]-g['gt'],axis=-1) for s in SEEDS]);r=rowstats(errors,base,mask,p['groups']);selection=[np.zeros_like(mask) if method=='anchor' else data[(arm,s)]['selected_'+method]>=0 for s in SEEDS];r['selected_fraction']=[float(t[mask].mean()) for t in selection];r['selected_points']=[int((t&mask).sum()) for t in selection];r['harm_over1px_fraction_all']=[float(((errors[i]-base[i])[mask]>1).mean()) for i in range(3)];r['harm_over1px_fraction_selected']=[float(((errors[i]-base[i])[mask&t]>1).mean()) if (mask&t).any() else None for i,t in enumerate(selection)];row[arm][method]=r
                if method not in ('anchor','mean_only'):uq[arm][method]=[uqstats(data[(arm,s)],method,mask,p) for s in SEEDS]
            geo=np.stack([np.linalg.norm(data[(arm,s)]['points_geometry_point']-g['gt'],axis=-1) for s in SEEDS]);const=np.stack([np.linalg.norm(data[(arm,s)]['points_constant_point']-g['gt'],axis=-1) for s in SEEDS]);mean=np.stack([np.linalg.norm(data[(arm,s)]['points_mean_only']-g['gt'],axis=-1) for s in SEEDS]);comp[arm]=dict(geometry_vs_constant=rowstats(geo,const,mask,p['groups']),geometry_vs_mean_only=rowstats(geo,mean,mask,p['groups']))
        results[key]=dict(points=int(mask.sum()),frames=int(mask.any(-1).sum()),methods=row);uncertainty[key]=uq;comparisons[key]=comp
    # Group-resolved coverage and calibration-derived reliability bins (no check chosen bins).
    reliability={};groupwise={}
    for arm in ARMS:
      for seed in SEEDS:
        a=data[(arm,seed)];err=np.abs(a['target']-a['mu']);rr={};gg={}
        for mode in SCALES:
            scale=a['scale_'+mode];edges=[wquantile(scale,p['groups'],part['calibration'],q) for q in (.2,.4,.6,.8)];b=np.searchsorted(edges,scale,side='right');rs=[]
            for k in range(5):
                mask=part['check'][:,None,None]&(b==k)
                if mask.any():rs.append(dict(bin=k,candidates=int(mask.sum()),mean_scale=float(scale[mask].mean()),mean_absolute_normal_error=float(err[mask].mean()),coverage=float((err[mask]<=a['width_'+mode+'_point'][mask]).mean())))
            rr[mode]=dict(calibration_bin_edges=edges,bins=rs)
            for group in sorted(set(p['groups'])):
                mask=np.broadcast_to((p['groups']==group)[:,None],(p['N'],8));gg[mode+'/'+group]=uqstats(a,mode+'_point',mask,p)
        reliability[f'{arm}_seed{seed}']=rr;groupwise[f'{arm}_seed{seed}']=gg
    key='check/all/all_detected';r=results[key]['methods']['both']['geometry_point'];q=uncertainty[key]['both']['geometry_point'];c=uncertainty[key]['both']['constant_point'];ugates=dict(coverage=all(x['simultaneous_point_coverage']>=.85 for x in q),ranking=all(x['scale_spearman'] is not None and x['scale_spearman']>=.2 for x in q),width=all(x['mean_halfwidth_px']<=.9*y['mean_halfwidth_px'] for x,y in zip(q,c)));gates=dict(uncertainty_useful=all(ugates.values()),point_improvement=min(r['gain_percent'])>=1 and r['difference']['ci95'][1]<0,all_condition_preservation=all(min(v['methods']['both']['geometry_point']['gain_percent'])>=-1 for k,v in results.items() if k.startswith('check/')),nontrivial_selection=min(r['selected_fraction'])>=.01,adds_over_constant=comparisons[key]['both']['geometry_vs_constant']['difference']['ci95'][1]<0);gates['accepted']=all(gates.values());write(RUN/'SUMMARY.json',dict(complete=True,hgb_fits=36,boosting_rounds=2160,neural_updates=0,results=results,uncertainty=uncertainty,comparisons=comparisons,uncertainty_gates=ugates,gates=gates,limits=read(RUN/'PROTOCOL.json')['limits']));write(RUN/'RELIABILITY.json',dict(complete=True,reliability=reliability,groupwise=groupwise));write(RUN/'EVALUATION_COMPLETE.json',dict(complete=True,files={str(q.relative_to(ROOT)):sha(q) for q in [RUN/'SUMMARY.json',RUN/'RELIABILITY.json']}));print('EVAL38',gates,flush=True)

if __name__=='__main__':main()
