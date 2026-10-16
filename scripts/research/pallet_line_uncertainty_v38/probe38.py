from context38 import *
from experiment38 import select,wquantile


def main():
    write(RUN/'PROBE_PROTOCOL.json',dict(exploratory=True,primary_sha256=sha(RUN/'SUMMARY.json'),reason='Primary uncertainty ranks errors but few actions and poor selected coverage. Separate mean-prediction, interval precision and acceptance failures using fixed-model arithmetic, no new fit.',oracles='oracle_mean: exactnormalGTmu with learnedwidth; oracle_radius: keep learnedmu but per-candidatewidth=abs(mu-target); oracle_both: exactmu zero width, equals bounded candidate point-GToracle. No deployable claims. Check condition strata and candidate gain ceilings.',curve='Geometry uncertainty scalar quantiles .50/.70/.80/.90/.95 computed only calibration4groups, both marginal and pointwise24max. No threshold chosen oncheck, no change to primary90%.',source_sha256=sha(KIT/'probe38.py')))
    p=population();part=partition(p);g=p['g'];rows={};curves={};cases=[];invariants=0;allerrors={};(RUN/'probes').mkdir(exist_ok=True)
    for arm in ARMS:
      for seed in SEEDS:
        a=dict(np.load(V37/'arrays'/f'v34_{arm}_seed{seed}.npz'));v=dict(np.load(RUN/'arrays'/f'{arm}_seed{seed}.npz'));mu=v['mu'];target=v['target'];width=v['width_geometry_point'];base=np.linalg.norm(a['anchor']-g['gt'],axis=-1);ce=np.linalg.norm(a['candidate']-g['gt'][:,:,None],axis=-1);mask=np.broadcast_to(part['check'][:,None],base.shape);actual=base[...,None]-ce;amount=a['candidate_amount']/p['diag'][:,None,None];norm2=(a['candidate_normal']**2).sum(-1);expected=2*amount*mu-amount**2*norm2;potential=(actual>1).any(-1);positiveuseful=((actual>1)&(expected>0)).any(-1);loweruseful=((actual>1)&(v['lower_geometry_point']>0)).any(-1);out={};r={}
        for name,m,w in [('oracle_mean',target,width),('oracle_radius',mu,np.abs(mu-target)),('oracle_both',target,np.zeros_like(width))]:
            point,sel,lo=select(a,m,w,p);error=np.linalg.norm(point-g['gt'],axis=-1);assert np.all(error<=base+1e-8);invariants+=1;out['points_'+name]=point;out['selected_'+name]=sel;allerrors[(arm,seed,name)]=error;r[name]=dict(mean_px=float(error[mask].mean()),selected=int(((sel>=0)&mask).sum()))
        assert np.allclose(allerrors[(arm,seed,'oracle_both')],np.minimum(base,ce.min(-1)),atol=1e-8)
        r['loss_funnel']=dict(check_points=int(mask.sum()),possible_over1px=int((potential&mask).sum()),possible_with_positive_predicted_gain=int((positiveuseful&mask).sum()),possible_with_positive_lower_gain=int((loweruseful&mask).sum()),primary_selected=int(((v['selected_geometry_point']>=0)&mask).sum()))
        c=[];ratio=np.abs(mu-target)/v['scale_geometry']
        for family,qdata in [('marginal',ratio),('point',ratio.max(-1))]:
          for level in (.5,.7,.8,.9,.95):
            q=wquantile(qdata,p['groups'],part['calibration'],level);w=q*v['scale_geometry'];point,sel,_=select(a,mu,w,p);error=np.linalg.norm(point-g['gt'],axis=-1);selected=mask&(sel>=0);covered=np.abs(mu-target)<=w;c.append(dict(family=family,calibration_level=level,q=q,check_mean_px=float(error[mask].mean()),check_selected_points=int(selected.sum()),check_harm_over1px_selected=int(((error-base>1)&selected).sum()),check_simultaneous_coverage=float(covered[mask].all(-1).mean()),check_candidate_coverage=float(covered[mask].mean()),check_halfwidth_px=float((w*p['diag'][:,None,None])[mask].mean())))
        # Record every selected primary point; few selected points are not a statistical safety guarantee.
        for i,j in zip(*np.where(mask&(v['selected_geometry_point']>=0))):
            k=int(v['selected_geometry_point'][i,j]);er=float(np.linalg.norm(v['points_geometry_point'][i,j]-g['gt'][i,j]));di=p['diag'][i];cases.append(dict(arm=arm,seed=seed,index=int(i),point=int(j),candidate=k,id=p['rows'][i]['id'],source_group=p['rows'][i]['source_group'],condition=p['rows'][i]['condition'],base_px=float(base[i,j]),after_px=er,actual_normal_px=float(target[i,j,k]*di),mean_normal_px=float(mu[i,j,k]*di),width_px=float(width[i,j,k]*di),interval_covered=bool(abs(target[i,j,k]-mu[i,j,k])<=width[i,j,k]),amount_px=float(a['candidate_amount'][i,j,k])))
        rows[f'{arm}_seed{seed}']=r;curves[f'{arm}_seed{seed}']=c;np.savez_compressed(RUN/'probes'/f'{arm}_seed{seed}.npz',**out)
    stats={};mask=np.broadcast_to(part['check'][:,None],(p['N'],8))
    for arm in ARMS:
        base=np.stack([np.linalg.norm(np.load(RUN/'arrays'/f'{arm}_seed{s}.npz')['anchor']-g['gt'],axis=-1) for s in SEEDS]);stats[arm]={}
        for name in ('oracle_mean','oracle_radius','oracle_both'):stats[arm][name]=rowstats(np.stack([allerrors[(arm,s,name)] for s in SEEDS]),base,mask,p['groups'])
    write(RUN/'PROBES.json',dict(complete=True,exploratory=True,oracles=rows,oracle_summary=stats,curves=curves,all_primary_selected_cases=cases,oracle_nonworsening_checks=invariants));print('PROBES38',stats['both'],[rows[f'both_seed{s}']['loss_funnel'] for s in SEEDS],flush=True)

if __name__=='__main__':main()
