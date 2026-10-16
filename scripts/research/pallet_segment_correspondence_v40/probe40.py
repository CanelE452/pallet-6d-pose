from context40 import *

def main():
    write(RUN/'PROBE_PROTOCOL.json',dict(exploratory=True,primary_sha256=sha(RUN/'SUMMARY.json'),question='Does averaging unsupported/unaligned candidates hide useful physical-line evidence, and are visible envelope endpoints the intended virtual corners?',definitions='GTsupported=role length>0; aligned=abs dot candidate/physicalnormal>=cos10deg. Endpoint distance is minimum Euclidean distance of two visible-envelope endpoints to intendedvirtual GTcorner onincidentrole, not endpointidentityaccuracy. Near/far extension not fitted. Analyticphysicalcenter uses V35old16trainmedianoffset withphysical GTline, projected onto fixedcandidate normal. Exactoffset analytic diagnostic is GTphysical observable component only and omitscandidate-normal projection ofphysicaltangenterror. This is postprimary descriptive,no tuning/refit/newpolicy.'))
    p=population();part=partition(p);g=p['g'];ir=np.repeat(np.array(INC),8,axis=1);valid=g['length'][:,ir]>0;normal=g['normal'][:,ir];support=g['support'][:,ir];offsets=read(V35/'OFFSET_CALIBRATION.json')['models'];med=np.array([[offsets[f'{j}_{r}']['median'] for r in ir[j]] for j in range(8)]);allv=[dict(np.load(RUN/'arrays'/f'seed{k}.npz')) for k in SEEDS];rows=[];endpoint=np.linalg.norm(support-g['gt'][:,:,None,None],axis=-1).min(-1)
    for seed,v in zip(SEEDS,allv):
        a=dict(np.load(V37/'arrays'/f'v34_both_seed{seed}.npz'));dot=(normal*a['candidate_normal']).sum(-1);hard=part['check'][:,None,None]&v['hard'][...,None];true=v['target']*p['diag'][:,None,None];residual=g['rho'][:,ir]+med[None]*p['diag'][:,None,None]-(normal*a['anchor'][:,:,None]).sum(-1);analytic=residual*dot;exact=(normal*(g['gt']-a['anchor'])[:,:,None]).sum(-1)*dot;result={}
        for name,mask in [('all',np.broadcast_to(hard,true.shape)),('supported',hard&valid),('unsupported',hard&~valid),('supported_aligned10',hard&valid&(np.abs(dot)>=np.cos(np.deg2rad(10))))]:
            if not mask.any():continue
            r=dict(candidate_targets=int(mask.sum()),distinct_points=int(mask.any(-1).sum()),normal_MAE_px={k:float(np.abs(v['mu_'+k+'_all']*p['diag'][:,None,None]-true)[mask].mean()) for k in ('baseline',*SOURCES)})
            if name in ('supported','supported_aligned10'):r.update(analytic_GT_line_train_median_MAE_px=float(np.abs(analytic-true)[mask].mean()),GT_exact_offset_observable_component_MAE_px=float(np.abs(exact-true)[mask].mean()),visible_envelope_endpoint_distance_to_corner_mean_px=float(endpoint[mask].mean()),visible_envelope_endpoint_distance_to_corner_median_px=float(np.median(endpoint[mask])))
            result[name]=r
        unique=part['check'][:,None,None]&v['hard'][...,None]&valid;result['endpoint_check']=dict(incident_role_count=int(unique[...,::8].sum()),endpoint_within3px_fraction=float((endpoint[unique]<=3).mean()),endpoint_within10px_fraction=float((endpoint[unique]<=10).mean()))
        rows.append(result)
    write(RUN/'PROBE.json',dict(complete=True,exploratory=True,rows=rows,limitations='Physicalvisible envelope endpoints are not virtual corner GT; no endpoint correspondence class or true junction detector was trained. Analytic privilegedtargets use frozenold16medianoffsets, not newly fit12. Exactoffset variant uses GTcorner and is not deployable.'))
    print('PROBE40 COMPLETE',rows,flush=True)
if __name__=='__main__':main()
