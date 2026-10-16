from context40 import *

def main():
    p=population();part=partition(p);g=p['g'];report=read(RUN/'PROBE.json');assert read(RUN/'PROBE_PROTOCOL.json')['primary_sha256']==sha(RUN/'SUMMARY.json');ir=np.repeat(np.array(INC),8,axis=1);valid=g['length'][:,ir]>0;normal=g['normal'][:,ir];t=np.stack((-normal[...,1],normal[...,0]),-1)
    for seed in SEEDS:
        a=dict(np.load(V37/'arrays'/f'v34_both_seed{seed}.npz'));v=dict(np.load(RUN/'arrays'/f'seed{seed}.npz'));error=(g['gt']-a['anchor'])[:,:,None];actual=(a['candidate_normal']*error).sum(-1);observable=(normal*error).sum(-1)*(a['candidate_normal']*normal).sum(-1);tangent=(t*error).sum(-1)*(a['candidate_normal']*t).sum(-1);assert np.allclose((actual-observable)[valid],tangent[valid],atol=1e-8)
        h=part['check'][:,None,None]&v['hard'][...,None];mask=h&valid;direct=np.abs(tangent[mask]).mean();r=report['rows'][seed-1];assert abs(direct-r['supported']['GT_exact_offset_observable_component_MAE_px'])<1e-8
        mask_unique=mask[...,::8];dist=np.hypot(g['support'][:,np.array(INC)][...,0]-g['gt'][:,:,None,None,0],g['support'][:,np.array(INC)][...,1]-g['gt'][:,:,None,None,1]).min(-1);assert int(mask_unique.sum())==r['endpoint_check']['incident_role_count'];assert abs((dist[mask_unique]<=3).mean()-r['endpoint_check']['endpoint_within3px_fraction'])<1e-12
        for source in ('baseline',*SOURCES):
            prediction=v['mu_'+source+'_all']*p['diag'][:,None,None];e=np.abs(actual-prediction);assert abs(e[mask].mean()-r['supported']['normal_MAE_px'][source])<1e-9
    write(RUN/'PROBE_AUDIT.json',dict(status='PASS',primary_unchanged=True,normal_tangent_decomposition_verified=True,endpoint_distances_deduplicated=True,supported_model_metrics_recalculated=True));print('PROBE AUDIT40 PASS',flush=True)
if __name__=='__main__':main()
