from context41 import *

def main():
    p=population();part=partition(p);g=p['g'];probe=read(RUN/'PROBE.json');assert read(RUN/'PROBE_PROTOCOL.json')['primary_sha256']==sha(RUN/'SUMMARY.json');ir=np.array(INC)
    for seed in SEEDS:
        v=dict(np.load(RUN/'arrays'/f'seed{seed}.npz'));q=dict(np.load(RUN/'arrays'/f'quality_oracle_seed{seed}.npz'));m=dict(np.load(V40/'arrays'/f'matching_seed{seed}.npz'));seg=m['pred_segment'];t=seg[:,:,1]-seg[:,:,0];length=np.maximum(np.hypot(t[...,0],t[...,1]),1e-12);delta=g['support']-seg[:,:,None,0];dist=np.abs(t[:,:,None,0]*delta[...,1]-t[:,:,None,1]*delta[...,0]).mean(-1)/length;gn=g['normal'];cos=np.abs((-t[...,1]*gn[...,0]+t[...,0]*gn[...,1])/length);angle=np.degrees(np.arccos(np.clip(cos,0,1)));quality=(g['length']>0)&(dist<=3)&(angle<=10);assert np.array_equal(quality,q['quality']);assert np.allclose(dist,q['distance'],atol=1e-8)
        selected=v['selected_pred_segment_physical3_residual_gate'];role=ir[np.arange(8)[None],np.maximum(selected,0)];keep=(selected>=0)&quality[np.arange(p['N'])[:,None],role];assert np.array_equal(keep,q['keep']);expected=np.where(keep[...,None],v['points_pred_segment_physical3_residual_gate'],v['anchor']);assert np.array_equal(expected,q['points']);err=np.hypot(expected[...,0]-g['gt'][...,0],expected[...,1]-g['gt'][...,1]);assert abs(err[part['check']].mean()-probe['GT_quality_oracle']['check/all/all_detected']['per_seed_px'][seed-1])<1e-9
    write(RUN/'PROBE_AUDIT.json',dict(status='PASS',primary_unchanged=True,physical_quality_cross_product_recomputed=True,GT_quality_oracle_replayed=True));print('PROBE AUDIT41 PASS',flush=True)
if __name__=='__main__':main()
