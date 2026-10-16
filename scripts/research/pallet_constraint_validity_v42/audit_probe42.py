from context42 import *
from experiment42 import load_inputs

def main():
    p=population();part=partition(p);g=p['g'];probe=read(RUN/'PROBE.json');assert read(RUN/'PROBE_PROTOCOL.json')['primary_sha256']==sha(RUN/'SUMMARY.json')
    for seed in SEEDS:
        a,v,X=load_inputs(p,seed);q=dict(np.load(RUN/'arrays'/f'seed{seed}.npz'));base=np.hypot(q['fallback'][...,0]-g['gt'][...,0],q['fallback'][...,1]-g['gt'][...,1]);ce=np.hypot(a['candidate'][...,0]-g['gt'][:,:,None,0],a['candidate'][...,1]-g['gt'][:,:,None,1])
        for pool in ('visible','physical','constraint','accepted'):
            allow=q['accepted_constraint_visual'] if pool=='accepted' else q['label_'+pool]&a['valid'];expected=np.minimum(base,np.where(allow,ce,np.inf).min(-1));pt=np.load(RUN/'arrays'/f'oracle_{pool}_seed{seed}.npz')['points'];err=np.hypot(pt[...,0]-g['gt'][...,0],pt[...,1]-g['gt'][...,1]);assert np.allclose(err,expected,atol=1e-9);mask=part['check'][:,None];op=(base-err>1)&mask;row=probe['rows'][seed-1]['pools'][pool];assert int(op.sum())==row['opportunities_over1px'];assert int((op&q['hard']).sum())==row['hard_opportunities_over1px'];assert abs(err[part['check']].mean()-probe['oracle_results'][pool]['check/all/all_detected']['per_seed_px'][seed-1])<1e-9
    write(RUN/'PROBE_AUDIT.json',dict(status='PASS',primary_unchanged=True,oracle_candidate_minima_recomputed=True,opportunity_counts_verified=True));print('PROBE AUDIT42 PASS',flush=True)
if __name__=='__main__':main()
