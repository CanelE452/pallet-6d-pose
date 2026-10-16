"""Training-only constant-scale optimum versus per-point oracle and learned models."""
from context22 import *
from scipy.optimize import minimize_scalar

def main():
    rows=[]
    for seed in SEEDS:
        view=torch.load(OLD.RUN/f'QUALITY_VIEWS_seed{seed}.pt',map_location='cpu',weights_only=True);base=view['bases'][0].numpy().astype(float);delta=view['deltas'][0].numpy().astype(float);ts=[];ids=[]
        for c in TRAIN_CONDITIONS:
            m=read(C.RUN/'cache/heldout'/c/'manifest.json');t=C.target_rows('heldout',c);ts.extend(x for x,r in zip(t,m['rows']) if r['detected']);ids.extend(r['id']+'::'+c for r in m['rows'] if r['detected'])
        assert ids==view['ids'];truth=np.array([x['points'] for x in ts]);valid=np.array([x['valid'] for x in ts]);outside=np.array([x['outside'] for x in ts]);cond=np.array([x.rsplit('::',1)[1] for x in ids]);oracle=np.clip(((truth-base)*delta).sum(-1)/np.maximum((delta*delta).sum(-1),1e-12),0,1)
        for tag,mask in [('all',valid),('outside',valid&outside),('crop20outside',valid&outside&(cond=='crop20')[:,None]),('crop40outside',valid&outside&(cond=='crop40')[:,None])]:
            b=base[mask];d=delta[mask];g=truth[mask]
            def cost(alpha):return float(np.linalg.norm(b+alpha*d-g,axis=-1).mean())
            opt=minimize_scalar(cost,bounds=(0,1),method='bounded');alpha=min((0.,float(opt.x),1.),key=cost)
            rows.append(dict(seed=seed,group=tag,points=int(mask.sum()),constant_optimal_alpha=alpha,constant_optimal_error_px=cost(alpha),base_error_px=cost(0),full_error_px=cost(1),oracle_per_point_error_px=float(np.linalg.norm(base+oracle[...,None]*delta-truth,axis=-1)[mask].mean())))
    write(RUN/'CONSTANT_DIAGNOSIS.json',dict(status='PASS',rows=rows,scope='Privileged synthetic-training diagnostic; not an inference policy, no real GT used. Group-specific constant optimizer is not a learned method.'));print('CONSTANT DIAGNOSIS PASS',flush=True)

if __name__=='__main__':main()
