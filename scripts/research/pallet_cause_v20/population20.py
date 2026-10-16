"""Posthoc residual-target shift with point-ID composition matched; no fitting."""
from context20 import *
from model20 import oracle_alpha

def summarize(base,delta,gt,valid):
    target=np.clip(np.sum((gt-base)*delta,-1)/np.maximum(np.sum(delta*delta,-1),1e-12),0,1);before=np.linalg.norm(gt-base,axis=-1);rows=[]
    for point in range(8):
        v=valid[:,point];rows.append(dict(point=point,n=int(v.sum()),alpha=float(target[:,point][v].mean()) if v.any() else None,stop_fraction=float((target[:,point][v]<=.05).mean()) if v.any() else None,base_error_px=float(before[:,point][v].mean()) if v.any() else None))
    return rows

def main():
    setup();rows=[]
    for seed in SEEDS:
        O,rep,gt,valid,ids,kind=proposal_training(seed);condition=np.array([x.rsplit('::',1)[1] for x in ids]);base=rep['base'].cpu().numpy();delta=rep['proposal_delta'].cpu().numpy();y=gt.cpu().numpy();v=valid.cpu().numpy()&(kind.cpu().numpy()==2)
        for c in ('crop20','crop40'):
            source=summarize(base,delta,y,v&(condition==c)[:,None]);groups={'train':source}
            for split in ('synth_val','real'):
                m,o,r=eval_inputs(split,c,seed);ts=targets(split,c);det=np.array([x['detected'] for x in m['rows']]);target=np.array([t['points'] for t in ts])[det];mask=(np.array([t['valid'] for t in ts])&np.array([t['outside'] for t in ts]))[det];groups[split]=summarize(r['base'].numpy(),r['proposal_delta'].numpy(),target,mask)
            weights=np.array([x['n'] for x in groups['real']],float);weights/=weights.sum();matched={}
            for split,rs in groups.items():
                assert all(x['n']>0 for w,x in zip(weights,rs) if w>0)
                matched[split]={key:float(sum(w*x[key] for w,x in zip(weights,rs) if w>0)) for key in ('alpha','stop_fraction','base_error_px')}
            rows.append(dict(seed=seed,condition=c,point_rows=groups,real_point_id_weights=weights.tolist(),point_id_matched_means=matched));print(c,seed,matched,flush=True)
    write(RUN/'POPULATION_DIAGNOSIS.json',dict(rows=rows,scope='Posthoc GT-based diagnostic only, point-ID matched to real valid outside-corner composition. Does not match pose, texture, image quality or latent feature distributions and is not causal proof. Weights never used for training or inference.'))

if __name__=='__main__':main()
