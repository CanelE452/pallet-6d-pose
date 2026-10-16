from context40 import *
from observations40 import extract, build
import pickle

def main():
    register();p=population();part=partition(p)
    for name in ('models','arrays','features'): (RUN/name).mkdir(exist_ok=True)
    write(RUN/'SOURCE_BINDINGS.json',dict(protocol_sha256=sha(RUN/'PROTOCOL.json'),files={str(q.relative_to(ROOT)):sha(q) for q in (KIT/'context40.py',KIT/'observations40.py',KIT/'experiment40.py')}))
    raw=extract(p);np.savez_compressed(RUN/'segments.npz',**raw);print('EXTRACT40',p['N'],int(raw['counts'].sum()),flush=True);fits=0
    for seed in SEEDS:
        z=load_seed(seed,p);a,X,support=observable(p,z,seed,'both');extra,matched=build(p,z,a,support,raw);v=dict(np.load(V39/'arrays'/f'both_seed{seed}.npz'));y=v['target'];out=dict(anchor=a['anchor'],target=y,hard=v['hard'],hard_prediction=v['hard_prediction_line'],mu_baseline_all=v['mu_standard'],mu_baseline_hard=v['mu_hard_geometry'],points_standard=v['points_standard'],points_hard_gate=v['points_line_gate'],points_GT_hard=v['points_GT_hard_geometry'],points_candidate_oracle=v['points_candidate_oracle']);bundle={}
        np.savez_compressed(RUN/'features'/f'seed{seed}.npz',geometry=X['geometry'],**extra);np.savez_compressed(RUN/'arrays'/f'matching_seed{seed}.npz',**matched)
        for source in SOURCES:
            feature=np.concatenate((X['geometry'],extra[source]),-1)
            for family in FAMILIES:
                key=source+'_'+family;m=fit_reg(feature,y,v['hard'],part['fit'],p['groups'],40000+seed,'hard_geometry' if family=='hard' else 'all');mu=pred(m,feature);bundle[key]=m;out['mu_'+key]=mu;pts,sel,gain=select(a,mu,np.zeros_like(y),p)
                if family=='all':out['points_'+key]=pts;out['selected_'+key]=sel
                else:
                    for gate,mask in [('hard_gate',v['hard_prediction_line']),('GT_hard',v['hard'])]:
                        name=source+'_'+gate;out['points_'+name]=np.where(mask[...,None],pts,a['anchor']);out['selected_'+name]=np.where(mask,sel,-1)
                fits+=1;print('FIT40',seed,key,fits,flush=True)
        (RUN/'models'/f'seed{seed}.pkl').write_bytes(pickle.dumps(bundle,protocol=4));np.savez_compressed(RUN/'arrays'/f'seed{seed}.npz',**out);write(RUN/'models'/f'seed{seed}.json',dict(complete=True,models=10,rounds=600,fit_groups=part['fit_groups']))
    files=[RUN/'segments.npz',*sorted((RUN/'models').glob('*')),*sorted((RUN/'arrays').glob('*.npz')),*sorted((RUN/'features').glob('*.npz'))];write(RUN/'TRAINING_COMPLETE.json',dict(complete=True,hgb_fits=fits,boosting_rounds=fits*60,neural_updates=0,files={str(q.relative_to(ROOT)):sha(q) for q in files}));print('TRAIN40 COMPLETE',fits,flush=True)
if __name__=='__main__':main()
