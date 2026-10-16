"""Independent update/score replay; privileged routing diagnostics after evaluation."""
from context12 import *

def main():
    setup();summary=read(RUN/'SUMMARY.json');train=read(RUN/'TRAIN_PROTOCOL.json')
    for name,expected in train['source_sha256'].items():assert sha(ROOT/name)==expected,name
    for arm in ARMS:
        finals=[];plans=[]
        for seed in SEEDS:
            d=RUN/'models'/f'{arm}_seed{seed}';c=read(d/'COMPLETE.json');assert c['complete'] and c['steps']==2000
            assert sha(d/'final.pt')==c['checkpoint_sha256'];finals.append(c['checkpoint_sha256'])
            trace=[json.loads(x) for x in (d/'TRACE.jsonl').read_text().splitlines()];assert len(trace)==2000
            assert all(r['step']==i+1 and np.isfinite([r['loss'],r['point_loss'],r['routing_loss'],r['gradient_norm']]).all() for i,r in enumerate(trace))
        assert len(set(finals))==3
    for seed in SEEDS:
        starts=[read(RUN/'models'/f'{a}_seed{seed}'/'START.json') for a in ARMS]
        assert len({r['plan_sha256'] for r in starts})==len({r['initial_state_sha256'] for r in starts})==1
    corner=read(RUN/'cache'/'train_corner20'/'manifest.json');t=read(RUN/'targets_train_corner20.json')['rows'];old={r['id']:r for r in read(V10/'targets_train.json')['rows']}
    for rec,r in zip(corner['rows'],t):
        assert rec['id']==r['id'];assert r['points']==old[r['id']]['points'] and r['valid']==old[r['id']]['valid']
        g=np.array(r['points']);a,u,b,v=rec['mask'];assert np.array_equal(r['hidden'],(g[:,0]>=a)&(g[:,0]<b)&(g[:,1]>=u)&(g[:,1]<v))
    maximum=dict(mixture_px=0.,metric=0.);checked=0;oracle={}
    for split in ('synth_val','real'):
        for condition in CONDITIONS:
            key=f'{split}/{condition}';m=read(V11/'cache'/split/condition/'manifest.json');t=read(V11/'targets'/f'{split}_{condition}.json')['rows']
            truth=np.array([r['points'] for r in t]);valid=np.array([r['valid'] for r in t]);kind=target_type(t).numpy()
            det=np.array([r['detected'] for r in m['rows']]);diag=np.linalg.norm([r['original_hw'] for r in m['rows']],axis=-1)
            for name,expected in m.get('chunk_sha256',{}).items():assert sha(V11/'cache'/split/condition/name)==expected
            stock=np.load(RUN/'predictions'/split/condition/'stock.npz')['points'];diagrows={}
            for path in (RUN/'predictions'/split/condition).glob('*.npz'):
                p=np.load(path);assert p['ids'].tolist()==[r['id'] for r in t];assert np.array_equal(det,p['detected'])
                assert np.array_equal(stock,p['original_points'],equal_nan=True);pred=p['points'].astype(float)
                assert np.isfinite(pred[det]).all() and np.isnan(pred[~det]).all()
                e=np.sqrt(((pred-truth)**2).sum(-1));frame=[e[i,valid[i]].mean() for i in np.flatnonzero(det)]
                full=sum(float(np.minimum(e[i,valid[i]]/diag[i],1).mean()) if det[i] else 1. for i in range(len(t)))/len(t)
                q=summary['results'][key][path.stem];err=max(abs(np.mean(frame)-q['frame_mean_px']),abs(full-q['full_normalized_mean']))
                assert err<1e-9;maximum['metric']=max(maximum['metric'],err)
                for typ,k in [('hidden',1),('outside',2)]:
                    mask=(kind==k)&valid&det[:,None]
                    if mask.any():assert abs(e[mask].mean()-q['strata'][typ]['mean_px'])<1e-9
                if 'keep' in p:
                    seed=int(path.stem.split('_seed')[1]);base=np.load(path.with_name(f'local_seed{seed}.npz'))['points']
                    assert np.array_equal(base,p['base_points'],equal_nan=True)
                    keep=p['keep'][det].astype(float);assert ((keep>=0)&(keep<=1)).all()
                    rec=stock[det].astype(float)+keep*(base[det].astype(float)-stock[det].astype(float))+p['recovery'][det].astype(float)
                    err=float(np.max(np.abs(rec-pred[det])));assert err<.001;maximum['mixture_px']=max(maximum['mixture_px'],err)
                    prob=p['probabilities'][det];assert np.allclose(prob.sum(-1),1,atol=1e-6)
                    if path.stem.startswith(('learned_gate','gated_amodal')):
                        # Privileged upper diagnostic; not a prediction file or acceptance metric.
                        privileged=stock[det].astype(float)+(kind[det]==0)[...,None]*(base[det]-stock[det])
                        if path.stem.startswith('gated_amodal'):
                            experts=p['experts'][det];privileged+=(kind[det]==1)[...,None]*experts[:,:,0]+(kind[det]==2)[...,None]*experts[:,:,1]
                        pe=np.linalg.norm(privileged-truth[det],axis=-1);vr=valid[det];typ=kind[det]
                        diagrows[path.stem]=dict(actual_mean_px=q['frame_mean_px'],oracle_mean_px=float(np.mean([r[v].mean() for r,v in zip(pe,vr)])),
                            oracle_hidden_px=float(pe[vr&(typ==1)].mean()) if (vr&(typ==1)).any() else None,
                            oracle_outside_px=float(pe[vr&(typ==2)].mean()) if (vr&(typ==2)).any() else None)
                checked+=int(det.sum())
            oracle[key]=diagrows
    write(RUN/'AUDIT.json',dict(status='PASS',prediction_frame_rows=checked,maxima=maximum,same_training_initialization_and_plan=True,
        training_sources_unchanged=True,distinct_seed_checkpoints=True,synthetic_corner_targets_exact=True,same_baseline_detections=True))
    write(RUN/'ORACLE_ROUTING_DIAGNOSTIC.json',dict(privileged=True,posthoc=True,rows=oracle,
        scope='Replace predicted mixture routing with known artificial exposure AFTER predictions and primary scores. Uses GT-derived exposure. Not deployable and not an improvement claim.'))
    print('AUDIT PASS',checked,maximum,flush=True)

if __name__=='__main__':main()
