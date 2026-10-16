"""Independent saved-coordinate, metric, source and manual GT audit."""
from context16 import *
from eval16 import prediction_path

def main():
    s=read(RUN/'SUMMARY.json');p=read(RUN/'PROTOCOL.json');train=read(RUN/'TRAIN_PROTOCOL.json');frames=0;selections=0;maximum=0.;metric=0.
    for path,h in read(V.RUN/'PROTOCOL.json')['parent_checkpoints'].items():assert sha(ROOT/path)==h
    for group in ('sources','targets','parents'):
        for path,h in train[group].items():assert sha(ROOT/path)==h,(group,path)
    for path,h in read(RUN/'EVALUATION_BINDINGS.json')['files'].items():assert sha(ROOT/path)==h
    finals=[]
    for seed in SEEDS:
        starts=[read(RUN/'models'/f'{a}_seed{seed}'/'START.json') for a in ARMS]
        for k in ('initial_sha256','plan_sha256','ids_sha256'):assert len({r[k] for r in starts})==1
        for arm in ARMS:
            d=RUN/'models'/f'{arm}_seed{seed}';trace=[json.loads(x) for x in (d/'TRACE.jsonl').read_text().splitlines()]
            assert len(trace)==2000 and all(x['step']==i+1 and np.isfinite(x['loss']) and np.isfinite(x['gradient_norm']) for i,x in enumerate(trace));finals.append(sha(d/'final.pt'))
    assert len(set(finals))==15
    for split in ('synth_val','real'):
        for condition in CONDITIONS:
            m=manifest(split,condition);t=targets(split,condition);gt=np.array([r['points'] for r in t]);valid=np.array([r['valid'] for r in t]);rows=s['results'][split+'/'+condition]
            if condition in NEW_CONDITIONS:
                original={r['id']:r for r in targets(split,'clean')}
                for r,x in zip(m['rows'],t):
                    src=original[r['id']];assert src['valid']==x['valid'];assert np.allclose(np.array(src['points'])-r['origin'],x['points'],atol=1e-9,rtol=0)
                    if split=='real':assert sha(ROOT/r['label'])==r['label_sha256']
            for name,reported in rows.items():
                path=prediction_path(split,condition,name);pred=np.load(path);det=pred['detected'];points=pred['points'].astype(float);assert np.isfinite(points[det]).all()
                assert pred['ids'].tolist()==[r['id'] for r in t]==[r['id'] for r in m['rows']]
                assert np.array_equal(det,[r['detected'] for r in m['rows']]);error=np.linalg.norm(points-gt,axis=-1)
                means=np.where(valid[det],error[det],0).sum(-1)/valid[det].sum(-1);diff=abs(float(means.mean())-reported['frame_mean_px']);metric=max(metric,diff);assert diff<1e-9
                diag=np.linalg.norm(np.array([r['original_hw'] for r in m['rows']]),axis=-1);norm=np.ones(len(det));norm[det]=np.where(valid[det],np.minimum(error[det]/diag[det,None],1),0).sum(-1)/valid[det].sum(-1)
                assert abs(norm.mean()-reported['full_normalized_mean'])<1e-9;frames+=len(det)
                for typ in ('hidden','outside'):
                    mask=valid&np.array([r[typ] for r in t])&det[:,None]
                    if mask.any():assert abs(error[mask].mean()-reported['strata'][typ]['mean_px'])<1e-9
                if 'candidates' not in pred:continue
                choice=pred['choice'][det].astype(int);eligible=pred['eligible'][det]>.5
                assert np.array_equal(choice,np.where(eligible,pred['gain'][det],-np.inf).argmax(-1));assert (pred['gain'][det,:,0]==0).all()
                replay=pred['candidates'][det][np.arange(det.sum())[:,None],np.arange(8)[None],choice];assert np.array_equal(replay,pred['points'][det]);selections+=det.sum()
                arm=name.rsplit('_seed',1)[0];seed=int(name.rsplit('_seed',1)[1]);rep=repcache(split,condition,seed,'actual')
                b=rep['base'].numpy().astype(float);a=rep['amodal'].numpy().astype(float);ld=rep['line_delta'].numpy().astype(float)
                basecand=np.stack((b,(b+a)/2,a,b+.5*ld,b+ld),2)
                weights=rep['line_weights'].numpy()[...,:-1];ix=np.argsort(-weights,axis=-1,kind='stable')[...,:3];l=rep['lines'].numpy().astype(float)
                picked=l[np.arange(len(b))[:,None,None],ix];normals=picked[...,:2]/np.maximum(np.linalg.norm(picked[...,:2],axis=-1,keepdims=True),1e-8);constant=picked[...,2]/np.maximum(np.linalg.norm(picked[...,:2],axis=-1),1e-8)
                bb=np.array([r['box'] for r in m['rows'] if r['detected']]);radius=.05*np.maximum(np.linalg.norm(bb[:,2:]-bb[:,:2],axis=-1),1)
                distance=(b[:,:,None]*normals).sum(-1)+constant;extra=b[:,:,None]-np.clip(distance,-radius[:,None,None],radius[:,None,None])[...,None]*normals
                reconstructed=np.concatenate((basecand,extra),2);err=float(np.abs(reconstructed-pred['candidates'][det]).max());maximum=max(maximum,err);assert err<.001
    write(RUN/'AUDIT.json',dict(status='PASS',prediction_frame_rows=int(frames),selection_frame_rows=int(selections),candidate_replay_max_px=maximum,metric_replay_max=metric,
        parents_and_training_sources_unchanged=True,all30000updates_finite=True,all_arms_same_initialization_and_plan=True,new_gt_shifts_and_manual_hashes_verified=True,exact_argmax=True))
    print('AUDIT PASS',frames,selections,maximum,metric,flush=True)

if __name__=='__main__':main()
