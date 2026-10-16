"""Independent float64 vector/radius/error replay, sources and GT bindings."""
from context20 import *
from eval20 import prediction_path

def main():
    s=read(RUN/'SUMMARY.json');tr=read(RUN/'TRAIN_PROTOCOL.json');maximum=0.;metric=0.;frames=0;vectors=0
    assert sha(RUN/'PROTOCOL.json')==tr['protocol_sha256']
    for key in ('sources','targets','parents'):
        for p,h in tr[key].items():assert sha(ROOT/p)==h,(key,p)
    assert sha(T.KIT/'model19.py')==tr['model19_sha256'];assert sha(T.KIT/'context19.py')==tr['context19_sha256']
    for p,h in read(V.RUN/'PROTOCOL.json')['parent_checkpoints'].items():assert sha(ROOT/p)==h
    for p,h in read(RUN/'EVALUATION_BINDINGS.json')['files'].items():assert sha(ROOT/p)==h
    assert sha(Path(__file__).with_name('eval20.py'))==read(RUN/'EVAL_PROTOCOL.json')['source_sha256']
    for seed in SEEDS:
        starts=[read(RUN/'models'/f'{a}_seed{seed}'/'START.json') for a in ARMS]
        for k in ('initial_sha256','plan_sha256','ids_sha256'):assert len({x[k] for x in starts})==1
        for arm in ARMS:
            d=RUN/'models'/f'{arm}_seed{seed}';trace=[json.loads(x) for x in (d/'TRACE.jsonl').read_text().splitlines()];assert len(trace)==2000 and all(x['step']==i+1 and all(np.isfinite(x[k]) for k in ('loss','point_error','gradient_norm')) for i,x in enumerate(trace));assert sha(d/'final.pt')==read(d/'COMPLETE.json')['sha256']
    assert sha(KIT/'resume20.py')==read(RUN/'RESUME_PROTOCOL.json')['source_sha256'];assert read(RUN/'REPRODUCTION.json')['status']=='PASS'
    train_ids=set(read(RUN/'TRAIN_GROUPS_seed1.json')['groups'])
    for split in ('real','synth_val'):
        assert not train_ids.intersection(t['id'] for t in targets(split,'clean'))
    for key,rows in s['results'].items():
        split,condition=key.split('/');m=manifest(split,condition);t=targets(split,condition);gt=np.array([x['points'] for x in t]);valid=np.array([x['valid'] for x in t]);det=np.array([r['detected'] for r in m['rows']])
        if split=='real':
            for r,x in zip(m['rows'],t):
                assert sha(ROOT/r['label'])==r['label_sha256'];doc=read(ROOT/r['label']);ann=doc['objects'][0]['keypoint_annotations'][:8]
                mask=[a.get('source')=='manual_click' and a.get('xy') is not None for a in ann];original=np.array([a['xy'] if a.get('xy') is not None else [0.,0.] for a in ann]);assert mask==x['valid'];assert np.allclose(original-np.array(r['origin']),x['points'],atol=1e-9,rtol=0)
        for name,reported in rows.items():
            pred=np.load(prediction_path(split,condition,name));assert pred['ids'].tolist()==[x['id'] for x in t]==[x['id'] for x in m['rows']];assert np.array_equal(pred['detected'],det);points=pred['points'].astype(float);assert np.isfinite(points[det]).all();error=np.linalg.norm(points-gt,axis=-1)
            frame=np.where(valid[det],error[det],0).sum(-1)/valid[det].sum(-1);diff=abs(frame.mean()-reported['frame_mean_px']);metric=max(metric,diff);assert diff<1e-9
            diag=np.linalg.norm(np.array([x['original_hw'] for x in m['rows']]),axis=-1);normalized=np.ones(len(det));normalized[det]=np.where(valid[det],np.minimum(error[det]/diag[det,None],1),0).sum(-1)/valid[det].sum(-1);assert abs(normalized.mean()-reported['full_normalized_mean'])<1e-9;frames+=len(det)
            for typ in ('hidden','outside'):
                mask=valid&det[:,None]&np.array([x[typ] for x in t])
                if mask.any():assert abs(error[mask].mean()-reported['strata'][typ]['mean_px'])<1e-9
            if 'alpha' not in pred:continue
            if name.startswith('raw_weighted_'):
                control=np.load(T.RUN/'predictions'/split/condition/(name.replace('raw_weighted','line_bce')+'.npz'));assert np.allclose(pred['points'],control['points'],atol=.001,rtol=0,equal_nan=True);assert np.allclose(pred['alpha'],control['alpha'],atol=1e-4,rtol=0,equal_nan=True)
            alpha=pred['alpha'][det].astype(float);delta=pred['proposal_delta'][det].astype(float);base=pred['base'][det].astype(float);old=np.load(Q.RUN/'predictions'/split/condition/f"line_vector_seed{name.rsplit('_seed',1)[1]}.npz")
            assert np.array_equal(pred['base'][det],old['base'][det]);assert np.array_equal(pred['proposal_delta'][det],old['points'][det]-old['base'][det]);assert ((alpha>=0)&(alpha<=1)).all()
            discrepancy=float(np.abs(base+alpha[...,None]*delta-points[det]).max());maximum=max(maximum,discrepancy);assert discrepancy<.001
            residual=gt[det]-base;optimal=np.clip(np.einsum('ijk,ijk->ij',residual,delta)/np.maximum(np.einsum('ijk,ijk->ij',delta,delta),1e-12),0,1);best=np.linalg.norm(base+optimal[...,None]*delta-gt[det],axis=-1)
            assert np.all(best<=np.linalg.norm(residual,axis=-1)+1e-8);assert np.all(best<=np.linalg.norm(base+delta-gt[det],axis=-1)+1e-8);assert np.all(best<=error[det]+.001);vectors+=int(det.sum())
    write(RUN/'AUDIT.json',dict(status='PASS',prediction_frame_rows=int(frames),vector_frame_rows=vectors,retention_replay_max_px=maximum,metric_replay_max=metric,all36000updates_finite=True,same_initialization_and_batches=True,original_manual_GT_and_shifts_verified=True,frozen_proposal_and_alpha_bounds=True,oracle_optimality=True,parents_unchanged=True))
    print('AUDIT PASS',frames,vectors,maximum,metric,flush=True)

if __name__=='__main__':main()
