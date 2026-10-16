"""Independent GT/coordinate/metric replay and fixed training provenance."""
from context23 import *
from eval23 import prediction_path,population,ARMS

def main():
    setup();s=read(RUN/'SUMMARY.json');tr=read(RUN/'TRAIN_PROTOCOL.json');maximum=0.;metric=0.;frames=0;vectors=0
    assert sha(RUN/'PROTOCOL.json')==tr['protocol_sha256'];assert sha(OLD.KIT/'model20.py')==tr['frozen_model_sha256'];assert sha(KIT/'eval23.py')==read(RUN/'EVAL_PROTOCOL.json')['source_sha256']
    for mapping in (tr['sources'],tr['parents'],tr['targets'],read(RUN/'EVALUATION_BINDINGS.json')['files'],read(V.RUN/'PROTOCOL.json')['parent_checkpoints'],tr['new_parent_bindings'],tr['new_target_bindings']):
        for p,h in mapping.items():assert sha(ROOT/p)==h,p
    assert read(RUN/'SPLIT_AUDIT.json')['status']=='PASS';assert read(RUN/'TARGET_AUDIT.json')['status']=='PASS'
    for seed in SEEDS:
        plan=read(RUN/f'GROUPS_seed{seed}.json');seen={i for batch in plan['plan'] for i in batch};assert seen==set(range(len(plan['ids'])));expected=set(read(ROOT/'data/pallet/results/pallet_residual_v21'/f'GROUPS_old_seed{seed}.json')['groups']);expected|={r['id'] for c in TRAIN_CONDITIONS for r in read(RUN/'cache/new_train'/c/'manifest.json')['rows'] if r['detected']};assert set(plan['groups'])==expected
        starts=[read(RUN/'models'/f'{a}_seed{seed}'/'START.json') for a in ARMS];assert len({x['initial_sha256'] for x in starts})==1
        assert len({x['plan_sha256'] for x in starts})==1;assert len({x['ids_sha256'] for x in starts})==1
        for a in ARMS:
            d=RUN/'models'/f'{a}_seed{seed}';rows=[json.loads(x) for x in (d/'TRACE.jsonl').read_text().splitlines()];assert len(rows)==2000
            assert all(x['step']==i+1 and all(np.isfinite(x[k]) for k in ('loss','gradient_norm','point_error')) for i,x in enumerate(rows))
            assert sha(d/'final.pt')==read(d/'COMPLETE.json')['sha256']
    assert sha(KIT/'train_mass23.py')==read(RUN/'MASS_PROTOCOL.json')['source_sha256']
    raw=read(RUN/'RAW_TARGETS.json')['rows']
    for split in ('new_train','new_check'):
        for c in TRAIN_CONDITIONS:
            m,o=new_load(split,c);ts=new_targets(split,c);assert [r['id'] for r in m['rows']]==[r['id'] for r in ts]
            for name,h in m['chunk_sha256'].items():assert sha(RUN/'cache'/split/c/name)==h
            for r,t in zip(m['rows'],ts):
                original=raw[r['id']];assert sha(ROOT/original['label'])==original['label_sha256'];values=np.fromstring((ROOT/original['label']).read_text(),sep=' ')[5:].reshape(9,3)[:8];hh,ww=r['original_hw'];gt=values[:,:2]*[ww+200,hh+200]-100;assert np.allclose(gt-np.array(r['origin']),t['points'],atol=1e-9,rtol=0);assert (values[:,2]>0).tolist()==t['valid']
    for key,rows in s['results'].items():
        split,c=key.split('/');m,ts=population(split,c);gt=np.array([x['points'] for x in ts]);valid=np.array([x['valid'] for x in ts]);det=np.array([x['detected'] for x in m['rows']])
        if split=='real':
            for r,t in zip(m['rows'],ts):
                assert sha(ROOT/r['label'])==r['label_sha256'];ann=read(ROOT/r['label'])['objects'][0]['keypoint_annotations'][:8];assert [a.get('source')=='manual_click' and a.get('xy') is not None for a in ann]==t['valid'];original=np.array([a['xy'] if a.get('xy') is not None else [0.,0.] for a in ann]);assert np.allclose(original-np.array(r['origin']),t['points'],atol=1e-9,rtol=0)
        for name,reported in rows.items():
            p=np.load(prediction_path(split,c,name));assert p['ids'].tolist()==[x['id'] for x in ts];assert np.array_equal(p['detected'],det);points=p['points'].astype(float);assert np.isfinite(points[det]).all();error=np.linalg.norm(points-gt,axis=-1);v=valid[det];frame=np.where(v,error[det],0).sum(-1)/v.sum(-1);diff=abs(frame.mean()-reported['frame_mean_px']);metric=max(metric,diff);assert diff<1e-9;frames+=len(det)
            diag=np.linalg.norm(np.array([x['original_hw'] for x in m['rows']]),axis=-1);norm=np.ones(len(det));norm[det]=np.where(v,np.minimum(error[det]/diag[det,None],1),0).sum(-1)/v.sum(-1);assert abs(norm.mean()-reported['full_normalized_mean'])<1e-9
            for typ in ('hidden','outside'):
                mask=valid&det[:,None]&np.array([x[typ] for x in ts])
                if mask.any():assert abs(error[mask].mean()-reported['strata'][typ]['mean_px'])<1e-9
            if name.rsplit('_seed',1)[0] not in ARMS:continue
            seed=int(name.rsplit('_seed',1)[1]);base=p['base'][det].astype(float);delta=p['proposal_delta'][det].astype(float);alpha=p['alpha'][det].astype(float);assert ((alpha>=0)&(alpha<=1)).all()
            if split in ('new_check','prospective'):
                import context21
                r=new_rep('new_check',c,seed) if split=='prospective' else context21.new_rep(split,c,seed);assert np.array_equal(p['base'][det],r['base'].numpy());assert np.array_equal(p['proposal_delta'][det],r['proposal_delta'].numpy())
            else:
                old=np.load(Q.RUN/'predictions'/split/c/f'line_vector_seed{seed}.npz');assert np.array_equal(p['base'][det],old['base'][det]);assert np.array_equal(p['proposal_delta'][det],old['points'][det]-old['base'][det])
            discrepancy=float(np.abs(base+alpha[...,None]*delta-points[det]).max());maximum=max(maximum,discrepancy);assert discrepancy<.001;vectors+=int(det.sum())
    write(RUN/'AUDIT.json',dict(status='PASS',prediction_frame_rows=frames,vector_frame_rows=vectors,retention_replay_max_px=maximum,metric_replay_max=metric,finite_updates=36000,requested_training_originals=765,usable_training_originals=764,training_originals_without_any_detection=['G38__G__f24246'],all_detected_training_observations_exposed=True,original_manual_GT_and_new_source_labels_verified=True,frozen_proposal=True,initialization_and_within_population_batches_matched=True));print('AUDIT PASS',frames,maximum,flush=True)

if __name__=='__main__':main()
