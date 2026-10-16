"""Independent replay of coordinate updates, label transforms and summaries."""
import copy
import numpy as np
from common import *


def main():
    setup();summary=read(RUN/'SUMMARY.json');train=read(RUN/'TRAIN_PROTOCOL.json')
    for name,expected in train['sources'].items():assert sha(ROOT/name)==expected,name
    pop=read(RUN/'POPULATIONS.json');repair=read(RUN/'IMAGE_LOCATOR_REPAIR.json')
    restored=copy.deepcopy(pop);old={r['id']:r['old'] for r in repair['repairs']}
    for r in restored['real']:
        if r['id'] in old:r['image']=old[r['id']]
    original=digest(json.dumps(restored,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    assert original==read(RUN/'PROTOCOL.json')['populations_sha256']==repair['previous_population_sha256']
    assert len({r['id'] for r in pop['real']})==297
    train_images={r['source_image_sha256'] for r in read(EXPORT/'train.json')['records']}
    synth_images={r['source_image_sha256'] for r in read(EXPORT/'synth_val.json')['records']}
    real_images={sha(ROOT/r['image']) for r in pop['real']}
    assert not train_images&synth_images and not train_images&real_images
    maxima=dict(coordinate_replay_px=0.,matrix_replay=0.,metric_replay=0.,target_transform_px=0.)
    rows_checked=0;point_rows=0;gate_stats={};state_shas={}
    for arm in ARMS:
        state_shas[arm]=[]
        for seed in SEEDS:
            cell=RUN/'models'/f'{arm}_seed{seed}';receipt=read(cell/'COMPLETE.json')
            assert receipt['complete'] and receipt['steps']==2000 and sha(cell/'final.pt')==receipt['checkpoint_sha256']
            trace=[json.loads(x) for x in (cell/'TRACE.jsonl').read_text().splitlines()]
            assert [r['step'] for r in trace]==list(range(1,2001))
            assert all(np.isfinite([r['loss'],r['grad_norm'],r['dht_grad_norm']]).all() for r in trace)
            if arm=='lines_joint':assert max(r['dht_grad_norm'] for r in trace)>0
            state_shas[arm].append(receipt['checkpoint_sha256'])
        assert len(set(state_shas[arm]))==3
    for split in ('train','synth_val','real'):
        original_targets={r['id']:r for r in read(V10/f'targets_{split}.json')['rows']} if split!='real' else {}
        for condition in CONDITIONS:
            if split=='train' and condition=='clean':continue
            directory=RUN/'cache'/split/condition;manifest=read(directory/'manifest.json')
            for name,expected in manifest['chunk_sha256'].items():assert sha(directory/name)==expected
            targets=read(RUN/'targets'/f'{split}_{condition}.json')['rows'];truth=np.array([r['points'] for r in targets])
            valid=np.array([r['valid'] for r in targets]);assert all(r['id']==t['id'] for r,t in zip(manifest['rows'],targets))
            for rec,t in zip(manifest['rows'],targets):
                if split=='real':
                    assert sha(ROOT/rec['label'])==rec['label_sha256']
                    ann=read(ROOT/rec['label'])['objects'][0]['keypoint_annotations'][:8]
                    p=np.array([a['xy'] if a.get('xy') is not None else [0,0] for a in ann])
                    v=np.array([a.get('source')=='manual_click' and a.get('xy') is not None for a in ann])
                else:p=np.array(original_targets[rec['id']]['points']);v=np.array(original_targets[rec['id']]['valid'])
                assert np.array_equal(v,t['valid'])
                err=np.max(np.abs(p-np.array(rec['origin'])-np.array(t['points'])))
                maxima['target_transform_px']=max(maxima['target_transform_px'],float(err));assert err==0
                h,w=rec['image_hw'];shifted=np.array(t['points'])
                outside=(shifted[:,0]<0)|(shifted[:,0]>=w)|(shifted[:,1]<0)|(shifted[:,1]>=h)
                assert np.array_equal(outside,t['outside'])
                point_rows+=1
            if split=='train':continue
            scores=summary['results'][f'{split}/{condition}'];stock=np.load(RUN/'predictions'/split/condition/'stock.npz')
            mask=np.array([r['detected'] for r in manifest['rows']]);diag=np.linalg.norm([r['original_hw'] for r in manifest['rows']],axis=-1)
            for path in (RUN/'predictions'/split/condition).glob('*.npz'):
                p=np.load(path);assert np.array_equal(p['detected'],mask)
                assert np.array_equal(p['original_points'],stock['points'],equal_nan=True)
                assert np.isfinite(p['points'][mask]).all() and np.isnan(p['points'][~mask]).all()
                errors=np.sqrt(np.sum((p['points'][mask].astype(float)-truth[mask])**2,axis=-1))
                weights=valid[mask]/valid[mask].sum(-1,keepdims=True)
                mean=float((errors*weights).sum(-1).mean())
                full=float(((np.minimum(errors/diag[mask,None],1)*weights).sum()+(~mask).sum())/len(mask))
                err=max(abs(mean-scores[path.stem]['frame_mean_px']),abs(full-scores[path.stem]['full_normalized_mean']))
                maxima['metric_replay']=max(maxima['metric_replay'],err);assert err<1e-9
                if 'base_points' in p:
                    seed=int(path.stem.split('_seed')[1][0]);base=np.load(path.with_name(f'v10_local_seed{seed}.npz'))['points'][mask]
                    assert np.array_equal(base,p['base_points'][mask])
                    gate=p['gate'][mask];assert ((gate>=0)&(gate<=1)).all()
                    A=p['A'][mask].astype(float);proposal=p['proposal'][mask].astype(float)
                    replay=base.astype(float)+(A@proposal[...,None])[...,0]*gate
                    error=float(np.max(np.abs(replay-p['points'][mask])));maxima['coordinate_replay_px']=max(maxima['coordinate_replay_px'],error)
                    assert error<.001
                    if not path.stem.startswith('local_extra'):
                        lines=p['lines'][mask].astype(float);length=np.linalg.norm(lines[...,:2],axis=-1);assert (length>1e-9).all()
                        n=lines[...,:2]/length[...,None];association=p['weights'][mask,:,:-1].astype(float)
                        expected=np.einsum('bpm,bmi,bmj->bpij',association,n,n)
                        error=float(np.max(np.abs(A-expected)));maxima['matrix_replay']=max(maxima['matrix_replay'],error);assert error<2e-5
                        assert np.linalg.eigvalsh(A).min()>-1e-6
                        gate_stats[f'{split}/{condition}/{path.stem}']=dict(median_gate=float(np.median(gate)),median_null=float(np.median(p['weights'][mask,:,-1])),
                            mean_extra_movement_px=float(np.linalg.norm(p['points'][mask]-base,axis=-1).mean()))
                rows_checked+=int(mask.sum())
    write(RUN/'AUDIT.json',dict(status='PASS',prediction_frame_rows=rows_checked,target_frame_rows=point_rows,maxima=maxima,
        exact_preserved_local_outputs=True,independent_metric_replay=True,training_sources_unchanged=True,
        distinct_seed_checkpoints=state_shas,no_exact_train_eval_image_overlap=True,
        population_locator_repair_verified_same_ids=True,current_population_sha256=sha(RUN/'POPULATIONS.json'),
        gate_statistics=gate_stats,scope='Numerical replay and lineage checks, not independent real annotation certification. No retraining or model selection.'))
    print('AUDIT PASS',rows_checked,maxima,flush=True)


if __name__=='__main__':main()
