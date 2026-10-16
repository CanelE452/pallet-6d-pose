"""Independent candidate reconstruction, readout and metric verification."""
from context13 import *

def main():
    setup();s=read(RUN/'SUMMARY.json');train=read(RUN/'TRAIN_PROTOCOL.json');protocol=read(RUN/'PROTOCOL.json')
    for path,expected in train['sources'].items():assert sha(ROOT/path)==expected,path
    for key,expected in protocol['parent_checkpoint_sha256'].items():
        arm,seed=key.rsplit('_',1);assert sha(V12/'models'/f'{arm}_seed{seed}'/'final.pt')==expected
    for seed in SEEDS:
        starts=[read(RUN/'models'/f'{arm}_seed{seed}'/'START.json') for arm in ARMS]
        assert len({r['initial_state_sha256'] for r in starts})==len({r['plan_sha256'] for r in starts})==1
    for arm in ARMS:
        finals=[]
        for seed in SEEDS:
            directory=RUN/'models'/f'{arm}_seed{seed}';c=read(directory/'COMPLETE.json');assert c['complete'] and c['steps']==2000
            assert sha(directory/'final.pt')==c['checkpoint_sha256'];finals.append(c['checkpoint_sha256'])
            rows=[json.loads(r) for r in (directory/'TRACE.jsonl').read_text().splitlines()]
            assert len(rows)==2000 and all(r['step']==i+1 and np.isfinite([r['loss'],r['regression'],r['ranking'],r['gradient_norm']]).all() for i,r in enumerate(rows))
        assert len(set(finals))==3
    maximum=dict(candidate_replay_px=0.,metric_replay=0.);count=0;selection_count=0
    for split in ('synth_val','real'):
        for condition in CONDITIONS:
            key=f'{split}/{condition}';m=read(V11/'cache'/split/condition/'manifest.json');targets=read(V11/'targets'/f'{split}_{condition}.json')['rows']
            gt=np.array([r['points'] for r in targets]);valid=np.array([r['valid'] for r in targets]);det=np.array([r['detected'] for r in m['rows']])
            diag=np.linalg.norm([r['original_hw'] for r in m['rows']],axis=-1);root=RUN/'predictions'/split/condition;stock=np.load(root/'stock.npz')['points']
            for path in root.glob('*.npz'):
                p=np.load(path);assert p['ids'].tolist()==[r['id'] for r in targets];assert np.array_equal(det,p['detected'])
                assert np.array_equal(p['original_points'],stock,equal_nan=True);assert np.isfinite(p['points'][det]).all() and np.isnan(p['points'][~det]).all()
                error=np.sqrt(((p['points'].astype(float)-gt)**2).sum(-1));frame=[error[i,valid[i]].mean() for i in np.flatnonzero(det)]
                full=sum(float(np.minimum(error[i,valid[i]]/diag[i],1).mean()) if det[i] else 1. for i in range(len(det)))/len(det)
                summary=s['results'][key][path.stem];err=max(abs(np.mean(frame)-summary['frame_mean_px']),abs(full-summary['full_normalized_mean']))
                assert err<1e-9;maximum['metric_replay']=max(maximum['metric_replay'],err)
                if 'gain' in p:
                    predgain=p['gain'][det];choice=p['choice'][det].astype(int);cand=p['candidates'][det]
                    assert np.array_equal(predgain[:,:,0],np.zeros_like(predgain[:,:,0]));assert np.array_equal(choice,predgain.argmax(-1))
                    selected=np.take_along_axis(cand,choice[:,:,None,None],axis=2)[:,:,0,:]
                    assert np.array_equal(selected,p['points'][det])
                    replay=p['base'][det].astype(float)[:,:,None]+np.array(ALPHAS)[None,None,:,None]*(p['full'][det].astype(float)-p['base'][det].astype(float))[:,:,None]
                    err=float(np.abs(cand-replay).max());assert err<.001;maximum['candidate_replay_px']=max(maximum['candidate_replay_px'],err)
                    best=np.linalg.norm(cand-gt[det,:,None],axis=-1).min(-1)
                    assert (best[valid[det]]<=error[det][valid[det]]+1e-9).all()
                    selection_count+=int(det.sum())
                count+=int(det.sum())
    write(RUN/'AUDIT.json',dict(status='PASS',prediction_frame_rows=count,selection_frame_rows=selection_count,maxima=maximum,
        exact_argmax_and_selected_candidate_replay=True,parents_unchanged=True,training_sources_unchanged=True,
        same_initialization_and_sample_plan=True,distinct_checkpoints=True,original_gt_used_without_reassignment=True))
    write(RUN/'JOINT_FOLLOWUP_DECISION.json',dict(stage1_pass=s['stage1_pass'],joint_training_required=s['stage1_pass'],
        joint_training_executed=False,reason='Complete preregistered stage1 gate required before joint followup; inspect SUMMARY.json for per-seed failures.'))
    print('AUDIT PASS',count,selection_count,maximum,flush=True)

if __name__=='__main__':main()
