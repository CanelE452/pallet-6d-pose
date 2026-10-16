"""Independent float64 vector/radius/error replay, sources and GT bindings."""
from context18 import *
from eval18 import prediction_path

def main():
    s=read(RUN/'SUMMARY.json');tr=read(RUN/'TRAIN_PROTOCOL.json');maximum=0.;metric=0.;frames=0;vectors=0
    assert sha(RUN/'PROTOCOL.json')==tr['protocol_sha256']
    for key in ('sources','targets','parents'):
        for p,h in tr[key].items():assert sha(ROOT/p)==h,(key,p)
    for p,h in read(V.RUN/'PROTOCOL.json')['parent_checkpoints'].items():assert sha(ROOT/p)==h
    for p,h in read(RUN/'EVALUATION_BINDINGS.json')['files'].items():assert sha(ROOT/p)==h
    for seed in SEEDS:
        starts=[read(RUN/'models'/f'{a}_seed{seed}'/'START.json') for a in ARMS]
        for k in ('initial_sha256','plan_sha256','ids_sha256'):assert len({x[k] for x in starts})==1
        for arm in ARMS:
            d=RUN/'models'/f'{arm}_seed{seed}';trace=[json.loads(x) for x in (d/'TRACE.jsonl').read_text().splitlines()];assert len(trace)==2000 and all(x['step']==i+1 and all(np.isfinite(x[k]) for k in ('loss','point_error','harm','gradient_norm')) for i,x in enumerate(trace));assert sha(d/'final.pt')==read(d/'COMPLETE.json')['sha256']
    for key,rows in s['results'].items():
        split,condition=key.split('/');m=manifest(split,condition);t=targets(split,condition);gt=np.array([x['points'] for x in t]);valid=np.array([x['valid'] for x in t]);det=np.array([r['detected'] for r in m['rows']]);bb=np.array([r['box'] for r in m['rows'] if r['detected']]);radius=RADIUS*np.linalg.norm(np.maximum(bb[:,2:]-bb[:,:2],1),axis=-1)
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
            if 'raw_vector' not in pred:continue
            raw=pred['raw_vector'][det].astype(float);delta=raw/np.sqrt(1+(raw*raw).sum(-1,keepdims=True))*radius[:,None,None];base=pred['base'][det].astype(float)
            discrepancy=float(np.abs(base+delta-points[det]).max());maximum=max(maximum,discrepancy);assert discrepancy<.001;assert np.all(np.linalg.norm(delta,axis=-1)<=radius[:,None]+1e-6)
            step=points[det]-base;direction=gt[det]-base;identity=(step*step).sum(-1)-2*(step*direction).sum(-1);actual=error[det]**2-(direction*direction).sum(-1);assert np.max(np.abs(identity-actual))<1e-6;vectors+=int(det.sum())
    write(RUN/'AUDIT.json',dict(status='PASS',prediction_frame_rows=int(frames),vector_frame_rows=vectors,vector_replay_max_px=maximum,metric_replay_max=metric,all24000updates_finite=True,same_initialization_and_batches=True,original_manual_GT_and_shifts_verified=True,squared_error_direction_overshoot_identity=True,parents_unchanged=True))
    print('AUDIT PASS',frames,vectors,maximum,metric,flush=True)

if __name__=='__main__':main()
