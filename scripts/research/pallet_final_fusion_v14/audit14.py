"""Independent numerical replay, target provenance and immutable-source checks."""
from context14 import *
from eval14 import BASELINES,path_for

def main():
    summary=read(RUN/'SUMMARY.json');protocol=read(RUN/'PROTOCOL.json');train=read(RUN/'TRAIN_PROTOCOL.json');maximum=0.;frames=0;selected_rows=0;metric_max=0.
    for path,expected in protocol['parent_checkpoints'].items():assert sha(ROOT/path)==expected
    for path,expected in train['sources'].items():assert sha(ROOT/path)==expected,(path,'training source changed')
    for path,expected in read(RUN/'EVALUATION_BINDINGS.json')['files'].items():assert sha(ROOT/path)==expected
    for path,expected in train['input_files'].items():assert sha(ROOT/path)==expected
    init={};plans={};finals=[]
    for seed in SEEDS:
        starts=[read(RUN/'models'/f'{a}_seed{seed}'/'START.json') for a in ARMS]
        assert len({s['initial_state_sha256'] for s in starts})==1
        assert starts[1]['plan_sha256']==starts[2]['plan_sha256'] and starts[1]['ids_sha256']==starts[2]['ids_sha256']
        for arm in ARMS:
            d=RUN/'models'/f'{arm}_seed{seed}';trace=[json.loads(x) for x in (d/'TRACE.jsonl').read_text().splitlines()]
            assert len(trace)==2000 and all(r['step']==i+1 and np.isfinite(r['loss']) and np.isfinite(r['gradient_norm']) for i,r in enumerate(trace))
            finals.append(sha(d/'final.pt'))
    assert len(set(finals))==9
    for split in ('synth_val','real'):
        for condition in CONDITIONS:
            manifest=read(V11/'cache'/split/condition/'manifest.json');t=target_rows(split,condition)
            gt=np.array([r['points'] for r in t]);valid=np.array([r['valid'] for r in t]);rows=summary['results'][split+'/'+condition]
            for name,reported in rows.items():
                arm='stock' if name=='stock' else name.rsplit('_seed',1)[0];seed=1 if name=='stock' else int(name.rsplit('_seed',1)[1]);path=path_for(split,condition,arm,seed);p=np.load(path)
                assert p['ids'].tolist()==[r['id'] for r in t];det=p['detected'];points=p['points'].astype(float);assert np.isfinite(points[det]).all()
                error=np.linalg.norm(points-gt,axis=-1);frame=np.where(valid[det],error[det],0).sum(-1)/valid[det].sum(-1)
                diff=abs(float(frame.mean())-reported['frame_mean_px']);metric_max=max(metric_max,diff);assert diff<1e-9
                diag=np.linalg.norm(np.array([r['original_hw'] for r in manifest['rows']]),axis=-1);normalized=np.ones(len(det));normalized[det]=np.where(valid[det],np.minimum(error[det]/diag[det,None],1),0).sum(-1)/valid[det].sum(-1)
                assert abs(normalized.mean()-reported['full_normalized_mean'])<1e-9;frames+=len(det)
                for typ in ('hidden','outside'):
                    mask=valid&np.array([r[typ] for r in t])&det[:,None]
                    if mask.any():assert abs(error[mask].mean()-reported['strata'][typ]['mean_px'])<1e-9
                if arm not in ARMS and not arm.startswith('fusion_'):continue
                choices=p['choice'][det].astype(int);eligible=p['eligible'][det]>.5
                replay_choice=np.where(eligible,p['gain'][det],-np.inf).argmax(-1);assert np.array_equal(choices,replay_choice)
                assert (p['gain'][det,:,0]==0).all()
                replay=p['candidates'][det][np.arange(det.sum())[:,None],np.arange(8)[None],choices]
                assert np.array_equal(replay,p['points'][det]);selected_rows+=int(det.sum())
                variant='donor' if arm=='fusion_donor' else 'actual';rep=repcache(split,condition,seed,variant)
                b=rep['base'].numpy().astype(float);a=rep['amodal'].numpy().astype(float);l=rep['line_delta'].numpy().astype(float)
                cand=np.stack((b,(b+a)/2,a,b+.5*l,b+l),2);err=np.abs(cand-p['candidates'][det]).max();maximum=max(maximum,float(err));assert err<.001
                if arm=='fusion_donor':
                    original=repcache(split,condition,seed)
                    for k in ('base','amodal','context','probabilities'):assert torch.equal(rep[k],original[k])
            # Independently recover all original manual target coordinates and IDs.
            if split=='real':
                for r,target in zip(manifest['rows'],t):
                    label=ROOT/r['label'];assert sha(label)==r['label_sha256'];ann=read(label)['objects'][0]['keypoint_annotations'][:8]
                    original=np.array([a['xy'] if a.get('xy') is not None else [0.,0.] for a in ann]);v=np.array([a.get('source')=='manual_click' and a.get('xy') is not None for a in ann])
                    assert np.array_equal(v,target['valid']);assert np.allclose(original-np.array(r['origin']),target['points'],atol=1e-9,rtol=0)
    # New train targets are exact shifts of original synthetic labels, not reassigned points.
    for split,original in [('insample','train'),('heldout','calibration')]:
        source={r['id']:r for r in read(V10/f'targets_{original}.json')['rows']}
        for condition in CONDITIONS:
            m=read(RUN/'cache'/split/condition/'manifest.json')
            for r,t in zip(m['rows'],target_rows(split,condition)):
                assert r['id']==t['id'];s=source[r['id']];assert s['valid']==t['valid']
                assert np.allclose(np.array(s['points'])-np.array(r['origin']),t['points'],atol=1e-9,rtol=0)
    write(RUN/'AUDIT.json',dict(status='PASS',prediction_frame_rows=frames,selection_frame_rows=selected_rows,candidate_replay_max_px=maximum,metric_replay_max=metric_max,
        parents_unchanged=True,training_sources_unchanged=True,heldout_control_same_initialization_and_plan=True,finite_all18000updates=True,
        manual_gt_hashes_and_coordinate_shifts=True,synthetic_target_shifts=True,exact_candidate_selection=True,donor_preserves_point_and_amodal_inputs=True))
    print('AUDIT PASS',frames,selected_rows,maximum,metric_max,flush=True)

if __name__=='__main__':main()
