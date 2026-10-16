"""Independent metrics/GT, artifact integrity, equal-budget and live-forward audit."""
from context32 import *
import eval32 as EV
from model32 import VisibleGeometry
from train32 import load_model
from geometry32 import roi_targets

def main():
    setup();summary=read(RUN/'SUMMARY.json');prior=read(RUN/'PRIOR_BINDINGS.json')['files']
    for p,h in prior.items():assert sha(ROOT/p)==h,('prior',p)
    protocol=read(RUN/'TRAIN_PROTOCOL.json');amendment=read(RUN/'INFERENCE_ABLATION_AMENDMENT.json')
    assert sha(RUN/'PROTOCOL.json')==protocol['protocol_sha256']
    for p,h in protocol['source_hashes'].items():
        if Path(p).name=='model32.py':
            assert h==amendment['old_sha256']==sha(RUN/'model32_training_source.py.txt')
            assert sha(ROOT/p)==amendment['new_sha256']
        else:assert sha(ROOT/p)==h,p
    updates=0
    for seed in SEEDS:
        starts=[]
        for arm in ARMS+('visible_aux_only',):
            folder=RUN/'models'/f'{arm}_seed{seed}';starts.append(read(folder/'START.json'));trace=[json.loads(x) for x in (folder/'TRACE.jsonl').read_text().splitlines()]
            assert len(trace)==2000 and all(r['step']==i+1 and all(np.isfinite(r[k]) for k in ('loss','point_error','gradient_norm')) for i,r in enumerate(trace))
            assert sha(folder/'final.pt')==read(folder/'COMPLETE.json')['checkpoint_sha256'];updates+=len(trace)
        for key in ('initial_sha256','plan_sha256','parameters'):assert len({x[key] for x in starts})==1,key
    checked=0;maxmetric=0.;manual=0
    bindings=read(RUN/'EVALUATION_BINDINGS.json')['files'];by_condition={}
    for path,h in bindings.items():
        assert sha(ROOT/path)==h,path;p=Path(path);by_condition.setdefault((p.parent.parent.name,p.parent.name),[]).append(p)
    for (split,c),paths in by_condition.items():
        m,ts=EV.population(split,c);gt=np.array([t['points'] for t in ts]);valid=np.array([t['valid'] for t in ts]);det=np.array([r['detected'] for r in m['rows']]);diag=np.linalg.norm(np.array([r['original_hw'] for r in m['rows']]),axis=-1)
        mm,O,S=EV.inputs(split,c,1);original=O.get('points')
        if original is None:original=EV.render_data(1,'check',c)[1]['points']
        if split=='real':
            for r,t in zip(m['rows'],ts):
                assert sha(ROOT/r['label'])==r['label_sha256'];ann=read(ROOT/r['label'])['objects'][0]['keypoint_annotations'][:8]
                assert [a.get('source')=='manual_click' and a.get('xy') is not None for a in ann]==t['valid']
                xy=np.array([a['xy'] if a.get('xy') is not None else [0.,0.] for a in ann]);assert np.allclose(xy-np.array(r['origin']),t['points'],atol=1e-9,rtol=0);manual+=1
        for path in paths:
            z=np.load(ROOT/path);assert z['ids'].tolist()==[r['id'] for r in m['rows']];assert np.array_equal(z['detected'],det)
            assert np.isfinite(z['points'][det]).all();assert np.array_equal(z['original_points'][det],original.numpy())
            error=np.linalg.norm(z['points'].astype(float)-gt,axis=-1);reported=summary['results'][split+'/'+c][path.stem]
            frame=np.where(valid[det],error[det],0).sum(-1)/valid[det].sum(-1);d=abs(frame.mean()-reported['frame_mean_px']);maxmetric=max(maxmetric,d);assert d<1e-9
            normalized=np.ones(len(det));normalized[det]=np.where(valid[det],np.minimum(error[det]/diag[det,None],1),0).sum(-1)/valid[det].sum(-1);assert abs(normalized.mean()-reported['full_normalized_mean'])<1e-9
            for typ in ('all','hidden','outside'):
                mask=valid&det[:,None]
                if typ!='all':mask &= np.array([t[typ] for t in ts])
                if mask.any():assert abs(error[mask].mean()-reported['strata'][typ]['mean_px'])<1e-9
            if 'residual' in z:
                wh=np.array([r['box'] for r in m['rows'] if r['detected']]);dd=np.linalg.norm(np.maximum(wh[:,2:]-wh[:,:2],1),axis=-1)
                assert np.max(np.abs(z['mix'][det]))<=1
                assert np.all(np.linalg.norm(z['residual'][det],axis=-1)<=.15*dd[:,None]+.001)
                reconstructed=z['base'][det]+z['mix'][det,...,None]*(z['strong'][det]-z['base'][det])+z['residual'][det]
                assert np.allclose(reconstructed,z['points'][det],atol=.001,rtol=0)
            checked+=1
    # Cropping a physical segment must not invent a physical endpoint at the crop.
    uv=np.column_stack((np.linspace(0,100,101),np.full(101,50.)))
    target,_=roi_targets(dict(segments=[dict(role=0,uv=uv.tolist(),visible=[True]*101)]),np.zeros((8,2)),[0,0,60,100],[20,0],[100,60])
    assert target[:12].sum()>0 and target[12:].sum()==0
    render=read(RUN/'RENDER_COMPLETE.json')['rows'];pairs=0
    for group in ('a0_g08_v0','a0_g08_v2','a1_g08_v0','a1_g08_v2'):
        rr=[next(r for r in render if r['id']==group+'_'+state) for state in ('clean','panel')]
        labels=[read(ROOT/r['label']) for r in rr]
        for a,b in zip(labels[0]['segments'],labels[1]['segments']):
            assert a['segment']==b['segment'] and np.allclose(a['uv'],b['uv'])
            assert not (np.array(b['visible'])&~np.array(a['visible'])).any()
            assert b['endpoint_visible']==[b['visible'][0],b['visible'][-1]]
        pairs+=1
    old={};exec(compile((RUN/'model32_training_source.py.txt').read_text(),'original_model32','exec'),old);replay=[]
    for seed in SEEDS:
        for split,c in [('real','crop20'),('render_check','clean')]:
            m,O,S=EV.inputs(split,c,seed);O={k:v[:3].cuda() for k,v in O.items()};S={k:v[:3].cuda() for k,v in S.items()};model=load_model('visible_hough',seed);original=old['VisibleGeometry'](seed).cuda().eval();original.load_state_dict(model.state_dict())
            with torch.no_grad():out=model(O,S);before=original(O,S);a=model(O,S,'point');b=model(O,S,'point','donor')
            assert torch.equal(out['points'],before['points']);assert torch.equal(a['points'],b['points'])
            p=np.load(RUN/'predictions'/split/c/f'visible_hough_seed{seed}.npz');difference=float(np.max(np.abs(out['points'].cpu().numpy()-p['points'][p['detected']][:3])));assert difference<.003
            replay.append(dict(seed=seed,split=split,max_px=difference,original_training_forward_identical=True))
    write(RUN/'AUDIT.json',dict(status='PASS',updates=updates,models=15,prior_bound_files=len(prior),prediction_files=checked,manual_GT_rows_verified=manual,manual_GT_changed=False,metric_max_difference=maxmetric,live_replay=replay,equal_initialization_and_batches=True,all_training_finite=True,crop_does_not_create_physical_endpoint=True,depth_occlusion_subset_pairs=pairs,original_points_metadata_correct=True,GT_forward=False,target_audit='TARGET_AUDIT.json',limits='Rule-defined physical roles; visibility raster tolerance; no independent human real-line GT; reused DEV.'))
    print('AUDIT32 PASS',checked,updates,flush=True)

if __name__=='__main__':main()
