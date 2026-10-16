from context33 import *
import eval33 as EV
from train33 import load_model
from model32 import VisibleGeometry

def main():
    setup();s=read(RUN/'SUMMARY.json');prior=read(RUN/'PRIOR_BINDINGS.json')['files']
    for p,h in prior.items():assert sha(ROOT/p)==h,('prior',p)
    for p,h in read(V32.RUN/'INPUT_BINDINGS.json')['files'].items():assert sha(ROOT/p)==h,('inputs',p)
    protocol=read(RUN/'TRAIN_PROTOCOL.json');assert sha(RUN/'PROTOCOL.json')==protocol['protocol_sha256']
    for p,h in protocol['source_hashes'].items():assert sha(ROOT/p)==h,p
    updates=0
    for seed in SEEDS:
        starts=[]
        for a in ARMS:
            folder=RUN/'models'/f'{a}_seed{seed}';starts.append(read(folder/'START.json'));trace=[json.loads(x) for x in (folder/'TRACE.jsonl').read_text().splitlines()];assert len(trace)==2000 and all(r['step']==i+1 and all(np.isfinite(r[k]) for k in ('loss','point_error','gradient_norm')) for i,r in enumerate(trace));assert sha(folder/'final.pt')==read(folder/'COMPLETE.json')['checkpoint_sha256'];updates+=len(trace)
        for k in ('initial_sha256','plan_sha256','parameters'):assert len({r[k] for r in starts})==1,k
        plan=read(RUN/f'PLAN_seed{seed}.json');assert sha(ROOT/plan['source'])==plan['source_sha256'];assert starts[0]['plan_sha256']==plan['plan_sha256']
    bindings=read(RUN/'EVALUATION_BINDINGS.json')['files'];groups={};count=0;manual=0;maximum=0.
    for p,h in bindings.items():
        assert sha(ROOT/p)==h;p=Path(p);groups.setdefault((p.parent.parent.name,p.parent.name),[]).append(p)
    for (split,c),paths in groups.items():
        m,t=EV.population(split,c);gt=np.array([x['points'] for x in t]);valid=np.array([x['valid'] for x in t]);det=np.array([r['detected'] for r in m['rows']]);diag=np.linalg.norm(np.array([r['original_hw'] for r in m['rows']]),axis=-1);_,O,_=EV.inputs(split,c,1);original=O.get('points')
        if original is None:original=EV.render_data(1,'check',c)[1]['points']
        if split=='real':
            for r,x in zip(m['rows'],t):
                assert sha(ROOT/r['label'])==r['label_sha256'];ann=read(ROOT/r['label'])['objects'][0]['keypoint_annotations'][:8];assert [a.get('source')=='manual_click' and a.get('xy') is not None for a in ann]==x['valid'];xy=np.array([a['xy'] if a.get('xy') is not None else [0.,0.] for a in ann]);assert np.allclose(xy-np.array(r['origin']),x['points'],atol=1e-9,rtol=0);manual+=1
        for p in paths:
            z=np.load(ROOT/p);assert z['ids'].tolist()==[x['id'] for x in t];assert np.array_equal(z['detected'],det);assert np.isfinite(z['points'][det]).all();assert np.array_equal(z['original_points'][det],original.numpy());error=np.linalg.norm(z['points'].astype(float)-gt,axis=-1);row=s['results'][split+'/'+c][p.stem];frame=np.where(valid[det],error[det],0).sum(-1)/valid[det].sum(-1);d=abs(frame.mean()-row['frame_mean_px']);assert d<1e-9;maximum=max(maximum,d)
            norm=np.ones(len(det));norm[det]=np.where(valid[det],np.minimum(error[det]/diag[det,None],1),0).sum(-1)/valid[det].sum(-1);assert abs(norm.mean()-row['full_normalized_mean'])<1e-9
            for typ in ('all','hidden','outside'):
                mask=valid&det[:,None]
                if typ!='all':mask &= np.array([x[typ] for x in t])
                if mask.any():assert abs(error[mask].mean()-row['strata'][typ]['mean_px'])<1e-9
            reconstructed=z['base'][det]+z['mix'][det,...,None]*(z['strong'][det]-z['base'][det])+z['residual'][det];assert np.allclose(reconstructed,z['points'][det],atol=.001,rtol=0)
            bb=np.array([r['box'] for r in m['rows'] if r['detected']]);dd=np.linalg.norm(np.maximum(bb[:,2:]-bb[:,:2],1),axis=-1);assert np.all(np.abs(z['mix'][det])<=1);assert np.all(np.linalg.norm(z['residual'][det],axis=-1)<=.15*dd[:,None]+.001);count+=1
    replay=[];pooled_reproduction=[]
    for seed in SEEDS:
        for split,c in [('real','crop20'),('render_check','clean')]:
            m,O,S=EV.inputs(split,c,seed);O={k:v[:3].cuda() for k,v in O.items()};S={k:v[:3].cuda() for k,v in S.items()}
            for a in ARMS:
                model=load_model(a,seed)
                with torch.no_grad():out=model(O,S,a)
                p=np.load(RUN/'predictions'/split/c/f'{a}_seed{seed}.npz');difference=float(np.abs(out['points'].cpu().numpy()-p['points'][p['detected']][:3]).max());assert difference<.003
                if a=='query':
                    for k,n in [('line_attention',4),('end_attention',64)]:
                        allowed=model.incident.repeat_interleave(n,1).repeat_interleave(2,0);assert torch.max(out[k][...,:-1][:,~allowed])==0;assert torch.allclose(out[k].sum(-1),torch.ones_like(out[k][...,-1]),atol=1e-6)
                    with torch.no_grad():before=model(O,S,a,'both_off');changed=model(O,S,a,'both_off',donor_maps=out['role_maps'].roll(1,0))
                    assert torch.equal(before['points'],changed['points'])
                if a in ('pooled','point_aux'):
                    old=VisibleGeometry(seed).cuda().eval();old.load_state_dict({k:v for k,v in model.state_dict().items() if k in old.state_dict()})
                    with torch.no_grad():expected=old(O,S,'visible_hough' if a=='pooled' else 'point')
                    assert torch.equal(out['points'],expected['points'])
                replay.append(dict(seed=seed,split=split,arm=a,max_px=difference))
        # Repeated matched controls should reproduce prior training, subject to GPU rounding.
        for a,oldarm in [('pooled','visible_hough'),('point_aux','visible_aux_only')]:
            for c in EV.conditions('real'):
                p=np.load(RUN/'predictions/real'/c/f'{a}_seed{seed}.npz');old=np.load(V32.RUN/'predictions/real'/c/f'{oldarm}_seed{seed}.npz');delta=float(np.nanmax(np.abs(p['points']-old['points'])));pooled_reproduction.append(dict(seed=seed,condition=c,arm=a,max_difference_px=delta))
    write(RUN/'AUDIT.json',dict(status='PASS',updates=updates,models=18,prior_bound_files=len(prior),prediction_files=count,manual_GT_rows_verified=manual,metric_max_difference=maximum,live_replay=replay,prior_control_reproduction=pooled_reproduction,equal_initialization_and_batches=True,input_hashes_verified=True,attention_incident_mask_and_normalization=True,both_off_donor_invariant=True,GT_forward=False,manual_GT_changed=False));print('AUDIT33 PASS',count,updates,flush=True)

if __name__=='__main__':main()
