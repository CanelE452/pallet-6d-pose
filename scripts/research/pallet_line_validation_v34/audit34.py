from context34 import *
import eval34 as EV
from train_dense34 import load_model
from probe34 import load_probe,PAIRS

def main():
    deterministic();s=read(RUN/'SUMMARY.json');previous=read(RUN/'PRIOR_BINDINGS.json')['files']
    for p,h in previous.items():assert sha(ROOT/p)==h,p
    for p,h in read(V32.RUN/'INPUT_BINDINGS.json')['files'].items():assert sha(ROOT/p)==h,p
    for name in ('TRAIN_PROTOCOL.json','CF_PROTOCOL.json'):
        d=read(RUN/name);assert d['protocol_sha256']==sha(RUN/'PROTOCOL.json')
        for p,h in d['source_hashes'].items():assert sha(ROOT/p)==h,p
    for p,h in read(RUN/'DENSE_PROTOCOL.json')['source_hashes'].items():assert sha(ROOT/p)==h,p
    amendment=read(RUN/'IMPLEMENTATION_AMENDMENT.json');assert amendment['completed_optimizer_updates_before_amendment']==0;assert amendment['preflight_sha256']==sha(RUN/'DETERMINISTIC_PREFLIGHT.json');assert not (RUN/'archive/unsupported_adaptive_pool/line_only_r0_seed1/TRACE.jsonl').read_text().strip()
    steps=0
    for seed in SEEDS:
      for family in (RUN,RUN/'dense'):
        for a in ARMS:
            starts=[]
            for repeat in REPEATS:
                folder=family/'models'/f'{a}_r{repeat}_seed{seed}';starts.append(read(folder/'START.json'));trace=[json.loads(x) for x in (folder/'TRACE.jsonl').read_text().splitlines()];assert len(trace)==2000 and all(x['step']==i+1 and all(np.isfinite(x[k]) for k in ('loss','gradient_norm','point_error')) for i,x in enumerate(trace));flags=starts[-1]['flags'];assert flags['enabled'] and not flags['warn_only'] and flags['cudnn_deterministic'];assert sha(folder/'final.pt')==read(folder/'COMPLETE.json')['checkpoint_sha256'];steps+=len(trace)
            assert starts[0]['process_id']!=starts[1]['process_id'];assert starts[0]['initial_sha256']==starts[1]['initial_sha256'];assert starts[0]['plan_sha256']==starts[1]['plan_sha256']
    bindings=read(RUN/'EVALUATION_BINDINGS.json')['files'];groups={};checked=0;manual=0
    for p,h in bindings.items():
        assert sha(ROOT/p)==h;p=Path(p);groups.setdefault((p.parent.parent.name,p.parent.name),[]).append(p)
    for (split,c),paths in groups.items():
        m,ts=EV.population(split,c);gt=np.array([t['points'] for t in ts]);valid=np.array([t['valid'] for t in ts]);det=np.array([r['detected'] for r in m['rows']]);diag=np.linalg.norm(np.array([r['original_hw'] for r in m['rows']]),axis=-1);_,O,_=EV.inputs(split,c,1);original=O.get('points')
        if original is None:original=EV.render_data(1,'check',c)[1]['points']
        if split=='real':
            for r,t in zip(m['rows'],ts):
                assert sha(ROOT/r['label'])==r['label_sha256'];ann=read(ROOT/r['label'])['objects'][0]['keypoint_annotations'][:8];assert [a.get('source')=='manual_click' and a.get('xy') is not None for a in ann]==t['valid'];xy=np.array([a['xy'] if a.get('xy') is not None else [0.,0.] for a in ann]);assert np.allclose(xy-np.array(r['origin']),t['points'],atol=1e-9,rtol=0);manual+=1
        for p in paths:
            z=np.load(ROOT/p);assert z['ids'].tolist()==[t['id'] for t in ts];assert np.array_equal(z['original_points'][det],original.numpy());assert np.isfinite(z['points'][det]).all();error=np.linalg.norm(z['points'].astype(float)-gt,axis=-1);r=s['results'][split+'/'+c][p.stem];frame=np.where(valid[det],error[det],0).sum(-1)/valid[det].sum(-1);assert abs(frame.mean()-r['frame_mean_px'])<1e-9;norm=np.ones(len(det));norm[det]=np.where(valid[det],np.minimum(error[det]/diag[det,None],1),0).sum(-1)/valid[det].sum(-1);assert abs(norm.mean()-r['full_normalized_mean'])<1e-9
            for typ in ('all','hidden','outside'):
                mask=valid&det[:,None]
                if typ!='all':mask &= np.array([t[typ] for t in ts])
                if mask.any():assert abs(error[mask].mean()-r['strata'][typ]['mean_px'])<1e-9
            checked+=1
    cf=read(RUN/'COUNTERFACTUAL.json');cf_files=read(RUN/'CF_INFERENCE_COMPLETE.json')['files'];cfc=0
    for p,h in cf_files.items():
        assert sha(ROOT/p)==h;p=Path(p);source=p.parent.name;stem=p.stem;seed=int(stem.rsplit('seed',1)[1]);pair=next((split,c) for split,c in PAIRS if stem==f'{split}_{c}_seed{seed}');m,ts=EV.population(*pair);ts=[t for t,r in zip(ts,m['rows']) if r['detected']];z=np.load(ROOT/p);assert z['ids'].tolist()==[t['id'] for t in ts];assert np.isfinite(z['shifted']).all();assert np.allclose(np.linalg.norm(z['normal'],axis=-1),1,atol=1e-6);sessions=[t['session'] for t in ts];assert all(sessions[i]!=sessions[j] for i,j in enumerate(z['donor']));valid=np.array([t['valid'] for t in ts]);r=cf['results'][source+'/'+pair[0]+'/'+pair[1]]
        for hi,shift in enumerate((4.,8.)):
            delta=z['shifted'][:,:,hi,1].astype(float)-z['shifted'][:,:,hi,0].astype(float);ratio=(delta*z['normal']).sum(-1)/(2*shift);row=next(x for x in r['response'] if x['seed']==seed and x['shift_px']==shift and x['stratum']=='all');assert abs(float(np.median(ratio[valid]))-row['median_signed_ratio'])<1e-6
        cfc+=1
    replay=[]
    for source in ('v33','deterministic'):
        for seed in SEEDS:
            m,O,S=EV.inputs('real','crop20',seed);O={k:v[:3].cuda() for k,v in O.items()};S={k:v[:3].cuda() for k,v in S.items()};model=load_probe(source,seed);z=np.load(RUN/'counterfactual'/source/f'real_crop20_seed{seed}.npz')
            with torch.no_grad():
                model.setting={};out=model(O,S,'line_only');features=model.last_token.clone();candidate=torch.tensor(z['candidate'][:3,0],device='cuda');model.setting=dict(mode='shift',candidate=candidate,pixels=4.);shift=model(O,S,'line_only');changed=model.changed_token;delta=changed-features
            assert torch.equal(changed[...,:42],features[...,:42]);assert torch.equal(changed[...,45:],features[...,45:]);rows=torch.arange(3,device='cuda');diag=(O['box'][:,2:]-O['box'][:,:2]).clamp_min(1).norm(dim=-1);assert torch.allclose(delta[rows,candidate,42],4/diag,atol=1e-7);assert torch.allclose(delta[rows,candidate,43:45],features[rows,candidate,40:42]*4/diag[:,None],atol=1e-7)
            other=torch.ones(3,48,dtype=torch.bool,device='cuda');other[rows,candidate]=False;assert torch.equal(changed[other],features[other]);diff=float(np.abs(out['points'].cpu().numpy()-z['baseline'][:3]).max());sd=float(np.abs(shift['points'][:,0].cpu().numpy()-z['shifted'][:3,0,0,1]).max());assert diff<.003 and sd<.003
            for mode in ('geometry_donor','appearance_donor'):
                model.setting=dict(mode=mode,donor=features.roll(1,0))
                with torch.no_grad():model(O,S,'line_only')
                changed=model.changed_token
                if mode=='geometry_donor':assert torch.equal(changed[...,:40],features[...,:40]) and torch.equal(changed[...,46:],features[...,46:])
                else:assert torch.equal(changed[...,32:],features[...,32:])
            model.setting={};path=V33.RUN/'models'/f'line_only_seed{seed}'/'final.pt' if source=='v33' else RUN/'dense/models'/f'line_only_r0_seed{seed}'/'final.pt';original=__import__('model33').Correspondence(seed).cuda().eval();original.load_state_dict(torch.load(path,weights_only=True,map_location='cpu')['state_dict'])
            with torch.no_grad():expected=original(O,S,'line_only')
            assert torch.equal(out['points'],expected['points']);replay.append(dict(source=source,seed=seed,baseline_max_px=diff,shift_max_px=sd,wrapper_exact=True))
    write(RUN/'AUDIT.json',dict(status='PASS',models=24,updates=steps,prediction_files=checked,counterfactual_files=cfc,manual_GT_rows_verified=manual,prior_bound_files=len(previous),all_inputs_unchanged=True,manual_GT_changed=False,GT_forward=False,isolated_geometry_and_appearance_verified=True,wrapper_replay=replay,initial_replication_result=read(RUN/'REPLICATION.json')['all_equal'],corrected_replication_result=read(RUN/'DENSE_REPLICATION.json')['all_equal']));print('AUDIT34 PASS',checked,cfc,flush=True)

if __name__=='__main__':main()
