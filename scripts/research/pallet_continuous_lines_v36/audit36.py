from context36 import *
from model36 import Frozen,Model
from train36 import load_model
from data32 import population
from eval32 import SPLITS
from data32 import conditions
from cli_train import state_sha
import adapt36

def main():
    setup36();assert read(RUN/'ADAPT_INFERENCE_COMPLETE.json')['complete'];steps=0
    for marker in ('PRIOR_BINDINGS.json','CACHE_COMPLETE.json','INFERENCE_COMPLETE.json','PROBES_COMPLETE.json','ADAPT_INFERENCE_COMPLETE.json'):
        for p,h in read(RUN/marker)['files'].items():assert sha(ROOT/p)==h,p
    for marker,key in [('CACHE_PROTOCOL.json','files'),('TRAIN_PROTOCOL.json','source_hashes')]:
        for p,h in read(RUN/marker)[key].items():assert sha(ROOT/p)==h,p
    assert read(RUN/'ADAPT_PROTOCOL.json')['source_sha256']==sha(KIT/'adapt36.py');assert read(RUN/'PROBE_PROTOCOL.json')['source_sha256']==sha(KIT/'probes36.py')
    for seed in SEEDS:
        expected=Frozen(seed).state_dict();pointinitial=[];fullinitial=[]
        for family,arms in [('',ARMS),('adapt',adapt36.ADAPT)]:
            for arm in arms:
                folder=RUN/family/'models'/f'{arm}_seed{seed}';start=read(folder/'START.json');complete=read(folder/'COMPLETE.json');trace=[json.loads(s) for s in (folder/'TRACE.jsonl').read_text().splitlines()];assert len(trace)==2000 and all(r['step']==i+1 and np.isfinite(r['loss']) and np.isfinite(r['gradient_norm']) for i,r in enumerate(trace));assert complete['checkpoint_sha256']==sha(folder/'final.pt');assert complete['trace_sha256']==sha(folder/'TRACE.jsonl');steps+=len(trace);state=torch.load(folder/'final.pt',weights_only=True,map_location='cpu')['state_dict']
                for k,v in expected.items():
                    if family and k.startswith(('line_token.','line_relation.')):continue
                    assert torch.equal(state['core.'+k],v),(family,arm,seed,k)
                if family:
                    pointinitial.append(start['point_path_initial_sha256']);rr={k.removeprefix('refiner.'):v for k,v in state.items() if k.startswith('refiner.')};assert state_sha(rr)==start['fixed_refiner_sha256']
                else:fullinitial.append(start['initial_sha256'])
        assert len(set(fullinitial))==1 and len(set(pointinitial))==1
    checked=0;manual=0
    for family,summary in [('',read(RUN/'POINT_SUMMARY.json')),('adapt',read(RUN/'ADAPT_SUMMARY.json'))]:
      for split in SPLITS:
        for c in conditions(split):
            m,ts=population(split,c);gt=np.array([t['points'] for t in ts]);valid=np.array([t['valid'] for t in ts]);det=np.array([r['detected'] for r in m['rows']]);diag=np.linalg.norm(np.array([r['original_hw'] for r in m['rows']]),axis=-1)
            if split=='real' and family=='':
                for r,t in zip(m['rows'],ts):
                    assert sha(ROOT/r['label'])==r['label_sha256'];a=read(ROOT/r['label'])['objects'][0]['keypoint_annotations'][:8];assert [p.get('source')=='manual_click' and p.get('xy') is not None for p in a]==t['valid'];xy=np.array([p['xy'] if p.get('xy') is not None else [0.,0.] for p in a]);assert np.allclose(xy-r['origin'],t['points'],atol=1e-9);manual+=1
            for p in (RUN/family/'predictions'/split/c).glob('*.npz'):
                z=np.load(p);assert z['ids'].tolist()==[t['id'] for t in ts];assert np.isfinite(z['points'][det]).all();old=np.load(V34.RUN/'predictions'/split/c/f'line_only_r0_seed{p.stem[-1]}.npz');assert np.array_equal(z['original_points'],old['original_points'],equal_nan=True);error=np.linalg.norm(z['points'].astype(float)-gt,axis=-1);frame=np.where(valid[det],error[det],0).sum(-1)/valid[det].sum(-1);ref=summary['results'][split+'/'+c][p.stem];assert abs(frame.mean()-ref['frame_mean_px'])<1e-9;norm=np.ones(len(det));norm[det]=np.where(valid[det],np.minimum(error[det]/diag[det,None],1),0).sum(-1)/valid[det].sum(-1);assert abs(norm.mean()-ref['full_normalized_mean'])<1e-9;checked+=1
    # Recompute line distance independently from output line and physical endpoints.
    meta=read(V35.RUN/'POPULATION.json')['rows'];geom=np.load(V35.RUN/'geometry.npz');lookup={r['id']:i for i,r in enumerate(meta)};line=read(RUN/'LINE_SUMMARY.json');errors=np.load(RUN/'LINE_ERRORS.npz');pointroles=errors['roles'];manual_line=0
    for seed in SEEDS:
      for arm in ARMS:
        for p in (RUN/'predictions/render_check').glob(f'*/{arm}_seed{seed}.npz'):
            z=np.load(p)
            for ri in np.where(z['detected'])[0]:
                i=lookup[z['ids'][ri]];bb=np.array(meta[i]['box']);diag=np.linalg.norm(np.maximum(bb[2:]-bb[:2],1));center=(bb[:2]+bb[2:])/2;ns=z['normal'][ri].reshape(12,8,2).astype(float);rs=z['offset'][ri].reshape(12,8).astype(float)*diag+(ns*center).sum(-1);scores=z['score'][ri].reshape(12,8)
                for j in np.where(errors['supported'][i])[0]:
                    role=pointroles[i,j];k=scores[role].argmax();end=geom['support'][i,role];d=np.abs(end@ns[role,k]-rs[role,k]).mean();assert abs(d-errors[f'top_{arm}_{seed}'][i,j])<1e-8;manual_line+=1
    replay=[]
    for family,arms,loader in [('',ARMS,load_model),('adapt',adapt36.ADAPT,adapt36.load)]:
      for seed in SEEDS:
        data=torch.load(RUN/'cache/real/crop20'/f'seed{seed}.pt',weights_only=True,map_location='cpu');x={k:v[:3].cuda() for k,v in data['x'].items()}
        for arm in arms:
            model=loader(arm,seed)
            with torch.no_grad():out=model(x,arm)
            z=np.load(RUN/family/'predictions/real/crop20'/f'{arm}_seed{seed}.npz');diff=float(np.abs(out['points'].cpu().numpy()-z['points'][z['detected']][:3]).max());assert diff<.003
            if family:
                original=load_model(arm,seed)
                with torch.no_grad():baseout=original(x,arm)
                assert torch.equal(out['normal'],baseout['normal']) and torch.equal(out['score'],baseout['score'])
            replay.append(dict(family=family or 'primary',arm=arm,seed=seed,max_px=diff))
    write(RUN/'AUDIT.json',dict(status='PASS',models=21,updates=steps,prediction_files=checked,manual_GT_condition_rows=manual,physical_line_point_seed_checks=manual_line,prior_files_unchanged=True,GT_training_leak=False,frozen_core_verified=True,adaptation_fixed_refiner_verified=True,matched_initialization_verified=True,checkpoint_replay=replay,limits='GT source/valid/crop/hash consistency checked; no new hand-click accuracy audit. Primary comparison and post-primary adaptation are separate.'))
    print('AUDIT36 PASS',steps,checked,flush=True)
if __name__=='__main__':main()
