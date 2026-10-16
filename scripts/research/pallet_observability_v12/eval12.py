"""Frozen paired evaluation and post-inference error stratification."""
import argparse
from context12 import *
from router12 import Router
E=import_file('v12_score_compat',KIT.parent/'pallet_optional_lines_v11'/'evaluate.py')

def save(split,condition,name,m,O,points,extra=None):
    directory=RUN/'predictions'/split/condition;directory.mkdir(parents=True,exist_ok=True)
    det=np.array([r['detected'] for r in m['rows']]);data=dict(ids=np.array([r['id'] for r in m['rows']]),detected=det)
    for key,value in dict(points=points,original_points=O['points'],**(extra or {})).items():
        v=value.detach().cpu().numpy();a=np.full((len(det),)+v.shape[1:],np.nan,dtype=np.float32);a[det]=v;data[key]=a
    np.savez_compressed(directory/f'{name}.npz',**data)

def inference():
    setup();assert read(RUN/'TRAINING_COMPLETE.json')['complete']
    lock=RUN/'EVAL_PROTOCOL.json'
    if lock.exists():raise SystemExit('Refusing to overwrite evaluation')
    write(lock,dict(source_sha256=sha(Path(__file__)),training_protocol_sha256=sha(RUN/'TRAIN_PROTOCOL.json'),
        policy='All six fixed V11 inputs, same detections. No GT targets, masks, condition identifiers or label flags supplied to model. Targets opened only in scoring phase.',
        diagnostics='Post-inference oracle routing uses true intervention exposure for diagnosis only; never a deployable model or primary result.'))
    audit=[]
    for split in ('synth_val','real'):
        for condition in CONDITIONS:
            m,O=C.load_observations(split,condition,'cuda:0');save(split,condition,'stock',m,O,O['points'])
            for seed in SEEDS:
                old=np.load(V11/'predictions'/split/condition/f'v10_local_seed{seed}.npz')
                assert old['ids'].tolist()==[r['id'] for r in m['rows']]
                bp=torch.tensor(old['points'][old['detected']],device='cuda:0');save(split,condition,f'local_seed{seed}',m,O,bp)
                for arm in ARMS:
                    ck=RUN/'models'/f'{arm}_seed{seed}'/'final.pt';model=Router(seed).cuda().eval()
                    model.load_state_dict(torch.load(ck,map_location='cpu',weights_only=False)['state_dict']);parts=[]
                    with torch.inference_mode():
                        for first in range(0,len(bp),32):
                            sl=slice(first,first+32);p=model(subset(O,sl),bp[sl],arm);parts.append({k:v.cpu() for k,v in p.items()})
                    out={k:torch.cat([p[k] for p in parts]) for k in parts[0]};points=out.pop('points');assert torch.isfinite(points).all()
                    save(split,condition,f'{arm}_seed{seed}',m,O,points,out)
                    audit.append(dict(split=split,condition=condition,arm=arm,seed=seed,frames=len(bp),checkpoint_sha256=sha(ck)));del model
            print('INFERRED',split,condition,len(O['points']),flush=True)
    write(RUN/'INFERENCE_COMPLETE.json',dict(complete=True,rows=audit,targets_used=False))

def compare(arrays,rows,a,b,sessions,split,metric='normalized',mask=None):
    an=[f'{a}_seed{s}' for s in SEEDS] if a!='stock' else ['stock']*3
    bn=[f'{b}_seed{s}' for s in SEEDS] if b!='stock' else ['stock']*3
    det=arrays['stock']['detected']
    if mask is None:
        result=E.bootstrap_difference(np.stack([arrays[n]['normalized'] for n in an]),np.stack([arrays[n]['normalized'] for n in bn]),sessions,split)
        result['paired_px']=E.bootstrap_difference(np.stack([arrays[n]['frame'][det] for n in an]),np.stack([arrays[n]['frame'][det] for n in bn]),np.array(sessions)[det].tolist(),split)
        result['relative_gain_percent']=[100*(1-rows[x]['frame_mean_px']/rows[y]['frame_mean_px']) for x,y in zip(an,bn)]
        return result
    eligible=mask&det[:,None];diff=np.stack([arrays[x]['error']-arrays[y]['error'] for x,y in zip(an,bn)])
    pooled=np.mean(diff,axis=0);sums=np.where(eligible,pooled,0.).sum(-1);counts=eligible.sum(-1)
    if split=='real':
        groups=np.unique(sessions);session=np.array(sessions)
        sums=np.array([sums[session==g].sum() for g in groups]);counts=np.array([counts[session==g].sum() for g in groups])
    good=counts>0;sums=sums[good];counts=counts[good]
    if not len(counts):return None
    rng=np.random.default_rng(121212);ix=rng.integers(0,len(counts),(5000,len(counts)))
    boots=sums[ix].sum(-1)/counts[ix].sum(-1)
    per=[float(d[eligible].mean()) for d in diff]
    ref=[float(arrays[n]['error'][eligible].mean()) for n in bn]
    return dict(mean_px=float(sums.sum()/counts.sum()),ci95=np.quantile(boots,[.025,.975]).tolist(),per_seed_difference_px=per,
        per_seed_gain_percent=[-100*d/r for d,r in zip(per,ref)],total_points=int(mask.sum()),detected_points=int(eligible.sum()),clusters=len(counts))

def scoring():
    assert read(RUN/'INFERENCE_COMPLETE.json')['complete'];results={};comparisons={};routing={}
    for split in ('synth_val','real'):
        for condition in CONDITIONS:
            m=read(V11/'cache'/split/condition/'manifest.json');t=read(V11/'targets'/f'{split}_{condition}.json')['rows']
            rows={};arrays={};qstats={};kind=target_type(t).numpy();valid=np.array([r['valid'] for r in t]);key=f'{split}/{condition}'
            for path in (RUN/'predictions'/split/condition).glob('*.npz'):
                rows[path.stem],arrays[path.stem]=E.score_file(path,m,t)
                p=np.load(path)
                if 'probabilities' in p:
                    det=p['detected'];qstats[path.stem]={}
                    for k,typ in enumerate(('intact','hidden','outside')):
                        mask=(kind==k)&valid&det[:,None]
                        qstats[path.stem][typ]=dict(points=int(mask.sum()),mean_keep=float(p['keep'][mask].mean()) if mask.any() else None,
                            routing_correct=float((p['probabilities'][mask].argmax(-1)==k).mean()) if mask.any() else None)
            comp={};sessions=[r['session'] for r in t]
            for a,b in [('learned_gate','local'),('learned_gate','scalar_gate'),('gated_amodal','learned_gate'),('gated_amodal','always_amodal'),('gated_amodal','stock'),('local','stock')]:
                r=compare(arrays,rows,a,b,sessions,split)
                r['subgroups']={typ:compare(arrays,rows,a,b,sessions,split,mask=valid&(kind==k)) for k,typ in enumerate(('intact','hidden','outside'))}
                comp[f'{a}_vs_{b}']=r
            results[key]=rows;comparisons[key]=comp;routing[key]=qstats
            print('SCORED',key,{a:[round(rows[f'{a}_seed{s}']['frame_mean_px'],3) for s in SEEDS] for a in ('local',)+ARMS},flush=True)
    def clean_limit(arm):
        r=results['real/clean'];return all(r[f'{arm}_seed{s}']['frame_mean_px']<=1.01*r[f'local_seed{s}']['frame_mean_px'] for s in SEEDS)
    h=comparisons['real/corner20'];h1=h['learned_gate_vs_local']['subgroups']['hidden'];h2=h['learned_gate_vs_scalar_gate']['subgroups']['hidden']
    gates=dict(suppression=dict(all_seeds_hidden_better_local=all(x<0 for x in h1['per_seed_difference_px']),
        all_seeds_hidden_better_scalar=all(x<0 for x in h2['per_seed_difference_px']),local_ci_below0=h1['ci95'][1]<0,scalar_ci_below0=h2['ci95'][1]<0,clean_limit=clean_limit('learned_gate')))
    for c in ('crop20','crop40'):
        r=comparisons[f'real/{c}']['gated_amodal_vs_learned_gate']['subgroups']['outside']
        gates[c]=dict(all_seeds_outside_gain3=all(x>=3 for x in r['per_seed_gain_percent']),ci_below0=r['ci95'][1]<0,clean_limit=clean_limit('gated_amodal'))
    for g in gates.values():g['pass']=all(g.values())
    write(RUN/'SUMMARY.json',dict(complete=True,results=results,comparisons=comparisons,routing=routing,gates=gates,
        scope='Reused synthetic505 and real DEV297/9sessions, artificial stress. Manual click GT only on real. No real tuning, no natural visibility or full6D claim.'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['inference','score'],default='inference');a=p.parse_args()
    inference() if a.phase=='inference' else scoring()
