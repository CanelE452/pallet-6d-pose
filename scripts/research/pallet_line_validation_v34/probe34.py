"""Frozen-token interventions: vary geometry separately from sampled appearance."""
from context34 import *
from model33 import Correspondence
from data32 import inputs,population

class Probe(Correspondence):
    def __init__(self,seed):
        super().__init__(seed);self.setting={};self.last_token=None

    def evidence(self,z,maps,bb,hw,blind=False):
        captured={}
        hook=self.line_token.register_forward_pre_hook(lambda module,args:captured.update(features=args[0]))
        try:e=super().evidence(z,maps,bb,hw,blind)
        finally:hook.remove()
        features=captured['features'];self.last_token=features.detach();mode=self.setting.get('mode')
        if mode is None:return e
        changed=features.clone()
        if mode=='shift':
            rows=torch.arange(len(bb),device=bb.device);k=self.setting['candidate'];normal=features[rows,k,40:42];diag=(bb[:,2:]-bb[:,:2]).clamp_min(1).norm(dim=-1);amount=self.setting['pixels']/diag
            changed[rows,k,42]+=amount;changed[rows,k,43:45]+=amount[:,None]*normal
        elif mode=='geometry_donor':changed[...,40:46]=self.setting['donor'][...,40:46]
        elif mode=='appearance_donor':changed[...,:32]=self.setting['donor'][...,:32]
        elif mode=='both_donor':
            changed[...,:32]=self.setting['donor'][...,:32];changed[...,40:46]=self.setting['donor'][...,40:46]
        else:raise ValueError(mode)
        e.update(line=self.line_token(changed),normal=changed[...,40:42],offset=changed[...,42],loc=changed[...,43:45],spread=changed[...,45]);self.changed_token=changed.detach();return e

def load_probe(source,seed):
    from sampler34 import install
    install();model=Probe(seed).cuda().eval();path=(V33.RUN/'models'/f'line_only_seed{seed}'/'final.pt') if source=='v33' else RUN/'dense/models'/f'line_only_r0_seed{seed}'/'final.pt';model.load_state_dict(torch.load(path,weights_only=True,map_location='cpu')['state_dict']);return model

PAIRS=(('real','clean'),('real','corner20'),('real','crop20'),('real','crop40'),('fresh_check','crop20'),('render_check','crop20'))

def infer():
    deterministic();records=[];write(RUN/'CF_PROTOCOL.json',dict(source_hashes={str(p.relative_to(ROOT)):sha(p) for p in (KIT/'probe34.py',KIT/'context34.py',KIT/'deterministic34.py',KIT/'sampler34.py',V33.KIT/'model33.py')},protocol_sha256=sha(RUN/'PROTOCOL.json'),GT_forward=False))
    for source in ('v33','deterministic'):
        for seed in SEEDS:
            model=load_probe(source,seed)
            for split,c in PAIRS:
                path=RUN/'counterfactual'/source/f'{split}_{c}_seed{seed}.npz';path.parent.mkdir(parents=True,exist_ok=True)
                if path.exists():continue
                m,O,S=inputs(split,c,seed);rows=[r for r in m['rows'] if r['detected']];n=len(rows);sessions=[r['session'] for r in rows];donor=[next((i+n//2+k)%n for k in range(n) if sessions[(i+n//2+k)%n]!=sessions[i]) for i in range(n)];O={k:v.cuda() for k,v in O.items()};S={k:v.cuda() for k,v in S.items()};baseline=[];tokens=[];weights=[]
                with torch.no_grad():
                    model.setting={}
                    for i in range(0,n,32):
                        out=model(subset(O,slice(i,i+32)),subset(S,slice(i,i+32)),'line_only');baseline.append(out['points']);tokens.append(model.last_token);weights.append(out['line_attention'][...,:-1].reshape(-1,8,2,48).mean(2))
                    baseline=torch.cat(baseline);tokens=torch.cat(tokens);weights=torch.cat(weights);top=weights.argmax(-1);mass=weights.gather(-1,top[...,None]).squeeze(-1);normal=tokens[torch.arange(n,device='cuda')[:,None],top,40:42];shifted=torch.empty(n,8,2,2,2,device='cuda');observed_variants={}
                    for mode in ('geometry_donor','appearance_donor','both_donor','line_off'):
                        parts=[]
                        for i in range(0,n,32):
                            sl=slice(i,i+32);model.setting={} if mode=='line_off' else dict(mode=mode,donor=tokens[donor[i:i+32]])
                            out=model(subset(O,sl),subset(S,sl),'line_only',intervention='line_off' if mode=='line_off' else None);parts.append(out['points'])
                        observed_variants[mode]=torch.cat(parts)
                    # Probe each point's highest-attention candidate independently.
                    for j in range(8):
                        for hi,h in enumerate((4.,8.)):
                            for si,sign in enumerate((-1.,1.)):
                                for i in range(0,n,32):
                                    sl=slice(i,i+32);model.setting=dict(mode='shift',candidate=top[sl,j],pixels=sign*h);out=model(subset(O,sl),subset(S,sl),'line_only');shifted[sl,j,hi,si]=out['points'][:,j]
                    model.setting={}
                selected=tokens[torch.arange(n,device='cuda')[:,None],top]
                values=dict(ids=np.array([r['id'] for r in rows]),baseline=baseline.cpu().numpy(),normal=normal.cpu().numpy(),line_offset=selected[...,42].cpu().numpy(),line_support=selected[...,43:45].cpu().numpy(),candidate=top.cpu().numpy(),top_attention=mass.cpu().numpy(),shifted=shifted.cpu().numpy(),donor=np.array(donor),**{k:v.cpu().numpy() for k,v in observed_variants.items()});np.savez_compressed(path,**values);records.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path)));print('CF34',source,seed,split,c,flush=True)
    files=list((RUN/'counterfactual').glob('*/*.npz'));write(RUN/'CF_INFERENCE_COMPLETE.json',dict(complete=True,GT_forward=False,files={str(p.relative_to(ROOT)):sha(p) for p in files}))

def cluster_ci(values,mask,sessions):
    # values seed x frame x point; average seeds first, then cluster sessions/groups.
    error=values.mean(0);groups=np.unique(sessions);sums=np.array([np.where(mask[np.array(sessions)==g],error[np.array(sessions)==g],0).sum() for g in groups]);counts=np.array([mask[np.array(sessions)==g].sum() for g in groups]);keep=counts>0;sums=sums[keep];counts=counts[keep]
    rng=np.random.default_rng(34001);ix=rng.integers(0,len(counts),(5000,len(counts)));boot=sums[ix].sum(-1)/counts[ix].sum(-1);return dict(mean=float(sums.sum()/counts.sum()),ci95=np.quantile(boot,[.025,.975]).tolist(),clusters=len(counts))

def score():
    results={}
    for source in ('v33','deterministic'):
        for split,c in PAIRS:
            m,ts=population(split,c);ts=[t for t,r in zip(ts,m['rows']) if r['detected']];gt=np.array([t['points'] for t in ts]);valid=np.array([t['valid'] for t in ts]);sessions=[t['session'] for t in ts];files=[np.load(RUN/'counterfactual'/source/f'{split}_{c}_seed{s}.npz') for s in SEEDS];rows=[]
            for seed,z in zip(SEEDS,files):
                assert z['ids'].tolist()==[t['id'] for t in ts]
                for hi,h in enumerate((4.,8.)):
                    plus=z['shifted'][:,:,hi,1];minus=z['shifted'][:,:,hi,0];change=plus-minus;ratio=np.sum(change*z['normal'],axis=-1)/(2*h);tangent=np.stack((-z['normal'][...,1],z['normal'][...,0]),-1);tratio=np.sum(change*tangent,axis=-1)/(2*h)
                    targetmask=valid&np.array([t['hidden' if c=='corner20' else 'outside'] for t in ts]) if c!='clean' else valid
                    for typ,mask in [('all',valid),('active',valid&(z['top_attention']>=.01)),('target_stratum',targetmask),('active_target',targetmask&(z['top_attention']>=.01))]:
                        if not mask.any():continue
                        rows.append(dict(seed=seed,shift_px=h,stratum=typ,points=int(mask.sum()),median_signed_ratio=float(np.median(ratio[mask])),mean_signed_ratio=float(ratio[mask].mean()),positive_fraction=float((ratio[mask]>1e-4).mean()),median_absolute_motion_px=float(np.median(np.linalg.norm(change,axis=-1)[mask]/2)),median_abs_tangent_ratio=float(np.median(np.abs(tratio[mask]))),mean_top_attention=float(z['top_attention'][mask].mean())))
            eb=np.stack([np.linalg.norm(z['baseline']-gt,axis=-1) for z in files]);variants={}
            for name in ('geometry_donor','appearance_donor','both_donor','line_off'):
                e=np.stack([np.linalg.norm(z[name]-gt,axis=-1) for z in files]);movement=np.stack([np.linalg.norm(z[name]-z['baseline'],axis=-1) for z in files]);v={}
                for typ,mask in [('all',valid),('target_stratum',valid&np.array([t['hidden' if c=='corner20' else 'outside'] for t in ts]) if c!='clean' else valid)]:
                    if not mask.any():continue
                    ci=cluster_ci(e-eb,mask,sessions);v[typ]=dict(base_px=float(eb[:,mask].mean()),changed_px=float(e[:,mask].mean()),movement_px=float(movement[:,mask].mean()),per_seed_difference_px=[float(x[mask].mean()) for x in e-eb],**ci)
                variants[name]=v
            results[source+'/'+split+'/'+c]=dict(response=rows,variants=variants)
    gates={}
    for source in ('v33','deterministic'):
        r=results[source+'/real/crop20'];active=[x for x in r['response'] if x['stratum']=='active'];direction=all(x['median_signed_ratio']>.02 and x['positive_fraction']>.6 for x in active) and len(active)==6;useful=r['variants']['geometry_donor']['target_stratum']['ci95'][0]>0;gates[source]=dict(directional_response=direction,geometry_donor_harm=useful,pass_gate=direction and useful)
    write(RUN/'COUNTERFACTUAL.json',dict(complete=True,results=results,gates=gates,limits='Synthetic token interventions keep observed appearance/points fixed, may be off-manifold. Direction gate active>=.01 is GT-free; GT used only scoring. Positive ratio threshold1e-4 ignores numerical jitter. Cluster CIs average seeds before resampling; repeated DEV, no independent generalization guarantee.'));print('COUNTERFACTUAL34',gates,flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['infer','score'],required=True);a=p.parse_args();dict(infer=infer,score=score)[a.phase]()
