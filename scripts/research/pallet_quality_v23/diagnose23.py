"""Post-training diagnosis; real GT is read only after training and inference finish."""
from context23 import *
from model20 import CauseGate,oracle_alpha
import train23
from eval23 import inputs,population,prediction_path,ARMS
from train_mass23 import mass_weights
from scipy.optimize import minimize_scalar

def analyze(base,delta,gt,valid,kind,ids,outputs,w=None):
    base=base.astype(float);delta=delta.astype(float);gt=gt.astype(float);before=np.linalg.norm(base-gt,axis=-1);full=np.linalg.norm(base+delta-gt,axis=-1);target=np.clip(((gt-base)*delta).sum(-1)/np.maximum((delta*delta).sum(-1),1e-12),0,1);bins=np.digitize(before,[5,15],right=True);result={}
    for c,typ in [('crop20',2),('crop40',2),('corner20',1)]:
        m=valid&(kind==typ)&np.array([x.endswith('::'+c) for x in ids])[:,None];rows=[]
        for i in (None,0,1,2):
            mask=m if i is None else m&(bins==i);n=int(mask.sum())
            if not n:continue
            d=dict(bin='all' if i is None else i,points=n,base_px=float(before[mask].mean()),full_px=float(full[mask].mean()),stop_fraction=float((target[mask]<=.05).mean()),oracle_px=float(np.linalg.norm(base+target[...,None]*delta-gt,axis=-1)[mask].mean()),models={})
            for arm,out in outputs.items():
                error=np.linalg.norm(out['points'].astype(float)-gt,axis=-1);bad=mask&(full>before+.1);good=mask&(full<before-.1)
                d['models'][arm]=dict(after_px=float(error[mask].mean()),mean_alpha=float(out['alpha'][mask].mean()),alpha_bad=float(out['alpha'][bad].mean()) if bad.any() else None,alpha_good=float(out['alpha'][good].mean()) if good.any() else None,remaining_harmful=int((mask&(error>before+.1)).sum()))
            if i is None:
                for label,weights0 in [('unweighted',np.ones_like(before)),('quality',w['quality'] if w is not None else np.ones_like(before)),('mass',w['mass'] if w is not None else np.ones_like(before))]:
                    def f(a):return float(np.average(np.linalg.norm(base[mask]+a*delta[mask]-gt[mask],axis=-1),weights=weights0[mask]))
                    x=minimize_scalar(f,bounds=(0,1),method='bounded');best=min((0.,1.,float(x.x)),key=f);d[label+'_constant_optimum']=dict(alpha=best,error_px=f(best),base_px=f(0),full_px=f(1))
            rows.append(d)
        result[c]=rows
    return result

def main():
    setup();assert read(RUN/'TRAINING_COMPLETE.json')['complete'];assert read(RUN/'INFERENCE_COMPLETE.json')['complete'];rows=[]
    train23.NEW_IDS={r['id'] for r in read(RUN/'POPULATIONS.json')['new_train']}
    for seed in SEEDS:
        O,r,gt,valid,ids,kind=train23.training(seed);w,probe=train23.weights(r,gt,valid,ids,kind);wm,mp=mass_weights(r,gt,valid,ids,kind);outs={}
        for arm in ARMS:
            model=CauseGate(seed).cuda().eval();model.load_state_dict(torch.load(RUN/'models'/f'{arm}_seed{seed}'/'final.pt',weights_only=True)['state_dict']);parts=[]
            with torch.inference_mode():
                for first in range(0,len(ids),32):
                    sl=slice(first,first+32);o=model(subset(O,sl),subset(r,sl),arm);parts.append({k:o[k].cpu().numpy() for k in ('points','alpha')})
            outs[arm]={k:np.concatenate([p[k] for p in parts]) for k in parts[0]};del model
        rows.append(dict(split='training',seed=seed,conditions=analyze(r['base'].cpu().numpy(),r['proposal_delta'].cpu().numpy(),gt.cpu().numpy(),valid.cpu().numpy(),kind.cpu().numpy(),ids,outs,dict(quality=w.cpu().numpy(),mass=wm.cpu().numpy()))))
        # Independently recompute the fixed bin weights in numpy.
        b=np.linalg.norm(r['base'].cpu().numpy()-gt.cpu().numpy(),axis=-1);mask=valid.cpu().numpy()&(kind.cpu().numpy()==2)&np.array([x.endswith('::crop20') for x in ids])[:,None];bins=np.digitize(b,[5,15],right=True);counts=np.bincount(bins[mask],minlength=3);f=np.clip(mask.sum()/(3*counts),.125,8);expected=f[bins[mask]];expected/=expected.mean();assert np.allclose(expected,w.cpu().numpy()[mask],atol=1e-6)
        move=r['proposal_delta'].norm(dim=-1).cpu().numpy();mass=np.array([move[mask&(bins==i)].sum() for i in range(3)]);f=np.clip(mass.sum()/(3*mass),.125,8);expected=f[bins[mask]];expected/=(expected*move[mask]).sum()/move[mask].sum();assert np.allclose(expected,wm.cpu().numpy()[mask],atol=1e-6)
        del O,r,gt,valid,kind,outs,w,wm
        for split in ('real','prospective'):
            pieces=[];outs={a:dict(points=[],alpha=[]) for a in ARMS};ids=[];g=[];v=[];ks=[];bs=[];ds=[]
            for c in ('crop20','crop40','corner20'):
                m,t=population(split,c);det=np.array([x['detected'] for x in m['rows']]);base=np.load(prediction_path(split,c,f'unit_expand_seed{seed}'));bs.append(base['base'][det]);ds.append(base['proposal_delta'][det]);g.extend(np.array([x['points'] for x in t])[det]);v.extend(np.array([x['valid'] for x in t])[det]);ks.extend(C.P.target_type(t).numpy()[det]);ids.extend(x['id']+'::'+c for x in m['rows'] if x['detected'])
                for a in ARMS:
                    p=np.load(prediction_path(split,c,f'{a}_seed{seed}'))
                    for k in outs[a]:outs[a][k].append(p[k][det])
            rows.append(dict(split=split,seed=seed,conditions=analyze(np.concatenate(bs),np.concatenate(ds),np.array(g),np.array(v),np.array(ks),ids,{a:{k:np.concatenate(z) for k,z in out.items()} for a,out in outs.items()})))
        print('DIAGNOSED',seed,flush=True)
    write(RUN/'DIAGNOSIS.json',dict(rows=rows,weight_independent_replay='PASS',scope='Posthoc privileged error bins and constant-alpha optima diagnose distribution; not fit parameters, inference routing, or proof of a sole cause. No matching of pose, texture or scale.'))
if __name__=='__main__':main()
