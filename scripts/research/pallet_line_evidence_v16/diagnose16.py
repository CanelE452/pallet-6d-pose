"""Posthoc role supervision audit and selection diagnostics, never model tuning."""
from context16 import *
from model16 import EvidenceEvaluator,role_targets,geometry,EDGES,HEIGHT

def confusion_stats(conf):
    p=np.diag(conf)/np.maximum(conf.sum(0),1);r=np.diag(conf)/np.maximum(conf.sum(1),1)
    return dict(confusion=conf.tolist(),precision=p.tolist(),recall=r.tolist(),f1=(2*p*r/np.maximum(p+r,1e-9)).tolist(),accuracy=float(np.trace(conf)/max(conf.sum(),1)))

def independent_labels(O,rep,gt,valid,masks,labels):
    # Independent float64 segment distance, with fixed random sample selection.
    x=geometry(O,rep,candidates(O,rep))['center'].cpu().numpy().astype(float);gt=gt.cpu().numpy().astype(float);valid=valid.cpu().numpy();masks=masks.cpu().numpy();labels=labels.cpu().numpy()
    rng=np.random.default_rng(16);checked=0
    for _ in range(800):
        b=int(rng.integers(len(gt)));p=int(rng.integers(8));k=int(rng.integers(8));l=int(rng.integers(3));j=int(rng.integers(7));q=x[b,p,k,l,j]
        if not valid[b].all():want=-100
        else:
            ds=[]
            for a,z in EDGES:
                d=gt[b,z]-gt[b,a];u=np.clip(np.dot(q-gt[b,a],d)/max(d@d,1e-8),0,1);ds.append(np.linalg.norm(q-gt[b,a]-u*d))
            idx=int(np.argmin(ds));a,z=EDGES[idx];h,w=O['image_hw'][b].cpu().numpy();xm,ym,xn,yn=masks[b]
            unsupported=not(0<=q[0]<w and 0<=q[1]<h) or (xm<=q[0]<xn and ym<=q[1]<yn)
            want=0 if ds[idx]>2.5 or unsupported else (2 if idx in HEIGHT else 1) if p in (a,z) else 3
        assert want==labels[b,p,k,l,j],(want,labels[b,p,k,l,j]);checked+=1
    return checked

def main():
    setup();rows={};checks=0
    for seed in SEEDS:
        O,rep,gt,valid,ids,kind=training(seed);cache=torch.load(RUN/f'training_roles_seed{seed}.pt',weights_only=True);labels=cache['labels'].cuda();masks=cache['mask'].cuda()
        assert ids==cache['ids'];checks+=independent_labels(O,rep,gt,valid,masks,labels)
        for split in ('training','synth_val'):
            if split=='synth_val':
                _,O=load(split,'clean','cuda:0');rep=repcache(split,'clean',seed,device='cuda:0');m=manifest(split,'clean');t=[x for x,r in zip(targets(split,'clean'),m['rows']) if r['detected']]
                gt=torch.tensor([x['points'] for x in t],dtype=torch.float32,device='cuda:0');valid=torch.tensor([x['valid'] for x in t],device='cuda:0');masks=torch.full((len(t),4),-1.,device='cuda:0')
                labels=torch.cat([role_targets(subset(O,slice(i,i+32)),subset(rep,slice(i,i+32)),gt[i:i+32],valid[i:i+32],masks[i:i+32]) for i in range(0,len(gt),32)])
            for arm in ('strip','role'):
                model=EvidenceEvaluator(seed).cuda().eval();model.load_state_dict(torch.load(RUN/'models'/f'{arm}_seed{seed}'/'final.pt',weights_only=True)['state_dict']);conf=np.zeros((4,4),np.int64)
                with torch.inference_mode():
                    for first in range(0,len(gt),32):
                        sl=slice(first,first+32);out=model(subset(O,sl),subset(rep,sl),arm);pred=out['role_logits'].argmax(-1);lab=labels[sl];ok=lab>=0
                        conf+=torch.bincount(lab[ok]*4+pred[ok],minlength=16).reshape(4,4).cpu().numpy()
                rows[f'{split}/{arm}/seed{seed}']=confusion_stats(conf);del model
    s=read(RUN/'SUMMARY.json');selection={}
    for key,models in s['diagnostics'].items():
        selection[key]={name:{typ:{k:d[k] for k in ('choice_counts','harmful_applied_points','applied_points','oracle_px','regret_px','catastrophes_base5_to20')} for typ,d in strata.items()} for name,strata in models.items()}
    sessions={}
    from eval16 import prediction_path
    for condition,typ in [('clean',None),('corner20','hidden'),('crop20','outside'),('crop40','outside')]:
        t=targets('real',condition);gt=np.array([x['points'] for x in t]);valid=np.array([x['valid'] for x in t]);groups=np.array([x['session'] for x in t]);session_rows={}
        for arm in ('control','role','balanced_role','median_role'):
            diffs=[];det=None
            for seed in SEEDS:
                p=np.load(prediction_path('real',condition,f'{arm}_seed{seed}'));base=np.load(prediction_path('real',condition,f'base_seed{seed}'));det=p['detected']
                diffs.append(np.linalg.norm(p['points']-gt,axis=-1)-np.linalg.norm(base['points']-gt,axis=-1))
            diff=np.mean(diffs,axis=0);mask=valid&det[:,None]
            if typ:mask&=np.array([x[typ] for x in t])
            session_rows[arm]={g:dict(points=int((mask&(groups==g)[:,None]).sum()),difference_px=float(diff[mask&(groups==g)[:,None]].mean())) for g in np.unique(groups) if (mask&(groups==g)[:,None]).any()}
        sessions[condition]=session_rows
    write(RUN/'DIAGNOSIS.json',dict(role_metrics=rows,independent_label_samples=checks,selection=selection,real_session_vs_base=sessions,
        limitations='Finite cuboid-edge proxy, not actual visible mesh silhouette. Auxiliary metrics only complete synthetic annotation frames. Correlated dense samples, no independent-sample CI claimed. Unsupervised strip role readout is a mechanism control, not a calibrated semantic head.',
        causal_scope='role vs strip isolates auxiliary loss; strip vs control isolates evidence branch. balanced/median change objective too. Wrong-image/off interventions are OOD and do not alone establish useful causal improvement.'))
    print('DIAGNOSIS COMPLETE, label replay',checks,flush=True)

if __name__=='__main__':main()
