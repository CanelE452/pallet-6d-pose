"""Independent paired-target replay and same-reference role evaluation."""
from context17 import *
from model17 import DetailEvaluator,revised_targets,geometry,EDGES,HEIGHT
from diagnose16 import confusion_stats

def independent(O,rep,gt,valid,masks,boundary,has,proxy,masked):
    q=geometry(O,rep,candidates(O,rep))['center'].cpu().numpy().astype(float);gt=gt.cpu().numpy().astype(float);valid=valid.cpu().numpy();masks=masks.cpu().numpy();bb=O['box'].cpu().numpy().astype(float);hw=O['image_hw'].cpu().numpy();boundary=boundary.cpu().numpy();has=has.cpu().numpy();proxy=proxy.cpu().numpy();masked=masked.cpu().numpy();rng=np.random.default_rng(1717);count=0
    for _ in range(1000):
        i=int(rng.integers(len(q)));p=int(rng.integers(8));c=int(rng.integers(8));l=int(rng.integers(3));j=int(rng.integers(7));x=q[i,p,c,l,j];uv=(x-bb[i,:2])/np.maximum(bb[i,2:]-bb[i,:2],1)*95
        if not valid[i].all() or not has[i] or not ((uv>=0)&(uv<=95)).all():a=b=-100
        else:
            ds=[]
            for u,v in EDGES:
                d=gt[i,v]-gt[i,u];t=np.clip((x-gt[i,u])@d/max(d@d,1e-8),0,1);ds.append(np.linalg.norm(x-gt[i,u]-t*d))
            e=int(np.argmin(ds));h,w=hw[i];xm,ym,xn,yn=masks[i];unsupported=not (0<=x[0]<w and 0<=x[1]<h) or (xm<=x[0]<xn and ym<=x[1]<yn)
            a=0 if ds[e]>2.5 or unsupported else (2 if e in HEIGHT else 1) if p in EDGES[e] else 3
            xx,yy=uv;u,v=int(xx),int(yy);u1,v1=min(u+1,95),min(v+1,95);dx,dy=xx-u,yy-v
            z=(1-dy)*((1-dx)*boundary[i,v,u]+dx*boundary[i,v,u1])+dy*((1-dx)*boundary[i,v1,u]+dx*boundary[i,v1,u1]);b=0 if a>0 and z>2.5 else a
        assert a==proxy[i,p,c,l,j] and b==masked[i,p,c,l,j],(a,b,proxy[i,p,c,l,j],masked[i,p,c,l,j]);count+=1
    return count

def main():
    setup();rows={};samples=0
    for seed in SEEDS:
        O,rep,gt,valid,ids,kind=training(seed);cache=torch.load(RUN/f'training_roles_seed{seed}.pt',weights_only=True)
        sup=[torch.load(RUN/'supervision/heldout'/f'{c}.pt',weights_only=True) for c in TRAIN_CONDITIONS];boundary=torch.cat([x['boundary'] for x in sup]).cuda();has=torch.cat([x['has_mask'] for x in sup]).cuda()
        proxy=cache['proxy'].cuda();masked=cache['masked'].cuda();masks=cache['mask'].cuda();samples+=independent(O,rep,gt,valid,masks,boundary,has,proxy,masked)
        for split in ('training','synth_val'):
            if split=='synth_val':
                m,O=load(split,'clean','cuda:0');rep=repcache(split,'clean',seed,device='cuda:0');t=[x for x,r in zip(targets(split,'clean'),m['rows']) if r['detected']];gt=torch.tensor([x['points'] for x in t],dtype=torch.float32,device='cuda:0');valid=torch.tensor([x['valid'] for x in t],device='cuda:0');masks=torch.full((len(t),4),-1.,device='cuda:0')
                sup=torch.load(RUN/'supervision/synth_val/clean.pt',weights_only=True);boundary=sup['boundary'].cuda();has=sup['has_mask'].cuda();pair=[revised_targets(subset(O,slice(i,i+32)),subset(rep,slice(i,i+32)),gt[i:i+32],valid[i:i+32],masks[i:i+32],boundary[i:i+32],has[i:i+32]) for i in range(0,len(gt),32)];proxy=torch.cat([x[0] for x in pair]);masked=torch.cat([x[1] for x in pair])
            for arm in ARMS:
                model=DetailEvaluator(seed).cuda().eval();model.load_state_dict(torch.load(RUN/'models'/f'{arm}_seed{seed}'/'final.pt',weights_only=True)['state_dict']);confs={k:np.zeros((4,4),np.int64) for k in ('proxy','mask')}
                with torch.inference_mode():
                    for first in range(0,len(gt),32):
                        sl=slice(first,first+32);out=model(subset(O,sl),subset(rep,sl),arm);pred=out['role_logits'].argmax(-1)
                        for k,labels in [('proxy',proxy),('mask',masked)]:
                            lab=labels[sl];ok=lab>=0;confs[k]+=torch.bincount(lab[ok]*4+pred[ok],minlength=16).reshape(4,4).cpu().numpy()
                rows[f'{split}/{arm}/seed{seed}']={k:confusion_stats(v) for k,v in confs.items()};del model
    write(RUN/'DIAGNOSIS.json',dict(status='PASS',independent_aux_samples=samples,role_metrics=rows,
        reference='All arms scored against BOTH common proxy and common visible-outline reference on same paired complete-corner G38 frames. Unknown/ROI-outside samples ignored. Dense correlated samples; not independent observations.',
        scope='Renderer mask gives outer object silhouette, not internal face creases. RGB detail test is a separate learned branch, not end-to-end YOLO/DHT feature retraining.'))
    print('DIAGNOSIS PASS',samples,flush=True)

if __name__=='__main__':main()
