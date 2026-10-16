"""Controlled C1/C4 output-layer trial: indexed, point-selected, joint-selected.

All arms use identical per-object OKS/visibility/RLE reduction. This is a
one-assigned-anchor output-head experiment, not a patch to stock batch-global
RLE reduction or a whole-network training claim. Upstream files stay untouched.
"""
from pathlib import Path
import argparse,hashlib,json,sys,time,os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/research/pallet_dimension_symmetry_v1'))
import core as old
OUT=ROOT/'data/pallet/results/pallet_c4_joint_loss_v1'
DS=ROOT/'challenge/yolo_pose_one_model/datasets/live_gt_v6_clean'
PERMS=torch.tensor([[0,1,2,3,4,5,6,7,8],[1,5,6,2,0,4,7,3,8],
                    [5,4,7,6,1,0,3,2,8],[4,0,3,7,5,1,2,6,8]])
ARMS=['indexed','point_select','joint_select']
sha=old.sha;read=old.read;write=old.write

def load(split,device='cpu'):
    return {k:v.to(device) for k,v in torch.load(OUT/f'{split}.pt',weights_only=True).items()}

def objective(pred,z,flow,arm):
    totals=[];parts=[]
    for p in PERMS.to(pred.device):
        total,terms=old.loss_terms(pred,z['target'][:,p],z['area'],flow)
        totals.append(total);parts.append(terms)
    total=torch.stack(totals,1);parts=torch.stack(parts,1)
    choice=(parts[:,:,0] if arm=='point_select' else total).argmin(1)
    if arm=='indexed':choice=choice*0
    choice=torch.where(z['order']==4,choice,torch.zeros_like(choice))
    loss=total.gather(1,choice[:,None]).squeeze(1)
    return loss,choice,total,parts

def freeze():
    OUT.mkdir(exist_ok=True)
    if (OUT/'PROTOCOL.json').exists():return
    files=[old.CKPT,ROOT/'pallet_yolo_loss/c4.py',ROOT/'pallet_yolo_loss/symmetry.py',
        ROOT/'scripts/research/pallet_dimension_symmetry_v1/core.py',
        ROOT/'challenge/config/SQUARE_PALLET_SYMMETRY_CONTRACT.json']
    files += [old.OUT/n for n in ['train.pt','val.pt','train_records.json','val_records.json','COMPLETION.json']]
    manifests={}
    for split in ['train','val']:
        rows=[]
        for ip in sorted((DS/'images'/split).glob('*.png')):
            lp=DS/'labels'/split/(ip.stem+'.txt');assert lp.exists()
            rows.append(dict(id=ip.stem,image=str(ip.relative_to(ROOT)),label=str(lp.relative_to(ROOT)),
                image_sha256=sha(ip),label_sha256=sha(lp)))
        manifests[split]=rows;write(OUT/f'real_{split}_manifest.json',rows)
    assert len(manifests['train'])==696 and len(manifests['val'])==155
    assert not {r['image_sha256'] for r in manifests['train']}&{r['image_sha256'] for r in manifests['val']}
    write(OUT/'PROTOCOL.json',dict(locked=True,arms=ARMS,seeds=[1,2,3],steps=1000,batch=32,lr=.0001,weight_decay=.0001,
        model='Same pretrained one2one keypoint/sigma outputs + zero-output FiLM in ALL arms; correct dimensions supplied to all. Backbone, detector, spatial features, BN and flow frozen.',
        primary='Real C4 all-frame normalized valid8-corner error. Same physical C4 scoring for every arm.',
        protected_population='Synthetic C1 subset of prior 2048/512 caches; C1 uses original per-object loss and identity scoring.',
        reduction='One TAL anchor per training object; identical per-object OKS+visibility+RLE clamped per object, gains12/1/1. Not stock multi-object batch-global RLE clamp.',
        selection='Only C4 switches. indexed=identity; point_select=min OKS; joint_select=min full12*OKS+visibility+RLE. One choice shared by all9 points/visibility/RLE.',
        sources={str(p.relative_to(ROOT)):sha(p) for p in files},
        input='Known dimensions + registered C1/C4 policy. No C2 objects used; no shape inference, no line features, no new backbone.',
        gate=dict(each_seed_C4_primary_not_worse_than_indexed_and_point=True,
            each_seed_C1_primary_not_worse_than_indexed_and_point=True,
            no_new_le10_to_gt20_against_both=True,
            C4_p90_nonworse_against_both=True,
            improvement_claim_requires_C4_paired_CI_upper_below_zero=True),
        evaluation_policy='No evaluation until all9 final checkpoints exist; no checkpoint/seed/threshold selection; no extra sweep.',
        limits=['Reused real interleave DEV155; NOT session-independent real generalization.',
                'Synthetic C1 protection screen, not real directional-pallet validation.',
                'Read-only upstream model and labels; raw coordinates subtract existing100px border.',
                'C1 and C4 differ in domain: within-C1 before/after is interpretable; between-type difference is not symmetry-only causality.']))

def cache():
    import cv2
    from ultralytics.data.augment import LetterBox
    from ultralytics.utils.loss import PoseLoss26
    freeze();assert not (OUT/'CACHE_COMPLETE.json').exists()
    torch.set_num_threads(3);cv2.setNumThreads(1)
    m=old.base_model().cuda();head=m.model[-1];criterion=PoseLoss26(m,tal_topk=7,tal_topk2=1)
    captured={};hooks=[mod.register_forward_hook(lambda module,args,out,i=i:captured.__setitem__(i,out.detach()))
                         for i,mod in enumerate(head.one2one_cv4)]
    audit={}
    for split in ['train','val']:
        original=torch.load(old.OUT/f'{split}.pt',weights_only=True)
        oldrows=read(old.OUT/f'{split}_records.json');ix=torch.where(original['order']==1)[0]
        chunks=[{k:v[ix] for k,v in original.items()}]
        records=[dict(oldrows[i],population='C1',session='C1:'+oldrows[i]['group']) for i in ix.tolist()]
        new=[];skips=[];parity=[]
        for n,row in enumerate(read(OUT/f'real_{split}_manifest.json')):
            ip=ROOT/row['image'];lp=ROOT/row['label'];assert sha(ip)==row['image_sha256'] and sha(lp)==row['label_sha256']
            im=cv2.imread(str(ip));ih,iw=im.shape[:2];assert (ih,iw)==(680,840)
            gain=min(640/ih,640/iw);offset=np.array([round((640-round(iw*gain))/2-.1),round((640-round(ih*gain))/2-.1)])
            image=LetterBox((640,640),auto=False)(image=im)
            x=torch.from_numpy(image[:,:,::-1].transpose(2,0,1).copy()).float().cuda()[None]/255
            with torch.no_grad():out=m(x);raw=out[1]['one2one'];decoded=head._inference(raw)
            conf,j=raw['scores'][0,0].sigmoid().max(0);j=int(j);selected=j
            label=np.loadtxt(lp,ndmin=2);assert label.shape==(1,32)
            gt=label[0,5:].reshape(9,3).copy();gt_raw=gt[:,:2]*[iw,ih]-100
            gt[:,:2]=(gt[:,:2]*[iw,ih]*gain+offset)/640
            box=label[0,1:5].copy();box[:2]=(box[:2]*[iw,ih]*gain+offset)/640;box[2:]*=np.array([iw,ih])*gain/640
            if split=='train':
                batch=dict(img=x,batch_idx=torch.zeros(1,device='cuda'),cls=torch.zeros(1,1,device='cuda'),
                    bboxes=torch.tensor(box[None],dtype=torch.float32,device='cuda'),keypoints=torch.tensor(gt[None],dtype=torch.float32,device='cuda'))
                with torch.no_grad():assignment,_,_=criterion.get_assigned_targets_and_loss(raw,batch)
                fg=assignment[0][0].nonzero().flatten()
                if not len(fg):skips.append(row['id']);continue
                assert len(fg)==1;selected=int(fg[0])
            sizes=[v.shape[-2]*v.shape[-1] for v in captured.values()]
            level=int(np.searchsorted(np.cumsum(sizes),selected,side='right'));local=selected-sum(sizes[:level]);fw=captured[level].shape[-1]
            feature=captured[level][0].reshape(45,-1)[:,local].cpu().numpy().copy()
            stride=float(head.stride[level]);anchor=np.array([local%fw+.5,local//fw+.5],dtype=np.float32)
            with torch.no_grad():
                f=torch.from_numpy(feature)[None,:,None,None].cuda()
                recovered=head.one2one_cv4_kpts[level](f).reshape(9,3).cpu().numpy()
                sigma=head.one2one_cv4_sigma[level](f).reshape(9,2).cpu().numpy()
            actual=raw['kpts'][0,:,selected].reshape(9,3).cpu().numpy()
            parity.append(float(abs(actual-recovered).max()))
            target=gt.copy();target[:,:2]*=640/stride
            xywh=box*640;targetbox=np.r_[xywh[:2]-xywh[2:]/2,xywh[:2]+xywh[2:]/2]
            observed=decoded[0,:4,j].cpu().numpy()
            new.append(dict(features=feature,level=np.array(level,dtype=np.int64),anchor=anchor,stride=np.array(stride,dtype=np.float32),
                dims_whd=np.array([1.1,1.1,.15],dtype=np.float32),target=target.astype('float32'),
                area=np.array(xywh[2]*xywh[3]/stride**2,dtype=np.float32),order=np.array(4,dtype=np.int64),perm=PERMS[2].numpy(),
                letterbox_offset=offset.astype('float32'),letterbox_gain=np.array(gain,dtype=np.float32),pad=np.array(100,dtype=np.float32),
                stock_grid=np.c_[actual[:,:2]+anchor,actual[:,2],sigma].astype('float32')))
            # Verify exact inverse of preprocessing against immutable label coordinates.
            assert np.max(np.abs((target[:,:2]*stride-offset)/gain-100-gt_raw))<1e-8
            records.append(dict(id='real:'+row['id'],group='real:'+row['id'],session=row['id'].split('__')[0],
                population='C4',asset='plastic_standard_110x110x15',order=4,image=row['image'],label=row['label'],raw_hw=[480,640],
                confidence=float(conf),box_raw=((observed-np.tile(offset,2))/gain-100).tolist(),
                gt_box_raw=((targetbox-np.tile(offset,2))/gain-100).tolist(),gt_raw=gt_raw.tolist(),valid=(gt[:,2]>0).tolist(),
                observation_anchor=j,training_anchor=selected if split=='train' else None))
            if (n+1)%100==0:print('CACHE',split,n+1,flush=True)
        assert max(parity)<.002
        chunks.append({k:torch.from_numpy(np.stack([r[k] for r in new])) for k in new[0]})
        z={k:torch.cat([c[k] for c in chunks]) for k in chunks[0]}
        torch.save(z,OUT/f'{split}.pt');write(OUT/f'{split}_records.json',records)
        audit[split]=dict(C1=len(ix),C4=len(new),skipped_C4=skips,public_layer_parity_max=float(max(parity)))
    for h in hooks:h.remove()
    write(OUT/'CACHE_COMPLETE.json',dict(PASS=True,splits=audit,cache_sha256={s:sha(OUT/f'{s}.pt') for s in ['train','val']}))
    print('CACHE COMPLETE',audit,flush=True)

def verify():
    torch.set_num_threads(2);m=old.base_model().cuda();flow=m.model[-1].flow_model.eval();z=load('train','cuda')
    ids=torch.cat([torch.where(z['order']==1)[0][:16],torch.where(z['order']==4)[0][:16]])
    zz=old.subset(z,ids);pred=old.OutputAdapter(m.model[-1],1).cuda()(zz,True).detach().requires_grad_()
    ops={a:objective(pred,zz,flow,a) for a in ARMS}
    mask=zz['order']==1;grads=[]
    for a in ARMS:grads.append(torch.autograd.grad(ops[a][0][mask].sum(),pred,retain_graph=True)[0])
    assert torch.equal(ops['indexed'][0][mask],ops['joint_select'][0][mask])
    assert max(float(abs(g-grads[0]).max()) for g in grads)==0
    assert bool((ops['joint_select'][0]<=ops['indexed'][0]+1e-6).all())
    assert bool((ops['joint_select'][0]<=ops['point_select'][0]+1e-6).all())
    # An independent candidate-by-candidate oracle checks the full weighted sum.
    brute=torch.stack([old.loss_terms(pred,zz['target'][:,p.to(pred.device)],zz['area'],flow)[0] for p in PERMS],1)
    expected=torch.where(mask,brute[:,0],brute.min(1).values)
    assert torch.equal(expected,ops['joint_select'][0])
    # Coordinate/visibility reordering is coupled; identity remains available.
    rotated=zz['target'].clone();rotated[~mask]=zz['target'][~mask][:,PERMS[1].cuda()]
    test=dict(zz,target=rotated)
    assert torch.allclose(objective(pred,test,flow,'joint_select')[0],ops['joint_select'][0],atol=1e-6)
    counts={};lossdrops=[];initial=old.OutputAdapter(m.model[-1],1).cuda().eval()
    with torch.no_grad():
        for start in range(0,len(z['order']),64):
            b=old.subset(z,slice(start,start+64));p=initial(b,True)
            a=objective(p,b,flow,'point_select');c=objective(p,b,flow,'joint_select');is4=b['order']==4
            counts['C4']=counts.get('C4',0)+int(is4.sum())
            counts['point_nonidentity']=counts.get('point_nonidentity',0)+int(((a[1]!=0)&is4).sum())
            counts['joint_nonidentity']=counts.get('joint_nonidentity',0)+int(((c[1]!=0)&is4).sum())
            counts['disagreement']=counts.get('disagreement',0)+int(((a[1]!=c[1])&is4).sum())
            assert bool((c[0]<=a[0]+1e-6).all());lossdrops.extend((a[0]-c[0])[is4].cpu().tolist())
    write(OUT/'PREFLIGHT.json',dict(PASS=True,C1_loss_and_gradient_exact=True,full_loss_nonincrease=True,
        brute_four_candidate_replay=True,C4_rotation_invariance=True,initial=counts,mean_full_loss_drop_C4=float(np.mean(lossdrops))))
    print('PREFLIGHT',read(OUT/'PREFLIGHT.json'),flush=True)

def train():
    assert read(OUT/'PREFLIGHT.json')['PASS'];cfg=read(OUT/'PROTOCOL.json')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    m=old.base_model().cuda();flow=m.model[-1].flow_model.eval();z=load('train','cuda');N=len(z['order'])
    lock=dict(script_sha256=sha(Path(__file__)),protocol_sha256=sha(OUT/'PROTOCOL.json'),cache=read(OUT/'CACHE_COMPLETE.json')['cache_sha256'])
    if (OUT/'TRAIN_LOCK.json').exists():assert read(OUT/'TRAIN_LOCK.json')==lock
    else:write(OUT/'TRAIN_LOCK.json',lock)
    cells={};start=time.time()
    for seed in cfg['seeds']:
        rng=np.random.default_rng(seed);ii=[]
        while len(ii)<cfg['steps']*cfg['batch']:ii.extend(rng.permutation(N).tolist())
        plan=np.array(ii[:cfg['steps']*cfg['batch']]).reshape(cfg['steps'],cfg['batch'])
        for arm in ARMS:
            folder=OUT/'models'/f'{arm}_s{seed}';done=folder/'COMPLETE.json'
            if done.exists():cells[folder.name]=read(done);continue
            assert not folder.exists();folder.mkdir(parents=True)
            model=old.OutputAdapter(m.model[-1],seed).cuda();init=old.state_sha(model.state_dict())
            opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
            history=[];switched=0;disagreed=0;count4=0
            for step,idx in enumerate(plan):
                b=old.subset(z,idx);opt.zero_grad(set_to_none=True);p=model(b,True)
                loss,choice,total,parts=objective(p,b,flow,arm);value=loss.mean();assert torch.isfinite(value)
                value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10,error_if_nonfinite=True);opt.step()
                is4=b['order']==4;switched+=int(((choice!=0)&is4).sum());count4+=int(is4.sum())
                disagreed+=int(((parts[:,:,0].argmin(1)!=total.argmin(1))&is4).sum())
                if (step+1)%200==0:
                    h=dict(step=step+1,loss=float(value.detach()),C4_seen=count4,switched=switched,choice_disagreements=disagreed)
                    history.append(h);print('TRAIN',seed,arm,h,flush=True)
            path=folder/'final.pt';torch.save({k:v.cpu() for k,v in model.state_dict().items()},path)
            cell=dict(complete=True,seed=seed,arm=arm,steps=cfg['steps'],initial_sha256=init,
                plan_sha256=hashlib.sha256(plan.tobytes()).hexdigest(),model_sha256=sha(path),history=history)
            write(done,cell);cells[folder.name]=cell
    for seed in cfg['seeds']:
        cc=[cells[f'{a}_s{seed}'] for a in ARMS];assert len({c['initial_sha256'] for c in cc})==len({c['plan_sha256'] for c in cc})==1
    write(OUT/'TRAIN_COMPLETE.json',dict(PASS=True,cells=cells,optimizer_steps=9000,elapsed_s=time.time()-start))

def evaluate():
    assert read(OUT/'TRAIN_COMPLETE.json')['PASS'];assert sha(Path(__file__))==read(OUT/'TRAIN_LOCK.json')['script_sha256']
    torch.set_num_threads(2);m=old.base_model();z=load('val');rows=read(OUT/'val_records.json');P=PERMS.numpy()
    from evaluate import iou
    matched=np.array([r['confidence']>=.25 and iou(r['box_raw'],r['gt_box_raw'])>=.5 for r in rows])
    population=np.array([r['population'] for r in rows]);diag=np.array([np.linalg.norm(r['raw_hw']) for r in rows])
    predictions={'stock':old.prepared_points(z['stock_grid'],z).numpy()}
    for seed in [1,2,3]:
        for arm in ARMS:
            model=old.OutputAdapter(m.model[-1],seed);model.load_state_dict(torch.load(OUT/'models'/f'{arm}_s{seed}'/'final.pt',weights_only=True));model.eval()
            with torch.no_grad():predictions[f'{arm}_s{seed}']=old.prepared_points(model(z,True),z).numpy()
    frame={};results={}
    for name,pred in predictions.items():
        scored=[]
        for i,r in enumerate(rows):
            gt=np.array(r['gt_raw']);valid=np.array(r['valid']);options=P if r['order']==4 else P[:1];err=[]
            for perm in options:
                v=valid[perm][:8];distance=np.linalg.norm(pred[i,:8]-gt[perm][:8],axis=-1)
                err.append(float(distance[v].mean()) if v.any() else float(diag[i]))
            e=min(err) if matched[i] else float(diag[i])
            scored.append(dict(id=r['id'],session=r['session'],population=r['population'],matched=bool(matched[i]),
                px=e,normalized=min(e/diag[i],1),indexed_px=err[0],chosen=int(np.argmin(err))))
        frame[name]=scored;results[name]={}
        for pop in ['C1','C4']:
            rr=[r for r in scored if r['population']==pop];mm=[r['px'] for r in rr if r['matched']]
            results[name][pop]=dict(n=len(rr),matched=len(mm),primary=float(np.mean([r['normalized'] for r in rr])),
                matched_mean_px=float(np.mean(mm)),matched_median_px=float(np.median(mm)),matched_p90_px=float(np.quantile(mm,.9)),
                bad_gt20_including_misses=sum(r['px']>20 for r in rr))
    def ci(delta,mask):
        sessions=np.array([r['session'] for r in rows]);labels=sorted(set(sessions[mask]));sums=np.array([delta[mask&(sessions==g)].sum() for g in labels]);counts=np.array([(mask&(sessions==g)).sum() for g in labels]);draw=np.random.default_rng(778).integers(0,len(labels),(10000,len(labels)));values=sums[draw].sum(1)/counts[draw].sum(1)
        return dict(mean=float(delta[mask].mean()),low=float(np.quantile(values,.025)),high=float(np.quantile(values,.975)),clusters=len(labels),scope='paired session/group bootstrap of reused DEV; not independent-session generalization')
    comparison={}
    checks=[]
    for control in ['indexed','point_select','stock']:
        item={}
        for pop in ['C1','C4']:
            mask=population==pop;deltas=[];harm=[];p90=[];darray=[]
            for seed in [1,2,3]:
                c=control if control=='stock' else f'{control}_s{seed}';j=f'joint_select_s{seed}'
                a=np.array([r['normalized'] for r in frame[j]]);b=np.array([r['normalized'] for r in frame[c]])
                px=np.array([r['px'] for r in frame[j]]);base=np.array([r['px'] for r in frame[c]])
                deltas.append(float((a-b)[mask].mean()));darray.append(a-b);harm.append(int(((base<=10)&(px>20)&mask).sum()))
                p90.append(results[j][pop]['matched_p90_px']<=results[c][pop]['matched_p90_px'])
            item[pop]=dict(primary_delta_per_seed=deltas,new_good_to_bad_per_seed=harm,p90_nonworse=p90,CI=ci(np.mean(darray,0),mask))
            if control!='stock':
                checks += [all(d<=0 for d in deltas),not any(harm)]
                if pop=='C4':checks += [all(p90)]
        comparison['joint_minus_'+control]=item
    write(OUT/'PER_FRAME.json',frame);np.savez_compressed(OUT/'predictions.npz',**predictions)
    write(OUT/'SUMMARY.json',dict(complete=True,results=results,comparisons=comparison,nonharm_gate_pass=all(checks),
        decision='NONHARM_SCREEN_PASSED' if all(checks) else 'NONHARM_NOT_ESTABLISHED',full_network_training=False,
        real_C4_evaluation=True,independent_real_session=False,training_updates=9000))
    print('EVALUATION',json.dumps(read(OUT/'SUMMARY.json'),ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['cache','verify','train','evaluate']);args=p.parse_args()
    globals()[args.phase]()
