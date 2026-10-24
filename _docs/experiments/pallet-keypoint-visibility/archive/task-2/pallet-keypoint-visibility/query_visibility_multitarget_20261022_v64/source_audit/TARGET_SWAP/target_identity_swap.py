#!/usr/bin/env python3
"""Public-only target-identity counterfactual. No training, no PnP.

Commands: prepare (CPU), sam (explicit GPU release required), evaluate (CPU).
All writes are confined to this standalone audit directory.
"""
from pathlib import Path
import argparse, csv, hashlib, importlib.util, json, sys, time
from collections import defaultdict
import cv2
import numpy as np
from PIL import Image

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
V63=ROOT.parent/'query_visibility_public_real_20261022_v63'
DATA=(V63/'ycb/data/train_real').resolve()
CACHE=ROOT/'source_cache/YCB_VALIDATION_WITH_SAM.npz'
PAIRS=np.array([[0,0],[1,0],[1,1],[2,0],[2,1],[2,2]],np.int8)
JOINT=np.array([[0,-1,-1],[1,2,-1],[3,4,5]],np.int8)
cv2.setNumThreads(1)

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,data): Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def readrows(path):
    with Path(path).open(newline='')as f:return list(csv.DictReader(f))
def loadnp(path,keys=None):
    with np.load(path,allow_pickle=False)as z:return {k:z[k]for k in(z.files if keys is None else keys)}
def check_source(path):
    path=Path(path).resolve();assert path.is_relative_to(DATA),path;return path
def mask(path):
    arr=cv2.imread(str(check_source(path)),cv2.IMREAD_UNCHANGED)
    assert arr is not None and arr.shape==(480,640)and set(np.unique(arr))<={0,255}
    return arr>0
def grid(q,side,hw=(480,640)):
    q=np.asarray(q,float);h,w=hw;axis=(np.arange(9)-4)*side/9;dx,dy=np.meshgrid(axis,axis)
    xy=q[:,None,None]+np.stack([dx,dy],-1)[None];pix=np.floor(xy+.5).astype(int)
    valid=(xy[...,0]>=0)&(xy[...,0]<w)&(xy[...,1]>=0)&(xy[...,1]<h)&(pix[...,0]>=0)&(pix[...,0]<w)&(pix[...,1]>=0)&(pix[...,1]<h)
    return xy,pix,valid
def sample(arr,pix):
    h,w=arr.shape;return arr[np.clip(pix[...,1],0,h-1),np.clip(pix[...,0],0,w-1)]
def boxgrid(xy,side,box):
    x0,y0,x1,y1=box;bw,bh=x1-x0,y1-y0
    return np.clip(np.stack([2*(xy[...,0]-x0)/bw-1,2*(xy[...,1]-y0)/bh-1,np.full(xy.shape[:-1],np.log2(side/bw)),np.full(xy.shape[:-1],np.log2(side/bh))],1),-4,4).astype(np.float32)
def counts(arr):return {str(v):int((arr==v).sum())for v in np.unique(arr)}

def prepare():
    assert not(OUT/'PREPARE_SEAL.json').exists()
    protocol=OUT/'PROTOCOL.json';assert protocol.exists()
    source_inputs=V63/'sam_source/INPUTS.jsonl';allinputs=[json.loads(s)for s in source_inputs.read_text().splitlines()]
    inputs={r['id']:r for r in allinputs if r['source']=='YCB'and r['split']=='validation'}
    a=loadnp(CACHE,['id','sample_id','native_corner','context','image_sha256','dense_valid','classes','joint'])
    rows=readrows(ROOT/'source_audit/YCB_VALIDATION_QUERY_EVIDENCE.csv')
    assert len(rows)==len(a['sample_id'])==3200
    assert np.array_equal(a['sample_id'].astype(str),[r['sample_id']for r in rows])
    instances=[json.loads(s)for s in(V63/'ycb/val_instances.jsonl').read_text().splitlines()]
    metadata={f"YCB:{r['scene_id']:06d}:{r['frame_id']:06d}:{r['instance_id']:06d}":r for r in instances}
    frameall=defaultdict(list)
    for r in instances:frameall[(r['scene_id'],r['frame_id'])].append(r)
    frames=defaultdict(list);groups=defaultdict(list)
    for i,ident in enumerate(a['id'].astype(str)):
        groups[ident].append(i)
    for ident in sorted(groups):
        r=metadata[ident];frames[(r['scene_id'],r['frame_id'])].append(ident)
    assert len(frames)==200 and all(len(v)==2 for v in frames.values())
    n=len(rows);context_other=np.empty_like(a['context']);box_other=np.empty((n,4,9,9),np.float32)
    other_ids=np.empty(n,dtype=a['id'].dtype);classes_other=np.empty_like(a['classes']);joint_other=np.empty_like(a['joint'])
    ekeys=['external_corroborated','external_ambiguous','strict_external_5x5','strict_external_corroborated_5x5','strict_external_ambiguous_5x5','visible_boundary_r1','full_boundary_r1','visible_overlaps_other_visible']
    evidence_other={k:np.zeros(n,bool)for k in ekeys};probeinputs=[]
    sourcehash={str(p):sha(p)for p in [CACHE,protocol,source_inputs,V63/'ycb/val_instances.jsonl',ROOT/'source_audit/YCB_VALIDATION_QUERY_EVIDENCE.csv']}
    for fi,(frame,targets)in enumerate(sorted(frames.items())):
        vm={r['instance_id']:mask(r['mask_vis'])for r in frameall[frame]};vs=sum(v.astype(np.uint16)for v in vm.values())
        for ident,other in [(targets[0],targets[1]),(targets[1],targets[0])]:
            ix=sorted(groups[ident],key=lambda i:int(a['native_corner'][i]));oi=groups[other]
            assert len(ix)==len(oi)==8 and np.array_equal(a['native_corner'][ix],np.arange(8))
            assert np.all(a['context'][oi]==a['context'][oi[0]])
            r=inputs[ident];ro=inputs[other];assert r['image_sha256']==ro['image_sha256']
            xy,pix,valid=grid(r['query'],r['side']);assert np.array_equal(valid,a['dense_valid'][ix])
            context_other[ix]=a['context'][oi[0]];box_other[ix]=boxgrid(xy,r['side'],ro['box']);other_ids[ix]=other
            meta=metadata[other];v=vm[meta['instance_id']];f=mask(meta['mask_full']);assert not(v&~f).any()
            other_v=vs-v.astype(np.uint16)>0;ext=f&~v
            v2=cv2.dilate(v.astype(np.uint8),np.ones((5,5),np.uint8))>0;f2=cv2.dilate(f.astype(np.uint8),np.ones((5,5),np.uint8))>0
            cc=np.stack([np.where(sample(vv,pix),0,np.where(sample(ff,pix),1,2))for vv,ff in [(v,f),(v2,f2)]],1).astype(np.int8)
            classes_other[ix]=cc;joint_other[ix]=JOINT[cc[:,0],cc[:,1]];assert(joint_other[ix][valid]>=0).all()
            ss=lambda m,k:cv2.boxFilter(m.astype(np.uint8),cv2.CV_32S,(k,k),normalize=False,borderType=cv2.BORDER_CONSTANT)
            area=ss(np.ones(v.shape,bool),5);nf,nv,no=ss(f,5),ss(v,5),ss(other_v,5);strict=(area==25)&(nf==25)&(nv==0)
            mix=lambda m:(ss(m,3)>0)&(ss(m,3)<ss(np.ones(m.shape,bool),3))
            maps={'external_corroborated':ext&other_v,'external_ambiguous':ext&~other_v,'strict_external_5x5':strict,'strict_external_corroborated_5x5':strict&(no==25),'strict_external_ambiguous_5x5':strict&(no==0),'visible_boundary_r1':mix(v),'full_boundary_r1':mix(f),'visible_overlaps_other_visible':v&other_v}
            for name,m in maps.items():evidence_other[name][ix]=sample(m,pix)[:,4,4]
            probeinputs.append(dict(recipient_id=ident,conditioning_id=other,image=r['image'],image_sha256=r['image_sha256'],raw_hw=r['raw_hw'],query=r['query'],side=r['side'],box=ro['box'],decoder_native_query=ro['query'],decoder_native_side=ro['side'],recipient_cache_rows=ix,decoder_native_cache_rows=sorted(oi,key=lambda i:int(a['native_corner'][i]))))
        if(fi+1)%100==0:print('PREPARE',fi+1,len(frames),flush=True)
    labels=dict(sample_id=a['sample_id'],conditioned_id_other=other_ids,classes_original=a['classes'],joint_original=a['joint'],classes_other=classes_other,joint_other=joint_other,dense_valid=a['dense_valid'])
    labels.update({f'other_{k}':v for k,v in evidence_other.items()})
    for key in ['strict_external_5x5','strict_external_corroborated_5x5','strict_external_ambiguous_5x5','visible_boundary_r1','full_boundary_r1']:
        labels['original_'+key]=np.array([bool(int(r[key]))for r in rows])
    labels['original_external_corroborated']=np.array([bool(int(r['exact_external_corroborated']))for r in rows]);labels['original_external_ambiguous']=np.array([bool(int(r['exact_external_ambiguous']))for r in rows])
    np.savez_compressed(OUT/'PAIR_LABELS.npz',**labels)
    np.savez_compressed(OUT/'PAIR_CONDITIONS.npz',sample_id=a['sample_id'],conditioned_id_other=other_ids,context_other=context_other,box_grid_other=box_other)
    with(OUT/'SAM_INPUTS.jsonl').open('w')as s:
        for r in probeinputs:s.write(json.dumps(r)+'\n')
    for p,h in sourcehash.items():assert sha(p)==h
    old=a['classes'][:,:,4,4];new=classes_other[:,:,4,4]
    summary=dict(protocol_sha256=sha(protocol),code_sha256=sha(__file__),queries=n,frames=len(frames),SAM_target_decodes=len(probeinputs),source_only=True,GPU_used=False,private_data_read=False,source_inputs=sourcehash,
                 original_classes_center={str(s):counts(old[:,s])for s in range(2)},other_classes_center={str(s):counts(new[:,s])for s in range(2)},
                 changed_center={str(s):dict(state=int((old[:,s]!=new[:,s]).sum()),visible=int(((old[:,s]==0)!=(new[:,s]==0)).sum()),full=int(((old[:,s]!=2)!=(new[:,s]!=2)).sum()))for s in range(2)},
                 other_evidence_counts={k:int(v.sum())for k,v in evidence_other.items()},outputs={p.name:sha(p)for p in [OUT/'PAIR_LABELS.npz',OUT/'PAIR_CONDITIONS.npz',OUT/'SAM_INPUTS.jsonl']})
    write(OUT/'PREPARE_SEAL.json',summary);print(json.dumps(summary,indent=2),flush=True)

def sam(args):
    assert args.gpu_released_by_parent, 'GPU execution requires explicit parent release'
    assert not(OUT/'SWAPPED_SAM.npz').exists()
    seal=json.loads((OUT/'PREPARE_SEAL.json').read_text());assert seal['code_sha256']==sha(__file__)
    path=OUT/'SAM_INPUTS.jsonl';assert sha(path)==seal['outputs'][path.name]
    rows=[json.loads(s)for s in path.read_text().splitlines()]
    byimage=defaultdict(list)
    for r in rows:byimage[r['image']].append(r)
    import torch
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False
    loader=V63/'sam/standalone_mobile_sam.py';spec=importlib.util.spec_from_file_location('targetswap_frozen_sam',loader);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    model,predictor=mod.load_model(args.device)
    a=loadnp(CACHE,['sam','sample_id']);swapped=np.empty_like(a['sam']);written=np.zeros(len(swapped),bool)
    replay_cells=replay_bad=0;start=time.monotonic();quality=[]
    with torch.inference_mode():
        for fi,(image,rr)in enumerate(byimage.items()):
            check_source(image);assert sha(image)==rr[0]['image_sha256']
            rgb=np.asarray(Image.open(image).convert('RGB'));assert list(rgb.shape[:2])==rr[0]['raw_hw'];predictor.set_image(rgb)
            for r in rr:
                alternatives,q,_=predictor.predict(box=np.asarray(r['box'],np.float32),multimask_output=True,return_logits=True)
                assert np.isfinite(q).all();choice=int(np.argmax(q));logit=alternatives[choice];near=cv2.dilate(logit,np.ones((5,5),np.uint8))
                def sample_two(query,side):
                    _,pix,valid=grid(query,side,r['raw_hw']);return np.stack([np.where(valid,np.clip(sample(m,pix)/8,-4,4),0).astype(np.float16)for m in[logit,near]],1)
                native=sample_two(r['decoder_native_query'],r['decoder_native_side']);expected=a['sam'][r['decoder_native_cache_rows']]
                replay_cells+=native.size;bad=int((native!=expected).sum());replay_bad+=bad;assert bad==0,('own-grid SAM replay mismatch',r['conditioning_id'],bad)
                ix=r['recipient_cache_rows'];assert not written[ix].any();swapped[ix]=sample_two(r['query'],r['side']);written[ix]=True
                quality.append(dict(id=r['conditioning_id'],chosen=choice,quality=q.astype(float).tolist()))
            if(fi+1)%25==0:print('SWAP_SAM',fi+1,len(byimage),'seconds',round(time.monotonic()-start,2),flush=True)
    assert written.all();assert(swapped[:,1]>=swapped[:,0]).all()
    np.savez_compressed(OUT/'SWAPPED_SAM.npz',sample_id=a['sample_id'],sam_other=swapped)
    report=dict(source_only=True,private_data_read=False,GT_masks_read=False,training=False,PnP=False,device=args.device,frames=len(byimage),target_decodes=len(rows),original_replay_values=replay_cells,original_replay_mismatches=replay_bad,seconds=time.monotonic()-start,prepare_seal_sha256=sha(OUT/'PREPARE_SEAL.json'),code_sha256=sha(__file__),loader_sha256=sha(loader),SAM_checkpoint_sha256=mod.CHECKPOINT_SHA,swapped_SAM_sha256=sha(OUT/'SWAPPED_SAM.npz'),quality=quality)
    write(OUT/'SAM_SEAL.json',report);print(json.dumps({k:v for k,v in report.items()if k!='quality'},indent=2),flush=True)

def marginals(p):
    sp=np.stack([np.stack([p[:,0],p[:,1]+p[:,2],p[:,3]+p[:,4]+p[:,5]],1),np.stack([p[:,0]+p[:,1]+p[:,3],p[:,2]+p[:,4],p[:,5]],1)],1)
    visible=sp[:,:,0];full=sp[:,:,:2].sum(2);hard=np.where(visible>=.5,0,np.where(full>=.5,1,2));return sp,visible,full,hard
def mean(v):return float(np.mean(v))if len(v)else None
def probe_metrics(p0,ps,old,new):
    sp0,v0,f0,h0=marginals(p0);sp1,v1,f1,h1=marginals(ps);u0=p0.max(1)<.5;u1=ps.max(1)<.5
    result={'all_pairs':len(old),'original_U_rate':mean(u0),'swapped_U_rate':mean(u1),'paired_confidence_coverage':mean(~u0&~u1),'scales':{}}
    for s,name in enumerate(['exact','radius2']):
        changed=old[:,s]!=new[:,s];ix=np.flatnonzero(changed)
        margin=(sp0[ix,s,old[ix,s]]-sp0[ix,s,new[ix,s]])-(sp1[ix,s,old[ix,s]]-sp1[ix,s,new[ix,s]])
        both=(h0[:,s]==old[:,s])&(h1[:,s]==new[:,s]);conf=(~u0&~u1)
        d={'class_changed_N':int(changed.sum()),'class_correct_direction_rate':mean(margin>0),'class_direction_tie_rate':mean(margin==0),'class_direction_mean_margin':mean(margin),'both_targets_correct_all':mean(both),'both_targets_correct_changed':mean(both[changed]),'both_targets_correct_changed_jointly_confident':mean(both[changed&conf]),'changed_jointly_confident_N':int((changed&conf).sum())}
        for task,oldbit,newbit,pa,pb in [('visible',old[:,s]==0,new[:,s]==0,v0[:,s],v1[:,s]),('full',old[:,s]!=2,new[:,s]!=2,f0[:,s],f1[:,s])]:
            cc=oldbit!=newbit;sign=newbit.astype(int)-oldbit.astype(int);delta=(pb-pa)*sign;bothbinary=((pa>=.5)==oldbit)&((pb>=.5)==newbit)
            d[task]={'changed_N':int(cc.sum()),'correct_direction_rate':mean(delta[cc]>0),'tie_rate':mean(delta[cc]==0),'mean_correct_direction_probability_change':mean(delta[cc]),'both_targets_binary_correct_changed':mean(bothbinary[cc])}
        result['scales'][name]=d
    return result

def evaluate(args):
    import torch
    torch.set_num_threads(1)
    checkpoint=Path(args.checkpoint).resolve();assert checkpoint.is_relative_to((ROOT/'runs').resolve())
    out=OUT/f'EVAL_{checkpoint.parent.name}';assert not out.exists();out.mkdir()
    a=loadnp(CACHE,['x','rgb_native','sam','context','box_grid','sample_id']);conditions=loadnp(OUT/'PAIR_CONDITIONS.npz');labels=loadnp(OUT/'PAIR_LABELS.npz');otherSAM=loadnp(OUT/'SWAPPED_SAM.npz')
    assert all(np.array_equal(a['sample_id'],v['sample_id'])for v in [conditions,labels,otherSAM])
    sys.path.insert(0,str(ROOT/'model'));from query_multitarget_head import load_model
    model,payload=load_model(checkpoint,'cpu');probs={};n=len(a['sample_id']);start=time.monotonic()
    arms=['ORIGINAL','CONTEXT_ONLY_SWAP','FULL_CONDITION_SWAP']
    with torch.inference_mode():
        for arm in arms:
            values=[]
            for lo in range(0,n,128):
                ix=slice(lo,min(n,lo+128));t=lambda arr:torch.as_tensor(arr[ix],dtype=torch.float32)
                logits=model(x=t(a['x']),rgb_native=t(a['rgb_native'])/255,sam=t(otherSAM['sam_other']if arm=='FULL_CONDITION_SWAP'else a['sam']),context=t(a['context']if arm=='ORIGINAL'else conditions['context_other']),box_grid=t(conditions['box_grid_other']if arm=='FULL_CONDITION_SWAP'else a['box_grid']))
                values.append(torch.softmax(logits.double(),1).numpy())
            probs[arm]=np.concatenate(values)
    old=labels['classes_original'][:,:,4,4];new=labels['classes_other'][:,:,4,4];result={}
    for arm in arms[1:]:result[arm]=probe_metrics(probs['ORIGINAL'],probs[arm],old,new)
    p0=probs['ORIGINAL'];ps=probs['FULL_CONDITION_SWAP'];sp0,v0,f0,h0=marginals(p0);sp1,v1,f1,h1=marginals(ps)
    slices={}
    for target,truth,pred,p in [('original',old,h0,p0),('other',new,h1,ps)]:
        for key in ['external_corroborated','external_ambiguous','strict_external_5x5','strict_external_corroborated_5x5','strict_external_ambiguous_5x5']:
            take=labels[target+'_'+key];conf=p.max(1)>=.5;confhard=PAIRS[p.argmax(1)]
            slices[target+'_'+key]={'N':int(take.sum()),'logical_exact_E_recall':mean((pred[:,0]==1)[take]),'logical_exact_state_accuracy':mean((pred[:,0]==truth[:,0])[take]),'U_rate':mean((~conf)[take]),'confident_jointMAP_exact_E_recall_allGT':mean(((confhard[:,0]==1)&conf)[take]),'confident_jointMAP_exact_E_precision':mean((truth[:,0]==1)[take&conf&(confhard[:,0]==1)]),'confident_jointMAP_exact_E_recall_decidedGT':mean((confhard[:,0]==1)[take&conf])}
    rawsam={}
    for scale,name in enumerate(['exact','radius2']):
        raw0=8*a['sam'][:,scale,4,4].astype(float);raw1=8*otherSAM['sam_other'][:,scale,4,4].astype(float);pp0=1/(1+np.exp(-raw0));pp1=1/(1+np.exp(-raw1));b0=old[:,scale]==0;b1=new[:,scale]==0;cc=b0!=b1;delta=(pp1-pp0)*(b1.astype(int)-b0.astype(int))
        rawsam[name]={'changed_N':int(cc.sum()),'correct_direction_rate':mean(delta[cc]>0),'tie_rate':mean(delta[cc]==0),'both_targets_correct_changed':mean((((raw0>0)==b0)&((raw1>0)==b1))[cc]),'both_targets_correct_all':mean(((raw0>0)==b0)&((raw1>0)==b1))}
    np.savez_compressed(out/'OUTPUTS.npz',sample_id=a['sample_id'],**{arm:probs[arm]for arm in arms})
    report=dict(protocol_sha256=sha(OUT/'PROTOCOL.json'),checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),source_cache_sha256=sha(CACHE),SAM_seal_sha256=sha(OUT/'SAM_SEAL.json'),code_sha256=sha(__file__),source_only=True,private_data_read=False,training=False,GPU_used=False,thresholds_fitted=False,context_only_is_inconsistent_conditioning_probe=True,metrics=result,external_evidence_slices=slices,rawSAM_visible_only=rawsam,seconds=time.monotonic()-start)
    write(out/'REPORT.json',report)
    records=[]
    for i,sid in enumerate(a['sample_id']):
        r={'sample_id':str(sid),'other_id':str(conditions['conditioned_id_other'][i]),'GT_original_exact':int(old[i,0]),'GT_other_exact':int(new[i,0]),'GT_original_radius2':int(old[i,1]),'GT_other_radius2':int(new[i,1])}
        for arm,p in probs.items():
            sp,v,f,h=marginals(p[i:i+1]);r.update({arm+'_pV0':float(v[0,0]),arm+'_pV2':float(v[0,1]),arm+'_pFull0':float(f[0,0]),arm+'_pFull2':float(f[0,1]),arm+'_state0':int(h[0,0]),arm+'_state2':int(h[0,1]),arm+'_U':int(p[i].max()<.5)})
        records.append(r)
    with(out/'QUERIES.csv').open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare','sam','evaluate']);ap.add_argument('--device',default='cuda:0');ap.add_argument('--gpu-released-by-parent',action='store_true');ap.add_argument('--checkpoint');args=ap.parse_args()
    if args.command=='prepare':prepare()
    elif args.command=='sam':sam(args)
    else:assert args.checkpoint;evaluate(args)
