"""Exact-source edge targets, fresh variant features, and three equal small fits."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
import torch

from . import common as C
from . import source_audit as S
from . import model as M

CACHE=C.SCRATCH/'learned_cache'
FITS=C.SCRATCH/'learned_fits'

def state_sha(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def gpu():
    s=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,temperature.gpu,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
    p=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits'],text=True).strip()
    processes=[]
    for line in p.splitlines():
        fields=line.split(',')
        if len(fields)>=3:processes.append(dict(pid=fields[0].strip(),process=Path(fields[1].strip()).name,memory_MiB=fields[2].strip()))
    return dict(gpu=s,processes=processes)

def freeze_protocol():
    torch.manual_seed(1);head=M.CorrespondenceHead();initial=state_sha(head.state_dict())
    order=np.random.default_rng(1).integers(0,768,size=(3000,16),dtype=np.int64)
    value=dict(schema='new_source_learning_protocol_v1',arms=list(M.ARMS),model=M.CONFIG,
        parameters=sum(p.numel() for p in head.parameters()),seed=1,updates=3000,batch=16,
        optimizer=dict(name='AdamW',lr=.001,weight_decay=.0001,warmup=100,decay='cosine to zero',gradient_clip_norm=10),
        normalization='fixed physical units: brightness[-1,1], Sobel/4, pixel offsets/32, xy/width-height, length/diagonal; frozen neck raw FP16 grouped means; no estimated normalization',
        checkpoint='last only',initial_state_sha256=initial,batch_order_sha256=hashlib.sha256(order.tobytes()).hexdigest(),
        supervised_fields_separate=True,source_test_scores_cannot_change_configuration=True,
        no_real_GT_read=True,no_model_reselection=True,shared_total_parameters=True,
        physical_distance_tolerance_relative_diagonal=1e-5,depth_visibility_tolerance_relative_diagonal=.001,
        external_mask_support_radius_px=1.5,variants=['existing_P0_original'],new_RGB_generation=0,
        four_controlled_variants='unrepresented: existing valid supervision takes precedence over new RGB creation',
        family_split=C.binding(C.DOC/'SOURCE_FAMILY_SPLIT.json'))
    path=C.DOC/'LEARNING_PROTOCOL.json'
    if path.exists():assert C.read(path)==value
    else:C.write(path,value)
    CACHE.mkdir(parents=True,exist_ok=True);FITS.mkdir(parents=True,exist_ok=True)
    if not (CACHE/'order.npy').exists():np.save(CACHE/'order.npy',order)
    return value

def targets(query,g,scene,visible):
    """Source truth only. Unsupported bounding edges receive IGNORE, never fake positive."""
    import open3d as o3d
    from .solver import project
    uv=project(g['X'],g['R'],g['t'],g['K']);lo=np.full(84,65,np.int16);hi=lo.copy();weight=np.zeros(84,np.float32);valid=np.zeros(84,bool);kind=np.full(84,'ignore',dtype='<U9')
    positions=np.zeros((84,2));expected_depth=np.zeros(84);physical=np.zeros(84,bool);radius=np.linalg.norm(g['dims'])
    # Solve each initial-normal search line against the known *physical* edge.
    source3d=[];sourceids=[]
    for i,(edge,a,b,u,length) in enumerate(query['identity']):
        if not query['valid'][i]:continue
        tangent=uv[b]-uv[a];matrix=np.column_stack([query['normal'][i],-tangent])
        if abs(np.linalg.det(matrix))<1e-8:continue
        offset,fraction=np.linalg.solve(matrix,uv[a]-query['center'][i]);valid[i]=True
        if not 0<=fraction<=1 or abs(offset)>32:kind[i]='no_match';continue
        za=(g['X'][a]@g['R'].T+g['t'])[2];zb=(g['X'][b]@g['R'].T+g['t'])[2]
        uu=(fraction/zb)/((1-fraction)/za+fraction/zb)
        X=(1-uu)*g['X'][a]+uu*g['X'][b];positions[i]=query['center'][i]+offset*query['normal'][i]
        expected_depth[i]=(X@g['R'].T+g['t'])[2];source3d.append(X);sourceids.append(i)
    if sourceids:
        xs=np.asarray(source3d,np.float32);closest=scene.compute_closest_points(o3d.core.Tensor(xs))['points'].numpy()
        actual=np.linalg.norm(closest-xs,axis=1)<=1e-5*radius
        for j,i in enumerate(sourceids):
            if not actual[j]:valid[i]=False;kind[i]='ignore'
            else:physical[i]=True
        K=g['K'];direction=np.column_stack([(positions[sourceids,0]-K[0,2])/K[0,0],(positions[sourceids,1]-K[1,2])/K[1,1],np.ones(len(sourceids))])@g['R'];origin=-g['R'].T@g['t']
        # A tiny inset along the segment avoids exact silhouette triangle-edge rounding.
        ray=np.column_stack([np.broadcast_to(origin,direction.shape),direction]).astype(np.float32)
        hit=scene.cast_rays(o3d.core.Tensor(ray))['t_hit'].numpy()
        for j,i in enumerate(sourceids):
            if not physical[i]:continue
            x,y=positions[i];h,w=visible.shape
            if not (0<=x<w and 0<=y<h):valid[i]=False;kind[i]='ignore';continue
            close=np.isfinite(hit[j]) and abs(float(hit[j])-expected_depth[i])<=.001*radius
            ix,iy=int(round(x-.5)),int(round(y-.5));l=max(0,ix-1);r=min(w,ix+2);t=max(0,iy-1);b=min(h,iy+2)
            observed=bool((visible[t:b,l:r]>127).any())
            if not close or not observed:kind[i]='no_match';continue
            offset=float((positions[i]-query['center'][i])@query['normal'][i])+32
            lower=int(np.clip(math.floor(offset),0,64));upper=min(64,lower+1)
            lo[i]=lower;hi[i]=upper;weight[i]=offset-lower;kind[i]='positive'
    # Nonphysical vertical bounding edges must be ignored even when no segment intersection exists.
    edge_physical=S.physical_samples(scene,g['X'])<1e-5*radius
    for e in range(12):
        if not edge_physical[e].any():valid[e*7:(e+1)*7]=False;kind[e*7:(e+1)*7]='ignore'
    return lo,hi,weight,valid,dict(Counter(kind.tolist()))

def prepare():
    protocol=freeze_protocol();assert C.read(C.DOC/'SOURCE_SUPERVISION_STATUS.json')['status']=='VALID_SOURCE_SUBSET'
    C.source_modules()
    from scripts.research.pallet_line_pose_v1.features import FrozenYoloFeatures
    start=time.monotonic();families=S.selected_families();v,tri,mesh=S.mesh_normalized();n=1024
    specs=dict(features=((n,84,28,65),np.float16),lo=((n,84),np.int16),hi=((n,84),np.int16),weight=((n,84),np.float32),valid=((n,84),bool))
    arrays={k:np.lib.format.open_memmap(CACHE/(k+'.npy'),mode='w+',dtype=d,shape=s) for k,(s,d) in specs.items()}
    detector=FrozenYoloFeatures(weights=C.ROOT/'challenge/yolo_pose_one_model/spatial_concat_scratch/runs/YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42/weights/best.pt')
    rows=[];counts=Counter();targetcounts=Counter();initialcalls=0
    for family_i,row in enumerate(families):
        paths=S.locate(row);ann=C.read(paths['label']);g=S.geometry(row,ann);original=cv2.imread(str(paths['rgb']));visible=cv2.imread(str(paths['visible']),0);amodal=cv2.imread(str(paths['amodal']),0)
        scene=S.ray_scene(v,tri,g['dims']);gt=np.array(ann['objects'][0]['projected_cuboid'])
        for j in range(1):
            index=family_i;image,vm,meta=original,visible,dict(kind='existing_P0_original',generated=False);captured=detector.predict(image)
            selected=captured['selected_index'];counts['detector_forward']+=1
            if selected is None:
                arrays['features'][index]=0;arrays['lo'][index]=65;arrays['hi'][index]=65;arrays['weight'][index]=0;arrays['valid'][index]=False;tc=dict(ignore=84);points=None
            else:
                points=captured['candidates'][selected]['keypoints_xy'];features,query=M.inputs(image,captured,points,g['K'],g['dims'])
                arrays['features'][index]=features.detach().cpu().numpy().astype(np.float16);lo,hi,weight,valid,tc=targets(query,g,scene,vm)
                for k,a in [('lo',lo),('hi',hi),('weight',weight),('valid',valid)]:arrays[k][index]=a
                initialcalls+=1
            targetcounts.update(tc);counts['generated_RGB_arrays']+=int(meta['generated'])
            rows.append(dict(index=index,id=row['id'],family=row['scenario_id'],
                partition=row['observation_partition'],variant=j,rgb_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                original_rgb=C.binding(paths['rgb']) if j==0 else None,composition=meta,target_counts=tc,
                selected_detector_candidate=selected,selected_points=None if points is None else points.tolist()))
        if (family_i+1)%32==0:
            print('FEATURE_CACHE',family_i+1,1024,round(time.monotonic()-start,1),flush=True)
            C.write(C.DOC/'LEARNING_PROGRESS.json',dict(stage='feature_cache',families=family_i+1,images=len(rows),seconds=time.monotonic()-start,gpu=gpu()))
        if time.monotonic()-start>1200:raise RuntimeError('feature preparation 20min budget reached; preserve partial files')
    for a in arrays.values():a.flush()
    detector.close();torch.cuda.empty_cache()
    C.write(CACHE/'CACHE_MANIFEST.json',dict(complete=True,records=rows,specs={k:dict(shape=s,dtype=str(np.dtype(d))) for k,(s,d) in specs.items()},protocol=C.binding(C.DOC/'LEARNING_PROTOCOL.json')))
    C.write(C.DOC/'SUPERVISION_PREPARATION.json',dict(complete=True,families=1024,RGB_inputs=1024,existing_original_RGB_reused=1024,
        generated_RGB_2D_arrays=0,rendered_RGB_3D=0,new_original_scene_RGB=0,
        raw_RGB_files_written=0,cache_bytes=sum((CACHE/(k+'.npy')).stat().st_size for k in specs),
        depth_ID_auxiliary_producer='Open3D raycasting of actual composed scene.usd triangles',
        counts=dict(counts),initial_standard_pose_calls=initialcalls,target_counts=dict(targetcounts),
        cache_manifest=C.binding(CACHE/'CACHE_MANIFEST.json'),wall_seconds=time.monotonic()-start,gpu=gpu(),
        true_ignore_supported=True,vertical_bbox_not_supervised=True,
        four_controlled_variants='not represented; no new RGB generated because existing supervision is valid'))

def probe(head,arm,arrays,indices):
    values=[];counts=Counter();positive=[];negative=[]
    head.eval()
    with torch.no_grad():
        for start in range(0,len(indices),16):
            ids=indices[start:start+16];x=torch.tensor(np.array(arrays['features'][ids]),device='cuda',dtype=torch.float32)
            targets={k:torch.tensor(np.array(arrays[k][ids]),device='cuda',dtype=torch.float32 if k=='weight' else torch.bool if k=='valid' else torch.long) for k in ['lo','hi','weight','valid']}
            logits=head(x,arm);values.append(float(M.loss(logits,**targets)));choice=logits.argmax(-1);valid=targets['valid'];pos=valid&(targets['lo']!=65);neg=valid&(targets['lo']==65)
            accepted=pos&(choice<65)
            counts.update(images=len(ids),valid=int(valid.sum()),ignored=int((~valid).sum()),positive=int(pos.sum()),
                no_match=int(neg.sum()),selected=int(((choice!=65)&valid).sum()),positive_accepted=int(accepted.sum()))
            if accepted.any():positive.extend((abs(choice[accepted].float()-(targets['lo'][accepted]+targets['weight'][accepted]))).cpu().tolist())
            if neg.any():negative.extend((choice[neg]!=65).cpu().tolist())
    head.train()
    return dict(loss_image_average=float(np.mean(values)),counts=dict(counts),positive_adoption_rate=counts['positive_accepted']/max(1,counts['positive']),
        positive_candidate_error_px_mean_conditional_accepted=float(np.mean(positive)) if positive else None,
        no_match_false_acceptance=float(np.mean(negative)) if negative else None)

def train():
    lock=freeze_protocol();manifest=C.read(CACHE/'CACHE_MANIFEST.json');assert manifest['complete']
    arrays={k:np.load(CACHE/(k+'.npy'),mmap_mode='r') for k in ['features','lo','hi','weight','valid']};order=np.load(CACHE/'order.npy');logs=[];start=time.monotonic();torch.set_num_threads(1)
    source_test=np.arange(896,1024);train_probe=np.arange(128);checkpoints=[]
    for arm in M.ARMS:
        torch.manual_seed(1);torch.cuda.manual_seed_all(1);head=M.CorrespondenceHead().cuda();initial=state_sha(head.state_dict());assert initial==lock['initial_state_sha256']
        optimizer=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.0001)
        begin=time.monotonic();initial_probe=probe(head,arm,arrays,source_test);first=None
        # Exactly100 throw-away numerical preflight updates, same fixed source batch; never published checkpoint.
        if arm==M.ARMS[0]:
            torch.manual_seed(1);smoke=M.CorrespondenceHead().cuda();opt=torch.optim.AdamW(smoke.parameters(),lr=.001,weight_decay=.0001);sstart=time.monotonic()
            for step in range(100):
                ids=order[step];x=torch.tensor(np.array(arrays['features'][ids]),device='cuda',dtype=torch.float32);ts={k:torch.tensor(np.array(arrays[k][ids]),device='cuda',dtype=torch.float32 if k=='weight' else torch.bool if k=='valid' else torch.long) for k in ['lo','hi','weight','valid']}
                opt.zero_grad();loss=M.loss(smoke(x,arm),**ts);assert torch.isfinite(loss);loss.backward();opt.step()
            torch.cuda.synchronize();logs.append(dict(kind='throwaway_preflight',updates=100,seconds=time.monotonic()-sstart,checkpoint_retained=False));del smoke,opt
        for step in range(1,3001):
            ids=order[step-1];x=torch.tensor(np.array(arrays['features'][ids]),device='cuda',dtype=torch.float32);ts={k:torch.tensor(np.array(arrays[k][ids]),device='cuda',dtype=torch.float32 if k=='weight' else torch.bool if k=='valid' else torch.long) for k in ['lo','hi','weight','valid']}
            lr=.001*(step/100 if step<=100 else .5*(1+math.cos(math.pi*(step-100)/2900)))
            for group in optimizer.param_groups:group['lr']=lr
            optimizer.zero_grad(set_to_none=True);logits=head(x,arm);value=M.loss(logits,**ts);assert torch.isfinite(value)
            if step==1:
                target_grad={}
                for label,mask in [('positive',ts['valid']&(ts['lo']!=65)),('no_match',ts['valid']&(ts['lo']==65)),('ignored',torch.zeros_like(ts['valid']))]:
                    val=M.loss(logits,ts['lo'],ts['hi'],ts['weight'],mask)
                    gg=torch.autograd.grad(val,head.body[0].weight,retain_graph=True)[0]
                    target_grad[label]=dict(queries=int(mask.sum()),loss=float(val),gradient_norm=float(gg.norm()))
                assert target_grad['ignored']['gradient_norm']==0
            value.backward()
            gradient=float(torch.nn.utils.clip_grad_norm_(head.parameters(),10,error_if_nonfinite=True))
            if step==1:
                first=dict(gradients={k:float(p.grad.norm()) if p.grad is not None else None for k,p in head.named_parameters()},image_channel_gradient=float(head.body[0].weight.grad[:,:19].norm()),role_channel_gradient=float(head.body[0].weight.grad[:,25:28].norm()),target_gradient=target_grad)
                if arm=='GEOMETRY_ONLY':assert first['image_channel_gradient']==0
                if arm=='IMAGE_NO_ROLE':assert first['role_channel_gradient']==0
            optimizer.step()
            if step==1 or step%100==0:
                torch.cuda.synchronize();probs=logits.detach().softmax(-1);correct=(1-ts['weight'])*probs.gather(-1,ts['lo'].clamp(min=0)[...,None])[...,0]+ts['weight']*probs.gather(-1,ts['hi'].clamp(min=0)[...,None])[...,0]
                record=dict(kind='formal',arm=arm,step=step,loss_image_average=float(value),lr=lr,gradient_norm=gradient,seconds=time.monotonic()-begin,
                    batch_images=len(ids),valid_queries=int(ts['valid'].sum()),positive_queries=int((ts['valid']&(ts['lo']!=65)).sum()),
                    no_match_queries=int((ts['valid']&(ts['lo']==65)).sum()),ignored_queries=int((~ts['valid']).sum()),
                    target_correct_probability=float(correct[ts['valid']].mean()) if ts['valid'].any() else None)
                logs.append(record);print('TRAIN',arm,step,3000,round(float(value),4),round(record['seconds'],1),flush=True)
            if step in [1000,2000,3000]:logs.append(dict(kind='source_curve',arm=arm,step=step,train=probe(head,arm,arrays,train_probe),source_test=probe(head,arm,arrays,source_test)))
            if time.monotonic()-start>3600:raise RuntimeError('formal training60min budget reached')
        checkpoint=FITS/(arm+'.pt');torch.save(dict(model=head.cpu().state_dict(),arm=arm,config=M.CONFIG,steps=3000,initial_state_sha256=initial,batch_order_sha256=lock['batch_order_sha256'],protocol_sha256=C.sha(C.DOC/'LEARNING_PROTOCOL.json')),checkpoint)
        checkpoints.append(dict(arm=arm,updates=3000,exposures=48000,initial_probe=initial_probe,first_step=first,checkpoint=C.binding(checkpoint),seconds=time.monotonic()-begin))
        del head,optimizer;torch.cuda.empty_cache()
    with (C.DOC/'TRAIN_LOGS.jsonl').open('w') as f:
        for row in logs:f.write(json.dumps(C.finite(row),allow_nan=False)+'\n')
    C.write(C.DOC/'TRAINING_COMPLETION.json',dict(complete=True,formal_updates=9000,throwaway_updates=100,total_updates=9100,
        formal_RGB_exposures=144000,batch=16,seed=1,checkpoints=checkpoints,wall_seconds=time.monotonic()-start,
        same_initial_tensor_sha=True,same_batch_order=True,same_update_budget=True,source_scores_model_selection=False,gpu=gpu()))

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','train']);a=p.parse_args();cv2.setNumThreads(1);torch.set_num_threads(1)
    if a.stage=='prepare':prepare()
    else:train()

if __name__=='__main__':main()
