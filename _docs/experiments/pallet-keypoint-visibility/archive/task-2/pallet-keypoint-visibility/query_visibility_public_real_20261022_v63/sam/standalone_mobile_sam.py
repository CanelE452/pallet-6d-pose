"""Frozen official MobileSAM source-only box baseline, with isolated scoring.

No legacy experiment module, CAL model, human pallet annotation, or pose solver
is imported. Source queries are pre-existing supervised query sets; they are
never SAM positive prompts. GT masks are opened only by score(), after seal.
"""
import argparse,csv,gzip,hashlib,json,sys,time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
TASK=ROOT.parent.parent
V60=TASK/'external_ray_proxy_head_20261022_v60'
V61=TASK/'target_presence_classifier_20261022_v61'
VENDOR=Path('/home/minjae/Documents/github/pallet-pose/_docs/experiments/pallet_visibility_curriculum_20261013_v1/foundation_mask_v1/vendor/MobileSAM')
CHECKPOINT=VENDOR/'weights/mobile_sam.pt'
CHECKPOINT_SHA='6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,x):Path(path).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def lines(path):return[json.loads(s)for s in Path(path).read_text().splitlines()if s.strip()]
def csvrows(path):
    with Path(path).open(newline='')as f:return list(csv.DictReader(f))
def seed(s):return int(hashlib.sha256(s.encode()).hexdigest()[:8],16)
def dumpcsv(path,rows):
    with Path(path).open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def roundpixel(q,hw):
    q=np.array(q,float);p=np.floor(q+.5).astype(int);h,w=hw
    return p, bool(np.isfinite(q).all() and 0<=q[0]<w and 0<=q[1]<h and 0<=p[0]<w and 0<=p[1]<h)


def prepare():
    assert not(HERE/'SOURCE_INPUTS.jsonl').exists(),'preserve existing registered subset'
    denied=set(json.loads((ROOT/'contract/EVALUATION_SHA256_DENYLIST.json').read_text())['image_sha256'])
    ycb=ROOT/'ycb/SOURCE_VAL_LOCKED_200.jsonl'
    queries_path=ROOT/'source_cache/v2/YCB_VALIDATION_QUERIES.csv'
    querygroup={}
    for q in csvrows(queries_path):querygroup.setdefault(q['id'],[]).append(q)
    manifest=lines(ycb);frames={}
    for r in manifest:frames.setdefault((r['scene_id'],r['frame_id']),[]).append(r)
    assert len(frames)==200
    pred=[];truth=[]
    for scene,frame in sorted(frames)[::4]:
        candidates=[r for r in frames[scene,frame]if r['full_pixels']>=128 and r['vis_pixels']>=16]
        candidates.sort(key=lambda r:seed(f"20261022:{scene}:{frame}:{r['instance_id']}"))
        for r in candidates[:2]:
            ident=f"YCB:{scene:06d}:{frame:06d}:{r['instance_id']:06d}"
            qq=sorted(querygroup[ident],key=lambda x:int(x['query_index']));assert len(qq)==8
            h,w=r['height'],r['width'];bx,by,bw,bh=r['bbox_obj'];center=np.array([bx+(bw-1)/2,by+(bh-1)/2])
            factor=float(np.random.default_rng(seed('SAM_SOURCE_BOX:'+ident)).uniform(.9,1.1))
            half=np.array([max(1,bw-1),max(1,bh-1)])*factor/2
            box=np.r_[np.maximum(center-half,[0.,0.]),np.minimum(center+half,[w-1.,h-1.])]
            assert r['image_sha256']not in denied
            pred.append(dict(id=ident,source='YCB',image=r['image'],image_sha256=r['image_sha256'],raw_hw=[h,w],prompt_box_xyxy=box.tolist(),prompt_box_source='source annotation bbox_obj with fixed hash scale jitter',prompt_scale_factor=factor,query_coordinates=[[float(q['query_x']),float(q['query_y'])]for q in qq],query_ids=[q['sample_id']for q in qq],query_native_ids=[int(q['query_index'])for q in qq],supervised_source_query_sampling=True))
            truth.append(dict(id=ident,source='YCB',mask_vis=r['mask_vis'],mask_full=r['mask_full']))
    prepared=V60/'PREPARED_INPUTS_SYNTH.json.gz'
    with gzip.open(prepared,'rt')as f:synth=json.load(f)
    maskaudit={r['id']:r for r in csvrows(V61/'MASK_CONTRACT_AUDIT.csv')}
    val=sorted([r for r in synth if r['split']=='validation'],key=lambda r:r['id'])
    assert len(val)==235
    for r in val[::4]:
        ident=r['id'];q=np.array(r['initial_points'],float)[:8];finite=np.isfinite(q).all(1)&~(q==-1).all(1)
        h,w=r['raw_hw'];lo=np.maximum(q[finite].min(0),[0.,0.]);hi=np.minimum(q[finite].max(0),[w-1.,h-1.])
        assert finite.sum()>=2 and (hi-lo>=2).all(),('Invalid registered initial box',ident)
        assert r['image_sha256']not in denied
        pred.append(dict(id=ident,source='SYNTH',image=r['image'],image_sha256=r['image_sha256'],raw_hw=[h,w],prompt_box_xyxy=np.r_[lo,hi].tolist(),prompt_box_source='unchanged initial query8 minmax clipped to raw image; centroid excluded',query_coordinates=q.tolist(),query_ids=[f'SYNTH:{ident}:{k}'for k in range(8)],query_native_ids=list(range(8)),supervised_source_query_sampling=False))
        truth.append(dict(id=ident,source='SYNTH',mask_vis=maskaudit[ident]['visible_path'],mask_full=maskaudit[ident]['amodal_path']))
    assert len({r['image_sha256']for r in pred if r['source']=='YCB'})==50
    assert len({r['image_sha256']for r in pred if r['source']=='SYNTH'})==59
    with (HERE/'SOURCE_INPUTS.jsonl').open('w')as f:
        for r in pred:f.write(json.dumps(r)+'\n')
    with (HERE/'SCORING_ONLY_MASKS.jsonl').open('w')as f:
        for r in truth:f.write(json.dumps(r)+'\n')
    lock=dict(schema='STANDALONE_FROZEN_MOBILESAM_SOURCE_ONLY_V1',source_subset='YCB200 sorted(scene,frame)[::4]=50frames,2stable-source-hash-selected targets/frame; SYNTHVAL235 sortedid[::4]=59frames',targets=len(pred),source_frame_counts=dict(YCB=50,SYNTH=59),
              checkpoint=str(CHECKPOINT),checkpoint_sha256=CHECKPOINT_SHA,model_type='vit_t',vendor=str(VENDOR),prompt='box only; no point prompts; unchanged existing query coordinates are readouts, not prompts',mask_selection='highest finite uncalibrated predicted IoU; exact tie first index',mask_logit_threshold=0.,
              output_targets={'radius0':'exact nearest pixel presence','radius1':'any visible-instance support in clipped3x3 square centered at nearest query pixel','radius2':'any visible-instance support in clipped5x5 square centered at nearest query pixel'},
              target_distinction='radius1/radius2 are separate finite-neighborhood targets, never relabel exact pixel metric',query_sampling='YCB stored source-mask-supervised query set reused unchanged; masks do not enter SAM prompt, inference or mask alternative selection',
              private_evaluation_accessed=False,real_CAL_model_accessed=False,new_PnP_calls=0,training=False,
              sources=[dict(path=str(p),sha256=sha(p))for p in [ycb,queries_path,prepared,V61/'MASK_CONTRACT_AUDIT.csv']],input_sha256=sha(HERE/'SOURCE_INPUTS.jsonl'),scoring_mask_manifest_sha256=sha(HERE/'SCORING_ONLY_MASKS.jsonl'),code_sha256=sha(__file__))
    write(HERE/'PREPARATION_LOCK.json',lock);print(json.dumps(dict(prepared_targets=len(pred),YCB_frames=50,SYNTH_frames=59)))


def load_model(device):
    import torch
    assert sha(CHECKPOINT)==CHECKPOINT_SHA
    sys.path.insert(0,str(VENDOR))
    from mobile_sam import SamPredictor,sam_model_registry
    import mobile_sam
    assert Path(mobile_sam.__file__).resolve().is_relative_to(VENDOR)
    model=sam_model_registry['vit_t']()
    model.load_state_dict(torch.load(CHECKPOINT,map_location='cpu',weights_only=True),strict=True)
    model=model.to(device).eval().requires_grad_(False)
    assert not any(p.requires_grad for p in model.parameters()) and model.mask_threshold==0
    return model,SamPredictor(model)


def smoke():
    import torch
    torch.set_num_threads(1)
    model,_=load_model('cpu')
    result=dict(cpu_model_loaded=True,checkpoint_sha256=sha(CHECKPOINT),parameters=sum(p.numel()for p in model.parameters()),frozen=True,model_forward_calls=0,current_eval_or_source_image_reads=0,real_CAL_model_accessed=False,new_PnP_calls=0)
    write(HERE/'CPU_LOAD_SMOKE.json',result);print(json.dumps(result))


def infer(device):
    import torch
    from PIL import Image
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    lock=json.loads((HERE/'PREPARATION_LOCK.json').read_text())
    assert lock['code_sha256']==sha(__file__) and lock['input_sha256']==sha(HERE/'SOURCE_INPUTS.jsonl')
    assert not(HERE/'PREDICTION_SEAL.json').exists(),'preserve completed inference'
    denied=set(json.loads((ROOT/'contract/EVALUATION_SHA256_DENYLIST.json').read_text())['image_sha256'])
    model,predictor=load_model(device);out=HERE/'masks';out.mkdir(exist_ok=True)
    rows=lines(HERE/'SOURCE_INPUTS.jsonl');previous=None;predictions=[];queryrows=[];frames=0;start=time.monotonic()
    if device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for i,r in enumerate(rows):
            # The inference manifest contains no GT mask paths or human labels.
            assert not any('mask'in k for k in r)
            assert r['image_sha256']not in denied
            if previous!=r['image']:
                assert sha(r['image'])==r['image_sha256']
                image=np.array(Image.open(r['image']).convert('RGB'));assert list(image.shape[:2])==r['raw_hw']
                predictor.set_image(image);previous=r['image'];frames+=1
            logits,quality,_=predictor.predict(box=np.array(r['prompt_box_xyxy'],np.float32),multimask_output=True,return_logits=True)
            quality=np.asarray(quality,float);assert quality.shape==(3,)and np.isfinite(quality).all()
            choice=int(np.argmax(quality));chosen=logits[choice];mask=chosen>0
            path=out/(hashlib.sha256(r['id'].encode()).hexdigest()[:20]+'.npz')
            assert not path.exists(),'preserve prior partial inference; use separate new run after diagnosing'
            np.savez_compressed(path,mask=mask.astype(np.uint8),image_sha256=r['image_sha256'])
            h,w=r['raw_hw']
            for qi,(qid,q,k)in enumerate(zip(r['query_ids'],r['query_coordinates'],r['query_native_ids'])):
                (px,py),valid=roundpixel(q,[h,w])
                for radius in[0,1,2]:
                    score=float(chosen[max(0,py-radius):min(h,py+radius+1),max(0,px-radius):min(w,px+radius+1)].max())if valid else None
                    queryrows.append(dict(id=r['id'],source=r['source'],query_id=qid,native_corner=k,query_x=q[0],query_y=q[1],pixel_x=int(px),pixel_y=int(py),radius=radius,query_valid=valid,visible_support_logit=score,pred_visible_support=bool(score>0)if score is not None else None))
            predictions.append(dict(id=r['id'],source=r['source'],image_sha256=r['image_sha256'],mask_path=str(path),mask_sha256=sha(path),selected_index=choice,uncalibrated_mask_quality=quality.tolist(),mask_pixels=int(mask.sum())))
            if(i+1)%20==0:print('SAM_SOURCE',i+1,len(rows),'seconds',round(time.monotonic()-start,1),flush=True)
    if device.startswith('cuda'):torch.cuda.synchronize()
    with(HERE/'TARGET_PREDICTIONS.jsonl').open('w')as f:
        for r in predictions:f.write(json.dumps(r)+'\n')
    dumpcsv(HERE/'QUERY_PREDICTIONS.csv',queryrows)
    seal=dict(input_sha256=lock['input_sha256'],code_sha256=sha(__file__),checkpoint_sha256=CHECKPOINT_SHA,targets=len(rows),image_encoder_forwards=frames,prompt_decoder_calls=len(rows),device=device,seconds=time.monotonic()-start,peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated())if device.startswith('cuda')else None,GT_masks_read_during_inference=False,current_eval_accessed=False,real_CAL_model_accessed=False,new_PnP_calls=0,files=[dict(path=str(p),sha256=sha(p))for p in[HERE/'TARGET_PREDICTIONS.jsonl',HERE/'QUERY_PREDICTIONS.csv']])
    write(HERE/'PREDICTION_SEAL.json',seal);print(json.dumps(seal),flush=True)


def score():
    from PIL import Image
    lock=json.loads((HERE/'PREPARATION_LOCK.json').read_text());seal=json.loads((HERE/'PREDICTION_SEAL.json').read_text())
    assert lock['scoring_mask_manifest_sha256']==sha(HERE/'SCORING_ONLY_MASKS.jsonl')
    for b in seal['files']:assert sha(b['path'])==b['sha256']
    targets={r['id']:r for r in lines(HERE/'TARGET_PREDICTIONS.jsonl')};truth={r['id']:r for r in lines(HERE/'SCORING_ONLY_MASKS.jsonl')}
    masks={};ious=[];rows=[]
    for ident,r in truth.items():
        mask_threshold=127 if r['source']=='SYNTH' else 0
        v=np.array(Image.open(r['mask_vis']))>mask_threshold;f=np.array(Image.open(r['mask_full']))>mask_threshold;assert v.shape==f.shape and not(v&~f).any()
        p=targets[ident];assert sha(p['mask_path'])==p['mask_sha256']
        with np.load(p['mask_path'])as z:m=z['mask'].astype(bool)
        assert m.shape==v.shape;intersection=int((m&v).sum());union=int((m|v).sum())
        ious.append(dict(id=ident,source=r['source'],intersection=intersection,union=union,visible_mask_IoU=intersection/union if union else None))
        masks[ident]=v,f
    for q in csvrows(HERE/'QUERY_PREDICTIONS.csv'):
        if q['query_valid']!='True':continue
        v,f=masks[q['id']];h,w=v.shape;px,py,rad=int(q['pixel_x']),int(q['pixel_y']),int(q['radius'])
        patch=(slice(max(0,py-rad),min(h,py+rad+1)),slice(max(0,px-rad),min(w,px+rad+1)))
        vy,fy=bool(v[patch].any()),bool(f[patch].any())
        state='VISIBLE_TARGET_SUPPORT'if vy else'EXTERNAL_OCCLUDED_SUPPORT'if fy else'OUTSIDE_TARGET_SUPPORT'
        rows.append(dict(**q,target_visible_support=vy,source_support_state=state))
    metrics=[];conf=[]
    for domain in['SYNTH','YCB']:
        for rad in[0,1,2]:
            a=[r for r in rows if r['source']==domain and int(r['radius'])==rad]
            tp=sum(r['target_visible_support']and r['pred_visible_support']=='True'for r in a);fp=sum(not r['target_visible_support']and r['pred_visible_support']=='True'for r in a)
            fn=sum(r['target_visible_support']and r['pred_visible_support']=='False'for r in a);tn=sum(not r['target_visible_support']and r['pred_visible_support']=='False'for r in a)
            ratio=lambda x,y:x/y if y else None
            metrics.append(dict(source=domain,radius=rad,target_name='exact_query_pixel_presence'if rad==0 else f'clipped_square_radius{rad}_any_visible_support',N=len(a),TP=tp,FP=fp,FN=fn,TN=tn,visible_precision=ratio(tp,tp+fp),visible_recall=ratio(tp,tp+fn),absent_precision=ratio(tn,tn+fn),absent_recall=ratio(tn,tn+fp)))
            for state in['VISIBLE_TARGET_SUPPORT','EXTERNAL_OCCLUDED_SUPPORT','OUTSIDE_TARGET_SUPPORT']:
                z=[r for r in a if r['source_support_state']==state];pv=sum(r['pred_visible_support']=='True'for r in z)
                conf.append(dict(source=domain,radius=rad,source_support_state=state,N=len(z),pred_visible_support=pv,pred_absent=len(z)-pv))
    dumpcsv(HERE/'SOURCE_QUERY_SCORED.csv',rows);dumpcsv(HERE/'SOURCE_METRICS.csv',metrics);dumpcsv(HERE/'SOURCE_SUPPORT_CONFUSION.csv',conf);dumpcsv(HERE/'VISIBLE_MASK_IOU.csv',ious)
    result=dict(scope='source-only frozen box-prompted general MobileSAM; no current private eval',code_sha256=sha(__file__),prediction_seal_sha256=sha(HERE/'PREDICTION_SEAL.json'),metrics=metrics,mean_visible_mask_IoU_by_source={d:float(np.mean([r['visible_mask_IoU']for r in ious if r['source']==d]))for d in['SYNTH','YCB']},semantic_self='not supplied by target mask; no semantic self performance claim',new_PnP_calls=0,source_GT_masks_opened_only_in_score=True,source_mask_thresholds=dict(SYNTH='>127',YCB='>0'))
    write(HERE/'SOURCE_EVALUATION.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare','smoke','infer','score']);ap.add_argument('--device',default='cuda:0');args=ap.parse_args()
    {'prepare':prepare,'smoke':smoke,'infer':lambda:infer(args.device),'score':score}[args.command]()
