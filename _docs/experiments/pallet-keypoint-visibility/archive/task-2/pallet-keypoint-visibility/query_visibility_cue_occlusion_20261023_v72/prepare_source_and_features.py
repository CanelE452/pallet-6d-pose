"""Normalize sealed source inputs; fresh whole-RGB frozen P3, fixed query/L.

GT metadata is stored separately and never passed to the backbone or head.
No private/current data, native keypoint recentering, or PnP is used.
"""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
from PIL import Image

R=Path(__file__).resolve().parent
P=R.parent
OUT=R/'source_cache'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def lines(p):return [json.loads(l) for l in Path(p).read_text().splitlines()]

def prepare():
    if OUT.exists():raise FileExistsError(OUT)
    OUT.mkdir()
    s=P/'query_visibility_reference_jitter_20261023_v71/data/synth_reference_jitter/BUILD_V2'
    y=R/'data/ycb_reference_cover'
    a=np.load(s/'MODEL_INPUTS_ONLY.npz',allow_pickle=False)
    b=np.load(y/'MODEL_INPUT_ONLY.npz',allow_pickle=False)
    sr=lines(s/'ROWS.jsonl');yi=lines(y/'INPUT_INDEX_LINEAGE_ONLY.jsonl');yg=lines(y/'GT_ONLY_TEACHERS.jsonl')
    cert=json.loads((s/'INTERVENTION_CERTIFICATES.json').read_text())['variants']
    refs={c['rank']:c['reference_GT_only'] for c in cert}
    assert len(sr)==260 and len(yi)==len(yg)==3645
    inp=dict(rgb96=np.concatenate((a['rgb'].transpose(0,3,1,2),b['rgb96'])),
      crop_valid96=np.concatenate((a['valid'][:,None].astype(bool),b['crop_valid96'])),
      query_valid=np.concatenate((np.isfinite(a['initial_query_xy']).all(1),b['query_valid'])),
      query_xy=np.concatenate((a['initial_query_xy'],b['query_xy'])),
      box_xyxy=np.concatenate((a['target_box_xyxy'],b['box_xyxy'])),L=np.concatenate((a['input_crop_L'],b['L'])))
    index=[];gt=[]
    for i,r in enumerate(sr):
        m=r['model_inputs'];ref=refs[r['cue_rank']]
        index.append(dict(input_index=i,row_id='SYNTH:'+r['row_id'],image=m['rgb']['path'],image_sha256=m['rgb']['sha256']))
        state={'D_teacher':'D','E_teacher':'E','U':'U'}[r['training_target_GT_only']]
        gt.append(dict(input_index=i,row_id=index[-1]['row_id'],source='SYNTH',split='TRAIN',cue_refID=r['reference_GT_only'],
          training_state=state,crop_availability=r['availability_GT_only'],query_valid=bool(inp['query_valid'][i]),
          reference_xy=ref['reference_xy'] if ref else None,uncertainty_bbox_xyxy=ref['uncertainty_bbox'] if ref else None,
          query_xy=inp['query_xy'][i].tolist(),L=float(inp['L'][i]),variant=r['variant'],
          ureason='LOCATION_STRESS' if r['jitter_role_GT_only']=='LOCATION_STRESS' else r['availability_GT_only'],
          source_group=r['source_group']))
    for j,(r,g) in enumerate(zip(yi,yg)):
        i=j+260;assert r['input_index']==g['input_index']==j and r['row_id']==g['row_id']
        index.append(dict(input_index=i,row_id='YCB:'+r['row_id'],image=r['image'],image_sha256=r['image_sha256']))
        gt.append(dict(input_index=i,row_id=index[-1]['row_id'],source='YCB',split=r['split'],cue_refID=g['cue_refID'],
          training_state=g['training_state'],crop_availability=g['availability'],query_valid=bool(inp['query_valid'][i]),
          reference_xy=g['reference_xy'],uncertainty_bbox_xyxy=g['uncertainty_bbox_xyxy'],query_xy=inp['query_xy'][i].tolist(),
          L=float(inp['L'][i]),variant=r['variant'],ureason=g['ureason'],source_group=r['scene_family_group']))
    np.savez_compressed(OUT/'MODEL_INPUT_ONLY.npz',**inp)
    for name,data in [('INPUT_INDEX_ONLY.jsonl',index),('GT_ONLY.jsonl',gt)]:
        (OUT/name).write_text(''.join(json.dumps(x,allow_nan=False)+'\n' for x in data))
    counts={}
    for r in gt:
        key='|'.join((r['source'],r['split'],r['training_state']));counts[key]=counts.get(key,0)+1
    write(OUT/'PREPARED.json',dict(N=len(gt),counts=counts,unique_RGB=len({r['image_sha256'] for r in index}),
      current_data=0,source_conditional_AI_teacher=True,native8_or_human_gold=False,
      source_bindings=[dict(path=str(x),sha256=sha(x)) for x in [s/'PILOT_RESULT.json',s/'MODEL_INPUTS_ONLY.npz',s/'ROWS.jsonl',s/'INTERVENTION_CERTIFICATES.json',y/'DATASET_RECEIPT.json',y/'MODEL_INPUT_ONLY.npz',y/'INPUT_INDEX_LINEAGE_ONLY.jsonl',y/'GT_ONLY_TEACHERS.jsonl']],
      outputs=[dict(path=str(OUT/x),sha256=sha(OUT/x)) for x in ['MODEL_INPUT_ONLY.npz','INPUT_INDEX_ONLY.jsonl','GT_ONLY.jsonl']]))

def features():
    import cv2,torch
    from torch.nn import functional as F
    sys.path.insert(0,'/tmp/pallet-bi-full-execution-20261019-v1/scripts/research')
    from pallet_line_pose_v1.features import FrozenYoloFeatures,canvas_affine
    out=OUT/'P3_ROIS.npz'
    if out.exists():raise FileExistsError(out)
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
    data=np.load(OUT/'MODEL_INPUT_ONLY.npz',allow_pickle=False);ix=lines(OUT/'INPUT_INDEX_ONLY.jsonl')
    groups={}
    for i,r in enumerate(ix):groups.setdefault(r['image_sha256'],[]).append(i)
    weights=P.parent/'..'/'..'  # explicitly replaced below, no project discovery
    weights=Path('/home/minjae/Documents/github/pallet-pose/challenge/yolo_pose_one_model/spatial_concat_scratch/runs/YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42/weights/best.pt')
    model=FrozenYoloFeatures(weights=weights,device='cuda');model.yolo.model.fuse(verbose=False);model.yolo.model.requires_grad_(False).eval()
    before={k:v.detach().cpu().clone() for k,v in model.yolo.model.state_dict().items()}
    yy,xx=torch.meshgrid(torch.arange(9,device='cuda'),torch.arange(9,device='cuda'),indexing='ij')
    offsets=torch.stack((xx-4,yy-4),-1).float()/9
    x=np.zeros((len(ix),65,9,9),np.float16);records=[];start=time.monotonic()
    with torch.no_grad():
      for h,indices in groups.items():
        r=ix[indices[0]];assert sha(r['image'])==h
        rgb=np.asarray(Image.open(r['image']).convert('RGB'));bgr=np.ascontiguousarray(rgb[:,:,::-1])
        c=model.predict(bgr,cpu_features=False);gain,off=canvas_affine(c['canvas_shape'],c['input_shape'])
        q=torch.tensor(data['query_xy'][indices],device='cuda',dtype=torch.float32)
        L=torch.tensor(data['L'][indices],device='cuda',dtype=torch.float32)
        grid=q[:,None,None,:]+offsets[None]*L[:,None,None,None]
        valid=torch.isfinite(grid).all(-1)&(grid[...,0]>=0)&(grid[...,0]<rgb.shape[1])&(grid[...,1]>=0)&(grid[...,1]<rgb.shape[0])
        coord=((grid+100)*gain+torch.tensor(off,device='cuda').float())/8
        norm=2*(coord+.5)/80-1
        roi=F.grid_sample(c['p3'].float().expand(len(indices),-1,-1,-1),norm,mode='bilinear',padding_mode='zeros',align_corners=False)
        roi=roi*valid[:,None]
        x[indices]=torch.cat((roi,valid[:,None].float()),1).cpu().numpy().astype(np.float16)
        records.append(dict(image=r['image'],image_sha256=h,rows=len(indices),raw_hw=list(rgb.shape[:2]),gain=gain,offset=off.tolist(),input_shape=list(c['input_shape'])))
        print(json.dumps(dict(images=len(records),total=len(groups),rows_done=sum(z['rows'] for z in records))),flush=True)
    assert all(torch.equal(v,model.yolo.model.state_dict()[k].detach().cpu()) for k,v in before.items())
    np.savez_compressed(out,x=x)
    write(OUT/'FEATURE_RECEIPT.json',dict(completed=True,rows=len(ix),whole_RGB_forwards=len(groups),seconds=time.monotonic()-start,
      GT_in_backbone_or_sampling=False,initial_q_L_box_preserved=True,backbone_state_unchanged=True,
      feature_shape=list(x.shape),feature_sha256=sha(out),input_sha256=sha(OUT/'MODEL_INPUT_ONLY.npz'),
      index_sha256=sha(OUT/'INPUT_INDEX_ONLY.jsonl'),code_sha256=sha(__file__),weights_sha256=sha(weights),RGB_records=records,
      PnP_calls=0,current_inputs=0))
    model.close()

if __name__=='__main__':
    {'prepare':prepare,'features':features}[sys.argv[1]]()
