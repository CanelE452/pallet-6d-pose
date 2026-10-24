"""V96 root-authorized fixed first changedRGB numerical replay; no query selection."""
from pathlib import Path
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import hashlib
import importlib.util
import json
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import cv2

HERE=Path(__file__).resolve().parent;V=HERE.parent


def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(x)for x in Path(p).read_text().splitlines()]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(p),sha256=sha(p))
def write(p,x):
    with Path(p).open('x')as f:json.dump(x,f,indent=2);f.write('\n')
def state(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):
        a=v.detach().cpu().contiguous().numpy();h.update(k.encode()+b'\0'+str(a.dtype).encode()+b'\0'+str(a.shape).encode()+b'\0'+a.tobytes())
    return h.hexdigest()


def main():
    assert torch.__version__=='2.1.1+cu118'and cv2.__version__=='4.9.0'
    assert sha(V/'actual_source_pair_input_pack/INPUT_PACK_SEAL.json')=='eb1607140fb1466a5aaad4ce839e4b9471c5514abd49288c7572a916123773f6'
    assert read(V/'actual_source_pair_input_pack/INPUT_PACK_SEAL.json')['completed']
    p=read(V/'source_input_preparation/FEATURE_PROTOCOL_BEFORE_SOURCE_EXECUTION.json')
    rows=lines(V.parent/'query_visibility_same_q_counterfactual_20261023_v95/source_pair_intake/ALL960_NORMALIZED_INPUT_ONLY.jsonl')
    occ=lines(V/'actual_query_anchored_image_intake/OCCURRENCES31680_LINEAGE.jsonl')
    actual=lines(V/'actual_source_pair_input_pack/ALL31680_INPUT_LINEAGE.jsonl')
    first=next(x for x in occ if x['full_rgb_changed']);assert first['slot_index']==176 and first['occurrence_index']==5813 and first['donor_rank']==2 and first['variant']=='OPAQUE_COVER'
    row=rows[176];rec=actual[5813]
    for d in p['assets'].values():assert sha(d['path'])==d['sha256']
    assert sha(first['edited_fullRGB_file']['path'])==first['edited_fullRGB_file']['sha256']
    write(HERE/'FIRST_CHANGED_REPLAY_STARTED.json',dict(code=bind(__file__),base_row_index=176,occurrence_index=5813,variant='OPAQUE_COVER',
          selected_edited_source=first['edited_fullRGB_file'],root_authorized_source_only=True,source_image_predict_calls_planned=1,blank_warmup_planned=1,
          initializer_query_reselection=0,head_forward_fit_current_PnP=0))
    rgb=np.asarray(Image.open(first['edited_fullRGB_file']['path']).convert('RGB')).copy()
    assert hashlib.sha256(rgb.tobytes()).hexdigest()==first['full_rgb_decoded_sha256']
    torch.set_num_threads(1);cv2.setNumThreads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    os.environ['YOLO_CONFIG_DIR']=str(HERE/'runtime/ultralytics')
    path=p['assets']['backbone_code']['path'];spec=importlib.util.spec_from_file_location('independent_v96_source_frozen_backbone',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    back=module.FrozenYoloFeatures(p['assets']['backbone_checkpoint']['path'],device='cuda:0')
    with torch.inference_mode():back.predict(np.zeros((480,640,3),np.uint8),already_padded=False,cpu_features=True)
    before=state(back.yolo.model)
    with torch.inference_mode():z=back.predict(np.ascontiguousarray(rgb[...,::-1]),already_padded=False,cpu_features=True)
    assert before==state(back.yolo.model)
    full=z['p3'];assert full.shape==(64,80,80)and full.dtype==torch.float16
    q=np.array(row['query_xy'],np.float64);L=float(row['L']);h,w=row['raw_hw']
    padded_hw=[h+200,w+200];gain=min(640/padded_hw[0],640/padded_hw[1]);rw,rh=round(padded_hw[1]*gain),round(padded_hw[0]*gain)
    input_hw=[rh+(640-rh)%32,rw+(640-rw)%32];offset=np.array([round((input_hw[1]-rw)/2-.1),round((input_hw[0]-rh)/2-.1)],np.float64)
    assert z['added_border']==100 and list(z['canvas_shape'])==padded_hw and list(z['input_shape'])==input_hw
    t=(np.arange(9,dtype=np.float64)+.5)/9-.5
    gx,gy=np.meshgrid(q[0]+t*L,q[1]+t*L);native=np.stack((gx,gy),-1)
    valid=(gx>=0)&(gx<w)&(gy>=0)&(gy<h)
    index=((native+100)*gain+offset)/8
    norm=2*(index+.5)/80-1
    with torch.inference_mode():
        sampled=F.grid_sample(full[None].float(),torch.tensor(norm[None],dtype=torch.float32),mode='bilinear',padding_mode='zeros',align_corners=False)
        roi=torch.cat((sampled*torch.tensor(valid[None,None],dtype=torch.float32),torch.tensor(valid[None,None],dtype=torch.float32)),1).half().numpy()[0]
    ref=rec['input_ref'];assert sha(ref['file']['path'])==ref['file']['sha256']
    with np.load(ref['file']['path'],allow_pickle=False)as f:saved=f['p3_roi'][ref['index']]
    equal=bool(np.array_equal(roi,saved));diff=np.abs(roi.astype(np.float32)-saved.astype(np.float32))
    np.savez_compressed(HERE/'FIXED_FIRST_CHANGED_ROI_REPLAY.npz',independent_roi=roi,saved_roi=saved)
    manifest=read(V/'actual_source_pair_input_pack/PACK_MANIFEST.json')
    result=dict(completed=True,PASS=equal,base_row_index=176,occurrence_index=5813,variant='OPAQUE_COVER',
       source_image_predict_calls=1,blank_warmup_calls=1,initializer_query_reselection=0,head_forward_fit_current_PnP=0,
       source_checkpoint=p['assets']['backbone_checkpoint'],backbone_code=p['assets']['backbone_code'],
       runtime=dict(python=sys.executable,torch=torch.__version__,cv2=cv2.__version__),
       independent_sampler='Own float64 native grid/affine -> FP32 normalized grid -> CPU grid_sample align_cornersFalse -> F16; no producer sampler import.',
       ROI_F16_bitexact=equal,different_values=int(np.count_nonzero(diff)),max_abs_difference=float(diff.max()),
       model_unchanged=True,warmed_model_state=before,state_matches_producer=before==manifest['model_state_before'],
       original_q=row['query_xy'],original_L=row['L'],raw_editedRGB_sha=first['full_rgb_decoded_sha256'],replay_arrays=bind(HERE/'FIXED_FIRST_CHANGED_ROI_REPLAY.npz'))
    back.close();write(HERE/'FIRST_CHANGED_BACKBONE_REPLAY.json',result)
    assert equal and result['state_matches_producer'],json.dumps(result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
