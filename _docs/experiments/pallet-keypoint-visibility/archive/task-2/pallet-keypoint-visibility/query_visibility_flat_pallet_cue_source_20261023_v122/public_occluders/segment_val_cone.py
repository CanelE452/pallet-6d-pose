from pathlib import Path
import json,hashlib,sys,time
import numpy as np
from PIL import Image
P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility');V=Path(__file__).parent;R=P/'foundation_runtime_sam21_20261023';sys.path.insert(0,str(R/'sam2_repo'))
policy={'source_only':True,'image':'VAL_Small-Traffic-Cone-Edited.png','purpose':'Separate cone body from supplied photograph opaque cast shadow before use as foreground. Candidate SAM mask, visually review before AUX use.','box_xyxy':[35,8,720,992],'positive_xy':[[400,100],[350,500],[350,850],[620,930]],'negative_xy':[[900,550],[700,700],[750,900],[100,100]],'current_or_V112':False,'fit':False,'PnP':False}
(V/'VAL_CONE_MASK_POLICY_BEFORE_FORWARD.json').write_text(json.dumps(policy,indent=2)+'\n')
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
start=time.monotonic();model=build_sam2('configs/sam2.1/sam2.1_hiera_b+.yaml',str(R/'checkpoints/sam2.1_hiera_base_plus.pt'),device='cuda',mode='eval',apply_postprocessing=False,hydra_overrides_extra=['++model.compile_image_encoder=false']);model.requires_grad_(False)
pr=SAM2ImagePredictor(model,mask_threshold=0.,max_hole_area=0.,max_sprinkle_area=0.)
a=np.array(Image.open(V/policy['image']).convert('RGBA'));rgb=a[:,:,:3].copy();rgb[a[:,:,3]==0]=255
with torch.inference_mode():
 pr.set_image(rgb)
 masks,scores,_=pr.predict(point_coords=np.asarray(policy['positive_xy']+policy['negative_xy'],np.float32),point_labels=np.array([1]*4+[0]*4,np.int32),box=np.asarray(policy['box_xyxy'],np.float32),multimask_output=False,return_logits=False)
mask=np.asarray(masks[0],bool)&(a[:,:,3]>0)
np.savez_compressed(V/'VAL_CONE_BODY_MASK_CANDIDATE.npz',mask=mask)
over=rgb.copy();over[mask]=np.round(.55*over[mask]+.45*np.array([0,255,255])).astype(np.uint8);Image.fromarray(over).save(V/'VAL_CONE_MASK_OVERLAY.png')
Image.fromarray(np.dstack([a[:,:,:3],np.where(mask,a[:,:,3],0).astype(np.uint8)])).save(V/'VAL_CONE_BODY_RGBA_CANDIDATE.png')
(V/'VAL_CONE_MASK_RESULT.json').write_text(json.dumps({'completed':True,'quality_score':float(scores[0]),'mask_pixels':int(mask.sum()),'source_RGBA_sha256':hashlib.sha256((V/policy['image']).read_bytes()).hexdigest(),'candidate_not_gold':True,'sam_calls':1,'set_image_calls':1,'fit':0,'current_V112':0,'seconds':time.monotonic()-start},indent=2)+'\n')
print(json.dumps({'completed':True,'quality':float(scores[0]),'mask_pixels':int(mask.sum())}))
