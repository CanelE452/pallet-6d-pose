"""GT-free whole-image R0 plus separate RGB PoseFix-derived backbone."""
import copy,math
import cv2,numpy as np,torch
from env import *
from prior_model import PoseFixPallet9,expectation
from posefix_contract_math import axis_aligned_crop_matrix,transform_points
def correct(points,raw,valid,rule,shape):
    p=np.asarray(points).copy();valid=np.asarray(valid,bool)&np.isfinite(p).all(-1);valid[8]=False
    if rule['lam']==0 or rule['max_move_image_diagonal_fraction']==0:return p
    assert np.isfinite(raw[valid]).all(),'Nonfinite network prediction; never silently exclude from evaluation'
    delta=(raw-p)*rule['lam'];fraction=rule['max_move_image_diagonal_fraction']
    if fraction is not None:
        cap=fraction*math.hypot(*shape);norm=np.linalg.norm(delta,axis=-1)
        delta*=np.minimum(1,cap/np.maximum(norm,1e-12))[:,None]
    p[valid]+=delta[valid];return p
class PriorInference:
    def __init__(self,seed,device='cuda'):
        selection=read(DOC/'PRIOR_SELECTION.json');path=RAW/f'runs/PRIOR{seed}/last.pt'
        assert selection['complete'] and sha(path)==selection['checkpoints'][str(seed)]
        ck=torch.load(path,map_location='cpu',weights_only=False);assert ck['complete'] and ck['step']==6000
        self.head=PoseFixPallet9().to(device);self.head.load_state_dict(ck['model_state_dict']);self.head.eval().requires_grad_(False)
        self.rule=selection['selected_rule'];self.device=device;self.extractor=old('features').FrozenYoloFeatures(R0,device=device)
    @torch.no_grad()
    def predict(self,bgr):
        captured=self.extractor.predict(bgr);selected=captured['selected_index'];candidates=copy.deepcopy(captured['candidates'])
        result=dict(candidates=candidates,selected_index=selected,head_used=False,lam=self.rule['lam'],baseline_selected=None if selected is None else copy.deepcopy(candidates[selected]))
        if selected is None:return result
        c=candidates[selected]
        if c['keypoints_xy'] is None:return result
        points=np.asarray(c['keypoints_xy'],float);box=np.asarray(c['box_xyxy'],float)
        if not np.isfinite(box).all() or not (box[2:]>box[:2]).all():return result
        valid=np.isfinite(points).all(-1)&~(points==-1).all(-1);matrix=axis_aligned_crop_matrix(box)
        rgb=cv2.warpAffine(bgr,matrix[:2],(288,384),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)[:,:,::-1].astype(np.float32)-np.array([123.68,116.78,103.94],np.float32)
        cp=transform_points(np.where(valid[:,None],points,0),matrix).astype(np.float32)
        tensors=[torch.as_tensor(x,device=self.device)[None] for x in (rgb.transpose(2,0,1),cp,valid)]
        with torch.backends.cudnn.flags(enabled=True,benchmark=False,deterministic=False,allow_tf32=False):
            q=expectation(self.head(*tensors))[0].cpu().numpy()
        raw=transform_points(q,np.linalg.inv(matrix));raw[~valid]=points[~valid];raw[8]=points[8]
        result['raw_keypoints_xy']=raw
        result['candidates'][selected]['keypoints_xy']=correct(points,raw,valid,self.rule,bgr.shape[:2]);result['head_used']=True
        return result
    def close(self):self.extractor.close()
