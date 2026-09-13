"""GT-free deployment wrapper using the exact frozen baseline feature path."""
import copy
import math
import numpy as np
import torch
from common import B, BRAW, R0, R0_SHA, old, read, sha
from generic_point_refiner import GenericPointRefiner, decode


def replace_selected(candidates, selected, points_input, refined, gain, lam):
    result=copy.deepcopy(candidates)
    if selected is None or lam==0:return result
    original=np.asarray(candidates[selected]['keypoints_xy'])
    moved=old('features').restore_refinement(candidates[selected],points_input,refined,gain,lam=lam)
    delta=refined-points_input
    changed=np.isfinite(points_input).all(-1)&np.any(delta!=0,axis=-1)
    changed[8]=False
    final=original.copy();final[changed]=moved[changed]
    result[selected]['keypoints_xy']=final
    return result


class PointInference:
    def __init__(self,seed,device='cuda'):
        selection=read(B/'P_SELECTION.json')
        assert selection['complete'] and selection['no_real_selection']
        self.device=device;self.seed=seed
        path=BRAW/f'runs/seed{seed}/last.pt'
        assert sha(path)==selection['checkpoints'][str(seed)]
        ck=torch.load(path,map_location='cpu',weights_only=False)
        assert ck['complete'] and ck['step']==6000 and ck['baseline_checkpoint_sha256']==R0_SHA
        self.head=GenericPointRefiner(**ck['config']).to(device)
        self.head.load_state_dict(ck['model_state_dict'],strict=True)
        self.head.requires_grad_(False).eval()
        self.rule=selection['selected_rule'];self.temperature=selection['temperatures'][str(seed)]['temperature']
        self.extractor=old('features').FrozenYoloFeatures(R0,device=device)

    @torch.no_grad()
    def predict(self,bgr):
        assert bgr.ndim==3 and bgr.shape[2]==3 and bgr.dtype==np.uint8
        captured=self.extractor.predict(bgr);selected=captured['selected_index']
        candidates=copy.deepcopy(captured['candidates']);lam=self.rule['lam']
        result=dict(candidates=candidates,selected_index=selected,head_used=False,lam=lam,
                    baseline_selected=None if selected is None else copy.deepcopy(candidates[selected]))
        if selected is None or lam==0:return result
        inputs=old('features').branch_inputs(captured)
        def tensor(x,dtype=torch.float32):return torch.as_tensor(x,device=self.device,dtype=dtype)[None]
        output=self.head(captured['p3'],captured['p4'],tensor(inputs['points']),tensor(inputs['boxes']),
                         tensor(inputs['point_valid'],torch.bool),tensor(inputs['input_shape']),lam=0)
        fraction=self.rule['max_move_image_diagonal_fraction']
        cap=None if fraction is None else fraction*math.hypot(*bgr.shape[:2])*inputs['gain']
        refined=decode(output,self.temperature,lam,cap)[0].cpu().numpy()
        assert np.isfinite(refined[inputs['point_valid']]).all()
        result['candidates']=replace_selected(candidates,selected,inputs['points'],refined,inputs['gain'],lam)
        result['head_used']=True
        result['diagnostics']=dict(null_probability=(output['logits']/self.temperature).softmax(-1)[0,:,-1].cpu().numpy(),
            move_px=np.linalg.norm(result['candidates'][selected]['keypoints_xy']-candidates[selected]['keypoints_xy'],axis=-1))
        return result

    def close(self):self.extractor.close()
