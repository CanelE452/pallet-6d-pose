"""Deployment wrapper for direct residual control with canonical R0 detection."""
import copy,math
import numpy as np
import torch
from env import *
from direct_residual_control import DirectResidualControl,decode_direct
from point_inference import replace_selected
class DirectInference:
    def __init__(self,seed,device='cuda'):
        sel=read(DOC/'D_SELECTION.json');assert sel['complete'] and sel['no_real_selection']
        path=RAW/f'runs/D{seed}/last.pt';assert sha(path)==sel['checkpoints'][str(seed)]
        ck=torch.load(path,map_location='cpu',weights_only=False);assert ck['complete'] and ck['step']==6000
        self.head=DirectResidualControl(**ck['config']).to(device);self.head.load_state_dict(ck['model_state_dict']);self.head.requires_grad_(False).eval()
        self.device=device;self.rule=sel['selected_rule'];self.extractor=old('features').FrozenYoloFeatures(R0,device=device)
    @torch.no_grad()
    def predict(self,bgr):
        captured=self.extractor.predict(bgr);selected=captured['selected_index'];cand=copy.deepcopy(captured['candidates']);lam=self.rule['lam']
        result=dict(candidates=cand,selected_index=selected,head_used=False,lam=lam,baseline_selected=None if selected is None else copy.deepcopy(cand[selected]))
        if selected is None or lam==0:return result
        i=old('features').branch_inputs(captured)
        tensor=lambda a,dtype=torch.float32:torch.as_tensor(a,device=self.device,dtype=dtype)[None]
        out=self.head(captured['p3'],captured['p4'],tensor(i['points']),tensor(i['boxes']),tensor(i['point_valid'],torch.bool),tensor(i['input_shape']),lam=0)
        f=self.rule['max_move_image_diagonal_fraction'];cap=None if f is None else f*math.hypot(*bgr.shape[:2])*i['gain']
        q=decode_direct(out,lam,cap)[0].cpu().numpy()
        result['candidates']=replace_selected(cand,selected,i['points'],q,i['gain'],lam);result['head_used']=True
        return result
    def close(self):self.extractor.close()
