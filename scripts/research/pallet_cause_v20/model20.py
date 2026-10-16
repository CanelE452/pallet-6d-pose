"""Invertible proposal encoding and loss weighting controls; frozen proposal remains unchanged."""
from context20 import *
from model19 import RetentionGate,oracle_alpha,observed_samples
import torch.nn.functional as F

def encode_direction(x):
    y=x.clone();d=x[...,225:227];length=x[...,227:228]
    y[...,223:225]=x[...,223:225]/.25
    y[...,225:227]=d/length.clamp_min(1e-8)
    y[...,227:228]=length/.15
    return y

class CauseGate(RetentionGate):
    def __init__(self,seed):
        super().__init__(seed);self.use_unit=False;self.point_encode.register_forward_pre_hook(self._encode);self.gate.register_forward_hook(self._capture)
    def _encode(self,module,args):return (encode_direction(args[0]),) if self.use_unit else args
    def _capture(self,module,args,out):self.output_logits=out[...,0]
    def forward(self,O,rep,arm='unit_equal',evidence_off=False,evidence_features=None):
        self.use_unit=arm.startswith('unit_');mode='local_gate' if arm.endswith('_local') else 'area_gate' if arm.endswith('_area') else 'line_gate';out=super().forward(O,rep,mode,evidence_off=evidence_off,evidence_features=evidence_features);out['logits']=self.output_logits;return out

def objective(out,gt,valid,arm):
    target=oracle_alpha(out['base'],out['proposal_delta'],gt).detach();weight=out['proposal_delta'].norm(dim=-1).detach() if arm.endswith('_weighted') else torch.ones_like(target);loss=(F.binary_cross_entropy_with_logits(out['logits'],target,reduction='none')*weight)[valid].mean();error=(out['points']-gt).norm(dim=-1)[valid].mean();return loss,error,target
