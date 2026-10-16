"""Same parameter tensors; optional removal of only the frozen 64D parent context."""
from context22 import *
from model20 import CauseGate,objective as bce_objective,oracle_alpha

class ObjectiveGate(CauseGate):
    def forward(self,O,rep,arm='unit_pixel',**kwargs):
        if 'nocontext' in arm:rep=dict(rep,context=torch.zeros_like(rep['context']))
        return super().forward(O,rep,arm,**kwargs)

def objective22(out,gt,valid,arm):
    if 'bce' in arm:return bce_objective(out,gt,valid,'unit_weighted')
    error=torch.linalg.vector_norm(out['points']-gt,dim=-1)[valid].mean()
    return error,error,oracle_alpha(out['base'],out['proposal_delta'],gt).detach()
