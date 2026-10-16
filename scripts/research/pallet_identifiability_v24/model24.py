from context24 import *
from model20 import CauseGate,oracle_alpha,objective
class DirectionGate(CauseGate):
    def forward(self,O,rep,arm='unit_sign',**kwargs):
        if arm.endswith('_parent'):
            O=dict(O,features=torch.zeros_like(O['features']));rep=dict(rep,context=torch.zeros_like(rep['context']))
        return super().forward(O,rep,arm,**kwargs)
def direction_target(rep,gt,valid):
    residual=gt-rep['base'];d=rep['proposal_delta'];label=(residual*d).sum(-1)>0
    mask=valid&(residual.norm(dim=-1)>.1)&(d.norm(dim=-1)>.1)
    return label.float(),mask
