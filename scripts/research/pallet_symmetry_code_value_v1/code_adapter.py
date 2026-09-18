"""Same existing eight-channel model; only the final three input values vary."""
import numpy as np
import torch
import cv_env as E
from refiner import DimensionConditionedPointRefiner,forward,train_loss,local_phase,context,decode,targets
from data import PaperData
from square_data import MixedData,SquareData

def model():return DimensionConditionedPointRefiner(context_dim=8,**E.read(E.D.DOC/'MODEL_AND_PARAMETER_AUDIT.json')['config'])
def alter(z,mode,donor_groups=None):
    out=z.clone()
    if mode in ['BLIND','NEUTRAL']:out[:,5:]=1/3
    elif mode=='ZERO':out[:,5:]=0
    elif mode=='WRONG':out[:,5:]=torch.roll(z[:,5:],1,-1)
    elif mode=='SHUFFLED':
        g=torch.as_tensor(donor_groups,device=z.device);assert g.shape==(len(z),) and torch.isin(g,g.new_tensor([1,2,4])).all()
        out[:,5:]=(g[:,None]==g.new_tensor([1,2,4])).to(z.dtype)
    else:assert mode=='CORRECT'
    assert torch.equal(out[:,:5],z[:,:5]);return out
def mode_for(arm):return 'BLIND' if 'BLIND' in arm else 'CORRECT'
class TrainingData:
    def __init__(self,track):
        self.track=track;self.base=PaperData() if track=='A' else MixedData()
    def order(self,seed):return np.load(E.RAW/f'orders/{self.track}_seed{seed}.npy',mmap_mode='r')
    def batch(self,rows,arm,device='cuda',supervision=True):
        b=self.base.batch(rows,'N4_META_SYM',device=device,supervision=supervision)
        b['context']=alter(b['context'],mode_for(arm));return b
