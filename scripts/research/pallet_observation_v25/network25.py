"""Inference composition: frozen representation and trainable correction in one module."""
from context25 import *
from network18 import VectorPointLineNetwork
from model25 import Correction25

class ObservationCorrectionNetwork(torch.nn.Module):
    def __init__(self,seed=1,arm='vector_line_edge'):
        super().__init__();self.arm=arm
        self.parent=VectorPointLineNetwork(seed,'line_vector').eval().requires_grad_(False)
        self.correction=Correction25(seed).cuda()
        self.correction.load_state_dict(torch.load(RUN/'models'/f'{arm}_seed{seed}'/'final.pt',weights_only=True,map_location='cpu')['state_dict'])
        self.eval()

    def forward(self,O):
        p=self.parent.parent({k:v for k,v in O.items() if k!='edge'})
        rep=dict(p['rep'],lines=p['lines'],line_weights=p['line_weights'])
        rep['proposal_delta']=self.parent.evaluator(O,rep,'line_vector')['points']-rep['base']
        return self.correction(O,rep,self.arm)
