"""Full frozen YOLO feature/DHT parent + point proposal + learned V27 module."""
from context27 import *
from network18 import VectorPointLineNetwork
from model25 import Correction25
from model27 import DirectionUtility27

class DirectionUtilityNetwork(torch.nn.Module):
    def __init__(self,seed=1,arm='line_role_utility'):
        super().__init__();self.arm=arm;self.parent=VectorPointLineNetwork(seed,'line_vector').eval().requires_grad_(False);self.point=Correction25(seed).cuda().eval().requires_grad_(False);self.point.load_state_dict(torch.load(PREV25.RUN/'models'/f'vector_local_edge_seed{seed}'/'final.pt',weights_only=True,map_location='cpu')['state_dict']);self.correction=DirectionUtility27(seed).cuda();self.correction.load_state_dict(torch.load(RUN/'models'/f'{arm}_seed{seed}'/'final.pt',weights_only=True,map_location='cpu')['state_dict']);self.eval()
    def forward(self,O):
        p=self.parent.parent({k:v for k,v in O.items() if k!='edge'});r=dict(p['rep'],lines=p['lines'],line_weights=p['line_weights']);r['proposal_delta']=self.parent.evaluator(O,r,'line_vector')['points']-r['base'];point_delta=self.point(O,r,'vector_local_edge')['points']-r['base'];return self.correction(O,r,point_delta,self.arm)
