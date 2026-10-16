import torch
from torch import nn
import torch.nn.functional as F

PERMS=[[0,1,2,3,4,5,6,7,8],[1,5,6,2,0,4,7,3,8],
       [5,4,7,6,1,0,3,2,8],[4,0,3,7,5,1,2,6,8]]
EDGES=[[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]

class SelectorResidual(nn.Module):
    def __init__(self,arm,seed):
        super().__init__();torch.manual_seed(seed);self.arm=arm
        self.register_buffer('perms',torch.tensor(PERMS));self.register_buffer('edges',torch.tensor(EDGES))
        # Selector and residual share inputs, not trainable parameters.
        self.risk_point=nn.Sequential(nn.LayerNorm(45),nn.Linear(45,24),nn.SiLU())
        self.risk_roi=nn.Sequential(nn.LayerNorm(45),nn.Linear(45,24),nn.SiLU())
        self.risk=nn.Sequential(nn.Linear(9*27+24+3+12+6+1,128),nn.SiLU(),nn.Linear(128,64),nn.SiLU(),nn.Linear(64,1))
        self.res_point=nn.Sequential(nn.LayerNorm(45),nn.Linear(45,24),nn.SiLU())
        self.res_roi=nn.Sequential(nn.LayerNorm(45),nn.Linear(45,24),nn.SiLU())
        self.residual=nn.Sequential(nn.Linear(9*27+24+3,128),nn.SiLU(),nn.Linear(128,64),nn.SiLU(),nn.Linear(64,18))
        for layer in [self.risk[-1],self.residual[-1]]:nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)

    def forward(self,z,mode='normal'):
        B=len(z['stock']);P=self.perms
        candidates=z['stock'][:,P]
        coords=z['point_coord'][:,P].clamp(-4,4);vis=z['visibility'][:,P,None]
        dims=torch.log(z['dims']/z['dims'].new_tensor([1.,1.,.15]))
        if self.arm=='D_select_residual_no_dims':dims=torch.zeros_like(dims)
        dims=dims[:,None].expand(-1,4,-1)
        extent=(z['stock'][:,:8].amax(1)-z['stock'][:,:8].amin(1)).clamp_min(1)
        scale=extent.norm(dim=-1).clamp_min(32)
        rp=self.risk_point(z['point_feat'])[:,P]
        rp=torch.cat([rp,coords,vis],-1).reshape(B,4,-1)
        roi=self.risk_roi(z['roi_feat']).mean(1)[:,None].expand(-1,4,-1)
        lengths=(coords[:,:,self.edges[:,0]]-coords[:,:,self.edges[:,1]]).norm(dim=-1)
        faces=[[0,1,2,3],[4,5,6,7],[0,1,5,4],[3,2,6,7],[0,4,7,3],[1,5,6,2]]
        areas=[]
        for face in faces:
            q=coords[:,:,face];areas.append((q[:,:,:,0]*q.roll(-1,2)[:,:,:,1]-q[:,:,:,1]*q.roll(-1,2)[:,:,:,0]).sum(-1).abs()/2)
        aspect=(extent[:,0]/extent[:,1]).log()[:,None,None].expand(-1,4,-1)
        risk=self.risk(torch.cat([rp,roi,dims,lengths,torch.stack(areas,-1),aspect],-1)).squeeze(-1)
        fp=self.res_point(z['point_feat'])[:,P]
        fp=torch.cat([fp,coords,vis],-1).reshape(B,4,-1)
        fr=self.res_roi(z['roi_feat']).mean(1)[:,None].expand(-1,4,-1)
        delta=16*torch.tanh(self.residual(torch.cat([fp,fr,dims],-1)).reshape(B,4,9,2))
        penalty=risk.new_tensor([0,2,2,2])[None]
        choice=(risk*scale[:,None]+penalty).argmin(-1)
        if self.arm=='A_direct' or mode=='no_selector':choice=torch.zeros_like(choice)
        if self.arm=='B_select_only' or mode=='no_residual':delta=delta*0
        allpoints=candidates+delta
        chosen=allpoints[torch.arange(B,device=risk.device),choice]
        return dict(points=chosen,choice=choice,risk=risk,scale=scale,candidates=candidates,allpoints=allpoints,delta=delta,
                    visibility=z['visibility'][:,P][torch.arange(B,device=risk.device),choice])

def objective(out,gt,arm):
    candidate_error=(out['candidates'][:,:,:8]-gt[:,None]).norm(dim=-1).mean(-1)
    target=candidate_error/out['scale'][:,None]
    selector=F.smooth_l1_loss(out['risk'],target,beta=.05)
    best=candidate_error.argmin(-1)
    if arm=='A_direct':best=torch.zeros_like(best)
    chosen=out['allpoints'][torch.arange(len(gt),device=gt.device),best,:8]
    residual=F.smooth_l1_loss(chosen,gt,beta=4)/16
    if arm=='A_direct':selector=selector*0
    if arm=='B_select_only':residual=residual*0
    return selector+residual,selector,residual
