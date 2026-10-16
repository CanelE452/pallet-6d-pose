"""A shared residual model with nested dimension/role/relation interventions."""
import torch
from torch import nn
import torch.nn.functional as F

class Residual(nn.Module):
    def __init__(self,arm,seed):
        super().__init__();torch.manual_seed(seed);self.arm=arm
        self.point=nn.Sequential(nn.LayerNorm(45),nn.Linear(45,64))
        self.coord=nn.Linear(3,64);self.image=nn.Sequential(nn.LayerNorm(45),nn.Linear(45,64))
        self.image_coord=nn.Linear(2,64);self.dim=nn.Sequential(nn.Linear(3,64),nn.SiLU(),nn.Linear(64,64))
        self.role=nn.Sequential(nn.Linear(3,64),nn.SiLU(),nn.Linear(64,64))
        self.relation=nn.Sequential(nn.Linear(5,32),nn.SiLU(),nn.Linear(32,1))
        self.norm1=nn.LayerNorm(64);self.norm2=nn.LayerNorm(64)
        self.q=nn.Linear(64,64);self.k=nn.Linear(64,64);self.v=nn.Linear(64,64)
        self.iq=nn.Linear(64,64);self.ik=nn.Linear(64,64);self.iv=nn.Linear(64,64)
        self.ff=nn.Sequential(nn.Linear(64,128),nn.SiLU(),nn.Linear(128,64))
        self.gate=nn.Linear(64,1);self.output=nn.Linear(64,2)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
        signs=torch.tensor([[-1,1,1],[1,1,1],[1,-1,1],[-1,-1,1],[-1,1,-1],[1,1,-1],[1,-1,-1],[-1,-1,-1],[0,0,0]],dtype=torch.float32)
        self.register_buffer('signs',signs)
        adjacent=((signs[:,None]!=signs[None]).sum(-1)==1).float();adjacent[8]=0;adjacent[:,8]=0
        self.register_buffer('adjacent',adjacent)

    def forward(self,z):
        # The signature accepts ONLY observation keys; targets cannot enter this path.
        B=len(z['stock']);dims=z['dims'];d=torch.log(dims/dims.new_tensor([1.,1.,.15]))
        if self.arm=='A_points':d=torch.zeros_like(d)
        base=self.point(z['point_feat'])+self.coord(torch.cat([z['point_coord'],z['visibility'][:,:,None]],-1))
        base=base+self.dim(d)[:,None]
        extents=torch.stack([dims[:,[0,2,1]],dims[:,[1,2,0]]],1)
        geometry=extents[:,:,None,:]*self.signs[None,None]/2
        use_role=self.arm in ['C_geometry','D_no_relations']
        if not use_role:geometry=torch.zeros_like(geometry)
        node=base[:,None]+self.role(geometry)
        node=node.reshape(B*2,9,64);g=geometry.reshape(B*2,9,3)
        delta=g[:,:,None]-g[:,None,:]
        relation=torch.cat([delta,delta.norm(dim=-1,keepdim=True),self.adjacent[None,:,:,None].expand(B*2,-1,-1,-1)],-1)
        if self.arm!='C_geometry':relation=torch.zeros_like(relation)
        bias=self.relation(relation).squeeze(-1)
        n=self.norm1(node);a=((self.q(n)@self.k(n).transpose(-1,-2))/8+bias).softmax(-1)
        node=node+a@self.v(n)
        image=self.image(z['roi_feat'])+self.image_coord(z['roi_coord'])
        image=image[:,None].expand(-1,2,-1,-1).reshape(B*2,49,64)
        a=(self.iq(self.norm2(node))@self.ik(image).transpose(-1,-2)/8).softmax(-1)
        node=node+a@self.iv(image);node=node+self.ff(node)
        node=node.reshape(B,2,9,64)
        weight=self.gate(node.mean(2)).softmax(1)
        delta=(64*torch.tanh(self.output(node))*weight[:,:,None,:]).sum(1)
        return z['stock']+delta
