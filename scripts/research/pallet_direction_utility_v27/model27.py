"""Learn candidate-specific utility and amount, with optional incident-role supervision."""
import torch
from torch import nn
import torch.nn.functional as F
EDGES=((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))
PAIRS=((0,1),(0,2),(1,2))

def bounded(v,radius):
    return v*(radius/v.norm(dim=-1).clamp_min(1e-8)).clamp_max(1)[...,None]

def candidates(O,r,point_delta,line=True):
    b=r['base'];B,P,_=b.shape;diag=(O['box'][:,2:]-O['box'][:,:2]).clamp_min(1).norm(dim=-1);radius=.15*diag[:,None];d=bounded(point_delta,radius)
    dx=d* d.new_tensor([1,0]);dy=d*d.new_tensor([0,1]);zero=torch.zeros_like(d)
    idx=r['line_weights'][...,:-1].argsort(dim=-1,descending=True,stable=True)[...,:3]
    ll=r['lines'][:,None].expand(-1,P,-1,-1).gather(2,idx[...,None].expand(-1,-1,-1,3));norm=ll[...,:2].norm(dim=-1);ll=ll/norm.clamp_min(1e-8)[...,None]
    sign=torch.where((ll[...,0]<-1e-7)|((ll[...,0].abs()<=1e-7)&(ll[...,1]<0)),-1.,1.);ll=ll*sign[...,None];n=ll[...,:2];signed=(n*b[:,:,None]).sum(-1)+ll[...,2];normal=-signed[...,None]*n
    pairs=[];oks=[];angles=[]
    for a,z in PAIRS:
        n1,n2=n[:,:,a],n[:,:,z];det=n1[...,0]*n2[...,1]-n1[...,1]*n2[...,0];ok=(det.abs()>=.1)&(norm[:,:,a]>1e-8)&(norm[:,:,z]>1e-8);safe=torch.where(ok,det,torch.ones_like(det));s1,s2=-signed[:,:,a],-signed[:,:,z]
        v=torch.stack((s1*n2[...,1]-n1[...,1]*s2,n1[...,0]*s2-s1*n2[...,0]),-1)/safe[...,None];pairs.append(v*ok[...,None]);oks.append(ok);angles.append(det.abs())
    if line:
        raw=torch.cat((torch.stack((zero,d,dx,dy),2),normal,torch.stack(pairs,2)),2);mask=torch.cat((torch.ones((B,P,4),device=b.device,dtype=torch.bool),norm>1e-8,torch.stack(oks,2)),2)
    else:
        # Equal candidate count; no DHT-derived geometry in point control.
        raw=torch.stack((zero,d,dx,dy,.5*d,1.5*d,-d,-dx,-dy,dx-dy),2);mask=torch.ones((B,P,10),device=b.device,dtype=torch.bool)
    delta=bounded(raw,radius[...,None]);assoc=r['line_weights'][...,:-1].gather(-1,idx)
    return dict(delta=delta,mask=mask,lines=ll,normals=n,association=assoc,null=r['line_weights'][...,-1],diag=diag,angles=torch.stack(angles,2))

class DirectionUtility27(nn.Module):
    def __init__(self,seed):
        super().__init__();torch.manual_seed(27000+seed)
        self.project=nn.Conv2d(192,16,1);self.identity=nn.Embedding(8,8);self.candidate_id=nn.Embedding(10,8)
        self.point_encode=nn.Sequential(nn.Linear(9*19+64+8+6,96),nn.SiLU(),nn.Linear(96,64),nn.SiLU())
        self.line_encode=nn.Sequential(nn.Linear(12*22,64),nn.SiLU(),nn.Linear(64,32),nn.SiLU())
        self.head=nn.Sequential(nn.Linear(64+32+19+8+8,96),nn.SiLU(),nn.Linear(96,3))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)

    def forward(self,O,r,point_delta,arm='line_role_utility',line_off=False):
        isline=arm!='point_utility';c=candidates(O,r,point_delta,isline);base=r['base'];B,P,_=base.shape;diag=c['diag'];bb=O['box'];wh=(bb[:,2:]-bb[:,:2]).clamp_min(1);hw=O['image_hw'].flip(-1)
        features=torch.cat((self.project(O['features'])*O['content'],F.interpolate(O['edge'],size=O['features'].shape[-2:],mode='bilinear',align_corners=True)),1)
        def sample(pos):
            grid=2*(pos-bb[:,None,None,:2])/wh[:,None,None]-1;valid=((pos>=0)&(pos<hw[:,None,None])).all(-1)&(grid.abs()<=1).all(-1)
            z=F.grid_sample(features,grid,align_corners=True,padding_mode='zeros').permute(0,2,3,1)*valid[...,None];return z,valid
        yy,xx=torch.meshgrid(base.new_tensor([-1,0,1]),base.new_tensor([-1,0,1]),indexing='ij');off=torch.stack((xx.flatten(),yy.flatten()),-1);patch,_=sample(base[:,:,None]+.03*diag[:,None,None,None]*off)
        ident=self.identity(torch.arange(P,device=base.device))[None].expand(B,-1,-1);q=self.point_encode(torch.cat((patch.flatten(-2),r['context'],ident,2*(base-bb[:,None,:2])/wh[:,None]-1,base/hw[:,None],point_delta/diag[:,None,None]),-1))
        line_token=base.new_zeros((B,P,3,32));coverage=base.new_zeros((B,P,3))
        if isline and not line_off:
            n=c['normals'];t=torch.stack((-n[...,1],n[...,0]),-1);foot=base[:,:,None]-((base[:,:,None]*n).sum(-1)+c['lines'][...,2])[...,None]*n
            u=torch.linspace(-.5,.5,12,device=base.device);pos=foot[...,None,:]+u[None,None,None,:,None]*diag[:,None,None,None,None]*t[...,None,:];z,v=sample(pos.flatten(2,3));z=z.reshape(B,P,3,12,19);v=v.reshape(B,P,3,12);desc=torch.cat((z,n[...,None,:].expand(-1,-1,-1,12,-1),v[...,None].float()),-1);line_token=self.line_encode(desc.flatten(-2));coverage=v.float().mean(-1)
        tok=[base.new_zeros((B,P,32)) for _ in range(4)]+list(line_token.unbind(2))+[(line_token[:,:,a]+line_token[:,:,z])/2 for a,z in PAIRS];tok=torch.stack(tok,2)
        destination=base[:,:,None]+c['delta'];dest,inside=sample(destination);d=c['delta']/diag[:,None,None,None];length=d.norm(dim=-1,keepdim=True);assoc=base.new_zeros((B,P,10));cov=torch.zeros_like(assoc);angle=torch.zeros_like(assoc)
        if isline and not line_off:
            assoc[:,:,4:7]=c['association'];cov[:,:,4:7]=coverage
            for k,(a,z) in enumerate(PAIRS):assoc[:,:,7+k]=torch.sqrt(c['association'][:,:,a]*c['association'][:,:,z]);cov[:,:,7+k]=torch.minimum(coverage[:,:,a],coverage[:,:,z]);angle[:,:,7+k]=c['angles'][:,:,k]
        geom=torch.cat((d,length,inside[...,None].float(),assoc[...,None],cov[...,None],angle[...,None],c['null'][:,:,None,None].expand(-1,-1,10,-1) if isline and not line_off else base.new_zeros(B,P,10,1)),-1)
        cid=self.candidate_id(torch.arange(10,device=base.device))[None,None].expand(B,P,-1,-1);raw=self.head(torch.cat((q[:,:,None].expand(-1,-1,10,-1),tok,dest,cid,geom),-1));utility=raw[...,0];alpha=raw[...,1].sigmoid();role=raw[...,2]
        logits=utility/.02;logits=logits.clone();logits[:,:,0]=0
        if arm=='line_role_utility':logits[:,:,4:]+=F.logsigmoid(role[:,:,4:])
        mask=c['mask'].clone()
        if line_off and isline:mask[:,:,4:]=False
        weights=logits.masked_fill(~mask,-1e4).softmax(-1);delta=(weights[...,None]*alpha[...,None]*c['delta']).sum(2)
        return dict(points=base+delta,base=base,delta=delta,utility=utility,alpha=alpha,role=role,weights=weights,candidates=c['delta'],candidate_mask=mask,lines=c['lines'],diag=diag)

def role_targets(lines,gt,valid,diag):
    targets=[];supported=[]
    for p in range(8):
        edges=[(a,b) for a,b in EDGES if p in (a,b)];values=[];masks=[]
        for a,b in edges:
            d1=(lines[:,p,:,:2]*gt[:,a,None]).sum(-1)+lines[:,p,:,2];d2=(lines[:,p,:,:2]*gt[:,b,None]).sum(-1)+lines[:,p,:,2];rms=torch.sqrt((d1.square()+d2.square())/2);ok=valid[:,a]&valid[:,b]&((gt[:,a]-gt[:,b]).norm(dim=-1)>1);values.append(rms.masked_fill(~ok[:,None],1e6));masks.append(ok)
        best=torch.stack(values,-1).amin(-1);targets.append(torch.exp(-best/(.02*diag[:,None]).clamp_min(1)));supported.append(torch.stack(masks,-1).any(-1))
    t=torch.stack(targets,1);support=torch.stack(supported,1);pair=torch.stack([torch.minimum(t[:,:,a],t[:,:,b]) for a,b in PAIRS],2)
    return torch.cat((t,pair),2),support

def objective(out,gt,valid,arm):
    residual=gt-out['base'];d=out['candidates'];length=d.norm(dim=-1);target_alpha=((residual[:,:,None]*d).sum(-1)/d.square().sum(-1).clamp_min(1e-9)).clamp(0,1).detach();before=residual.norm(dim=-1)
    after=(out['base'][:,:,None]+out['alpha'].detach()[...,None]*d-gt[:,:,None]).norm(dim=-1);target_utility=((before[:,:,None]-after)/out['diag'][:,None,None]).detach();mask=valid[:,:,None]&out['candidate_mask'];mask=mask.clone();mask[:,:,0]=False
    point=(out['points']-gt).norm(dim=-1)[valid].mean();utility=F.smooth_l1_loss(out['utility'][mask],target_utility[mask],beta=.01);amount=((out['alpha']-target_alpha).abs()*length)[mask].mean();role=point.new_zeros(())
    if arm=='line_role_utility':
        target,support=role_targets(out['lines'],gt,valid,out['diag']);rm=mask[:,:,4:]&support[:,:,None];role=F.binary_cross_entropy_with_logits(out['role'][:,:,4:][rm],target[rm]) if rm.any() else role
    loss=point if arm=='line_distance' else point+10*utility+.25*amount+.25*role
    return loss,dict(point=point,utility=utility,amount=amount,role=role)
