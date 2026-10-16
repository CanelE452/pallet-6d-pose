from context36 import *
from model33 import Correspondence
from torch import nn
import torch.nn.functional as F

class Frozen(Correspondence):
    def __init__(self,seed):
        super().__init__(seed)
        self.load_state_dict(torch.load(V34.RUN/'dense/models'/f'line_only_r0_seed{seed}'/'final.pt',weights_only=True,map_location='cpu')['state_dict'])
        self.requires_grad_(False);self.eval()
        k=torch.arange(1152);dt=(k[:,None]//48-k[None]//48).abs();dr=(k[:,None]%48-k[None]%48).abs();flip=(k[:,None]%48+k[None]%48-47).abs();self.register_buffer('suppress',((dt<=1)&(dr<=1))|((dt>=23)&(flip<=1)))
    def evidence(self,z,maps,bb,hw,blind=False):
        e=super().evidence(z,maps,bb,hw,blind);H=e['hough'].flatten(-2);work=H.clone();indices=[]
        for _ in range(CANDIDATES):
            idx=work.argmax(-1);indices.append(idx);work=work.masked_fill(self.suppress[idx],-1e9)
        index=torch.stack(indices,-1).flatten(1);score=H.gather(-1,torch.stack(indices,-1)).flatten(1);role=torch.arange(12,device=z.device).repeat_interleave(CANDIDATES);theta=(index//48)*math.pi/24;rho=(index%48)*2*math.sqrt(2)/47-math.sqrt(2);wh=(bb[:,2:]-bb[:,:2]).clamp_min(1);half=wh/2;diag=wh.norm(dim=-1);center=(bb[:,:2]+bb[:,2:])/2
        normal=torch.stack((theta.cos(),theta.sin()),-1)/half[:,None];length=normal.norm(dim=-1);normal=normal/length[...,None];offset=rho/length/diag[:,None];xy=self.xy24[None]*half[:,None]/diag[:,None,None];pixel=xy*diag[:,None,None]+center[:,None];observed=((pixel>=0)&(pixel<hw.flip(-1)[:,None])).all(-1);weight=F.interpolate(maps[:,:12],size=(24,24),mode='bilinear',align_corners=True).flatten(-2)[:,role];distance=torch.einsum('bnd,bkd->bnk',normal,xy)-offset[...,None];ridge=torch.exp(-.5*(distance/.025).square());support=weight*ridge*observed[:,None];mass=support.sum(-1);weights=support/mass[...,None].clamp_min(1e-6);loc=torch.einsum('bnk,bkd->bnd',weights,xy);tangent=torch.stack((-normal[...,1],normal[...,0]),-1);along=torch.einsum('bnkd,bnd->bnk',xy[:,None]-loc[:,:,None],tangent);spread=(weights*along.square()).sum(-1).clamp_min(1e-8).sqrt();visual=torch.einsum('bnk,bck->bnc',weights,F.interpolate(z,size=(24,24),mode='bilinear',align_corners=True).flatten(-2));geom=torch.cat((normal,offset[...,None],loc,spread[...,None],score[...,None],(mass/576)[...,None]),-1)
        self.proposal=dict(features=torch.cat((visual,self.role_embed(role)[None].expand(len(z),-1,-1),geom),-1),theta=theta,rho=rho,index=index);return e
    def capture(self,O,S):
        values={};a=self.head.register_forward_pre_hook(lambda m,args:values.update(context=args[0]));b=self.query_encode.register_forward_hook(lambda m,args,out:values.update(query=out))
        try:out=self(O,S,'line_only')
        finally:a.remove();b.remove()
        bb=O['box'];wh=(bb[:,2:]-bb[:,:2]).clamp_min(1);diag=wh.norm(dim=-1);center=(bb[:,:2]+bb[:,2:])/2;anchors=torch.stack((S['base'],S['strong']),2).flatten(1,2);values.update(self.proposal);values.update(box=bb,anchor=(anchors-center[:,None])/diag[:,None,None],base=S['base'],strong=S['strong'],v34=out['points']);return values

class Model(nn.Module):
    def __init__(self,seed):
        super().__init__();self.core=Frozen(seed);self.head=copy.deepcopy(self.core.head).requires_grad_(True);torch.manual_seed(36000+seed);self.refiner=nn.Sequential(nn.Linear(52,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,3));nn.init.zeros_(self.refiner[-1].weight);nn.init.zeros_(self.refiner[-1].bias)
    def forward(self,x,arm,intervention=None,donor=None):
        bb=x['box'];wh=(bb[:,2:]-bb[:,:2]).clamp_min(1);diag=wh.norm(dim=-1);half=wh/2;aspect=(wh[:,0]/wh[:,1]).log();input=torch.cat((x['features'],x['theta'].cos()[...,None],x['theta'].sin()[...,None],x['rho'][...,None],aspect[:,None,None].expand(-1,96,1)),-1);raw=self.refiner(input);residual=raw[...,:2].tanh();theta=x['theta']+(residual[...,0]*math.pi/24 if arm in ('refine','both') else 0);rho=x['rho']+(residual[...,1]*2*math.sqrt(2)/47 if arm in ('refine','both') else 0)
        normal=torch.stack((theta.cos(),theta.sin()),-1)/half[:,None];length=normal.norm(dim=-1);normal=normal/length[...,None];offset=rho/length/diag[:,None];prior=x['features'][...,46].clamp(1e-5,1-1e-5);logit=torch.logit(prior)+raw[...,2];score=logit.sigmoid() if arm in ('rank','both') else prior
        # Exact same baseline feature at zero refinement; geometry-dependent center correction.
        loc=x['features'][...,43:45];oldn=x['features'][...,40:42];oldoffset=x['features'][...,42];oldres=(oldn*loc).sum(-1)-oldoffset;newloc=loc+(offset+oldres-(normal*loc).sum(-1))[...,None]*normal
        feature=x['features'].clone();feature[...,40:42]=normal;feature[...,42]=offset;feature[...,43:45]=newloc;feature[...,46]=score
        # Physical line learning is isolated from the point objective.
        feature=feature.detach();pointnormal=normal.detach();pointoffset=offset.detach();pointloc=newloc.detach();pointscore=score.detach()
        if intervention=='geometry_donor':
            feature=feature.clone();feature[...,40:46]=donor['feature'][...,40:46];pointnormal=donor['normal'];pointoffset=donor['offset'];pointloc=donor['loc']
        role=torch.arange(12,device=bb.device).repeat_interleave(8);e=dict(line=self.core.line_token(feature),line_role=role,line_score=pointscore,normal=pointnormal,offset=pointoffset,loc=pointloc,spread=feature[...,45]);line,attention=self.core.attend(x['query'],x['anchor'],e,'line',False)
        if intervention=='line_off':line=torch.zeros_like(line)
        context=x['context'].clone();assert context.shape[-1]==1230;context[...,1088:1152]=line.reshape(-1,8,64);context[...,1152:1216]=0;out=self.head(context);mix=out[...,0].tanh();delta=out[...,1:].tanh();delta=delta/delta.norm(dim=-1,keepdim=True).clamp_min(1);points=x['base']+mix[...,None]*(x['strong']-x['base'])+.15*diag[:,None,None]*delta
        return dict(points=points,normal=normal,offset=offset,loc=newloc,feature=feature,score=score,quality_logit=logit,residual=residual,theta=theta,rho=rho,attention=attention,diag=diag)

def loss36(out,x,gt,valid,support,hasline,arm):
    point=((out['points']-gt).norm(dim=-1)/out['diag'][:,None])[valid].mean();geo=point*0;rank=point*0;reachable=torch.zeros_like(out['score'],dtype=torch.bool)
    if support is not None:
        bb=x['box'];half=(bb[:,2:]-bb[:,:2]).clamp_min(1)/2;center=(bb[:,:2]+bb[:,2:])/2;uv=support.repeat_interleave(8,1);exists=hasline.repeat_interleave(8,1);v=uv[:,:,1]-uv[:,:,0];gn=torch.stack((-v[...,1],v[...,0]),-1);gn=gn/gn.norm(dim=-1,keepdim=True).clamp_min(1e-8);gr=(gn*uv[:,:,0]).sum(-1);rnormal=gn*half[:,None];l=rnormal.norm(dim=-1).clamp_min(1e-8);angle=torch.atan2(rnormal[...,1],rnormal[...,0]);targetrho=(gr-(gn*center[:,None]).sum(-1))/l;diff=angle-x['theta'];sign=torch.where(diff.cos()<0,-1.,1.);da=torch.remainder(diff+math.pi/2,math.pi)-math.pi/2;dr=targetrho*sign-x['rho'];target=torch.stack((da/(math.pi/24),dr/(2*math.sqrt(2)/47)),-1);reachable=exists&(target.abs()<=1).all(-1)
        if arm in ('refine','both') and reachable.any():geo=F.smooth_l1_loss(out['residual'][reachable],target[reachable])
        rhop=out['offset']*out['diag'][:,None]+(out['normal']*center[:,None]).sum(-1);distance=((out['normal'][:,:,None]*uv).sum(-1)-rhop[...,None]).abs().mean(-1);quality=torch.exp(-distance.detach()/5)*exists;known=x['supervised'][:,None].expand_as(quality);pos=known&(quality>=.5);neg=known&~pos
        if arm in ('rank','both') and known.any():
            bce=F.binary_cross_entropy_with_logits(out['quality_logit'],quality,reduction='none');rank=(bce[pos].mean() if pos.any() else bce.sum()*0)+(bce[neg].mean() if neg.any() else bce.sum()*0)
    return point+.05*geo+.05*rank,dict(point=point,geometry=geo,rank=rank,reachable=reachable.sum(),point_px=(out['points']-gt).norm(dim=-1)[valid].mean())
