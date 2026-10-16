"""Separate frozen point choice from incremental directional gain and risk."""
import torch
from torch import nn
import torch.nn.functional as F
from backbone31 import FeatureBackbone,ANCHORS

class PointSelector(nn.Module):
    def __init__(self,seed):
        super().__init__();self.core=FeatureBackbone(seed)
        self.relative=nn.Sequential(nn.Linear(128,64),nn.SiLU(),nn.Linear(64,1))
        nn.init.zeros_(self.relative[-1].weight);nn.init.zeros_(self.relative[-1].bias)
    def forward(self,O,S):
        out=self.core(O,S,'point_gain');emb=out['point_embedding']
        fraction=self.relative(torch.cat((emb[:,:,0],emb[:,:,1]),-1)).squeeze(-1).tanh()
        distance=(S['strong']-S['base']).norm(dim=-1)
        gain=fraction*distance;anchor_id=(gain>0).long()
        points=torch.where(anchor_id[...,None]==1,S['strong'],S['base'])
        return dict(points=points,anchor_id=anchor_id,parent_gain=gain,parent_fraction=fraction,parent_distance=distance)

def selector_loss(out,S,gt,valid):
    eb=(S['base']-gt).norm(dim=-1);es=(S['strong']-gt).norm(dim=-1)
    target=((eb-es)/out['parent_distance'].clamp_min(1e-6)).clamp(-1,1)
    mask=valid&(out['parent_distance']>.1)
    loss=F.smooth_l1_loss(out['parent_fraction'][mask],target[mask],beta=.2)
    return loss,dict(point_error=(out['points']-gt).norm(dim=-1)[valid].mean())

class SeparatedGain(nn.Module):
    def __init__(self,seed,selector):
        super().__init__();self.selector=selector.eval().requires_grad_(False)
        self.core=FeatureBackbone(seed)
        self.numeric=nn.Sequential(nn.Linear(110,64),nn.SiLU(),nn.Linear(64,2))
        nn.init.zeros_(self.numeric[-1].weight);nn.init.zeros_(self.numeric[-1].bias)
    def forward(self,O,S,arm='split',line_off=False):
        with torch.no_grad():parent=self.selector(O,S)
        out=self.core(O,S,'generic' if arm=='generic' else 'normal_gain')
        raw=self.numeric(out['embedding']);mean=raw[...,0].tanh()
        lower=mean-(mean+1)*raw[...,1].sigmoid() # ordered bounded lower quantile
        move=(out['candidate_points']-out['anchors']).norm(dim=-1).detach()
        allowed=(torch.tensor(ANCHORS,device=move.device)[None,None]==parent['anchor_id'][...,None])
        allowed[:,:,:8]=False
        gain=mean*move;low=lower*move
        eligible=allowed&(low>0)&(move>.1)
        score=out['gain'].masked_fill(~eligible,-1e9);index=score.argmax(-1)
        apply=eligible.any(-1)&(score.max(-1).values>0)
        if line_off:apply=torch.zeros_like(apply)
        proposal=out['candidate_points'].gather(2,index[...,None,None].expand(-1,-1,1,2))[:,:,0]
        points=torch.where(apply[...,None],proposal,parent['points'])
        return dict(points=points,parent_points=parent['points'],parent_id=parent['anchor_id'],parent_gain=parent['parent_gain'],parent_fraction=parent['parent_fraction'],chosen=index,applied=apply,alpha=out['alpha'],rank=out['gain'],gain=gain,lower=low,mean_fraction=mean,lower_fraction=lower,move=move,allowed=allowed,candidate_points=out['candidate_points'],anchors=out['anchors'],directions=out['directions'],diag=out['diag'])

def correction_loss(out,gt,valid):
    d=out['directions'];length=d.norm(dim=-1)
    alpha=(((gt[:,:,None]-out['anchors'])*d).sum(-1)/d.square().sum(-1).clamp_min(1e-9)).clamp(-1,1).detach()
    anchors=out['anchors'];end=anchors+out['alpha'].detach()[...,None]*d
    target_gain=((anchors-gt[:,:,None]).norm(dim=-1)-(end-gt[:,:,None]).norm(dim=-1)).detach()
    target_fraction=(target_gain/out['move'].clamp_min(1e-6)).clamp(-1,1)
    mask=valid[:,:,None].expand_as(length).clone();mask[:,:,:8]=False
    moving=mask&(out['move']>.1)
    amount=(((out['alpha']-alpha).abs()*length/out['diag'][:,None,None])[mask]).mean()
    numeric=F.smooth_l1_loss(out['mean_fraction'][moving],target_fraction[moving],beta=.2)
    diff=target_fraction-out['lower_fraction'];quantile=torch.maximum(.1*diff,-.9*diff)[moving].mean()
    target=target_gain.masked_fill(~out['allowed'],-1e9)
    # Extra fixed zero candidate means retain the selected point.
    target=torch.cat((torch.zeros_like(target[:,:,:1]),target),-1).argmax(-1)
    scores=torch.cat((torch.zeros_like(out['rank'][:,:,:1]),out['rank'].masked_fill(~out['allowed'],-1e9)), -1)
    rank=F.cross_entropy((scores/.02)[valid],target[valid])
    loss=10*amount+numeric+quantile+.05*rank
    return loss,dict(amount=amount,numeric=numeric,quantile=quantile,rank=rank,point_error=(out['points']-gt).norm(dim=-1)[valid].mean())
