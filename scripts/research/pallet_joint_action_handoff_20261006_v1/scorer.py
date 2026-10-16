"""Dynamic-M adapter retaining every N3 parameter and its radial forward API."""
import torch
from .a_common import DimensionConditionedPointRefiner,finite,sample,local_phase

class JointActionScorer(DimensionConditionedPointRefiner):
    def forward_bank(self,p3,p4,points,boxes,point_valid,input_shape,*,context,candidate_points,action_valid):
        dtype=self.role_embedding.weight.dtype
        points,boxes,input_shape=points.to(dtype),boxes.to(dtype),input_shape.to(dtype)
        q=candidate_points.to(dtype);b,m=q.shape[:2]
        assert q.shape==(b,m,9,2) and action_valid.shape==(b,m)
        assert torch.all((q[:,0]==points)|(torch.isnan(q[:,0])&torch.isnan(points))) and action_valid[:,0].all()
        if m==1:
            supported=torch.zeros((b,8),device=points.device,dtype=torch.bool)
            return dict(logits=points.new_zeros((b,8,1)),points_raw=points,point_valid=finite(points,point_valid),point_support=supported,candidate_points=q,action_valid=action_valid,box_diagonal=(boxes[:,2:]-boxes[:,:2]).norm(dim=-1).clamp_min(1),boxes=boxes,coverage=points.new_zeros((b,8,0)))
        valid=finite(points,point_valid);box_valid=torch.isfinite(boxes).all(-1)&(boxes[:,2:]>boxes[:,:2]).all(-1)
        safe=torch.where(valid[...,None],points,torch.zeros_like(points));bb=torch.where(box_valid[:,None],boxes,boxes.new_tensor([0,0,1,1]))
        size=bb[:,2:]-bb[:,:2];diag=size.norm(dim=-1).clamp_min(1);center=(bb[:,2:]+bb[:,:2])*.5
        safe_q=torch.where(valid[:,None,:,None],q,safe[:,None])
        locations=safe_q[:,1:,:8].transpose(1,2)
        positions=locations[:,:,:,None]+diag[:,None,None,None,None]*self.stencil_fraction*self.stencil[None,None,None]
        inside=valid[:,:8,None,None]&(positions[...,0]>=0)&(positions[...,1]>=0)&(positions[...,0]<input_shape[:,None,None,None,1])&(positions[...,1]<input_shape[:,None,None,None,0])
        evidence=torch.cat([sample(self.adapt3(p3.to(dtype)),positions,input_shape,8),sample(self.adapt4(p4.to(dtype)),positions,input_shape,16)],-2)*inside[...,None,:]
        _,k,c,channels,_=evidence.shape
        patch=self.patch_body(evidence.reshape(b*k*c,channels,4,8));pooled=torch.cat([patch.mean((-2,-1)),patch.amax((-2,-1))],-1).reshape(b,k,c,-1)
        own=(safe[:,:8]-center[:,None])/diag[:,None,None]
        geom=torch.cat([own,(size/diag[:,None])[:,None].expand(-1,8,-1),(size[:,0]/size[:,1].clamp_min(1e-6)).log()[:,None,None].expand(-1,8,1)],-1)
        raw_context=torch.cat([geom,self.role_embedding.weight[None].expand(b,-1,-1)],-1)
        displacement=(safe_q[:,:, :8]-safe[:,None,:8]).transpose(1,2)/diag[:,None,None,None]/.08
        logits=self.scorer(torch.cat([pooled,raw_context[:,:,None].expand(-1,-1,c,-1),displacement[:,:,1:],inside.to(dtype).mean(-1,keepdim=True)],-1)).squeeze(-1)
        motion_valid=action_valid[:,1:].to(dtype)
        null_pool=(pooled*motion_valid[:,None,:,None]).sum(-2)/motion_valid.sum(-1).clamp_min(1)[:,None,None]
        null=self.null_scorer(torch.cat([null_pool,raw_context],-1));logits=torch.cat([null,logits],-1)
        embedding=self.metadata_encoder(context.to(self.role_embedding.weight))
        role=self.role_embedding.weight[None,:,None].expand(b,-1,m,-1);null_flag=torch.zeros((b,8,m,1),device=points.device,dtype=dtype);null_flag[:,:,0]=1
        meta=torch.cat([embedding[:,None,None].expand(-1,8,m,-1),role,displacement,null_flag],-1)
        logits=logits+self.metadata_scorer(meta).squeeze(-1)
        logits=logits.masked_fill(~action_valid[:,None],float('-inf'))
        support=valid[:,:8]&box_valid[:,None]&(inside.any(-1)&action_valid[:,None,1:]).any(-1)
        return dict(logits=logits,points_raw=points,point_valid=valid,point_support=support,candidate_points=q,action_valid=action_valid,box_diagonal=diag,coverage=inside.to(dtype).mean(-1),boxes=boxes)

def action_scores(output):
    mask=output['point_support'];count=mask.sum(-1)
    # Avoid -inf times zero at masked corners.
    scores=torch.where(mask[:,:,None],output['logits'],torch.zeros_like(output['logits'])).sum(1)/count.clamp_min(1)[:,None]
    scores=scores.masked_fill(~output['action_valid'],float('-inf'))
    scores=torch.where((count>0)[:,None],scores,torch.cat([scores.new_zeros((len(scores),1)),scores.new_full((len(scores),scores.shape[1]-1),float('-inf'))],-1))
    return scores

def decode_bank(output,readout='J'):
    q=output['candidate_points'];points=output['points_raw'];b=len(q)
    if readout=='J':
        index=action_scores(output).argmax(-1);selected=q[torch.arange(b,device=q.device),index]
    elif readout=='I':
        index=output['logits'].argmax(-1);row=torch.arange(b,device=q.device)[:,None];corner=torch.arange(8,device=q.device)[None]
        selected=torch.cat([q[row,index,corner],points[:,8:]],1)
    else:raise ValueError('Readout must be I or J')
    supported=output['point_support']
    if readout=='I':selected=torch.cat([torch.where(supported[...,None],selected[:,:8],points[:,:8]),points[:,8:]],1)
    else:selected=torch.cat([selected[:,:8],points[:,8:]],1)
    return selected,index

def joint_loss(output,batch):
    gt,valid,branch,_=local_phase(output['points_raw'],output['point_valid'],batch['gt_points'],batch['gt_valid'],batch['permutations'],batch['group_valid'],output['box_diagonal'])
    mask=finite(gt,valid)[:,:8]&output['point_support'];count=mask.sum(-1)
    error=(output['candidate_points'][:,:,:8]-gt[:,None,:8]).square().sum(-1)
    error=torch.where(mask[:,None],error,torch.zeros_like(error)).sum(-1)/count.clamp_min(1)[:,None]
    sigma=output['box_diagonal']*.08/17
    target=(-error/(2*sigma[:,None].square())).masked_fill(~output['action_valid'],float('-inf')).softmax(-1)
    logp=action_scores(output).log_softmax(-1)
    ce=-(target*torch.where(output['action_valid'],logp,torch.zeros_like(logp))).sum(-1)
    return torch.where(count>0,ce,torch.zeros_like(ce)).sum()/(count>0).sum().clamp_min(1),dict(excluded_frames=int((count==0).sum()),branch=branch)
