"""Fixed, equally parameterized residual heads over one inherited forward.

NoOp uses the inherited moving-patch aggregate, not a newly sampled RAW patch.
The legacy scorer's arithmetic and prediction-only support are retained here.
"""
from pathlib import Path
import torch
from torch import nn
from scripts.research.pallet_pose_target_6d_20261006_v1.baseline import BASELINE_ROOT
from scripts.research.pallet_joint_action_handoff_20261006_v1.a_common import (
    finite, sample, read, dcp_env)
from scripts.research.pallet_joint_action_handoff_20261006_v1.scorer import (
    JointActionScorer, action_scores, decode_bank)

METHODS = ('LOCAL_CAP', 'JOINT8')
ROOT = Path(__file__).resolve().parents[3]
DOC = ROOT / '_docs/experiments/pallet_quick_joint_scorer_20261006_v1'
OLD_DOC = BASELINE_ROOT / '_docs/experiments/pallet_joint_action_handoff_20261006_v1'
_PHI_STATES = {}


def projected_tokens(phi, descriptors, support, action_valid):
    """Mask before and after phi; a missing corner's mask bit is also zero."""
    assert descriptors.shape[:3] == (*action_valid.shape, 8)
    assert support.shape == (descriptors.shape[0], 8)
    mask = support[:, None, :, None] & action_valid[:, :, None, None]
    safe = torch.where(mask, descriptors, torch.zeros_like(descriptors))
    assert torch.isfinite(safe).all(), 'Supported inference descriptor is nonfinite'
    encoded = phi(safe)
    encoded = torch.where(mask, encoded, torch.zeros_like(encoded))
    return torch.cat([encoded, mask.to(encoded.dtype).expand(*encoded.shape[:-1], 1)], -1)


def residual_score(method, head, tokens, support, action_valid):
    """Ordered joint nonlinearity or independently scored shared corner tokens."""
    assert tokens.shape == (*action_valid.shape, 8, 16)
    if method == 'LOCAL_CAP':
        individual = head(tokens).squeeze(-1)
        individual = torch.where(support[:, None], individual, torch.zeros_like(individual))
        delta = individual.sum(-1) / support.sum(-1).clamp_min(1)[:, None]
    elif method == 'JOINT8':
        delta = head(tokens.flatten(-2)).squeeze(-1)
    else:
        raise ValueError(method)
    usable = action_valid & (support.sum(-1) > 0)[:, None]
    return torch.where(usable, delta, torch.zeros_like(delta))


def descriptors_from_inherited(pooled, null_pool, raw_context, displacement,
                               coverage, embedding, action_valid, support):
    """Use existing tensors only; no labels, candidate indices or extra patches."""
    b, k, motions, _ = pooled.shape
    m = motions + 1
    assert k == 8 and action_valid.shape == (b, m)
    motion_valid = action_valid[:, 1:].to(pooled.dtype)
    null_coverage = (coverage * motion_valid[:, None]).sum(-1, keepdim=True) / \
        motion_valid.sum(-1).clamp_min(1)[:, None, None]
    all_pool = torch.cat([null_pool[:, :, None], pooled], -2)
    all_coverage = torch.cat([null_coverage, coverage], -1)[..., None]
    # Its definition is explicitly zero, including missing corners.
    all_displacement = torch.cat([torch.zeros_like(displacement[:, :, :1]), displacement[:, :, 1:]], -2)
    noop = pooled.new_zeros((b, 8, m, 1)); noop[:, :, 0] = 1
    descriptors = torch.cat([all_pool,
        raw_context[:, :, None].expand(-1, -1, m, -1), all_displacement,
        all_coverage, embedding[:, None, None].expand(-1, 8, m, -1), noop], -1).transpose(1, 2)
    mask = support[:, None, :, None] & action_valid[:, :, None, None]
    return torch.where(mask, descriptors, torch.zeros_like(descriptors))


class ResidualJointScorer(JointActionScorer):
    def __init__(self, method, **config):
        if method not in METHODS:
            raise ValueError(method)
        super().__init__(5, **config)
        self.method = method
        self._inherited_keys = tuple(self.state_dict())
        original = self.inherited_state_dict()
        inherited_sha = dcp_env.state_sha(original)
        expected = read(OLD_DOC / 'A_fits/GEO_seed1.json')['first_step']['initial_state_sha256']
        assert inherited_sha == expected, 'Original seed1 initial state differs'
        pooled_channels = 2 * self.patch_body[2].out_channels
        raw_channels = 5 + self.role_embedding.embedding_dim
        embedding_channels = self.metadata_encoder[2].out_features
        self.descriptor_dim = pooled_channels + raw_channels + 2 + 1 + embedding_channels + 1
        # Generate the common projection once per actual D, then copy its state.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(101)
            self.phi = nn.Sequential(nn.Linear(self.descriptor_dim, 15, bias=True), nn.ReLU())
            if self.descriptor_dim not in _PHI_STATES:
                _PHI_STATES[self.descriptor_dim] = {k: v.detach().clone() for k, v in self.phi.state_dict().items()}
            self.phi.load_state_dict(_PHI_STATES[self.descriptor_dim])
            torch.manual_seed(102)
            first, hidden = (16, 260) if method == 'LOCAL_CAP' else (128, 36)
            self.residual_head = nn.Sequential(nn.Linear(first, hidden, bias=True), nn.ReLU(),
                                               nn.Linear(hidden, 1, bias=False))
            nn.init.zeros_(self.residual_head[2].weight)
        assert sum(p.numel() for p in self.residual_head.parameters()) == 4680
        assert self.descriptor_dim == 81, 'Fixed inherited config has changed'
        assert sum(p.numel() for p in self.parameters()) == 26169
        self.initialization_metadata = dict(method=method, original_seed=1, projection_seed=101,
            head_seed=102, inherited_state_sha256=inherited_sha,
            projection_state_sha256=dcp_env.state_sha(self.phi.state_dict()),
            common_state_sha256=self.common_state_sha(),
            initial_state_sha256=dcp_env.state_sha(self.state_dict()), descriptor_dim=self.descriptor_dim,
            inherited_parameters=20259, projection_parameters=1230, residual_head_parameters=4680,
            total_parameters=26169, final_weight_zero=True, learned_warmstart=False)

    def inherited_state_dict(self):
        state = self.state_dict()
        return {k: state[k] for k in self._inherited_keys}

    def common_state_dict(self):
        return {k: v for k, v in self.state_dict().items() if not k.startswith('residual_head.')}

    def common_state_sha(self):
        return dcp_env.state_sha(self.common_state_dict())

    def forward_bank(self, p3, p4, points, boxes, point_valid, input_shape, *,
                     context, candidate_points, action_valid):
        # This is the original forward arithmetic, executed once. The extension
        # begins only after its support/logits/intermediate tensors are known.
        dtype=self.role_embedding.weight.dtype
        points,boxes,input_shape=points.to(dtype),boxes.to(dtype),input_shape.to(dtype)
        q=candidate_points.to(dtype);b,m=q.shape[:2]
        assert q.shape==(b,m,9,2) and action_valid.shape==(b,m)
        assert torch.all((q[:,0]==points)|(torch.isnan(q[:,0])&torch.isnan(points))) and action_valid[:,0].all()
        if m==1:
            supported=torch.zeros((b,8),device=points.device,dtype=torch.bool)
            logits=points.new_zeros((b,8,1))
            descriptors=points.new_zeros((b,1,8,self.descriptor_dim))
            tokens=points.new_zeros((b,1,8,16))
            return dict(logits=logits,inherited_logits=logits,points_raw=points,
                point_valid=finite(points,point_valid),point_support=supported,candidate_points=q,
                action_valid=action_valid,box_diagonal=(boxes[:,2:]-boxes[:,:2]).norm(dim=-1).clamp_min(1),
                boxes=boxes,coverage=points.new_zeros((b,8,0)),descriptors=descriptors,tokens=tokens,
                residual_scores=points.new_zeros((b,1)))
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
        displacement=(safe_q[:,:,:8]-safe[:,None,:8]).transpose(1,2)/diag[:,None,None,None]/.08
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
        descriptors=descriptors_from_inherited(pooled,null_pool,raw_context,displacement,
            inside.to(dtype).mean(-1),embedding,action_valid,support)
        assert descriptors.shape == (b,m,8,self.descriptor_dim), 'Executed descriptor D differs'
        tokens=projected_tokens(self.phi,descriptors,support,action_valid)
        delta=residual_score(self.method,self.residual_head,tokens,support,action_valid)
        inherited_logits=logits
        logits=logits+delta[:,None,:]
        return dict(logits=logits,inherited_logits=inherited_logits,points_raw=points,point_valid=valid,
            point_support=support,candidate_points=q,action_valid=action_valid,box_diagonal=diag,
            coverage=inside.to(dtype).mean(-1),boxes=boxes,descriptors=descriptors,tokens=tokens,
            residual_scores=delta)


def initialize(method, config, device='cpu'):
    """Independent original seed1 initialization, never a fitted checkpoint."""
    assert (DOC / 'PROTOCOL.json').is_file(), 'Save fixed PROTOCOL before model construction'
    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1)
    return ResidualJointScorer(method, **config).to(device)
