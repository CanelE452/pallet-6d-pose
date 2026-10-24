"""Pure V189 synthetic END-local candidate. No file/model/image operations.

The declared finite convention is separate from the old any-visible broad patch.
It does not relabel an old native D/E/S, owner, quality, or support target.
"""
import numpy as np

END_FRACTION = .2  # nearest existing V127 edge/face witness spacing
PLANE_TRI = np.asarray([[0,1,3],[0,3,2]],np.int32)


def require(ok, message):
    if not bool(ok):
        raise AssertionError(message)


def finite_end(data, native_slot, permCF_to_canonical):
    """Use explicit input slot -> declared canonical role, never q-nearest/GT-best."""
    perm = np.asarray(permCF_to_canonical)
    require(perm.shape == (8,) and sorted(perm.tolist()) == list(range(8)), 'Fixed bijection')
    require(isinstance(native_slot, (int,np.integer)) and 0 <= native_slot < 8, 'Input native slot')
    c = int(perm[native_slot])
    anchor = np.asarray(data['native_anchor3D'][c],np.float64)
    broad = np.asarray(data['native_patch_quads'][c],np.float64)
    require(broad.shape == (3,4,3) and np.isfinite(broad).all(), 'Exact three incident old patches')
    require(np.array_equal(broad[:,0],np.broadcast_to(anchor,(3,3))), 'Constructive common endpoint')
    end = anchor + END_FRACTION * (broad-anchor)
    old_witness = np.asarray(data['native_witness_points'][c],np.float64)
    require(old_witness.shape == (37,3) and np.array_equal(old_witness[0],anchor), 'Endpoint index0')
    # All37 fixed local points are scaled about their declared endpoint, not about q.
    points = anchor + END_FRACTION * (old_witness-anchor)
    return dict(native_slot=int(native_slot),canonical_index=c,
                primary_part_id=int(data['native_primary_part_id'][c]),
                anchor=anchor,finite_end_quads=end,broad_patch_quads=broad,
                finite_end_witness37=points,outer_broad_witnesses=old_witness[10:],
                convention='FINITE_REGISTERED_END_INNER_0P2_OF_OLD_PATCH',
                physical_native_gold=False)


def project(points, camera):
    points = np.asarray(points,np.float64)
    K,R,t = (np.asarray(camera[k],np.float64) for k in ('K','R','t'))
    require(K.shape == R.shape == (3,3) and t.shape == (3,), 'Camera shapes')
    pc = points @ R.T + t
    require(np.isfinite(pc).all() and (pc[...,2] > float(camera.get('near',.001))).all(), 'Near/behind -> UNKNOWN')
    hp = pc @ K.T
    return hp[...,:2]/hp[...,2,None],pc


def propose_pair(data, native_slot, permutation, camera, query_xy, L, raw_hw):
    """Single deterministic proposal, not a search/retry or proof of pixel validity.

    Camera and original q/L are supplied unchanged. Only opaque foreign geometry
    is added. Later actual source raster/crop checks decide PASS or UNKNOWN.
    """
    q=np.asarray(query_xy,np.float64)
    require(q.shape==(2,) and np.isfinite(q).all() and np.isfinite(L) and L>0, 'Missing original q/L -> UNKNOWN')
    g=finite_end(data,native_slot,permutation)
    xy,pc=project(g['finite_end_quads'].reshape(-1,3),camera)
    # Fixed integer-source-pixel margin. No pixel/quality-selected shrinking.
    lo=np.floor(xy.min(0))-1.; hi=np.ceil(xy.max(0))+1.
    h,w=map(int,raw_hw)
    require((lo>=0).all() and hi[0]<w and hi[1]<h, 'End enclosure OOF -> UNKNOWN')
    z0=.5*float(pc[:,2].min())
    K,R,t=(np.asarray(camera[k],np.float64) for k in ('K','R','t'))
    require(K[0,1]==K[1,0]==K[2,0]==K[2,1]==0 and K[2,2]==1 and K[0,0]>0 and K[1,1]>0,'Existing pinhole convention')
    corners=np.asarray([[lo[0],lo[1]],[hi[0],lo[1]],[lo[0],hi[1]],[hi[0],hi[1]]])
    front=np.column_stack(((corners[:,0]-K[0,2])*z0/K[0,0],(corners[:,1]-K[1,2])*z0/K[1,1],np.full(4,z0)))
    # Opaque two-sided planar foreign rectangle, same shape and depth in both
    # variants. Integer image shift preserves its exact finite alpha footprint.
    shift_px=int(np.ceil(hi[0]-lo[0]+8.))
    dx=shift_px*z0/K[0,0]
    sham_cam=front+np.asarray([dx,0.,0.])
    cover=(front-t)@R;sham=(sham_cam-t)@R
    require(np.allclose(sham-cover,np.broadcast_to(np.asarray([dx,0.,0.])@R,cover.shape),atol=1e-14,rtol=0),'Same rigid translated foreign object')
    return dict(**g,cover_vertices=cover,sham_vertices=sham,foreign_triangles=PLANE_TRI.copy(),
                target_triangle_count=int(len(data['triangles'])),
                cover_enclosure_xyxy=np.r_[lo,hi],cover_front_z=z0,
                sham_translation_camera=np.asarray([dx,0.,0.]),sham_translation_image_xy=[shift_px,0],query_xy=q.copy(),L=float(L),
                camera_unchanged=True,native_unchanged=True,requires_actual_pixel_and_support_proofs=True)


def direct_targets(closed_plan):
    """Consume the separately frozen B plan/actual-proof contract, not old labels.

    This pure candidate adapter is exercised on invented closed-proof records only in preparation.
    The future generator must bind B's exact module and all actual proof assets.
    """
    if closed_plan.get('status')!='CONTROLLED_DIRECT_END_WEAK_PAIR_SUPPORTED':
        return dict(BEFORE=-1,COVER=-1,SHAM=-1,reasons=list(closed_plan.get('reasons',[])),
                    native_visibility_target=-1,point_quality_target=-1,training_admission=False)
    require(closed_plan['original_input']['original_query_available'] is True,'InputU never direct known')
    require(closed_plan['direct_end_weak_states']==dict(BEFORE='VISIBLE_WEAK',COVER='EXTERNALLY_BLOCKED_WEAK',SHAM='VISIBLE_WEAK'),'Exact separate weak semantics')
    require(closed_plan['new_native_visibility_target']==closed_plan['new_quality_target']==-1 and closed_plan['training_admission'] is False,'No old native or quality admission')
    return dict(BEFORE=0,COVER=1,SHAM=0,reasons=[],
                target_order='VISIBLE_END0_EXTERNAL_BLOCKED_END1_UNKNOWNminus1',
                authority=closed_plan['authority'],
                native_visibility_target=-1,point_quality_target=-1,training_admission=False)


def requested_target8(native_slot, value):
    require(0<=native_slot<8 and value in (-1,0,1),'Independent selected-native target')
    out=np.full(8,-1,np.int8);out[native_slot]=value
    return out


def balanced_direct_end_loss(logits, targets, available):
    """Equal D/E class means; unknown/U are excluded without pseudo-labels.

    CPU invented-logit gradients are allowed preparation. No model/data loads.
    Missing either class fails closed; caller must not substitute BG/owner/broad.
    """
    import torch
    import torch.nn.functional as F
    require(logits.ndim==1 and targets.shape==available.shape==logits.shape,'Direct shapes')
    require(logits.dtype==torch.float32 and targets.dtype==torch.int8 and available.dtype==torch.bool,'Direct dtypes')
    require(bool(((targets==-1)|(targets==0)|(targets==1)).all()),'Only direct known/UNKNOWN states')
    require(bool((targets[~available]==-1).all()),'Original inputU remains UNKNOWN')
    values=[]
    for cls in (0,1):
        mask=available&(targets==cls)
        require(bool(mask.any()),'No invented zero-class denominator')
        require(bool(torch.isfinite(logits[mask]).all()),'Known direct logits finite')
        values.append(F.binary_cross_entropy_with_logits(logits[mask],torch.full_like(logits[mask],float(cls))))
    return (values[0]+values[1])*.5


def add_direct_objective(base_five_loss, direct_logits, targets, available, weight):
    """Old five-loss mean stays intact, additive direct objective only.

    weight0 bypasses all direct computation/gradients for a genuine replay.
    Fit steps/draw/weight require a separate root prospective fit design.
    """
    require(np.isfinite(weight) and weight>=0,'Prospective nonnegative weight')
    if weight==0:
        return base_five_loss
    return base_five_loss+weight*balanced_direct_end_loss(direct_logits,targets,available)


def make_candidate_head_class(existing_head_module):
    """Inert factory, not called by preparation. Existing 115157 + 65 params.

    The new logit is a separate semantic task. Old broad2 and owner3 outputs
    keep names and meanings; no quality gate and no targets/native in inputs.
    """
    import torch
    from torch import nn
    class DirectEndCandidateHead(existing_head_module.Raw96SupportHead):
        def __init__(self):
            super().__init__()
            with torch.random.fork_rng(devices=[]):
                self.direct_end_head=nn.Linear(64,1)
            nn.init.zeros_(self.direct_end_head.weight);nn.init.zeros_(self.direct_end_head.bias)

        def forward(self,x):
            require(x.ndim==4 and x.shape[1:]==(8,96,96) and x.dtype==torch.float32 and bool(torch.isfinite(x).all()),'Same8 input')
            features=self.shared(self.pool(self.encoder(x)))
            return dict(support_logits=self.support_head(features),owner_logits=self.owner_head(features),
                        direct_end_external_logit=self.direct_end_head(features)[:,0])
    return DirectEndCandidateHead
