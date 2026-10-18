"""Prediction-only H1 inputs with the fixed B-profile feature block neutralized."""
from __future__ import annotations
from copy import deepcopy
import numpy as np
from scripts.research.pallet_boundary_action_repair_v1 import action_data as D

FEATURE_NAMES=(
    'fixed_energy','A_energy','B_energy','AB_energy','edge_energy',
    'full_energy','partial_energy','outline_energy','A_owner_entropy',
    'A_none_mean','AB_owner_entropy','complete_n','partial_n',
    'B_supported_n','B_profile_n','B_entropy_mean','fallback_observation_fraction',
    'point_covariance_sigma_mean','point_residual_mean_px','point_residual_p90_px',
    'partial_normal_mean_sigma','N0_translation_fraction','N0_rotation_radians',
    'min_constraint_slack','positive_depth_fraction','B_consumed_fraction',
    'score_gap_fixed','score_gap_A','score_gap_B','score_gap_AB','score_gap_edge')
REMOVED_NAMES=('B_energy','AB_energy','edge_energy','AB_owner_entropy',
    'B_supported_n','B_profile_n','B_entropy_mean','min_constraint_slack',
    'B_consumed_fraction','score_gap_B','score_gap_AB','score_gap_edge')
REMOVED_INDICES=tuple(FEATURE_NAMES.index(n)for n in REMOVED_NAMES)
LOCAL_MASK=tuple(sorted(REMOVED_INDICES+tuple(i+31 for i in REMOVED_INDICES)))
FINAL_MASK=tuple(i+offset for offset in (0,62,124)for i in LOCAL_MASK)
KEPT_SCALARS=tuple(i for i in range(31)if i not in REMOVED_INDICES)
KEPT_FINAL=tuple(i for i in range(188)if i not in FINAL_MASK)


def dependency_manifest():
    definitions={
        'B_energy':'candidate_scores B objective; fixed observations plus B-profile region likelihood',
        'AB_energy':'candidate_scores AB objective; joint owner/B-profile likelihood',
        'edge_energy':'candidate_scores edge objective; B-profile selected transition location',
        'AB_owner_entropy':'global_owner_responsibilities from the AB audit',
        'B_supported_n':'number of consumed native IDs in the B-profile bundle',
        'B_profile_n':'number of supported queries in the B-profile bundle',
        'B_entropy_mean':'mean entropy of supported offset_probability distributions',
        'min_constraint_slack':'max of per-owner minimum slacks for AB constraints',
        'B_consumed_fraction':'consumed native IDs / max(1, ordinary measurement count)',
        'score_gap_B':'B score minus the original bank/family B-score minimum',
        'score_gap_AB':'AB score minus the original bank/family AB-score minimum',
        'score_gap_edge':'edge score minus the original bank/family edge-score minimum'}
    return dict(schema='NO_B_FIXED_FEATURE_DEPENDENCY_V1',scalar_count=31,input_dim=188,
        removed=[dict(name=n,index=FEATURE_NAMES.index(n),definition=definitions[n],
            final_slots=[FEATURE_NAMES.index(n)+offset for offset in(0,31,62,93,124,155)])for n in REMOVED_NAMES],
        removed_final_slots=list(FINAL_MASK),removed_final_slot_count=72,retained_final_slot_count=116,
        retained_names=[FEATURE_NAMES[i]for i in KEPT_SCALARS],
        pooling='Zero normalized value AND validity before original float32 mean/max pooling',
        candidate_count_slot=186,is_N0_slot=187,
        original_feature_source='scripts/research/pallet_statewise_end_to_end_recovery_v1/solver.py:features_for_bank',
        original_input_source='scripts/research/pallet_boundary_action_repair_v1/action_data.py:inputs',
        indirect_candidate_production_eligibility_and_dedup_dependencies_preserved=True,
        scope='Fixed-bank feature-bundle ablation; not a B-free candidate producer and not a measured inference-time saving')


def inputs(frame,normalizer):
    """Keep original actions/order/188 layout; close every masked pooling path."""
    D.validate_frame(frame)
    if 'feature_names'in frame and tuple(frame['feature_names'])!=FEATURE_NAMES:
        raise ValueError('Original31 scalar meanings changed')
    mean=np.asarray(normalizer['mean']);scale=np.asarray(normalizer['scale'])
    seen=np.asarray(normalizer['valid_train_counts'])>0
    if mean.shape!=(31,)or scale.shape!=(31,)or seen.shape!=(31,)or not np.isfinite(mean).all()or not np.isfinite(scale).all()or np.any(scale<=0):
        raise ValueError('Original31-dimensional normalization contract required')
    entries=D.actions(frame,False);n=len(entries)
    local=np.zeros((n,62),np.float32)
    if n:
        raw=np.asarray([c['features']for c in entries]);valid=raw[:,31:]>0
        # Removed values are never normalized. Retained scalar arithmetic and
        # float32 row layout match the original builder bit for bit.
        keep=np.asarray(KEPT_SCALARS)
        normalized=np.zeros((n,31),dtype=np.result_type(raw,mean,scale))
        normalized[:,keep]=np.where(valid[:,keep]&seen[None,keep],
            (raw[:,keep]-mean[keep])/scale[keep],0.)
        local[:,:31]=normalized;local[:,31:]=raw[:,31:]
        local[:,LOCAL_MASK]=0.
        pm,px=local.mean(0),local.max(0)
    else:pm=px=np.zeros(62,np.float32)
    out=np.zeros((n,188),np.float32)
    if n:out[:,:62]=local
    out[:,62:124]=pm;out[:,124:186]=px;out[:,186]=np.log1p(n)
    if n:out[:,187]=[float(c['is_N0'])for c in entries]
    if not np.isfinite(out).all()or np.any(out[:,FINAL_MASK]!=0):
        raise ValueError('No-B feature closure is invalid')
    return out,entries


def infer(frame,model,normalizer):
    """Same original H1 tie/readout; return an unchanged actual stored pose."""
    import torch
    from scripts.research.pallet_boundary_action_repair_v1 import action_models as M
    x,entries=inputs(frame,normalizer)
    if not entries:
        return dict(actual_pose=dict(available=False),fallback=False,prediction_status='NO_POSE',
            selected_candidate_id=None,action_role='NO_POSE',selected_source_aliases=[],logits=[],actions=[],eligible_candidates=0,archived_candidates=len(frame['candidates']))
    with torch.inference_mode():
        logits=model(torch.from_numpy(x)).squeeze(-1);scores=logits.numpy();probabilities=logits.sigmoid().numpy()
    selected=M.choose(scores,entries,'H1')
    action=entries[selected];aliases=sorted(D.source_aliases(action))
    role='N0'if'N0'in aliases else'E0'if'E0'in aliases else'OTHER'
    return dict(actual_pose=deepcopy(action['actual_pose']),fallback=False,none_selected=False,
        prediction_status='ACTUAL_ACTION_SELECTED',selected_candidate_id=action['candidate_id'],action_role=role,
        selected_source_aliases=aliases,selected_score=float(scores[selected]),probability=float(probabilities[selected]),
        probability_is_calibrated_confidence=False,eligible_candidates=len(entries),archived_candidates=len(frame['candidates']),
        logits=scores.tolist(),actions=[dict(candidate_id=c['candidate_id'],role=c['action_role'],
            source_aliases=sorted(D.source_aliases(c)),score=float(scores[i]),sigmoid_or_softmax=float(probabilities[i]))for i,c in enumerate(entries)])
