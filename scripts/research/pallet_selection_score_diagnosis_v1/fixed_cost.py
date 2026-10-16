"""Read-only original G10/G13 objective accounting at already stored poses.

No solver, initializer, network, line search, or observation generator is
called.  Original preparation, objective and inequality evaluators are used.
The outline scalar fast path is opt-in only after ``check_outline_scalar``
has established parity with the actual scalar-and-gradient implementation.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from itertools import product
import importlib
import time

import numpy as np
from scipy.spatial import ConvexHull, QhullError

PREFIX = 'scripts.research.pallet_visibility_curriculum_20261013_v1.'
ATOL = 1e-9
FAMILIES = ('ORIGINAL_WD', 'SWAPPED_WD')
LOSS_FAMILIES = ('full_BI', 'partial_normal', 'outline', 'prior')


def _module(name):
    return importlib.import_module(PREFIX + name)


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    return value


def weighted_components(parts):
    """Only additive, actually weighted scalars; never metadata as a loss."""
    return dict(full_BI=float(parts.get('soft_BI_robust_loss', 0.)),
                partial_normal=float(parts.get('partial_boundary_normal_robust_loss', 0.)),
                outline=float(parts.get('global_outline_robust_mean_loss', 0.))
                * float(parts.get('global_outline_factor_weight', 0.)),
                prior=float(parts.get('coarse_pose_prior_loss', 0.)))


@dataclass
class Context:
    packet: dict
    stage: str
    family_name: str
    backend: object
    xyz: np.ndarray
    K: np.ndarray
    states: list
    measurements: list
    domains: list
    image_hw: tuple
    cfg: dict
    boundary: list
    regions: list
    Q: np.ndarray
    prior: dict
    requested_prior: float
    scale: float
    outline: dict
    outline_cfg: dict
    use_outline: bool
    actual_backend: str
    admission: bool
    scalar_only: bool = False


def prepare(packet, stage, family, *, scalar_only=False):
    """Prepare only immutable original inputs; does not enter any solver body."""
    if stage not in ('G10', 'G13') or family not in FAMILIES:
        raise ValueError('An original G10/G13 and original W/D family are required')
    spec = packet['solver_input']
    fam = packet['original_e_stage1_family_seed_bank'][family]['family']
    typed = stage == 'G13' and any(r.get('explicit_complete_BI_likelihood_revocation') is True
                                 for r in spec['boundary_observations'])
    name = ('statewise_partial_BI_outline_solver_v4_13' if typed else
            'statewise_global_outline_solver_v4_10' if stage == 'G10' else
            'statewise_global_outline_solver_v4_11')
    backend = _module(name)
    raw_boundary = (spec['boundary_observations'] if stage == 'G13' else
                    packet['original_e_legacy_boundary_observations'])
    outline = deepcopy(spec['outline_observation'])
    ocfg = backend._outline_config(outline.get('configuration'))
    values = backend._prepare(fam['native_xyz'], spec['K'], deepcopy(spec['states']),
        deepcopy(spec['visible']), deepcopy(spec['domains']), spec['image_hw'],
        deepcopy(fam['coarse_pose']), deepcopy(spec['config']),
        deepcopy(raw_boundary), deepcopy(spec['u_regions']))
    xyz,K,states,measurements,domains,image_hw,cfg,boundary,regions,Q,prior,requested = values
    admission = len(measurements) >= 2 and (len(measurements) != 2 or bool(domains or regions or boundary))
    use_outline = bool(outline.get('supported') and ocfg['outline_factor_weight'] > 0 and admission)
    actual_backend = name
    if not use_outline:
        # The original non-typed outline consumer returns its first WD result
        # untouched.  It does not suddenly apply its legacy partial factors.
        # Typed partial-only does run its own normal/range package (already
        # stored); here we evaluate that package without calling its solver.
        if typed:
            actual_backend = 'statewise_partial_BI_u_solver_v4_13'
        else:
            boundary = []
            actual_backend = 'DELEGATED_ORIGINAL_WD_FULL_BI_U_RESULT'
        covariance = cfg['prior_covariance6']
        if covariance is None:
            sigma = np.r_[np.repeat(np.deg2rad(cfg['prior_rotation_sigma_deg']),3),
                          np.repeat(cfg['prior_ray_sigma'],2),cfg['prior_log_depth_sigma']]
            covariance = np.diag(sigma**2)
        prior = {**prior, 'whitening': np.linalg.inv(np.linalg.cholesky(np.asarray(covariance)))}
    return Context(packet,stage,family,backend,xyz,K,states,measurements,domains,tuple(image_hw),cfg,
                   boundary,regions,Q,prior,requested,
                   max(float(np.linalg.norm(np.ptp(xyz,axis=0))),1e-3),outline,ocfg,use_outline,
                   actual_backend,admission,scalar_only)


def _candidate_record(candidate):
    # Original retained sample supplies comparator scalar/name; the logging
    # endpoint supplies the otherwise absent components and branch. Preserve
    # their source records outside this read-only merged evaluator view.
    record = dict(candidate.get('original_record', candidate.get('record', candidate)))
    endpoint = candidate.get('original_endpoint_record') or {}
    return {**record, **endpoint}


def _pose(context, candidate):
    V = context.backend.V
    record = _candidate_record(candidate)
    R = np.asarray(candidate.get('R',candidate.get('R_cf',record.get('R'))),float)
    t = np.asarray(candidate.get('t',candidate.get('centroid',record.get('t'))),float)
    if R.shape != (3,3) or t.shape != (3,) or not V._valid_rotation(R) or not np.isfinite(t).all():
        raise ValueError('Actual stored proper finite native R/t required')
    Q = candidate.get('Q',candidate.get('physical_registration_Q'))
    if Q is not None and not np.allclose(Q, context.Q, atol=1e-12,rtol=0):
        raise ValueError('Candidate uses another native/physical family registration')
    p = np.asarray(record.get('p',V._pose_vector(R,t,context.scale)),float)
    restored_R,restored_t = V._pose(p,context.scale)
    if not (np.allclose(R,restored_R,atol=ATOL,rtol=0) and np.allclose(t,restored_t,atol=ATOL,rtol=0)):
        raise ValueError('Stored candidate p and native R/t disagree')
    return R,t,p


def check_outline_scalar(context, candidate):
    """Reference-independent preflight for skipping an unused gradient only."""
    if not context.use_outline:
        return dict(applicable=False,reason='No active original outline factor')
    _,_,p = _pose(context,candidate)
    original,gradient,audit = context.backend.outline_value_and_gradient(
        p,context.xyz,context.K,context.scale,context.outline,context.outline_cfg)
    scalar,scalar_audit = context.backend._outline_value(
        p,context.xyz,context.K,context.scale,context.outline,context.outline_cfg)
    difference = abs(float(original)-float(scalar))
    if difference > ATOL:
        raise ValueError(f'Outline scalar/gradient value parity failed: {difference}')
    return dict(applicable=True,pass_=True,atol=ATOL,rtol=0.,difference=difference,
                original_value=float(original),scalar_value=float(scalar),
                original_gradient_evaluated=True,gradient_norm=float(np.linalg.norm(gradient)),
                audit_valid=bool(audit.get('valid_projection')),
                scalar_audit_valid=bool(scalar_audit.get('valid_projection')))


def _outline_observations(ctx,p,*,dense):
    """Exact original samples, loss and arc normalizers, never new line fitting."""
    b=ctx.backend; R,t=b.V._pose(p,ctx.scale);q,depth=b.V.project_native(ctx.xyz,R,t,ctx.K)
    if not np.isfinite(q).all() or np.min(depth)<=1e-8:
        return [],False
    try:polygon=q[ConvexHull(q).vertices]
    except QhullError:return [],False
    observed=np.asarray(ctx.outline['observed_boundary_samples_xy'],float)
    actual=np.asarray(ctx.outline['observed_boundary_segments_xy'],float)
    reverse=b._nearest_residual(observed,polygon)
    h,w=ctx.outline['image_hw'];margin=ctx.outline_cfg['frame_margin_px']
    bounds=((margin,w-1-margin),(margin,h-1-margin));points=[];weights=[]
    for a,z in zip(*b._segments(polygon)):
        clipped=b._clip(a,z,bounds)
        if clipped is None:continue
        a,z=clipped;length=float(np.linalg.norm(z-a));n=ctx.outline_cfg['model_samples_per_segment']
        points.extend(a[None,:]+((np.arange(n)+.5)/n)[:,None]*(z-a));weights.extend([length/n]*n)
    if not points:return [],False
    forward=b._nearest_segment_residual(np.asarray(points),actual)
    delta=ctx.outline_cfg['huber_delta'];sigma=ctx.outline_cfg['normal_sigma_px']
    factor=ctx.outline_cfg['outline_factor_weight'];rows=[]
    for direction,residual,weight,xy in [('model_to_observed',forward,np.asarray(weights),np.asarray(points)),
                                        ('observed_to_model',reverse,np.ones(len(reverse)),observed)]:
        radius=np.linalg.norm(residual,axis=1)/sigma
        loss=delta**2*(np.sqrt(1+(radius/delta)**2)-1)
        contribution=.5*factor*weight*loss/weight.sum()
        entry=dict(loss_family='outline',observation_id='outline:'+direction,direction=direction,
            state=None,observation_type='GLOBAL_CONVEX_SHAPE',measurement_source='SAVED_GLOBAL_MASK_OUTLINE',
            semantic_owner_status='GLOBAL_SHAPE_NOT_A_PHYSICAL_CORNER_CERTIFICATE',
            source_binding=deepcopy(ctx.outline.get('image_binding')),
            count=len(residual),normalizer=float(weight.sum()),normalizer_type=('arc_length_px' if direction=='model_to_observed' else 'observed_sample_count'),
            factor_weight=factor,direction_weight=.5,raw_loss_sum=float(np.sum(loss)),
            robust_mean=float(np.average(loss,weights=weight)),weighted_contribution=float(np.sum(contribution)),
            residual_mean_px=float(np.linalg.norm(residual,axis=1).mean()),active=True)
        if dense:
            entry.update(sample_xy=xy.tolist(),residual_xy_px=residual.tolist(),
                         standardized_radius=radius.tolist(),robust_losses=loss.tolist(),
                         arc_or_sample_weights=weight.tolist(),weighted_sample_contributions=contribution.tolist())
        rows.append(entry)
    return rows,True


def _constraint_labels(ctx,domains,sides,boundary,regions):
    labels=[dict(kind='positive_depth',native_corner=k,units='depth_over_model_diagonal') for k in range(8)]
    for m in ctx.measurements:
        k=m['native_corner']
        if ctx.cfg['enforce_visible_geometry']:
            labels.append(dict(kind='visible_geometry',native_corner=k,state=ctx.states[k],units='cosine_plus_sine_tolerance'))
        if ctx.cfg['enforce_visible_canvas']:
            labels.extend(dict(kind='visible_canvas',native_corner=k,state=ctx.states[k],side=side,units='canvas_fraction')
                          for side in ('left','top','right','bottom'))
    for d,side in zip(domains,sides):
        k=d['native_corner'];common=dict(native_corner=k,state=d['state'])
        if d['shape']=='ellipse':labels.append(dict(kind='OT_ellipse',units='one_minus_quadratic',**common))
        if d['shape']=='box':labels.extend(dict(kind='OT_box',side=s,units='box_span_fraction',**common) for s in ('left','top','right','bottom'))
        labels.extend(dict(kind='OT_halfplane',halfplane_index=i,units='pixels',**common)
                      for i,_ in enumerate(d['halfplanes']))
        if d['state']=='T':labels.append(dict(kind='T_outside_branch',side=side,units='pixels',**common))
    labels.extend(dict(kind='U_ellipse',native_corner=r['native_corner'],region_index=r['u_region_index'],units='one_minus_quadratic') for r in regions)
    for b in boundary:
        for i,_ in enumerate(b['lines']):
            labels.extend(dict(kind='partial_finite_tangent',native_corner=b['native_corner'],line_index=i,side=s,units='pixels') for s in ('lower','upper'))
    return labels


def evaluate(packet, stage, family, candidate, *, context=None, dense_outline=False, strict=True):
    """Account for the unmodified objective at one stored candidate pose.

    Dense per-outline-sample arrays are optional to avoid duplicating thousands
    of samples in a public compact ledger. Direction totals, counts, and exact
    normalizers are always retained. Raw packet plus pose deterministically
    reproduces the dense arrays with ``dense_outline=True``.
    """
    started=time.monotonic();cpu=time.process_time()
    ctx=context or prepare(packet,stage,family)
    if (ctx.packet['id'],ctx.stage,ctx.family_name)!=(packet['id'],stage,family):
        raise ValueError('Prepared context belongs to another packet/stage/family')
    R,t,p=_pose(ctx,candidate);b=ctx.backend;P=b.P;BASE=P.BASE
    record=_candidate_record(candidate)
    seed=(candidate.get('kind')=='partial_BI_seed_fit' or candidate.get('objective_scope')=='BI_SEED_ONLY')
    boundary=[] if seed else ctx.boundary
    measurements=ctx.measurements
    if seed:
        value,gradient,parts=BASE._objective_pair(p,ctx.xyz,ctx.K,measurements,ctx.scale,ctx.prior,ctx.cfg)
    else:
        value,gradient,parts=P._joint_objective_pair(p,ctx.xyz,ctx.K,measurements,ctx.scale,ctx.prior,ctx.cfg,boundary,1.)
    use_outline=ctx.use_outline and not seed
    if use_outline:
        if ctx.scalar_only:
            extra,audit=b._outline_value(p,ctx.xyz,ctx.K,ctx.scale,ctx.outline,ctx.outline_cfg)
            outline_gradient=False
        else:
            extra,eg,audit=b.outline_value_and_gradient(p,ctx.xyz,ctx.K,ctx.scale,ctx.outline,ctx.outline_cfg)
            gradient=gradient+ctx.outline_cfg['outline_factor_weight']*eg
            outline_gradient=True
        value+=ctx.outline_cfg['outline_factor_weight']*extra
        parts={**parts,'global_outline_robust_mean_loss':extra,
               'global_outline_factor_weight':ctx.outline_cfg['outline_factor_weight'],
               'global_outline_audit':audit,'coarse_pose_prior_loss':0.,
               'S_keypoint_image_terms':0,'U_keypoint_image_terms':0}
    else:outline_gradient=False
    weights=weighted_components(parts);summed=sum(weights.values())
    if not np.isclose(summed,value,atol=ATOL,rtol=0):
        raise ValueError(f'Objective additive reconstruction failed: {summed} vs {value}')
    stored=record.get('objective',candidate.get('objective'))
    storedparts=record.get('objective_components',record.get('components',candidate.get('objective_components')))
    same_package=candidate.get('objective_scope') not in ('COUNTERFACTUAL_FIXED_POSE','EXTERNAL_POSE_NOT_IN_E_BANK','INCOMING_FROM_OTHER_BACKEND')
    difference=None if stored is None else float(value)-float(stored)
    objective_parity=None if stored is None or not same_package else abs(difference)<=ATOL
    component_differences={}
    if storedparts and same_package:
        original_weights=weighted_components(storedparts)
        component_differences={k:weights[k]-original_weights[k] for k in LOSS_FAMILIES}
        component_parity=all(abs(x)<=ATOL for x in component_differences.values())
        storedsum=sum(original_weights.values())
        storedsum_parity=stored is None or abs(storedsum-float(stored))<=ATOL
    else:component_parity=None;storedsum_parity=None
    if strict and (objective_parity is False or component_parity is False or storedsum_parity is False):
        raise ValueError(f'Original fixed cost parity failed {packet["id"]}/{stage}/{family}: objective={difference}, components={component_differences}')
    # Saved original components are authoritative when available and matched.
    authoritative=weighted_components(storedparts) if storedparts and component_parity else weights
    rows=[];removed={r['native_corner'] for r in boundary};source={r['native_corner']:r for r in packet['solver_input']['visible']}
    q,depth=b.V.project_native(ctx.xyz,R,t,ctx.K)
    for m in measurements:
        k=m['native_corner'];raw=q[k]-m['xy'];white=m['whitening']@raw
        radius2=float(white@white);delta=ctx.cfg['huber_scale'];reliability=m['fit_reliability_weight']
        robust=2*delta**2*(np.sqrt(1+radius2/delta**2)-1)
        entry=dict(loss_family='full_BI',observation_id=f'full:{k}',native_corner=k,state=ctx.states[k],
            observation_type='COMPLETE_2D',measurement_source=source[k].get('source','VISIBLE_BACKBONE_FALLBACK'),
            inherited_predictor_fallback=source[k].get('source','VISIBLE_BACKBONE_FALLBACK')=='VISIBLE_BACKBONE_FALLBACK',
            semantic_owner_status='SEMANTIC_OWNER_UNVERIFIED',assumed_physical_owner_native_xyz=ctx.xyz[k].tolist(),
            assumed_physical_owner_Q=ctx.Q.tolist(),measurement_xy=m['xy'].tolist(),projected_xy=q[k].tolist(),
            raw_residual_xy_px=raw.tolist(),raw_radius_px=float(np.linalg.norm(raw)),
            whitened_residual=white.tolist(),observed_covariance_px2=m['observed_covariance_px2'].tolist(),
            fit_covariance_px2=m['covariance_px2'].tolist(),whitening=m['whitening'].tolist(),
            reliability_weight=float(reliability),huber_scale=delta,raw_robust_loss=float(robust),
            weighted_contribution=float(reliability*robust) if k not in removed else 0.,
            active=k not in removed,reason='complete_2D_revoked_by_partial' if k in removed else 'active_original_full_BI',
            count=1,normalizer=1.,aggregation='sum',provenance=deepcopy(source[k].get('provenance')))
        rows.append(entry)
    N=P.N
    for r in boundary:
        normal_value,_=N._normal_objective_pair(p,ctx.xyz,ctx.K,[r],ctx.scale,r['fit_reliability_weight'],ctx.cfg['huber_scale'])
        lines=[]
        for i,line in enumerate(r['lines']):
            raw=float(line['normal_unit_xy']@q[r['native_corner']]+line['offset_px'])
            lines.append(dict(line_index=i,normal_unit_xy=line['normal_unit_xy'].tolist(),offset_px=line['offset_px'],
                normal_variance_px2=line['normal_variance_px2'],normal_residual_px=raw,
                standardized_residual=raw/np.sqrt(line['normal_variance_px2']),
                support_tangent_interval_px=line['support_tangent_interval_px'].tolist(),
                support_extrapolation_px=line['support_extrapolation_px'],provenance=deepcopy(line['provenance'])))
        rows.append(dict(loss_family='partial_normal',observation_id=f'partial:{r["native_corner"]}',
            native_corner=r['native_corner'],state=ctx.states[r['native_corner']],observation_type='PARTIAL_NORMAL_AND_FINITE_TANGENT',
            measurement_source=source[r['native_corner']].get('source','VISIBLE_BACKBONE_FALLBACK'),
            semantic_owner_status='SEMANTIC_OWNER_UNVERIFIED',lines=lines,
            grouped_radial_robust_loss=True,lines_are_not_individually_additive=True,
            weighted_contribution=float(normal_value),reliability_weight=r['fit_reliability_weight'],
            active=True,count=len(lines),normalizer=1.,aggregation='sum_of_corner_group_losses',
            association_provenance=deepcopy(r['association_provenance']),image_binding=deepcopy(r['image_binding'])))
    if use_outline:
        outline_rows,outline_valid=_outline_observations(ctx,p,dense=dense_outline)
        rows.extend(outline_rows)
    else:outline_valid=None
    obs_sums={k:sum(r['weighted_contribution'] for r in rows if r['loss_family']==k) for k in LOSS_FAMILIES if k!='prior'}
    obs_differences={k:obs_sums[k]-weights[k] for k in obs_sums}
    if strict and any(abs(x)>ATOL for x in obs_differences.values()):
        raise ValueError(f'Per-observation objective decomposition failed: {obs_differences}')
    domains=[] if seed else ctx.domains;regions=[] if seed else ctx.regions
    choices=[(list(d['outside_canvas_side']) if isinstance(d['outside_canvas_side'],tuple) else
              [d['outside_canvas_side']] if d['outside_canvas_side'] else list(b.V._SIDES)) if d['state']=='T' else [None] for d in domains]
    branches=list(product(*choices));branch=record.get('branch',candidate.get('branch',0))
    sides=record.get('sides')
    if sides is None:
        if branch is None:branch=0
        if not isinstance(branch,int) or not 0<=branch<len(branches):raise ValueError('Original branch is unavailable')
        sides=branches[branch]
    if seed:
        slacks,_=BASE._constraints(p,ctx.xyz,ctx.K,measurements,domains,sides,ctx.image_hw,ctx.scale,ctx.cfg)
    else:
        slacks,_=P._joint_constraint_pair(p,ctx.xyz,ctx.K,measurements,domains,sides,ctx.image_hw,ctx.scale,ctx.cfg,boundary,regions)
    labels=_constraint_labels(ctx,domains,sides,boundary,regions)
    if len(labels)!=len(slacks):raise ValueError(f'Constraint row mapping mismatch {len(labels)} vs {len(slacks)}')
    constraints=[{**label,'row_index':i,'slack':float(x),'passes':bool(np.isfinite(x) and x>=-ctx.cfg['feasibility_tolerance']),
                  'role':'HARD_CONSTRAINT_NOT_ADDITIVE_COST'} for i,(label,x) in enumerate(zip(labels,slacks))]
    feasible=bool(np.isfinite(q).all() and np.isfinite(slacks).all() and np.min(slacks)>=-ctx.cfg['feasibility_tolerance']
                  and np.min(depth)>=ctx.cfg['positive_depth_m']-1e-10 and b.V._valid_rotation(R))
    return _plain(dict(schema='ORIGINAL_FIXED_POSE_COST_ACCOUNT_V1',id=packet['id'],stage=stage,family=family,
        actual_backend=ctx.actual_backend,objective_scope='BI_SEED_ONLY' if seed else 'ORIGINAL_STAGE_FAMILY_PACKAGE',
        objective=float(stored) if stored is not None and objective_parity else float(value),
        objective_recomputed=float(value),objective_stored=stored,weighted_components=authoritative,
        objective_components_original=deepcopy(storedparts),objective_components_recomputed=parts,
        additive_sum=float(summed),additive_difference=float(summed-value),
        stored_objective_parity=objective_parity,stored_component_parity=component_parity,
        stored_component_sum_parity=storedsum_parity,stored_objective_difference=difference,
        stored_component_differences=component_differences,per_observation=rows,
        observation_component_differences=obs_differences,constraints=constraints,
        minimum_constraint_slack=float(np.min(slacks)),recomputed_joint_feasible=feasible,
        original_joint_feasible=record.get('joint_feasible',record.get('feasible',candidate.get('feasible'))),
        term_status=dict(
            full_BI=dict(active=bool(len(measurements)-len(removed)),reason='sum_of_unrevoked_complete_BI_likelihoods'),
            partial_normal=dict(active=bool(boundary),reason='grouped_normal_likelihood' if boundary else 'inactive_no_original_partial_factor'),
            outline=dict(active=use_outline,reason='original_global_outline_factor' if use_outline else 'inactive_original_delegation_or_partial_only'),
            prior=dict(active=ctx.cfg['prior_weight']>0,reason='inactive_effective_prior_weight_zero' if ctx.cfg['prior_weight']==0 else 'original_effective_prior')),
        branch=branch,sides=list(sides),actual_prior_weight=ctx.cfg['prior_weight'],
        requested_prior_weight=ctx.requested_prior,outline_active=use_outline,outline_valid=outline_valid,
        full_2D_revoked_ids=sorted(removed),S_point_terms=0,U_point_terms=0,
        gradient_calculated_for_BI_partial_prior=True,gradient_calculated_for_outline=outline_gradient,
        outline_scalar_only=ctx.scalar_only,dense_outline_arrays=dense_outline,
        new_pose_solve_calls=0,new_candidates=0,new_observation_calls=0,
        elapsed_seconds=time.monotonic()-started,worker_cpu_seconds=time.process_time()-cpu))


def pilot(old_root, selection_directory, output):
    """Original ordered CAL first six only; no selection using reference errors."""
    from collections import Counter
    from pathlib import Path
    from .common import rows, save_rows, sha, write, timed_stage
    from .runtime_guard import FixedPoseOnly
    old_root,selection_directory,output=map(Path,(old_root,selection_directory,output))
    output.mkdir(parents=True,exist_ok=False)
    packets=[]
    for packet in rows(old_root/'e_trace_v2/TYPED_PACKETS.jsonl.gz'):
        if packet['partition']=='CAL':packets.append(packet)
        if len(packets)==6:break
    ids=[p['id'] for p in packets]
    by_id={p['id']:p for p in packets}
    candidates=[c for c in rows(selection_directory/'RETAINED_CANDIDATES.jsonl.gz') if c['id'] in by_id]
    contexts={};records=[];max_original=0.;max_scalar=0.;full_elapsed=0.;fast_elapsed=0.;counts=Counter()
    with timed_stage(output,'fixed_cost_first_six_original_CAL',new_pose_calls=0,new_model_forwards=0) as timing:
        with FixedPoseOnly() as guard:
            for candidate in candidates:
                packet=by_id[candidate['id']];stage=candidate['stage'];family=candidate['family']
                key=(packet['id'],stage,family)
                if key not in contexts:contexts[key]=prepare(packet,stage,family)
                ctx=contexts[key]
                scalar=check_outline_scalar(ctx,candidate)
                full=evaluate(packet,stage,family,candidate,context=ctx,dense_outline=True)
                full_elapsed+=full['elapsed_seconds'];max_original=max(max_original,abs(full['stored_objective_difference'] or 0.))
                ctx.scalar_only=True
                fast=evaluate(packet,stage,family,candidate,context=ctx,dense_outline=True)
                ctx.scalar_only=False
                fast_elapsed+=fast['elapsed_seconds']
                d=abs(full['objective_recomputed']-fast['objective_recomputed']);max_scalar=max(max_scalar,d)
                if d>ATOL:raise ValueError('Full/scalar-only objective parity failed')
                for k in LOSS_FAMILIES:
                    if abs(full['weighted_components'][k]-fast['weighted_components'][k])>ATOL:
                        raise ValueError('Full/scalar-only component parity failed')
                counts[stage]+=1
                records.append(dict(candidate_id=candidate['candidate_id'],id=packet['id'],stage=stage,family=family,
                    stored_objective_parity=full['stored_objective_parity'],stored_component_parity=full['stored_component_parity'],
                    additive_difference=full['additive_difference'],observation_component_differences=full['observation_component_differences'],
                    stored_objective_difference=full['stored_objective_difference'],outline_scalar_parity=scalar,
                    scalar_only_objective_difference=d,elapsed_original=full['elapsed_seconds'],elapsed_scalar_only=fast['elapsed_seconds']))
            guard_receipt=guard.receipt()
    save_rows(output/'PILOT_COST_ROWS.jsonl.gz',records)
    receipt=dict(schema='FIXED_COST_ORIGINAL_CAL_FIRST_SIX_PILOT_V1',ids=ids,selection_rule='original ordered CAL list first six; no reference-dependent selection',
        candidate_rows=len(records),stage_counts=dict(counts),all_stored_objective_parity=True,
        max_stored_objective_difference=max_original,max_outline_scalar_only_difference=max_scalar,
        atol=ATOL,rtol=0.,original_gradient_elapsed_seconds=full_elapsed,scalar_only_elapsed_seconds=fast_elapsed,
        scalar_only_mean_seconds=fast_elapsed/len(records),projected_11603_scalar_seconds=11603*fast_elapsed/len(records),
        original_candidate_source_sha256=sha(selection_directory/'RETAINED_CANDIDATES.jsonl.gz'),
        typed_packet_sha256=sha(old_root/'e_trace_v2/TYPED_PACKETS.jsonl.gz'),source_sha256=sha(Path(__file__)),
        row_sha256=sha(output/'PILOT_COST_ROWS.jsonl.gz'),runtime_guard=guard_receipt,**timing)
    write(output/'PILOT_COST_RECEIPT.json',receipt)
    return receipt


if __name__=='__main__':
    import argparse,json
    parser=argparse.ArgumentParser()
    parser.add_argument('--pilot-old-root',required=True)
    parser.add_argument('--selection-directory',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    print(json.dumps(pilot(args.pilot_old_root,args.selection_directory,args.output),ensure_ascii=False,indent=2))
