"""Shared actual SE(3) solving, carried candidates, and reference-free features.

Every original G10/G13 pose is evaluated with ONE G13 input package per family.
No cross-objective comparison is made. The automatic methods share proposal
budgets, fixed observations and a numerical-failure N0 fallback.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import time

import numpy as np
import cv2
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.spatial.transform import Rotation

from scripts.research.pallet_selection_score_diagnosis_v1.fixed_cost import prepare
from scripts.research.pallet_visibility_curriculum_20261013_v1 import joint_ot_jacobian_v2 as J
from scripts.research.pallet_visibility_curriculum_20261013_v1 import joint_ot_pnp as V
from . import owner_model as A


FEATURE_NAMES = (
    'fixed_energy', 'A_energy', 'B_energy', 'AB_energy', 'edge_energy',
    'full_energy', 'partial_energy', 'outline_energy', 'A_owner_entropy',
    'A_none_mean', 'AB_owner_entropy', 'complete_n', 'partial_n',
    'B_supported_n', 'B_profile_n', 'B_entropy_mean', 'fallback_observation_fraction',
    'point_covariance_sigma_mean', 'point_residual_mean_px', 'point_residual_p90_px',
    'partial_normal_mean_sigma', 'N0_translation_fraction', 'N0_rotation_radians',
    'min_constraint_slack', 'positive_depth_fraction', 'B_consumed_fraction',
    'score_gap_fixed', 'score_gap_A', 'score_gap_B', 'score_gap_AB', 'score_gap_edge',
)


def pose_distance(a, b, scale):
    aa, bb = np.asarray(a['R_physical']), np.asarray(b['R_physical'])
    angle = Rotation.from_matrix(aa@bb.T).magnitude()
    delta = np.linalg.norm(np.asarray(a['centroid'])-b['centroid'])/scale
    return float(angle+delta)


def unique_poses(candidates, scale):
    """Inherited original numerical duplicate tolerance (strict pose equality).

    Source roles are retained on the first physical pose; evaluation uses C2,
    while duplicate removal here deliberately does not equate distinct images
    of the same object's C2 phase or collapse 90-degree owner alternatives.
    """
    result = []
    for c in candidates:
        R, t = np.asarray(c['R_physical']), np.asarray(c['centroid'])
        if not np.isfinite(R).all() or not np.isfinite(t).all():
            continue
        if any(np.max(np.abs(R-np.asarray(x['R_physical']))) <= 1e-7
               and np.max(np.abs(t-np.asarray(x['centroid']))) <= 1e-7*scale for x in result):
            continue
        result.append(deepcopy(c))
    return result


def _native_pose(candidate, ctx):
    R = np.asarray(candidate['R_physical'])@ctx.Q.T
    t = np.asarray(candidate['centroid'])-R@ctx.xyz.mean(0)
    return V._pose_vector(R, t, ctx.scale)


def _actual_pose(p, ctx):
    R, t = V._pose(p, ctx.scale)
    return dict(available=True, R_cf=R.tolist(), R_physical=(R@ctx.Q).tolist(),
                centroid=(R@ctx.xyz.mean(0)+t).tolist(), native_t=t.tolist(),
                cf_extents=np.ptp(ctx.xyz, axis=0).tolist(),
                physical_registration_Q=ctx.Q.tolist())


def _rebuild_geometry(records, ctx):
    normals = V.registered_incident_normals(ctx.xyz)
    for r in records:
        k = r['native_corner']
        if 'object_xyz' in r:
            r['object_xyz'] = ctx.xyz[k].copy()
        if 'incident_normals' in r:
            r['incident_normals'] = normals[k].copy()
    return records


@dataclass
class FamilyModel:
    ctx: object
    owner: dict
    bundle: dict | None
    constraints_by_h: list
    replaced_constraints_by_h: list


def compile_frame(frame, config, *, bundle=None):
    models = {}
    for name in frame['families']:
        ctx = prepare(frame['packet'], 'G13', name, scalar_only=True)
        originals = {m['native_corner']: m for m in frame['packet']['solver_input']['visible']}
        for m in ctx.measurements:
            source = originals[m['native_corner']]
            m['source'] = source.get('source', 'UNKNOWN_ORIGINAL_SOURCE')
            m['state'] = ctx.states[m['native_corner']]
            m['image_measurement'] = source.get('image_measurement', True)
            m['measurement_is_pose_projection'] = source.get('measurement_is_pose_projection', False)
        owner = A.compile_owner_groups(ctx, none_prior=config['none_prior'])
        consumed = set(bundle.get('consumed_native_ids', [])) if bundle else set()
        original, replaced = [], []
        for perm in owner['permutations']:
            ms = _rebuild_geometry(A.map_owner_records(ctx.measurements, perm, ctx.xyz), ctx)
            bs = _rebuild_geometry(A.map_owner_records(ctx.boundary, perm, ctx.xyz), ctx)
            ds = A.map_owner_records(ctx.domains, perm, ctx.xyz)
            us = A.map_owner_records(ctx.regions, perm, ctx.xyz)
            original.append((ms, bs, ds, us))
            # Region is a replacement for old B image support, including its
            # borrowed tangent range. Geometry/canvas guards remain on all ms.
            replaced.append((ms, [b for b, source in zip(bs, ctx.boundary)
                                  if source['native_corner'] not in consumed], ds, us))
        models[name] = FamilyModel(ctx, owner, bundle, original, replaced)
    return models


def _sides(domains, q, image_hw):
    H, W = image_hw
    result = []
    for d in domains:
        if d['state'] != 'T':
            result.append(None)
            continue
        side = d.get('outside_canvas_side')
        allowed = list(side) if isinstance(side, (tuple, list)) else [side] if side else list(V._SIDES)
        x, y = q[d['native_corner']]
        margin = d.get('outside_margin_px', 0.)
        values = dict(left=-margin-x, right=x-W-margin, top=-margin-y, bottom=y-H-margin)
        result.append(max(allowed, key=lambda s: values[s]))
    return tuple(result)


def constraints(model, p, h=0, *, replace_B=False):
    c = model.ctx
    ms, bs, ds, us = (model.replaced_constraints_by_h if replace_B else model.constraints_by_h)[h]
    R, t = V._pose(p, c.scale)
    q, _ = V.project_native(c.xyz, R, t, c.K)
    sides = _sides(ds, q, c.image_hw)
    return c.backend.P._joint_constraint_pair(p, c.xyz, c.K, ms, ds, sides,
                    c.image_hw, c.scale, c.cfg, bs, us)


def _old_components(model, p, consumed=(), *, outline=True):
    c = model.ctx
    consumed = set(consumed)
    ms = [m for m in c.measurements if m['native_corner'] not in consumed]
    bs = [b for b in c.boundary if b['native_corner'] not in consumed]
    value, gradient, parts = c.backend.P._joint_objective_pair(
        p, c.xyz, c.K, ms, c.scale, c.prior, c.cfg, bs, 1.)
    hull = 0.
    if outline and c.use_outline:
        hull, g, _ = c.backend.outline_value_and_gradient(p, c.xyz, c.K, c.scale, c.outline, c.outline_cfg)
        hull *= c.outline_cfg['outline_factor_weight']
        value += hull
        gradient += c.outline_cfg['outline_factor_weight']*g
    return value, gradient, dict(full=float(parts.get('soft_BI_robust_loss', 0.)),
        partial=float(parts.get('partial_boundary_normal_robust_loss', 0.)), outline=float(hull))


def _boundary_energy(model, q, kind='region'):
    if model.bundle is None:
        return 0., np.zeros((8,2))
    from . import boundary_model as B
    if hasattr(B, 'energy_and_gradient'):
        value, grad = B.energy_and_gradient(model.bundle, q, model=kind)
        return float(value), np.asarray(grad)
    value = B.candidate_energy(model.bundle, q, model=kind)
    grad = np.zeros((8,2))
    step = 1e-4
    for k in range(8):
        for axis in range(2):
            hi, lo = q.copy(), q.copy()
            hi[k,axis] += step; lo[k,axis] -= step
            grad[k,axis] = (B.candidate_energy(model.bundle,hi,model=kind)
                            -B.candidate_energy(model.bundle,lo,model=kind))/(2*step)
    return float(value), grad


def objective(model, p, mode, *, conditional_h=None, audit=False):
    """Scalar+analytic parameter gradient; all models evaluated on fixed RGB."""
    c = model.ctx
    q, depth, jac, _ = J.projection_and_jacobian(p,c.xyz,c.K,c.scale)
    if not np.isfinite(q).all():
        return 1e20, np.zeros(6), dict(invalid_projection=True)
    bundle = model.bundle
    consumed = set(bundle.get('consumed_native_ids', [])) if bundle else set()
    supported = bool(consumed)
    if mode in ('fixed','edge','B') and (mode=='fixed' or not supported):
        value, grad, parts = _old_components(model,p)
        return value, grad, dict(parts=parts, B_supported=False)
    if mode in ('B','edge'):
        base, grad, parts = _old_components(model,p,consumed,outline=False)
        extra, qgrad = _boundary_energy(model,q,'edge' if mode=='edge' else 'region')
        return base+extra, grad+np.einsum('ni,nij->j',qgrad,jac), dict(parts=parts,region=extra,B_supported=True)
    if mode not in ('A','AB'):
        raise ValueError('Unknown fixed scientific model: '+mode)
    replacements = []
    extras = np.zeros(4); extra_gradients = np.zeros((4,8,2))
    if mode=='AB' and supported:
        replacements = [g['group_id'] for g in model.owner['groups'] if g['native_corner'] in consumed]
        for h, perm in enumerate(model.owner['permutations']):
            extra, g = _boundary_energy(model,q[list(perm)],'region')
            extras[h] = extra
            for k, owner in enumerate(perm):
                extra_gradients[h,owner] += g[k]
    values, gradients, details = A.energies_per_global_h(model.owner,q,
        exclude_group_ids=replacements,extra_h_energies=extras,extra_h_gradients=extra_gradients,return_audit=True)
    if conditional_h is not None:
        value, qgrad = values[conditional_h], gradients[conditional_h]
    else:
        value, qgrad, ad = A.energy(model.owner,q,exclude_group_ids=replacements,
            extra_h_energies=extras,extra_h_gradients=extra_gradients,
            return_gradient=True,return_audit=True)
        details.update(ad)
    grad = np.einsum('ni,nij->j',qgrad,jac)
    hull = 0.
    if (mode=='A' or not supported) and c.use_outline:
        hull,g,_ = c.backend.outline_value_and_gradient(p,c.xyz,c.K,c.scale,c.outline,c.outline_cfg)
        hull *= c.outline_cfg['outline_factor_weight']; value += hull
        grad += c.outline_cfg['outline_factor_weight']*g
    return float(value),grad,dict(owner=details,outline=hull,B_supported=supported,
                                  replacements=replacements)


def candidate_scores(candidate, models):
    """All five models use the same bank and one score per actual pose."""
    results = {}
    for family, model in models.items():
        p = _native_pose(candidate,model.ctx)
        result = dict(family=family, scores={}, audits={}, h_min_slack={}, feasible={})
        for mode in ('fixed','A','B','AB','edge'):
            value,_,detail = objective(model,p,mode,audit=True)
            count = 4 if mode in ('A','AB') else 1
            slacks = [float(np.min(constraints(model,p,h,replace_B=mode in ('B','AB','edge'))[0]))
                      for h in range(count)]
            if mode in ('A','AB'):
                # A single globally consistent h must satisfy the hard image
                # constraints as well as explain the likelihood. Never use
                # one h's cheap residual with another h's feasible region.
                good = np.asarray(slacks) >= -model.ctx.cfg['feasibility_tolerance']
                values = np.asarray(detail['owner']['per_h_total_energies']
                    if 'per_h_total_energies' in detail['owner'] else
                    A.energies_per_global_h(model.owner,
                        J.projection_and_jacobian(p,model.ctx.xyz,model.ctx.K,model.ctx.scale)[0]),float)
                value = (float(-2*logsumexp(np.log(model.owner['normalized_global_priors'][good])
                                             -.5*values[good]))+detail['outline']) if good.any() else 1e20
                detail['jointly_feasible_global_h'] = good.tolist()
                posterior=np.zeros(len(good))
                if good.any():
                    logits=np.log(model.owner['normalized_global_priors'][good])-.5*values[good]
                    posterior[good]=np.exp(logits-logsumexp(logits))
                detail['owner']['global_owner_responsibilities']=posterior.tolist()
            result['scores'][mode] = float(value)
            result['audits'][mode] = detail
            result['h_min_slack'][mode] = slacks
            result['feasible'][mode] = max(slacks) >= -model.ctx.cfg['feasibility_tolerance']
        results[family] = result
    return results


def choose_starts(candidates, scored, family, scale, budget=4):
    """Same fixed-score first start, then farthest geometry diversity, all arms."""
    order = sorted(range(len(candidates)), key=lambda i:(scored[i][family]['scores']['fixed'],i))
    if not order:
        return []
    selected = [order[0]]
    while len(selected)<min(budget,len(order)):
        remaining = [i for i in order if i not in selected]
        index = max(remaining,key=lambda i:(min(pose_distance(candidates[i],candidates[j],scale)
                                               for j in selected),-order.index(i)))
        selected.append(index)
    # Up to four distinct owner problems still exist when an upstream stage
    # supplies only one numerical pose. Repeat the base in that case; the
    # explicit yaw proposal below, shared by ALL methods, makes four starts.
    original=list(selected)
    while len(selected)<budget:
        selected.append(original[len(selected)%len(original)])
    return selected


def owner_start_proposal(candidate,model,h):
    """Reference-free proper native-yaw proposal, identically used by controls.

    A quarter turn is an initialization, never an evaluation symmetry or a
    guarantee of correct rectangular owner geometry. Translation is centered.
    """
    c=model.ctx
    base_R=np.asarray(candidate['R_physical'])@c.Q.T
    yaw=Rotation.from_rotvec(np.array([0.,-h*np.pi/2,0.])).as_matrix()
    return dict(candidate_id=candidate['candidate_id']+f'/PROPOSED_NATIVE_YAW_{h}',
                source='PREDICTION_ONLY_YAW_START',R_physical=(base_R@yaw@c.Q).tolist(),
                centroid=deepcopy(candidate['centroid']),base_candidate_id=candidate['candidate_id'])


def solve_one(model, start, mode, h, config):
    before = time.monotonic()
    p0 = _native_pose(start,model.ctx)
    replace = mode in ('B','AB','edge')
    conditional = h if mode in ('A','AB') else None
    calls = dict(objective=0,constraint=0)
    def fg(p):
        calls['objective'] += 1
        v,g,_=objective(model,p,mode,conditional_h=conditional)
        return v,g
    def cons(p):
        calls['constraint'] += 1
        return constraints(model,p,h if conditional is not None else 0,replace_B=replace)
    result=minimize(fg,p0,jac=True,method='SLSQP',
        constraints=[dict(type='ineq',fun=lambda p:cons(p)[0],jac=lambda p:cons(p)[1])],
        options=dict(maxiter=config['max_iterations'],ftol=config['optimizer_ftol'],disp=False))
    pose=_actual_pose(result.x,model.ctx)
    slack=float(np.min(cons(result.x)[0]))
    feasible=(slack>=-model.ctx.cfg['feasibility_tolerance'] and np.isfinite(result.fun)
              and V._valid_rotation(np.asarray(pose['R_cf'])))
    return dict(candidate_id=f'NEW/{mode}/{model.ctx.family_name}/h{h}/{start["candidate_id"]}',
                source='NEW_'+mode,original_native_family=model.ctx.family_name,
                R_physical=pose['R_physical'],centroid=pose['centroid'],
                solve=dict(mode=mode,global_owner_h=h if conditional is not None else 0,
                    start_id=start['candidate_id'],start_pose=deepcopy(start),actual_pose=pose,
                    optimizer_success=bool(result.success),optimizer_status=int(result.status),
                    optimizer_message=str(result.message),iterations=int(result.nit),
                    objective=float(result.fun),minimum_constraint_slack=slack,feasible=bool(feasible),
                    calls=calls,elapsed_seconds=time.monotonic()-before))


def choose(candidates, scored, mode):
    eligible=[]
    for i,(candidate,score) in enumerate(zip(candidates,scored)):
        for family, s in score.items():
            if s['feasible'][mode] and np.isfinite(s['scores'][mode]):
                eligible.append((s['scores'][mode],i,family))
    return min(eligible,key=lambda x:(x[0],x[1],x[2])) if eligible else None


def prediction_from_choice(choice,candidates,frame,mode):
    if choice is None:
        return dict(actual_pose=deepcopy(frame['fallback']),fallback=True,
                    prediction_status='NO_FEASIBLE_OUTPUT_FALLBACK_N0',selected_candidate_id='N0',mode=mode)
    score,index,family=choice
    c=candidates[index]
    return dict(actual_pose=dict(available=True,R_physical=deepcopy(c['R_physical']),
                                 centroid=deepcopy(c['centroid'])),
                fallback=False,prediction_status='SELECTED',selected_candidate_id=c['candidate_id'],
                selected_family=family,selected_energy=score,mode=mode)


def generate_candidates(frame,config,*,bundle=None,full_modes=('fixed','A','B','AB','edge')):
    """One finite common pipeline; no references or frame-membership decisions."""
    started=time.monotonic()
    models=compile_frame(frame,config,bundle=bundle)
    scale=max(m.ctx.scale for m in models.values())
    carried=unique_poses(frame['candidates'],scale)
    carried_scores=[candidate_scores(c,models) for c in carried]
    predictions={}
    # One initial N0 per W/D family: fixed observations and fixed owners only.
    n0 = dict(candidate_id='N0_LOCAL_START', source='N0',
              R_physical=frame['fallback']['R_physical'],centroid=frame['fallback']['centroid'])
    fixed_attempts=[]
    for family,model in models.items():
        try:
            fixed_attempts.append(solve_one(model,n0,'fixed',0,config))
        except (ValueError,RuntimeError,FloatingPointError,OverflowError) as exc:
            fixed_attempts.append(dict(candidate_id='FAILED/CTRL_FIXED/'+family,
                source='NEW_fixed',solve=dict(feasible=False,exception=repr(exc),start_id='N0_LOCAL_START')))
    fixed_valid=[c for c in fixed_attempts if c['solve']['feasible']]
    fixed_scores=[candidate_scores(c,models) for c in fixed_valid]
    predictions['CTRL_FIXED']=prediction_from_choice(choose(fixed_valid,fixed_scores,'fixed'),fixed_valid,frame,'fixed')
    for arm,mode in [('CTRL_CARRY','fixed'),('A_FIXED_BANK','A'),('B_FIXED_BANK','B')]:
        predictions[arm]=prediction_from_choice(choose(carried,carried_scores,mode),carried,frame,mode)
    banks={}; full_scores={}; attempts=list(fixed_attempts)
    names=dict(fixed='CTRL_BUDGET',A='A_FULL',B='B_FULL',AB='AB_FULL',edge='CTRL_EDGE_BOUNDARY')
    starts={family:choose_starts(carried,carried_scores,family,scale,config['starts_per_family'])
            for family in models}
    for mode in full_modes:
        new=[]
        for family,model in models.items():
            for j,index in enumerate(starts[family]):
                try:
                    proposal=owner_start_proposal(carried[index],model,j%4)
                    result=solve_one(model,proposal,mode,j%4,config)
                    attempts.append(result)
                    if result['solve']['feasible']:
                        new.append(result)
                except (ValueError,RuntimeError,FloatingPointError,OverflowError) as exc:
                    attempts.append(dict(candidate_id=f'FAILED/{mode}/{family}/{j}',
                        source='NEW_'+mode,solve=dict(feasible=False,exception=repr(exc),
                        start_id=carried[index]['candidate_id'],global_owner_h=j%4)))
        bank=unique_poses(carried+new,scale)
        scores=carried_scores+[candidate_scores(c,models) for c in bank[len(carried):]]
        banks[mode]=bank;full_scores[mode]=scores
        predictions[names[mode]]=prediction_from_choice(choose(bank,scores,mode),bank,frame,mode)
    predictions['CTRL_RANSAC']=ransac_control(frame,models,config)
    return dict(id=frame['id'],partition=frame['partition'],session=frame['session'],
                cluster_id=frame['cluster_id'],predictions=predictions,
                carried=carried,carried_scores=carried_scores,banks=banks,bank_scores=full_scores,
                attempts=attempts,starts={k:[carried[i]['candidate_id'] for i in v] for k,v in starts.items()},
                elapsed_seconds=time.monotonic()-started,
                configured_max_new_solves_per_arm=2*config['starts_per_family'],
                actual_new_solve_attempts=len(attempts),reference_inputs=False)


def ransac_control(frame,models,config):
    """OpenCV standard fixed-owner control; partials never become full points.

    100 iterations / 8px / .99 confidence are the official OpenCV defaults,
    fixed before the synthetic pilot. No inlier-threshold search is used.
    """
    candidates=[];attempts=[]
    for family,model in models.items():
        c=model.ctx
        revoked={b['native_corner'] for b in c.boundary}
        ms=[m for m in c.measurements if m['native_corner'] not in revoked]
        if len(ms)<4:
            attempts.append(dict(family=family,status='UNSUPPORTED_FEWER_THAN_FOUR_FULL_POINTS',full_points=len(ms)))
            continue
        cv2.setRNGSeed(20261018)
        try:
            ok,rv,tv,inliers=cv2.solvePnPRansac(np.ascontiguousarray([c.xyz[m['native_corner']] for m in ms]),
                np.ascontiguousarray([m['xy'] for m in ms]),c.K,None,
                iterationsCount=100,reprojectionError=8.,confidence=.99,flags=cv2.SOLVEPNP_EPNP)
        except cv2.error as exc:
            attempts.append(dict(family=family,status='FAILED_TECHNICAL',exception=repr(exc),full_points=len(ms)))
            continue
        attempts.append(dict(family=family,status='SOLVED' if ok else 'NO_SOLUTION',
                             full_points=len(ms),inliers=0 if inliers is None else len(inliers)))
        if ok:
            R=cv2.Rodrigues(rv)[0]; t=np.asarray(tv).reshape(3)
            candidates.append(dict(candidate_id='RANSAC/'+family,source='PNP_RANSAC',
                R_physical=(R@c.Q).tolist(),centroid=(R@c.xyz.mean(0)+t).tolist(),original_native_family=family))
    scores=[candidate_scores(c,models) for c in candidates]
    pred=prediction_from_choice(choose(candidates,scores,'fixed'),candidates,frame,'fixed')
    pred['ransac_attempts']=attempts
    return pred


def _finite_feature(x):
    return float(x) if x is not None and np.isfinite(x) and abs(x)<1e15 else None


def features_for_bank(frame,models,bank,scores):
    """Same permutation-invariant, reference-free features for synthetic/real.

    Returned features include an explicit mask for every missing scalar. Log
    transforms use no data-fit parameters. Train-only normalization is later
    applied identically to both learning models.
    """
    if not bank:
        return []
    minima={}
    for mode in ('fixed','A','B','AB','edge'):
        valid=[s[family]['scores'][mode] for s in scores for family in models
               if _finite_feature(s[family]['scores'][mode]) is not None]
        minima[mode]=min(valid) if valid else None
    result=[]
    for candidate,all_scores in zip(bank,scores):
        # Context is selected by the original common score, never by a label.
        family=min(models,key=lambda f:(all_scores[f]['scores']['fixed'],f))
        model=models[family];c=model.ctx;s=all_scores[family]
        p=_native_pose(candidate,c)
        q,depth,_,_=J.projection_and_jacobian(p,c.xyz,c.K,c.scale)
        revoked={b['native_corner'] for b in c.boundary}
        full=[m for m in c.measurements if m['native_corner'] not in revoked]
        errors=[np.linalg.norm(q[m['native_corner']]-m['xy']) for m in full]
        normal=[]
        for b in c.boundary:
            for line in b['lines']:
                normal.append(abs(np.dot(line['normal_unit_xy'],q[b['native_corner']])+line['offset_px'])
                              /np.sqrt(line['normal_variance_px2']))
        def owner_detail(mode):
            a=s['audits'][mode].get('owner',{})
            mass=np.asarray(a.get('global_owner_responsibilities',[.25]*4),float)
            entropy=-float(np.sum(mass[mass>0]*np.log(mass[mass>0])))
            none=np.asarray(a.get('per_h_none_responsibilities',[]),float)
            return entropy, float(none.mean()) if none.size else None
        ah,none=owner_detail('A');abh,_=owner_detail('AB')
        bundle=model.bundle or {}
        profiles=[q for g in bundle.get('groups',[]) for q in g['queries'] if q['supported']]
        consumed=bundle.get('consumed_native_ids',[])
        entropies=[]
        for profile in profiles:
            if 'offset_probability' in profile:
                prob=np.asarray(profile['offset_probability'],float)
                entropies.append(float(-np.sum(prob[prob>0]*np.log(prob[prob>0]))))
        base=s['audits']['fixed'].get('parts',{})
        sigmas=[np.sqrt(np.trace(m['covariance_px2'])/2) for m in full]
        dN=np.linalg.norm(np.asarray(candidate['centroid'])-np.asarray(frame['fallback']['centroid']))/c.scale
        rN=Rotation.from_matrix(np.asarray(candidate['R_physical'])@np.asarray(frame['fallback']['R_physical']).T).magnitude()
        raw=[s['scores'][k] for k in ('fixed','A','B','AB','edge')]
        raw += [base.get(k) for k in ('full','partial','outline')]
        raw += [ah,none,abh,len(full),len(c.boundary),len(consumed),len(profiles),
                float(np.mean(entropies)) if entropies else None,
                float(np.mean([m.get('source')=='VISIBLE_BACKBONE_FALLBACK' for m in full])) if full else None,
                float(np.mean(sigmas)) if sigmas else None,
                float(np.mean(errors)) if errors else None,
                float(np.quantile(errors,.9)) if errors else None,
                float(np.mean(normal)) if normal else None,
                float(dN),float(rN),max(s['h_min_slack']['AB']),float(np.mean(depth>0)),
                len(consumed)/max(1,len(c.measurements))]
        raw += [s['scores'][mode]-minima[mode] if minima[mode] is not None and
                _finite_feature(s['scores'][mode]) is not None else None
                for mode in ('fixed','A','B','AB','edge')]
        assert len(raw)==len(FEATURE_NAMES)
        values=[_finite_feature(x) for x in raw]
        transformed=[0. if x is None else np.sign(x)*np.log1p(abs(x)) for x in values]
        mask=[float(x is not None) for x in values]
        result.append(dict(candidate_id=candidate['candidate_id'],features=transformed+mask,
                           actual_pose=dict(available=True,R_physical=candidate['R_physical'],
                                            centroid=candidate['centroid']),feature_context_family=family))
        result[-1]['eligible'] = bool(candidate.get('source') in ('N0','E0') or any(
            score['feasible'][mode] for score in all_scores.values()
            for mode in ('fixed','A','B','AB','edge')))
        result[-1]['baseline_alternative_not_joint_certificate'] = candidate.get('source') in ('N0','E0')
    return result
