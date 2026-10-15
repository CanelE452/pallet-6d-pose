"""Independent selected-pose v29 scalar reconstruction.

The frozen native parser/old guard primitives are reused. The new consumer,
its ownership planner, its objective, and its support-constraint code are
never called to certify a selected energy. No RGB, labels or fitting occurs.
"""
from __future__ import annotations

from collections import Counter
from copy import copy
import hashlib
import json
from itertools import zip_longest
from pathlib import Path

import numpy as np

from . import native_edge_problem_v4_17 as N
from . import independent_geometry_edge_energy_v4_29 as KERNEL
from . import common as C

FALLBACK = 'VISIBLE_BACKBONE_FALLBACK'


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def observation_checks(task, observation, plan, noise):
    """Check exact saved support/floors/order against the original archive."""
    checks=[]
    refs=observation['selected_chain_refs']
    selected={_digest(r) for r in refs}
    records=observation['chain_observation']['edge_records']
    nodes={n['name']:n for n in observation['joint_graph']['line_candidates']}
    measurements={m['native_corner']:m for m in task['spec']['visible']}
    sites=set()
    for group in plan['active_groups']:
        ref=group['selected_chain_ref']
        checks.append((_digest(ref) in selected,'active edge bound to original graph winner'))
        ray=records[ref['ray_index']];chain=ray['chains'][ref['chain_index']]
        k,mate=ray['native_corner'],ray['native_neighbor']
        if task['initial_self_hidden'][k] or task['initial_self_hidden'][mate]:
            raise ValueError('S edge gate before original archive payload')
        checks.append((group['native_corner']==k and group['native_neighbor']==mate,'exact native edge IDs'))
        xy=np.asarray(chain['observed_points']);mask=np.asarray(chain['inlier_mask'],bool)
        actual=xy[mask]
        lattice=np.asarray(chain['actual_inlier_image_lattice_xy'])
        gradient=np.asarray(chain['image_gradient_normal_xy']);shift=np.asarray(chain['measured_subpixel_shift_px'])
        # Frozen NMS localization floor is one pixel.
        reconstructed=lattice+gradient[mask]*shift[mask,None]
        checks.append((np.allclose(actual,reconstructed,atol=1e-7,rtol=0),'original NMS lattice and subpixel support'))
        checks.append((np.array_equal(actual,group['actual_inlier_xy']) and
                       np.array_equal(lattice,group['actual_inlier_lattice_xy']),'exact actual support without refit'))
        normal=np.asarray(chain['normal'],float);offset=float(chain['offset'])
        if normal[np.argmax(np.abs(normal))]<0:normal=-normal;offset=-offset
        checks.append((np.array_equal(normal,group['normal_unit_xy']) and offset==group['offset_px'],
                       'exact observed canonical TLS coefficients'))
        node=nodes[ref['line_name']]
        checks.append((np.array_equal(node['actual_inliers'],actual),'graph winner actual support'))
        axis=np.array([-normal[1],normal[0]]);coordinates=actual@axis
        span=float(np.ptp(coordinates));rms=float(np.sqrt(np.mean((actual@normal+offset)**2)))
        fn=max(1.,rms);fs2=2.*fn*fn/(span*span+2.*fn*fn)
        anchor=actual.mean(axis=0)-normal*(normal@actual.mean(axis=0)+offset)
        checks.append((span>=11.,'original finite support at least eleven pixels'))
        checks.append((np.allclose(anchor,group['support_anchor_xy'],atol=1e-12,rtol=0),
                       'weighted TLS support anchor projected onto measured line'))
        checks.append((np.isclose(group['normal_variance_px2'],noise['normal_variance_component_px2']+fn*fn,atol=1e-10,rtol=1e-12)
                       and np.isclose(group['angular_variance_rad2'],noise['angular_variance_component_sine2']+fs2,atol=1e-12,rtol=1e-12),
                       'new CAL components and actual engineering floors once'))
        checks.append((group.get('normal_angle_correlation',0.)==0.,'declared zero normal angle correlation'))
        guide=np.asarray(chain['generating_guide_direction_xy']);reference=ref['guide_reference']
        digest_fields={key:value for key,value in reference.items() if key!='guide_reference_id'}
        checks.append((_digest(digest_fields)==ref['guide_reference_id']==chain['generating_guide_reference_id']
                       and reference.get('guide_reference_id',ref['guide_reference_id'])==ref['guide_reference_id'],
                       'exact generating guide digest'))
        checks.append((np.array_equal(guide,reference['actual_search_direction_xy']), 'exact generating search direction'))
        role='TARGET_BACKGROUND_OUTLINE' if task['states'][k]=='B' else 'TARGET_TARGET_INTERNAL'
        checks.append((chain['mask_context_kind']==group['mask_context_kind']==role,'actual mask role'))
        if role=='TARGET_BACKGROUND_OUTLINE':
            context=node['global_external_context']
            checks.append((context['available'] and context['actual_support_near_global_external_fraction']>=.6
                           and context['same_foreground_component_fraction']>=.6,'global exterior support excludes internal holes'))
        current_sites={tuple(p) for p in lattice}
        checks.append((not bool(sites&current_sites),'no cross-native reuse of same image lattice support'))
        sites|=current_sites
        for owner in group['activation_owners']:
            corner=owner['native_corner'];other=mate if corner==k else k
            if other in measurements:
                direction=np.asarray(measurements[other]['xy'])-np.asarray(measurements[corner]['xy'])
                provenance='IMMUTABLE_BI_ENDPOINT_HEMISPHERE'
            else:
                direction=guide if corner==k else -guide
                provenance='SELECTED_GENERATING_GUIDE_UNIQUE_ORDER'
            dot=float(axis@direction);sign=None if abs(dot)<=1e-12 else (1 if dot>0 else -1)
            checks.append((sign==owner['fixed_tangent_sign'] and
                (sign is None or owner['order_provenance']==provenance),'order sign from immutable image evidence'))
    return checks


def prepare_raw(task, family):
    spec = task['spec']
    return N.prepare_native_edge_problem(family['native_xyz'], spec['K'], spec['states'],
        spec['visible'], spec['domains'], spec['image_hw'], family['coarse_pose'],
        config={**spec['config'], 'prior_weight':0.},
        boundary_observations=task['boundary_merged'], u_regions=task['u_regions'],
        edge_bundle=dict(edges=[], replaces_normal_factor_refs=[]))


def recompute_native(task, family, R, t, plan, tangent_gamma, *, raw=None):
    """Manual factor sum and independently filtered original/new guards."""
    raw = prepare_raw(task, family) if raw is None else raw
    projected, depth = N.V.project_native(raw.xyz, np.asarray(R), np.asarray(t), raw.K)
    partial = {b['native_corner'] for b in raw.boundary}
    source = {m['native_corner']:m['source'] for m in task['spec']['visible']}
    parsed = {m['native_corner']:m for m in raw.measurements}
    restored = {k for k in partial if source[k] != FALLBACK}
    owners = set(plan['new_owner_ids'])
    groups = plan['active_groups']
    corners = {p['native_corner']:p for p in plan['corner_plans']}
    checks = []
    checks.append((owners == set(corners), 'every new owner has one corner plan'))
    checks.append((owners == {o['native_corner'] for g in groups for o in g['activation_owners']},
                   'active physical edges and owner ledger agree'))
    pairs = [tuple(sorted([g['native_corner'],g['native_neighbor']])) for g in groups]
    checks.append((len(pairs) == len(set(pairs)), 'one factor per physical edge'))
    for k in owners:
        checks.append((not task['initial_self_hidden'][k] and task['states'][k] in ('B','I'),
                       'nonS BI gate before owner archive'))
        if task['initial_self_hidden'][k] or task['states'][k] not in ('B','I'):
            raise ValueError('Only nonS BI owner archives may be read')
        if plan['source_policy'] == 'RGB_NONFALLBACK_PRESERVE':
            checks.append((source[k] == FALLBACK, 'source preserving policy only replaces exact fallback'))
        archive = corners[k]['source_archive'];measurement = parsed[k]
        for key, value in (
                ('original_measurement_source',source[k]),
                ('original_observed_xy',measurement['xy']),
                ('original_observed_covariance_px2',measurement['observed_covariance_px2']),
                ('original_effective_fit_covariance_px2',measurement['covariance_px2']),
                ('original_fit_reliability_weight',measurement['fit_reliability_weight'])):
            checks.append((np.array_equal(archive[key],value), 'immutable source archive '+key))
        incident = [g for g in groups if any(o['native_corner']==k for o in g['activation_owners'])]
        checks.append((set(corners[k]['selected_component_ids']) == {g['component_id'] for g in incident},
                       'owner selected component IDs exact'))
        checks.append((corners[k]['new_precise_KP'] is False, 'no inferred precise KP'))

    costs = []
    delta = raw.cfg['huber_scale']
    for measurement in raw.measurements:
        k = measurement['native_corner']
        if k not in owners and (k not in partial or k in restored):
            costs.append(dict(kind='COMPLETE_BI', native_corner=k,
                weighted_robust_cost=KERNEL.point_energy(projected,measurement,delta)))
    old = [b for b in raw.boundary if b['native_corner'] not in owners|restored]
    for boundary in old:
        costs.append(dict(kind='UNREPLACED_PARTIAL_NORMAL', native_corner=boundary['native_corner'],
            weighted_robust_cost=KERNEL.legacy_normal_energy(projected,boundary,delta)))
    # The legacy first-line archive parser remains unchanged. Its scalar
    # energy is independently recomputed, with original variance inflation4.
    for record in N.T._tangent_records(old,4.):
        k = record['native_corner']
        costs.append(dict(kind='UNCERTAIN_ARCHIVED_TANGENT', native_corner=k,
            weighted_robust_cost=KERNEL.tangent_energy(projected,k,record['original_xy'],
                record['tangent'],record['variance_px2'],record['reliability'],delta)))
    support = []
    for group in groups:
        endpoints = {group['native_corner'], group['native_neighbor']}
        checks.append((all(task['states'][k]!='S' for k in endpoints), 'no S image edge endpoint'))
        checks.append((np.count_nonzero(np.abs(raw.xyz[group['native_corner']]-raw.xyz[group['native_neighbor']])>1e-9)==1,
                       'named physical cuboid edge'))
        expected_weight = min(parsed[o['native_corner']]['fit_reliability_weight'] for o in group['activation_owners'])
        checks.append((group['fit_reliability_weight']==expected_weight, 'physical edge reliability once'))
        value = KERNEL.actual_line_energy(projected,group,delta)
        costs.append(dict(kind='PROJECTED_NATIVE_EDGE', native_corner=group['native_corner'],
            native_neighbor=group['native_neighbor'], weighted_robust_cost=value['energy']))
        support.append(dict(component_id=group['component_id'],
            **KERNEL.original_support_slacks(projected,group)))
    for k, corner in corners.items():
        incident = [g for g in groups if g['component_id'] in corner['selected_component_ids']]
        if corner['mode'] == 'ONE_SELECTED_LINE_ARCHIVED_Q_TANGENT':
            checks.append((len(incident)==1, 'one line preserves exactly one q tangent'))
            normal = np.asarray(incident[0]['normal_unit_xy']);axis=np.array([-normal[1],normal[0]])
            checks.append((np.array_equal(axis,corner['selected_line_tangent_unit_xy']),
                           'tangent is immutable measured line direction'))
            measurement=parsed[k]
            a=float(axis@measurement['covariance_px2']@axis)
            variance=1.+float(tangent_gamma)*a
            costs.append(dict(kind='ARCHIVED_Q_SELECTED_MEASURED_LINE_TANGENT', native_corner=k,
                weighted_robust_cost=KERNEL.tangent_energy(projected,k,measurement['xy'],axis,
                    variance,measurement['fit_reliability_weight'],delta)))
        else:
            checks.append((corner['mode']=='TWO_DIRECTION_LINES_NO_Q_COST' and len(incident)==2,
                           'rank2 removes q using exactly two lines'))
            if len(incident)==2:
                normals=np.asarray([g['normal_unit_xy'] for g in incident])
                sine=abs(float(np.linalg.det(normals)))
                checks.append((sine>=.25, 'rank2 independent measured directions'))
                if sine>=.25:
                    junction=np.linalg.solve(normals,-np.asarray([g['offset_px'] for g in incident]))
                    h,w=raw.image_hw
                    checks.append((0<=junction[0]<w and 0<=junction[1]<h, 'rank2 finite canvas intersection'))
                    for group in incident:
                        n=np.asarray(group['normal_unit_xy']);axis=np.array([-n[1],n[0]])
                        coordinates=np.asarray(group['actual_inlier_xy'])@axis
                        coordinate=float(axis@junction)
                        gap=max(float(coordinates.min())-coordinate,coordinate-float(coordinates.max()),0.)
                        checks.append((gap<=11.+1e-9, 'rank2 original actual finite support'))
                        owner=next(o for o in group['activation_owners'] if o['native_corner']==k)
                        sign=owner.get('fixed_tangent_sign')
                        fraction=None if sign is None else float(np.mean(sign*(coordinates-coordinate)>=-1.))
                        checks.append((fraction is not None and fraction>=.6, 'rank2 ordered actual support'))
    # Keep original visible/depth/OT/U geometry; remove only old partial
    # locality of new owners. Reuse frozen guard primitives, not new wrapper.
    guard = copy(raw)
    guard.boundary=[b for b in raw.boundary if b['native_corner'] not in owners]
    guarded=guard.evaluate(R,t)
    support_slacks=[x for record in support for x in record['coverage_slacks']]
    support_slacks += [x for record in support for d in record['directed'] for x in d['slacks']]
    minimum_support=float(min(support_slacks,default=np.inf))
    accepted=bool(guarded['accepted'] and minimum_support>=-raw.cfg['feasibility_tolerance'])
    return dict(objective=float(sum(c['weighted_robust_cost'] for c in costs)),native_costs=costs,
        checks=checks,accepted=accepted,old_guard_audit=guarded,actual_support=support,
        minimum_actual_support_slack=minimum_support,projected_native=projected,
        minimum_depth=float(np.min(depth)),restored_ids=sorted(restored),owner_ids=sorted(owners),raw=raw)


def cost_buckets(records):
    totals=Counter()
    for record in records:
        totals[(record['kind'],record.get('native_corner'),record.get('native_neighbor'))]+=record['weighted_robust_cost']
    return dict(totals)


def main(output=None):
    # Reuse only unchanged old O/S and C scalar primitives and model paths.
    from . import audit_source_preserving_tangent_mask_v4_24 as OLD_AUDIT
    from . import global_foreground_containment_v4_25 as CONTAINMENT
    from . import geometry_guided_edge_noise_calibration_v4_29 as NOISE
    from . import evaluate_prepartial_RGB_preservation_v4_26 as OLD_RUN
    B=OLD_RUN.OUTPUT.parent
    output=Path(B/'geometry_guided_edge_eval_v4_29' if output is None else output).resolve()
    dest=output/'independent_root_audit_v1'
    if dest.exists():raise FileExistsError('Preserve all audit attempts')
    arms=('SAME_FROZEN_E','GEOMETRY_EDGE_ALL_BI_NATIVE','GEOMETRY_EDGE_ALL_BI_CONTAINMENT',
          'GEOMETRY_EDGE_SOURCE_NATIVE','GEOMETRY_EDGE_SOURCE_CONTAINMENT')
    policy={arms[1]:'ALL_BI_REPLACE',arms[2]:'ALL_BI_REPLACE',
            arms[3]:'RGB_NONFALLBACK_PRESERVE',arms[4]:'RGB_NONFALLBACK_PRESERVE'}
    protocol=C.read(output/'PROTOCOL.json');inputseal=C.read(output/'INPUT_SEAL.json')
    seal=C.read(output/'PREDICTION_SEAL.json')
    if (seal['rows']!=238 or protocol['arms']!=list(arms)
            or C.sha(output/'PREDICTIONS.jsonl.gz')!=seal['predictions_sha256']
            or C.sha(output/'PROTOCOL.json')!=seal['protocol_sha256']
            or C.sha(output/'INPUT_SEAL.json')!=seal['input_seal_sha256']
            or C.sha(output/'INPUTS.jsonl.gz')!=inputseal['input_sha256']):
        raise ValueError('Exact all238 input/prediction/source seals required')
    verified={}
    for kind in ('source_sha256','input_sha256','protected_user_sha256'):
        verified[kind]=0
        for relative,digest in protocol[kind].items():
            if C.sha(C.ROOT/relative)!=digest:raise ValueError('Frozen binding changed: '+relative)
            verified[kind]+=1
    models={}
    for p in set(policy.values()):
        folder=B/'CAL_geometry_guided_edge_noise_v4_29'/p
        line=C.read(folder/'MODEL.json');tangent=C.read(folder/'TANGENT_MODEL.json')
        unavailable=line.get('status','').startswith('MODEL_UNAVAILABLE') or tangent.get('status','').startswith('MODEL_UNAVAILABLE')
        models[p]=dict(available=not unavailable)
        if not unavailable:
            models[p].update(noise=NOISE.mixture_noise_parameters(line,source_policy=p),
                             gamma=NOISE.tangent_variance_parameters(tangent,source_policy=p)['gamma'])
    omodel=C.read(OLD_AUDIT.OFIT.OUTPUT/'MODEL.json')
    smodel=C.read(OLD_AUDIT.SFIT.OUTPUT/'MODEL.json')
    oparams=OLD_AUDIT.OFIT.prior_parameters(omodel)
    total_checks=0;failures=[];maximum=0.;counts=Counter();accepted=0;fallback=0;rows=0
    observations=[]
    def check(condition,reason,ident,arm=None):
        nonlocal total_checks
        total_checks+=1
        if not bool(condition):failures.append(dict(id=ident,arm=arm,reason=reason))
    def near(first,second):
        nonlocal maximum
        gap=abs(float(first)-float(second));maximum=max(maximum,gap)
        return gap<=1e-8*max(1.,abs(float(first)),abs(float(second)))
    dest.mkdir(parents=True)
    for task,row,previous in zip_longest(C.rows(output/'INPUTS.jsonl.gz'),C.rows(output/'PREDICTIONS.jsonl.gz'),
                                        C.rows(OLD_RUN.OUTPUT/'PREDICTIONS.jsonl.gz')):
        if task is None or row is None or previous is None:raise ValueError('Exact ordered238 journals required')
        rows+=1;ident=task['id'];counts[task['partition']]+=1
        check(ident==row['id']==previous['id']==protocol['ids'][rows-1],'ordered238',ident)
        check(set(row['arms'])==set(arms),'all declared five arms',ident)
        check(row['arms'][arms[0]]==task['baseline'],'exact frozen e baseline',ident)
        check(row['states']==task['states']==task['spec']['states'],'unchanged initial states',ident)
        first=task['families']['ORIGINAL_WD'];coarse=first['coarse_pose']
        sprepared=OLD_AUDIT.S.prepare_bounded_coarse_visibility_prior(first['native_xyz'],coarse['R'],coarse['t'],
            np.asarray(task['initial_self_hidden'],bool),smodel)
        regions=OLD_AUDIT.O.prepare_o_localization_regions(
            [r['original_domain'] for r in task['O_localization_abstention']['records']],
            task['states'],task['initial_self_hidden'])
        cfg=OLD_AUDIT.MASK_CORE._outline_config(task['outline']['configuration'])
        supported=bool(task['outline'].get('supported'));weight=cfg['outline_factor_weight'] if supported else 0.
        old_searches=previous['bounded_prior_refinement_audit']['PREPARTIAL_RGB_NATIVE']['family_refinements']
        raw_cache={}
        for arm in arms[1:]:
            result=row['arms'][arm];solution=result['solution'];family=task['families'][result['WD_family']]
            # Frozen v17 exporter stores native t under the legacy name
            # "centroid". This experiment uses centered native XYZ, for which
            # it also equals the camera-frame physical model center.
            pose=result['actual_pose'];R=np.asarray(pose['R_cf']);t=np.asarray(pose['centroid'])
            check(np.allclose(np.asarray(family['native_xyz']).mean(axis=0),0.,atol=1e-12,rtol=0),
                  'current centered native XYZ required by legacy centroid field',ident,arm)
            current_policy=policy[arm];containment=arm.endswith('_CONTAINMENT')
            check(result['source_policy']==current_policy,'fixed source policy',ident,arm)
            check(np.allclose(R@family['physical_registration_Q'],pose['R_physical'],atol=1e-12,rtol=0),
                  'fixed native physical registration',ident,arm)
            searches=row['bounded_prior_refinement_audit'][arm]['family_refinements']
            check(set(searches)==set(task['families']),'both native WD hypotheses retained',ident,arm)
            for name,search in searches.items():
                check(search['start_selection']['selected_start_names']==old_searches[name]['start_selection']['selected_start_names'],
                      'exact original v19 native start names',ident,arm)
            plan=solution['v29_cost_plan'];owners=set(plan['new_owner_ids'])
            check(plan['source_policy']==current_policy,'selected cost ownership policy',ident,arm)
            q=C.project(family['native_xyz'],R,t,task['spec']['K'])
            readout=result['keypoint_readout']
            for k,state in enumerate(task['states']):
                point=readout['reported_points'][k]
                if state=='S':check(np.allclose(point,q[k],atol=1e-8,rtol=0),'S model projection only',ident,arm)
                if state=='U' or k in owners:check(point is None,'U/new line owner no precise point readout',ident,arm)
                if state in ('O','T'):
                    check((result['accepted'] and np.allclose(point,q[k],atol=1e-8,rtol=0)
                           and readout['point_sources'][k]=='FINITE_FEASIBLE_CANDIDATE_PROJECTION_NOT_UNIQUE')
                          or (not result['accepted'] and point is None),
                          'O/T only explicit nonunique model projection; fallback unresolved',ident,arm)
            check(readout['OT_solved'] is False and readout['OT_unique_solution_claimed'] is False,
                  'O/T model projection is not precise image observation or unique solution',ident,arm)
            if not result['accepted']:
                fallback+=1
                check(pose['fallback'] and result['WD_family']=='ORIGINAL_WD' and np.array_equal(R,coarse['R'])
                      and np.array_equal(t,coarse['t']) and result['fallback_audit']['old_best_pose_fallback'] is False,
                      'own original initial coarse fallback only',ident,arm)
                continue
            accepted+=1
            if result['WD_family'] not in raw_cache:raw_cache[result['WD_family']]=prepare_raw(task,family)
            raw=raw_cache[result['WD_family']];parameters=models[current_policy]
            if not parameters['available']:
                check(not owners and not plan['active_groups'],'unavailable new CAL causes uniform no-line abstention',ident,arm)
                gamma=0.
            else:
                gamma=parameters['gamma']
                for passed,reason in observation_checks(task,task['v29_observation'],plan,parameters['noise']):
                    check(passed,reason,ident,arm)
            fresh=recompute_native(task,family,R,t,plan,gamma,raw=raw)
            for passed,reason in fresh['checks']:check(passed,reason,ident,arm)
            check(fresh['accepted'],'independent old and new actual support guards',ident,arm)
            check(near(fresh['objective'],solution['native_observation_objective']),'independent native energy',ident,arm)
            first_cost=cost_buckets(fresh['native_costs']);stored=cost_buckets(solution['native_costs'])
            check(first_cost.keys()==stored.keys() and all(near(first_cost[k],stored[k]) for k in first_cost),
                  'all native factor identities and scalar costs once',ident,arm)
            p=N.V._pose_vector(R,t,raw.scale)
            ov,_,oa=OLD_AUDIT.O.bounded_o_localization_value_and_gradient(p,raw.xyz,raw.K,raw.scale,regions,**oparams)
            rebound=OLD_AUDIT.S.rebind_bounded_visibility_trial_geometry(sprepared,raw.xyz)
            sv,_,sa=OLD_AUDIT.S.coarse_visibility_prior_value_and_gradient(p,raw.xyz,raw.scale,rebound)
            expected_kinds=['O_COARSE_LOCALIZATION_PRIOR','S_COARSE_VISIBILITY_PRIOR']
            priors=solution['regularizer_costs']
            check([r['kind'] for r in priors]==expected_kinds,'only frozen O/S priors, no TP/full pose prior',ident,arm)
            values={r['kind']:r['weighted_cost'] for r in priors}
            check(near(values[expected_kinds[0]],ov) and near(values[expected_kinds[1]],sv),
                  'independent bounded O/S costs',ident,arm)
            check(oa['O_point_terms']==0 and sa['S_image_or_2D_terms']==0,'no O/S hidden image points',ident,arm)
            shape=0.
            if containment and supported and weight>0:
                shape,_,ca=CONTAINMENT.outline_value_and_gradient(p,raw.xyz,raw.K,raw.scale,task['outline'],cfg)
                check(ca['valid_projection'] and ca['model_to_observed_factor_weight']==0.
                      and ca['inside_model_hull_cost_zero'],'one sided foreground containment only',ident,arm)
            total=fresh['objective']+ov+sv+(weight*shape if containment else 0.)
            check(near(solution['regularizer_objective'],ov+sv) and near(solution['objective'],total),
                  'independent total N+O+S+C once',ident,arm)
            selection=result['candidate_selection']
            check(selection['old_full_mask_cost_or_numeric_tie_break_used'] is False
                  and selection['e_numeric_pose_prior_or_fallback'] is False,'no old full mask or e numeric routing',ident,arm)
            chosen=[r for r in selection['candidates'] if r['name']==selection['selected_name']]
            check(len(chosen)==1,'one selected contender',ident,arm)
            if chosen:
                chosen=chosen[0];minimum=min(r['total_energy'] for r in selection['candidates'])
                tolerance=32*np.finfo(float).eps*max(1.,abs(minimum),abs(chosen['total_energy']))
                check(near(chosen['native_energy'],fresh['objective']) and near(chosen['regularizer_energy'],ov+sv)
                      and near(chosen['weighted_containment_objective'],weight*shape if containment else 0.)
                      and near(chosen['total_energy'],total),'selected optimizer and selector identical components',ident,arm)
                check(chosen['total_energy']<=minimum+tolerance,'minimum stored feasible contender total',ident,arm)
            observations.append(dict(id=ident,arm=arm,native=fresh['objective'],O=ov,S=sv,
                weighted_containment=weight*shape if containment else 0.,total=total,
                owner_ids=fresh['owner_ids'],minimum_actual_support_slack=fresh['minimum_actual_support_slack']))
        if rows%20==0:print(dict(rows=rows,checks=total_checks,failures=len(failures)),flush=True)
    check(counts==Counter(CAL=46,TEST=49,REMAINDER=143),'all238 population',None)
    for kind in verified:
        for relative,digest in protocol[kind].items():
            if C.sha(C.ROOT/relative)!=digest:raise ValueError('Binding changed during audit: '+relative)
    C.save_rows(dest/'SELECTED_POSE_COMPONENTS.jsonl.gz',observations)
    report=dict(schema='INDEPENDENT_SELECTED_GEOMETRY_EDGE_AUDIT_V29',rows=rows,checks=total_checks,
        failure_count=len(failures),failures=failures,partition_counts=dict(counts),accepted_arms=accepted,
        fallback_arms=fallback,maximum_absolute_scalar_difference=maximum,verified_bindings=verified,
        prediction_seal_sha256=C.sha(output/'PREDICTION_SEAL.json'),
        prediction_seal=dict(path=str((output/'PREDICTION_SEAL.json').relative_to(C.ROOT)),sha256=C.sha(output/'PREDICTION_SEAL.json')),
        independent_manual_new_line_tangent_and_support_reconstruction=True,
        original_frozen_BI_OT_U_depth_geometry_guard_primitives_reused=True,
        runtime_new_problem_or_ownership_planner_calls=0,new_RGB_reference_optimization_or_inference_calls=0,
        scope='selected-pose arithmetic/guards/source support plus stored contender minima; graph global optimum and every rejected candidate are not independently proved',
        default_adoption=False)
    C.write(dest/'AUDIT.json',report)
    C.write(dest/'AUDIT_SEAL.json',dict(receipt_sha256=C.sha(dest/'AUDIT.json'),
        components_sha256=C.sha(dest/'SELECTED_POSE_COMPONENTS.jsonl.gz'),
        sources={str(Path(module.__file__).resolve().relative_to(C.ROOT)):C.sha(module.__file__)
                 for module in (N,KERNEL,OLD_AUDIT,CONTAINMENT,NOISE)},
        audit_source_sha256=C.sha(__file__),prediction_seal_sha256=C.sha(output/'PREDICTION_SEAL.json')))
    print(dict(rows=rows,checks=total_checks,failures=len(failures),maximum_difference=maximum),flush=True)
    if failures:raise SystemExit(1)


if __name__=='__main__':
    main()
