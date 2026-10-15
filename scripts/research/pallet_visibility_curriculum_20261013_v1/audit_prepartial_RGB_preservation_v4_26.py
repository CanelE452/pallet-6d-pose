"""Independent selected-pose v26 audit; full run requires root memory GO.

Reconstruct restored complete points and cost-only partial suppression from
the original frozen native kernels. No v26 native proxy/problem wrapper or
experiment factory certifies its own cost. Reuse the v25 selected-pose audit
contracts for pure O/S, direct old-P1, containment, starts and family locks.
No target, score, image, optimization or inference is used.
"""
from __future__ import annotations

from collections import Counter
from copy import copy,deepcopy
from itertools import zip_longest
from pathlib import Path
import math
import types

import numpy as np

from . import audit_coarse_prior_containment_v4_25 as PREVIOUS
from . import audit_source_preserving_tangent_mask_v4_24 as A24
from . import evaluate_prepartial_RGB_preservation_v4_26 as V
from . import native_edge_problem_v4_17 as NATIVE
from . import source_preserving_complete_bi_evidence_v4_24 as SOURCE
from . import calibrate_prepartial_RGB_edge_mixture_v4_26 as RESTORE_CAL


# Independent old authenticator, with only the new declared CAL generation.
# It has the ORIGINAL full boundary, before restored-owner cost filtering.
_AUTH=types.FunctionType(SOURCE._AUTH.__code__,
    {**SOURCE._AUTH.__globals__,'NOISE_OBSERVATION_FAMILY':V.RESTORE.NOISE_OBSERVATION_FAMILY},'_audit_auth')
_AUTHENTICATE=types.FunctionType(SOURCE._AUTHENTICATE.__code__,
    {**SOURCE._AUTHENTICATE.__globals__,'_AUTH':_AUTH,
     'NOISE_OBSERVATION_FAMILY':V.RESTORE.NOISE_OBSERVATION_FAMILY},'_audit_authenticate')
_SOURCE_AUTH=types.FunctionType(SOURCE.authenticate_complete_bi_image_chain_groups.__code__,
    {**SOURCE.authenticate_complete_bi_image_chain_groups.__globals__,'_AUTHENTICATE':_AUTHENTICATE},'_audit_source_auth')
_SOURCE_AUTH.__kwdefaults__=deepcopy(SOURCE.authenticate_complete_bi_image_chain_groups.__kwdefaults__)


def _original_prior_fields(task,model,model_sha256,mode):
    original=PREVIOUS._original_prior_fields(task,model,model_sha256,mode)
    original['factor_weight']=0. if mode=='FULL' else 1.
    return original


def direct_old_P1_prior(p,xyz,scale,original,trial_Q):
    value,gradient,audit=PREVIOUS.direct_old_P1_prior(p,xyz,scale,original,trial_Q)
    weight=float(original['factor_weight'])
    return weight*value,weight*gradient,{**audit,'factor_weight':weight,
        'FULL_schema_is_disabled_placeholder':original['mode']=='FULL'}


_source_binding=PREVIOUS._source_binding
_energies=PREVIOUS._energies
_native_endpoint_bank=types.FunctionType(PREVIOUS._native_endpoint_bank.__code__,
    {**PREVIOUS._native_endpoint_bank.__globals__,'V':V},'_native_endpoint_bank')
CONTAINMENT=PREVIOUS.CONTAINMENT


def _independent_plan(raw,records,bundle):
    partial={b['native_corner'] for b in raw.boundary}
    source={}
    for r in records:
        k=r['native_corner']
        if raw.states[k]=='S' or raw.states[k] not in ('B','I'):continue
        source[k]=r['source']
    restored={k for k in partial if source[k]!=SOURCE.FALLBACK_SOURCE}
    # Plain shallow native view for PLANNING only. No v26 factory/proxy is
    # called and no cost is evaluated through this augmented plan view.
    augmented=copy(raw)
    augmented.complete=[m for m in raw.measurements if m['native_corner'] not in partial or m['native_corner'] in restored]
    original=SOURCE.plan_complete_bi_replacement(augmented,bundle,overlap_policy=SOURCE.PROTOTYPE_OVERLAP_POLICY)
    active=[]
    for g in original['active_groups']:
        owners=[deepcopy(o) for o in g['activation_owners'] if o['native_corner'] not in restored]
        if not owners:continue
        active.append({**deepcopy(g),'activation_owners':owners,
            'guard_authorized_original_normal_factor_refs':deepcopy(g['replaces_normal_factor_refs']),
            'replaces_normal_factor_refs':[deepcopy(r) for r in g['replaces_normal_factor_refs'] if r['native_corner'] not in restored]})
    plan={**original,'active_groups':active,
        'replaces_normal_factor_refs':[r for g in active for r in g['replaces_normal_factor_refs']],
        'restored_prepartial_complete_ids':sorted(restored),
        'restored_prepartial_source_archives':{str(k):deepcopy(bundle['frozen_measurement_archives'][str(k)]) for k in sorted(restored)}}
    return plan,restored


class IndependentRestoredNativeProblem:
    """Direct old native residuals + mixture/TCAL kernels, same raw guards."""
    def __init__(self,raw,plan,restored):
        self.raw,self.plan,self.restored=raw,plan,set(restored)
        self.revoked=set(plan['revoked_complete_ids'])

    def __getattr__(self,name):return getattr(self.raw,name)

    def objective(self,p):
        view=copy(self.raw)
        partial={b['native_corner'] for b in self.raw.boundary}
        view.complete=[m for m in self.raw.measurements
            if (m['native_corner'] not in partial or m['native_corner'] in self.restored)
            and m['native_corner'] not in self.revoked]
        refs={(r['native_corner'],r['line_index']) for r in self.plan['replaces_normal_factor_refs']}
        view.boundary=[b for b in self.raw.boundary if b['native_corner'] not in self.restored]
        view.remaining_normals=[]
        for b in view.boundary:
            lines=[line for i,line in enumerate(b['lines']) if (b['native_corner'],i) not in refs]
            if lines:view.remaining_normals.append({**b,'lines':lines})
        view.replaced=refs
        value,gradient,parts=NATIVE.NativeEdgeProblem.objective(view,p)
        for group in self.plan['active_groups']:
            extra,derivative,_=SOURCE.M.mixture_edge_value_and_gradient(p,self.xyz,self.K,self.scale,
                group['components'],component_weights=group['component_weights'],
                reliability_weight=group['reliability_weight'],huber_scale=self.cfg['huber_scale'])
            value+=extra;gradient+=derivative
            parts['native_costs'].append(dict(kind='PROJECTED_NATIVE_EDGE',native_corner=group['native_corner'],
                native_neighbor=group['native_neighbor'],weighted_robust_cost=extra))
        q,_,jac,_=NATIVE.J.projection_and_jacobian(p,self.xyz,self.K,self.scale)
        delta=self.cfg['huber_scale']
        for record in self.plan['corner_plans']:
            if record.get('tangent_kind')!='ARCHIVED_COMPLETE_BI_TANGENT':continue
            archive=record['original_measurement_archive'];k=record['native_corner'];tangent=np.asarray(record['tangent_unit_xy'])
            sigma=np.sqrt(record['tangent_variance_px2']);r=float(tangent@(q[k]-archive['original_observed_xy'])/sigma)
            derivative=tangent@jac[k]/sigma;weight=archive['original_fit_reliability_weight']
            denominator=np.sqrt(1.+r*r/delta**2);extra=float(weight*2.*delta**2*(denominator-1.))
            value+=extra;gradient+=2.*weight*r/denominator*derivative
            parts['native_costs'].append(dict(kind='ARCHIVED_COMPLETE_BI_TANGENT',native_corner=k,weighted_robust_cost=extra))
        return float(value),gradient,parts

    def constraints(self,p,sides):return self.raw.constraints(p,sides)

    def evaluate(self,R,t):
        checked=self.raw.evaluate(R,t)
        if checked.get('objective') is None:return checked
        p=NATIVE.V._pose_vector(np.asarray(R),np.asarray(t),self.scale)
        value,_,parts=self.objective(p)
        return {**checked,**parts,'objective':value,'accepted':bool(checked['accepted'] and np.isfinite(value))}


def _reconstruct_native(task,family,edge_model,tangent_model,*,source_preserving=True):
    spec=task['spec']
    raw=NATIVE.prepare_native_edge_problem(family['native_xyz'],spec['K'],spec['states'],spec['visible'],
        spec['domains'],spec['image_hw'],family['coarse_pose'],config={**spec['config'],'prior_weight':0.},
        boundary_observations=task['boundary_merged'],u_regions=task['u_regions'],edge_bundle=dict(edges=[],replaces_normal_factor_refs=[]))
    bundle=_SOURCE_AUTH(raw,task['new_chain_observation'],junction_proposals=task['complete_BI_evidence']['junction_proposals'],
        visible_measurements=spec['visible'],initial_self_hidden=task['initial_self_hidden'],
        image_binding=dict(sha256=task['image_sha256'],image_hw=spec['image_hw']),
        initial_projection_xy=task['complete_BI_evidence']['initial_projection_xy'],
        **RESTORE_CAL.mixture_noise_parameters(edge_model))
    plan,restored=_independent_plan(raw,spec['visible'],bundle)
    parameters=A24.TCAL.tangent_variance_parameters(tangent_model)
    for r in plan['corner_plans']:
        if r.get('tangent_kind')!='ARCHIVED_COMPLETE_BI_TANGENT':continue
        k=r['native_corner']
        if task['initial_self_hidden'][k] or task['states'][k] not in ('B','I') or k in restored:
            raise ValueError('Only non-restored nonS BI may have new archived-complete tangent')
        tangent=np.asarray(r['tangent_unit_xy']);covariance=np.asarray(r['original_measurement_archive']['original_effective_fit_covariance_px2'])
        axis=float(tangent@covariance@tangent)
        if not np.isclose(r['tangent_variance_px2'],4.*axis,atol=1e-10,rtol=1e-10):raise ValueError('Original tangent covariance changed')
        r['tangent_variance_px2']=parameters['pixel_floor_variance_px2']+parameters['gamma']*axis
    return IndependentRestoredNativeProblem(raw,plan,restored),raw


def _model_readout_checks(task,result):
    partial={b['native_corner'] for b in task['boundary_merged']};restored=set();visible={}
    for m in task['spec']['visible']:
        k=m['native_corner']
        if task['initial_self_hidden'][k] or task['states'][k] not in ('B','I'):raise ValueError('Readout gate before q/source')
        visible[k]=m
        if k in partial and m['source']!=SOURCE.FALLBACK_SOURCE:restored.add(k)
    family=task['families'][result['WD_family']];pose=result['actual_pose'];readout=result['keypoint_readout']
    q=V.C.project(family['native_xyz'],pose['R_cf'],pose['centroid'],task['spec']['K'])
    checks=[]
    for k,state in enumerate(task['states']):
        point=readout['reported_points'][k]
        if state=='S':checks.append((np.allclose(point,q[k],atol=1e-8,rtol=0),'S projection only including fallback'))
        elif state=='U' or (k in partial and k not in restored):checks.append((point is None,'U/remaining partial not precise point'))
        elif k in restored:
            checks.append((point==visible[k]['xy'] and readout['point_sources'][k]==visible[k]['source'],'restored exact original RGB/unknown q and source'))
        elif state in ('O','T') and not result['accepted']:checks.append((point is None,'fallback does not solve O/T'))
    checks.append((readout['restored_prepartial_RGB_ids']==sorted(restored),'all restored IDs readout matches source gate'))
    return checks


# Only old independent helpers/constants are reused, with the native cost and
# readout reconstruction above. No current experiment/problem factory is used.
A=types.SimpleNamespace(**{k:getattr(A24,k) for k in ('V','NATIVE','S','O','OFIT','SFIT','MASK_CORE','TCAL','_plan_signature','_native_cost_buckets')},
    SOURCE_CAL=RESTORE_CAL,_reconstruct_native=_reconstruct_native,_model_readout_checks=_model_readout_checks)


def main(output=None):
    C, E = V.C, V.E
    output = Path(V.OUTPUT if output is None else output)
    dest = output / 'independent_root_audit_v1'
    if dest.exists():
        raise FileExistsError('Preserve all previous v26 audit evidence')
    protocol, seal = C.read(output / 'PROTOCOL.json'), C.read(output / 'PREDICTION_SEAL.json')
    inputseal = C.read(output / 'INPUT_SEAL.json')
    parent = A.V.OLD.V.OUTPUT
    parentseal = C.read(parent / 'PREDICTION_SEAL.json')
    expected_profiles=(('NATIVE_ZERO_COARSE','FULL',False,V.ARMS[1:3]),
        ('TRANSLATION_NATIVE','TRANSLATION',False,V.ARMS[3:5]),
        ('TRANSLATION_CONTAINMENT','TRANSLATION',True,V.ARMS[5:7]))
    if V.PROFILES!=expected_profiles:raise ValueError('Exact zero/TP/TP+C profile mapping required')
    if (seal['rows'] != 238 or seal['arms'] != list(V.ARMS) or seal['arm_count'] != 7
            or not seal['all7_arms_complete'] or not seal['all_declared_arms_complete']
            or protocol['arms'] != list(V.ARMS)
            or seal['predictions_sha256'] != C.sha(output / 'PREDICTIONS.jsonl.gz')
            or seal['protocol_sha256'] != C.sha(output / 'PROTOCOL.json')
            or seal['input_seal_sha256'] != C.sha(output / 'INPUT_SEAL.json')
            or inputseal['input_sha256'] != C.sha(output / 'INPUTS.jsonl.gz')
            or inputseal['protocol_sha256'] != C.sha(output / 'PROTOCOL.json')
            or parentseal['rows'] != 238
            or parentseal['predictions_sha256'] != C.sha(parent / 'PREDICTIONS.jsonl.gz')
            or protocol['coarse_prior_weights'] != dict(noP=0., translation=1., beta_bias=1.)
            or protocol['old_full_outline_factor_or_tie_break_used'] is not False):
        raise ValueError('Exact all238 seven-arm seal and zero/TP1 beta1 profiles required')
    for key in ('source_sha256', 'input_sha256', 'protected_user_sha256'):
        E._verify_sources(protocol[key])
    binding = protocol['coarse_prior_model_binding']
    if C.sha(C.ROOT / binding['path']) != binding['sha256']:
        raise ValueError('Frozen CAL coarse covariance/mean binding changed')
    coarsemodel = C.read(C.ROOT / binding['path'])
    if coarsemodel['fit_partition'] != 'CAL':
        raise ValueError('Only the existing frozen CAL prior model is allowed')
    edge = C.read(A.SOURCE_CAL.OUTPUT / 'MODEL.json')
    tangent = C.read(A.TCAL.OUTPUT / 'MODEL.json')
    A.SOURCE_CAL.mixture_noise_parameters(edge)
    A.TCAL.tangent_variance_parameters(tangent)
    omodel, smodel = C.read(A.OFIT.OUTPUT / 'MODEL.json'), C.read(A.SFIT.OUTPUT / 'MODEL.json')
    op = A.OFIT.prior_parameters(omodel)
    A.S.validate_model(smodel)
    checks, failures, maximum, rows = 0, [], 0., 0
    counts, accepted, fallback = Counter(), 0, 0

    def check(condition, what, ident, arm=None):
        nonlocal checks
        checks += 1
        if not condition:
            failures.append(dict(id=ident, arm=arm, check=what))

    def near(first, second):
        nonlocal maximum
        delta = abs(float(first) - float(second))
        maximum = max(maximum, delta)
        return delta <= 1e-8 * max(1., abs(float(first)), abs(float(second)))

    for task, row, previous in zip_longest(C.rows(output / 'INPUTS.jsonl.gz'),
            C.rows(output / 'PREDICTIONS.jsonl.gz'), C.rows(parent / 'PREDICTIONS.jsonl.gz')):
        if task is None or row is None or previous is None:
            raise ValueError('All ordered frames including fallbacks must remain')
        rows += 1
        ident = task['id']
        counts[task['partition']] += 1
        check(ident == row['id'] == previous['id'] == protocol['ids'][rows-1], 'ordered238', ident)
        check(row['arms'][V.ARMS[0]] == task['baseline'], 'exact frozen e baseline', ident)
        check(set(row['arms']) == set(V.ARMS), 'all seven declared arms', ident)
        check(task['states'] == row['states'] == task['spec']['states'], 'unchanged authoritative states', ident)
        first = task['families']['ORIGINAL_WD']
        coarse = first['coarse_pose']
        sprepared = A.S.prepare_bounded_coarse_visibility_prior(first['native_xyz'], coarse['R'], coarse['t'],
            np.asarray(task['initial_self_hidden'], bool), smodel)
        regions = []
        for r in task['O_localization_abstention']['records']:
            d = r['original_domain']
            k = d['native_corner']
            if task['initial_self_hidden'][k] or task['states'][k] != 'O':
                raise ValueError('O state gate precedes localization fields')
            check(d['binding']['calibration_model_sha256'] == omodel['source_O_model_digest'], 'O source-radius model', ident)
            regions.append(d)
        regions = A.O.prepare_o_localization_regions(regions, task['states'], task['initial_self_hidden'])
        cfg = A.MASK_CORE._outline_config(task['outline']['configuration'])  # Config only; never old mask cost.
        supported = bool(task['outline'].get('supported'))
        weight = cfg['outline_factor_weight'] if supported else 0.
        originals = {mode: _original_prior_fields(task, coarsemodel, binding['sha256'], mode)
            for mode in ('TRANSLATION', 'FULL')}
        for mode, fields in originals.items():
            stored = row['original_first_pose_prior_inputs'][mode]
            check(all(E._digest(stored[k]) == E._digest(value) for k, value in fields.items())
                and stored['source_binding_sha256'] == _source_binding(fields),
                'original first physical pose/CAL parameters fixed before every WD trial', ident)
        oldsearches = previous['native_refinement_audit'][A.V.OLD.V.ARMS[1]]['family_refinements']
        problems = {}
        for profile, mode, containment, arms in V.PROFILES:
            if mode not in ('TRANSLATION', 'FULL') or V.PROFILE_ARMS[profile] != arms or len(arms) != 2:
                raise ValueError('Exact three source-defined optimization profiles required')
            searches = row['bounded_prior_refinement_audit'][arms[0]]['family_refinements']
            check(set(searches) == set(oldsearches) == set(task['families']), 'same two WD searches', ident, arms[0])
            for name, search in searches.items():
                check(search['start_selection']['selected_start_names'] == oldsearches[name]['start_selection']['selected_start_names'],
                    'exact frozen v19-native start names', ident, arms[0])
                check(search['search']['old_full_outline_factor_used'] is False
                    and search['search']['mask_objective'] is False,
                    'no old bidirectional mask optimization', ident, arms[0])
                check(search['search']['CAL_coarse_prior_mode']==mode
                    and search['search']['zero_weight_FULL_schema_not_active_rotation_prior'] is (mode=='FULL'),
                    'zero-FULL placeholder and active TP search metadata', ident, arms[0])
            for index, arm in enumerate(arms):
                result = row['arms'][arm]
                family = task['families'][result['WD_family']]
                pose, solution = result['actual_pose'], result['solution']
                R, t = np.asarray(pose['R_cf']), np.asarray(pose['centroid'])
                locked = index == 1
                lock = result['early_family_audit']
                check(lock['enabled'] is locked and lock['e_numeric_pose_used'] is False,
                    'only declared locked arm uses predicted WD label', ident, arm)
                check(np.allclose(R @ family['physical_registration_Q'], pose['R_physical'], atol=1e-12, rtol=0),
                    'selected physical Q unchanged', ident, arm)
                for condition, reason in A._model_readout_checks(task, result):
                    check(condition, reason, ident, arm)
                check(result['keypoint_readout']['old_full_outline_factor_used'] is False,
                    'no old full-mask readout claim', ident, arm)
                if locked:
                    check(lock['requested_predicted_family'] == task['baseline']['WD_family'], 'predicted e label only', ident, arm)
                if not result['accepted']:
                    fallback += 1
                    check(pose['fallback'] and result['WD_family'] == 'ORIGINAL_WD'
                        and np.array_equal(R, coarse['R']) and np.array_equal(t, coarse['t'])
                        and result['fallback_audit']['old_best_pose_fallback'] is False,
                        'own original coarse fallback only', ident, arm)
                    if locked:
                        check(lock['locked_family_constraint_satisfied'] is False
                            and result['fallback_audit']['lock_requested_but_unsatisfied']
                            and result['candidate_selection']['selected_name'] is None,
                            'failed family constraint is explicit', ident, arm)
                    continue
                accepted += 1
                if result['WD_family'] not in problems:
                    problems[result['WD_family']] = A._reconstruct_native(task, family, edge, tangent, source_preserving=True)
                native_problem, raw = problems[result['WD_family']]
                fresh = native_problem.evaluate(R, t)
                check(fresh['accepted'], 'independent original scalar all-guard acceptance', ident, arm)
                p = A.NATIVE.V._pose_vector(R, t, native_problem.scale)
                values, jac = native_problem.constraints(p, native_problem.sides[0])
                oldvalues, oldjac = raw.constraints(p, raw.sides[0])
                check(np.array_equal(values, oldvalues) and np.array_equal(jac, oldjac), 'hardguard values/Jacobian exact', ident, arm)
                check(E._digest(A._plan_signature(native_problem.plan)) == E._digest(A._plan_signature(solution['complete_BI_replacement_audit'])),
                    'exact SOURCE/TCAL ownership and original q/noise archives', ident, arm)
                check(solution['replaced_old_normal_factor_count'] == len(native_problem.plan['replaces_normal_factor_refs']),
                    'old normals replaced once', ident, arm)
                stored_plan = solution['complete_BI_replacement_audit']
                restored = sorted(native_problem.restored)
                check(stored_plan['restored_prepartial_complete_ids'] == restored
                    and E._digest(stored_plan['restored_prepartial_source_archives']) == E._digest(native_problem.plan['restored_prepartial_source_archives']),
                    'restored exact source/q/observedSigma/fitSigma/reliability archives', ident, arm)
                check(all(o['native_corner'] not in native_problem.restored for g in stored_plan['active_groups'] for o in g['activation_owners'])
                    and all(r['native_corner'] not in native_problem.restored for r in stored_plan['replaces_normal_factor_refs']),
                    'restored owner/ref removed; shared surviving owners retained', ident, arm)
                counts_native = Counter((c['kind'], c.get('native_corner')) for c in fresh['native_costs'])
                check(all(counts_native[('COMPLETE_BI', k)] == 1
                    and counts_native[('UNREPLACED_PARTIAL_NORMAL', k)] == 0
                    and counts_native[('UNCERTAIN_ARCHIVED_TANGENT', k)] == 0
                    and counts_native[('ARCHIVED_COMPLETE_BI_TANGENT', k)] == 0 for k in restored),
                    'restored full2D once and own old normal/tangents zero', ident, arm)
                check(stored_plan['ledger']['restored_prepartial_complete_q_factors_once'] == len(restored)
                    and solution['complete_BI_likelihood_count'] == len(raw.complete) + len(restored) - len(native_problem.revoked),
                    'complete restoration/revocation ledger exact', ident, arm)
                native = float(fresh['objective'])
                check(near(native, solution['native_observation_objective']), 'independent native image cost', ident, arm)
                left, right = A._native_cost_buckets(fresh['native_costs']), A._native_cost_buckets(solution['native_costs'])
                check(left.keys() == right.keys() and all(near(left[k], right[k]) for k in left), 'native factor identity and cost once', ident, arm)
                ov, _, oa = A.O.bounded_o_localization_value_and_gradient(p, native_problem.xyz, native_problem.K,
                    native_problem.scale, regions, **op)
                sr = A.S.rebind_bounded_visibility_trial_geometry(sprepared, native_problem.xyz)
                sv, _, sa = A.S.coarse_visibility_prior_value_and_gradient(p, native_problem.xyz, native_problem.scale, sr)
                pv, _, pa = direct_old_P1_prior(p, native_problem.xyz, native_problem.scale, originals[mode], family['physical_registration_Q'])
                parts = solution['regularizer_costs']
                kinds = ['O_COARSE_LOCALIZATION_PRIOR', 'S_COARSE_VISIBILITY_PRIOR', 'CAL_COARSE_POSE_PRIOR']
                check([part['kind'] for part in parts] == kinds, 'exact three separate priors; C excluded', ident, arm)
                actual = {part['kind']: part['weighted_cost'] for part in parts}
                check(all(near(actual[k], value) for k, value in zip(kinds, (ov, sv, pv))), 'independent O/S/direct-old-P1 prior costs', ident, arm)
                check(oa['O_point_terms'] == 0 and sa['S_image_or_2D_terms'] == 0, 'no hidden-point image prior', ident, arm)
                check(ov <= -2*math.log(op['epsilon']) + 1e-10 and sv <= -2*math.log(smodel['epsilon']) + 1e-10,
                    'O/S one-frame caps; coarse Gaussian not falsely capped', ident, arm)
                prior_audit = solution['CAL_coarse_pose_prior_audit']
                check(prior_audit['mode'] == mode and prior_audit['factor_weight'] == (0. if mode == 'FULL' else 1.)
                    and prior_audit['bias_strength'] == 1. and prior_audit['calibration_model_digest_sha256'] == binding['sha256']
                    and prior_audit['mean_bias_subtracted_once'] is True and prior_audit['S_U_raw_image_2D_or_regions'] == 0,
                    'fixed CAL model/beta1/zero-FULL placeholder or TP1/no S/U image inputs', ident, arm)
                check(np.allclose(prior_audit['whitened_residual'], pa['residual'], atol=1e-9, rtol=1e-10)
                    and prior_audit['original_source_binding_sha256'] == _source_binding(originals[mode]),
                    'direct residual vector and original physical/CAL binding, not only squared norm', ident, arm)
                shape, ca = 0., None
                if containment and supported and weight > 0:
                    shape, _, ca = CONTAINMENT.outline_value_and_gradient(p, native_problem.xyz, native_problem.K,
                        native_problem.scale, task['outline'], cfg)
                    check(ca['valid_projection'] and ca['model_to_observed_factor_weight'] == 0.
                        and ca['inside_model_hull_cost_zero'] is True, 'pure outside-only C, no complete silhouette matching', ident, arm)
                energies = _energies(native, ov, sv, pv, shape, weight, containment)
                check(near(solution['regularizer_objective'], energies['regularizer'])
                    and near(solution['objective'], energies['total']), 'optimizer total N+O+S+P+C once', ident, arm)
                check(solution['coarse_pose_prior_count'] == 1 and near(solution['CAL_coarse_pose_prior_objective'], pv),
                    'coarse prior ledger once', ident, arm)
                if containment:
                    check(near(solution['global_containment_cost'], shape)
                        and near(solution['weighted_containment_objective'], energies['weighted_containment'])
                        and near(solution['native_plus_coarse_regularized_objective'], energies['regularized_native'])
                        and solution['containment_count_in_optimizer'] == int(supported and weight > 0)
                        and solution['old_bidirectional_full_outline_factor_used'] is False,
                        'independent C ledger and no old M', ident, arm)
                selection = result['candidate_selection']
                check(selection['old_full_outline_factor_or_tie_break_used'] is False
                    and selection['e_numeric_pose_used'] is False,
                    'no old mask tie-break or numeric e fallback', ident, arm)
                chosen = [c for c in selection['candidates'] if c['name'] == selection['selected_name']]
                check(len(chosen) == 1, 'one selected authenticated representative', ident, arm)
                if chosen:
                    chosen = chosen[0]
                    check(near(chosen['regularized_native_energy'], energies['regularized_native'])
                        and near(chosen['global_containment_cost'], shape)
                        and near(chosen['weighted_containment_objective'], energies['weighted_containment'])
                        and near(chosen['total_energy'], energies['total']), 'selector uses identical independent components once', ident, arm)
                    minimum = min(c['total_energy'] for c in selection['candidates'])
                    tol = 32*np.finfo(float).eps * max(1., abs(minimum), abs(chosen['total_energy']))
                    check(chosen['total_energy'] <= minimum + tol, 'minimum stored contender total', ident, arm)
                if locked:
                    check(result['WD_family'] == task['baseline']['WD_family'] and lock['locked_family_constraint_satisfied'],
                        'predicted family constraint satisfied', ident, arm)
                for k in native_problem.revoked:
                    check(result['keypoint_readout']['reported_points'][k] is None, 'revoked q not precise new KP', ident, arm)
            unlocked, locked = (row['arms'][arm]['candidate_selection'] for arm in arms)
            expected = {c['name']: c for c in unlocked['candidates'] if c['family'] == task['baseline']['WD_family']}
            observed = {c['name']: c for c in locked['candidates']}
            check(expected.keys() == observed.keys(), 'same feasible bank, predicted-family-only filter', ident, arms[1])
            check(all(c['family'] == task['baseline']['WD_family'] for c in observed.values()), 'no other family in lock bank', ident, arms[1])
            fields = ('native_energy', 'regularizer_energy', 'regularized_native_energy', 'global_containment_cost',
                'weighted_containment_objective', 'total_energy')
            check(all(all(near(expected[name][key], observed[name][key]) for key in fields) for name in observed),
                'lock changes eligibility only, no pose/cost substitution', ident, arms[1])
            if containment:
                check(row['bounded_prior_refinement_audit'][arms[0]]['same_mode_native_endpoints_retained'] is True,
                    'containment bank declares same-mode native endpoint retention', ident, arms[0])
                names = {c['name'] for c in unlocked['candidates']}
                for endpoint in _native_endpoint_bank(task, row, mode):
                    if endpoint['name'] in names:
                        check(True, 'same-mode native endpoint retained in accepted C bank', ident, arms[0])
                        continue
                    # A missing native endpoint is allowed only if the new C
                    # kernel itself is numerically invalid. No pose fit occurs.
                    valid = True
                    if supported and weight > 0:
                        f = task['families'][endpoint['family']]
                        xyz = np.asarray(f['native_xyz'])
                        scale = max(float(np.linalg.norm(np.ptp(xyz, axis=0))), 1e-3)
                        p = A.NATIVE.V._pose_vector(np.asarray(endpoint['R']), np.asarray(endpoint['t']), scale)
                        cv, cg, ca = CONTAINMENT.outline_value_and_gradient(p, xyz, np.asarray(task['spec']['K']),
                            scale, task['outline'], cfg)
                        valid = bool(ca['valid_projection'] and np.isfinite(cv) and cv >= 0 and np.isfinite(cg).all())
                    check(not valid, 'native endpoint absent only for independent C numeric invalidity', ident, arms[0])
    check(rows == 238 and counts == Counter(CAL=46, TEST=49, REMAINDER=143), 'all238 severe0 retained', None)
    for key in ('source_sha256', 'input_sha256', 'protected_user_sha256'):
        E._verify_sources(protocol[key])
    receipt = dict(schema='INDEPENDENT_PREPARTIAL_RGB_SELECTED_POSE_AUDIT_V26', rows=rows,
        partitions=dict(counts), primary192=192, severe=0, checks=checks, failure_count=len(failures), failures=failures,
        selected_arm_poses=accepted, fallback_arm_poses=fallback, maximum_energy_difference=maximum,
        source=E._binding(Path(__file__)), prediction_seal=E._binding(output / 'PREDICTION_SEAL.json'),
        frozen_start_policy_prediction_seal=E._binding(parent / 'PREDICTION_SEAL.json'),
        coarse_model_binding=binding, prior_and_containment_wrappers_not_used_to_certify_own_objective=True,
        direct_old_BIAS_API_with_full_or_marginal_whitening_and_mean_once=True,
        no_old_full_outline_factor_or_numeric_e_fallback=True,
        restored_native_costs_direct_original_NativeEdgeProblem_not_v26_proxy=True,
        restored_full_q_once_and_own_oldnormal_T4_owner_costs_removed=True,
        FULL_schema_is_exact_zero_weight_placeholder_not_active_rotation_prior=True,
        same_mode_native_endpoint_retention_checked=True,
        missing_native_endpoint_containment_numeric_validity_only_recomputed_if_needed=True,
        nonselected_candidate_numeric_energies_recomputed=False,
        scope='independent selected-pose energy/guards/archives; stored nonselected energies for minimum check',
        new_pose_fits_RGB_NN_or_reference_decodes=0, default_adoption=False)
    C.write(dest / 'AUDIT.json', receipt)
    C.write(dest / 'AUDIT_SEAL.json', dict(receipt_sha256=C.sha(dest / 'AUDIT.json')))
    print(dict(rows=rows, checks=checks, failures=len(failures), max_energy_difference=maximum), flush=True)
    if failures:
        raise ValueError('Independent v26 selected-pose audit failed')


if __name__ == '__main__':
    main()
