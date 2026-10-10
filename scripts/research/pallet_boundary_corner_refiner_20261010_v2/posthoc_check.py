"""Independent saved-row arithmetic audit; no inference or reference loader.

This module deliberately imports neither diagnostics.py nor the deployment
pipeline. All references below are already stored in the scored public rows.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

REPO = Path(__file__).resolve().parents[3]
DOC = REPO/'_docs/experiments/pallet_boundary_corner_refiner_20261010_v2'
METHODS = ('VALIDATED_ROLE_ONLY','N3_VALIDATED_ROLE','N3_VALIDATED_ROLE_NO_MASK',
           'N3_BASIN_ROBUST','N3_BASIN_STANDARD')
POSE = ('translation_cm','rotation_deg','ADDsym_m')
POOLS = ('eligible','used','fit_input','final_inlier','accepted_new_pose_final_inlier')
GROUPS = ('DIRECT_VISIBLE','SELF_OCCLUDED','ACTUALLY_REPROJECTED_H')
OUTCOMES = ('BOTH_TRANSLATION_ROTATION_IMPROVED','BOTH_TRANSLATION_ROTATION_WORSENED',
            'MIXED_OR_EQUAL','NO_COMMON_OPERATIONAL_POSE')


def read(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    with gzip.open(path,'rt') as f:
        return [json.loads(line) for line in f if line.strip()]


def binding(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):
            h.update(b)
    return dict(path=str(path.relative_to(REPO)) if path.is_relative_to(REPO) else path.name,
                sha256=h.hexdigest(),bytes=path.stat().st_size)


class Audit:
    def __init__(self):
        self.checks = 0
        self.max_absolute_float_difference = 0.

    def require(self, condition, path):
        self.checks += 1
        if not condition:
            raise AssertionError('POSTHOC_CHECK: '+path)

    def equal(self, actual, expected, path):
        if isinstance(expected,dict):
            for k,v in expected.items():
                self.require(str(k) in actual,path+'.'+str(k)+' missing')
                self.equal(actual[str(k)],v,path+'.'+str(k))
        elif isinstance(expected,(list,tuple)):
            self.require(isinstance(actual,list) and len(actual)==len(expected),path+' length')
            for i,v in enumerate(expected):
                self.equal(actual[i],v,path+'['+str(i)+']')
        elif isinstance(expected,float):
            if not math.isfinite(expected):
                self.require(actual is None,path+' missing/nonfinite convention')
            else:
                self.require(actual is not None and isinstance(actual,(float,int)),path+' numeric type')
                error = abs(float(actual)-expected)
                self.max_absolute_float_difference = max(self.max_absolute_float_difference,error)
                self.require(error<=1e-8+1e-10*abs(expected),path+' numeric difference')
        else:
            self.require(actual==expected,path+' value')


def arr(value):
    return np.asarray(value,dtype=float)[:8]


def available(a):
    return np.isfinite(a).all(1)&~np.all(a==-1,axis=1)


def error_array(a,g,known):
    answer = np.full(8,np.nan)
    take = known & available(a)
    # Per-coordinate dot products, independent of diagnostics' norm expression.
    difference = a[take]-g[take]
    answer[take] = np.sqrt(np.einsum('ij,ij->i',difference,difference))
    return answer


def moments(values):
    values = [float(x) for x in values]
    assert all(math.isfinite(x) for x in values)
    n = len(values)
    mean = math.fsum(values)/n if n else None
    variance = math.fsum((x-mean)**2 for x in values)/(n-1) if n>1 else None
    return dict(n=n,mean=mean,sample_variance=variance,sample_std=math.sqrt(variance) if variance is not None else None,
        median=float(np.quantile(values,.5)) if n else None,P90=float(np.quantile(values,.9)) if n else None,
        maximum=max(values) if n else None,ddof=1,quantile_method='numpy linear')


def shape(a,which):
    chosen = np.asarray(a,float)[which]
    good = np.isfinite(chosen).all(1)
    chosen = chosen[good]
    dimension = np.asarray(a).shape[1]
    sv = np.zeros(dimension)
    if len(chosen)>1:
        returned = np.linalg.svd(chosen-chosen.mean(0),compute_uv=False)
        sv[:len(returned)] = returned
    rel = sv/sv[0] if sv[0]>0 else np.zeros(dimension)
    return dict(ids=which,finite_ids=[k for k,v in zip(which,good) if v],n=len(chosen),
        singular_values=sv.tolist(),relative_singular_values=rel.tolist(),
        centered_numerical_rank=int(np.count_nonzero(rel>1e-10)),rank_threshold_relative=1e-10,
        rms_radius=float(np.sqrt(np.einsum('ij,ij->',chosen-chosen.mean(0),chosen-chosen.mean(0))/len(chosen)))
        if len(chosen) else None)


def cube(dimensions):
    # IDs are the unchanged native cuboid phase, not a physical-mask assertion.
    signs = np.asarray([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                        [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],float)
    return signs*np.asarray(dimensions)/2


def category(which,e0,ei,eo):
    pair = [k for k in which if math.isfinite(e0[k]) and math.isfinite(eo[k])]
    pair_input = [k for k in which if math.isfinite(ei[k]) and math.isfinite(eo[k])]
    return dict(ids=which,reference_count=len(which),
        initial_N3_available_ids=[k for k in which if math.isfinite(e0[k])],
        input_available_ids=[k for k in which if math.isfinite(ei[k])],
        output_available_ids=[k for k in which if math.isfinite(eo[k])],
        N3_to_output_comparable_ids=pair,input_to_output_comparable_ids=pair_input,
        damage_N3_le5_to_output_gt10_ids=[k for k in pair if e0[k]<=5 and eo[k]>10],
        improved_vs_N3_ids=[k for k in pair if eo[k]-e0[k]<-1e-9],
        worsened_vs_N3_ids=[k for k in pair if eo[k]-e0[k]>1e-9])


def verify_row(check,r,d,o,f,label):
    fid = r['id'];prefix=r['method']+'/'+fid
    reference = r['evaluation_reference']
    check.require(reference['phase_control']=='N3_SUBPIX',prefix+' phase')
    g = arr(reference['native_points_px'])
    n3 = arr(f['native_points']); incoming=arr(r['input_points']);outgoing=arr(r['native_points'])
    known = np.zeros(8,bool); known[reference['valid_native_ids']]=True
    known &= available(g)&bool(reference['matched'])
    e0,ei,eo = [error_array(a,g,known) for a in (n3,incoming,outgoing)]
    good = set(np.flatnonzero(np.isfinite(ei)&(ei<=8)).tolist())
    bad = set(np.flatnonzero(np.isfinite(ei)&(ei>8)).tolist())
    unknown = set(range(8))-good-bad
    states=reference['human_states_native'];direct={i for i,s in enumerate(states) if s=='DIRECT_VISIBLE'}
    self_h={i for i,s in enumerate(states) if s=='SELF_OCCLUDED'}
    known_mask={i for i,s in enumerate(states) if s!='UNANNOTATED'}
    H=set(r['hidden_initial']);predicted_H=set(r['predicted_initial_N3_hidden'])
    s=r['solver'];new=bool(s['available']);fallback=not new and bool(r['initial_pose']['available'])
    check.require(new==r['new_pose_estimated'] and fallback==r['fallback_used'],prefix+' status flags')
    check.require(H==(set() if r['method']=='N3_VALIDATED_ROLE_NO_MASK' else predicted_H),prefix+' no-mask state')
    check.require(predicted_H==set(o['predicted_N3_hidden']),prefix+' shared mask')
    check.require(r['reprojected_ids']==(sorted(H) if new else []),prefix+' actual H reprojection')
    check.require(not new or H.isdisjoint(s['fit_input_ids']),prefix+' hidden final fit')
    status='NEW_POSE' if new else 'N3_BASELINE_FALLBACK' if fallback else 'POSE_FAILURE'
    check.equal(r['output_status'],status,prefix+' status')
    check.equal(r['pose']['available'],bool(r['actual_pose']['available']),prefix+' scored availability')
    check.equal(r['hidden_set_changed'],bool(new and set(r['hidden_after'])!=H),prefix+' changed mask')
    contract=r['observation_contract'];lines={l['edge']:l for l in o['lines']}
    invalid=set();line_checks=[]
    for edge,line in lines.items():
        support=np.asarray(line['support_points'],float);normal=np.asarray(line['normal'],float)
        residual=np.abs(np.einsum('ij,j->i',support,normal)-line['offset'])
        radii=np.asarray(line['query_radii_px'],float)
        accept=bool(np.isfinite(residual).all() and np.isfinite(radii).all() and (radii>=0).all()
                    and (residual<=radii+1e-9).all())
        if not accept:invalid.add(edge)
        line_checks.append(dict(edge=edge,support_queries=line['queries'],absolute_residuals_px=residual.tolist(),
            query_radii_px=radii.tolist(),accepted=accept,tolerance_px=1e-9,
            reason='accepted' if accept else 'FINAL_LINE_CONSENSUS_INCONSISTENT'))
    admitted={};admission=[]
    for corner in sorted(o['corners'],key=lambda c:c['id']):
        radius=float(corner['radius_px'])
        reason='FINAL_LINE_CONSENSUS_INCONSISTENT' if set(corner['edges'])&invalid else (
            'UNCERTAINTY_EXCEEDS_PNP_OBSERVATION_RADIUS' if not math.isfinite(radius) or radius<0 or radius>8 else 'accepted')
        admission.append(dict(id=corner['id'],edges=corner['edges'],radius_px=radius,accepted=reason=='accepted',reason=reason))
        if reason=='accepted':admitted[corner['id']]=corner
    replaced=[]; rejected=[];hybrid=np.asarray(f['native_points'],float).copy()
    sparse=np.full_like(hybrid,np.nan);sparse[8]=hybrid[8]
    for k,c in admitted.items():
        xy=np.asarray(c['xy'],float);sparse[k]=xy
        accept=k not in predicted_H and available(n3[[k]])[0] and np.linalg.norm(xy-n3[k])<=8
        if accept:replaced.append(k);hybrid[k]=xy
        else:rejected.append(k)
    replaced.sort();rejected.sort()
    check.equal(contract['invalid_final_line_edges'],sorted(invalid),prefix+' line veto IDs')
    check.equal(contract['final_line_consensus_checks'],line_checks,prefix+' final line checks')
    check.equal(contract['corner_admission'],admission,prefix+' radius admission')
    check.equal(contract['validated_boundary_corner_ids'],sorted(admitted),prefix+' admitted IDs')
    check.equal(contract['hybrid_boundary_corner_ids'],replaced,prefix+' hybrid IDs')
    check.equal(contract['hybrid_rejected_boundary_corner_ids'],rejected,prefix+' hybrid rejected IDs')
    expect_input=sparse if r['method']=='VALIDATED_ROLE_ONLY' else hybrid if r['method'].startswith('N3_VALIDATED_ROLE') else np.asarray(f['native_points'],float)
    check.equal(r['input_points'],expect_input.tolist(),prefix+' input assembly')
    expect_output=np.asarray(f['native_points'],float).copy()
    if new:
        for k in range(8):
            if k not in H and available(incoming[[k]])[0]:expect_output[k]=incoming[k]
        for k in H:expect_output[k]=s['projected'][k]
    check.equal(r['native_points'],expect_output.tolist(),prefix+' output/fallback assembly')
    used=set(s['used']);eligible=set(s['eligible']);inliers=set(s['final_inliers']);fit=set(s['fit_input_ids'])
    check.require(used<=eligible and inliers<=used and fit<=used,prefix+' solver ID nesting')
    dimensions=s.get('cf_extents') or r['initial_pose'].get('cf_extents')
    model=cube(dimensions) if dimensions is not None else None
    pools={}
    for name,which in [('eligible',eligible),('used',used),('fit_input',fit),('final_inlier',inliers),
                       ('accepted_new_pose_final_inlier',inliers if new else set())]:
        which=sorted(which);correct=sorted(set(which)&good)
        pools[name]=dict(ids=which,correct_ids=correct,incorrect_ids=sorted(set(which)&bad),
            unknown_ids=sorted(set(which)&unknown),human_direct_ids=sorted(set(which)&direct),
            correct_human_direct_ids=sorted(set(which)&good&direct),image_layout=shape(incoming,which),
            correct_image_layout=shape(incoming,correct),cuboid_ID_layout=shape(model,which) if model is not None else None,
            correct_cuboid_ID_layout=shape(model,correct) if model is not None else None)
    comparable=bool(r['pose']['available'] and f['pose']['available'])
    delta={k:float(r['pose'][k]-f['pose'][k]) for k in POSE} if comparable else {}
    better=comparable and delta['translation_cm'] < -1e-9 and delta['rotation_deg'] < -1e-9
    worse=comparable and delta['translation_cm'] > 1e-9 and delta['rotation_deg'] > 1e-9
    outcome='NO_COMMON_OPERATIONAL_POSE' if not comparable else OUTCOMES[0] if better else OUTCOMES[1] if worse else OUTCOMES[2]
    wrong=bool((H^self_h)&known_mask)
    groups={name:category(sorted(which&set(np.flatnonzero(known))),e0,ei,eo)
            for name,which in [('DIRECT_VISIBLE',direct),('SELF_OCCLUDED',self_h),('ACTUALLY_REPROJECTED_H',H if new else set())]}
    evidence=[]
    for c in o['corners']:
        k=c['id'];candidate_error=float(np.sqrt(np.dot(np.asarray(c['xy'])-g[k],np.asarray(c['xy'])-g[k]))) if known[k] else None
        base_error=float(e0[k]) if math.isfinite(e0[k]) else None
        coordinate_used=k in used and (r['method']=='VALIDATED_ROLE_ONLY' and k in admitted or
                                       r['method'].startswith('N3_VALIDATED_ROLE') and k in replaced)
        displayed=new and k not in H and (r['method']=='VALIDATED_ROLE_ONLY' and k in admitted or
                                        r['method'].startswith('N3_VALIDATED_ROLE') and k in replaced)
        evidence.append(dict(id=k,xy=c['xy'],edges=c['edges'],radius_px=c['radius_px'],reference_available=candidate_error is not None,
            error_px=candidate_error,correct_within8px=candidate_error<=8 if candidate_error is not None else None,
            N3_error_px=base_error,error_delta_candidate_minus_N3_px=candidate_error-base_error
            if candidate_error is not None and base_error is not None else None,
            admitted=k in admitted,selected_for_hybrid=k in replaced,ID_in_solver_pool=k in used,
            ID_is_final_inlier=k in inliers,boundary_coordinate_in_solver_pool=bool(coordinate_used),
            displayed_as_boundary_observation=bool(displayed),no_hybrid_coordinate_substitution_on_fallback=True))
    obs=dict(selected_query_count=o['selected_queries'],decoded_line_count=len(o['lines']),computed_corner_count=len(o['corners']),
        computed_corner_ids=[c['id'] for c in o['corners']],computed_corner_radii_px=[c['radius_px'] for c in o['corners']],
        admitted_corner_ids=sorted(admitted),hybrid_replaced_corner_ids=replaced,hybrid_rejected_corner_ids=rejected,
        final_line_consensus_checks=line_checks,invalid_final_line_edges=sorted(invalid),corner_admission=admission,
        boundary_corner_evidence=evidence,line_veto_count=len(invalid),corner_admission_reasons=dict(Counter(c['reason'] for c in admission)))
    expected=dict(id=fid,session=r['session'],method=r['method'],difficulty_label=label,output_status=status,
        new_pose_estimated=new,fallback_used=fallback,pose_available=bool(r['pose']['available']),solver_state=s['state'],solver_reason=s['reason'],
        pose_jacobian=s['geometry'].get('jacobian'),reference_phase='fresh fixed N3_SUBPIX',reference=reference['reference'],
        reference_valid_ids=np.flatnonzero(known).tolist(),reference_native_points_px=g.tolist(),input_native_points_px=incoming.tolist(),
        output_native_points_px=outgoing.tolist(),fixed_N3_native_points_px=n3.tolist(),human_states_native=states,
        input_error_px=ei.tolist(),output_error_px=eo.tolist(),fixed_N3_error_px=e0.tolist(),input_correct_ids=sorted(good),
        input_incorrect_ids=sorted(bad),input_unknown_ids=sorted(unknown),input_correct_threshold_px=8.,pools=pools,
        final_inlier_scope='accepted_new_pose' if new else 'rejected_candidate_or_no_new_pose',hidden_initial_ids=sorted(H),
        human_self_hidden_ids=sorted(self_h),human_direct_visible_ids=sorted(direct),
        human_direct_reference_missing_ids=sorted(direct-set(np.flatnonzero(known))),
        human_self_reference_missing_ids=sorted(self_h-set(np.flatnonzero(known))),known_mask_ids=sorted(known_mask),
        mask_fully_annotated=len(known_mask)==8,mask_wrong_on_known=wrong,false_excluded_direct_ids=sorted(H&direct),
        false_retained_self_ids=sorted((self_h-H)&eligible),hidden_set_changed=r['hidden_set_changed'],
        actual_reprojected_ids=sorted(H) if new else [],
        actual_reprojected_reference_missing_ids=sorted((H if new else set())-set(np.flatnonzero(known))),
        comparable_to_N3=comparable,delta_vs_N3=delta,pose_outcome_vs_N3=outcome,
        wrong_known_mask_both_pose_improved=bool(wrong and better),matching_known_mask_both_pose_worsened=bool(not wrong and worse),
        corner_groups=groups,observations=obs)
    check.equal(d,expected,prefix+' posthoc')
    return expected


def summary(values):
    result=dict(n=len(values),status_counts=dict(Counter(r['output_status'] for r in values)),
        new_pose=sum(r['new_pose_estimated'] for r in values),fallback=sum(r['fallback_used'] for r in values),
        no_pose=sum(not r['pose_available'] for r in values),
        wrong_known_mask_both_pose_improved_ids=[r['id'] for r in values if r['wrong_known_mask_both_pose_improved']],
        matching_known_mask_both_pose_worsened_ids=[r['id'] for r in values if r['matching_known_mask_both_pose_worsened']],
        actual_hidden_reprojection_frames=sum(bool(r['actual_reprojected_ids']) for r in values),
        actual_hidden_reprojection_reference_missing=sum(len(r['actual_reprojected_reference_missing_ids']) for r in values),
        hidden_set_changed=sum(r['hidden_set_changed'] for r in values))
    result['mask_pose_groups']={}
    for wrong in (False,True):
        for outcome in OUTCOMES:
            selected=[r for r in values if r['mask_wrong_on_known']==wrong and r['pose_outcome_vs_N3']==outcome]
            key=('wrong_on_known' if wrong else 'matching_on_known')+'__'+outcome
            result['mask_pose_groups'][key]=dict(n=len(selected),ids=[r['id'] for r in selected],
                fully_annotated_mask_count=sum(r['mask_fully_annotated'] for r in selected),new_pose=sum(r['new_pose_estimated'] for r in selected),
                fallback=sum(r['fallback_used'] for r in selected),
                delta_vs_N3={k:moments([r['delta_vs_N3'][k] for r in selected if r['comparable_to_N3']]) for k in POSE})
    result['correspondences']={pool:dict(correct_count=moments([len(r['pools'][pool]['correct_ids']) for r in values]),
        incorrect_count=moments([len(r['pools'][pool]['incorrect_ids']) for r in values]),
        unknown_count=moments([len(r['pools'][pool]['unknown_ids']) for r in values]),
        correct_image_rank_counts=dict(Counter(str(r['pools'][pool]['correct_image_layout']['centered_numerical_rank']) for r in values))) for pool in POOLS}
    result['corner_errors']={}
    for group in GROUPS:
        pair=[(r,k) for r in values for k in r['corner_groups'][group]['N3_to_output_comparable_ids']]
        all_points=[(r,k) for r in values for k in r['corner_groups'][group]['ids']]
        result['corner_errors'][group]=dict(reference_count=len(all_points),comparable_N3_output=len(pair),
            N3_error_px=moments([r['fixed_N3_error_px'][k] for r,k in pair]),
            output_error_same_N3_set_px=moments([r['output_error_px'][k] for r,k in pair]),
            delta_output_minus_N3_px=moments([r['output_error_px'][k]-r['fixed_N3_error_px'][k] for r,k in pair]),
            input_error_available_px=moments([r['input_error_px'][k] for r,k in all_points if math.isfinite(r['input_error_px'][k])]),
            output_error_available_px=moments([r['output_error_px'][k] for r,k in all_points if math.isfinite(r['output_error_px'][k])]),
            damage_N3_le5_to_output_gt10=sum(len(r['corner_groups'][group]['damage_N3_le5_to_output_gt10_ids']) for r in values),
            damaged_frame_ids=[r['id'] for r in values if r['corner_groups'][group]['damage_N3_le5_to_output_gt10_ids']])
    result['observation_counts']={k:moments([r['observations'][k] for r in values])
        for k in ('decoded_line_count','computed_corner_count','line_veto_count')}
    result['observation_counts'].update({k:moments([len(r['observations'][k]) for r in values])
        for k in ('admitted_corner_ids','hybrid_replaced_corner_ids','hybrid_rejected_corner_ids')})
    result['computed_corner_radii_px']=moments([v for r in values for v in r['observations']['computed_corner_radii_px']])
    result['boundary_corner_quality']={}
    for scope in ('computed','admitted','selected_for_hybrid','displayed_as_boundary_observation'):
        selected=[c for r in values for c in r['observations']['boundary_corner_evidence'] if scope=='computed' or c[scope]]
        known=[c for c in selected if c['reference_available']];paired=[c for c in known if c['N3_error_px'] is not None]
        result['boundary_corner_quality'][scope]=dict(total_count=len(selected),known_count=len(known),unknown_count=len(selected)-len(known),
            correct_within8px=sum(c['correct_within8px'] for c in known),error_px=moments([c['error_px'] for c in known]),
            N3_error_same_paired_set_px=moments([c['N3_error_px'] for c in paired]),
            candidate_error_same_paired_set_px=moments([c['error_px'] for c in paired]),
            candidate_minus_N3_error_px=moments([c['error_delta_candidate_minus_N3_px'] for c in paired]),
            improves_vs_N3=sum(c['error_delta_candidate_minus_N3_px'] < -1e-9 for c in paired),
            harms_vs_N3=sum(c['error_delta_candidate_minus_N3_px'] > 1e-9 for c in paired))
    return result


def run(args):
    start=time.monotonic();doc=Path(args.doc);output=Path(args.output) if args.output else doc/'POSTHOC_CHECKS.json'
    assert not output.exists() and not output.is_symlink(), 'Preserve existing check receipt'
    names=('PROTOCOL.json','DIAGNOSTICS.json','POSTHOC_ROWS.jsonl.gz','PREDICTIONS.jsonl.gz','FIXED_PREDICTIONS.jsonl.gz',
           'OBSERVATIONS.jsonl.gz','SCORING_RECEIPT.json','GEOMETRY_SEAL.json')
    paths={name:doc/name for name in names};paths['code']=Path(__file__)
    protocol=read(paths['PROTOCOL.json']);cohort_path=REPO/protocol['fixed_inputs']['cohort']['path']
    paths['cohort']=cohort_path
    before={name:binding(path) for name,path in paths.items()}
    check=Audit();diagnostics=read(paths['DIAGNOSTICS.json']);cohort=read(cohort_path)
    check.require(diagnostics['complete'] and diagnostics['frames']==245 and diagnostics['rows']==1225,'diagnostic completion')
    recorded_code=diagnostics['input_bindings']['code'];actual_code=binding(REPO/recorded_code['path'])
    check.require(recorded_code['sha256']==actual_code['sha256'] and recorded_code['bytes']==actual_code['bytes'],'executed diagnostics code binding')
    seal=read(paths['GEOMETRY_SEAL.json']);receipt=read(paths['SCORING_RECEIPT.json'])
    check.require(seal['complete'] and seal['GT_read_allowed'] is False and receipt['complete'],'seal before scored rows')
    for name,key in [('PREDICTIONS.jsonl.gz','predictions'),('FIXED_PREDICTIONS.jsonl.gz','fixed_predictions')]:
        check.require(receipt[key]['sha256']==before[name]['sha256'] and receipt[key]['bytes']==before[name]['bytes'],name+' scoring SHA')
    for name,key in [('POSTHOC_ROWS.jsonl.gz','rows_binding'),('OBSERVATIONS.jsonl.gz','observations')]:
        expected=diagnostics[key] if key=='rows_binding' else seal[key]
        check.require(expected['sha256']==before[name]['sha256'] and expected['bytes']==before[name]['bytes'],name+' SHA')
    scored=read_rows(paths['PREDICTIONS.jsonl.gz']);posthoc=read_rows(paths['POSTHOC_ROWS.jsonl.gz'])
    fixed=read_rows(paths['FIXED_PREDICTIONS.jsonl.gz']);obs=read_rows(paths['OBSERVATIONS.jsonl.gz'])
    expected_ids=set(cohort['ids']);labels={f['id']:f['label'] for f in cohort['frames']}
    check.require(len(expected_ids)==245 and Counter(labels.values())==dict(clean=153,moderate=92),'frozen difficulty scope')
    for rows,methods in [(scored,METHODS),(posthoc,METHODS),(fixed,('BASE','N3_SUBPIX'))]:
        check.require(len(rows)==len(methods)*245 and len({(r['method'],r['id']) for r in rows})==len(rows),'method unique rows')
        for method in methods:check.require({r['id'] for r in rows if r['method']==method}==expected_ids,'cohort IDs '+method)
    check.require(len(obs)==245 and len({r['id'] for r in obs})==245 and {r['id'] for r in obs}==expected_ids,'observation IDs')
    by_d={(r['method'],r['id']):r for r in posthoc};by_o={r['id']:r for r in obs}
    by_f={r['id']:r for r in fixed if r['method']=='N3_SUBPIX'}
    reconstructed=[verify_row(check,r,by_d[(r['method'],r['id'])],by_o[r['id']],by_f[r['id']],labels[r['id']]) for r in scored]
    checked_summaries=0
    for scope,label in [('combined',None),('easy','clean'),('medium','moderate')]:
        for method in METHODS:
            selected=[r for r in reconstructed if r['method']==method and (label is None or r['difficulty_label']==label)]
            check.equal(diagnostics['strata'][scope][method],summary(selected),scope+'/'+method+' full moments')
            checked_summaries+=1
    after={name:binding(path) for name,path in paths.items()}
    check.require(before==after,'all inputs preserved')
    receipt=dict(schema='independent_boundary_posthoc_arithmetic_check_v2',passed=True,frames=245,
        scored_rows=1225,posthoc_rows=1225,fixed_rows=490,observation_rows=245,strata_method_summaries=checked_summaries,
        scalar_and_structure_checks=check.checks,max_absolute_float_difference=check.max_absolute_float_difference,
        float_tolerance=dict(atol=1e-8,rtol=1e-10),summary_mean_variance='independent math.fsum; sample variance ddof1',
        verified=['known matched proxy-reference 8px input labels and all error arrays','eligible/used/fit/final and accepted-final inlier IDs',
            '2D/3D centered SVD layouts','no-mask/excluded-H states and exact fallback/output coordinate assembly',
            'independent final-line radius veto, corner8px admission, hybrid8px+hidden exclusion',
            'mask disagreement and paired N3 translation/rotation outcomes','boundary computed/admitted/hybrid/applied errors and counts',
            'DIRECT/SELF/actually-reprojected-H errors, damage<=5 to >10, missing references','all 15 full summary moments'],
        limitations=['references are the previously reconstructed geometric proxy, not independent measured physical truth',
            'a supported source edge or query confidence does not certify real physical edge ownership',
            '8px coordinate correctness here is posthoc proxy agreement, not occlusion/physical-boundary truth',
            'correlated corners/line samples are not additional independent 3D observations',
            'fallback candidate inliers remain diagnostics; accepted-new-pose inliers are separately verified'],
        inputs=before,executed_diagnostic_code=actual_code,new_GT_loaders=0,new_detector_forwards=0,new_head_forwards=0,
        new_pose_fits=0,new_rays=0,new_images=0,new_training_updates=0,elapsed_seconds=time.monotonic()-start)
    with output.open('x') as f:json.dump(receipt,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(passed=True,checks=check.checks,rows=1225,summaries=checked_summaries,
                         output=str(output.relative_to(REPO)) if output.is_relative_to(REPO) else output.name)))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--doc',default=str(DOC));p.add_argument('--output')
    run(p.parse_args())


if __name__=='__main__':main()
