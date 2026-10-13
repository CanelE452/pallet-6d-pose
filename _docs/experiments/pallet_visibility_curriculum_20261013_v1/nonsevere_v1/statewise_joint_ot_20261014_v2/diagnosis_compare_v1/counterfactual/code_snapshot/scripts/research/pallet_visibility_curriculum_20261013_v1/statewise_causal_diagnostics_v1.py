"""Controlled POSTSEAL oracle diagnostics. Never a production evaluation.

Only new diagnostic artifacts are written. Existing state models, current19
sources, real predictions, primary population and protected user files are
hash checked before and after. No RGB reads or classifier/mask forwards occur.
"""
from __future__ import annotations
import os
for _thread_setting in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_thread_setting]='1'
import argparse
import gzip
import json
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
from copy import deepcopy
from datetime import datetime,timezone
from itertools import product
from multiprocessing import get_context
from pathlib import Path
import hashlib
import shutil
import time

import numpy as np
from threadpoolctl import threadpool_limits

from . import common as C
from . import joint_ot_pnp as P
from .scoring import pose_metric,distribution,rotation_y
from .statewise_joint_api_v2 import registered_geometry

STUDY=C.DOC/'nonsevere_v1/statewise_joint_ot_20261014_v2'
DIAG=STUDY/'diagnosis_compare_v1'
PLAN=DIAG/'COUNTERFACTUAL_PLAN_V1.json'
CURRENT='CURRENT_CORRECTED_BI_OT'
METRICS=('translation_cm','rotation_deg','ADDsym_cm','success_5cm_5deg')
PIXEL_ARMS={'ORACLE_BINARY_RAW_STANDARD_COV','FULL_HUMAN_RAW_STANDARD_COV',
            'CURRENT_DROP_KNOWN_FALSE_OT','CURRENT_REFERENCE_MODEL_CENTER_O'}


def digest(value):
    import json
    return hashlib.sha256(json.dumps(C.finite(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def binding(path):
    path=Path(path)
    return dict(path=str(path.relative_to(C.ROOT)),sha256=C.sha(path),bytes=path.stat().st_size)

def valid_query(xy,hw):
    q=np.asarray(xy,float);h,w=hw
    return bool(q.shape==(2,) and np.isfinite(q).all() and not np.array_equal(q,[-1.,-1.]) and 0<=q[0]<w and 0<=q[1]<h)

def reference_alignment(row,label):
    truth=label['target']['truth'];xyz=np.asarray(row['runtime']['initial_geometry']['native_xyz']);Q=np.asarray(row['runtime']['initial_geometry']['physical_registration_Q'])
    mapping=np.asarray(label['metadata']['native_to_reference_corner_ids'],int)
    Qref=np.asarray(truth['body_R']).T@np.asarray(truth['R']);mapped=(C.cuboid(*truth['body_xyz'])@Qref)[mapping]
    phases=[]
    for k in range(truth['order']):
        symmetry=rotation_y(k*2*np.pi/truth['order']);gap=float(np.linalg.norm(xyz@Q@symmetry.T-mapped,axis=1).max())
        phases.append((gap,k,symmetry))
    gap,k,symmetry=min(phases,key=lambda v:v[0]);physical=np.asarray(truth['R'])@symmetry;R=physical@Q.T;t=np.asarray(truth['t'])
    q,depth=P.project_native(xyz,R,t,np.asarray(row['K']));bodyq,_=P.project_native(C.cuboid(*truth['body_xyz']),np.asarray(truth['body_R']),t,np.asarray(row['K']));bodyq=bodyq[mapping]
    compatible=gap<1e-6
    if compatible:assert float(np.linalg.norm(q-bodyq,axis=1).max())<1e-8
    gt=np.asarray(label['target']['gt'])[:8][mapping];valid=np.asarray(label['target']['valid'])[:8][mapping]
    return dict(compatible=compatible,physical_native_gap_m=gap,phase=k,
        fixed_phase_selection_is_3D_mapping_only=True,reference_R_cf=R.tolist(),reference_t=t.tolist(),
        reference_projection=q.tolist(),reference_depth=depth.tolist(),mapped_annotation_xy=gt.tolist(),mapped_annotation_valid=valid.tolist(),human_native=label['metadata']['human_states_native'])


def base_input(row,config):
    rt=row['runtime'];geom=rt['initial_geometry'];coarse=row['initial_pose']
    return dict(xyz=deepcopy(geom['native_xyz']),K=deepcopy(row['K']),states=deepcopy(rt['states']),
                visible=deepcopy(rt['frozen_visible_measurements']),domains=deepcopy(rt['ot_domains']['domains']),
                image_hw=deepcopy(row['raw_hw']),initializations=[dict(R=deepcopy(coarse['R_cf']),t=deepcopy(coarse['centroid']))],config=deepcopy(config))


def physics_key(spec):
    # Only quantities actually consumed by the pure solver define reuse.
    visible=[{k:r[k]for k in ('native_corner','xy','covariance_px2','protection_px')}for r in spec['visible']]
    keys=('native_corner','state','shape','center_xy','half_axes_px','angle_deg','bounds_xyxy','halfplanes','outside_canvas_side','outside_margin_px','canvas_hw','normal','offset')
    domains=[{k:d[k]for k in keys if k in d}for d in spec['domains']]
    return digest(dict(xyz=spec['xyz'],K=spec['K'],states=spec['states'],visible=visible,domains=domains,image_hw=spec['image_hw'],initializations=spec['initializations'],config=spec['config']))


def standard_visible(states,q,hw,*,ideal=False):
    records=[];unavailable=[]
    for k,state in enumerate(states):
        if state not in ('B','I'):continue
        if not valid_query(q[k],hw):
            states[k]='U';unavailable.append(k);continue
        sigma=1. if ideal else 2. if state=='B' else 3.
        records.append(dict(native_corner=k,state=state,xy=np.asarray(q[k]).tolist(),
            covariance_px2=[[sigma*sigma,0.],[0.,sigma*sigma]],protection_px=2.,
            source='ORACLE_MODEL_PROJECTION' if ideal else 'RAW_IMAGE_PREDICTION_STANDARD_COV',
            image_measurement=not ideal,measurement_is_pose_projection=ideal))
    return records,unavailable


def diagnostic_domains(states,q,hw,model,scale,*,center_source):
    h,w=hw;radius=max(float(model['minimum_radius_px']),float(scale)*float(model['normalized_radius']));domains=[]
    for k,state in enumerate(states):
        common=dict(native_corner=k,native_id=k,state=state,canvas_hw=list(hw),
            diagnostic_input=True,production_region=False,point_source=center_source,
            raw_scale_held_fixed=True,CAL_model_sha256=model['model_sha256'],GT_coordinates_used=center_source=='ORACLE_MODEL_REFERENCE_PROJECTION')
        if state=='O':
            if valid_query(q[k],hw):
                d=dict(common,shape='ellipse',center_xy=np.asarray(q[k]).tolist(),half_axes_px=[radius,radius],angle_deg=0.,
                       halfplanes=[[-1.,0.,0.],[1.,0.,float(w)],[0.,-1.,0.],[0.,1.,float(h)]])
            else:d=dict(common,shape='box',bounds_xyxy=[0.,0.,float(w),float(h)],halfplanes=[],missing_O_coordinate_broad_canvas=True)
        elif state=='T':d=dict(common,shape='halfplane',outside_canvas_side='unknown',outside_margin_px=0.,halfplanes=[])
        else:continue
        domains.append(d)
    return domains


def transform(row,arm,plan,model,alignment,label):
    if arm=='FULL_HUMAN_RAW_STANDARD_COV_BUDGET6':
        spec,provenance=transform(row,'FULL_HUMAN_RAW_STANDARD_COV',plan,model,alignment,label)
        if alignment['compatible']:spec['config']['maximum_visible_mahalanobis']=6.
        provenance.update(arm=arm,uncertainty_interaction_eligible86_only=True,
                          incompatible9_keep_exact_FULL_HUMAN_standard3_control=True)
        return spec,provenance
    spec=base_input(row,plan['solver_configuration']);rt=row['runtime'];raw=np.asarray(rt['original_raw_points'])[:8];hw=row['raw_hw'];states=spec['states'];roles=rt['initial_geometry']['initial_roles'];human=alignment['human_native'];scale=rt['ot_domains']['nonself_prediction_scale_px'];notes=[]
    compatible=alignment['compatible'];oracle=False;unchanged=False;missing=[]
    if arm in PIXEL_ARMS and not compatible:
        if arm in {'ORACLE_BINARY_RAW_STANDARD_COV','FULL_HUMAN_RAW_STANDARD_COV'}:
            # Identical RAW_STANDARD control, not identical corrected-current arm.
            for m in spec['visible']:m['xy']=raw[m['native_corner']].tolist();m['covariance_px2']=(np.eye(2)*(2. if m['state']=='B' else 3.)**2).tolist()
        unchanged=True;notes.append('incompatible_native_reference_mapping_exact_control_reused')
    elif arm in {'RAW_BI_OT','RAW_BI_NO_OT','CURRENT_RAW_STANDARD_COV'}:
        for m in spec['visible']:
            m['xy']=raw[m['native_corner']].tolist()
            if arm=='CURRENT_RAW_STANDARD_COV':m['covariance_px2']=(np.eye(2)*(2. if m['state']=='B' else 3.)**2).tolist()
        if arm=='RAW_BI_NO_OT':spec['states']=['U'if s in ('O','T')else s for s in states];spec['domains']=[]
    elif arm in {'ORACLE_BINARY_RAW_STANDARD_COV','FULL_HUMAN_RAW_STANDARD_COV'}:
        oracle=True
        new=[]
        for k,s in enumerate(states):
            if arm=='ORACLE_BINARY_RAW_STANDARD_COV' and s in ('S','T'):new.append(s)
            elif human[k]=='SELF_OCCLUDED':new.append('S'if arm=='FULL_HUMAN_RAW_STANDARD_COV'else'U')
            elif human[k]=='DIRECT_VISIBLE':new.append(roles[k])
            elif human[k]=='EXTERNAL_OCCLUDED':new.append('O')
            else:new.append('U')
        spec['states']=new;spec['visible'],missing=standard_visible(new,raw,hw)
        spec['domains']=diagnostic_domains(new,raw,hw,model,scale,center_source='RAW_IMAGE_PREDICTION')
        # Frozen conditional T geometry is retained exactly for binary control.
        if arm=='ORACLE_BINARY_RAW_STANDARD_COV':
            oldT={d['native_corner']:d for d in row['runtime']['ot_domains']['domains']if d['state']=='T'}
            spec['domains']=[deepcopy(oldT[d['native_corner']])if d['state']=='T'else d for d in spec['domains']]
    elif arm=='CURRENT_DROP_KNOWN_FALSE_OT':
        oracle=True;removed=[];gt=np.asarray(alignment['mapped_annotation_xy']);valid=alignment['mapped_annotation_valid']
        for d in spec['domains']:
            k=d['native_corner'];falseO=d['state']=='O'and human[k]in('DIRECT_VISIBLE','SELF_OCCLUDED')
            falseT=d['state']=='T'and human[k]in('DIRECT_VISIBLE','EXTERNAL_OCCLUDED')and valid[k]and valid_query(gt[k],hw)
            if falseO or falseT:removed.append(k);states[k]='U'
        spec['domains']=[d for d in spec['domains']if d['native_corner']not in removed];notes.append(dict(removed_known_false_OT_ids=removed))
    elif arm=='CURRENT_BI_BUDGET6':spec['config']['maximum_visible_mahalanobis']=6.
    elif arm=='CURRENT_REFERENCE_MODEL_CENTER_O':
        oracle=True
        for d in spec['domains']:
            if d['state']=='O':
                assert d['shape']=='ellipse','Only an existing O center may move; no fabricated shape/radius'
                d['center_xy']=deepcopy(alignment['reference_projection'][d['native_corner']]);d['diagnostic_input']=True;d['GT_coordinates_used']=True;d['point_source']='ORACLE_MODEL_REFERENCE_PROJECTION'
    elif arm=='CURRENT_REFERENCE_INIT':
        oracle=True;truth=label['target']['truth'];Q=np.asarray(rt['initial_geometry']['physical_registration_Q'])
        for k in range(truth['order']):spec['initializations'].append(dict(name='ORACLE_PHYSICAL_REFERENCE_SYMMETRY_'+str(k),R=(np.asarray(truth['R'])@rotation_y(k*2*np.pi/truth['order'])@Q.T).tolist(),t=deepcopy(truth['t'])))
    elif arm=='MODEL_IDEAL_PROJECTIONS':
        oracle=True
        # For incompatible mappings this is an explicitly constructed virtual
        # native model positive control, not a human-coordinate oracle.
        Q=np.asarray(rt['initial_geometry']['physical_registration_Q'])
        R=np.asarray(alignment['reference_R_cf']) if compatible else np.asarray(label['target']['truth']['R'])@Q.T
        t=np.asarray(alignment['reference_t']);pose=dict(R_cf=R,R_physical=R@Q,centroid=t,cf_extents=row['initial_pose']['cf_extents'])
        geo=registered_geometry(hw,row['K'],rt['initial_geometry']['native_xyz'],pose);q=np.asarray(geo['initial_projection_xy']);new=[]
        for k in range(8):
            if geo['initial_self_hidden'][k]:new.append('S')
            elif not valid_query(q[k],hw):new.append('T')
            elif compatible and human[k]=='EXTERNAL_OCCLUDED':new.append('O')
            else:new.append(geo['initial_roles'][k])
        spec['states']=new;spec['visible'],missing=standard_visible(new,q,hw,ideal=True);spec['domains']=diagnostic_domains(new,q,hw,model,scale,center_source='ORACLE_MODEL_REFERENCE_PROJECTION')
        spec['initializations'].append(dict(name='ORACLE_CONSTRUCTION_REFERENCE',R=R.tolist(),t=t.tolist()))
    else:raise ValueError('Unknown declared diagnostic arm: '+arm)
    assert set(m['native_corner']for m in spec['visible'])=={k for k,s in enumerate(spec['states'])if s in('B','I')}
    assert set(d['native_corner']for d in spec['domains'])=={k for k,s in enumerate(spec['states'])if s in('O','T')}
    qhash=digest(spec['visible']);original=digest(rt['frozen_visible_measurements']);dhash=digest(spec['domains']);old_domains=digest(rt['ot_domains']['domains'])
    provenance=dict(arm=arm,postseal_exploratory_diagnostic=True,production_evaluation=False,oracle_inputs_used=oracle,
        native_mapping_compatible=compatible,incompatible_control_reused=unchanged,raw_nonself_scale_held_fixed=scale,
        visible_input_sha256=qhash,original_visible_input_sha256=original,domains_sha256=dhash,original_domains_sha256=old_domains,
        visible_unchanged=qhash==original,domains_unchanged=dhash==old_domains,missing_or_outside_visible_queries_become_U=missing,
        no_RGB_forward=True,no_new_parameters=True,no_target_based_parameter_selection=True,notes=notes)
    return spec,provenance


def pose_from_solution(row,solution):
    coarse=row['initial_pose'];seed=solution.get('corrected_BI_seed')or{};accepted=bool(solution['accepted'])
    if accepted:R,t=np.asarray(solution['R']),np.asarray(solution['t']);source='DIAGNOSTIC_JOINT_FEASIBLE_POSE_FAMILY_REPRESENTATIVE'if(solution.get('ambiguity')or{}).get('detected')else'DIAGNOSTIC_JOINT_FEASIBLE'
    elif seed.get('visible_support_guards_pass')and not seed.get('contains_underconstrained_pose_family'):
        R,t=np.asarray(seed['R']),np.asarray(seed['t']);source='EXPLICIT_VALID_BI_SEED_FALLBACK'
    else:R,t=np.asarray(coarse['R_cf']),np.asarray(coarse['centroid']);source='EXPLICIT_ORIGINAL_COARSE_FALLBACK'
    Q=np.asarray(coarse['R_cf']).T@np.asarray(coarse['R_physical'])
    return dict(available=True,R_cf=R.tolist(),R_physical=(R@Q).tolist(),centroid=t.tolist(),cf_extents=deepcopy(coarse['cf_extents']),
        source=source,joint_feasible=accepted,fallback=not accepted,OT_unique_solution_claimed=False,
        material_ambiguity=bool((solution.get('ambiguity')or{}).get('detected')))


def compact(solution):
    s=deepcopy(solution);journal=s.pop('candidates',[])
    s['candidate_journal_summary']=dict(count=len(journal),feasible_count=sum(v.get('joint_feasible',False)for v in journal),
        complete_journal_sha256=digest(journal),all_distinct_feasible_samples_retained=True)
    return s


def reference_audit(row,spec,label):
    cfg=dict(P._DEFAULTS,**spec['config']);xyz,K,states,measurements,domains,hw=P._inputs(spec['xyz'],spec['K'],spec['states'],spec['visible'],spec['domains'],spec['image_hw'],cfg)
    truth=label['target']['truth'];Q=np.asarray(row['runtime']['initial_geometry']['physical_registration_Q']);scale=max(float(np.linalg.norm(np.ptp(xyz,axis=0))),1e-3);reference=np.asarray(row['runtime']['initial_geometry']['initial_projection_xy']);audits=[]
    alignment=reference_alignment(row,label)
    for phase in range(truth['order']):
        R=np.asarray(truth['R'])@rotation_y(phase*2*np.pi/truth['order'])@Q.T;t=np.asarray(truth['t']);q,depth=P.project_native(xyz,R,t,K);sides=[]
        for d in domains:
            choice=d['outside_canvas_side'];options=choice if isinstance(choice,tuple)else(choice,)if choice is not None else P._SIDES
            if d['state']!='T':sides.append(None)
            else:sides.append(max(options,key=lambda side:min(P._domain_slacks(d,q[d['native_corner']],side,hw,cfg['outside_canvas_margin_px'])[0])))
        values=P._constraint_values(P._pose_vector(R,t,scale),xyz,K,measurements,domains,tuple(sides),reference,hw,scale,cfg)
        ma=P._measurement_audit(q,measurements,reference,cfg,-R.T@t,hw);da=P._domain_audit(q,domains,tuple(sides),hw,cfg)
        objective=P._objective(P._pose_vector(R,t,scale),xyz,K,measurements,scale,cfg['huber_scale'])if measurements else 0.
        audits.append(dict(physical_symmetry_phase=phase,matches_fixed_native_reference_phase=bool(alignment['compatible']and phase==alignment['phase']),R_cf=R.tolist(),t=t.tolist(),objective=objective,
            minimum_normalized_constraint_slack=float(min(values)),positive_depth_pass=bool(min(depth)>=cfg['positive_depth_m']-1e-10),
            minimum_depth_m=float(min(depth)),visible=ma,domains=da,all_constraints_pass=bool(np.isfinite(values).all()and min(values)>=-cfg['feasibility_tolerance']and min(depth)>=cfg['positive_depth_m']-1e-10),
            failed_BI_mahalanobis_ids=[a['native_corner']for a in ma if not a['measurement_budget_pass']],failed_BI_geometry_ids=[a['native_corner']for a in ma if not a['visible_geometry_pass']],failed_BI_canvas_ids=[a['native_corner']for a in ma if not a['visible_canvas_pass']],failed_OT_ids=[a['native_corner']for a in da if not a['feasible']]))
    return dict(allowed_physical_representatives=audits,any_representative_feasible=any(a['all_constraints_pass']for a in audits),
        native_mapping_compatible=alignment['compatible'],fixed_native_phase=alignment['phase']if alignment['compatible']else None,
        fixed_native_reference_feasible=audits[alignment['phase']]['all_constraints_pass']if alignment['compatible']else None,
        precise_reference_is_historical_derived_not_independently_measured=True,GT_used_for_postseal_diagnostic=True)


def solve_task(task):
    key,spec=task
    with threadpool_limits(limits=1):
        before=digest(spec);s=P.solve_joint_ot_pnp(spec['xyz'],spec['K'],spec['states'],spec['visible'],spec['domains'],spec['image_hw'],initializations=spec['initializations'],config=spec['config'])
        assert digest(spec)==before
    return key,compact(s)


def selector_diagnostics(row,label):
    solution=row['runtime']['joint_pose'];samples=deepcopy(solution.get('feasible_pose_samples')or[]);xyz=np.asarray(row['runtime']['initial_geometry']['native_xyz']);K=np.asarray(row['K']);coarse=np.asarray(row['runtime']['initial_geometry']['initial_projection_xy']);scale=max(float(np.linalg.norm(np.ptp(coarse,axis=0))),16.);Q=np.asarray(row['runtime']['initial_geometry']['physical_registration_Q']);truth=label['target']['truth']
    records=[]
    for index,s in enumerate(samples):
        projected,_=P.project_native(xyz,np.asarray(s['R']),np.asarray(s['t']),K)
        distance=float(np.sqrt(np.mean(np.sum((projected-coarse)**2,axis=1)))/scale)
        pose=dict(R_cf=s['R'],R_physical=(np.asarray(s['R'])@Q).tolist(),centroid=s['t'],cf_extents=row['initial_pose']['cf_extents'])
        records.append(dict(index=index,normalized_all8_projection_RMS_to_coarse=distance,objective=s['objective'],pose=pose,metrics=pose_metric(pose,truth)))
    if not records:return dict(pool_count=0,SELECT_CLOSEST_COARSE_FIXED=dict(reused_current_fallback=True,actual_pose=deepcopy(row['runtime']['actual_pose'])),ORACLE_BEST_FOUND=dict(reused_current_fallback=True,actual_pose=deepcopy(row['runtime']['actual_pose'])),candidate_metrics=[])
    nearest=min(records,key=lambda r:(r['normalized_all8_projection_RMS_to_coarse'],r['index']));oracle=min(records,key=lambda r:(r['metrics']['translation_cm'],r['index']))
    return dict(pool_count=len(records),exact_current_feasible_pool=True,GT_not_used_for_nearest_selector=True,all_S_raw_queries_unavailable=True,
        SELECT_CLOSEST_COARSE_FIXED=dict(index=nearest['index'],actual_pose=nearest['pose'],normalized_distance=nearest['normalized_all8_projection_RMS_to_coarse'],reused_current_fallback=False,material_ambiguity=bool((solution.get('ambiguity')or{}).get('detected'))),
        ORACLE_BEST_FOUND=dict(index=oracle['index'],actual_pose=oracle['pose'],selected_metric='minimum_translation_cm',oracle_not_deployable=True,reused_current_fallback=False,per_metric_best_found_lower_bounds={k:min(float(r['metrics'][k])for r in records)for k in METRICS[:3]}),candidate_metrics=records)


def paired(rows,control,metric):
    sessions=sorted({r['session']for r in rows});groups={s:[r for r in rows if r['session']==s]for s in sessions};delta=np.array([float(r['metrics'][metric])-float(control[r['id']]['metrics'][metric])for r in rows]);rng=np.random.default_rng(20261014);draw=rng.integers(0,len(sessions),size=(10000,len(sessions)));sums=np.array([sum(float(r['metrics'][metric])-float(control[r['id']]['metrics'][metric])for r in groups[s])for s in sessions]);counts=np.array([len(groups[s])for s in sessions]);boot=sums[draw].sum(1)/counts[draw].sum(1)
    improve=delta>1e-8 if metric=='success_5cm_5deg'else delta<-1e-8;worse=delta<-1e-8 if metric=='success_5cm_5deg'else delta>1e-8
    return dict(mean_difference=float(delta.mean()),session_cluster_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),improved=int(improve.sum()),worsened=int(worse.sum()),tied=int(len(rows)-improve.sum()-worse.sum()))


def score(output):
    output=Path(output);seal=C.read(output/'PREDICTION_SEAL.json');assert C.sha(output/'PREDICTIONS.jsonl.gz')==seal['predictions_sha256']and C.sha(output/'PROTOCOL.json')==seal['protocol_sha256'];protocol=C.read(output/'PROTOCOL.json')
    assert all(C.sha(C.ROOT/p)==e for p,e in protocol['source_sha256'].items())
    assert all(C.sha(C.ROOT/p)==e for p,e in protocol['evidence_sha256'].items())
    assert all(C.sha(C.ROOT/v['path'])==v['sha256']for v in protocol['label_bindings'].values())
    assert all(C.sha(C.ROOT/p)==e for p,e in protocol['protected_user_sha256'].items())
    labels={}
    for part in ('CAL','TEST'):labels.update(C.read(C.ROOT/protocol['label_bindings'][part]['path'])['rows'])
    records=list(C.rows(output/'PREDICTIONS.jsonl.gz'));groups=defaultdict(list)
    for r in records:
        truth=labels[r['id']]['target']['truth'];r['metrics']=pose_metric(r['actual_pose'],truth);groups[r['arm']].append(r)
    summaries={};pairs={};control={r['id']:r for r in groups[CURRENT]}
    for arm,rs in groups.items():
        summaries[arm]={};pairs[arm]={}
        for part in ('CAL','TEST'):
            selected=[r for r in rs if r['partition']==part];summaries[arm][part]=dict(frames=len(selected),success_count=sum(r['metrics']['success_5cm_5deg']for r in selected),accepted=sum(r['accepted']for r in selected),material_ambiguity=sum(r['material_ambiguity']for r in selected),metrics={k:distribution([float(r['metrics'][k])for r in selected])for k in METRICS},compatible_subset={k:distribution([float(r['metrics'][k])for r in selected if r['compatible']])for k in METRICS})
            pairs[arm][part]=dict(by_metric={k:paired(selected,control,k)for k in METRICS},success_recover_ids=[r['id']for r in selected if r['metrics']['success_5cm_5deg']and not control[r['id']]['metrics']['success_5cm_5deg']],success_loss_ids=[r['id']for r in selected if not r['metrics']['success_5cm_5deg']and control[r['id']]['metrics']['success_5cm_5deg']])
    matched=[('RAW_BI_OT',CURRENT),('RAW_BI_NO_OT','CURRENT_CORRECTED_BI_NO_OT'),('RAW_BI_OT','RAW_BI_NO_OT'),(CURRENT,'CURRENT_CORRECTED_BI_NO_OT'),('CURRENT_RAW_STANDARD_COV','RAW_BI_OT'),('ORACLE_BINARY_RAW_STANDARD_COV','CURRENT_RAW_STANDARD_COV'),('FULL_HUMAN_RAW_STANDARD_COV','CURRENT_RAW_STANDARD_COV')];matched_out={}
    if 'FULL_HUMAN_RAW_STANDARD_COV_BUDGET6'in groups:matched.append(('FULL_HUMAN_RAW_STANDARD_COV_BUDGET6','FULL_HUMAN_RAW_STANDARD_COV'))
    for a,b in matched:
        ref={r['id']:r for r in groups[b]};matched_out[a+'-'+b]={p:dict(all_frames={k:paired([r for r in groups[a]if r['partition']==p],ref,k)for k in METRICS},compatible_only={k:paired([r for r in groups[a]if r['partition']==p and r['compatible']],ref,k)for k in METRICS})for p in('CAL','TEST')}
    C.save_rows(output/'SCORED_ROWS.jsonl.gz',records);C.write(output/'POSTSEAL_SCORING.json',dict(schema='ORACLE_DIAGNOSTIC_POSTSEAL_METRICS_V1',summary=summaries,paired_current_control=pairs,matched_factorial_controls=matched_out,production_or_generalization_claim=False,all95_including_failures=True,oracle_inputs_used_explicitly=True,prediction_seal_sha256=C.sha(output/'PREDICTION_SEAL.json')))
    print({arm:{p:(round(v['metrics']['translation_cm']['mean'],4),round(v['metrics']['rotation_deg']['mean'],4),v['success_count'])for p,v in s.items()}for arm,s in summaries.items()},flush=True)


def predict(output,plan_path=PLAN,workers=3):
    output=Path(output).resolve();plan_path=Path(plan_path).resolve()
    if output.exists():raise FileExistsError('Preserve previous diagnostic run')
    plan=C.read(plan_path);assert workers==plan['execution']['independent_CPU_workers']==3
    assert all(C.sha(C.ROOT/p)==e for p,e in plan['frozen_source_sha256'].items())
    assert all(C.sha(C.ROOT/p)==e for p,e in plan['evidence_sha256'].items())
    protected=C.read(C.DOC/'STATEWISE_PUBLICATION_REMOTE_RECEIPT_V1.json')['user_files_preserved_sha256'];assert all(C.sha(C.ROOT/p)==e for p,e in protected.items())
    cached=list(C.rows(STUDY/'evaluation_v3/PREDICTIONS.jsonl.gz'));cached=[r for r in cached if r['cohort']=='PRIMARY_NONSEVERE'];assert [r['id']for r in cached]==plan['primary_ids']and len(cached)==95
    labels={};label_bindings={}
    # Explicit ORACLE inputs are already postseal targets. Bind them before tasks.
    for part,path in [('CAL',STUDY.parent/'OOF_pose_v1/ISOLATED_CAL_SCORING.json'),('TEST',STUDY.parent/'test_reused_v1/ISOLATED_TEST_SCORING.json')]:labels.update(C.read(path)['rows']);label_bindings[part]=binding(path)
    model_path=STUDY/'ot_domains/EXTERNAL_O_CAL_MAX_MODEL_N3_SUBPIX.json';model=C.read(model_path);all_spec=[];solutions={};tasks={};reuse={};alignments={}
    for row in cached:
        i=row['id'];a=reference_alignment(row,labels[i]);alignments[i]=a;assert a['compatible']==(i in plan['pixel_oracle_eligible_ids']);rt=row['runtime'];base=base_input(row,plan['solver_configuration']);key=physics_key(base);solutions[key]=compact(rt['joint_pose']);reuse[key]=dict(source='CURRENT_SEALED_CACHE',id=i)
        noOT=deepcopy(base);noOT['states']=['U'if s in('O','T')else s for s in noOT['states']];noOT['domains']=[];k=physics_key(noOT);solutions[k]=compact(row['BI_only']['solution']);reuse[k]=dict(source='CURRENT_BI_ONLY_SEALED_CACHE',id=i)
        for arm in plan['diagnostic_arms']:
            spec,provenance=transform(row,arm['name'],plan,model,a,labels[i]);key=physics_key(spec)
            all_spec.append(dict(id=i,arm=arm['name'],spec=spec,provenance=provenance,physics_key=key))
            tasks.setdefault(key,spec)
    assert sum(a['compatible']for a in alignments.values())==86
    tasks={k:s for k,s in tasks.items()if k not in solutions}
    output.mkdir(parents=True);source_files=dict(plan['frozen_source_sha256']);source_files[str(Path(__file__).resolve().relative_to(C.ROOT))]=C.sha(__file__)
    test_path=Path(__file__).parent/'test_statewise_causal_diagnostics_v1.py';source_files[str(test_path.relative_to(C.ROOT))]=C.sha(test_path)
    protocol=dict(schema='POSTSEAL_ORACLE_CAUSAL_DIAGNOSTICS_PROTOCOL_V1',plan=binding(plan_path),plan_exact=plan,
        source_sha256=source_files,evidence_sha256=plan['evidence_sha256'],label_bindings=label_bindings,
        diagnostic_state_coordinate_reference_GT_inputs_explicit=True,production_benchmark=False,no_RGB_reads=True,
        all95=95,compatible86=86,workers=3,threads_per_worker=1,raw_scale_frozen_from_current=True,
        original_model_and_C2_metric_preserved=True,unique_solver_jobs=len(tasks),diagnostic_arm_frame_inputs=len(all_spec),
        focused_contract_tests=dict(count=4,status='passed_before_execution',source=binding(test_path)),
        incompatible_ideal_model_policy='physical reference phase0 in current native Q basis; constructed native model positive control; no human E labeling on incompatible9',
        GT_best_found_selection_metric='minimum_translation_cm; other metric lower bounds separately, not one pose simultaneously best',protected_user_sha256=protected)
    C.write(output/'PROTOCOL.json',protocol);C.save_rows(output/'DIAGNOSTIC_INPUTS.jsonl.gz',all_spec);C.write(output/'INPUT_SEAL.json',dict(before_new_solves=True,inputs_sha256=C.sha(output/'DIAGNOSTIC_INPUTS.jsonl.gz'),protocol_sha256=C.sha(output/'PROTOCOL.json'),oracle_GT_inputs_explicit=True))
    for relative,e in source_files.items():
        target=output/'code_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(C.ROOT/relative,target);assert C.sha(target)==e
    start=time.monotonic();done=0
    with gzip.open(output/'PURE_SOLVER_CHECKPOINTS.jsonl.gz','wt',compresslevel=5)as checkpoints,ProcessPoolExecutor(max_workers=3,mp_context=get_context('spawn'))as pool:
        futures={pool.submit(solve_task,(k,s)):k for k,s in tasks.items()}
        for future in as_completed(futures):
            key,s=future.result();assert key==futures[future];solutions[key]=s;done+=1
            checkpoints.write(json.dumps(C.finite(dict(physics_key=key,solution=s)),separators=(',',':'),allow_nan=False)+'\n');checkpoints.flush()
            C.write(output/'LIVE_PROGRESS.json',dict(completed_unique_jobs=done,total_unique_jobs=len(tasks),elapsed_seconds=time.monotonic()-start))
            if done%10==0 or done==len(tasks):print(f'{done}/{len(tasks)} unique pure solves elapsed={time.monotonic()-start:.1f}s',flush=True)
    by_id={r['id']:r for r in cached};records=[]
    for row in cached:
        i=row['id'];a=alignments[i]
        for arm,pose,s in [('ORIGINAL_PNP',row['initial_pose'],None),('CURRENT_CORRECTED_BI_NO_OT',row['BI_only']['actual_pose'],row['BI_only']['solution']),(CURRENT,row['runtime']['actual_pose'],row['runtime']['joint_pose'])]:
            records.append(dict(id=i,session=row['session'],partition=row['partition'],arm=arm,actual_pose=deepcopy(pose),accepted=bool(s and s['accepted']),material_ambiguity=bool(s and(s.get('ambiguity')or{}).get('detected')),compatible=a['compatible'],provenance=dict(frozen_real_cached_control=True,production_prediction_preserved=True),solution=compact(s)if s else None))
    for item in all_spec:
        row=by_id[item['id']];s=deepcopy(solutions[item['physics_key']]);pose=pose_from_solution(row,s)
        s['provenance'].update(diagnostic_GT_arguments_supplied=bool(item['provenance']['oracle_inputs_used']),
            GT_inputs_field_describes_solver_internal_target_file_reads_only=True,
            caller_oracle_inputs_recorded_in_outer_provenance=True,production_evaluation=False)
        record=dict(id=item['id'],session=row['session'],partition=row['partition'],arm=item['arm'],actual_pose=pose,accepted=bool(s['accepted']),material_ambiguity=bool((s.get('ambiguity')or{}).get('detected')),compatible=alignments[item['id']]['compatible'],physics_key=item['physics_key'],solver_input=deepcopy(item['spec']),provenance=dict(item['provenance'],pure_input_reuse=reuse.get(item['physics_key']),input_exactly_unchanged_after_solver=True),solution=s,reference_constraint_audit=reference_audit(row,item['spec'],labels[item['id']]))
        if item['arm']=='MODEL_IDEAL_PROJECTIONS':
            construction=item['spec']['initializations'][-1];scale=max(float(np.linalg.norm(np.ptp(np.asarray(item['spec']['xyz']),axis=0))),1e-3)
            construction_phase=alignments[item['id']]['phase']if alignments[item['id']]['compatible']else 0
            record['ideal_model_consistency']=dict(construction_phase=construction_phase,
                construction_reference_constraints_feasible=record['reference_constraint_audit']['allowed_physical_representatives'][construction_phase]['all_constraints_pass'],
                construction_reference_retained_in_found_samples=any(P._same_pose((np.asarray(construction['R']),np.asarray(construction['t'])),(np.asarray(v['R']),np.asarray(v['t'])),scale)for v in s.get('feasible_pose_samples',[])),
                accepted_ambiguous_is_possible=bool((s.get('ambiguity')or{}).get('detected')),
                exact_reference_pose_error_not_required_for_ambiguous_case=True)
        records.append(record)
    selectors=[]
    for row in cached:
        selection=selector_diagnostics(row,labels[row['id']]);selectors.append(dict(id=row['id'],partition=row['partition'],selection=selection))
        for arm in('SELECT_CLOSEST_COARSE_FIXED','ORACLE_BEST_FOUND'):
            selected=selection[arm];records.append(dict(id=row['id'],session=row['session'],partition=row['partition'],arm=arm,actual_pose=selected['actual_pose'],accepted=bool(row['runtime']['accepted']),material_ambiguity=bool(row['runtime']['ambiguity_audit']['material_ambiguity']),compatible=alignments[row['id']]['compatible'],provenance=dict(oracle_GT_selection=arm=='ORACLE_BEST_FOUND',exact_cached_pool_only=True,new_solves=False),selector=selected))
    assert Counter(r['arm']for r in records)=={arm:95 for arm in [*plan['frozen_real_arms'],*[a['name']for a in plan['diagnostic_arms']],*plan['selector_diagnostics']]}
    assert all(C.sha(C.ROOT/p)==e for p,e in source_files.items())and all(C.sha(C.ROOT/p)==e for p,e in plan['evidence_sha256'].items())and all(C.sha(C.ROOT/p)==e for p,e in protected.items())
    C.save_rows(output/'PREDICTIONS.jsonl.gz',records);C.save_rows(output/'CACHED_SELECTOR_POOLS.jsonl.gz',selectors);C.write(output/'PREDICTION_SEAL.json',dict(schema='ORACLE_DIAGNOSTIC_PREDICTION_SEAL_V1',before_new_metric_aggregation=True,oracle_GT_inputs_explicit=True,production_evaluation=False,rows=len(records),predictions_sha256=C.sha(output/'PREDICTIONS.jsonl.gz'),protocol_sha256=C.sha(output/'PROTOCOL.json'),input_seal_sha256=C.sha(output/'INPUT_SEAL.json'),source_sha256=source_files,severe_frames=0,elapsed_seconds=time.monotonic()-start));print('DIAGNOSTIC_PREDICTIONS_SEALED',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('predict','score'));parser.add_argument('--output',type=Path,default=DIAG/'counterfactual');parser.add_argument('--plan',type=Path,default=PLAN);args=parser.parse_args()
    if args.action=='predict':predict(args.output,args.plan)
    else:score(args.output)

if __name__=='__main__':main()
