"""Independent read-only controls audit; no RGB reads or solver optimization."""
from pathlib import Path
from copy import deepcopy
import json, gzip, hashlib, collections
import numpy as np
from scripts.research.pallet_visibility_curriculum_20261013_v1 import common as C
from scripts.research.pallet_visibility_curriculum_20261013_v1 import joint_ot_pnp as P
from scripts.research.pallet_visibility_curriculum_20261013_v1 import statewise_causal_diagnostics_v1 as D

N=C.DOC/'nonsevere_v1/statewise_joint_ot_20261014_v2'
HERE=N/'diagnosis_compare_v1/theory_review'
plan=C.read(N/'diagnosis_compare_v1/COUNTERFACTUAL_PLAN_V1.json')
source=Path(D.__file__).resolve();source_before=C.sha(source)
labels={}
for p in (N.parent/'OOF_pose_v1/ISOLATED_CAL_SCORING.json',N.parent/'test_reused_v1/ISOLATED_TEST_SCORING.json'):
    labels.update(C.read(p)['rows'])
rows=[r for r in C.rows(N/'evaluation_v3/PREDICTIONS.jsonl.gz') if r['cohort']=='PRIMARY_NONSEVERE']
assert [r['id'] for r in rows]==plan['primary_ids']
model=C.read(N/'ot_domains/EXTERNAL_O_CAL_MAX_MODEL_N3_SUBPIX.json')
checks=collections.defaultdict(lambda: {'checked':0,'failures':[]})
def ck(key, value, rid):
    checks[key]['checked']+=1
    if not value:checks[key]['failures'].append(rid)
def same_except(a,b,key):
    a,b=deepcopy(a),deepcopy(b);a.pop(key);b.pop(key);return a==b
def input_digest(row):
    r=row['runtime']
    return D.digest(dict(initial_pose=row['initial_pose'],K=row['K'],raw_hw=row['raw_hw'],
        runtime={k:r[k] for k in ('initial_geometry','states','original_raw_points','frozen_visible_measurements','ot_domains')}))
compatible=collections.Counter();arm_names=[a['name'] for a in plan['diagnostic_arms']]
keys=[];current_keys=set();ideal_bi=collections.Counter();oracle_flags=collections.Counter()
for row in rows:
    rid=row['id'];rt=row['runtime'];before=input_digest(row);label=labels[rid]
    align=D.reference_alignment(row,label)
    ck('fixed_native_compatible_eligibility',align['compatible']==(rid in plan['pixel_oracle_eligible_ids']),rid)
    if align['compatible']:compatible[row['partition']]+=1
    base=D.base_input(row,plan['solver_configuration']);current_keys.add(D.physics_key(base))
    no=deepcopy(base);no['states']=['U' if s in ('O','T') else s for s in base['states']];no['domains']=[];current_keys.add(D.physics_key(no))
    specs={};prov={}
    for arm in arm_names:
        spec,p=D.transform(row,arm,plan,model,align,label);specs[arm]=spec;prov[arm]=p;keys.append(D.physics_key(spec))
        cfg=dict(P._DEFAULTS,**spec['config'])
        P._inputs(spec['xyz'],spec['K'],spec['states'],spec['visible'],spec['domains'],spec['image_hw'],cfg)
        ck('solver_input_contract',True,rid+':'+arm)
        ck('no_S_visible_or_domain_record',all(spec['states'][m['native_corner']] in ('B','I') for m in spec['visible']) and all(spec['states'][d['native_corner']] in ('O','T') for d in spec['domains']),rid+':'+arm)
        ck('fixed_K_native_dimensions_scale',spec['xyz']==base['xyz'] and spec['K']==base['K'] and spec['image_hw']==base['image_hw'] and p['raw_nonself_scale_held_fixed']==rt['ot_domains']['nonself_prediction_scale_px'],rid+':'+arm)
        ck('no_input_mutation',input_digest(row)==before,rid+':'+arm)
        oracle_flags[(arm,p['oracle_inputs_used'])]+=1
    raw=deepcopy(base)
    for m in raw['visible']:m['xy']=deepcopy(rt['original_raw_points'][m['native_corner']])
    ck('raw_xy_only_same_covariance',specs['RAW_BI_OT']==raw,rid)
    raw_no=deepcopy(raw);raw_no['states']=no['states'];raw_no['domains']=[]
    ck('raw_noOT_exact_matched_control',specs['RAW_BI_NO_OT']==raw_no,rid)
    std=deepcopy(raw)
    for m in std['visible']:
        v=(2 if m['state']=='B' else 3)**2;m['covariance_px2']=[[float(v),0.],[0.,float(v)]]
    ck('standard_raw_covariance_only',specs['CURRENT_RAW_STANDARD_COV']==std,rid)
    budget=deepcopy(base);budget['config']['maximum_visible_mahalanobis']=6.
    ck('current_budget6_only_cap',specs['CURRENT_BI_BUDGET6']==budget,rid)
    full=specs['FULL_HUMAN_RAW_STANDARD_COV'];joint=specs['FULL_HUMAN_RAW_STANDARD_COV_BUDGET6']
    expected=deepcopy(full)
    if align['compatible']:expected['config']['maximum_visible_mahalanobis']=6.
    ck('fullhuman_budget6_exact_interaction',joint==expected,rid)
    if not align['compatible']:
        for arm in ('ORACLE_BINARY_RAW_STANDARD_COV','FULL_HUMAN_RAW_STANDARD_COV','FULL_HUMAN_RAW_STANDARD_COV_BUDGET6'):
            ck('incompatible9_exact_standard_control',specs[arm]==std,rid+':'+arm)
        for arm in ('CURRENT_DROP_KNOWN_FALSE_OT','CURRENT_REFERENCE_MODEL_CENTER_O'):
            ck('incompatible9_exact_current_control',specs[arm]==base,rid+':'+arm)
    else:
        ck('binary_initial_ST_retained',all(specs['ORACLE_BINARY_RAW_STANDARD_COV']['states'][i]==s for i,s in enumerate(base['states']) if s in ('S','T')),rid)
        ck('fullhuman_no_manufactured_T','T' not in full['states'],rid)
        for arm in ('ORACLE_BINARY_RAW_STANDARD_COV','FULL_HUMAN_RAW_STANDARD_COV'):
            s=specs[arm]
            ck('oracle_visible_raw_standard_cov',all(m['xy']==rt['original_raw_points'][m['native_corner']] and m['covariance_px2']==[[float((2 if m['state']=='B' else 3)**2),0.],[0.,float((2 if m['state']=='B' else 3)**2)]] and D.valid_query(m['xy'],row['raw_hw']) for m in s['visible']),rid+':'+arm)
            radius=max(model['minimum_radius_px'],rt['ot_domains']['nonself_prediction_scale_px']*model['normalized_radius'])
            ck('oracle_O_radius_current_fixed_scale',all(d['half_axes_px']==[radius,radius] for d in s['domains'] if d['state']=='O' and d['shape']=='ellipse'),rid+':'+arm)
        drop=specs['CURRENT_DROP_KNOWN_FALSE_OT'];dropids=set(i for i,(a,b) in enumerate(zip(base['states'],drop['states'])) if a!=b)
        ck('knownfalse_drop_domains_only',drop['visible']==base['visible'] and all(base['states'][i] in ('O','T') and drop['states'][i]=='U' for i in dropids) and drop['domains']==[d for d in base['domains'] if d['native_corner'] not in dropids],rid)
        center=specs['CURRENT_REFERENCE_MODEL_CENTER_O']
        ck('reference_O_center_BI_T_fixed',center['visible']==base['visible'] and center['states']==base['states'] and center['config']==base['config'] and all(d==old for d,old in zip(center['domains'],base['domains']) if d['state']=='T') and all(d['half_axes_px']==old['half_axes_px'] and d['halfplanes']==old['halfplanes'] and d['center_xy']==align['reference_projection'][d['native_corner']] for d,old in zip(center['domains'],base['domains']) if d['state']=='O'),rid)
    init=specs['CURRENT_REFERENCE_INIT']
    ck('reference_init_exact_F_loss',same_except(init,base,'initializations') and init['initializations'][:1]==base['initializations'] and len(init['initializations'])==1+label['target']['truth']['order'],rid)
    Q=np.asarray(rt['initial_geometry']['physical_registration_Q']);G=np.asarray(label['target']['truth']['R'])
    ck('reference_init_physical_Q_all_symmetry',all(np.allclose(a['R'],G@D.rotation_y(k*2*np.pi/label['target']['truth']['order'])@Q.T,rtol=0,atol=1e-12) for k,a in enumerate(init['initializations'][1:])),rid)
    ideal=specs['MODEL_IDEAL_PROJECTIONS'];construction=ideal['initializations'][-1]
    q,depth=P.project_native(np.asarray(ideal['xyz']),np.asarray(construction['R']),np.asarray(construction['t']),np.asarray(ideal['K']))
    cfg=dict(P._DEFAULTS,**ideal['config']);xyz,K,st,ms,ds,hw=P._inputs(ideal['xyz'],ideal['K'],ideal['states'],ideal['visible'],ideal['domains'],ideal['image_hw'],cfg)
    sides=[]
    for d in ds:
        if d['state']!='T':sides.append(None)
        else:sides.append(max(P._SIDES,key=lambda side:min(P._domain_slacks(d,q[d['native_corner']],side,hw,cfg['outside_canvas_margin_px'])[0])))
    scale=max(float(np.linalg.norm(np.ptp(xyz,axis=0))),1e-3)
    vals=P._constraint_values(P._pose_vector(np.asarray(construction['R']),np.asarray(construction['t']),scale),xyz,K,ms,ds,tuple(sides),q,hw,scale,cfg)
    ck('ideal_construction_feasible_not_metric_claim',np.min(vals)>=-cfg['feasibility_tolerance'] and min(depth)>=cfg['positive_depth_m']-1e-10,rid)
    ck('ideal_measurements_declared_model_not_RGB',all(m['measurement_is_pose_projection'] and not m['image_measurement'] and m['covariance_px2']==[[1.,0.],[0.,1.]] and np.allclose(m['xy'],q[m['native_corner']],atol=1e-12,rtol=0) for m in ideal['visible']),rid)
    ideal_bi[len(ideal['visible'])]+=1
    # Merely inspect saved pools. No oracle candidate or metric is selected here.
    ck('current_sample_flags',all(x.get('all_OT_individual_constraints_pass') for x in rt['joint_pose'].get('feasible_pose_samples',[])),rid)
assert C.sha(source)==source_before,'Driver changed during this read-only audit'
result=dict(schema='INDEPENDENT_PURE_DIAGNOSTIC_INPUT_AUDIT_V1',driver_sha256=source_before,plan_sha256=C.sha(D.PLAN),new_solver_fits=0,new_RGB_reads=0,core_source_mutations=0,rows95=len(rows),compatible=dict(compatible),arms=arm_names,input_cases=len(keys),distinct_physics_keys=len(set(keys)),cached_physics_matches=len(set(keys)&current_keys),new_physics_jobs=len(set(keys)-current_keys),ideal_BI_count_histogram=dict(ideal_bi),checks=dict(checks),failed_checks=sum(len(v['failures']) for v in checks.values()),total_checks=sum(v['checked'] for v in checks.values()))
HERE.mkdir(parents=True,exist_ok=True)
C.write(HERE/'PURE_INPUT_AUDIT_V1.json',result)
print(json.dumps({k:v for k,v in result.items() if k!='checks'},ensure_ascii=False))
if result['failed_checks']:print(json.dumps({k:v for k,v in checks.items() if v['failures']},ensure_ascii=False));raise SystemExit(1)
