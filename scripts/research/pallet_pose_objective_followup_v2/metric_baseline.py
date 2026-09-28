"""Pose-first locked metric contract and CPU reuse of frozen historical outputs.

No fit, inference, selector tuning, Git, or old artifact write is performed.
The fixed-candidate T/R oracles remain private diagnostic selections.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
NAME = 'pallet_pose_objective_followup_v2'
DOC = ROOT / '_docs/experiments' / NAME
RAW = ROOT / 'data/pallet/results' / NAME / 'metric_baseline'
OLD_RAW = ROOT / 'data/pallet/results/pallet_oracle_mechanism_followup_v1'
OLD_DOC = ROOT / '_docs/experiments/pallet_oracle_mechanism_followup_v1'
PRIMARY = 'PRIMARY_MODERATE_PLUS_SEVERE'
SEVERITIES = ('CLEAN', 'MODERATE_OCCLUSION', 'SEVERE_OCCLUSION')
KEYS = ('translation_cm', 'rotation_deg', 'yaw_deg')
OLD_CYCLES = ('C2_REAL_AFFINE_OFF', 'C3_MANUAL38_CAPABILITY')
MAIN_ALIASES = {'PLASTIC': {'R0':'R0', 'OLD_RAW':'RAW_LR5', 'OLD_REF':'REF_LR5', 'SYN':'SYN_LR5'},
                'WOOD': {'R0':'R0', 'OLD_RAW':'WOOD_RAW_LR5', 'OLD_REF':'WOOD_REF_LR5', 'SYN':'SYN_LR5'}}


def read(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def bind(path):
    path = Path(path).resolve()
    return dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)


def verify(binding):
    assert bind(ROOT / binding['path']) == binding, 'Frozen input binding changed'


def clean(value):
    if isinstance(value, dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value, (tuple,list,np.ndarray)):return [clean(v) for v in value]
    if isinstance(value, np.generic):return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):return None
    return value


def save(path, value):
    path = Path(path).resolve()
    assert path.is_relative_to(DOC) or path.is_relative_to(RAW)
    text = value if isinstance(value,str) else json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if path.exists():
        assert path.read_text() == text, 'Refusing to overwrite a frozen metric artifact'
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp');temporary.write_text(text);temporary.replace(path)


def group_ids(records, material):
    out={'ALL':[r['id'] for r in records]}
    for severity in SEVERITIES:
        out['severity:'+severity]=[r['id'] for r in records if r['severity']==severity]
    for recording in sorted({r.get('recording',r.get('recording_group')) for r in records}):
        out['recording:'+recording]=[r['id'] for r in records if r.get('recording',r.get('recording_group'))==recording]
    if material=='PLASTIC':
        out[PRIMARY]=[r['id'] for r in records if r['severity'] in SEVERITIES[1:]]
    return out


def extended_quantile(values, q):
    """Linear rank quantile in extended reals, without inf-minus-inf NaNs."""
    values=sorted(values)
    if not values:return None
    rank=(len(values)-1)*q;low=int(math.floor(rank));high=int(math.ceil(rank))
    if low==high:return float(values[low])
    if math.isinf(values[high]):return float('inf')
    return float(values[low]+(rank-low)*(values[high]-values[low]))


def distribution(values):
    values=list(values)
    out={}
    for name,q in [('q10',.1),('q25',.25),('median',.5),('q75',.75),('P90',.9),('P95',.95),('P99',.99),('max',1.)]:
        value=extended_quantile(values,q)
        out[name]=value if value is None or np.isfinite(value) else None
        out[name+'_status']='NA_EMPTY' if value is None else 'POSITIVE_INFINITY' if math.isinf(value) else 'FINITE'
    return out


def summarize(rows):
    """Compatible with original D.metric records; never drops failed frames."""
    from scripts.research.pallet_oracle_mechanism_followup_v1 import pose_oracle as O
    rows=list(rows);valid=[r for r in rows if r['available']]
    out=dict(frames=len(rows),valid_pose=len(valid),failed_pose=len(rows)-len(valid),
             coverage=len(valid)/len(rows) if rows else None,
             status='OK' if rows else 'NA_EMPTY_POPULATION',
             conditional={k:distribution([r[k] for r in valid]) for k in KEYS},
             full_population={k:distribution([r[k] if r['available'] else float('inf') for r in rows]) for k in KEYS},
             axis_mismatch_count=sum(r.get('axis_correct') is False for r in valid),
             axis_available_count=sum('axis_correct' in r for r in valid),
             ADDsym_AUC=O.D.Pose.pose_auc([r['ADDsym_normalized'] if r['available'] else float('inf') for r in rows],1.) if rows else None,
             IoU3D=distribution([r['IoU3D'] for r in valid]))
    for axis in ('x','z'):
        values=[r['camera_'+axis+'_signed_cm'] for r in valid if 'camera_'+axis+'_signed_cm' in r]
        out['camera_'+axis]=dict(available=len(values),absolute_cm=distribution(map(abs,values)),
                                 signed_bias_mean_cm=float(np.mean(values)) if values else None,
                                 signed_bias_median_cm=float(np.median(values)) if values else None)
    return out


def paired(before, after, ids=None):
    ids=list(ids if ids is not None else before)
    assert set(ids)<=set(before) and set(ids)<=set(after)
    common=[i for i in ids if before[i]['available'] and after[i]['available']]
    counts=Counter();deltas={k:[] for k in KEYS}
    for fid in common:
        dt=after[fid]['translation_cm']-before[fid]['translation_cm']
        dr=after[fid]['rotation_deg']-before[fid]['rotation_deg']
        sign=lambda value:'IMPROVE' if value<0 else 'WORSEN' if value>0 else 'TIE'
        counts['T_'+sign(dt)+'__R_'+sign(dr)]+=1
        for key in KEYS:deltas[key].append(after[fid][key]-before[fid][key])
    old,new=[summarize([data[i] for i in ids]) for data in (before,after)]
    common_old,common_new=[summarize([data[i] for i in common]) for data in (before,after)]
    diff={key:(new['conditional'][key]['median']-old['conditional'][key]['median'])
          if new['conditional'][key]['median'] is not None and old['conditional'][key]['median'] is not None else None for key in KEYS}
    return dict(frames=len(ids),common_valid_frames=len(common),before_valid=old['valid_pose'],after_valid=new['valid_pose'],
                available_to_failed=sum(before[i]['available'] and not after[i]['available'] for i in ids),
                failed_to_available=sum(not before[i]['available'] and after[i]['available'] for i in ids),
                difference_of_conditional_medians=diff,
                median_of_common_frame_differences={k:float(np.median(v)) if v else None for k,v in deltas.items()},
                common_frame_delta_distributions={k:distribution(v) for k,v in deltas.items()},
                paired_direction_counts={f'T_{t}__R_{r}':counts[f'T_{t}__R_{r}'] for t in ('IMPROVE','TIE','WORSEN') for r in ('IMPROVE','TIE','WORSEN')},
                common_valid_before=common_old,common_valid_after=common_new,
                axis_correct_to_wrong=sum(before[i].get('axis_correct') is True and after[i].get('axis_correct') is False for i in common),
                axis_wrong_to_correct=sum(before[i].get('axis_correct') is False and after[i].get('axis_correct') is True for i in common))


def classify_candidate(new, baseline):
    """Sign-only label, not operational significance or independent evidence."""
    fields=('translation_cm','rotation_deg')
    a=[new['conditional'][k]['median'] for k in fields];b=[baseline['conditional'][k]['median'] for k in fields]
    if any(v is None for v in a+b):return dict(label='UNRESOLVED',eligible_joint=False)
    dt,dr=[x-y for x,y in zip(a,b)]
    label=('JOINT_GAIN_ON_REUSED_DEV' if dt<0 and dr<0 else 'TRADEOFF' if dt*dr<0 else
           'T_ONLY' if dt<0 else 'R_ONLY' if dr<0 else 'NO_GAIN')
    coverage_ok=new['valid_pose']>=baseline['valid_pose'] and new['frames']==baseline['frames']
    return dict(label=label,eligible_joint=label=='JOINT_GAIN_ON_REUSED_DEV' and coverage_ok,
                coverage_guard_pass=coverage_ok,translation_delta_cm=dt,rotation_delta_deg=dr,
                near_numeric_parity_limit=abs(dt)<=1e-7 or abs(dr)<=1e-7,
                significance='No meaningful/practical/independent gain automatically inferred from sign')


def pareto_front(candidates):
    """Input card ID, T/R median, added inference cost, training resources."""
    result=[]
    for candidate in candidates:
        point=(candidate['translation_cm'],candidate['rotation_deg'])
        dominated=any(other is not candidate and other['translation_cm']<=point[0] and other['rotation_deg']<=point[1]
                      and (other['translation_cm']<point[0] or other['rotation_deg']<point[1]) for other in candidates)
        if not dominated:result.append(candidate)
    return sorted(result,key=lambda c:c['card_id'])


def choose_repeat(candidates):
    eligible=[c for c in pareto_front(candidates) if c['eligible_joint']]
    if not eligible:return None
    return min(eligible,key=lambda c:(c['added_inference_seconds'],c['new_training_GPU_seconds'],c['card_id']))['card_id']


def extend_metric(metric, pose, truth):
    """Independently recompute origin/units/C2 rotation, preserving original rows."""
    row=dict(metric)
    assert bool(row['available'])==bool(pose['available'])
    if not row['available']:return row
    from scripts.paper.pose_metric_closure_v1.symmetry_aware_pose_metrics import rotation_error_degrees
    assert truth['order']==2
    delta=(np.asarray(pose['centroid'])-np.asarray(truth['t']))*100
    rotation=rotation_error_degrees(pose['R_physical'],truth['R'])
    assert np.isclose(np.linalg.norm(delta),row['translation_cm'],atol=1e-7,rtol=1e-7)
    assert np.isclose(rotation,row['rotation_deg'],atol=1e-7,rtol=1e-7)
    row.update(camera_x_signed_cm=float(delta[0]),camera_z_signed_cm=float(delta[2]))
    return row


def freeze():
    path=DOC/'METRIC_AND_SELECTION_LOCK.json'
    if path.exists():
        for binding in read(path)['inputs']:verify(binding)
        print('METRIC_LOCK_ALREADY_FROZEN',flush=True);return
    from scripts.research.pallet_oracle_mechanism_followup_v1 import pose_oracle as O
    members={};inputs=[]
    for material in MAIN_ALIASES:
        records,_,_,paths=O.population(material)
        expected={'CLEAN':29,'MODERATE_OCCLUSION':21,'SEVERE_OCCLUSION':78} if material=='PLASTIC' else {'CLEAN':38,'MODERATE_OCCLUSION':7}
        assert dict(Counter(r['severity'] for r in records))==expected
        groups=group_ids(records,material)
        if material=='PLASTIC':assert len(groups[PRIMARY])==99
        members[material]=dict(records=[{k:r[k] for k in ('id','recording','severity')} for r in records],groups=groups)
        inputs.extend(paths)
        oldroot=O.P.RAW if material=='PLASTIC' else O.W.RAW
        inputs.extend(oldroot/name for name in ('POSE_METRICS.json','FRAME_METRICS.json'))
        inputs.extend(OLD_RAW/'pose_oracle'/f'{material}_{kind}.json' for kind in ('CANDIDATES','METRICS'))
        for cycle in OLD_CYCLES:
            inputs.extend(OLD_RAW/'cycles'/cycle/f'{material}_{kind}.json' for kind in ('POSES','POSE_METRICS','FRAME_METRICS','METADATA'))
            inputs.extend([OLD_RAW/'cycles'/cycle/'PREDICTIONS_LOCK.json',OLD_DOC/'cycles'/cycle/'RESULTS.json'])
    member_path=RAW/'POPULATION_LOCK_PRIVATE.json';save(member_path,members);inputs.append(member_path)
    inputs += [ROOT/'scripts/research/pallet_clean19_pose_mismatch_v1/diagnose.py',
               ROOT/'scripts/research/pallet_dim_conditioned_p_v1/pose.py',
               ROOT/'scripts/paper/pose_metric_closure_v1/symmetry_aware_pose_metrics.py',
               ROOT/'scripts/paper/pose_metric_closure_v1/run_pose_evaluation.py',
               ROOT/'scripts/paper/pose_metric_closure_v1/build_geometry_resolved_pose_gt.py',
               ROOT/'challenge/evaluation_v2/pnp_selector.py',
               ROOT/'data/pallet/results/paper_pose_metric_closure_v1/GEOMETRY_RESOLVED_POSE_GT.json',
               ROOT/'data/pallet/results/paper_pose_metric_closure_v1/AXIS_REVIEW_MANIFEST.json']
    save(path,dict(created_at=now(),status='FROZEN_BEFORE_V2_FITS_AND_FRESH_BASELINE_AGGREGATION',
       primary_population=dict(material='PLASTIC',natural_severities=list(SEVERITIES[1:]),frames=99,
           grouping='Pool all original99 frame errors; never average severity medians',reference_role='REUSED_DEV'),
       populations={'PLASTIC':dict(frames=128,Clean=29,Moderate=21,Severe=78),'WOOD':dict(frames=45,Clean=38,Moderate=7,Severe=0)},
       translation=dict(name='translation_cm',origin='Centroid of origin-centered physical cuboid',frame='OpenCV camera coordinates',
           formula='100 * norm(t_pred_centroid_m - t_reference_centroid_m)',unit='cm',camera_components='x,z signed prediction-minus-reference and absolute error; no vehicle extrinsic provided, not forklift lateral/forward'),
       rotation=dict(name='rotation_deg',formula='min_Q degrees(acos(clamp((trace((R_gt Q)^T R_pred)-1)/2,-1,1)))',
           symmetry='C2: I and proper Ry(180deg); no 90deg W/D symmetry',frame='Object-to-camera physical registry frame',
           correspondence='R_physical=R_camera_facing @ Q_WD; same deterministic registry-X conversion for reference, no per-prediction truth branch',
           signed_axis_limit='180deg equivalence follows legacy geometry contract; not independently verified physical insertion-face orientation'),
       yaw=dict(name='yaw_deg',definition='Minimum existing atan2(relative[0,2],relative[2,2]) absolute wrap over same C2, degrees; supplementary, not fullR'),
       summaries=dict(primary='Pair of conditional T/R medians with nondecreasing coverage guard',
           secondary='P90, yaw, camera-x/z errors and biases, registered W/D-axis mismatch count, full error quantiles, AUC/PCK/IoU',
           legacy='Original valid-pose conditional numpy median/linear quantile P90 reproduced',
           full_population='Unavailable ranked as positive infinity; linear extended-real quantiles. JSON null plus POSITIVE_INFINITY marker, never failure=0 or an invented finite penalty',
           common_valid='Same valid frame intersection reported for every paired comparison',
           large_angle='No new operational angular threshold: report fullR quantiles and original axis_correct extent-parity mismatch separately',
           difference='NEW-minus-BASE; difference of medians and median of per-frame differences distinct'),
       selection=dict(A='Both T/R medians decrease versus OLD_REF on primary99 => JOINT_GAIN_ON_REUSED_DEV, subject to no coverage loss',
           B='One decreases: preserve T_ONLY/R_ONLY or opposing-sign TRADEOFF; never claim both improve',
           C='Compare both T/R separately versus R0 and matched new RAW',
           D='Keep T/R Pareto-nondominated candidates; do not add cm and degrees',
           E='Among joint Pareto candidates repeat lowest added inference runtime; then lowest new measured GPU training runtime; then predeclared card ID lexicographic. Fits/updates also disclosed',
           F='No joint candidate: a justified single-axis replication may be separately declared before repetition; no automatic PCK/AUC fallback',
           G='Small signs are not meaningful-gain thresholds: retain numeric parity tolerance1e-7, actual magnitude, repeat variability and reference uncertainty',
           H='No AUC/PCK posthoc tie-break',checkpoint='last at locked update budget, all seeds retained',
           historical='C2/C3 T/R evaluation is retrospective descriptive reuse, not prospectively selected success or matched main217/361 intervention'),
       oracle=dict(objectives=['translation_cm','rotation_deg'],pool='Same original D9 candidate set or four original whole-output R0/RAW/REF/TEACHER poses',
           choice='Minimize named objective then lexical candidate name; optimize separately; report complete T/R vector of each chosen pose',
           joint='Count same single candidate strictly improving both current T and R; no combining minima from different candidates',
           isolation='GT_DEPENDENT / DIAGNOSTIC_ONLY, selections private, never training/inference input'),
       numeric_tolerance=dict(atol=1e-7,rtol=1e-7,purpose='Historical numeric parity only, not operational acceptance threshold'),
       sources_read='Actual Pose.metric/infer/metadata, D.metric/aggregate, original cuboid/solver/GT constructor and C2 metric helpers',
       inputs=[bind(p) for p in sorted(set(inputs))]))
    print('METRIC_AND_SELECTION_LOCK_FROZEN',flush=True)


def baseline():
    start=time.perf_counter();cpu=time.process_time();lock=read(DOC/'METRIC_AND_SELECTION_LOCK.json')
    for binding in lock['inputs']:verify(binding)
    destination=DOC/'BASELINE_POSE_RESULTS.json'
    if destination.exists():print('BASELINE_ALREADY_COMPLETE',flush=True);return
    from scripts.research.pallet_oracle_mechanism_followup_v1 import pose_oracle as O
    from scripts.research.pallet_oracle_mechanism_followup_v1.cycle_affine_eval import summary2
    members=read(RAW/'POPULATION_LOCK_PRIVATE.json');_,truth=O.D.Pose.metadata('REAL_DEV')
    output={};private={};parity=0
    for material in MAIN_ALIASES:
        oldroot=O.P.RAW if material=='PLASTIC' else O.W.RAW
        poses=read(oldroot/'POSE_PREDICTIONS.json');metrics=read(oldroot/'POSE_METRICS.json');twoD=read(oldroot/'FRAME_METRICS.json')
        models={alias:(metrics[original],poses[original],twoD[original]) for alias,original in MAIN_ALIASES[material].items()}
        for cycle,aliases in [(OLD_CYCLES[0],{'C2_RAW':'NEW_RAW','C2_REF':'NEW_REF'}),(OLD_CYCLES[1],{'C3_RAW9':'RAW9','C3_MANUAL9':'MANUAL9'})]:
            root=OLD_RAW/'cycles'/cycle
            pm,pp,ff=[read(root/f'{material}_{kind}.json') for kind in ('POSE_METRICS','POSES','FRAME_METRICS')]
            models.update({alias:(pm[original],pp[original],ff[original]) for alias,original in aliases.items()})
        records=members[material]['records'];ids=members[material]['groups']['ALL'];private[material]={}
        for arm,(mm,pp,_) in models.items():
            assert set(mm)==set(pp)==set(ids)
            private[material][arm]={fid:extend_metric(mm[fid],pp[fid],truth[fid]) for fid in ids};parity+=len(ids)
        groups={}
        for name,group in members[material]['groups'].items():
            groups[name]={arm:dict(**summarize([private[material][arm][fid] for fid in group]),
                 auxiliary_2D=summary2([models[arm][2][fid] for fid in group]) if group else None) for arm in models}
        contrasts={};pairs=[('OLD_RAW','OLD_REF'),('R0','OLD_REF'),('OLD_REF','C2_REF'),('C2_RAW','C2_REF'),('R0','C2_REF'),
                           ('OLD_REF','C3_MANUAL9'),('C3_RAW9','C3_MANUAL9'),('R0','C3_MANUAL9')]
        for name,group in members[material]['groups'].items():
            contrasts[name]={after+'-minus-'+before:paired(private[material][before],private[material][after],group) for before,after in pairs}
        loro={}
        primary=members[material]['groups'].get(PRIMARY,ids)
        for recording in sorted({r['recording'] for r in records}):
            subset=[r['id'] for r in records if r['recording']!=recording and r['id'] in primary]
            loro[recording]={after+'-minus-'+before:paired(private[material][before],private[material][after],subset) for before,after in pairs}
        output[material]=dict(groups=groups,contrasts=contrasts,leave_one_recording_out=loro,
          LORO_population=PRIMARY if material=='PLASTIC' else 'ALL',reference='Legacy geometry-derived, not independent physical6D',
          historical_C2_C3='Posthoc T/R descriptive reference only; not V2 candidates, C3 training population differs')
        if material=='PLASTIC':
            print('PRIMARY99_BASELINES',json.dumps({a:{k:groups[PRIMARY][a]['conditional'][k]['median'] for k in KEYS} for a in ('R0','OLD_RAW','OLD_REF')}),flush=True)
    path=RAW/'FRAME_METRICS_PRIVATE.json';save(path,private)
    save(destination,dict(created_at=now(),materials=output,metric_lock=bind(DOC/'METRIC_AND_SELECTION_LOCK.json'),
       private_metric_artifact=bind(path),numeric_parity_frame_arm_comparisons=parity,
       primary_population='Plastic natural Moderate21 + Severe78 =99 original frame errors',
       new_fits=0,optimizer_updates=0,GPU_seconds=0,wall_seconds=time.perf_counter()-start,CPU_seconds=time.process_time()-cpu,
       implementation=bind(Path(__file__))))
    print('BASELINE_COMPLETE',parity,flush=True)


def oracle_for(ids, choices, current):
    """choices[fid]: named complete metric records; never merge their fields."""
    selected={objective:{} for objective in KEYS[:2]};rows={objective:[] for objective in KEYS[:2]}
    joint=0;different=0;unavailable=0
    for fid in ids:
        available=[h for h in choices[fid] if h['metric']['available']]
        if not available:unavailable+=1
        for objective in selected:
            best=min(available,key=lambda h:(h['metric'][objective],h['name'])) if available else None
            selected[objective][fid]=best['name'] if best else None
            rows[objective].append(best['metric'] if best else dict(id=fid,available=False))
        different+=selected['translation_cm'][fid]!=selected['rotation_deg'][fid]
        base=current[fid]
        if base['available']:
            joint+=any(h['metric']['translation_cm']<base['translation_cm'] and h['metric']['rotation_deg']<base['rotation_deg'] for h in available)
            for objective in selected:
                assert rows[objective][-1][objective]<=base[objective]+1e-7
    baseline_summary=summarize([current[fid] for fid in ids])
    result=dict(current=baseline_summary,translation_optimal=summarize(rows['translation_cm']),rotation_optimal=summarize(rows['rotation_deg']),
                frames_with_same_candidate_joint_gain=joint,T_R_optimal_choices_different=different,no_available_candidate=unavailable,
                separate_objectives_not_one_attainable_pose=True)
    return result,selected


def oracles():
    start=time.perf_counter();cpu=time.process_time();destination=DOC/'TR_ORACLE_RESULTS.json'
    if destination.exists():print('TR_ORACLES_ALREADY_COMPLETE',flush=True);return
    lock=read(DOC/'METRIC_AND_SELECTION_LOCK.json')
    for binding in lock['inputs']:verify(binding)
    from scripts.research.pallet_oracle_mechanism_followup_v1 import pose_oracle as O
    members=read(RAW/'POPULATION_LOCK_PRIVATE.json');out={};private={}
    for material in MAIN_ALIASES:
        metrics=read(OLD_RAW/'pose_oracle'/f'{material}_METRICS.json')['arms']
        out[material]={};private[material]={}
        for group,ids in members[material]['groups'].items():
            fixed={};selection={}
            for arm in O.ARMS[material]:
                current={fid:metrics[arm][fid]['current'] for fid in ids}
                choices={fid:metrics[arm][fid]['hypotheses'] for fid in ids}
                fixed[arm],selection[arm]=oracle_for(ids,choices,current)
            pool={fid:[dict(name=arm,metric=metrics[arm][fid]['current']) for arm in O.ARMS[material]] for fid in ids}
            whole={};pool_selection={}
            for arm in O.ARMS[material]:
                whole[arm],pool_selection[arm]=oracle_for(ids,pool,{fid:metrics[arm][fid]['current'] for fid in ids})
            out[material][group]=dict(fixed_D9_candidates=fixed,whole_output_expert_pool=whole)
            private[material][group]=dict(fixed=fixed and selection,whole_output=pool_selection)
    path=RAW/'TR_ORACLE_CHOICES_PRIVATE.json';save(path,dict(is_oracle=True,GT_DEPENDENT=True,DIAGNOSTIC_ONLY=True,production_input=False,choices=private))
    save(destination,dict(created_at=now(),is_oracle=True,GT_DEPENDENT=True,DIAGNOSTIC_ONLY=True,production_input=False,
       materials=out,metric_lock=bind(DOC/'METRIC_AND_SELECTION_LOCK.json'),private_choices=bind(path),
       caveats=['T-best and R-best complete poses reported separately; extrema are not combined',
                'Counts of one candidate improving both metrics do not demonstrate an input-only selector',
                'No new candidate generation, no per-point mixing or solver intervention; reused metric caches only',
                'Prior AUC-optimal oracle does not upper-bound T/R; new objectives do not rewrite its historical conclusion'],
       new_fits=0,optimizer_updates=0,GPU_seconds=0,wall_seconds=time.perf_counter()-start,CPU_seconds=time.process_time()-cpu,
       implementation=bind(Path(__file__))))
    print('TR_ORACLES_COMPLETE',flush=True)


def report():
    result=read(DOC/'BASELINE_POSE_RESULTS.json');oracle=read(DOC/'TR_ORACLE_RESULTS.json')
    lines=['# Translation / full rotation 평가 계약','',
      '[확인] 새 지시문 §3을 새 학습 전에 잠갔다. 기존 원고·AUC 판정·참조·후보 생성은 수정하지 않았다. 새 T/R 주집합은 반복 DEV인 Plastic 자연 Moderate21+Severe78의 실제99 frame 합집합이다. 아래 과거 C2/C3 표는 새 기준의 사후 기술 통계이며, 새 성공 후보로 선정하지 않았다.','',
      '## 지표와 참조','',
      'T는 원점 중심 cuboid의 centroid 두 위치를 같은 OpenCV camera 좌표계에서 비교한 L2 norm×100(cm)이다. full R은 object-to-camera physical registry rotation의 geodesic angle(deg)을 기존 C2={I,Ry180}에서 최소화한다. camera-facing W/D→registry 변환은 예측/참조에 동일하게 고정되며 전치나 inverse로 바꾸지 않는다. 직사각형 90° 혼동은 대칭으로 제거하지 않는다. yaw는 같은 C2에서 상대회전 atan2의 wrap 최솟값이며 full R을 대체하지 않는다.','',
      '참조는 같은 annotation/치수로 복원된 geometry-derived pose다. 180° signed-axis 등가류이지 독립 physical6D 또는 삽입면 방향 검증이 아니다. camera-x/z는 예측−참조 signed bias와 절대오차로 보고하며, 차량 extrinsic이 확인되지 않아 forklift lateral/forward라고 부르지 않는다.','',
      '기존 conditional median/P90은 valid pose만 집계하고 coverage를 병기한다. 실패 포함 표는 실패를 +∞로 순서화한 extended-real linear quantile이다. JSON null은 POSITIVE_INFINITY/NA_EMPTY 상태와 구분한다. 임의 cm/deg failure penalty와 0오차 대체는 없다. common-valid 교집합 표·실패 전이·T/R 9칸 개선/동률/악화 교차표를 함께 제공한다. 기존 W/D extent axis-mismatch를 세고 신규 각도 합격선을 발명하지 않았다.','',
      '## 선택 규칙','',
      'primary99에서 OLD_REF 대비 T/R median이 둘 다 감소하고 coverage가 감소하지 않은 후보만 joint로 표시한다. R0 및 matched RAW 비교는 별도다. T/R Pareto 비지배 후보 중 추가 추론시간→신규 학습 GPU시간→사전 card ID 순으로 반복 후보를 택한다. cm+degree 합산이나 AUC/PCK tie-break는 없다. 하나만 좋아지거나 tradeoff인 경우 축을 명시하며, 작은 부호를 실질적 향상으로 자동 인정하지 않는다.','',
      '## Plastic primary99 baseline / 과거 참고선','',
      '| arm | valid/N | T median / P90 (cm) | R median / P90 (deg) | yaw median / P90 (deg) | axis mismatch |','|---|---:|---:|---:|---:|---:|']
    primary=result['materials']['PLASTIC']['groups'][PRIMARY]
    for arm,row in primary.items():
        values=[' / '.join(f"{row['conditional'][key][q]:.6f}" for q in ('median','P90')) for key in KEYS]
        lines.append(f"| {arm} | {row['valid_pose']}/{row['frames']} | {' | '.join(values)} | {row['axis_mismatch_count']} |")
    lines+=['','## 서로 다른 T/R oracle','',
      '| material/group | old REF T / R | T-best pose T / R | R-best pose T / R | same-candidate joint gain frames | distinct T/R choices |','|---|---:|---:|---:|---:|---:|']
    for material,group,arm in [('PLASTIC',PRIMARY,'REF_LR5'),('PLASTIC','ALL','REF_LR5'),('WOOD','ALL','WOOD_REF_LR5')]:
        value=oracle['materials'][material][group]['fixed_D9_candidates'][arm]
        fields=[' / '.join(f"{value[name]['conditional'][key]['median']:.6f}" for key in KEYS[:2]) for name in ('current','translation_optimal','rotation_optimal')]
        lines.append(f"| {material}/{group} | {' | '.join(fields)} | {value['frames_with_same_candidate_joint_gain']} | {value['T_R_optimal_choices_different']} |")
    lines+=['','T-best의 R과 R-best의 T도 각각 동일하게 선택된 한 pose에서 산출했다. 서로 다른 최솟값을 하나의 가능한 pose로 합치지 않는다. whole-output R0/teacher/RAW/REF pool의 별도 oracle도 JSON에 보존했다. GT로 고른 선택은 private diagnostic이며 학생 타깃·일반 추론에 제공하지 않는다.','',
      '모든 severity/recording/전체128·45·paired/LORO 수치와 camera-x/z는 [BASELINE_POSE_RESULTS.json](BASELINE_POSE_RESULTS.json), oracle은 [TR_ORACLE_RESULTS.json](TR_ORACLE_RESULTS.json)에 있다. Wood Severe는 N=0, NA_EMPTY_POPULATION이며 성능0으로 해석하지 않는다.','',
      '재현: `python -m scripts.research.pallet_pose_objective_followup_v2.metric_baseline all`. 잠긴 산출물이 있으면 hash를 검증하고 재학습 없이 재사용한다.']
    save(DOC/'METRIC_CONTRACT.md','\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','baseline','oracle','report','all']);args=parser.parse_args()
    if args.stage in ('freeze','all'):freeze()
    if args.stage in ('baseline','all'):baseline()
    if args.stage in ('oracle','all'):oracles()
    if args.stage in ('report','all'):report()


if __name__=='__main__':main()
