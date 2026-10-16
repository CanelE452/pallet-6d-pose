"""One saved-row decomposition; no model, solver, scorer or reference loader.

Associations and exact mean contributions do not identify a physical cause.
Freeze creates new audit bindings; run performs one bounded 245-frame join.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

REPO = Path(__file__).resolve().parents[3]
DOC = REPO / '_docs/experiments/pallet_cornerwise_independent_20261010_v4'
PRIMARY = 'N3_INDEPENDENT_CORNERWISE_ROLE'
NATIVE_H = 'N3_INDEPENDENT_ROBUST_H'
NO_MASK = 'N3_INDEPENDENT_ROBUST_NO_MASK'
OLD = 'N3_CORNERWISE_ROLE'
METHODS = (PRIMARY, NATIVE_H, NO_MASK, OLD)
METRICS = ('translation_cm', 'rotation_deg', 'ADDsym_cm')
QUALITY = ('CORRECT_WITHIN8PX', 'INCORRECT_OVER8PX', 'UNKNOWN_REFERENCE_OR_INPUT')
ARTIFACTS = ('FAULT_ISOLATION_STARTED.json', 'FAULT_ISOLATION_ROWS.jsonl.gz',
             'FAULT_ISOLATION_CHECKS.json', 'FAULT_ISOLATION_TABLE.csv', 'FAULT_ISOLATION_KO.md')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with gzip.open(path, 'rt') as stream:
        for line in stream:
            yield json.loads(line)


def binding(path):
    path = Path(path).resolve()
    require(path.is_relative_to(REPO), 'public audit input must be inside publication repository')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return dict(path=str(path.relative_to(REPO)), sha256=h.hexdigest(), bytes=path.stat().st_size)


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def inputs(folder):
    paths = dict(predictions=folder/'PREDICTIONS.jsonl.gz', fixed=folder/'FIXED_PREDICTIONS.jsonl.gz',
                 posthoc=folder/'POSTHOC_ROWS.jsonl.gz', metrics=folder/'METRICS.json',
                 formal_protocol=folder/'PROTOCOL.json', scoring=folder/'SCORING_RECEIPT.json',
                 control_parity=folder/'V3_CONTROL_PARITY.json', inference=folder/'INFERENCE_RECEIPT.json',
                 cohort=REPO/'_docs/experiments/pallet_kp_corrected_supervision_20261010_v1/COHORT.json',
                 code=Path(__file__))
    return {key: binding(path) for key, path in paths.items()}


def distribution(values):
    a = [float(v) for v in values if v is not None]
    require(all(math.isfinite(v) for v in a), 'nonfinite recorded value')
    n = len(a); mean = math.fsum(a)/n if n else None
    variance = math.fsum((v-mean)**2 for v in a)/(n-1) if n > 1 else None
    return dict(n=n, mean=mean, sample_variance=variance,
                sample_std=math.sqrt(variance) if variance is not None else None,
                median=float(np.quantile(a, .5)) if n else None,
                P90=float(np.quantile(a, .9)) if n else None, max=max(a) if n else None)


def pose_values(row):
    if not row['pose']['available']:
        return None
    return {k: float(row['pose']['ADDsym_m'])*100 if k == 'ADDsym_cm' else float(row['pose'][k])
            for k in METRICS}


def delta(a, b):
    return {k: a[k]-b[k] for k in METRICS} if a is not None and b is not None else None


def branch(actual, xyz):
    if not actual.get('available') or actual.get('cf_extents') is None:
        return 'UNAVAILABLE'
    d, x = np.asarray(actual['cf_extents'], float), np.asarray(xyz, float)
    require(d.shape == x.shape == (3,) and np.isfinite(d).all(), 'invalid recorded extents')
    if abs(x[0]-x[2]) < 1e-9:
        require(np.allclose(d, x, atol=1e-9, rtol=0), 'square extents differ from registry')
        return 'SQUARE_IDENTICAL'
    if np.allclose(d, x, atol=1e-9, rtol=0):
        return 'REGISTRY_WD'
    if np.allclose(d, x[[2, 1, 0]], atol=1e-9, rtol=0):
        return 'REGISTRY_DW'
    raise ValueError('actual output extents match neither supplied registry hypothesis')


def relation(a, b):
    if 'UNAVAILABLE' in (a, b):
        return 'UNAVAILABLE'
    if 'SQUARE_IDENTICAL' in (a, b):
        require(a == b, 'square branch mismatch')
        return 'SQUARE_IDENTICAL'
    return 'SAME' if a == b else 'WD_SWITCHED'


def compact(row):
    solver = row.get('solver') or {}
    return dict(id=row['id'], session=row['session'], method=row['method'], xyz=row['xyz'],
        new_pose=row['new_pose_estimated'], fallback=row['fallback_used'], status=row['output_status'],
        values=pose_values(row), actual_pose={k: row['actual_pose'].get(k)
            for k in ('available', 'cf_extents', 'R_physical')},
        selected_boundary_ids=list(row.get('selected_corner_ids', [])),
        used_ids=list(solver.get('used', [])), fit_ids=list(solver.get('fit_input_ids', []))
            if row['new_pose_estimated'] else [],
        final_inliers=list(solver.get('final_inliers', [])) if row['new_pose_estimated'] else [],
        geometry=solver.get('geometry', {}), solver_state=solver.get('state'),
        solver_reason=solver.get('reason'), prior_used=solver.get('prior_used'))


def aggregate(data):
    result = dict(n=len(data), ids=[r['id'] for r in data],
        state_counts=dict(Counter(r['state_transition'] for r in data)),
        branch_counts=dict(Counter(r['branch_vs_N3'] for r in data)),
        nativeH_branch_counts=dict(Counter(r['nativeH_branch_vs_N3'] for r in data)),
        nativeH_branch_switch_ids=[r['id'] for r in data if r['nativeH_branch_vs_N3']=='WD_SWITCHED'],
        primary_nativeH_branch_relation=dict(Counter(r['primary_nativeH_branch_relation'] for r in data)),
        accepted_final_inlier_count=dict(Counter(str(r['primary']['final_inlier_count']) for r in data
                                                if r['primary']['new_pose'])),
        used_count=dict(Counter(str(r['primary']['used_count']) for r in data)),
        object_rank=dict(Counter(str(r['primary']['object_rank']) for r in data)),
        jacobian_rank=dict(Counter(str(r['primary']['jacobian_rank']) for r in data)),
        jacobian_condition=distribution([r['primary']['jacobian_condition_number'] for r in data]),
        retained_input_quality={q:sum(r['primary']['used_quality_counts'].get(q,0) for r in data) for q in QUALITY},
        accepted_final_inlier_quality={q:sum(r['primary']['final_inlier_quality_counts'].get(q,0) for r in data) for q in QUALITY},
        large_rotation_error_45deg_ids=[r['id'] for r in data if r['primary']['values'] is not None
                                       and r['primary']['values']['rotation_deg']>=45],
        primary_output_rotation_error=distribution([r['primary']['values']['rotation_deg']
            for r in data if r['primary']['values'] is not None]),
        physical_rotation_delta_vs_N3=distribution([r['primary']['physical_rotation_delta_vs_N3_deg'] for r in data]))
    result['paired'] = {}
    for comparison in ('N3', 'nativeH', 'oldv3'):
        result['paired'][comparison] = {}
        for key in METRICS:
            values=[r['delta_'+comparison][key] for r in data if r['delta_'+comparison] is not None]
            result['paired'][comparison][key]=dict(**distribution(values),
                sum_delta=math.fsum(values), contribution_to_full245_mean=math.fsum(values)/245)
    matched=[r for r in data if r['primary_nativeH_branch_relation'] in ('SAME','SQUARE_IDENTICAL')
             and r['primary']['new_pose'] and r['nativeH']['new_pose']]
    result['boundary_comparison_same_branch_both_NEW'] = dict(n=len(matched), ids=[r['id'] for r in matched],
        metrics={k:distribution([r['delta_nativeH'][k] for r in matched]) for k in METRICS})
    candidates=[c for r in data for c in r['boundary_candidates']]
    result['boundary']={}
    for name, selected in [('all_candidates',candidates),('adopted',[c for c in candidates if c['accepted']]),
                           ('displayed',[c for c in candidates if c['displayed_in_final_output']])]:
        known=[c for c in selected if c['delta_vs_N3_px'] is not None]
        result['boundary'][name]=dict(n=len(selected), known=len(known), unknown=len(selected)-len(known),
            improved=sum(c['delta_vs_N3_px']<0 for c in known), worsened=sum(c['delta_vs_N3_px']>0 for c in known),
            unchanged=sum(c['delta_vs_N3_px']==0 for c in known),
            delta_error_px=distribution([c['delta_vs_N3_px'] for c in known]))
    return result


def freeze(folder):
    require(folder.resolve()==DOC.resolve(), 'use exact new v4 audit namespace')
    for name in ('FAULT_ISOLATION_PROTOCOL.json',)+ARTIFACTS:
        require(not (folder/name).exists(), 'preserve '+name)
    write(folder/'FAULT_ISOLATION_PROTOCOL.json', dict(
        schema='fixed_saved_row_fault_isolation_protocol_v4', inputs=inputs(folder),
        frames=245, methods=list(METHODS), actual_arithmetic_runs_allowed=1,
        group_factors=['oldv3_to_v4_output_state','actual_output_extents_vs_N3','adopted_boundary_0_or_positive'],
        states=['BOTH_NEW','ADDITIONAL_NEW','LOST_NEW','BOTH_BASELINE_RETURN'],
        branch_rule='actual returned cf_extents matched to supplied physical xyz or xyz[[2,1,0]] at1e-9; square merged; fallback uses returnedN3, not rejectedcandidate',
        exact_sum_decomposition='sum paired delta pergroup /245; groups partition all245; compare recordedfullMETRICS',
        descriptive_rotation_threshold_deg=45, no_threshold_selection=True,
        physical_rotation_difference='arccos(clamp((trace(R_N3.T@R_output)-1)/2)) from recordedphysicalR only; no GT',
        boundary_effect='sameframe primary minusnativeH; samebranch bothNEW subset separately reported, not randomcausalintervention',
        quality='reuse fixedN3phase POSTHOC known8px/unknown, acceptedNEWinliers only',
        no_new_model_PnP_ray_training_RGB_GT_scoring_calls=True,
        limitations=['WDswitch association is not proof of a wrongphysicalbranch',
                     'PhysicalRchange and proxyrotationerror are distinct quantities',
                     '238minus217 isnetNEWchange; additional/lost requireexactIDjoin',
                     'No physicalownershipcausefraction or posthocsettingselection']))


def run(folder):
    for name in ARTIFACTS:
        require(not (folder/name).exists(), 'preserve '+name)
    protocol=read(folder/'FAULT_ISOLATION_PROTOCOL.json'); before=inputs(folder)
    require(before==protocol['inputs'], 'audit code/input drift before arithmetic')
    write(folder/'FAULT_ISOLATION_STARTED.json', dict(protocol=binding(folder/'FAULT_ISOLATION_PROTOCOL.json'),
        inputs=before, actual_arithmetic_runs=1, new_model_PnP_ray_training_RGB_GT_scoring_calls=0))
    start=time.monotonic(); counts=Counter(); failure=None; output=[]; checks=[]
    try:
        saved=read(folder/'METRICS.json'); cohort=read(REPO/protocol['inputs']['cohort']['path']); ids=cohort['ids']
        require(len(ids)==len(set(ids))==245,'cohort population differs')
        by={m:{} for m in METHODS+('BASE','N3_SUBPIX')}
        for filename in ('PREDICTIONS.jsonl.gz','FIXED_PREDICTIONS.jsonl.gz'):
            for row in rows(folder/filename):
                counts['scored_rows_read']+=1
                require(row['method'] in by and row['id'] not in by[row['method']], 'duplicate/mismatched row')
                by[row['method']][row['id']]=compact(row)
        require(all(set(x)==set(ids) for x in by.values()), 'scored method population differs')
        post={}
        for row in rows(folder/'POSTHOC_ROWS.jsonl.gz'):
            counts['posthoc_rows_read']+=1
            key=(row['method'],row['id']);require(key not in post,'duplicate posthoc row');post[key]=row
        require(len(post)==980,'posthoc population differs')
        for fid in ids:
            primary,old,native,n3=(by[m][fid] for m in (PRIMARY,OLD,NATIVE_H,'N3_SUBPIX'))
            p=post[(PRIMARY,fid)]
            status='BOTH_NEW' if primary['new_pose'] and old['new_pose'] else (
                'ADDITIONAL_NEW' if primary['new_pose'] else 'LOST_NEW' if old['new_pose'] else 'BOTH_BASELINE_RETURN')
            require(primary['values'] is not None and n3['values'] is not None,'operational output missing')
            b={m:branch(by[m][fid]['actual_pose'],primary['xyz']) for m in METHODS+('N3_SUBPIX',)}
            rp=np.asarray(primary['actual_pose']['R_physical'],float); rn=np.asarray(n3['actual_pose']['R_physical'],float)
            angle=float(np.degrees(np.arccos(np.clip((np.trace(rn.T@rp)-1)/2,-1,1))))
            counts['physical_rotation_matrix_comparisons']+=1
            g=primary['geometry'];corners={c['id']:c for c in p['corners']}
            used=primary['used_ids']; final=primary['final_inliers']
            require(set(final)==set(p['pools']['accepted_final_inliers']['ids']), 'accepted inlier IDs differ')
            require(set(used)==set(p['pools']['solver_used']['ids']), 'used IDs differ')
            usedq=dict(Counter(corners[k]['input_quality'] for k in used))
            finalq=dict(Counter(corners[k]['input_quality'] for k in final))
            boundary=p['boundary_candidates']; adopted=[c['id'] for c in boundary if c['accepted']]
            require(sorted(adopted)==sorted(primary['selected_boundary_ids']), 'adopted IDs differ')
            primary.update(used_count=len(used),final_inlier_count=len(final),used_quality_counts=usedq,
                final_inlier_quality_counts=finalq,object_rank=g.get('object',{}).get('numerical_rank'),
                jacobian_rank=g.get('jacobian',{}).get('numerical_rank'),
                jacobian_condition_number=g.get('jacobian',{}).get('condition_number'),
                physical_rotation_delta_vs_N3_deg=angle)
            output.append(dict(id=fid,session=primary['session'],state_transition=status,
                branch_vs_N3=relation(b[PRIMARY],b['N3_SUBPIX']),
                nativeH_branch_vs_N3=relation(b[NATIVE_H],b['N3_SUBPIX']),
                primary_nativeH_branch_relation=relation(b[PRIMARY],b[NATIVE_H]),
                branches=b,boundary_flag='ADOPTED_BOUNDARY' if adopted else 'NO_ADOPTED_BOUNDARY',
                primary=primary,nativeH=native,oldv3=old,N3=n3,
                delta_N3=delta(primary['values'],n3['values']),
                delta_nativeH=delta(primary['values'],native['values']),
                delta_oldv3=delta(primary['values'],old['values']),boundary_candidates=boundary))
        grouped=defaultdict(list)
        for row in output:
            grouped[(row['state_transition'],row['branch_vs_N3'],row['boundary_flag'])].append(row)
        table=[dict(state_transition=k[0],branch_vs_N3=k[1],boundary_flag=k[2],**aggregate(data))
               for k,data in sorted(grouped.items())]
        total=aggregate(output)
        require(sum(r['n'] for r in table)==245 and len(output)==245,'group partition differs')
        for comparator, method in [('N3','N3_SUBPIX'),('nativeH',NATIVE_H),('oldv3',OLD)]:
            recorded=saved['contrasts'][PRIMARY+'_minus_'+method]['common_operational']
            require(recorded['pair_ids']==ids,'recorded paired IDs differ')
            for key in METRICS:
                actual=total['paired'][comparator][key]['mean'];expected=recorded['metrics'][key]['mean_delta']
                difference=abs(actual-expected);require(difference<=1e-10,'delta replay differs')
                contribution=math.fsum(r['paired'][comparator][key]['contribution_to_full245_mean'] for r in table)
                require(abs(contribution-actual)<=1e-10,'group contribution decomposition differs')
                checks.append(dict(comparator=comparator,metric=key,recorded_mean_delta=expected,
                    replay_mean_delta=actual,group_contribution_sum=contribution,absolute_difference=difference))
                counts['paired_scalar_deltas_computed']+=245
        counts.update(frames_joined=245,groups=len(table),mean_decomposition_checks=len(checks),
            boundary_candidates_examined=sum(len(r['boundary_candidates']) for r in output))
        after=inputs(folder);require(after==before,'audit input bytes changed')
        with gzip.open(folder/'FAULT_ISOLATION_ROWS.jsonl.gz','xt') as stream:
            for row in output:stream.write(json.dumps(row,allow_nan=False,separators=(',',':'))+'\n')
        with (folder/'FAULT_ISOLATION_TABLE.csv').open('x',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['state','branch_vs_N3','boundary','n','ids',
                'deltaT_vsN3_cm','deltaR_vsN3_deg','T_contribution_cm','R_contribution_deg','deltaT_vsNativeH_cm',
                'deltaR_vsNativeH_deg','nativeH_branch_switched','adopted','adopted_improved','adopted_worsened'])
            for r in table:
                writer.writerow([r['state_transition'],r['branch_vs_N3'],r['boundary_flag'],r['n'],'|'.join(r['ids']),
                    r['paired']['N3']['translation_cm']['mean'],r['paired']['N3']['rotation_deg']['mean'],
                    r['paired']['N3']['translation_cm']['contribution_to_full245_mean'],
                    r['paired']['N3']['rotation_deg']['contribution_to_full245_mean'],
                    r['paired']['nativeH']['translation_cm']['mean'],r['paired']['nativeH']['rotation_deg']['mean'],
                    r['nativeH_branch_counts'].get('WD_SWITCHED',0),r['boundary']['adopted']['n'],
                    r['boundary']['adopted']['improved'],r['boundary']['adopted']['worsened']])
        result=dict(schema='fixed_saved_row_fault_isolation_checks_v4',passed=True,complete=True,
            protocol=binding(folder/'FAULT_ISOLATION_PROTOCOL.json'),input_bindings=before,
            unchanged_inputs_after=True,counts=dict(counts),mean_decomposition_checks=checks,
            total=total,table=table,rows=binding(folder/'FAULT_ISOLATION_ROWS.jsonl.gz'),
            table_csv=binding(folder/'FAULT_ISOLATION_TABLE.csv'),actual_arithmetic_runs=1,
            new_model_PnP_ray_training_RGB_GT_scoring_calls=0,physical_causal_ownership_certification=False)
        lines=['# 저장 원행에 의한 v4 오류 분해','',
            '새 모델·PnP·학습·ray·RGB·GT 채점 없이 기존245장 원행을 한 번 조인했다. ',
            '실제 반환 cf_extents로 W/D 분기를 비교했고 기본 반환의 거절 후보를 새 자세로 세지 않았다.',
            '', '|v3→v4 상태|N3 대비 분기|경계 채택|n|Δ위치 cm|Δ회전 °|전체245 위치 기여 cm|전체245 회전 기여 °|',
            '|---|---|---|---:|---:|---:|---:|---:|']
        for r in table:
            values=[r['paired']['N3']['translation_cm']['mean'],r['paired']['N3']['rotation_deg']['mean'],
                    r['paired']['N3']['translation_cm']['contribution_to_full245_mean'],r['paired']['N3']['rotation_deg']['contribution_to_full245_mean']]
            lines.append('|'+ '|'.join([r['state_transition'],r['branch_vs_N3'],r['boundary_flag'],str(r['n'])]+[f'{v:.6f}' for v in values])+'|')
        lines+=['','모든 그룹의 기여도를 더하면 전체 원행 평균 차이와 일치한다. 음수는 N3보다 오차가 작다는 뜻이다.',
            '', '경계 교체 없는 nativeH와의 같은 영상 차이, 같은 분기에서 양쪽 새 자세인 부분집합, ',
            '채택/표시 코너의 상대 참조 정확도, 사용 대응점·최종 inlier의 정확/부정확/미상과 수·배치는 CHECKS와 ROWS에 보존했다.',
            '', 'W/D 분기 변화는 오차와의 연관이다. 실제 물리 분기가 틀렸다는 독립 인증이 아니며, ',
            '경계 채택과 branch 선택은 같은 solver 안에서 상호작용한다. maskH·Base 제안 특징의 의존도 남아 있다.',
            '기존 geometric proxy 점수와 알려진8px 기준을 재사용했으며 새로운 물리 소유권 정답·원인 비율을 만들지 않았다.',
            'rotation≥45°는 이 표의 고정된 설명용 집계이며 설정 선택이나 실패 제외 기준이 아니다.',
            '', '[프로토콜](FAULT_ISOLATION_PROTOCOL.json) · [원행245](FAULT_ISOLATION_ROWS.jsonl.gz) · ',
            '[표 CSV](FAULT_ISOLATION_TABLE.csv) · [검산과 모든 그룹](FAULT_ISOLATION_CHECKS.json)']
        (folder/'FAULT_ISOLATION_KO.md').open('x').write('\n'.join(lines)+'\n')
    except Exception as error:
        failure=dict(type=type(error).__name__,message=str(error))
        result=dict(passed=False,complete=False,exception=failure,counts=dict(counts),
                    input_bindings=before,actual_arithmetic_runs=1,new_model_PnP_ray_training_RGB_GT_scoring_calls=0)
    result['elapsed_seconds']=time.monotonic()-start
    write(folder/'FAULT_ISOLATION_CHECKS.json',result)
    print(json.dumps(dict(passed=result['passed'],counts=result['counts'],exception=failure)),flush=True)
    if failure:raise SystemExit(1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('freeze','run'));parser.add_argument('--input',type=Path,default=DOC)
    args=parser.parse_args();(freeze if args.stage=='freeze' else run)(args.input)


if __name__=='__main__':main()
