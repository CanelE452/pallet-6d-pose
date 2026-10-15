"""Independent postseal accounting using score JSON files only.

Standard-library arithmetic; no research scorer, reference, prediction,
image, model, PnP or objective import/access. Actual IO requires ROOT GO.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
B = ROOT/'_docs/experiments/pallet_visibility_curriculum_20261013_v1/nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1'
NEW = B/'clean_fresh_BI_pose_eval_v4_32_r2'
OLD = B/'geometry_guided_edge_eval_v4_30'
OUTPUT = B/'independent_clean_fresh_BI_score_review_v4_32'
OLD_SEAL_SHA = 'c2c85714f81dd90b78d46fd9e82541c9dc2ec1578e5e93c8f2209815ccf91d87'
PREDICTION_SEAL_SHA = 'eff80efdb1cb9c17f7eb13df50b953467f82bbe3415c8385b9aa146456948b48'
COUNTS = dict(CAL=46, TEST=49, REMAINDER=143, PRIMARY=192)
ARMS = ('SAME_FROZEN_E','CLEAN_FRESH_BI_ALL_BI_NATIVE','CLEAN_FRESH_BI_ALL_BI_CONTAINMENT',
        'CLEAN_FRESH_BI_SOURCE_NATIVE','CLEAN_FRESH_BI_SOURCE_CONTAINMENT')
LABELS = ('고정 e','전체 B/I · native','전체 B/I · 포함 비용','RGB q 보존 · native','RGB q 보존 · 포함 비용')


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


class Checks:
    def __init__(self):self.count=0;self.failures=[];self.max_numeric_difference=0.
    def check(self,condition,context):
        self.count+=1
        if not condition:self.failures.append(context)
    def equal_number(self,actual,expected,context):
        if actual is None or expected is None:
            self.check(actual is expected,context);return
        self.max_numeric_difference=max(self.max_numeric_difference,abs(float(actual)-float(expected)))
        self.check(math.isclose(float(actual),float(expected),rel_tol=1e-12,abs_tol=1e-10),context)


def moments(values):
    finite=[float(value)for value in values if value is not None]
    if any(not math.isfinite(value)for value in finite):raise ValueError('Saved finite errors required')
    stats=dict(count=len(finite),mean=math.fsum(finite)/len(finite)if finite else None,
               max=max(finite)if finite else None)
    return dict(count=len(values),missing_or_undefined_count=len(values)-len(finite),
        mean=stats['mean']if len(finite)==len(values)else None,
        max=stats['max']if len(finite)==len(values)else None,finite_only_diagnostic=stats)


def aggregate(report,arms,checks):
    rows=report['rows'];count=len(rows);result={}
    checks.check(report['count']==count,'report count')
    checks.check(len({row['id']for row in rows})==count,'unique cohort IDs')
    for arm in arms:
        ts=[];rs=[];tpass=rpass=joint=fallback=missing=unaccepted=0
        for row in rows:
            checks.check(set(row['metrics'])==set(arms),'all declared metrics '+row['id'])
            checks.check(set(row['accepted'])==set(arms),'all declared accepted flags '+row['id'])
            value=row['metrics'][arm];t=value['translation_cm'];r=value['rotation_deg']
            available=value['pose_available'];accepted=row['accepted'][arm]
            checks.check(isinstance(available,bool)and isinstance(accepted,bool),'typed availability/accepted '+row['id']+'/'+arm)
            checks.check(row['pose_available'][arm]is available,'row metric pose availability '+row['id']+'/'+arm)
            checks.check(available or(t is None and r is None and not accepted),'no pose remains failure '+row['id']+'/'+arm)
            trule=bool(available and t is not None and t<=5.)
            rrule=bool(available and r is not None and r<=5.)
            jrule=(None if available and r is None else bool(trule and rrule))
            checks.check(value['translation_le5']is trule,'inclusive translation threshold '+row['id']+'/'+arm)
            checks.check(value['success_5cm_5deg']is jrule,'inclusive joint threshold '+row['id']+'/'+arm)
            ts.append(t);rs.append(r);tpass+=trule;rpass+=rrule;joint+=bool(jrule)
            missing+=not available;fallback+=bool(not accepted and available);unaccepted+=not accepted
        q=dict(count=count,translation_le5=tpass,rotation_le5=rpass,successes_5cm_5deg=joint,
            translation_cm=moments(ts),rotation_deg=moments(rs),missing_pose_count=missing,
            rotation_unresolved=sum(value is None for value in rs),fallback_count=fallback,unaccepted_count=unaccepted)
        finite=[row for row in rows if row['metrics'][arm]['translation_cm']is not None]
        absent=[row['id']for row in rows if not row['metrics'][arm]['pose_available']]
        q['max_translation_id']=absent[0]if absent else max(finite,key=lambda row:row['metrics'][arm]['translation_cm'])['id']if finite else None
        summary=report['summary'][arm]
        for key in ('translation_le5','successes_5cm_5deg','missing_pose_count','rotation_unresolved','fallback_count','unaccepted_count','max_translation_id'):
            checks.check(q[key]==summary[key],'summary '+arm+'/'+key)
        checks.check(summary['denominator']==count,'summary full denominator '+arm)
        for metric in ('translation_cm','rotation_deg'):
            for field in ('mean','max'):
                checks.equal_number(summary[metric][field],q[metric][field],'summary '+arm+'/'+metric+'/'+field)
        result[arm]=q
    return result


def regressions(rows,arm):
    threshold=[];joint=[];increases=[]
    for row in rows:
        e=row['metrics'][ARMS[0]];value=row['metrics'][arm]
        record=dict(id=row['id'],e_translation_cm=e['translation_cm'],translation_cm=value['translation_cm'],
            e_rotation_deg=e['rotation_deg'],rotation_deg=value['rotation_deg'],
            accepted=row['accepted'][arm],pose_available=value['pose_available'])
        if e['translation_cm']is not None and e['translation_cm']<=5. and(value['translation_cm']is None or value['translation_cm']>5.):threshold.append(record)
        if bool(e['success_5cm_5deg'])and not bool(value['success_5cm_5deg']):joint.append(record)
        if e['translation_cm']is not None and(value['translation_cm']is None or value['translation_cm']>e['translation_cm']):increases.append(record)
    return dict(translation_threshold_regressions=threshold,joint_threshold_regressions=joint,
        translation_error_increases=increases,definition='Separate threshold losses from any finite numerical increase; postscore only, never runtime routing.')


def largest_errors(rows,arm):
    ordered=sorted(enumerate(rows),key=lambda item:(-(float('inf')if item[1]['metrics'][arm]['translation_cm']is None else item[1]['metrics'][arm]['translation_cm']),item[0]))
    return [dict(id=row['id'],translation_cm=row['metrics'][arm]['translation_cm'],
        rotation_deg=row['metrics'][arm]['rotation_deg'],accepted=row['accepted'][arm],
        pose_available=row['metrics'][arm]['pose_available'])for _,row in ordered[:5]]


def _identity(binding):return(binding['path'],binding['sha256'])


def compare_e(before,after,checks):
    checks.check([row['id']for row in before]==[row['id']for row in after],'V30 e ordered cohort IDs')
    for old,new in zip(before,after):
        checks.check(old['accepted'][ARMS[0]]==new['accepted'][ARMS[0]],'V30 e accepted '+new['id'])
        a,b=old['metrics'][ARMS[0]],new['metrics'][ARMS[0]]
        for key in ('translation_cm','rotation_deg','ADDsym_cm'):
            checks.check(a[key]==b[key],'V30 e exact metric '+new['id']+'/'+key)
        checks.check(a['success_5cm_5deg']==b['legacy_strict_success_5cm_5deg'],'V30 legacy strict boolean '+new['id'])


def load_score_files(folder,seal,parts):
    reports={};bindings={}
    for part in parts:
        path=folder/(part+'_SCORE.json')
        if sha(path)!=seal['score_sha256'][part]:raise ValueError('Changed sealed score '+part)
        reports[part]=read(path);bindings[str(path.relative_to(ROOT))]=sha(path)
    return reports,bindings


def markdown(data):
    lines=['# V32 전체 점수 독립 집계','',
        '점수 JSON만 재집계했습니다. e 수치·수용 여부는 V30과 동일하며 새 기본 방법을 채택하지 않았습니다.',
        '위치와 회전 기준은 각각 ≤5cm와 ≤5°입니다. 공동은 두 조건을 모두 만족한 수입니다. 실패·fallback·회전 미확정 행도 원래 분모에 남습니다.','']
    fmt=lambda value:'미확정'if value is None else f'{value:.3f}'
    for part,title in (('PRIMARY','전체 192'),('TEST','기존 TEST 49'),('REMAINDER','severe 제외 추가 143'),('CAL','별도 CAL 46')):
        lines.extend(['## '+title,'','| 방법 | 수 | 위치≤5cm | 회전≤5° | 공동 | 평균 위치 cm | 최대 위치 cm | 평균 회전 ° | 최대 회전 ° | fallback |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|'])
        for arm,label in zip(ARMS,LABELS):
            q=data['populations'][part][arm]
            values=[label,str(q['count']),str(q['translation_le5']),str(q['rotation_le5']),str(q['successes_5cm_5deg']),
                fmt(q['translation_cm']['mean']),fmt(q['translation_cm']['max']),fmt(q['rotation_deg']['mean']),fmt(q['rotation_deg']['max']),str(q['fallback_count'])]
            lines.append('| '+' | '.join(values)+' |')
        lines.append('')
    lines.extend(['## 원래 49장의 퇴행','',
        '위치 기준 퇴행은 e가 ≤5cm인 사례가 새 방법에서 >5cm 또는 자세 없음이 된 경우입니다. 전체 ID·위치·회전 값과 공동 기준 퇴행, 단순 위치 오차 증가 목록은 [집계 JSON](AUDIT.json)에 구분해 저장했습니다.',''])
    for arm,label in zip(ARMS[1:],LABELS[1:]):
        item=data['TEST49_regressions'][arm]
        lines.append(f'- {label}: 위치 기준 {len(item["translation_threshold_regressions"])}건, 공동 기준 {len(item["joint_threshold_regressions"])}건.')
    lines.extend(['','## 전체 192의 최대 위치 오차','',
        '| 방법 | 사례 | 위치 cm | 회전 ° |','|---|---|---:|---:|'])
    for arm,label in zip(ARMS,LABELS):
        worst=data['largest_errors']['PRIMARY'][arm][0]
        lines.append('| '+' | '.join((label,worst['id'],fmt(worst['translation_cm']),fmt(worst['rotation_deg'])))+' |')
    lines.extend(['','각 집단·방법의 상위 5건은 집계 JSON에 보존했습니다. 이는 사후 설명용이며 방법이나 매개변수 선택에 사용하지 않습니다.'])
    lines.extend(['','같은 TEST를 반복 사용한 탐색 결과입니다. 추가 143의 참조는 annotation/PnP 유도값이며 독립 물리 계측이 아닙니다.',''])
    return '\n'.join(lines)


def run(*,root_score_review_GO=False,expected_score_seal_sha256=None,output=OUTPUT):
    if root_score_review_GO is not True:raise PermissionError('ROOT postscore review GO before any IO')
    if not isinstance(expected_score_seal_sha256,str)or len(expected_score_seal_sha256)!=64:raise ValueError('Caller pins completed score seal')
    if sha(NEW/'SCORE_SEAL.json')!=expected_score_seal_sha256 or sha(OLD/'SCORE_SEAL.json')!=OLD_SEAL_SHA:raise ValueError('Exact new/old score seal bytes required')
    output=Path(output)
    if output.exists():raise FileExistsError('Append-only independent score review')
    score=read(NEW/'SCORE_SEAL.json');oldseal=read(OLD/'SCORE_SEAL.json')
    if score['prediction_seal']['sha256']!=PREDICTION_SEAL_SHA or not score['all_rows_kept']:raise ValueError('Complete frozen238x5 prediction and score required')
    reports,inputs=load_score_files(NEW,score,COUNTS)
    oldreports,oldinputs=load_score_files(OLD,oldseal,('CAL','TEST','REMAINDER'));inputs.update(oldinputs)
    checks=Checks();populations={}
    checks.check(score['default']=='SAME_FROZEN_E unchanged','default remains frozen e')
    for part,count in COUNTS.items():
        checks.check(reports[part]['count']==count,'expected '+part+' cohort')
        expected_parts=Counter(TEST=49,REMAINDER=143)if part=='PRIMARY'else Counter({part:count})
        checks.check(Counter(row['partition']for row in reports[part]['rows'])==expected_parts,'exact cohort membership '+part)
        populations[part]=aggregate(reports[part],ARMS,checks)
    checks.check(reports['PRIMARY']['rows']==reports['TEST']['rows']+reports['REMAINDER']['rows'],'PRIMARY exact TEST+REMAINDER rows')
    checks.check(len({row['id']for part in ('CAL','TEST','REMAINDER')for row in reports[part]['rows']})==238,'unique full238')
    for part in ('CAL','TEST','REMAINDER'):
        compare_e(oldreports[part]['rows'],reports[part]['rows'],checks)
        checks.check(_identity(score['reference_bindings'][part])==_identity(oldseal['reference_bindings'][part]),'same reference path/SHA '+part)
        checks.check(_identity(reports[part]['reference_binding'])==_identity(score['reference_bindings'][part]),'score reference metadata '+part)
        checks.check(_identity(reports['PRIMARY']['reference_binding'][part])==_identity(score['reference_bindings'][part]),'PRIMARY reference metadata '+part)
    inputs.update({str((NEW/'SCORE_SEAL.json').relative_to(ROOT)):expected_score_seal_sha256,str((OLD/'SCORE_SEAL.json').relative_to(ROOT)):OLD_SEAL_SHA})
    data=dict(schema='INDEPENDENT_SCORE_ONLY_V32_ACCOUNTING_V1',UTC=datetime.now(timezone.utc).isoformat(),
        checks=checks.count,failure_count=len(checks.failures),failures=checks.failures,
        max_numeric_difference=checks.max_numeric_difference,populations=populations,
        TEST49_regressions={arm:regressions(reports['TEST']['rows'],arm)for arm in ARMS[1:]},
        largest_errors={part:{arm:largest_errors(report['rows'],arm)for arm in ARMS}for part,report in reports.items()},
        rows238=True,partitions=dict(CAL=46,TEST=49,REMAINDER=143,PRIMARY=192),arms=list(ARMS),
        exact_V30_e_numeric_metrics_and_accepted_verified=not checks.failures,baseline_default_unchanged=not checks.failures,
        reference_path_SHA_metadata_unchanged=not checks.failures,source_sha256={str(Path(__file__).relative_to(ROOT)):sha(__file__)},
        input_sha256=inputs,new_reference_RGB_model_PnP_optimizer_or_objective_calls=0,
        independent_numerical_pose_cost_audit_performed_here=False,
        no_runtime_method_or_parameter_selection=True,missing_or_undefined_rows_preserve_full_denominator=True)
    output.mkdir(parents=True)
    path=output/('AUDIT.json'if not checks.failures else 'FAILED_AUDIT.json')
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    if checks.failures:raise AssertionError('Independent score checks failed: '+str(len(checks.failures)))
    report=output/'REPORT_KO.md';report.write_text(markdown(data),encoding='utf-8')
    seal=dict(receipt_sha256=sha(path),report_sha256=sha(report),source_sha256=data['source_sha256'],input_sha256=inputs)
    (output/'AUDIT_SEAL.json').write_text(json.dumps(seal,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(checks=checks.count,failures=0,receipt_sha256=sha(path),report_sha256=sha(report),seal_sha256=sha(output/'AUDIT_SEAL.json'))))
    return data


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-score-review-GO',action='store_true')
    parser.add_argument('--score-seal-sha256',required=True)
    args=parser.parse_args();run(root_score_review_GO=args.root_score_review_GO,expected_score_seal_sha256=args.score_seal_sha256)
