"""Posthoc common-image outlier sensitivity; keep the original full evaluation."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import time

import numpy as np

ARMS=('POSEFIX_RAW','R0','N0','N1','E0')
NAMES={'POSEFIX_RAW':'PoseFix RAW','R0':'RAW YOLO + F','N0':'N3→SubPix + F','N1':'N3→SubPix + 최소 B/I','E0':'기존 전체 방법 e'}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,obj):
    with Path(p).open('x') as f:
        f.write(obj if isinstance(obj,str) else json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def distribution(values):
    x=np.asarray(values,dtype=np.float64)
    if not np.isfinite(x).all():raise ValueError('Report unavailable errors explicitly; never silently trim them')
    return dict(n=len(x),mean=float(x.mean()),variance_sample=float(x.var(ddof=1)),SD_sample=float(x.std(ddof=1)),
                median=float(np.median(x)),max=float(x.max()))


def summarize(rows):
    arms={}
    for arm in ARMS:
        arms[arm]=dict(full_n=len(rows),success=sum(r['metrics'][arm]['success'] for r in rows),
            errors={metric:distribution([r['metrics'][arm][metric] for r in rows]) for metric in ('translation_cm','rotation_deg')})
    pairs={}
    for arm,baseline in (('N1','N0'),('E0','N0'),('N0','POSEFIX_RAW'),('N1','POSEFIX_RAW'),('E0','POSEFIX_RAW')):
        groups={name:[] for name in ('recovery','damage','both_success','both_failure')}
        for r in rows:
            a=r['metrics'][arm]['success'];b=r['metrics'][baseline]['success']
            group='both_success' if a and b else 'recovery' if a else 'damage' if b else 'both_failure'
            groups[group].append(r['id'])
        pairs[arm+'-'+baseline]=dict(transition={k:dict(count=len(v),ids=v) for k,v in groups.items()},
            mean_difference={metric:arms[arm]['errors'][metric]['mean']-arms[baseline]['errors'][metric]['mean'] for metric in ('translation_cm','rotation_deg')})
    return dict(full_n=len(rows),arms=arms,paired=pairs)


def table(pop):
    lines=['| 경로 | 성공/전체 | 위치 평균 cm | 위치 SD cm | 위치 분산 cm² | 회전 평균 ° | 회전 SD ° | 회전 분산 °² |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        a=pop['arms'][arm];t=a['errors']['translation_cm'];r=a['errors']['rotation_deg']
        lines.append(f'| {NAMES[arm]} | {a["success"]}/{a["full_n"]} | {t["mean"]:.3f} | {t["SD_sample"]:.3f} | {t["variance_sample"]:.3f} | {r["mean"]:.3f} | {r["SD_sample"]:.3f} | {r["variance_sample"]:.3f} |')
    return '\n'.join(lines)


def run(summary,output):
    start=time.monotonic();summary,output=map(Path,(summary,output))
    if output.exists():raise FileExistsError('Preserve prior outlier analysis')
    seal=read(summary/'SEAL.json')
    for name,value in seal['artifacts'].items():
        if sha(summary/name)!=value:raise ValueError('Original saved-result summary changed')
    records=[]
    with (summary/'ALL_ROWS.csv').open(newline='') as f:
        for row in csv.DictReader(f):
            if row['partition']=='CAL':continue
            metrics={}
            for arm in ARMS:
                if row[arm+'__pose_available']!='True':raise ValueError('Retain no-pose in full denominator before any exclusions')
                t=float(row[arm+'__translation_cm']);r=float(row[arm+'__rotation_deg']);success=t<=5 and r<=5
                if success!=(row[arm+'__success_le5']=='True'):raise ValueError('Fixed <=5 success changed')
                metrics[arm]=dict(translation_cm=t,rotation_deg=r,success=success)
            records.append(dict(id=row['id'],session=row['session'],partition=row['partition'],metrics=metrics))
    if len(records)!=192 or len({r['id'] for r in records})!=192:raise ValueError('Full PRIMARY192 required')
    thresholds={}
    for metric in ('translation_cm','rotation_deg'):
        q1,q3=np.percentile([r['metrics']['N0'][metric] for r in records],[25,75],method='linear')
        thresholds[metric]=dict(Q1=float(q1),Q3=float(q3),IQR=float(q3-q1),upper=float(q3+1.5*(q3-q1)),quantile_method='linear')
    rules=[('FULL192','전체192장 · 제외 없음',lambda r:False),
        ('N0_IQR_T_R','N0 위치 또는 회전 상단 IQR 초과 제외',lambda r:any(r['metrics']['N0'][m]>thresholds[m]['upper'] for m in thresholds)),
        ('ALL5_IQR_T_R','보조: 같은 IQR 상한을 다섯 경로 모두에 적용해 공통 제외',lambda r:any(r['metrics'][a][m]>thresholds[m]['upper'] for a in ARMS for m in thresholds)),
        ('N0_IQR_T_ONLY','보조: N0 위치 상단 IQR 초과만 제외',lambda r:r['metrics']['N0']['translation_cm']>thresholds['translation_cm']['upper']),
        ('N0_T_GT50','보조: N0 위치50cm 초과 제외',lambda r:r['metrics']['N0']['translation_cm']>50),
        ('ALL5_T_GT50','보조: 다섯 경로 중 하나라도 위치50cm 초과인 영상 제외',lambda r:any(r['metrics'][a]['translation_cm']>50 for a in ARMS)),
        ('N0_TOP10_T','보조: N0 위치 오차 상위10장 제외',lambda r:r['id'] in top10)]
    top10={r['id'] for r in sorted(records,key=lambda r:(-r['metrics']['N0']['translation_cm'],r['id']))[:10]}
    cohorts={};flat=[]
    for name,label,predicate in rules:
        excluded=[r for r in records if predicate(r)];kept=[r for r in records if not predicate(r)]
        cohort=dict(label=label,original_n=192,excluded_n=len(excluded),excluded_ids=[r['id'] for r in excluded],retained_ids=[r['id'] for r in kept],
                    same_IDs_for_every_method=True,summary=summarize(kept))
        cohorts[name]=cohort
        for arm in ARMS:
            a=cohort['summary']['arms'][arm];t=a['errors']['translation_cm'];rot=a['errors']['rotation_deg']
            flat.append(dict(cohort=name,arm=arm,original_n=192,excluded_n=len(excluded),retained_n=len(kept),success=a['success'],
                T_mean_cm=t['mean'],T_variance_cm2=t['variance_sample'],T_SD_cm=t['SD_sample'],
                R_mean_deg=rot['mean'],R_variance_deg2=rot['variance_sample'],R_SD_deg=rot['SD_sample']))
    result=dict(schema='COMMON_IMAGE_POSTHOC_OUTLIER_SENSITIVITY_V1',original_primary_n=192,CAL_excluded=True,
        success_definition='T<=5cm AND physical/C2 R<=5deg',variance_ddof=1,threshold_anchor='frozen N0, not fitted again',thresholds=thresholds,
        main_rule='N0_IQR_T_R',cohorts=cohorts,original_full_metrics_unchanged=True,
        not_new_primary_or_unseen_test=True,not_proof_algorithm_improved=True,
        method_specific_independent_trimming=False,exclusions_use_reference_errors=True,new_training_or_prediction_or_reference_fit=0)
    output.mkdir(parents=True)
    write(output/'OUTLIER_RESULTS.json',result)
    with (output/'SUMMARY.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    sections=['# 아웃라이어 제외 후의 동일 영상 비교',
        '**이 결과는 사용자의 요청에 따른 사후 민감도 분석이다. 원래192장 평가와 실패를 삭제하거나 대체하지 않는다.** '
        '모든 경로에서 같은 ID를 제외했다. 방법별로 각자 잘 나온 영상만 남기는 집계는 하지 않았다. '
        '제외 기준은 참조 오차를 사용하므로 실제 추론 시 동작하는 gate나 새 독립 평가 성능이 아니다.',
        f'주 기준은 N3→SubPix + F(N0)의 위치 또는 회전 오차가 상단 Q3+1.5×IQR을 넘는 영상이다. '
        f'전체192장의 위치 상한은{thresholds["translation_cm"]["upper"]:.6f}cm, 회전 상한은{thresholds["rotation_deg"]["upper"]:.6f}°다. '
        '분산은 ddof=1 표본분산이며, 단위는cm²/°²다. CAL46은 합산하지 않았다.']
    for name,c in cohorts.items():
        sections.extend([f'## {c["label"]}',f'192장 중{c["excluded_n"]}장 제외, 공통{c["summary"]["full_n"]}장 유지.',table(c['summary'])])
        pair=c['summary']['paired']['N1-N0'];tr=pair['transition'];diff=pair['mean_difference']
        sections.append(f'최소 B/I 대 N0: 회복{tr["recovery"]["count"]}장·손상{tr["damage"]["count"]}장, '
            f'위치 평균 차이{diff["translation_cm"]:+.6f}cm·회전 평균 차이{diff["rotation_deg"]:+.6f}°. 음수면 B/I가 개선한 것이다.')
    sections.extend(['## 해석의 제한',
        '기준을 바꾸면 남는 영상과 평균이 달라진다. N0로만 제외하면 N1에서 새로 발생한 큰 실패는 남을 수 있다. '
        'N0 기준으로 남긴 자료는 N0가 잘 맞은 조건의 분석이므로 N0에 유리한 선택이기도 하다. '
        '반대로 모든 경로의 큰 오차 합집합을 제외하면 N1의 손상 사례까지 제거하므로 후단의 안전성을 평가할 수 없다. '
        '그 차이를 공개하기 위해 보조 기준도 함께 보존한다. 어느 기준이 더 좋은 숫자를 보이는지에 따라 운영 정책을 선택하지 않았다.',
        '[모든 제외·유지 ID와 짝지은 승패](OUTLIER_RESULTS.json), [통계 CSV](SUMMARY.csv).'])
    write(output/'OUTLIER_RESULTS_KO.md','\n\n'.join(sections)+'\n')
    write(output/'RECEIPT.json',dict(input_summary_seal_sha256=sha(summary/'SEAL.json'),input_rows_sha256=sha(summary/'ALL_ROWS.csv'),
        source_sha256=sha(__file__),wall_seconds=time.monotonic()-start,original_records=192,cohorts=len(cohorts),new_model_calls=0,new_pose_fits=0,
        original_output_modified=False))
    write(output/'SEAL.json',dict(assets={p.name:sha(p) for p in output.iterdir() if p.is_file()}))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--summary',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.summary,a.output)
