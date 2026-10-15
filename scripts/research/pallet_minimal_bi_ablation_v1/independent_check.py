"""Independent saved-score recount: stdlib/numpy only, no scorer imports.

No image/reference/model/prediction loader exists here. This checker consumes
only the sealed score artifacts after every238x4 prediction has been closed.
Bootstrap uses the same declared random draws but independently expands each
recording multiplicity back to frame weights for the pooled estimand.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import time

import numpy as np

ARMS=('R0','R1','N0','N1')
CONTRASTS=(('R1','R0'),('N1','N0'),('R1','N1'),('R1','N0'))
ERRORS=('translation_cm','rotation_deg','ADDsym_cm')
PARTITIONS={'CAL':46,'TEST':49,'REMAINDER':143,'PRIMARY':192}
BOOTSTRAP_SEED=20260917
BOOTSTRAP_DRAWS=10000


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path)
    with path.open('x') as stream:
        stream.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def sample_distribution(values):
    values=list(values)
    finite=[float(v) for v in values if v is not None]
    if not all(math.isfinite(v) for v in finite):raise ValueError('Nonfinite saved error')
    n=len(finite);mean=math.fsum(finite)/n if n else None
    variance=math.fsum((v-mean)**2 for v in finite)/(n-1) if n>1 else None
    return dict(n=n,denominator=len(values),missing=len(values)-n,mean=mean,
                sample_std=math.sqrt(variance) if variance is not None else None,max=max(finite) if n else None)


def transition_table(rows,new,before):
    ids={k:[] for k in ('failure_to_success','success_to_failure','both_success','both_failure','reference_unresolved')}
    for row in rows:
        a,b=[row['metrics'][arm]['success_le5'] for arm in (new,before)]
        if a is None or b is None:key='reference_unresolved'
        elif a and not b:key='failure_to_success'
        elif b and not a:key='success_to_failure'
        elif a:key='both_success'
        else:key='both_failure'
        ids[key].append(row['id'])
    return dict(counts={k:len(v) for k,v in ids.items()},ids=ids,full_denominator=len(rows))


def frame_resample_weights(rows):
    """Draw recordings, then explicitly give every original frame its weight."""
    clusters=sorted({r['cluster_id'] for r in rows})
    if len(rows)==0 or not clusters:raise ValueError('Nonempty fixed paired population required')
    counts=np.random.default_rng(BOOTSTRAP_SEED).multinomial(
        len(clusters),[1/len(clusters)]*len(clusters),size=BOOTSTRAP_DRAWS).astype(np.uint16)
    position={c:i for i,c in enumerate(clusters)}
    frame_weights=counts[:,[position[r['cluster_id']] for r in rows]].astype(np.float64)
    return frame_weights,dict(seed=BOOTSTRAP_SEED,resamples=BOOTSTRAP_DRAWS,units=len(clusters),clusters=clusters,
        cluster_sizes=dict(Counter(r['cluster_id'] for r in rows)),session_sizes=dict(Counter(r['session'] for r in rows)),
        draw_sha256=hashlib.sha256(counts.tobytes()).hexdigest())


def paired_interval(values,rows,frame_weights):
    valid=np.asarray([v is not None for v in values],bool)
    x=np.asarray([float(v) for v in values if v is not None],float)
    if not np.isfinite(x).all():raise ValueError('Nonfinite saved paired difference')
    selected=frame_weights[:,valid]
    denominator=selected.sum(axis=1)
    keep=denominator>0
    estimates=(selected[keep]@x)/denominator[keep]
    units=len({r['cluster_id'] for r,ok in zip(rows,valid) if ok})
    return dict(mean_difference=math.fsum(x.tolist())/len(x) if len(x) else None,
        CI95=np.quantile(estimates,[.025,.975]).tolist() if units>1 and len(estimates) else None,
        common_available_n=int(valid.sum()),full_denominator=len(values),excluded_unavailable_n=int((~valid).sum()),
        eligible_clusters=units,valid_resamples=int(keep.sum()),empty_resamples=int((~keep).sum()),
        improved=int((x<0).sum()),worsened=int((x>0).sum()),tied=int((x==0).sum()))


def recount_population(rows):
    rows=list(rows)
    weights,bootstrap=frame_resample_weights(rows)
    arms={}
    for arm in ARMS:
        metrics=[r['metrics'][arm] for r in rows]
        statuses=[r['prediction_status'][arm] for r in rows]
        common={key:[r['metrics'][arm][key] for r in rows if all(r['metrics'][a][key] is not None for a in ARMS)] for key in ERRORS}
        arms[arm]=dict(denominator=len(rows),successes_le5=sum(m['success_le5'] is True for m in metrics),
            success_rate_le5=sum(m['success_le5'] is True for m in metrics)/len(rows),
            reference_unresolved_n=sum(m['success_le5'] is None for m in metrics),
            successes_strict_lt5=sum(m['success_strict_lt5'] is True for m in metrics),
            translation_exactly_5_n=sum(m['translation_cm']==5. for m in metrics),
            rotation_exactly_5_n=sum(m['rotation_deg']==5. for m in metrics),
            strict_vs_inclusive_changed_ids=[r['id'] for r in rows if r['metrics'][arm]['success_le5'] is True and r['metrics'][arm]['success_strict_lt5'] is False],
            pose_available_n=sum(m['pose_available'] for m in metrics),no_pose_n=sum(not m['pose_available'] for m in metrics),
            accepted_n=sum(s['accepted'] is True for s in statuses),accepted_available_n=sum(s['accepted'] is not None for s in statuses),
            fallback_n=sum(s['fallback'] is True for s in statuses),fallback_available_n=sum(s['fallback'] is not None for s in statuses),
            status_counts=dict(Counter(str(s['status']) for s in statuses)),
            errors={key:sample_distribution(m[key] for m in metrics) for key in ERRORS},
            common_all_four_available_errors={key:sample_distribution(common[key]) for key in ERRORS})
    paired={}
    for new,before in CONTRASTS:
        binary=[None if r['metrics'][new]['success_le5'] is None or r['metrics'][before]['success_le5'] is None
            else int(r['metrics'][new]['success_le5'])-int(r['metrics'][before]['success_le5']) for r in rows]
        rate=paired_interval(binary,rows,weights)
        rate['gains']=rate.pop('worsened');rate['losses']=rate.pop('improved')
        if any(x is None for x in binary):
            rate['conditional_known_label_mean_difference']=rate['mean_difference']
            rate['mean_difference']=None;rate['CI95']=None
        paired[new+'-'+before]=dict(new=new,reference=before,success_rate=rate,transition=transition_table(rows,new,before),
            continuous={key:paired_interval([None if r['metrics'][new][key] is None or r['metrics'][before][key] is None
                else r['metrics'][new][key]-r['metrics'][before][key] for r in rows],rows,weights) for key in ERRORS})
    return dict(rows=len(rows),arms=arms,comparisons=paired,bootstrap=bootstrap,
        common_all_four_available_n={key:sum(all(r['metrics'][a][key] is not None for a in ARMS) for r in rows) for key in ERRORS})


def compare_subset(expected,actual,path='',failures=None):
    failures=[] if failures is None else failures
    if isinstance(expected,dict):
        if not isinstance(actual,dict):failures.append(dict(path=path,reason='missing dict'));return 1,failures
        count=0
        for key,value in expected.items():
            if key not in actual:failures.append(dict(path=path+'/'+key,reason='missing key'));count+=1
            else:
                n,_=compare_subset(value,actual[key],path+'/'+key,failures);count+=n
        return count,failures
    same=expected==actual
    if isinstance(expected,float) and isinstance(actual,(int,float)):
        same=math.isclose(expected,float(actual),rel_tol=0,abs_tol=1e-9)
    if isinstance(expected,list) and isinstance(actual,list) and len(expected)==len(actual):
        if all(isinstance(x,(int,float)) for x in expected+actual):
            same=bool(np.allclose(expected,actual,rtol=0,atol=1e-9))
    if not same:failures.append(dict(path=path,expected=expected,actual=actual))
    return 1,failures


def run(*,score_folder,input_seal,expected_score_seal_sha256,output,root_recount_GO=False):
    if root_recount_GO is not True:raise PermissionError('Full952 sealed scores and explicit recount GO required before IO')
    if not re.fullmatch('[0-9a-f]{64}',expected_score_seal_sha256):raise ValueError('Exact SCORE_SEAL SHA256 required')
    start=time.monotonic();cpu=time.process_time();folder=Path(score_folder);output=Path(output)
    if output.exists():raise FileExistsError('Preserve every independent recount attempt')
    if sha(folder/'SCORE_SEAL.json')!=expected_score_seal_sha256:raise ValueError('Score seal changed')
    seal=read(folder/'SCORE_SEAL.json')
    if seal['rows']!=238 or seal['arm_rows']!=952 or seal.get('all_rows_kept') is not True:raise ValueError('Full238x4 score inventory required')
    for name,value in seal['assets'].items():
        if Path(name).name!=name or sha(folder/name)!=value:raise ValueError('Closed score asset binding mismatch')
    source_rows=read(folder/'SCORED_ROWS.json')
    rows=source_rows['rows'];frozen=read(input_seal)
    if len(rows)!=238 or source_rows['arm_rows']!=952 or [r['id'] for r in rows]!=frozen['ordered_ids']:
        raise ValueError('No dropped/reordered/duplicate frame allowed')
    expected_parts=['CAL']*46+['TEST']*49+['REMAINDER']*143
    if [r['partition'] for r in rows]!=expected_parts or len({r['id'] for r in rows})!=238:
        raise ValueError('Original ordered CAL46/TEST49/R143 required')
    for r in rows:
        if set(r['metrics'])!=set(ARMS) or set(r['prediction_status'])!=set(ARMS):raise ValueError('All4 arms including failures required')
        for arm in ARMS:
            m=r['metrics'][arm];t=m['translation_cm'];rotation=m['rotation_deg']
            if t is not None and rotation is not None:
                if m['success_le5']!=(t<=5 and rotation<=5) or m['success_strict_lt5']!=(t<5 and rotation<5):raise ValueError('Threshold truth table mismatch')
    actual=read(folder/'RESULTS.json')['populations']
    populations={key:[r for r in rows if r['partition']==key] for key in ('CAL','TEST','REMAINDER')}
    populations['PRIMARY']=[r for r in rows if r['partition']!='CAL']
    reconstructed={key:recount_population(v) for key,v in populations.items()}
    checks,failures=compare_subset(reconstructed,actual)
    output.mkdir(parents=True)
    write(output/'RECOUNT.json',reconstructed)
    receipt=dict(schema='INDEPENDENT_MINIMAL_BI_SAVED_SCORE_RECOUNT_V1',checks=checks,failure_count=len(failures),failures=failures,
        rows=238,arm_rows=952,partitions=PARTITIONS,score_seal_sha256=expected_score_seal_sha256,input_seal_sha256=sha(input_seal),
        source_sha256={str(Path(__file__).resolve()):sha(__file__)},input_sha256={str(folder/name):value for name,value in seal['assets'].items()},
        no_scorer_or_statistics_imports=True,new_reference_or_prediction_or_image_or_model_opens=0,
        bootstrap='Independent frame-weight expansion of same seed20260917 multinomial10000 recording draws',
        resources=dict(wall_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu,processes=1))
    write(output/'RECEIPT.json',receipt)
    lines=['# 독립 저장 점수 재집계','',f'고정 238장×4방법 952개 점수, 검사 {checks:,}개 / 실패 {len(failures)}개. 참조·영상·추론·PnP를 열거나 실행하지 않았다.','',
        '| PRIMARY 192장 | 5cm·5° 이하 | 위치 평균 cm | 위치 표본 SD | 위치 최대 cm | 회전 평균 ° | 회전 최대 ° |','|---|---:|---:|---:|---:|---:|---:|']
    for arm,v in reconstructed['PRIMARY']['arms'].items():
        t=v['errors']['translation_cm'];r=v['errors']['rotation_deg']
        fmt=lambda x:'NA' if x is None else f'{x:.6f}'
        lines.append(f"| {arm} | {v['successes_le5']}/192 | {fmt(t['mean'])} | {fmt(t['sample_std'])} | {fmt(t['max'])} | {fmt(r['mean'])} | {fmt(r['max'])} |")
    lines+=['','성공률 분모에는 모든 실패·no-pose를 유지한다. 수치 오차 요약은 available-only이며 누락 수를 JSON에 보존한다. 6개 원촬영 군집의 frame-weighted bootstrap10,000회이며 적은 군집·반복 개발 자료의 한계를 유지한다.','',
        '이 집계는 현재 네 경로의 비교이며 기본 방법 승격이나 전체 192장 목표 달성을 자동으로 선언하지 않는다.']
    (output/'REPORT_KO.md').write_text('\n'.join(lines)+'\n')
    write(output/'SEAL.json',dict(receipt_sha256=sha(output/'RECEIPT.json'),assets={p.name:sha(p) for p in output.iterdir() if p.is_file()}))
    if failures:raise ValueError('Independent recount found mismatches; attempt preserved')
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--score-folder',type=Path,required=True);p.add_argument('--input-seal',type=Path,required=True)
    p.add_argument('--score-seal-sha256',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--root-recount-GO',action='store_true')
    a=p.parse_args();print(run(score_folder=a.score_folder,input_seal=a.input_seal,expected_score_seal_sha256=a.score_seal_sha256,output=a.output,root_recount_GO=a.root_recount_GO))
