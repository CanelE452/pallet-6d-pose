"""Validation-only mass-dispersion diagnosis from two sealed synthetic ledgers.

Positive membership is a stored GT-derived label used only for explanation.
It is never a deployable selector or a counterfactual pose-performance result.
No model, pose solver, reference image or real evaluation is imported/read.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import math
from pathlib import Path
import resource
import statistics
import time

from .common import read, rows, sha, write

ARMS=tuple(kind+'_s'+str(seed) for kind in ('L_LINEAR','L_HEAD') for seed in (20261018,20261019,20261020))


def summarize(values):
    x=sorted(float(v) for v in values if v is not None)
    if not x:return dict(n=0,mean=None,sample_std=None,median=None,P10=None,P90=None,min=None,max=None)
    if not all(math.isfinite(v) for v in x):raise ValueError('Nonfinite diagnostic value; do not drop a frame')
    def q(p):
        z=(len(x)-1)*p;i=int(z)
        return x[i]+(z-i)*(x[min(i+1,len(x)-1)]-x[i])
    return dict(n=len(x),mean=statistics.fmean(x),sample_std=statistics.stdev(x) if len(x)>1 else None,
                median=statistics.median(x),P10=q(.1),P90=q(.9),min=x[0],max=x[-1])


def masses(logp,good_indices,none_index):
    probability=[math.exp(x) for x in logp];k=len(good_indices);pnone=probability[none_index]
    if not k:return dict(positive_count=0,P_none=pnone,P_max_good=None,P_sum_good=0.,
                         log_sum_good_minus_log_none=None,sum_good_gt_none=False,uniform_positive_CE=None,
                         marginal_good_NLL=None,log_positive_count=None,conditional_uniform_KL=None,decomposition_error=None)
    logs=[logp[i] for i in good_indices];maximum=max(logs)
    logsum=maximum+math.log(math.fsum(math.exp(x-maximum) for x in logs))
    ce=-statistics.fmean(logs);logk=math.log(k)
    kl=statistics.fmean(-logk-(x-logsum) for x in logs)
    return dict(positive_count=k,P_none=pnone,P_max_good=math.exp(maximum),P_sum_good=math.exp(logsum),
                log_sum_good_minus_log_none=logsum-logp[none_index],sum_good_gt_none=logsum>logp[none_index],
                uniform_positive_CE=ce,marginal_good_NLL=-logsum,log_positive_count=logk,conditional_uniform_KL=kl,
                decomposition_error=abs(ce-(-logsum+logk+kl)))


def self_check():
    # Hand-constructed counterexamples: atomic none can win while the union of
    # two positive atoms has greater mass; with one positive it cannot do so.
    a=masses([math.log(.31),math.log(.30),math.log(.39)],[0,1],2)
    b=masses([math.log(.20),math.log(.10),math.log(.70)],[0,1],2)
    c=masses([math.log(.30),math.log(.30),math.log(.40)],[0],2)
    assert a['P_max_good']<a['P_none']<a['P_sum_good'] and a['sum_good_gt_none']
    assert b['P_sum_good']<b['P_none'] and not b['sum_good_gt_none']
    assert not c['sum_good_gt_none']
    assert max(a['decomposition_error'],b['decomposition_error'],c['decomposition_error'])<1e-12
    return dict(passed=True,hand_constructed_probability_cases=3,no_model_inference=True)


def category_summary(subset):
    fields=('P_none','P_max_good','P_sum_good','log_sum_good_minus_log_none','uniform_positive_CE',
            'marginal_good_NLL','log_positive_count','conditional_uniform_KL')
    return dict(n=len(subset),positive_count_distribution=dict(sorted(Counter(r['positive_count'] for r in subset).items())),
                sum_good_gt_none_n=sum(r['sum_good_gt_none'] for r in subset),
                summaries={key:summarize(r[key] for r in subset) for key in fields})


def render(result):
    fmt=lambda x:'N/A' if x is None else f'{x:.4f}'
    lines=['# 합성 validation의 성공 후보 질량과 보류 진단','',
           '저장된 분포와 학습 라벨만 읽었다. 모델·학습·예측·선택 규칙은 바꾸지 않았다. '
           '아래의 “성공 후보”는 정답을 이용한 사후 집합이므로 런타임에서 알 수 없으며 새 선택기 성능을 뜻하지 않는다.','',
           '| arm | validation | good 존재 | good 존재·보류 | 그중 good합 > none | 그중 good합 ≤ none | good 부재·보류 |','|---|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        a=result['arms'][arm];c=a['categories']
        lines.append(f"| {arm} | {a['denominator']} | {c['good_available']['n']} | {c['good_available_abstained']['n']} | {c['good_available_abstained_sum_gt_none']['n']} | {c['good_available_abstained_sum_le_none']['n']} | {c['no_good_abstained']['n']} |")
    lines += ['','## 좋은 후보가 있지만 보류한 군의 확률','',
              '| arm | none 평균±SD / 중앙값 | 최대 good 평균±SD / 중앙값 | good합 평균±SD / 중앙값 |','|---|---:|---:|---:|']
    for arm in ARMS:
        p=result['arms'][arm]['categories']['good_available_abstained']['summaries']
        cell=lambda k:f"{fmt(p[k]['mean'])}±{fmt(p[k]['sample_std'])} / {fmt(p[k]['median'])}"
        lines.append('| '+' | '.join([arm,cell('P_none'),cell('P_max_good'),cell('P_sum_good')])+' |')
    lines += ['','## 성공 후보 수 k의 분포','',
              '각 셀은 `good 존재·보류 수 / 그중 good합>none 수`다. 미기록을0으로 바꾼 것이 아니라 각 보존 행에서 직접 센 값이다.','',
              '| k | '+' | '.join(ARMS)+' |','|---|'+'---:|'*len(ARMS)]
    counts=sorted({int(k) for arm in ARMS for k in result['arms'][arm]['good_abstention_by_positive_count']})
    for k in counts:
        cells=[]
        for arm in ARMS:
            v=result['arms'][arm]['good_abstention_by_positive_count'].get(str(k),{})
            cells.append(f"{v.get('n',0)}/{v.get('sum_good_gt_none_n',0)}")
        lines.append('| '+' | '.join([str(k)]+cells)+' |')
    a=result['arms']['L_HEAD_s20261018']['categories'];n=a['good_available_abstained']['n'];yes=a['good_available_abstained_sum_gt_none']['n'];no=a['good_available_abstained_sum_le_none']['n']
    lines += ['','## 근거와 반례','',
              f"사전 대표 head의 good 존재·보류 {n}장 중 {yes}장은 성공 후보 합의 질량이 none보다 크다. "
              '각 개별 성공 후보보다 none이 큰 atomic argmax와, 여러 성공 후보를 합친 질량이 동시에 성립하는 사례다. '
              '여러 올바른 후보에 질량이 나뉜다는 설명과 관측상 일치하지만 손실의 인과 효과를 분리한 실험은 아니다.','',
              f"나머지 {no}장은 성공 후보 질량을 합쳐도 none 이하인 반례다. "
              '이 군의 보류를 성공 후보 사이의 분산만으로 설명할 수 없다. k별 분포와 선형/다른 seed도 함께 남겨 선택적으로 해석하지 않는다.','',
              '`CE_uniform = −log(P_good_sum) + log(k) + KL(U_k || p_good/P_good_sum)`를 수치 검산했다. '
              '균등 positive CE는 좋은 집합에 질량을 보내면서 그 내부 분포도 균등하게 만드는 목표다. '
              '`log(k)`는 해당 frame에서 상수이므로 그 항 자체가 none을 밀어 올린다고 해석하면 안 된다. '
              'none label의 빈도·후보 지원 범위·특징의 구별력 효과는 이 저장 분포 분석만으로 분리하지 못한다.','',
              'GT-derived good합은 배포 시 계산할 수 없다. 이를 런타임 규칙으로 제안하거나 이 집계로 회복 수·새 선택기 성공률을 만들지 않았다. '
              '원 validation235장 전체와 모든6seed 행은 `POSITIVE_MASS_ROWS.csv`에 보존한다.']
    return '\n'.join(lines)+'\n'


def run(run_dir,output=None,*,synthetic_GO=False):
    if synthetic_GO is not True:raise PermissionError('Root synthetic-only postseal diagnostic GO required before IO')
    start,cpu=time.monotonic(),time.process_time();run=Path(run_dir);dist=run/'head_synthetic_distributions_v1';labels=run/'head_labels_v1'
    ds=read(dist/'DISTRIBUTION_SEAL.json');ls=read(labels/'LABEL_SEAL.json');dp=dist/'DISTRIBUTIONS.jsonl.gz';lp=labels/'SYNTHETIC_HEAD_TARGETS.jsonl.gz'
    if (ds.get('complete') is not True or ds.get('before_reference_decode') is not True or ds.get('reference_reads')!=0
            or ds.get('rows')!=1003 or ls.get('rows')!=1003 or set(ds['arms'])!=set(ARMS)
            or ds['features_sha256']!=ls['features_sha256'] or sha(dp)!=ds['distributions_sha256'] or sha(lp)!=ls['labels_sha256']):
        raise ValueError('Matching complete sealed synthetic distributions and labels required')
    checks=self_check();all_distributions=list(rows(dp));all_labels={r['id']:r for r in rows(lp)}
    if len(all_distributions)!=1003 or len(all_labels)!=1003 or {r['id'] for r in all_distributions}!=set(all_labels):raise ValueError('Exact original1003 identity set required')
    if [r['id'] for r in all_distributions]!=ds['ordered_ids']:raise ValueError('Original distribution order changed')
    validation=[r for r in all_distributions if r['split']=='validation']
    if len(validation)!=235:raise ValueError('Preserve all235 original validation frames')
    result_rows=[];maximum_partition_error=maximum_decomposition_error=0.;single_good_mass_contradictions=0
    for r in validation:
        label=all_labels[r['id']]
        if label['split']!='validation' or set(r['predictions'])!=set(ARMS):raise ValueError('Validation split/arm inventory changed')
        metrics=label['candidate_metrics'];pos=label['positive_candidate_indices'];goodids={metrics[i]['candidate_id'] for i in pos};k=len(goodids)
        if len(pos)!=k or label['none_target'] is not (k==0):raise ValueError('Stored positive labels inconsistent')
        target=label['target_distribution'];positive=set(pos)
        expected=[1/k if i in positive else 0. for i in range(len(metrics))]+[1. if not k else 0.]
        if target!=expected:raise ValueError('Original uniform-positive targets changed')
        eligible=[m['candidate_id'] for m in metrics if m['eligible']]
        for arm in ARMS:
            d=r['predictions'][arm];ids=d['candidate_ids'];logp=d['log_probabilities'];ni=d['none_index'];chosen=d['selected_index']
            if ids!=eligible or ni!=len(ids) or len(logp)!=len(ids)+1 or not all(math.isfinite(x) for x in logp):raise ValueError('Frozen eligible softmax domain changed')
            if d['none_selected'] is not (chosen==ni) or logp[chosen]<max(logp)-1e-6:raise ValueError('Stored atomic selection inconsistent')
            good=[i for i,cid in enumerate(ids) if cid in goodids]
            if len(good)!=k:raise ValueError('Positive candidate absent from original eligible distribution')
            v=masses(logp,good,ni);error=abs(math.fsum(math.exp(x) for x in logp)-1.)
            maximum_partition_error=max(maximum_partition_error,error)
            if error>1e-5:raise ValueError('Stored probabilities fail normalization; preserve source')
            maximum_decomposition_error=max(maximum_decomposition_error,v['decomposition_error'] or 0.)
            single_good_mass_contradictions+=int(k==1 and d['none_selected'] and v['sum_good_gt_none'])
            result_rows.append(dict(id=r['id'],arm=arm,representative_seed=arm.endswith('20261018'),split='validation',
                                    none_selected=d['none_selected'],good_available=bool(k),**v))
    if single_good_mass_contradictions:raise ValueError('One-positive argmax/mass contradiction; inspect numeric archive')
    arms={}
    for arm in ARMS:
        selected=[r for r in result_rows if r['arm']==arm];good=[r for r in selected if r['good_available']];ab=[r for r in good if r['none_selected']];ng=[r for r in selected if not r['good_available']]
        categories=dict(all_validation=selected,good_available=good,good_available_abstained=ab,
                        good_available_abstained_sum_gt_none=[r for r in ab if r['sum_good_gt_none']],
                        good_available_abstained_sum_le_none=[r for r in ab if not r['sum_good_gt_none']],
                        good_available_not_abstained=[r for r in good if not r['none_selected']],
                        no_good=ng,no_good_abstained=[r for r in ng if r['none_selected']],no_good_not_abstained=[r for r in ng if not r['none_selected']])
        arms[arm]=dict(denominator=len(selected),abstentions=sum(r['none_selected'] for r in selected),
                       categories={name:category_summary(value) for name,value in categories.items()},
                       good_abstention_by_positive_count={str(k):category_summary([r for r in ab if r['positive_count']==k]) for k in sorted({r['positive_count'] for r in ab})})
    value=dict(schema='SYNTHETIC_VALIDATION_POSITIVE_MASS_DIAGNOSIS_V1',complete=True,validation_frames=235,rows=len(result_rows),arms=arms,
               representative_seed=20261018,original_reference_symmetry_counts=dict(Counter(str(all_labels[r['id']]['reference_symmetry_order']) for r in validation)),
               probability_comparison='strict stored-log-probability logsumexp(good)>logp(none); no fitted threshold',
               no_counterfactual_selector_accuracy=True,no_runtime_GT_aggregate_rule_proposed=True,no_method_training_prediction_or_setting_changes=True,
               no_PRIMARY_or_real_results_read=True,new_NN_training_or_inference_calls=0,new_PnP_or_optimization_calls=0,
               checks={**checks, 'maximum_probability_partition_error':maximum_partition_error,'maximum_CE_decomposition_error':maximum_decomposition_error,
                       'single_positive_mass_contradictions':single_good_mass_contradictions},
               input_bindings=dict(distributions=sha(dp),distribution_seal=sha(dist/'DISTRIBUTION_SEAL.json'),labels=sha(lp),label_seal=sha(labels/'LABEL_SEAL.json'),features_sha256=ds['features_sha256']),
               source_sha256=sha(__file__),cost=dict(wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time()-cpu,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    out=Path(output or run/'positive_mass_v1');out.mkdir(parents=True,exist_ok=False)
    with (out/'POSITIVE_MASS_ROWS.csv').open('x',newline='')as f:
        writer=csv.DictWriter(f,fieldnames=list(result_rows[0]));writer.writeheader();writer.writerows(result_rows)
    write(out/'POSITIVE_MASS_DIAGNOSIS.json',value);(out/'POSITIVE_MASS_DIAGNOSIS_KO.md').write_text(render(value))
    write(out/'POSITIVE_MASS_SEAL.json',dict(schema='POSITIVE_MASS_DIAGNOSIS_SEAL_V1',complete=True,synthetic_only=True,
          input_bindings=value['input_bindings'],source_sha256=sha(__file__),artifacts={p.name:sha(p) for p in out.iterdir() if p.is_file()},
          cost=dict(wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time()-cpu,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)))
    return value


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--output');p.add_argument('--synthetic-GO',action='store_true');a=p.parse_args()
    r=run(a.run_dir,a.output,synthetic_GO=a.synthetic_GO)
    for arm in ARMS:
        c=r['arms'][arm]['categories'];print(arm,dict(good_abstain=c['good_available_abstained']['n'],good_sum_gt_none=c['good_available_abstained_sum_gt_none']['n'],counterexamples=c['good_available_abstained_sum_le_none']['n']))


if __name__=='__main__':main()
