"""Descriptive mechanism diagnosis after complete prediction/evaluation seals.

Consumes existing success/stage records; never imports a solver, metric,
reference, model or training module. Old strata are direct aliases of the
original sealed E0 diagnosis, not newly inferred physical failure labels.
"""
from __future__ import annotations

import argparse
from collections import Counter
import math
from pathlib import Path
import resource
import statistics
import time

from .common import read, rows, sha, write

GROUPS=('FINAL_SUCCESS_CONTROL','FAMILY_SELECTION_LOSS','WITHIN_FAMILY_SELECTION_LOSS','NO_SUCCESS_CANDIDATE')
PRIMARY_COUNTS=dict(zip(GROUPS,(114,10,22,46)))
OLD_ALIAS={'family_selection':GROUPS[1],'within_family_final_selection':GROUPS[2],
           'confirmed_original_consumer_observation_or_proposal_limit':GROUPS[3]}
POPULATIONS={'PRIMARY192':('TEST','REMAINDER'),'TEST49':('TEST',),'REMAINDER143':('REMAINDER',),'CAL46':('CAL',)}
EXPECTED_DENOMINATORS=dict(PRIMARY192=192,TEST49=49,REMAINDER143=143,CAL46=46)
COMPARATORS=('E0','N0','POSEFIX_SUBPIX_PF_ANCHOR','POSEFIX_SUBPIX_YOLO_ANCHOR')
SEEDS=(20261018,20261019,20261020)
MECHANISM_CONTRASTS=(('CTRL_CARRY','CTRL_BUDGET'),('A_FIXED_BANK','A_FULL'),
                     ('B_FIXED_BANK','B_FULL'),('CTRL_BUDGET','AB_FULL'),('CTRL_EDGE_BOUNDARY','B_FULL'))
BOTTLENECK_ARM='AB_FULL'  # Fixed before any new outcome read; never PRIMARY-best arm.


def stats(values):
    x=sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    if not x:return dict(n=0,mean=None,sample_std=None,sample_variance=None,median=None,P90=None,P95=None,max=None)
    def quantile(p):
        z=(len(x)-1)*p;i=int(z)
        return x[i]+(z-i)*(x[min(i+1,len(x)-1)]-x[i])
    variance=statistics.variance(x) if len(x)>1 else None
    return dict(n=len(x),mean=statistics.fmean(x),sample_variance=variance,
                sample_std=math.sqrt(variance) if variance is not None else None,
                median=statistics.median(x),P90=quantile(.9),P95=quantile(.95),max=x[-1])


def original_group(row):
    if row['final_joint_success'] is True:
        if row['earliest_observed_failure_stage'] is not None:raise ValueError('Original success has a failure stage')
        return GROUPS[0]
    if row['final_joint_success'] is not False:raise ValueError('Unresolved original success must not become a new label')
    if row.get('original_consumer_generated_bank_complete') is not True:raise ValueError('Original bank coverage unresolved')
    stage=row['earliest_observed_failure_stage']
    if stage not in OLD_ALIAS:raise ValueError('Unmapped original stage; preserve the gap: '+str(stage))
    return OLD_ALIAS[stage]


def metric_summary(subset,arm):
    ms=[r['metrics'][arm] for r in subset]
    return dict(denominator=len(subset),successes=sum(m['success'] for m in ms),
                success_rate=sum(m['success'] for m in ms)/len(ms) if ms else None,
                no_pose=sum(not m['pose_available'] for m in ms),metric_errors=sum(m.get('metric_error') is not None for m in ms),
                translation_cm=stats(m['translation_cm'] for m in ms),rotation_deg=stats(m['rotation_deg'] for m in ms),
                T_gt50cm=sum(m['translation_cm'] is not None and m['translation_cm']>50 for m in ms),
                T_gt100cm=sum(m['translation_cm'] is not None and m['translation_cm']>100 for m in ms),
                R_gt45deg=sum(m['rotation_deg'] is not None and m['rotation_deg']>45 for m in ms))


def paired_summary(subset,before,after):
    counts=Counter();ids={k:[] for k in ('BOTH_SUCCESS','RECOVERY','DAMAGE','BOTH_FAILURE')}
    for r in subset:
        a,b=r['metrics'][before]['success'],r['metrics'][after]['success']
        name='BOTH_SUCCESS' if a and b else 'RECOVERY' if b else 'DAMAGE' if a else 'BOTH_FAILURE'
        counts[name]+=1;ids[name].append(r['id'])
    delta={}
    for axis in ('translation_cm','rotation_deg'):
        common=[r for r in subset if r['metrics'][before][axis] is not None and r['metrics'][after][axis] is not None]
        values=[r['metrics'][after][axis]-r['metrics'][before][axis] for r in common]
        failed=[r for r in common if not r['metrics'][before]['success'] and not r['metrics'][after]['success']]
        ds=[r['metrics'][after][axis]-r['metrics'][before][axis] for r in failed]
        delta[axis]=dict(common_available_n=len(common),delta=stats(values),both_failure_common_available_n=len(failed),
                         both_failure_delta=stats(ds),both_failure_improved_n=sum(x<0 for x in ds),both_failure_worsened_n=sum(x>0 for x in ds))
    return dict(denominator=len(subset),counts={k:counts[k] for k in ids},
                success_difference=counts['RECOVERY']-counts['DAMAGE'],
                recovery_ids=ids['RECOVERY'],damage_ids=ids['DAMAGE'],continuous=delta)


def stage_summary(subset,arm,stage_by_id):
    selected=[stage_by_id[r['id']] for r in subset];values=[r.get('arms',{}).get(arm,{}) for r in selected]
    result=dict(denominator=len(subset),status_counts=dict(Counter(v.get('status','UNREPORTED') for v in values)),
                stage_loss_class_counts=dict(Counter(v.get('stage_loss_class','UNREPORTED') for v in values)),
                family_intermediate_status_counts=dict(Counter(v.get('family_intermediate_status','APPLICABLE' if 'any_family_winner_success' in v else 'UNREPORTED') for v in values)))
    def known_boolean(items,key):
        known=[v[key] for v in items if type(v.get(key)) is bool]
        return dict(known_n=len(known),unknown_n=len(items)-len(known),success_n=sum(known) if known else None)
    result['carried_oracle']=known_boolean([r.get('carried',{}) for r in selected],'has_success')
    for source,name in (('generated_before_feasibility','generated_before_constraints_oracle'),
                        ('after_constraints','after_constraints_oracle'),('retained_bank','retained_bank_oracle')):
        result[name]=known_boolean([v.get(source,{}) for v in values],'has_success')
    result['family_winner_oracle']=known_boolean(values,'any_family_winner_success')
    result['new_success_beyond_carried_explicit']=known_boolean(values,'success_created_beyond_carried')
    result['final']=known_boolean(values,'final_success')
    for label in ('optimizer_reported_failed_attempts','exception_attempts','infeasible_attempts'):
        known=[v[label] for v in values if isinstance(v.get(label),int)]
        result[label]=dict(known_frames=len(known),unknown_frames=len(values)-len(known),attempt_count=sum(known) if known else None)
    for r,v in zip(subset,values,strict=True):
        if 'final_success' in v and v['final_success'] is not r['metrics'][arm]['success']:
            raise ValueError('Postseal stage/final metric disagreement')
    return result


def training_summary(run,synthetic_evaluation_dir=None):
    folder=run/'head_training_v1';manifest_path=folder/'TRAINING_MANIFEST.json';notrun=folder/'TRAINING_NOT_RUN.json'
    if not manifest_path.exists():
        if not notrun.exists():return dict(status='ARCHIVE_GAP',reason='No actual training or nonexecution receipt')
        value=read(notrun)
        if value.get('training_runs')!=0:raise ValueError('Invalid training nonexecution receipt')
        return dict(status='NOT_APPLICABLE_LABEL_SUPPORT_GATE',training_runs=0,receipt_sha256=sha(notrun),
                    label_support_gate=value.get('label_support_gate'),no_PRIMARY_seed_selection=True)
    if notrun.exists():raise ValueError('Training and nonexecution receipts conflict')
    manifest=read(manifest_path)
    if (manifest.get('complete') is not True or manifest.get('total_training_runs')!=6
            or manifest.get('representative_seed')!=SEEDS[0]):raise ValueError('Actual fixed six-run training incomplete')
    expected={(kind,s) for kind in ('L_LINEAR','L_HEAD') for s in SEEDS}
    runs={(r['kind'],r['seed']):r for r in manifest['runs']}
    if set(runs)!=expected or len(manifest['runs'])!=6:raise ValueError('Frozen model/seed inventory changed')
    info=[]
    for (kind,seed),r in runs.items():
        history_path=Path(r['checkpoint']).parent/'HISTORY.json';h=read(history_path)
        if len(h)!=r['epochs'] or min(x['validation_loss'] for x in h)!=r['best_validation_loss']:
            raise ValueError('Actual history/manifest mismatch')
        info.append(dict(kind=kind,seed=seed,representative=seed==SEEDS[0],epochs=r['epochs'],best_epoch=r['best_epoch'],
                         best_validation_CE=r['best_validation_loss'],optimizer_steps=r['optimizer_steps'],
                         parameters=r['parameters'],checkpoint_sha256=r['checkpoint_sha256'],history_sha256=sha(history_path),cost=r['cost']))
    comparisons=[dict(seed=s,representative=s==SEEDS[0],linear_validation_CE=runs[('L_LINEAR',s)]['best_validation_loss'],
                      head_validation_CE=runs[('L_HEAD',s)]['best_validation_loss'],
                      head_minus_linear_CE=runs[('L_HEAD',s)]['best_validation_loss']-runs[('L_LINEAR',s)]['best_validation_loss']) for s in SEEDS]
    result=dict(status='EXECUTED',training_runs=6,manifest_sha256=sha(manifest_path),representative_seed=SEEDS[0],runs=info,
                paired_seed_comparisons=comparisons,
                all_three_seed_best_validation_CE={k:stats(r['best_validation_loss'] for (kind,_),r in runs.items() if kind==k) for k in ('L_LINEAR','L_HEAD')},
                no_PRIMARY_seed_selection=True,validation_used_for_epoch_selection=True,
                loss_comparison_is_not_independent_test_accuracy=True)
    evaluation=Path(synthetic_evaluation_dir or run/'synthetic_head_evaluation_v1')
    if (evaluation/'EVALUATION_SEAL.json').exists():
        seal=read(evaluation/'EVALUATION_SEAL.json');path=evaluation/'SYNTHETIC_LEARNED_EVALUATION.json'
        if (seal.get('complete') is not True or sha(path)!=seal['artifacts'][path.name]
                or seal['input_bindings']['training_manifest']!=sha(manifest_path)):
            raise ValueError('Synthetic head evaluation does not bind these actual checkpoints')
        value=read(path)
        result['synthetic_postprediction_joint_and_abstention']=value['populations']
        result['synthetic_evaluation_seal_sha256']=sha(evaluation/'EVALUATION_SEAL.json')
    else:result['synthetic_postprediction_joint_and_abstention_status']='NOT_AVAILABLE; validation CE only, no invented pose accuracy'
    return result


def verify_inputs(run,evaluation,oracle_root,independent_receipt=None):
    ps=read(run/'PREDICTION_SEAL.json');es=read(evaluation/'EVALUATION_SEAL.json')
    if (ps.get('complete') is not True or ps.get('before_scoring') is not True or ps.get('reference_inputs') is not False
            or ps['plan_lock_sha256']!=sha(run/'PLAN_LOCK.json') or es.get('explicit_root_scoring_GO') is not True
            or es['input_bindings']['new_prediction_seal']!=sha(run/'PREDICTION_SEAL.json')
            or es['input_bindings']['plan_lock']!=sha(run/'PLAN_LOCK.json')
            or es['input_bindings']['new_predictions']!=ps['predictions_sha256']):raise ValueError('Complete prediction→evaluation seal chain required')
    consumed=('SCORED_ROWS.jsonl.gz','CANDIDATE_STAGE_DIAGNOSIS.json','RESULTS.json')
    for name in consumed:
        if sha(evaluation/name)!=es['artifacts'][name]:raise ValueError('Evaluated artifact changed: '+name)
    oldseal=read(oracle_root/'ORACLE_SEAL.json');oldpath=oracle_root/'ORACLE_ROWS.jsonl.gz'
    expected=oldseal.get('artifacts',oldseal.get('files',{})).get(oldpath.name)
    if isinstance(expected,dict):expected=expected.get('sha256')
    if expected is None or sha(oldpath)!=expected:raise ValueError('Original E0 sealed oracle changed')
    if independent_receipt is not None:
        independent=read(independent_receipt)
        if independent.get('status')!='PASS' or independent.get('failure_count')!=0 or sha(evaluation/'EVALUATION_SEAL.json') not in independent.get('input_bindings',{}).values():
            raise ValueError('Independent receipt does not bind this evaluation')
    return dict(prediction_seal=sha(run/'PREDICTION_SEAL.json'),evaluation_seal=sha(evaluation/'EVALUATION_SEAL.json'),
                plan_lock=sha(run/'PLAN_LOCK.json'),original_oracle=sha(oldpath),original_oracle_seal=sha(oracle_root/'ORACLE_SEAL.json'),
                **{name:sha(evaluation/name) for name in consumed},
                independent_receipt=sha(independent_receipt) if independent_receipt is not None else None,
                prediction_journal_rehashed=False,scope='Only consumed evaluated artifacts rehashed; original prediction journal is already bound by the evaluated seal chain')


def bottleneck(summary,stage,arms):
    if BOTTLENECK_ARM not in arms:return dict(status='UNAVAILABLE',predeclared_arm=BOTTLENECK_ARM,reason='No actual AB_FULL output')
    classes=stage['arms'][BOTTLENECK_ARM]['stage_loss_class_counts']
    failures={k:v for k,v in classes.items() if k!='FINAL_SUCCESS' and k!='UNREPORTED'}
    if not failures:return dict(status='NO_IDENTIFIABLE_FAILED_STAGE',predeclared_arm=BOTTLENECK_ARM,counts=classes)
    name,n=min(failures.items(),key=lambda p:(-p[1],p[0]))
    return dict(status='DESCRIPTIVE_FIXED_REFERENCE_STAGE',predeclared_arm=BOTTLENECK_ARM,stage=name,count=n,
                full_denominator=summary['denominator'],final_successes=summary['arms'][BOTTLENECK_ARM]['successes'],
                original_E0_successes=summary['arms']['E0']['successes'],
                paired_vs_E0=summary['paired'][BOTTLENECK_ARM+'_minus_E0'],all_failure_class_counts=classes,
                caveat='Largest recorded failure-stage count for the prespecified AB_FULL only. Finite-bank oracle, not physical owner truth or a proof of global impossibility; no PRIMARY-best method selected.')


def render(value):
    def fmt(x):return 'N/A' if x is None else f'{x:.2f}'
    lines=['# 원 E0 실패 층별 후봉인 기전 진단','',
           '원 봉인 진단의114성공/10family 선택/22같은-family 선택/46원 bank 후보부재를 그대로 연결했다. '
           '새 실패 라벨이나 새 선택기를 만들지 않았다. 후보 oracle은 같은 평가 참조를 사후 적용한 유한 bank 상한이다.','']
    b=value['predeclared_AB_FULL_bottleneck']
    if b['status']=='DESCRIPTIVE_FIXED_REFERENCE_STAGE':
        p=b['paired_vs_E0']['counts'];lines += [f"사전 지정 AB_FULL에서 가장 많은 기록상 실패 단계는 `{b['stage']}`: {b['count']}/{b['full_denominator']}장이다. "
            f"최종 성공은{b['final_successes']}장, 원 E0는{b['original_E0_successes']}장이며 E0 대비 회복{p['RECOVERY']}·손상{p['DAMAGE']}장이다.",'',
            '이 진단은 실제 표본의 최빈 손실 위치를 설명하며 물리 owner 정답·독립 일반화·전역 해의 부재를 인증하지 않는다.','']
    for pop in ('PRIMARY192','CAL46'):
        lines += [f'## {pop}: 원 층별 성공과 오차','',
                  '| 원 층 | 방법 | 공동성공/전체 | E0 대비 회복/손상 | 위치cm 평균±SD | 회전° 평균±SD |','|---|---|---:|---:|---:|---:|']
        for group,g in value['populations'][pop]['strata'].items():
            for arm,a in g['arms'].items():
                if a.get('execution_status')=='NOT_EXECUTED':
                    lines.append(f"| {group} | {arm} | N/A/{g['denominator']} | N/A | N/A | N/A |");continue
                p=g['paired'].get(arm+'_minus_E0');pair=f"{p['counts']['RECOVERY']}/{p['counts']['DAMAGE']}" if p else '—'
                t,r=a['translation_cm'],a['rotation_deg']
                lines.append(f"| {group} | {arm} | {a['successes']}/{a['denominator']} | {pair} | {fmt(t['mean'])}±{fmt(t['sample_std'])} | {fmt(r['mean'])}±{fmt(r['sample_std'])} |")
        lines += ['','### 후보 단계: 성공 후보가 남은 영상 수','',
                  '| 원 층 | 새 arm | carried | 생성 전제약 | 제약 후 | family winner | 최종 |','|---|---|---:|---:|---:|---:|---:|']
        for group,g in value['populations'][pop]['strata'].items():
            for arm,s in g['candidate_stages']['arms'].items():
                fields=('carried_oracle','generated_before_constraints_oracle','after_constraints_oracle','family_winner_oracle','final')
                cells=[]
                for f in fields:
                    x=s.get(f,{})
                    cells.append('N/A' if x.get('success_n') is None else f"{x['success_n']}/{x['known_n']}")
                lines.append('| '+' | '.join([group,arm]+cells)+' |')
        lines += ['','분수의 분모는 해당 단계가 실제 보존된 영상 수다. 원 전체 분모와 archive gap은 JSON에 따로 남겼으며, 미기록을 실패0으로 바꾸지 않았다.','']
    h=value['learned_head_validation'];lines += ['## 실제 학습과 validation','']
    if h.get('status')=='EXECUTED':
        lines += ['대표 seed는20261018로 사전 고정했다. 세 seed 모두의 실제 best-validation CE를 선형과 나란히 보존하며 PRIMARY 최고 seed를 선택하지 않는다.','',
                  '| seed | 대표 | 선형 validation CE | head validation CE | head−선형 |','|---|---|---:|---:|---:|']
        for r in h['paired_seed_comparisons']:
            lines.append(f"| {r['seed']} | {r['representative']} | {r['linear_validation_CE']:.6f} | {r['head_validation_CE']:.6f} | {r['head_minus_linear_CE']:+.6f} |")
        lines += ['','validation은 epoch 선택에 사용한 개발자료다. 별도 합성 후봉인 평가가 있으면 전체 train/validation 공동성공·보류·후보부재 CE를 JSON에 함께 연결한다.']
    else:lines += [f"학습 상태: `{h['status']}`. 실행하지 않은 head의 수치 성과를 만들지 않는다."]
    return '\n'.join(lines)+'\n'


def run(run_dir,*,original_oracle_root,evaluation_dir=None,output=None,independent_receipt=None,
        synthetic_evaluation_dir=None,diagnosis_GO=False):
    if diagnosis_GO is not True:raise PermissionError('Explicit root final-seal mechanism GO required before any IO')
    start,cpu=time.monotonic(),time.process_time();run=Path(run_dir);evaluation=Path(evaluation_dir or run/'evaluation_v1');old=Path(original_oracle_root)
    bindings=verify_inputs(run,evaluation,old,independent_receipt)
    scored=list(rows(evaluation/'SCORED_ROWS.jsonl.gz'));stage=read(evaluation/'CANDIDATE_STAGE_DIAGNOSIS.json');results=read(evaluation/'RESULTS.json')
    oracle=list(rows(old/'ORACLE_ROWS.jsonl.gz'));ids=[r['id'] for r in scored];old_by_id={r['id']:r for r in oracle};stage_by_id={r['id']:r for r in stage['rows']}
    if len(ids)!=238 or len(set(ids))!=238 or set(ids)!=set(old_by_id) or set(ids)!=set(stage_by_id):raise ValueError('Exact original238 identity union required')
    groups={};memberships=[]
    for r in scored:
        source=old_by_id[r['id']]
        if any(r[k]!=source[k] for k in ('partition','session')) or r['metrics']['E0']['success'] is not source['final_joint_success']:
            raise ValueError('Original E0 identity/success changed')
        group=original_group(source);groups[r['id']]=group
        memberships.append(dict(id=r['id'],partition=r['partition'],session=r['session'],cluster_id=r['cluster_id'],
                                original_stratum=group,original_earliest_observed_failure_stage=source['earliest_observed_failure_stage']))
    if Counter(groups[r['id']] for r in scored if r['partition']!='CAL')!=Counter(PRIMARY_COUNTS):raise ValueError('Original114/10/22/46 strata changed')
    arms=tuple(results['arms']);new=tuple(results['new_arms']);unexecuted=results.get('unexecuted_arms',{})
    if any(set(r['metrics'])!=set(arms) for r in scored):raise ValueError('Method cannot disappear on failures')
    populations={}
    def summarize(subset):
        metrics={a:metric_summary(subset,a) for a in arms}
        for a,why in unexecuted.items():metrics[a]=dict(execution_status='NOT_EXECUTED',denominator=len(subset),reason=why)
        pairs={a+'_minus_'+b:paired_summary(subset,b,a) for a in new for b in COMPARATORS}
        contrasts={a+'_minus_'+b:paired_summary(subset,b,a) for b,a in MECHANISM_CONTRASTS if a in arms and b in arms}
        stages={a:stage_summary(subset,a,stage_by_id) for a in new}
        for a,why in unexecuted.items():stages[a]=dict(execution_status='NOT_EXECUTED',denominator=len(subset),reason=why)
        return dict(denominator=len(subset),arms=metrics,paired=pairs,predeclared_mechanism_contrasts=contrasts,
                    candidate_stages=dict(denominator=len(subset),arms=stages))
    for name,parts in POPULATIONS.items():
        subset=[r for r in scored if r['partition'] in parts];whole=summarize(subset)
        if len(subset)!=EXPECTED_DENOMINATORS[name]:raise ValueError('Original full population denominator changed')
        whole['strata']={g:summarize([r for r in subset if groups[r['id']]==g]) for g in GROUPS}
        populations[name]=whole
    value=dict(schema='POSTSEAL_ORIGINAL_E0_STRATIFIED_MECHANISM_DIAGNOSIS_V1',complete=True,
               original_stratum_source_field='final_joint_success and earliest_observed_failure_stage; original complete consumer bank scope',
               original_aliases=OLD_ALIAS,expected_primary_strata=PRIMARY_COUNTS,original_membership=memberships,
               arms=list(arms),unexecuted_arms=unexecuted,populations=populations,
               predeclared_AB_FULL_bottleneck=bottleneck(populations['PRIMARY192'],populations['PRIMARY192']['candidate_stages'],arms),
               learned_head_validation=training_summary(run,synthetic_evaluation_dir),
               no_new_metric_or_solver_or_model_calls=True,no_primary_seed_or_method_selection=True,
               candidate_oracles_are_postseal_upper_bounds=True,not_physical_owner_ground_truth=True,
               all_denominators_failures_and_archive_gaps_preserved=True,input_bindings=bindings,source_sha256=sha(__file__),
               cost=dict(wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time()-cpu,
                         peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    out=Path(output or run/'mechanism_v1');out.mkdir(parents=True,exist_ok=False)
    write(out/'MECHANISM_DIAGNOSIS.json',value);(out/'MECHANISM_DIAGNOSIS_KO.md').write_text(render(value))
    write(out/'MECHANISM_SEAL.json',dict(schema='POSTSEAL_MECHANISM_DIAGNOSIS_SEAL_V1',complete=True,explicit_root_GO=True,
          input_bindings=bindings,source_sha256=sha(__file__),diagnosis_sha256=sha(out/'MECHANISM_DIAGNOSIS.json'),
          markdown_sha256=sha(out/'MECHANISM_DIAGNOSIS_KO.md'),artifacts={p.name:sha(p) for p in out.iterdir() if p.is_file()},cost=value['cost']))
    return value


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--original-oracle-root',required=True)
    p.add_argument('--evaluation-dir');p.add_argument('--output');p.add_argument('--independent-receipt')
    p.add_argument('--synthetic-evaluation-dir');p.add_argument('--diagnosis-GO',action='store_true');a=p.parse_args()
    r=run(a.run_dir,original_oracle_root=a.original_oracle_root,evaluation_dir=a.evaluation_dir,output=a.output,
          independent_receipt=a.independent_receipt,synthetic_evaluation_dir=a.synthetic_evaluation_dir,diagnosis_GO=a.diagnosis_GO)
    print(dict(complete=r['complete'],bottleneck=r['predeclared_AB_FULL_bottleneck']))


if __name__=='__main__':main()
