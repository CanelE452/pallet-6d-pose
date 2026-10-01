"""Publish fixed pairwise experiment results, including negative results."""
from . import common as C
from . import figures
import csv
import json
from pathlib import Path
import numpy as np


def write_csv(name, rows):
    with (C.DOC/name).open('w', newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        w.writeheader();w.writerows(rows)


def table(summaries):
    rows=['| 모델 | T 중앙값 cm | R 중앙값 ° | T P90 cm | R P90 ° | 실패 |',
          '|---|---:|---:|---:|---:|---:|']
    for model,s in summaries.items():
        p=s['full_population']
        values=[p[a][q] for a,q in [('translation_cm','median'),('rotation_deg','median'),
                                   ('translation_cm','P90'),('rotation_deg','P90')]]
        rows.append('| '+model+' | '+' | '.join('∞/NA' if v is None else f'{v:.6f}' for v in values)+f" | {s['failed_pose']} |")
    return '\n'.join(rows)


def main():
    p=C.protocol()
    gate=C.read(C.DOC/'SOURCE_VAL_GATE.json')
    complete=C.read(C.DOC/'TRAINING_COMPLETE.json')
    training_check=C.read(C.DOC/'TRAIN_CONVERGENCE.json')
    assert training_check['complete'] and training_check['independent_check_PASS']
    assert gate['complete'] and complete['complete'] and complete['all_certified']
    choices=C.read(C.RAW/'SOURCE_VAL_CHOICES.json')
    real_path=C.DOC/'REAL_RESULTS.json'
    real=C.read(real_path) if real_path.exists() else None
    assert not gate['PASS'] or (real is not None and real['complete']), 'Source PASS requires full real evaluation before final report.'
    assert gate['PASS'] or real is None
    achieved=bool(real and real['stability']['PASS'])
    exports={};fits={};logs=[]
    (C.DOC/'model_parameters').mkdir(exist_ok=True)
    for model in C.MODEL_NAMES:
        fit=C.read(C.DOC/f'FIT_{model}.json');fits[model]=fit
        origin=C.ROOT/fit['checkpoint']['path']
        target=C.DOC/'model_parameters'/f'{model}.json'
        target.write_bytes(origin.read_bytes())
        assert C.sha(target)==fit['checkpoint']['sha256']
        exports[model]=dict(local=fit['checkpoint'],published=C.bind(target),new_fit=model in C.NEW_MODELS)
        if model in C.NEW_MODELS:
            trace=[json.loads(line) for line in (C.ROOT/fit['trace']['path']).read_text().splitlines()]
            logs.extend(dict(model=model,**row) for row in trace if row['event']=='objective')
    write_csv('TRAINING_OBJECTIVE_LOG.csv',logs)
    rows=[]
    with np.load(C.RAW/'SOURCE_VAL_METRICS.npz') as z:
        for model in z['models'].tolist():
            for fid,error in zip(z['ids'].tolist(),z[model]):
                pick=choices['records'].get(model,{}).get(fid,{})
                rows.append(dict(id=fid,model=model,split='VAL',T_cm=float(error[0]),R_deg=float(error[1]),
                    available=bool(np.isfinite(error).all()),candidate=pick.get('candidate_name','fixed_GEO'),
                    fallback=pick.get('fallback',False)))
    write_csv('SOURCE_VAL_FRAME_RESULTS.csv',rows)
    checks=[dict(model=m,baseline=b,criterion=k,PASS=v) for m,cc in gate['comparisons'].items()
            for b,d in cc.items() for k,v in d['checks'].items()]
    write_csv('SOURCE_VAL_CHECKS.csv',checks)
    figures.supervision_graph();figures.measured(gate,logs)
    calls=sum(fits[m]['objective_calls'] for m in C.NEW_MODELS)
    iterations=sum(fits[m]['iterations'] for m in C.NEW_MODELS)
    elapsed=sum(fits[m]['wall_seconds'] for m in C.NEW_MODELS)
    fit_rows=[]
    for model in C.MODEL_NAMES:
        fit=fits[model];cert=fit['certificate']
        fit_rows.append(f"| {model} | {'신규 학습' if model in C.NEW_MODELS else '기존 weight 재사용'} | "
            f"{fit['iterations']} | {fit['objective_calls']} | {cert['gradient_l2']:.3e} | "
            f"{cert['gradient_l2_squared_over_2lambda']:.3e} |")
    transfer_rows=['| 모델 | 최선 후보 정확도: 이전→신규 | 평균 regret: 이전→신규 | 최종 가설 오류: 이전→신규 |',
                   '|---|---:|---:|---:|']
    for model in C.NEW_MODELS:
        row=training_check['models'][model]
        before=row['old_converged_same_pool']['metrics'];after=row['new_TRAIN_metrics']
        transfer_rows.append(f"| {model} | {100*before['whole_best_accuracy']:.2f}% → {100*after['whole_best_accuracy']:.2f}% | "
            f"{before['mean_regret_defined_rows']:.4f} → {after['mean_regret_defined_rows']:.4f} | "
            f"{before['global_choice_different_hypothesis_from_best']} → {after['global_choice_different_hypothesis_from_best']} |")
    if not gate['PASS']:
        lead=f"**이번 방법도 안정적 T·R 개선에 실패했다.** 신규 선택기3개를 실제 학습해 수렴을 확인했지만, 합성 VAL45개 조건 중{gate['checks_passed']}개만 통과했다. 기존 R0_ONLY 대조는 같은 목적함수임을 확인하고 재사용했다. 사전 중단 조건에 따라 새 실사 선택·평가는0회다."
    else:
        lead='**합성 검증45개 조건을 통과해 실사173장 전체를 평가했다.** '+('사전 고정한 실사 안정성 조건을 모두 통과했다.' if achieved else '실사 안정성 조건은 통과하지 못했으므로 목표는 아직 미달이다.')
    lines=['# W/D·expert pairwise 감독: 전체 실행 결과',lead,
        '입력은 **이미지 한 장과 팔레트 치수**, 기존 보정 K/기하 계약이다. 시간 정보나 새 실사 GT 학습을 추가하지 않았다. 이 보고서는 합성·실사 결과, 실제 학습·재사용, 수렴·성능 판정을 구분한다.',
        '## 바꾼 것과 검증하려는 설명',
        '[직전 수렴 실험](../pallet_pose_selector_convergence_20261001_v1/REPORT_KO.md)에서는 기존 one-hot CE 선택기를 충분히 수렴시켜도 source gate4개가 실패했다. 이번에는 기존94특징·후보4개·최종 전체 argmin을 유지하면서 감독 구조만 바꾼다. 최상위 후보 하나만 분류하던 방식에서, 각 expert 안의 long/short 비교2개와 같은 가설 안의 R0/DIVERSE 비교2개를 함께 학습한다.',
        '![사전 고정한 네 후보 비교 구조](figures/supervision_graph.png)',
        '각 edge의 정답은 기존 `max(T/sT,R/sR)`가 작은 후보다. W/D group이라는 이름은 비교하는 후보의 종류를 뜻하며 renderer의 축 parity 자체를 정답으로 삼았다는 뜻이 아니다. T와 R을 서로 다른 후보에서 가져오지 않는다. 런타임에서 W/D를 먼저 제거하는 계단식 router도 아니다.',
        '네 edge에 각각¼의 binary logistic loss를 주고, 전체2598행 평균에 명시적 ridge `0.5×1e−4×||w94||²`를 더한다. 비용 차이에 비례하는 가중이나 VAL을 보고 정한 임계값은 없다. 후보가 없거나 한쪽이 invalid인 edge의 loss는0으로 유지하고 분모를 재정규화하지 않는다.',
        '## 학습 전에 확인한 정답 계약',
        '[TARGET_CONTRACT_AUDIT_KO.md](TARGET_CONTRACT_AUDIT_KO.md)와 [원본 JSON](TARGET_CONTRACT_AUDIT.json)에 모든 TRAIN2598행을 검증했다. seed마다2597행은 네 후보가 모두 유효하고1행은 모두 invalid다. 실제 TRAIN의 exact-cost tie와 cycle은0이며, 새 전역 순서의 최상위 후보는 기존 target과2598/2598행에서 같다.',
        '이론적으로는 pair마다 Pareto 동점을 따로 처리할 때 순환 순서가 생길 수 있다. 학습 전에 비용→동일 비용 내 Pareto front→기존 R0/가설 tie key라는 하나의 전역 순서를 고정했다. 실제 TRAIN의 edge label 변경은0이다. 존재하지 않은 실제 데이터 오류를 수정한 성과로 포장하지 않는다.',
        '네 edge는 대각 비교2개를 포함하지 않는다. 따라서 모든 edge를 정확히 맞혀도 전체 cost-best를 항상 식별하는 것은 아니다. 실제 TRAIN에서 두 후보가 동시에 zero-indegree인 행은 seed별4/5/2개다. 이 한계를 유지한 채 원래 전체 pose 성능 기준으로 판정한다. edge 정확도 상승을 T/R 개선으로 대신하지 않는다.',
        '## 실제 실행량과 수렴',
        '| 모델 | 실행 | 새 iteration | 새 objective 호출 | gradient L2 | objective gap 상한 |',
        '|---|---|---:|---:|---:|---:|',*fit_rows,
        f'신규 fit3회, objective/gradient 호출{calls}회, optimizer iteration{iterations}회다. CPU에서 실행한 세 fit의 경과시간 합은{elapsed:.3f}초다. 기존 이미지 예측·pose 캐시를 사용해 새 이미지 forward와 PnP 계산은0회다.',
        'R0_ONLY의 한 long/short pair logistic loss는 기존 두 후보 CE와 같은 수학적 함수다. 기존 weight와 정규화는 bit 단위로 유지하고, 독립 objective/gradient 동등성 검사 후 새 protocol metadata wrapper에 원본 계보를 기록했다. 이 wrapper 작성은 새 학습이 아니다. 공개 wrapper JSON과 원래 checkpoint JSON 전체가 byte-identical하다고 주장하지 않는다.',
        '![신규 세 선택기의 학습 목적함수](figures/training_objective.png)',
        '세 신규 fit은 모두0 초기화, float32 정규화 후 float64, CPU1thread, L-BFGS-B 최대1000iteration/2000closure를 사용한다. optimizer success와 `||gradient||²/(2λ)≤1e−6`를 동시에 요구한다. 이 수렴 인증은 새로운 pairwise objective에 대한 것이며 T/R나 일반화 보장이 아니다. 서로 다른 one-hot CE와 pairwise loss의 숫자 크기를 직접 비교해 성능 향상이라고 해석하지 않는다.',
        '[사전 protocol](PROTOCOL.json), [SHA](PROTOCOL_SHA.json), [실행 완료](TRAINING_COMPLETE.json), [전체 학습 로그 CSV](TRAINING_OBJECTIVE_LOG.csv), [실제 최종 파라미터](model_parameters/)를 공개한다.',
        '## 작은 비교의 정확도가 최종 선택으로 이어졌는가',
        '별도 PyTorch float64 softplus/autograd로 저장 weight의 objective와 gradient를 검산했다. 세 모델의 수렴 인증과 R0 재사용은 모두 통과했다. 이전에 수렴시킨 one-hot CE weight도 같은 TRAIN 후보의 새 pairwise objective에 적용해 비교했다. 이 비교는 서로 다른 loss의 숫자를 직접 비교한 것이 아니다.',
        '\n'.join(transfer_rows),
        '새 pairwise objective는 내려갔고 W/D edge 정확도도 조금 올라갔지만, 최종 전체 argmin이 올바른 최선 후보를 고르는 비율은 세 경우 모두 낮아졌다. 위의 가설 오류는 renderer parity가 아니라 cost-best 후보와 다른 long/short를 고른 수다. 평균 regret는 비용이 정의되는2597행의 값이며 후보 없는1행은 별도 실패로 유지했다. TRAIN에서도 작은 비교의 개선이 전체 pose 선택의 개선으로 연결되지 않았으므로, 이번 실패를 합성 VAL 전이 문제만으로 설명할 수 없다.',
        '[독립 TRAIN 검산과 상세 해석](TRAIN_CONVERGENCE_KO.md), [전체 수치](TRAIN_CONVERGENCE.json)에 같은 pool의 local edge·global 선택·T/R 차이를 기록했다. 이 결과는 이번 고정 네-edge 목적함수를 지지하지 않으며 모든 pairwise 또는 RGB 방법의 불가능성을 증명하지 않는다.',
        '## 합성 VAL1024장 결과',
        '모든 학습과 baseline 재사용 검증이 끝난 뒤4×1024개 선택을 동결하고 source 참조를 읽었다. 전체 분모와 기존 C2 회전·중심 이동 오차를 유지한다. source VAL은 반복 사용한 개발 자료이며 독립 TEST가 아니다.',
        table(gate['summaries']), '![공통 대조와 UNION 세 모델의 합성 T/R](figures/source_val_results.png)',
        f"45개 검사 중{gate['checks_passed']}개 통과, {len(gate['failed_checks'])}개 실패다. 다음 실패 목록은 전체 목록이다.",
        '\n'.join('- `'+item+'`' for item in gate['failed_checks']) or '실패한 source 검사는 없다.',
        '[전체8192행 CSV](SOURCE_VAL_FRAME_RESULTS.csv), [45개 판정 CSV](SOURCE_VAL_CHECKS.csv), [원본 gate](SOURCE_VAL_GATE.json), [선택 동결 기록](SOURCE_VAL_ROUTING_LOCK.json)에서 모든 모델과 seed를 확인할 수 있다.',
        '[독립 source 검산과 이전 선택 대비 변화](SOURCE_VAL_ANALYSIS_KO.md), [검산 JSON](SOURCE_VAL_VERIFICATION.json)을 함께 제공한다.',
        '## 실사 T·R와 목표 상태']
    if real is None:
        lines += ['이번 모델의 새 실사 routing·평가는 실행하지 않았다. 기존 natural99/clean29/wood45의 결과는 그대로다. source 실패를 실사 성능이 개선됐다는 근거로 사용하지 않는다. [실사 미실행 근거](REAL_EVALUATION_NOT_RUN_KO.md), [기존 전체 실사 결과](../pallet_pose_stable_improvement_20261001_v1/REPORT_KO.md), [자연 가림99장 이미지](../pallet_pose_stable_improvement_20261001_v1/GALLERY_NATURAL99.md)를 연결한다.']
    else:
        lines += [table(real['summaries']['NATURAL99']),
                  '| 사전 안정성 gate | 판정 |','|---|---|']
        lines += [f"| {name} | {'PASS' if value['PASS'] else 'FAIL'} |" for name,value in real['stability']['gates'].items()]
        lines += ['[실사 전체 결과](REAL_RESULTS.json), [모든 프레임 CSV](REAL_FRAME_RESULTS.csv), [촬영별·bootstrap 상세](REAL_DETAILED_COMPARISONS.json)를 함께 공개한다. 실사 평가 역시 반복 DEV와 주석·K·치수에서 유도한 참조이며 독립 실측 TEST가 아니다.']
    lines += ['## 공개와 재현',
        '구현: [학습](../../../scripts/research/pallet_pose_selector_pairwise_20261001_v1/train.py), [source 평가](../../../scripts/research/pallet_pose_selector_pairwise_20261001_v1/evaluate_source.py), [protocol 봉인](../../../scripts/research/pallet_pose_selector_pairwise_20261001_v1/seal.py), [보고서 생성](../../../scripts/research/pallet_pose_selector_pairwise_20261001_v1/report.py).',
        '원본 학습·평가 artifact는 덮어쓰지 않는다. 대형 이미지·YOLO/PoseFix weight·캐시는 이번 GitHub commit에 포함하지 않고 작은 scorer 파라미터·CSV·MD·이미지·코드와 hash를 공개한다. [공개 manifest](PUBLICATION_MANIFEST.json)는 실제 게시 범위를 기록한다.']
    rendered=''
    previous=''
    for part in lines:
        if rendered:
            rendered += '\n' if previous.startswith('|') and part.startswith('|') else '\n\n'
        rendered += part
        previous=part
    (C.DOC/'REPORT_KO.md').write_text(rendered+'\n')
    data=dict(complete=True,code=C.bind(Path(__file__)),figures_code=C.bind(Path(figures.__file__)),
        protocol=C.bind(C.DOC/'PROTOCOL.json'),source_gate=C.bind(C.DOC/'SOURCE_VAL_GATE.json'),
        new_fits=3,reused_models=1,new_objective_calls=calls,new_iterations=iterations,
        csv_rows={'TRAINING_OBJECTIVE_LOG.csv':len(logs),'SOURCE_VAL_FRAME_RESULTS.csv':len(rows),'SOURCE_VAL_CHECKS.csv':len(checks)},
        exports=exports,new_image_forwards=0,new_PnP_solves=0,real_evaluated=real is not None,
        stable_joint_improvement_achieved=achieved,cpu_elapsed_fit_seconds=elapsed)
    (C.DOC/'REPORT_DATA.json').write_text(json.dumps(C.U.clean(data),ensure_ascii=False,indent=2)+'\n')
    print('PAIRWISE_REPORT_COMPLETE',data['csv_rows'],flush=True)


if __name__=='__main__':main()
