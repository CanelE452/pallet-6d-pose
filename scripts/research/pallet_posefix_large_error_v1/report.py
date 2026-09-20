"""Generate a source-linked report from completed, immutable screen artifacts."""
import json
from scripts.research.pallet_posefix_large_error_v1 import core as C


def main():
    protocol=C.E.read(C.DOC/'PROTOCOL.json'); C.verify_bindings(protocol['sources'])
    for b in protocol['code'].values(): C.F.verify(b)
    C.F.checked_lock()
    fit=C.E.read(C.DOC/'FIT.json'); C.F.verify(fit['checkpoint'])
    paths=[C.DOC/'EXISTING_RESULTS.json']
    if fit['trainability_pass']: paths.append(C.DOC/'FINETUNED_RESULTS.json')
    loaded=[]
    for path in paths:
        r=C.E.read(path); assert r['complete'] and r['evaluated_images']==222
        C.verify_bindings(r['source_bindings'])
        for b in r['data_bindings']+r['artifacts']: C.F.verify(b)
        assert r['preservation_checks']==888 and r['baseline_metric_parity_checks']==744
        loaded.append(r)
    history=C.E.read(C.RAW/'TRAIN_STEPS.json')
    assert [r['step'] for r in history]==list(range(1,301))
    assert min(r['update_norm'] for r in history)>0
    support=C.E.read(C.DOC/'TRAIN_SUPPORT.json')
    assert support['manual_corners']==support['crop_supported_corners']==38
    before,after=fit['before'],fit['after']
    deltas={}
    if fit['trainability_pass']:
        latest=loaded[-1]['results']
        deltas={ds:{arm:100*(latest[ds][arm]['PCK']['10']-latest[ds]['A_N2']['PCK']['10'])
            for arm in ('POSEFIX_RAW','POSEFIX_CAP8')} for ds in ('DEV72','GREEN150_MANUAL')}
    degraded=bool(deltas and all(v<0 for rows in deltas.values() for v in rows.values()))
    verdict='TRAINABILITY_PASS_GENERALIZATION_DEGRADED_NOT_PROMOTED' if degraded else 'BOUNDED_DIAGNOSTIC_COMPLETE_NOT_PROMOTED'
    C.freeze(C.DOC/'DECISION.json',dict(verdict=verdict,trainability_pass=fit['trainability_pass'],
        PCK10_delta_vs_N2_pp=deltas,independent_confirmation=False,auto_promoted=False,
        noise_causal_effect_estimated=False,more_training_runs=0))
    lines=['# PoseFix 큰 오차 복구 — 실제 실행 결과', '',
        f'판정: `{verdict}`.', '',
        '같은 학습 사진에서는 큰 이동을 배울 수 있었다. 그러나 이번 소량 추가학습 모델은 촬영이 다른 평가 사진에서 정상 코너를 훼손하고 전체 PCK10이 낮아졌다. 최종 모델로 교체하지 않는다. 이 결과가 PoseFix 또는 RGB 보정의 가능성 자체를 부정하지는 않는다.', '',
        '## 과거 실험에 대한 정정', '',
        'PoseFix를 아직 실험하지 않았다는 설명은 잘못이었다. 2026-09-15 submission_v1에서 공식 TF1 네트워크의 연산·가중치 parity를 확인한 RGB ResNet152 PoseFix-derived pallet9를 seed 1/2/3 × 6,000 step 학습·평가했다. 원 논문의 사람 포즈 전체 프로토콜 재현은 아니다.', '',
        '당시 DEV319의 고정 인덱스·중심점 포함 PCK10은 R0 63.7331%, 기존 P 3-seed 평균 67.3409%, PoseFix-derived 3-seed 평균 68.4055%였다. 아래의 새 8코너·전체 물체 대칭 지표와 직접 비교하지 않는다.', '',
        '근거: ../pallet_sensors_submission_v1/FINAL_REPORT_KO.md, TRAIN_COMPLETE.json, PRIOR_PROTOCOL_LOCK.json.', '',
        '연속 영상 역시 7프레임 정렬·합의를 109센터에서 실행했다. 다만 후속 평가 감사에서 정식 적격 평가 대상이 0장으로 확인되어 FORMAL_TEMPORAL_PILOT=POPULATION_LIMITED, 원래 결과=EXPLORATORY_DIAGNOSTIC_ONLY로 정정했다. Student 학습은 없었다. 따라서 연속 영상이 효과 없다고 정식 입증한 실험은 아니다.', '',
        '근거: data/pallet/results/paper_temporal_selftrain_v1/evaluation_closure_v1/TEMPORAL_EVALUATION_CLOSURE_REPORT.md.', '',
        '## 이번 실제 실행', '',
        '- 기존 PRIOR1 재평가 222장: raw 출력과 과거의 이미지 대각선 1% 이동 제한을 동시에 보고한다.',
        '- 새 학습: 기존 PRIOR1 복사본, 실사 TRAIN 9장/직접 클릭 38점, seed1 × 300 step, batch8/micro2, TFAdam 1e-4. BN 통계 고정, 전층 파라미터 학습.',
        '- RGB와 R0 초기점·예측 박스만 사용한다. 깊이, CAD, 렌더링, PnP, 치수 입력 없음. 최종 모델/원본 가중치/어노테이션/논문 표 변경 없음.',
        '- 각 학습 이미지 50%는 원래 R0 입력, 50%는 manual 정답 위치에 예측 박스 대각선 5–15% 길이의 독립 오차를 추가한다. 미확인/PnP/중심점은 정답으로 쓰지 않는다.',
        '- 직접 클릭 38점 전부 R0 crop 내부. 코너별 감독수 [8,5,4,4,8,7,1,1]; 매우 불균형하다. 기존 loss는 유효점 수가 아닌 전체9채널 평균을 그대로 사용한다.',
        '- 학습9장과 평가 촬영을 분리한 기존 split 유지. 평가 데이터는 과거에 연구에 재사용되었으므로 독립 검증이 아니다.',
        '- 동일 학습 사진의 독립 noise seed 8회씩을 검사한다. 304개는 서로 다른 실제 코너304개가 아니라 38점을 8번 교란한 것이다.', '',
        '## 학습능력 검사: 고정 인덱스, 원본 이미지 px', '',
        '| 입력/검사 | 입력 평균오차 | 기존 PoseFix 출력 | 300 step 출력 | 이후 PCK10 | 이후 큰오차 복구 |',
        '|---|---:|---:|---:|---:|---:|']
    for key,label in [('original_R0','실제 R0 입력 / 같은 학습9장'),('noisy_manual','주입한 큰 오차 / 같은 학습9장')]:
        a,b=before[key],after[key]
        lines.append(f"| {label} | {a['input_mean_px']:.3f} | {a['output_mean_px']:.3f} | {b['output_mean_px']:.3f} | {100*b['PCK10']:.2f}% | {b['recovered']}/{b['hard']} |")
    lines += ['',f"사전 trainability 기준 통과: {fit['trainability_pass']}. 학습 및 최종 probe/저장 경과 {fit['elapsed_seconds']:.1f}초.", '',
        '이 표는 학습능력 검사다. 실제 R0 입력에서도 학습한 사진의 오차 감소일 뿐, 실사 일반화의 증명이 아니다.', '',
        '## 촬영 분리 평가', '',
        'PCK10은 전체 유효 GT 코너 분모이며 검출 매칭 실패에는 패널티를 부여한다. median/P90은 매칭된 코너. 복구는 같은 R0 >20px 코너가 <=10px, 훼손은 같은 R0 <5px가 >10px. 전체 물체에 허용된 C1/C2/C4 순열만 사용하며 개별 코너 GT를 재배정하지 않는다.', '',
        '| 모델 | DEV72 PCK10 | DEV72 median / P90 px | DEV 복구 / hard | DEV 훼손 / good | GREEN150 manual PCK10 | GREEN 매칭 복구 / hard | GREEN 훼손 / good |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    rows=[('R0','R0',loaded[0]),('현재 DIM-only','A_N2',loaded[0])]
    for result in loaded:
        label='기존 PoseFix' if result['model']=='existing' else '실사9장 추가학습 PoseFix'
        rows += [(label+' raw','POSEFIX_RAW',result),(label+' 1% cap','POSEFIX_CAP8',result)]
    for label,arm,result in rows:
        d=result['results']['DEV72'][arm]; g=result['results']['GREEN150_MANUAL'][arm]
        dr=d['matched_recovery_damage']; gr=g['matched_recovery_damage']
        lines.append(f"| {label} | {100*d['PCK']['10']:.3f}% | {d['matched_pooled_corner8_median_px']:.3f} / {d['matched_pooled_corner8_P90_px']:.3f} | {dr['recovered']}/{dr['hard']} | {dr['damaged']}/{dr['good']} | {100*g['PCK']['10']:.3f}% | {gr['recovered']}/{gr['hard']} | {gr['damaged']}/{gr['good']} |")
    lines += ['',
        'GREEN150 manual 전체681점 중 매칭610점/134장, >20px인 매칭 코너는4개뿐이다. 전체 hard75 중71은 검출 박스 매칭 실패 패널티라 점 보정으로 복구할 수 없다. 이 작은4개 분모로 일반적인 큰 오차 성능을 주장하지 않는다.', '',
        '## 해석 경계', '',
        '- 작은 실사 집합에 과적합하는 이 검사는 구조가 큰 이동을 표현·학습할 수 있는지 보는 것이다. 새로운 최종 모델 훈련이나 논문 기여 검증이 아니다.',
        '- 기존 initialization과 실사 감독/오류 주입을 함께 사용했다. 정상 R0 입력만 사용한 동일예산 새 학습군이 없으므로 오류 주입의 독립 인과효과는 분리하지 않는다.',
        '- 기존 geometry/flip 필터를 새 실험에 넣지 않았고 평가 이미지를 골라 버리지 않았다. 필터를 통과한다고 위치가 정확한 것은 아니다.',
        '- 사용자 예제에는 코너 역할/대응 문제가 함께 있다. 전체 대칭 지표의 큰 오차를 전부 국소 위치 오차라고 해석하지 않는다.',
        '- 검출 후보·박스·confidence·centroid는 모두 유지된다. Negative 오검출 개선/AP 개선 실험이 아니다.',
        '- 최종 checkpoint는 항상300step이다. 평가로 seed/step/상한을 선택하지 않았으며 raw/cap 결과를 모두 보존했다.', '',
        'PoseFix 원 논문은 RGB와 초기 포즈로 보정하고 오류 통계로 입력 포즈를 합성한다. 이번은 그 아이디어를 따르되, 사람의 swap/inversion prior 대신 명시한 팔레트 코너 위치 교란을 사용한 소량 진단이다. https://arxiv.org/abs/1812.03595', '',
        '## 산출물', '',
        '- PROTOCOL.json: 실행 전 코드·원본 checkpoint·평가 자료 hash 고정.',
        '- FIT.json, TRAIN_SUPPORT.json: 300step 실제학습과 학습능력 검사.',
        '- EXISTING_RESULTS.json / FINETUNED_RESULTS.json: 동일 데이터의 재평가(후자는 sanity 통과 시).',
        '- outputs/pallet_posefix_large_error_v1/{existing,finetuned}/: 사용자 예제 전체·오른쪽 확대, 가장 개선/악화된 두 사례. 모두 저장된 좌표의 실제 이미지 overlay.',
        '- AUDIT.json: 원본 보존, step수, 분모 및 예측 보존 점검.', '']
    (C.DOC/'RESULTS_KO.md').write_text('\n'.join(lines))
    C.freeze(C.DOC/'AUDIT.json',dict(complete=True,original_bindings_verified=True,
        final_release_verified=True,actual_updates=300,minimum_update_norm=min(r['update_norm'] for r in history),
        BN_buffers_bit_exact=fit['BN_buffers_bit_exact'],manual_supported=38,
        inference_passes=len(loaded),evaluated_images_per_pass=222,
        baseline_metric_parity_checks_per_pass=744,preservation_checks_per_pass=888,
        original_model_replaced=False,pushed=False,negative_FP_test=False,
        runtime=C.E.bound(C.DOC/'INFERENCE_RUNTIME.json'),
        artifacts=[C.E.bound(p) for p in [C.DOC/'FIT.json',*paths,C.DOC/'DECISION.json',C.DOC/'RESULTS_KO.md']]))
    C.write(C.DOC/'STATUS.json',dict(stage='COMPLETE',trainability_pass=fit['trainability_pass'],
        evaluation_passes=len(loaded),auto_promoted=False))
    print(C.DOC/'RESULTS_KO.md')


if __name__=='__main__': main()
