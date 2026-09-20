"""Apply preregistered safety gates without selecting a cap or checkpoint."""
from scripts.research.pallet_posefix_replay_v1 import core as N


def decide(protocol,before,fit,result):
    g=protocol['promotion_gate'];checks={}
    def check(key,value,op,threshold):
        ok=value is not None and (value>=threshold-1e-12 if op=='>=' else value<=threshold+1e-12)
        checks[key]=dict(value=value,comparison=op,threshold=threshold,passed=bool(ok))
    checks['trainability']=dict(passed=bool(fit['trainability_pass']))
    source0=before['PRIOR_SOURCE']['clean'];source1=fit['source_probe']['clean']
    check('source_clean_PCK10_delta_pp',100*(source1['PCK10']-source0['PCK10']),'>=',-g['source_clean_PCK10_drop_max_pp'])
    check('source_clean_P90_ratio',source1['P90_px']/source0['P90_px'] if source0['P90_px']>0 else None,'<=',g['source_clean_P90_ratio_max'])
    for ds in g['populations']:
        raw=result['results'][ds]['POSEFIX_RAW'];base=result['results'][ds]['A_N2']
        rd=raw['matched_recovery_damage']
        check(ds+'_raw_damage_rate',rd['damage_rate'],'<=',g['raw_good5_to_bad10_rate_max'])
        check(ds+'_raw_PCK10_delta_vs_N2_pp',100*(raw['PCK']['10']-base['PCK']['10']),'>=',-g['raw_PCK10_vs_N2_drop_max_pp'])
        denom=base['matched_pooled_corner8_P90_px']
        check(ds+'_raw_P90_ratio_vs_N2',raw['matched_pooled_corner8_P90_px']/denom if denom>0 else None,'<=',g['raw_P90_vs_N2_ratio_max'])
    recovered=result['results']['DEV72']['POSEFIX_RAW']['matched_recovery_damage']['recovered']
    check('DEV72_raw_recovered',recovered,'>=',g['DEV_raw_min_recovered_hard_corners'])
    passed=all(c['passed'] for c in checks.values())
    return dict(complete=True,gate_pass=passed,checks=checks,
        failed_checks=[k for k,c in checks.items() if not c['passed']],
        verdict='PASS_FOR_BOUNDED_SELECTIVE_SELFTTRAIN' if passed else 'REPLAY_SCREEN_FAILED_DO_NOT_START_SELFTTRAIN',
        selective_selftraining_started=False,auto_promoted=False,independent_confirmation=False,
        no_cap_selection=True,checkpoint='last300',
        next_action='LOCK_AND_RUN_SELECTIVE_SELFTTRAIN' if passed else 'STOP_BOUND_SCREEN_KEEP_EXISTING_FINAL_MODEL')


def report(decision,before,fit,result):
    lines=['# PoseFix 합성 GT replay 보존 통제 실험', '',f"판정: `{decision['verdict']}`.", '',
        '## 실제 실행과 비교 조건', '',
        '- 추가학습 전의 같은 PRIOR1에서 시작. 기존 real-only와 seed1/300step/TFAdam1e-4/전층 학습/BN통계고정 동일.',
        '- 매 step 실사8장의 index와 입력 교란을 기존 RNG6401로 재생성해 bit-exact 검증. 실사 노출2400회 그대로 유지.',
        '- 합성8장/step, 총2400노출 추가. 합성 GT supervision만 추가하며 새 pseudo-label/teacher anchor/추가 유지loss 없음.',
        '- 목적함수=real 데이터loss + source 데이터loss + 기존 L2 한 번. real loss 또는 L2를 절반으로 줄이거나 두 배로 늘리지 않음.',
        '- 합성은 정상R0입력과 GT+큰교란을50:50확률로 사용. 정상/교란replay의 개별효과는 분리하지 않음.',
        '- 전체 계산량과 노출은 증가하므로 compute-matched 주장이 아님. replay가 더 적은 real노출 때문에 좋아졌다는 설명은 배제.',
        '- 같은9장/38manual감독, 미확인/PnP/중심정답제외. RGB+R0점/박스만 입력; 깊이/CAD/PnP/치수는 추론 입력 아님.',
        '- 새 source heldout256은 replay train과분리. 기존R0/과거연구노출은배제하지않아독립최종검증아님.',
        '- 실사평가는같은DEV72+GREEN150, 촬영분리·고정R0검출·GT분모·전체물체대칭평가유지. 평가로checkpoint/cap/threshold를선택하지않음.', '',
        f"학습 및 최종 probe/저장 경과 {fit['elapsed_seconds']:.1f}초, peak allocated {fit['peak_allocated_MiB']:.1f}MiB.", '',
        '## 같은 학습9장: 학습능력 검사일 뿐 일반화 아님', '',
        '| 입력 | 입력평균 px | replay 출력평균 px | PCK10 | 복구/hard |',
        '|---|---:|---:|---:|---:|']
    for k,r in fit['real_probe'].items():
        lines.append(f"| {k} | {r['input_mean_px']:.3f} | {r['output_mean_px']:.3f} | {100*r['PCK10']:.3f}% | {r['recovered']}/{r['hard']} |")
    lines+=['', 'noisy_manual304개는38개정답×8교란이며새로운304개실사정답이아님.', '',
        '## 합성 heldout256: 원본 prepared RGB px, fixed index', '',
        '| 모델 | 입력 | 평균 px | P90 px | PCK10 | 복구/hard | 훼손/good |',
        '|---|---|---:|---:|---:|---:|---:|']
    for label,probes in [('기존합성PoseFix',before['PRIOR_SOURCE']),('실사만',before['REAL_ONLY_SOURCE']),('실사+합성replay',fit['source_probe'])]:
        for mode,r in probes.items():
            lines.append(f"| {label} | {mode} | {r['output_mean_px']:.3f} | {r['P90_px']:.3f} | {100*r['PCK10']:.3f}% | {r['recovered']}/{r['hard']} | {r['damaged']}/{r['good']} |")
    lines+=['', '## 실사 촬영분리 개발평가', '',
        '| 모델 | DEV PCK10 | DEV P90 px | DEV 복구/hard | DEV 훼손/good | GREEN manual PCK10 | GREEN P90 px | GREEN 복구/hard | GREEN 훼손/good |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    rows=[('R0','R0',result['results']),('현재 DIM-only','A_N2',result['results'])]
    for label,data in [('기존 PoseFix',result['historical_results']['EXISTING_SYNTHETIC']),('실사만',result['historical_results']['REAL_ONLY']),('실사+replay',result['results'])]:
        for arm in ('POSEFIX_RAW','POSEFIX_CAP8'):rows.append((label+(' raw' if arm=='POSEFIX_RAW' else ' 1%cap'),arm,data))
    for label,arm,data in rows:
        a=data['DEV72'][arm];b=data['GREEN150_MANUAL'][arm]
        ar=a['matched_recovery_damage'];br=b['matched_recovery_damage']
        lines.append(f"| {label} | {100*a['PCK']['10']:.3f}% | {a['matched_pooled_corner8_P90_px']:.3f} | {ar['recovered']}/{ar['hard']} | {ar['damaged']}/{ar['good']} | {100*b['PCK']['10']:.3f}% | {b['matched_pooled_corner8_P90_px']:.3f} | {br['recovered']}/{br['hard']} | {br['damaged']}/{br['good']} |")
    lines+=['', 'GREEN matched hard4개뿐. 전체hard75중71은박스매칭실패패널티라점보정으로복구불가. 1%cap은640×480에서8px로, 같은GT대응의>20→<=10복구에필요한이동보다작음. 그러므로 raw가복구검사의주출력,cap은안전성참고이며좋은모드선택금지.', '',
        '## 실행 전 고정한 후속 self-training 진입 기준', '',
        '| 검사 | 관측값 | 조건 | 통과 |','|---|---:|---|---|']
    for key,c in decision['checks'].items():
        value=c.get('value');value='—' if value is None else f'{value:.6g}'
        condition=f"{c.get('comparison','')} {c.get('threshold','same training sanity criteria')}"
        lines.append(f"| {key} | {value} | {condition} | {c['passed']} |")
    lines+=['', '모든 기준을 만족해야 후속 선택적self-training에진입한다. 기준은개발screen의사전안전범위이지통계적비열등성검증이아니다.', '',
        '## 실행 범위와 보존', '',
        f"- 후속 선택적self-training 시작 여부: {decision['selective_selftraining_started']}.",
        '- 실패 시 새 pseudo-label을만들거나학생을추가학습하지않음. threshold/seed/lr/cap변경으로결과를구제하는추가탐색없음.',
        '- 원본 R0/PoseFix/N2 checkpoint, 기존 결과, 최종 모델, 어노테이션과 논문 표를 보존함. 자동 교체/commit/push 없음.',
        '- 222장 전체평가,baseline metric parity744개,검출/중심등보존888개검사.',
        '- PCK10은 전체GT유효코너분모,매칭실패패널티포함. P90은매칭코너. 복구는R0>20→<=10,훼손은R0<5→>10.',
        '- 동일9장의부분/불균형감독과개발자료재사용은여전히한계다. 실패를RGB보정/PoseFix/self-training전체의불가능으로확대하지않음.',
        '- 정상합성GT replay와큰교란source둘다추가된개입이다. 계산량증가,source분포,정규화효과를각각분리한실험은아님.', '',
        '산출물: PROTOCOL.json, INPUT_LOCK.json, BEFORE.json, FIT.json, REAL_RESULTS.json, DECISION.json, AUDIT.json.',
        '그림: outputs/pallet_posefix_replay_v1/user_example_right_zoom.png 및 사용자전체/가장개선/가장악화사례.', '']
    (N.DOC/'RESULTS_KO.md').write_text('\n'.join(lines))


def main():
    p=N.verify();before=N.E.read(N.DOC/'BEFORE.json');fit=N.E.read(N.DOC/'FIT.json');r=N.E.read(N.DOC/'REAL_RESULTS.json')
    N.F.verify(fit['checkpoint'])
    from scripts.research.pallet_posefix_replay_v1.evaluate import verify_result
    verify_result(r)
    history=N.E.read(N.RAW/'TRAIN_STEPS.json');assert [x['step'] for x in history]==list(range(1,301))
    assert len(history)==300 and min(x['update_norm'] for x in history)>0
    assert fit['BN_buffers_bit_exact'] and fit['real_input_order_bit_exact']
    assert (r['evaluated_images'],r['preservation_checks'],r['baseline_metric_parity_checks'])==(222,888,744)
    d=decide(p,before,fit,r);N.freeze(N.DOC/'DECISION.json',d);report(d,before,fit,r)
    N.freeze(N.DOC/'AUDIT.json',dict(complete=True,original_bindings_verified=True,actual_updates=300,
        real_exposures=2400,synthetic_exposures=2400,real_inputs_bit_exact=True,BN_buffers_bit_exact=True,
        evaluated_images=222,preservation_checks=888,baseline_metric_parity_checks=744,
        artifacts=[N.E.bound(N.DOC/n) for n in ('PROTOCOL.json','INPUT_LOCK.json','BEFORE.json','FIT.json','REAL_RESULTS.json','DECISION.json','RESULTS_KO.md')],
        original_model_replaced=False,selective_selftraining_started=False,pushed=False))
    N.write(N.DOC/'STATUS.json',dict(stage='PRESERVATION_SCREEN_COMPLETE',gate_pass=d['gate_pass'],next_action=d['next_action']))
    print(d,flush=True)


if __name__=='__main__':main()
