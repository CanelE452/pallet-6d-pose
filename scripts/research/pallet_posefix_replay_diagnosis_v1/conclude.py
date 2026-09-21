"""Write evidence tables and bounded conclusions; never initiate Phase B training."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from scripts.research.pallet_posefix_replay_diagnosis_v1 import run as R
import numpy as np

def build():
    real=R.read(R.DOC/'PASS_METRICS_REAL_DEV.json')['splits']
    synth=R.read(R.DOC/'PASS_METRICS_SYNTH.json')['splits']
    numeric=R.read(R.DOC/'NUMERICAL_REPLAY.json')
    r=real['REAL_DEV']['mean_seed']['RAW']; c=real['REAL_DEV']['mean_seed']['CAPPED']
    square=real['SQUARE_DEV']['mean_seed']['RAW']; held=synth['heldout']['mean_seed']['RAW']
    perseed=real['REAL_DEV']['per_seed']; seeds=['1','2','3']
    phase=[perseed[s]['RAW']['symmetry']['1']['phase_only_frames'] for s in seeds]
    greenphase=[real['SQUARE_DEV']['per_seed'][s]['RAW']['symmetry']['1']['phase_only_frames'] for s in seeds]
    strong=sum(perseed[s]['RAW']['symmetry']['1']['phase_only_fraction']>=.1 for s in seeds)>=2
    recovery=[perseed[s]['RAW']['error_strata']['>20']['passes']['1']['raw_recovered_to10'] for s in seeds]
    n=int(r['error_strata']['>20']['n_corners']); h=r['error_strata']['>20']['passes']['1']
    objects={}
    for obj in perseed['1']['RAW']['objects']:
        objects[obj]=dict(frames=perseed['1']['RAW']['objects'][obj]['frames'],
            WDH=perseed['1']['RAW']['objects'][obj]['dimensions_WDH_unique'],
            passes={str(p):dict(median_px=float(np.mean([perseed[s]['RAW']['objects'][obj]['passes'][str(p)]['matched_pooled_corner8_median_px'] for s in seeds])),
                               PCK10=float(np.mean([perseed[s]['RAW']['objects'][obj]['passes'][str(p)]['PCK']['10'] for s in seeds]))) for p in range(4)},
            cap_hit_PASS1=float(np.mean([perseed[s]['RAW']['objects'][obj]['cap_hit_PASS1'] for s in seeds])),
            PASS3_vs_PASS1_corner_regression=float(np.mean([perseed[s]['RAW']['objects'][obj]['PASS3_vs_PASS1_corner_regression'] for s in seeds])))
    result=dict(complete=True,training_updates=0,model='canonical synthetic-only PRIOR1/2/3 last6000',
        old_replay_meaning='SYNTHETIC_TRAINING_DATA_REHEARSAL; neither iterative inference nor numerical parity',
        cases=dict(A_CAP_LIMITED='LOCAL_EFFECT_NOT_DOMINANT_FOR_LARGE_REAL_ERRORS',
            B_REPLAY_DISTRIBUTION_SHIFT='REPEAT_DEGRADATION_ON_SYNTHETIC_AND_SQUARE; realDEV mixed; observed feedback effects not causal OOD proof',
            C_FIXED_INDEX_AMBIGUITY='MAIN_STRONG_GATE_MET' if strong else 'MAIN_STRONG_GATE_NOT_MET; square minority already present in R0',
            D_POSEFIX_RAW_NOT_CORRECTIVE='LARGE_ERROR_RECOVERY_INSUFFICIENT; not a claim all raw corrections fail',
            E_NUMERICAL_REPLAY_ONLY='BIT_EXACT_FAIL_WITHIN_REFERENCE_TOLERANCE; cannot explain accuracy differences'),
        real_RAW_passes=r['passes'],real_CAPPED_passes=c['passes'],real_RAW_pose=r['pose'],
        real_hard_corners=n,real_PASS1_hard_recovery_by_seed=recovery,real_PASS1_hard_statistics=h,
        real_PASS1_cap=r['movement']['1'],real_oscillation=r['oscillation'],
        real_phase_only_by_seed=phase,square_phase_only_by_seed=greenphase,
        synthetic_heldout_RAW_passes=held['passes'],square_RAW_passes=square['passes'],objects=objects,
        numerical_crop_max=max(v['crop_max_abs'] for v in numeric['runs']),numerical_restored_max=max(v['restored_max_abs'] for v in numeric['runs']),
        numerical_all_within_3e4=all(v['within_reference_tolerance'] for v in numeric['runs']),
        phase_B_triggered=strong,training_started=False,model_promoted=False,
        inference_scope='Cannot transfer this causal decomposition automatically to later real+synthetic last300 Replay; that is a different checkpoint.',
        limitations=['Reused DEV, no independent generalization confirmation','No cap/pass/seed tuning or winner promotion',
            'Real and square 6D references reconstructed from annotations, not independently measured physical pose',
            'Whole-object phase diagnostic cannot exclude local corner swaps or averaging induced by training ambiguity',
            'Object-type differences do not causally prove absence of dimension conditioning is responsible'])
    R.write(R.DOC/'FINAL_DIAGNOSIS.json',result)
    lines=['# 최종 진단: PoseFix 1회·반복·cap·대칭·수치 재현성', '',
        '**기존 `pallet_posefix_replay_v1`의 replay는 합성 학습 데이터 재사용이다. 반복 추론(A)도 수치 재현(B)도 아니다.** 기존 last300의 실패는 완료된 1회 추론의 정확도 기준 미달이었다. 이번 새 진단은 원래 합성-only PRIOR1/2/3 last6000으로 수행했으며, 기존 last300의 실패 원인을 그대로 인과적으로 설명하는 실험은 아니다.', '',
        '학습 0 step. 합성 calibration1,004 + selection1,031 + heldout1,985 + 실사 DEV319 + 별도 C4 DEV155. 모두 기존 분할 전체, 최대PASS3. RAW와 CAPPED는 별도 피드백 경로다. 기존 최종 모델과 표는 바꾸지 않았다.', '',
        '## 1. 1회 보정은 동작하는가? — 동작한다', '',
        '현재 8코너·전체물체 대칭 평가, 세 seed 요약값의 산술평균:', '',
        '| DEV319 RAW | 중앙오차 px | P90 px | PCK10 % | rotation deg | translation cm | ADDsym AUC |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for p in range(4):
        m=r['passes'][str(p)];q=r['pose'][str(p)]
        lines.append(f'| PASS{p} | {m["matched_pooled_corner8_median_px"]:.3f} | {m["matched_pooled_corner8_P90_px"]:.3f} | {100*m["PCK"]["10"]:.3f} | {q["rotation_deg"]["median"]:.3f} | {q["translation_cm"]["median"]:.3f} | {q["ADDsym_AUC_full"]:.5f} |')
    lines += ['', '중앙오차/P90은 matched 관측점, PCK는 누락·매칭실패 벌점을 포함한 고정 GT 분모다. 과거 9점 표의 숫자와 직접 섞지 않는다.', '',
        '## 2. 반복부터 악화하는가? — 분할마다 다르다 (CASE B, 제한된 범위)', '',
        f'합성 heldout RAW 중앙오차 PASS0→3: '+ ' → '.join(f'{held["passes"][str(p)]["matched_pooled_corner8_median_px"]:.3f}' for p in range(4))+' px.',
        '합성에서는 1회 개선 후 반복으로 다시 악화한다. 실사 DEV319는 반복 시 일부 지표가 더 좋아져 “반복은 항상 실패”라고 말할 수 없다. 위 표의 2D/6D 전체를 함께 본다. 어떤 pass도 DEV 결과로 선택하거나 최종 방법으로 승격하지 않았다.', '',
        f'실사 연속 변위 reversal은 d1↔d2 {100*r["oscillation"]["pairs"]["d1_d2"]["reversal"]:.2f}%, d2↔d3 {100*r["oscillation"]["pairs"]["d2_d3"]["reversal"]:.2f}%다. 즉 대부분이 왕복 진동하는 것은 아니다. 같은 방향의 누적 이동/지나친 보정도 오류 증가를 만들 수 있다. 반복 입력 분포 변화와 부합하지만 OOD 원인의 인과 증명은 아니다.', '',
        '## 3. 1% cap이 큰 오차 복구의 주된 병목인가? — 아니다 (CASE A는 국소 효과)', '',
        f'실사 PASS1 cap-hit는 전체 matched 코너 {100*r["movement"]["1"]["cap_hit_corner_fraction"]:.2f}%, 초기>20px 코너 {100*h["cap_hit_fraction"]:.2f}%다. RAW 이동 중앙값 {r["movement"]["1"]["raw"]["median"]:.3f}px, P90 {r["movement"]["1"]["raw"]["P90"]:.3f}px. 같은 입력에서 raw는10px이하로 복구하지만 cap만 그 복구를 막은 PASS1 코너는 seed평균 {r["movement"]["1"]["same_input_cap_blocked_recovery"]:.1f}개다.',
        'cap이 개별 이동을 제한하는 것은 사실이나 cap을 풀면 큰 실사 오류가 대량 복구된다는 설명은 지지되지 않는다. 반복 경로 차이와 섞지 않은 same-input RAW/CAPPED 비교를 사용했다.', '',
        '## 4. raw부터 잘못되거나 충분히 못 움직이는가? — 큰 오류에서는 여전히 그렇다 (CASE D)', '',
        f'초기>20px matched 코너 {n}개 중 RAW 1회 후 ≤10px 복구는 seed별 {recovery}개다. 큰 오류에서 평균 변화는 {h["delta_from_R0"]["mean"]:.3f}px이고 {100*h["regression_fraction"]:.2f}%는 더 나빠진다. 전체 작은 오차 개선과 큰 오류 복구 능력은 다른 질문이다. “PoseFix가 전혀 학습하지 못했다”는 결론도 틀리다.', '',
        '## 5. 대칭 번호 문제 때문인가? — 주된 원인이라는 증거 부족 (CASE C main 미충족)', '',
        f'실사 matched311장 중 fixed평균>20px이지만 symmetry평균≤10px인 phase-only 사례는 seed별 {phase}장. C4 DEV155에서는 {greenphase}장이지만 대부분 R0에도 이미 있던 번호 차이다. 대칭 평가는 필요하지만 이를 fixed-index 학습이 실패 원인이라는 causal proof로 사용할 수 없다. 부분 코너 swap/중간좌표 평균화는 전체물체 phase-only 검사로 배제할 수 없다.', '',
        'C4 RAW PCK10 PASS0→3: '+' → '.join(f'{100*square["passes"][str(p)]["PCK"]["10"]:.3f}%' for p in range(4))+'. 중앙값만 좋아졌다는 이유로 전체 개선을 주장하지 않는다.', '',
        '## 6. 단순 bit-exact 문제뿐인가? — 아니다 (CASE E 수치 현상은 별도)', '',
        f'동일 입력 10회×3seed 모두 첫 차이는 up1. crop 최대 차이 {result["numerical_crop_max"]:.9f}px, 원본 좌표 최대 차이 {result["numerical_restored_max"]:.9f}px. 모두 기존 crop 절대 component 허용치0.0003px 이내다. bit-exact는 아니지만 이 규모로 실제 정확도 실패를 설명할 수 없다. 허용치는 변경하지 않았다.', '',
        '## 물체별 결과 — 치수 부재의 인과 증거는 아님', '',
        '| object | n | W,D,H m | RAW med PASS0→1→2→3 px | PASS1 cap-hit % |', '|---|---:|---|---|---:|']
    for name,obj in objects.items():
        lines.append(f'| {name} | {obj["frames"]} | {obj["WDH"]} | '+ ' → '.join(f'{obj["passes"][str(p)]["median_px"]:.3f}' for p in range(4))+f' | {100*obj["cap_hit_PASS1"]:.2f} |')
    lines += ['', '## 결정', '',
        '주된 남은 문제는 **큰 초기 오류에 대한 raw 보정 자체의 제한된 복구력**이다. cap 해제·무작정 반복·bit-exact 강제만으로 해결됐다고 볼 수 없다. 반복의 부작용은 합성/C4에서 확인되지만 실사 모든 지표에서 동일하지 않다. 물체별 차이만으로 dimension 입력 부재를 원인으로 확정하지 않는다.', '',
        'Phase B의 강한 main 대칭-모호성 기준은 충족하지 않아 추가 학습이나 scaffold를 시작하지 않는다. `TRAINING_PROPOSAL.md`는 NOT_TRIGGERED. 기존 모델 유지, 이번 결과는 진단용으로 보존한다. 전체 pass/seed·6D·이동 구간별 표와 시각자료는 `REPLAY_SUMMARY.md` 및 JSON/PNG에 있다.']
    assert not strong, 'Strong CASE C requires manual proposal/scaffold review; never launch training automatically'
    R.write(R.DOC/'FINAL_DIAGNOSIS.md','\n'.join(lines)+'\n')
    R.write(R.DOC/'TRAINING_PROPOSAL.md',f'# NOT_TRIGGERED\n\nPhase A strong CASE C gate: matched REAL_DEV phase-only fraction >=10% in at least2/3 seeds at PASS1 RAW. Observed counts {phase} /311 per seed. Gate not met.\n\nSquare C4 has {greenphase}/155 phase-only cases per seed, largely present in R0 already; not evidence that new symmetry supervision will repair the observed large errors. No scaffold or training is launched, no new split or square synthetic claim.\n\nA future explicitly authorized matched control would change only one whole-object GT correspondence chosen against R0, not per-corner Hungarian/reflection; retain center masking, initialization/order/optimizer/6000steps/batch. This is a conditional design constraint, not an executed experiment or a recommendation supported by strong main CASE C evidence.\n')
    print('FINAL_DIAGNOSIS_WRITTEN',result['cases'],flush=True)

if __name__=='__main__': build()
