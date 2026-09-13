"""Print an apply_patch payload; main applies it to new report files only."""
import numpy as np
from common import *

SENTENCES={
    'LINE_BIAS_DEVELOPMENT_SIGNAL':'Line-structured refinement retains development evidence beyond a matched generic local point refiner; independent confirmation is required.',
    'LOCAL_REFINEMENT_ONLY_SIGNAL':'Local refinement is useful, but the current evidence does not establish a line-specific inductive-bias contribution.',
    'GENERIC_POINT_REFINER_DOMINATES':'A generic point-local refiner matches or exceeds the structured line model; the line-specific architecture claim is closed.',
    'NO_REFINEMENT_SIGNAL':'Neither refinement provides sufficient controlled evidence for an ML contribution.'}

def report():
    verdict=read(B/'VERDICT.json');name=verdict['verdict'];last=SENTENCES[name]
    next_action=('다음 행동은 frozen 설정으로 새 independent real sessions confirmation을 수행하는 것 하나다.'
        if name=='LINE_BIAS_DEVELOPMENT_SIGNAL' else '다음 행동은 현재 line-specific architecture contribution 주장을 종료하는 것 하나다.')
    real=read(B/'PER_SEED_REAL.json')['methods'];paired=read(B/'REAL_PAIRED_STATISTICS.json')['comparisons']
    syn=read(B/'SYNTH_HELDOUT_RESULTS.json')['summaries'];runtime=read(B/'RUNTIME.json')
    mechanism=read(B/'MECHANISM_NORMAL_TANGENT.json')['groups']['overall']['all']
    select=read(B/'P_SELECTION.json');param=read(B/'PARAMETER_BUDGET_LOCK.json')
    mean=lambda arm,section,key:float(np.mean([real[f'{arm}{s}'][section][key] for s in (1,2,3)]))
    lines=['# 마지막 ML contribution 대조 실험','',f'최종 architecture 판정: `{name}`. DEV-only이며 independent confirmation·novelty 주장은 없다.','',
        '## 무엇을 비교했는가','',
        'R0는 기존 frozen YOLO26n이다. L은 기존 `image_line_only` seed 1/2/3이며 재학습하지 않았다. P는 구조 정보를 쓰지 않는 새 point-local voting head 하나다. 기존 line 실험 primary `image_joint`의 실패는 변경하지 않았다.','',
        f"P/L trainable params는 {param['P_params']:,}/{param['L_params']:,}개({param['relative_difference']*100:+.2f}%). 각각 8 roles × 221 non-null candidates × 32 samples × P3/P4로 113,152 spatial samples다. Null은 추가 sampling 없이 context를 사용한다. 파라미터·sample 수 일치는 FLOPs/latency 일치가 아니다.",
        '같은 cached FP16 P3/P4, 같은 55,915 matched usable train rows, 같은 seed별 96,000 exposures, 6,000 updates × 3 fits, batch16, FP32 head/no AMP, AdamW/warmup/cosine/clip을 재사용했다. 학습 source·목록·최종 sampler 상태로 old/new order parity를 검증했다.','',
        'P의 isotropic 4×8 patch와 L의 along-line evidence encoder는 다르다. P target은 Gaussian point displacement, L target은 bilinear line geometry/null이다. 동일 point validity를 사용하지만 edge length/support를 point mask로 오용하지 않는다. 기존 L에 C2 min-over-permutation 학습이 없어서 정본 index assignment를 그대로 사용했다. 이 차이는 matching 한계이며 새 symmetry-invariance 주장을 하지 않는다.','',
        '## Synthetic 선택과 heldout','',
        f"P T(seed1/2/3) = {[select['temperatures'][str(s)]['temperature'] for s in (1,2,3)]}; shared lambda={select['selected_rule']['lam']}, cap={select['selected_rule']['max_move_image_diagonal_fraction']}. 기존 T/lambda/cap grid와 tie rule만 사용했다. Selection은 기존 all-source-GT-denominator 8-corner capped/diagonal-normalized frame score이며, 아래 9kp heldout score로 바꾸지 않았다.",
        'Train55980/cal1004/selection1031/heldout1985의 원래 분할을 유지했다. P selection artifact를 저장한 뒤에만 heldout accuracy를 계산했고, real 결과로 조정하지 않았다. 원래 R0·과거 probe의 validation 재사용 때문에 전체 연구에 독립적인 test라는 뜻은 아니다.','',
        '| 모델 | synth9 median px | synth9 P90 px | PCK10/allGT | 원래8 selection-objective heldout |',
        '|---|---:|---:|---:|---:|']
    for model in ['R0','L1','L2','L3','P1','P2','P3']:
        x=syn[model];lines.append(f"| {model} | {x['nine']['pooled9_median_px']:.4f} | {x['nine']['pooled9_p90_px']:.4f} | {x['nine']['pck10_all_gt']:.4f} | {x['old8']['frame_score_mean']:.8f} |")
    lines+=['','Radius coverage, movement, null probability, point availability는 `SYNTH_HELDOUT_RESULTS.json`에 함께 보존했다. R0의 dormant P1 null 필드는 해당 없음으로, L의 null은 별도 계산해 `SYNTH_DIAGNOSTIC_FIELD_CLARIFICATION.json`에 명시했다. 초기 generic permutation 테스트의 C2 yaw 이름 오류와 학습 후 추가 정본 검사도 `C2_TEST_LABEL_CORRECTION.md`에 공개했다. 어느 정정도 학습/예측/선택을 바꾸지 않는다. 이 진단으로 radius/stencil/model을 다시 설계하지 않았다.','',
        '## Real DEV319: 고정한 일회 비교','',
        'P 3 seeds 각각 positive319+negative2689에서 실제 image inference를 수행했다. 모든 candidate의 box/score/order, 최고 score 선택, nonselected points와 center8 보존을 exact 검사했다. Negative detection/AP/ranking은 동일하다. 실제 negative P 점 출력도 별도 저장하되, 기존 canonical scorer는 negative point supervision이 없으므로 그 부분만 baseline raw cache를 재사용한다.','',
        '| 모델 | 9kp median px | P90 px | frame mean px | gross20 | rotation med ° | translation med cm | IoU3D med | ADDsym AUC |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for model in ['R0','L1','L2','L3','P1','P2','P3']:
        x=real[model]['2d'];p=real[model]['6d'];lines.append(f"| {model} | {x['keypoint_location_median_px']:.4f} | {x['keypoint_location_p90_px']:.4f} | {x['frame_mean_px']:.4f} | {x['gross20']:.4f} | {p['rotation_median_deg']:.4f} | {p['translation_median_cm']:.4f} | {p['iou3d_median']:.4f} | {p['add_sym_auc']:.4f} |")
    primary=paired['keypoint_location_median_px'];ci=primary['session_cluster'];fc=primary['frame_level']
    lines+=['',f"Primary seed-mean L−P median difference = {primary['difference']:+.6f}px; paired13-session 95% CI [{ci['low']:+.6f}, {ci['high']:+.6f}], paired-frame CI [{fc['low']:+.6f}, {fc['high']:+.6f}]. 10,000 draws, seed={STAT_SEED}. 각 draw에서 같은 frame/session ID를 모든 seed에 적용하고, seed별 metric(L)-metric(P)의 평균을 계산했다. R0를 3배 독립 표본으로 세지 않았다.",
        f"L/P seed-mean P90 = {mean('L','2d','keypoint_location_p90_px'):.6f}/{mean('P','2d','keypoint_location_p90_px'):.6f}px; gross20 = {mean('L','2d','gross20'):.6f}/{mean('P','2d','gross20'):.6f}. L/P median = {mean('L','2d','keypoint_location_median_px'):.6f}/{mean('P','2d','keypoint_location_median_px'):.6f}px; R0={real['R0']['2d']['keypoint_location_median_px']:.6f}px.",
        'Proj@5/10/20는 supervised keypoint 오차가 threshold 이내인 비율이다. 모든9점이 모든 프레임에서 supervised인 것은 아니다. Canonical evaluator/GT/reference를 그대로 사용하고 full-precision point 오류를 재계산해 기존 JSON 수치와 일치시켰다. Yaw·coverage·Proj는 `PER_SEED_REAL.json`에 포함했다. Lateral/depth는 기존 frozen per-frame pose export에 없어 새 지표를 추가하지 않았다.','',
        '| Gate | 결과 |','|---|---|']
    lines += [f"| {k} | {'PASS' if v else 'FAIL'} |" for k,v in verdict['gates'].items()]
    lines+=['','G5는 6D 우월성 확증이 아니라 downstream 안전성이다. 아래 차이는 모두 L−P다.','',
        '| Downstream | seed-mean difference | session95% low | high | L better direction | clear L harm |',
        '|---|---:|---:|---:|---|---|']
    for metric in old('aggregate_results').POSE_METRICS:
        r=paired[metric];c=r['session_cluster'];lines.append(f"| {metric} | {r['difference']:+.6f} | {c['low']:+.6f} | {c['high']:+.6f} | {verdict['downstream_directions_L_better'][metric]} | {verdict['downstream_clear_L_harm'][metric]} |")
    lines+=['','## Mechanism 및 비용','',
        f"GT-assisted edge-endpoint 평균 normal error의 L−P median delta={mechanism['L_minus_P']['normal']['median']:+.6f}px; tangent delta={mechanism['L_minus_P']['tangent']['median']:+.6f}px. 이는 기전 진단일 뿐 primary/gates를 뒤집지 않는다. Role/day-night/material/session 및 사전 고정 R0 mean error ≤5 / 5–10 / >10 px subgroup을 전부 보존했다.",
        'Visibility는 annotation code에 따른 기술적 subgroup이며 physical observed/occluded provenance를 독립 검증하지 못했다. 물리적 occlusion robustness 주장은 하지 않는다.','',
        'Runtime은 기존과 같은26장·5 warmup·3 repeats·batch1이며 baseline/integrated를 교대로 측정했다. BGR→reflect padding→YOLO→refinement→원본2D좌표, 동기화된 wall latency다. File decode/model load/PnP는 제외했다. Timing output과 accuracy output의 좌표도 exact 비교했다.','',
        '| 모델 | median ms | mean ms | P90 ms | paired added median ms | head used |','|---|---:|---:|---:|---:|---:|']
    for model,r in runtime['by_run'].items():
        x=r['integrated_ms'];lines.append(f"| {model} | {x['median']:.3f} | {x['mean']:.3f} | {x['p90']:.3f} | {r['added_ms']['median']:+.3f} | {r['head_used_fraction']:.3f} |")
    base=runtime['R0'];lines += [f"| R0 paired observations | {base['median']:.3f} | {base['mean']:.3f} | {base['p90']:.3f} | 0 | 0 |",'',
        '비용 수치는 현재 GPU/환경의 측정치다. Runtime 시작부에 CPU 점수 계산이 병행되어 CPU 부하가 완전히 격리되지는 않았다(`RUNTIME_CONDITIONS_NOTE.json`). 따라서 관측 wall-time 차이를 통제된 인과적 속도 향상으로 주장하지 않는다. 반복을 골라내거나 더 빠른 값을 얻기 위한 재측정은 하지 않았다. FLOP 일치도 주장하지 않는다. GT-assisted mechanism 결과만으로 line 필요성을 증명할 수 없다.','',
        '## 감사와 종료','',
        '최종 source/체크포인트/seed-order/metric/bootstrap/runtime/출력 보존 감사는 `FINAL_AUDIT.json`을 따른다. Audit PASS는 performance PASS가 아니다. 원래 task-risk/active/line 결과 및 paper/final은 수정하지 않았다.','',
        'Line-specific gates 전체를 통과할 때만 새 independent real sessions confirmation이 다음 단계다. 그 외에는 현재 line-specific claim을 닫고 추가 DHT/Hough/module/score 구조 탐색은 하지 않는다. Generic P를 자동으로 새로운 proposed novelty로 승격하지 않는다.','',last]
    btext='\n'.join(lines)+'\n'
    mech=['# GT-assisted normal/tangent diagnostic','',
        'Each GT edge defines its unit tangent/normal. Absolute projected endpoint errors are averaged over the two endpoints before reporting distributions. Shared corners therefore occur in multiple edge roles; these are not independent observations.','',
        '| Model | normal median | normal mean | normal P90 | tangent median | tangent mean | tangent P90 |','|---|---:|---:|---:|---:|---:|---:|']
    for model,x in mechanism['methods'].items():
        mech.append('| '+model+' | '+' | '.join(f"{x[a][s]:.6f}" for a in ('normal','tangent') for s in ('median','mean','p90'))+' |')
    mech += ['',f"Seed-mean L−P normal median delta: {mechanism['L_minus_P']['normal']['median']:+.6f}px; tangent median delta: {mechanism['L_minus_P']['tangent']['median']:+.6f}px.",
        'All role, difficulty, day/night, material, session and annotation-visibility subgroups are in MECHANISM_NORMAL_TANGENT.json. Difficulty is fixed from R0 frame-mean supervised error (≤5, 5–10, >10px). No subgroup search or new gate is used.',
        'Numeric annotation visibility codes do not independently establish physical observability. No physical occlusion mechanism claim is made. The primary paired real statistic and safety gates govern the verdict; this diagnostic cannot override them.']
    final=['# 최종 의사결정','',
        f"시작 commit: `{START}` (`main`). 기존 결과를 보존한 별도 최종 감사/대조 실험이다.",'',
        '## A: task-risk failure 감사','',
        '`TASK_RISK_AL_NO_SIGNAL` 유지. 기존 QA flag가 평가145장 전부에 있어 QA_CLEAN-only S1은 NOT_ESTIMABLE이다. QA flag를 근거로 문제 프레임 하나만 제거해 성공으로 바꾸지 않는다. 감사 판정은 `TASK_RISK_RESULT_QA_SENSITIVE_REQUIRES_CAUTION`이다.',
        '문제 프레임은 원본640×480 밖 x≈698–740의 padding-region 검출이 세 seed 모두 최고 score로 선택되었다. 2D 점 오차는 약286px이고 두 기존 치수 가설 모두 약27–32m 위치 오류다. 선택된 위치의 오류가 이미 존재하고 PnP에서 증폭되며, C2 순서·축 선택만으로 회복되지 않는다. 후보 교체/selector/padding 필터는 적용하지 않았다. Reference provenance는 제한사항이나 제안 모델만의 실패가 GT 오류 때문이라는 근거는 아니다.','',
        '## B: 마지막 controlled ML architecture 비교','',
        f"판정: `{name}`. Primary L−P={primary['difference']:+.6f}px, 13-session95% CI [{ci['low']:+.6f}, {ci['high']:+.6f}]. Gates: {verdict['gates']}.",
        'R0/L을 재학습하지 않았고 P만 seeds1/2/3×6000steps 학습했다. Synthetic 선택 후 실제 DEV319+negative2689 추론, 기존 evaluator, paired10k bootstrap, 기전 및 runtime 감사를 수행했다. 상세: `B_line_vs_point/REPORT_KO.md`.','',
        next_action+' 추가 DHT/Hough/selector/active score/self-training/module 탐색은 허용되지 않는다. Generic P의 novelty도 별도 입증 없이 주장하지 않는다.','',
        'Git은 main-only로 새 code/docs만 commit/push한다. 최종 commit의 자기참조 SHA를 보고서에 억지로 기록하지 않고, push 후 raw `GIT_PUSH_RECEIPT.json`과 CLI에서 local/origin/main 정확 일치를 확인한다. 기존 capacity_screen 미추적 폴더는 본 커밋에서 제외한다.','',last]
    patches={B/'REPORT_KO.md':btext,B/'MECHANISM_REPORT.md':'\n'.join(mech)+'\n',DOC/'FINAL_DECISION_KO.md':'\n'.join(final)+'\n'}
    print('*** Begin Patch')
    for path,text in patches.items():
        assert not path.exists(),path
        print('*** Add File: '+str(path))
        for line in text.splitlines():print('+'+line)
    print('*** End Patch')

if __name__=='__main__':report()
