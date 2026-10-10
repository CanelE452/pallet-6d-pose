"""Korean joint-refinement experiment report from locked public evidence only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
DOC=ROOT/'_docs/experiments/pallet_feature_gradient_joint_20261010'
METHODS=('BASE','N3_DIM_SYM','SUBPIX','N3_THEN_SUBPIX','JOINT_FIXED_ISOTROPIC','FG_JOINT_POSTERIOR')
METRICS=('corner_px','translation_cm','rotation_deg','ADDsym_cm')
COMPARATORS=('N3_THEN_SUBPIX','N3_DIM_SYM','SUBPIX','JOINT_FIXED_ISOTROPIC')
CODE='../../../scripts/research/pallet_feature_gradient_joint_20261010'


def read(doc,name,default=None):
    p=doc/name;return json.loads(p.read_text()) if p.exists() else default


def f(value):return '없음' if value is None else f'{value:.4f}'
def ci(interval):return '미산출' if interval is None else f'[{f(interval[0])}, {f(interval[1])}]'
def cellstat(stat,interval=True):return f"{f(stat['mean'])} ± {f(stat['sample_std'])}"+(f"; {ci(stat['CI95'])}" if interval else '')


def table(headers,rows):
    def escape(value):return str(value).replace('|',r'\|').replace('\n','<br>')
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(escape(v) for v in row)+' |' for row in rows])


METHOD_TEXT='''# Feature-Guided Joint Keypoint Refinement: 고정 계산과 전제

FG-JKR는 실험용 이름이다. N3와 영상 기울기를 같은 위치 목적함수에 넣는 **새 inference-only 융합 구현**이며, 학술적 최초·보편 우위를 주장하는 확정 논문명이 아니다. 기존 N3→OpenCV cornerSubPix는 직렬 후처리 기준선으로 그대로 보존했다. 이번 joint를 공동 학습된 N3 또는 새 end-to-end 모델로 부르지 않는다.

![기존 직렬과 공동 위치 계산](figures/01_joint_method.png)

입력은 원영상 grayscale I, 원래 Base q0[9,2], 고정 N3 qN[9,2], N3 logits[8,222], 원래 network 후보 이동[222,2], gain, prediction_support[9], point_support[8], 고정 seed별 temperature다. 참조/R,t/사람 visibility는 입력이 아니다. [fusion.py](../../../scripts/research/pallet_feature_gradient_joint_20261010/fusion.py)의 `joint_refine`에는 해당 입력을 받을 인자 자체가 없다.

## 222 후보에서 prior 공분산 대리량

221개 non-null 및 마지막 zero/null 후보를 모두 보존한다. 후보 순서와 마지막(0,0)은 실제 모델 buffer로 검산했다. bbox diagonal이 이미 곱해진 `candidate_displacements_network`를 기존 canvas/letterbox의 gain으로 나눈다. 원래 네트워크 prediction의 pad/offset을 이동에 다시 더하지 않는다.

```text
p_j = softmax(logit_j / temperature_s),  j=0,...,221
d_j = candidate_displacements_network[j] / gain
m = Σ_j p_j d_j
C = Σ_j p_j (d_j−m)(d_j−m)^T
μ = qN                         # 저장된 기존 cap 후 native N3 좌표
c = 0.01 × hypot(raw_width, raw_height)
Σ = eig_reconstruct(C + 1 px² I, eigenvalues clipped to [1 px², c²])
```

공분산은 **cap 전 후보 퍼짐의 대리량**이다. prior 중심은 기존 cap 후 qN이고 q0+m이 아니다. 따라서 Σ를 capped posterior의 정확한 covariance 또는 calibration된 confidence라고 해석하지 않는다. 큰 C가 실제 오류를 뜻한다거나 entropy가 신뢰도 정답이라고 확정하지 않는다. 방향 고유벡터는 유지하고 floor/ceiling만 적용한다. 확률·공분산·정상방정식 집계는 float64다.

## 원영상의 고정 창 기울기 제약

원영상 gray는 float32 [0,1]이다. `cv2.Sobel` ksize=3의 x/y 미분을 원영상 native pixel 격자에서 계산한다. integer (x,y)는 `image[y,x]` 픽셀 중심이며 half-pixel offset을 더하지 않는다. Sobel과 `cv2.remap(..., INTER_LINEAR)`은 모두 `BORDER_REFLECT_101`로 고정했다. INTER_LINEAR는 OpenCV의 interpolation table 정밀도를 그대로 사용한다.

qN 중심의 dx,dy∈−5,...,5, 총11×11 위치 x_i를 한 번 정의한다. 새로운 위치를 찾을 때 창을 반복해서 옮기지 않는다. 시작 qN이 이미지 밖이면 반사 창을 만들어 중심 자체를 안으로 복구하지 않고 N3 fallback을 사용한다. 중심이 안이고 창 일부만 밖이면 고정 반사 정책으로 샘플한다.

```text
x_i = qN + (dx_i,dy_i)
g_i = bilinear samples of native Sobel x/y derivatives at x_i
w_i = exp(−||x_i−qN||² / (2 × 2.5²))
S = Σ_i w_i ||g_i||²
A = (Σ_i w_i g_i g_i^T) / S
b = (Σ_i w_i g_i g_i^T x_i) / S
```

S<1e−4이면 영상 정보가 부족해 q_unconstrained=qN이다. 충분하면 trace(A)≈1로 정규화한다. 이는 OpenCV cornerSubPix의 기울기 직교성 원리를 참고한 **고정 창 근사 제약**이며 그 함수 내부 구현과 동일하지 않다. [OpenCV 4.9.0 공식 설명](https://docs.opencv.org/4.9.0/dd/d1a/group__imgproc__feature.html).

## 하나의 공동 위치 결정식

```text
E_joint(q) = (q−μ)^T Σ^−1 (q−μ)
             + (q^T A q − 2 q^T b + constant) / (4 px)²
q_unconstrained = solve(Σ^−1 + A/16, Σ^−1 μ + b/16)
```

precision은 `np.linalg.solve(Σ,I₂)`, 최종 식은 `np.linalg.solve(system,rhs)`로 계산한다. `np.linalg.inv`를 호출하지 않는다. 2×2 SPD 고유값과 조건수≤1e12를 검사한다. rank1 edge의 A가 한 방향만 제약해도 prior precision이 다른 방향을 안정화한다. A=0,b=0이면 계산 반올림을 추가하지 않고 qN을 정확히 반환한다.

`JOINT_FIXED_ISOTROPIC`은 **Σ만 16I₂로 교체**한다. 동일한 p/C 진단, μ=qN, gray/Sobel/샘플창/A/b, solver와 최종 cap을 사용한다. posterior covariance를 제거하는 유일한 ablation이며 좋은 결과의 방법을 사후 primary로 바꾸지 않았다.

마지막에 q0 중심 반경 c로 한 번 투영한다. `Δ=q_unconstrained−q0`, `qFinal=q0+Δ min(1,c/||Δ||)`이며 0 이동은 그대로 유지한다. N3 주변 추가1%가 아니다. center8과 unsupported/sentinel/nonfinite 입력은 원래 q0로 보존한다. image/posterior/SPD fallback은 cap 전 qN을 유지하고 같은 최종 q0 cap을 적용한다. 과거 N3 float32 미세 상한 초과는 이 엄격 cap에서만 제거될 수 있다.

검출 후보/selected_index/box/score/confidence/K/등록치수/대칭 계약은 wrapper에서 보존한다. canonical 문맥 치수는 [W,D,H], 기존 PnP API는 [W,H,D]다. 최종 F는 기존 prediction-only W/D 가설 선택, SQPnP 및 RefineLM 그대로다. joint 출력에 cornerSubPix를 추가하지 않았고 완성된 두 좌표를 평균하지 않았다.

[FUSION_METHOD_LOCK.json](FUSION_METHOD_LOCK.json)은 새 DEV 오차를 읽기 전에 설정과 사례 선택 규칙을 봉인했다. [UNIT_TESTS.json](UNIT_TESTS.json)의 8개 테스트는 합성 숫자 배열만 사용하며, 평탄 영상, 단일 방향 edge의 SPD, 경계 반사와 픽셀 중심, null과 높은 entropy, native gain, 결측과 중심 보존, 총 이동 cap, isotropic의 Sigma 단독 교체를 검사한다. cornerSubPix와 explicit inverse를 호출하면 예외가 나는 monkeypatch에서도 joint가 통과했다.

`POSTERIOR_CAPTURE.jsonl.gz`는 실제 고정 N3 forward에서 얻은 logits/후보/gain/q0/qN이다. `NEW_COORDINATES_SEALED.jsonl.gz`는 참조를 읽기 전에 저장한 두 joint의 q0/qN/qS/qFinal 및 정상방정식이다. 새 방법의 **qS는 joint의 unconstrained 해**이며 SUBPIX 좌표가 아니다. 실제 F는 qFinal만 받는다. `PREDICTIONS.jsonl.gz`의 actual_pose는 그 F가 반환한 R/t이고 pose/corner는 평가 후 부가한 오차다.

prior/영상/solver 레코드는 `correction.diagnostics.corner_records[k]`의 `C`, `Sigma`, `A`, `b`, `S`, `posterior_entropy`, `null_probability`, `probability_sum`, `q_unconstrained`, `q_final`, `fallback_reason`, `cap_active`로 추적한다. 공분산과 영상 방향 그림은 이 실제 배열에서 생성하며 독립 물리 측정이나 인과 기전의 증거가 아니다.
'''


REPRO_TEXT='''# 재현: 기존 환경과 공개 원행

설치·재학습·합성/실사 데이터 생성 없이 기존 환경을 사용한다. 정확도 실행은 기존 비공개 입력이 필요하고, 공개 원행의 통계·그림 검산은 원본 RGB/checkpoint/GPU 없이 가능하다.

```bash
export PALLET_PYTHON="/path/to/existing/pallet-pose/bin/python"
export PALLET_SOURCE_ROOT="/path/to/original/pallet-pose"
export PALLET_BASELINE_ROOT="/path/to/immutable/a22-baseline-worktree"
export PALLET_JOINT_PRIVATE="/path/to/private/replay-records"
export PALLET_PRIVATE_OUTPUT="$PALLET_JOINT_PRIVATE"
export PALLET_JOINT_OUTPUT="/path/to/new-joint-output"
export PYTHONDONTWRITEBYTECODE=1
```

게시된 입력/출력을 덮어쓰지 않고 **새 출력 디렉터리**로 전체 Gate A→B→C를 재실행:

```bash
"$PALLET_PYTHON" -B -m scripts.research.pallet_feature_gradient_joint_20261010.replay --source-root "$PALLET_SOURCE_ROOT" --private-dir "$PALLET_JOINT_PRIVATE" --output "$PALLET_JOINT_OUTPUT"
"$PALLET_PYTHON" -B -m scripts.research.pallet_feature_gradient_joint_20261010.review_pose
"$PALLET_PYTHON" -B -m scripts.research.pallet_feature_gradient_joint_20261010.visibility
"$PALLET_PYTHON" -B -m scripts.research.pallet_feature_gradient_joint_20261010.verify_posterior --docs "$PALLET_JOINT_OUTPUT"
"$PALLET_PYTHON" -B -m scripts.research.pallet_feature_gradient_joint_20261010.verify_auxiliary --docs "$PALLET_JOINT_OUTPUT"
"$PALLET_PYTHON" -B -c 'from scripts.research.pallet_feature_gradient_joint_20261010.finalize import mechanism_proof; mechanism_proof()'
"$PALLET_PYTHON" -B -m scripts.research.pallet_feature_gradient_joint_20261010.summarize --doc "$PALLET_JOINT_OUTPUT"
"$PALLET_PYTHON" -B -m scripts.research.pallet_feature_gradient_joint_20261010.figures --doc "$PALLET_JOINT_OUTPUT"
"$PALLET_PYTHON" -B -m scripts.research.pallet_feature_gradient_joint_20261010.verify --doc "$PALLET_JOINT_OUTPUT" --require-figures
"$PALLET_PYTHON" -B -m scripts.research.pallet_feature_gradient_joint_20261010.report --doc "$PALLET_JOINT_OUTPUT"
"$PALLET_PYTHON" -B -c 'from scripts.research.pallet_feature_gradient_joint_20261010.finalize import artifact_manifest; artifact_manifest()'
```

replay 내부 순서는 `preflight`(원본 hash와 metadata 인증) → `capture`(고정 N3의 seed별 319회 forward 및 native parity) → `test_fusion`(합성 배열을 이용한 8개 단위검사와 실행 영수증) → `coordinates`(16장 smoke와 두 방법의 1,914개 좌표를 참조 읽기 전에 봉인) → `evaluate`(각 새 좌표에 실제 F를 1회 호출)다. `UNIT_TESTS.json`과 방법 lock을 새 출력에 저장하고 기존 코드의 수치 설정은 변경하지 않는다. 아래 명령은 단위검사만 실행하므로 전체 재현에 필요한 실행 영수증은 replay가 생성한다.

```bash
"$PALLET_PYTHON" -B -m scripts.research.pallet_feature_gradient_joint_20261010.test_fusion
```

공개 원행만으로 검산·재집계·PNG 재생성:

```bash
export PALLET_PUBLIC_DOC="_docs/experiments/pallet_feature_gradient_joint_20261010"
mkdir -p "$PALLET_JOINT_PRIVATE"
"$PALLET_PYTHON" -B -m scripts.research.pallet_feature_gradient_joint_20261010.verify_posterior --docs "$PALLET_PUBLIC_DOC" --output "$PALLET_JOINT_PRIVATE/public_posterior_check.json"
"$PALLET_PYTHON" -B -m scripts.research.pallet_feature_gradient_joint_20261010.verify_auxiliary --docs "$PALLET_PUBLIC_DOC" --output "$PALLET_JOINT_PRIVATE/public_auxiliary_check.json"
PALLET_JOINT_OUTPUT="$PALLET_PUBLIC_DOC" "$PALLET_PYTHON" -B -m scripts.research.pallet_feature_gradient_joint_20261010.review_pose
"$PALLET_PYTHON" -B -m scripts.research.pallet_feature_gradient_joint_20261010.summarize --doc "$PALLET_PUBLIC_DOC"
"$PALLET_PYTHON" -B -m scripts.research.pallet_feature_gradient_joint_20261010.figures --doc "$PALLET_PUBLIC_DOC"
"$PALLET_PYTHON" -B -m scripts.research.pallet_feature_gradient_joint_20261010.verify --doc "$PALLET_PUBLIC_DOC" --require-figures
"$PALLET_PYTHON" -B -m scripts.research.pallet_feature_gradient_joint_20261010.report --doc "$PALLET_PUBLIC_DOC"
```

독립 검산기의 `--docs`는 공개 자료 디렉터리, `--output`은 새 검산 영수증 파일이다. 기존 파일을 덮어쓰지 않는 배타적 생성 방식이므로 다시 실행할 때는 새로운 출력 파일명을 지정한다. 통계와 그림 생성기의 옵션은 단수형 `--doc`이다. 저장된 자료만 검산할 때에는 비공개 RGB, checkpoint, CUDA가 필요하지 않다.

위의 전체 재현은 새로운 출력에 posterior와 보조 검산 영수증, `MECHANISM_PROOF.json`을 생성한다. `finalize.mechanism_proof()`는 저장된 좌표의 차이와 계산 계약만 검산하며 추론이나 F를 다시 호출하지 않는다. 게시용 `finalize`의 main은 이번 실행 세션의 START와 원래 checkout snapshot을 요구하므로 일반 재현에서는 호출하지 않는다. 게시 세션 전용 SHA manifest와 원격 게시 영수증도 일반 정확도 재현과 구분한다.

원본 의존성은 N3 seed 1·2·3의 last.pt, REAL_DEV의 기존 N3 좌표 JSON, detector/neck feature cache, 원영상 RGB, K와 등록 치수, 기하 참조다. [INPUT_LOCK.json](INPUT_LOCK.json)의 `models`, `input_manifest`, `source_bindings`에 상대 경로와 SHA256이 있다. checkpoint `data/pallet/results/pallet_dim_conditioned_p_v1/runs/N3_DIM_SYM_seed{1,2,3}/last.pt`와 예측 `predictions/REAL_DEV/N3_DIM_SYM_seed{1,2,3}.json`은 기존 TRAINING_COMPLETE/DEV_INFERENCE_COMPLETE 기록과 대조한다. logits가 cache에 있다고 가정하지 않고 기존 feature cache에서 N3만 다시 forward한다. 누락, 변조, native 복원 불일치가 있으면 BLOCKED로 종료하며 새 분포, 가중치, GT 보정으로 채우지 않는다.

정확도 재실행의 N3 capture에는 CUDA GPU와 기존 torch 2.1.1+cu118 환경이 필요하다. joint, 기존 F, 통계는 CPU 계산이며 OpenCV 4.9.0, NumPy 1.26.4, matplotlib 3.10.9를 사용했다. 추론 단계는 RGB와 추론 입력만 사용하며 참조는 좌표 봉인 후 평가에서만 읽는다. 공개 통계 검산은 표준 Python 통계 함수와 NumPy의 동일한 고정 bootstrap draw를 사용한다.

새 배포 파이프라인의 end-to-end latency는 측정하지 않았다. capture, joint, F의 실행 총시간을 검출부터의 배포 latency로 해석하거나 기존 직렬 15.8 ms를 JOINT의 시간으로 사용하지 않는다. 논문, LaTeX, PDF, 초록, PPT는 수정하지 않았다.
'''


def result(doc,m,p,ex,index,verification):
    sm=m['seed_mean']['ALL'];primary=p['seed_mean']['ALL']['FG_JOINT_POSTERIOR_minus_N3_THEN_SUBPIX']['statistics']
    dt=primary['translation_cm'];dr=primary['rotation_deg']
    verdict='TRADEOFF' if dt['mean_paired_difference']*dr['mean_paired_difference']<0 else 'FAIL'
    worse=[s for s in ('1','2','3') if any(p['by_seed'][s]['ALL']['FG_JOINT_POSTERIOR_minus_N3_THEN_SUBPIX']['statistics'][k]['mean_paired_difference']>0 for k in ('translation_cm','rotation_deg'))]
    complete=(verification.get('status')=='PASS' and ex.get('new_F_calls')==1914 and
              verification.get('figures',{}).get('status')=='PASS' and len(index.get('figures',[]))==6)
    lines=['# FG-JKR: 실제 3-seed 동일 REAL_DEV 평가','',
        f"[판정] JOINT vs N3→SubPix: {'상충' if verdict=='TRADEOFF' else '악화'}. 전체 319×3 seed 평균 ΔT={f(dt['mean_paired_difference'])} cm, ΔR={f(dr['mean_paired_difference'])}°, 두 95% CI={ci(dt['CI95'])}/{ci(dr['CI95'])}; 더 나쁜 seed={','.join(worse)}.",
        '직렬 대비 위치 오차 평균이 늘고 회전 오차 평균이 줄었으며 두 세션 CI 모두 0을 포함한다. 세 seed의 개선 방향도 일치하지 않아 확정 개선으로 선언하지 않는다. 결과를 보고 융합기, 상수, 주비교를 교체하지 않고 **상충으로 종료**했다.','',
        f"실행 상태 **{'COMPLETE' if complete else 'PARTIAL'}**, 과학적 판정 **{verdict}**. 새 학습 0회, 새 합성 RGB 0장, 새 촬영 및 수동 주석 0건이다. 기존 기준선의 보고서, 수치, 그림, 원고를 보존했다.",'',
        '## 실제 공동 계산','',
        '고정 N3의 222개 후보 분포로 native 이동 공분산을 계산하고, qN을 평균으로 하는 prior와 원영상의 고정 11×11 창에서 얻은 기울기 A,b를 **하나의 2×2 선형방정식**에 넣었다. μ=qN을 유지하고 Σ만 16I로 바꾸는 isotropic 비교도 실행했다. 수식, 스케일, 고정 수치, 실제 스키마는 [METHOD_KO.md](METHOD_KO.md)에 있다.','',
        '[MECHANISM_PROOF.json](MECHANISM_PROOF.json)은 실제 저장 좌표를 비교했다. 각 seed의 319장 모두에서 FG joint는 기존 직렬 및 isotropic의 최종 좌표와 1e−10 px 초과로 달랐다. source AST와 cornerSubPix 예외 monkeypatch도 통과했다. 이는 새 계산을 실제로 실행했다는 증거이며 정확도 우위나 학술적 최초성의 증거로 해석하지 않는다.','',
        '![실제계산차이](figures/01_joint_method.png)','',
        '## 자료·분모·불변계약','',
        '319장, 13세션이며 matched 영상은 311장, 참조 코너는 2,499개, 관측 코너는 2,445개다. clean 153장, moderate 92장, severe 74장을 모두 유지했다. 난도 등급은 주석 범주이며 물리적 가림률이 아니다. 이 319장은 반복 개발 DEV로 독립 holdout이 아니다. 참조는 2D 주석과 등록 치수로 재구성한 geometric proxy이며 독립 물리 계측 GT가 아니다. 중심 8, 미지원과 결측, 검출 선택, score, box, confidence, K, canonical 치수, proper 대칭 계약을 보존했다.','',
        '기존 네 방법의 3,828행을 공개된 고정 원행에서 재사용하고 새 두 방법에만 3 seed×319=1,914회 실제 F를 호출했다. 최종 5,742행에서 동일한 영상 ID를 유지했다. 참조 좌표, 사람 visibility, GT 가설 선택을 추론에 사용하지 않았다. 새 좌표를 먼저 SHA로 봉인한 뒤 참조를 읽었다.','',
        '## 전체 정확도: 평균±SD와95%CI','',
        'SD는 오차의 표본 산포(ddof=1)로 CI, 표준오차, seed 간 산포와 구분한다. 코너 오차는 관측된 2,445개 코너, 자세 오차는 실제 성공한 319장에 대해 계산했다. 큰 유한 오차를 clipping/winsorization하거나 영상과 seed를 제외하지 않았으며 실패를 0 또는 1e6으로 대치하지 않았다.','']
    scopes=[(s,m['by_seed'][s]['ALL']) for s in ('1','2','3')]+[('seed-mean',sm)]
    lines += [table(['seed','방법','T cm 평균±SD;CI','R ° 평균±SD;CI','ADDsym cm 평균±SD;CI','성공/실패'],
        [[s,method,*[cellstat(summaries[method]['metrics'][k]) for k in ('translation_cm','rotation_deg','ADDsym_cm')],
          f"{summaries[method]['pose']['available']}/{summaries[method]['pose']['failures']}"] for s,summaries in scopes for method in METHODS]),'',
        table(['seed','방법','코너px 평균±SD;CI','PCK@10','gross20수/비율','ADDsym-AUC'],
            [[s,method,cellstat(summaries[method]['metrics']['corner_px']),f(summaries[method]['corner']['PCK']['10']),
              f"{f(summaries[method]['corner']['gross20_count'])}/{f(summaries[method]['corner']['gross20_rate'])}",f(summaries[method]['ADDsym_AUC']['full'])] for s,summaries in scopes for method in METHODS]),'',
        'seed-mean 오차는 동일 ID와 동일 canonical 코너의 세 seed 오차를 먼저 평균한 319장 및 2,445개 코너의 분포다. R/t를 평균해 새 자세를 만들거나 957장의 독립 이미지로 취급하지 않았다. PCK, gross20, AUC는 세 seed별 임계 비율, 개수, AUC의 산술평균이므로 개수가 소수일 수 있다. 평균 오차에 임계값을 적용한 별도 값은 `threshold_of_seed_mean_errors`와 `AUC_of_seed_mean_errors`에 있다. AUC는 기존 ADDsym_normalized≤threshold를 0..0.1 범위의 1,001개 임계값에서 적분한다. 추가 5 cm/5° 성공률은 기존 계약과의 정합성이 확인되지 않아 **NOT_CONFIRMED**로 기록하고 숫자를 만들지 않았다.','',
        '![seed별자세](figures/02_seed_pose_comparison.png)','',
        '<details><summary>원행 재계산: 분산·중앙값·P90·최대</summary>','']
    for metric in METRICS:
        lines += [f'**{metric}**','',table(['seed','방법','n','표본분산','SD','median','P90','max'],
            [[s,method,summaries[method]['metrics'][metric]['n'],*[f(summaries[method]['metrics'][metric][k]) for k in ('sample_variance','sample_std','median','P90','max')]] for s,summaries in scopes for method in METHODS]),'']
    lines += ['</details>','',
        'ADDsym의 m→cm 변환에서 평균과 SD는 ×100, 분산은 ×10,000이다. 모든 정밀값과 등급별 CI는 [METRICS.csv](METRICS.csv)와 [METRICS.json](METRICS.json)에 있다.','',
        '## 공동 효과와 유일한 ablation','',
        '주비교는 Joint−기존 직렬이며 음수는 오차 감소다. 같은 영상의 공통 성공 짝을 비교했다. 원래 13개 세션의 paired cluster bootstrap을 10,000회(seed 20260917) 수행하고 동일한 draw를 모든 비교, seed, 등급에 사용했다. CI가 0을 포함하면 방향의 불확실성이 남으며 동등성의 증거도 아니다.','']
    pairs=[(s,p['by_seed'][s]['ALL']) for s in ('1','2','3')]+[('seed-mean',p['seed_mean']['ALL'])]
    lines += [table(['seed','Joint−기준','코너Δpx[CI]','TΔcm[CI]','RΔ°[CI]','ADDsymΔcm[CI]','공통자세/전체'],
        [[s,comp,*[f"{f(pp['FG_JOINT_POSTERIOR_minus_'+comp]['statistics'][k]['mean_paired_difference'])} {ci(pp['FG_JOINT_POSTERIOR_minus_'+comp]['statistics'][k]['CI95'])}" for k in METRICS],
          f"{pp['FG_JOINT_POSTERIOR_minus_'+comp]['coverage']['common_success']}/319"] for s,pp in pairs for comp in COMPARATORS]),'',
        '![짝CI](figures/03_paired_ci.png)','']
    isotropic=p['seed_mean']['ALL']['FG_JOINT_POSTERIOR_minus_JOINT_FIXED_ISOTROPIC']['statistics']
    lines += [f"직렬 대비 seed-mean 위치 중앙값도 {f(sm['N3_THEN_SUBPIX']['metrics']['translation_cm']['median'])} → {f(sm['FG_JOINT_POSTERIOR']['metrics']['translation_cm']['median'])} cm로 악화했고 ADDsym-AUC는 {f(sm['N3_THEN_SUBPIX']['ADDsym_AUC']['full'])} → {f(sm['FG_JOINT_POSTERIOR']['ADDsym_AUC']['full'])}로 낮아졌다. 코너 PCK@10은 {f(sm['N3_THEN_SUBPIX']['corner']['PCK']['10'])} → {f(sm['FG_JOINT_POSTERIOR']['corner']['PCK']['10'])}로 높지만 이것만으로 6D 자세 개선을 선언하지 않는다.",'',
        f"분포 Σ 대비 고정 16I의 추가 효과는 T {f(isotropic['translation_cm']['mean_paired_difference'])} cm {ci(isotropic['translation_cm']['CI95'])}, R {f(isotropic['rotation_deg']['mean_paired_difference'])}° {ci(isotropic['rotation_deg']['CI95'])}다. 두 CI 모두 0을 포함하므로 후보 분포의 방향과 퍼짐이 추가 정확도 이득을 준다고 확정하지 않는다. 이것이 실제 ablation 결과이며 좋은 사례 하나로 대체하지 않는다.",'',
        f"bootstrap draw SHA는 `{m['bootstrap']['draw_sha256']}`이다. 세 seed는 고정되어 있고 CI는 원래 세션의 변동만 재표집한다. 학습 seed 모집단의 불확실성을 세 seed로 확증하지 않았으며 다중비교 보정도 하지 않았다.",'',
        '## 모든 난도·코너 손상·fallback·가설','',
        table(['등급','방법','T cm 평균±SD','R ° 평균±SD','ADDsym cm 평균±SD','코너px 평균±SD','PCK10','gross20'],
            [[g,method,*[cellstat(m['seed_mean'][g][method]['metrics'][k],False) for k in ('translation_cm','rotation_deg','ADDsym_cm','corner_px')],
              f(m['seed_mean'][g][method]['corner']['PCK']['10']),f(m['seed_mean'][g][method]['corner']['gross20_rate'])]
             for g in ('clean','moderate','severe') for method in METHODS]),'',
        '![등급손상](figures/04_per_grade_damage.png)','']
    diag=m['diagnostics']['by_seed']
    lines += [table(['seed','새 방법','손상 vs Base/N3/직렬','회복 vs Base/N3/직렬','총 cap 코너/영상','fallback 사유/수','cap 전 N3부터 추가 이동 px 평균±SD'],
        [[s,method,'/'.join(str(diag[s][key][method]['good5_to_bad10']) for key in ('damage_BASE','damage_N3','damage_SEQUENTIAL')),
          '/'.join(str(diag[s][key][method]['bad20_to_good10']) for key in ('damage_BASE','damage_N3','damage_SEQUENTIAL')),
          f"{diag[s]['joint'][method]['cap_active_corners']}/{diag[s]['joint'][method]['cap_active_frames']}",
          json.dumps(diag[s]['joint'][method]['fallback_counts']),cellstat(diag[s]['joint'][method]['unconstrained_move_from_N3_px'],False)]
         for s in ('1','2','3') for method in ('JOINT_FIXED_ISOTROPIC','FG_JOINT_POSTERIOR')]),'',
        '손상은 before<5 px→after>10 px, 회복은 before>20 px→after≤10 px를 만족하는 동일 canonical 참조 코너로 판정했다. flat/image fallback에서는 cap 전 위치가 N3에 남는다. no_pose는 후단 자세를 산출하지 못한 영상이다. fallback을 자세 실패와 합치거나 0 오차로 계산하지 않는다. 모든 실패와 손상 ID는 [FAILURES.json](FAILURES.json)에, 상한 적용 ID는 [METRICS.json](METRICS.json)의 `diagnostics.by_seed.seed.joint.method.cap_active_ids`에 보존했다.','',
        table(['seed','Joint가설vs','전환/유지/미산출','T/R모두감소','T↓R↑','T↑R↓','모두증가'],
            [[s,comp,f"{diag[s]['hypothesis'][comp]['switch_frames']}/{diag[s]['hypothesis'][comp]['no_switch_frames']}/{diag[s]['hypothesis'][comp]['unavailable_frames']}",
              *[diag[s]['hypothesis'][comp]['pose_quadrants'].get(k,0) for k in ('BOTH_DECREASE','T_DECREASE_R_INCREASE','T_INCREASE_R_DECREASE','BOTH_INCREASE')]] for s in ('1','2','3') for comp in COMPARATORS]),'',
        'W/D 가설 전환은 기존 prediction-only 선택기의 사후 결과다. 각 전환 ID의 before/after 가설과 ΔT/ΔR은 `diagnostics.by_seed.seed.hypothesis.comparator.switch_records`에 있다. 전환을 자동으로 올바른 복구나 개선으로 판정하지 않는다.','']
    vis=read(doc,'VISIBILITY_RESULTS.json',{})
    if vis:
        lines += ['기존 사람이 작성한 코너 상태 주석도 **좌표 봉인 후 평가에서만** 읽었다. DIRECT_VISIBLE 1,776개, SELF_OCCLUDED 462개, EXTERNAL_OCCLUDED 218개, OUT_OF_FRAME 43개로 총 2,499개의 참조 코너다. 이는 앞의 영상 난도 등급과 다른 주석이며 물리 장치로 측정한 가림률이 아니다. 새 visibility classifier나 수동 주석을 추가하지 않았다.','',
            table(['seed','방법','기존코너상태','관측/참조','평균±SD px','PCK10','손상/회복vs직렬'],
                [[s,method,state,f"{v['observed_corners']}/{v['reference_corners']}",
                  f"{f(v['observed_error_px']['mean'])} ± {f(v['observed_error_px']['std'])}",f(v['PCK10']),
                  f"{v['damage_vs']['N3_THEN_SUBPIX']['good5_to_bad10']}/{v['damage_vs']['N3_THEN_SUBPIX']['bad20_to_good10']}"]
                 for s in ('1','2','3') for method in ('N3_THEN_SUBPIX','FG_JOINT_POSTERIOR')
                 for state,v in vis['by_seed'][s][method]['states'].items() if v['reference_corners']]),'',
            '누락 관측과 그룹별 median/P90/max는 [VISIBILITY_RESULTS.json](VISIBILITY_RESULTS.json)에 있다. 원래 주석을 읽은 시점과 좌표 봉인 순서는 [VISIBILITY_LABELS.json](VISIBILITY_LABELS.json) 및 [PUBLIC_AUXILIARY_VERIFICATION.json](PUBLIC_AUXILIARY_VERIFICATION.json)에서 확인한다.','']
    lines += ['## 사례와 큰 실패를 그대로 보존','',
        '원본 RGB의 배포 권한이 확인되지 않아 작업장 사진을 공개하지 않았다. 좌표 사례에는 방법 lock에서 미리 고정한 seed 1의 Joint−직렬 T 최소/최대, |ΔT cm|+|ΔR°| 최소, Joint 절대 T 최대 규칙을 적용했다. 동률은 ID 사전순으로 정했다. 평가 오차로 ID를 뽑되 규칙은 사전에 고정했으며 사례 선택을 전체 성능의 증거로 삼지 않는다. near-zero는 정확히 같다는 뜻이 아니다.','',
        '![사례](figures/05_qualitative.png)','',
        table(['사전선택범주','ID','seed','ΔT cm','ΔR °'],[[v['category'],v['id'],v['seed'],f(v['deltaT_cm']),f(v['deltaR_deg'])] for v in index.get('examples',[])]),'',
        '![공분산기울기축](figures/06_covariance_vs_gradient.png)','',
        '그림 06은 같은 사례의 지원 코너 중 실제 Joint−N3 이동이 가장 큰 코너를 확대한다. Σ의 sqrt(eigenvalue) 타원은 confidence region으로 해석하지 않는다. A축은 실제 방향과 설명용 4sqrt(eigenvalue) 길이를 사용한다. 기울기 방향, prior 퍼짐, 최종 이동이 일치하더라도 참조 정답을 알고 있다는 증거가 되지 않는다.','',
        table(['seed','Joint 최대T ID','주석등급','T cm'],[[s,v['id'],v['grade'],f(v['error'])] for s in ('1','2','3') for v in m['by_seed'][s]['ALL']['FG_JOINT_POSTERIOR']['largest_errors']['translation_cm'][:3]]),'',
        'no_pose와 큰 유한 오차를 구분한다. 모든 경로가 자세를 산출해도 수십 미터의 오차를 가진 영상이 남을 수 있으며 큰 오류 ID를 삭제하지 않았다. 전체 운영 분모 319장과 실패 ID, 공통 성공 짝의 분모 319장은 원행, FAILURES, PAIRED에서 각각 확인할 수 있다.','',
        '## 후보 분포와 영상 정보의 실제 진단','',
        table(['seed','Joint C최소/최대고유값평균 px²','안정Σ최소/최대고유값평균 px²','A최소/최대고유값평균','S 평균/P90','flat fallback'],
            [[s,'/'.join(f(diag[s]['joint']['FG_JOINT_POSTERIOR']['posterior_C_eigen_px2'][k]['mean']) for k in ('minimum','maximum')),
              '/'.join(f(diag[s]['joint']['FG_JOINT_POSTERIOR']['stabilized_Sigma_eigen_px2'][k]['mean']) for k in ('minimum','maximum')),
              '/'.join(f(diag[s]['joint']['FG_JOINT_POSTERIOR']['gradient_A_eigen'][k]['mean']) for k in ('minimum','maximum')),
              f"{f(diag[s]['joint']['FG_JOINT_POSTERIOR']['gradient_strength_S']['mean'])}/{f(diag[s]['joint']['FG_JOINT_POSTERIOR']['gradient_strength_S']['P90'])}",
              diag[s]['joint']['FG_JOINT_POSTERIOR']['fallback_counts'].get('flat_gradient',0)] for s in ('1','2','3')]),'',
        '관측된 상충의 해석 범위는 제한적이다. 새 고정 창 계산이 작은 qN 근방 이동을 만들더라도 cuboid의 의미 있는 코너가 영상 밝기 교차점과 일치한다고 보장하지 않는다. Σ는 상한에 clipping되는 대리량으로 calibration된 오류 확률이 아니다. 같은 F의 W/D 선택은 작은 코너 변화에도 큰 R 차이를 만들 수 있다. 이는 가능한 설명이며 인과 기전을 확정한 증거가 아니다. 실제 평균, median, P90, 손상, 전환, CI를 먼저 판정하고 기전 설명으로 부정 결과를 대체하지 않았다.','',
        '## 실행량·비용·검산·종료','']
    capture=ex.get('N3_forward_capture',{});seal=ex.get('inference_coordinate_seal',{})
    lines += [table(['실제 작업','횟수/시간','해석'],[
        ['고정 N3 forward',f"{capture.get('execution',{}).get('N3_forward_calls')}회 / {f(capture.get('execution',{}).get('elapsed_seconds'))} s",'기존 detector/neck cache 사용; checkpoint와 temperature 고정'],
        ['추론 전용 smoke','16장×3 seed×2방법=96회','참조 및 F 호출 0회; 정확도 평가 전 검사'],
        ['새 joint 좌표',f"1914개 / {f(seal.get('accuracy_coordinate_seconds'))} s",'RGB decode 319회 / Sobel 319회; 모든 좌표를 먼저 봉인'],
        ['새 최종 F',f"{ex.get('new_F_calls')}회 / 내부 F 시간 합 {f(ex.get('new_F_seconds'))} s",'각 새 좌표당 1회; 기준선 F 재실행 0회'],
        ['평가 전체 wall',f"{f(ex.get('evaluation_wall_seconds'))} s",'참조 채점과 입출력 포함; 배포 latency와 구분'],
        ['solvePnP / RefineLM',f"{ex.get('PnP_counts',{}).get('solvePnP')}/{ex.get('PnP_counts',{}).get('solvePnPRefineLM')}",'기존 가설 선택과 F 유지'],
        ['추가 학습 / 새 합성 RGB / detector forward','0 / 0 / 0','새 모델, 새 자료, 설정 sweep 없음'],
        ['새 joint cornerSubPix / end-to-end 벤치마크','0 / 0','기존 직렬 15.8 ms를 새 joint latency로 재사용하지 않음']]),'',
        '배포 검출→N3→joint→F 전체의 동일 환경 latency는 측정하지 않았으므로 속도 우위를 주장하지 않는다. 실행 총시간이나 좌표 단계 시간을 전체 latency로 환산하지 않는다. 추론 횟수, 단계 경계, 해시는 [EXECUTION_LEDGER.json](EXECUTION_LEDGER.json)에 있다.','',
        f"독립 수치 검산 **{verification.get('status','미완료')}**: 평균, 표본분산, SD, median, P90, max, m↔cm 변환, 고정 13세션 CI, seed-mean, 자세 실패 분모, 이미지와 표의 원행 일치, 그림 SHA를 [VERIFICATION.json](VERIFICATION.json)에 기록했다. 실제 N3 posterior 및 좌표 parity는 [POSTERIOR_PARITY.json](POSTERIOR_PARITY.json), 독립 posterior 재계산은 [POSTERIOR_NUMERIC_VERIFICATION.json](POSTERIOR_NUMERIC_VERIFICATION.json), 실제 R/t→pose 오차는 [POSE_NUMERIC_VERIFICATION.json](POSE_NUMERIC_VERIFICATION.json), 보조 자료와 봉인 순서 검산은 [PUBLIC_AUXILIARY_VERIFICATION.json](PUBLIC_AUXILIARY_VERIFICATION.json)에서 확인한다.",'',
        '한 종류의 공동 융합과 Σ만 바꾼 ablation을 고정된 3 seed 및 319장에서 검증하고 부정 결과까지 공개했다. 성능이 상충해 TRADEOFF로 종료하며 추가 하이퍼파라미터, 새 seed, 재학습, 가림 classifier, robust PnP를 도입하지 않았다. 반복 DEV, geometric proxy, 3 seed, 13세션, 미보정 다중비교, 고정 창 영상 제약 근사, cap 전 covariance 대리량이라는 한계를 유지한다. 새 추론 구현이 있다는 사실로 학술적 최초성이나 논문 성공을 선언하지 않는다.','',
        '[수식/원리](METHOD_KO.md) · [실행명령](REPRODUCE.md) · [모든수치](METRICS.csv) · [원행](PREDICTIONS.jsonl.gz) · [짝CI](PAIRED.json) · [실패ID](FAILURES.json) · [그림/권리](FIGURE_INDEX.json)']
    return '\n'.join(lines)+'\n',verdict,complete


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--doc',type=Path,default=DOC);args=p.parse_args();doc=args.doc
    metrics=read(doc,'METRICS.json');paired=read(doc,'PAIRED.json');ex=read(doc,'EXECUTION_LEDGER.json',{})
    index=read(doc,'FIGURE_INDEX.json',{});verification=read(doc,'VERIFICATION.json',{})
    if len(index.get('figures',[])) != 6 or len(index.get('examples',[])) != 4:
        raise ValueError('Generate all six figures and four locked examples before the report')
    text,verdict,complete=result(doc,metrics,paired,ex,index,verification)
    (doc/'RESULT_KO.md').write_text(text);(doc/'METHOD_KO.md').write_text(METHOD_TEXT);(doc/'REPRODUCE.md').write_text(REPRO_TEXT)
    readme=f'''# Feature-Guided Joint Keypoint Refinement (2026-10-10)

실행 상태 **{'COMPLETE' if complete else 'PARTIAL'}**, 과학적 판정 **{verdict}**. 새로운 공동 위치 계산을 기존 N3→SubPix와 비교했다. 동일한 DEV 319장과 3 seed를 사용했으며 추가 학습은 0회다. 직렬 대비 위치 오차 평균과 중앙값, ADDsym-AUC가 악화했고 회전 평균이 줄었다. 두 평균 차이의 95% CI가 모두 0을 포함하므로 확정 개선으로 선언하지 않는다.

[한국어결과](RESULT_KO.md) · [정확한수식/계산](METHOD_KO.md) · [재현](REPRODUCE.md) · [수치CSV](METRICS.csv) · [JSON](METRICS.json) · [짝95%CI](PAIRED.json) · [실패/손상ID](FAILURES.json)

[원행](PREDICTIONS.jsonl.gz) · [실제222후보capture](POSTERIOR_CAPTURE.jsonl.gz) · [참조전좌표봉인](NEW_COORDINATES_SEALED.jsonl.gz) · [원입력lock](INPUT_LOCK.json) · [방법lock](FUSION_METHOD_LOCK.json) · [코드lock](CONFIG_LOCK.json) · [실행ledger](EXECUTION_LEDGER.json)

[단위검사](UNIT_TESTS.json) · [N3 parity](POSTERIOR_PARITY.json) · [독립 posterior 검산](POSTERIOR_NUMERIC_VERIFICATION.json) · [R/t 검산](POSE_NUMERIC_VERIFICATION.json) · [보조 자료 검산](PUBLIC_AUXILIARY_VERIFICATION.json) · [새 계산 증거](MECHANISM_PROOF.json) · [통계/그림 검산](VERIFICATION.json) · [그림 manifest](FIGURE_INDEX.json) · [전체 SHA256 manifest](SHA256_MANIFEST.json)

원본 RGB의 유통 권한이 확인되지 않아 좌표와 집계 그림을 공개했다. 좋지 않은 seed, 큰 오류, 좋은 코너의 손상, CI의 0 포함을 보고서에 유지했다. 논문, LaTeX, PDF는 수정하지 않았다.

'''+table(['PNG','내용'],[[f"[{v['path']}]({v['path']})",v['title']] for v in index.get('figures',[])])+f'\n\n[새코드]({CODE}/) · [실제joint]({CODE}/fusion.py) · [단위검사]({CODE}/test_fusion.py) · [그림생성]({CODE}/figures.py) · [보고서생성]({CODE}/report.py)\n'
    if (doc/'PUBLICATION_VERIFIED.json').exists():
        readme += '\n[원격 게시 검증 영수증](PUBLICATION_VERIFIED.json)\n'
    (doc/'README.md').write_text(readme);print(json.dumps(dict(status='COMPLETE' if complete else 'PARTIAL',verdict=verdict)))


if __name__=='__main__':main()
