"""Generate Korean Markdown from actual public experiment outputs; no inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DOC = ROOT / '_docs/experiments/pallet_n3_subpix_final_20261010'
METHODS = ('BASE','N3_DIM_SYM','SUBPIX','N3_THEN_SUBPIX')
METRICS = ('corner_px','translation_cm','rotation_deg','ADDsym_cm')
KOREAN = {'corner_px':'코너 (px)','translation_cm':'위치 T (cm)',
          'rotation_deg':'회전 R (°)','ADDsym_cm':'ADDsym (cm)'}
REVERSE_SHA='d2ecb51c11886e45e8752226d3a2d33c65daed80'
REVERSE_URL=f'https://github.com/CanelE452/pallet-6d-pose/blob/{REVERSE_SHA}/_docs/experiments/pallet_subpix_order_20261009_v1/RESULT_KO.md'
CODE='../../../scripts/research/pallet_n3_subpix_final_20261010'


def read(path,default=None):
    return json.loads(Path(path).read_text(encoding='utf-8')) if Path(path).exists() else default


def f(x,precision=3):
    return '해당 없음' if x is None else f'{x:.{precision}f}'


def interval(ci):
    return '미산출' if ci is None else f'[{f(ci[0])}, {f(ci[1])}]'


def statcell(s,ci=True):
    return f"{f(s['mean'])} ± {f(s['sample_std'])}"+(f"; {interval(s['CI95'])}" if ci else '')


def table(headers, rows):
    def cell(value):
        return str(value).replace('|', r'\|').replace('\n', '<br>')
    return '\n'.join(['| '+' | '.join(headers)+' |',
                      '| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(cell(v) for v in row)+' |' for row in rows])


REPRODUCTION='''기존 Python 환경을 그대로 사용한다. 새 설치·학습·데이터 생성은 필요 없다. 아래 변수 값은 사용자가 보유한 기존 환경과 원본 입력 루트로 바꾼다. 공개 저장소만 받은 독자는 두 번째 묶음으로 원행→표→그림을 검산할 수 있다.

```bash
export PALLET_PYTHON="/path/to/existing/environment/bin/python"
export PALLET_SOURCE_ROOT="/path/to/original/pallet-pose"
export PALLET_BASELINE_ROOT="/path/to/unchanged/a22-worktree"
export PALLET_FINAL_OUTPUT="/path/to/new-accuracy-output"
export PALLET_FINAL_PRIVATE="/path/to/private/audit-output"
export PYTHONDONTWRITEBYTECODE=1
```

원본을 갖고 정확도부터 다시 실행하려면 `PALLET_FINAL_OUTPUT`을 기존 PREDICTIONS가 없는 새 디렉터리로 정한다. `replay`가 그 위치로만 새 산출물을 기록하고 게시 원행은 보존한다. 다른 실행의 출력이 있으면 삭제하지 말고 새 경로를 사용한다.

```bash
"$PALLET_PYTHON" -m scripts.research.pallet_n3_subpix_final_20261010.preflight --source-root "$PALLET_SOURCE_ROOT" --output "$PALLET_FINAL_OUTPUT/INPUT_AUDIT.json" --private-dir "$PALLET_FINAL_PRIVATE"
"$PALLET_PYTHON" -m scripts.research.pallet_n3_subpix_final_20261010.replay --output "$PALLET_FINAL_OUTPUT"
"$PALLET_PYTHON" -m scripts.research.pallet_n3_subpix_final_20261010.summarize --doc "$PALLET_FINAL_OUTPUT"
"$PALLET_PYTHON" -m scripts.research.pallet_n3_subpix_final_20261010.figures --doc "$PALLET_FINAL_OUTPUT"
"$PALLET_PYTHON" -m scripts.research.pallet_n3_subpix_final_20261010.verify --doc "$PALLET_FINAL_OUTPUT" --require-figures
"$PALLET_PYTHON" -m scripts.research.pallet_n3_subpix_final_20261010.report --doc "$PALLET_FINAL_OUTPUT"
```

공개 원행만으로 다시 검산·집계·그림 생성하기:

```bash
"$PALLET_PYTHON" -m scripts.research.pallet_n3_subpix_final_20261010.summarize
"$PALLET_PYTHON" -m scripts.research.pallet_n3_subpix_final_20261010.figures
"$PALLET_PYTHON" -m scripts.research.pallet_n3_subpix_final_20261010.verify --require-figures
"$PALLET_PYTHON" -m scripts.research.pallet_n3_subpix_final_20261010.report
```

원본 필요 항목은 N3 seed 1/2/3 checkpoint와 REAL_DEV prediction JSON, 기존 detector-feature 캐시, 원본 RGB, K 및 평가 참조다. `PALLET_BASELINE_ROOT`는 기존 immutable `a22fb14beb5e8df08076385000e0d53503c1ae29`의 scorer/자료 binding이 남아 있는 worktree이며 기존 resolver가 파일 바이트를 확인한다. 모든 상대경로와 SHA256은 [INPUT_AUDIT.json](INPUT_AUDIT.json)에 있다. checkpoint는 `data/pallet/results/pallet_dim_conditioned_p_v1/runs/N3_DIM_SYM_seed{1,2,3}/last.pt`, N3 예측은 같은 결과 루트 `predictions/REAL_DEV/N3_DIM_SYM_seed{1,2,3}.json`이다. `preflight`가 TRAINING_COMPLETE/DEV_INFERENCE_COMPLETE와 해시를 대조하므로 누락·변조를 새 학습이나 추정 필드로 채우지 않는다.

캐시 정확도 재생에는 GPU가 필요 없다. 기존 torch/OpenCV/NumPy 및 후단 F 의존성이 필요하고 checkpoint 검사는 CPU에서 수행한다. detector/N3 신경망 재추론을 하지 않는다. 공개 통계·검산·그림 생성에는 원본 RGB·checkpoint·GPU가 필요 없다. 기존 시간 측정의 GPU 사용 조건은 정확도 재생과 구분한다. 이 명령들은 원고·LaTeX·PDF를 수정하지 않는다.'''


def build_report(doc,metrics,paired,lock,ex,verification,index,audit):
    sm=metrics['seed_mean']['ALL'];pair=paired['seed_mean']
    p_n=pair['N3_THEN_SUBPIX_minus_N3_DIM_SYM']['statistics']
    p_s=pair['N3_THEN_SUBPIX_minus_SUBPIX']['statistics']
    dt=p_n['translation_cm'];dr=p_n['rotation_deg']
    sig_t=dt['CI95'][1]<0;sig_r=dr['CI95'][1]<0
    uncertain=[]
    if not sig_t:uncertain.append('위치')
    if not sig_r:uncertain.append('회전')
    uncertainty=('·'.join(uncertain)+'의 결합−N3 95% 구간이 0을 포함한다') if uncertain else '두 지표의 결합−N3 95% 구간은 음수다'
    status='COMPLETE' if ex.get('complete') and verification.get('status')=='PASS' else 'PARTIAL'
    lines=[
        '# N3→cornerSubPix 3-seed 동일 REAL_DEV 검증 보고서',
        '',
        f"N3→cornerSubPix를 3-seed 동일 실사 평가에서 검증한 결과, N3 대비 seed-mean 위치 평균은 {f(abs(dt['mean_paired_difference']))} cm {'감소' if dt['mean_paired_difference']<0 else '증가'}, 회전 평균은 {f(abs(dr['mean_paired_difference']))}° {'감소' if dr['mean_paired_difference']<0 else '증가'}했으며 {uncertainty}.",
        f"SUBPIX 대비 위치 변화는 {f(p_s['translation_cm']['mean_paired_difference'])} cm {interval(p_s['translation_cm']['CI95'])}, 회전 변화는 {f(p_s['rotation_deg']['mean_paired_difference'])}° {interval(p_s['rotation_deg']['CI95'])}다. 이 반복 개발자료만으로 독립 테스트의 확정 우위를 주장할 수 없다.",
        f"코너 PCK@10은 N3 {sm['N3_DIM_SYM']['corner']['PCK']['10']:.6f} → 결합 {sm['N3_THEN_SUBPIX']['corner']['PCK']['10']:.6f}이고 손상·큰 오류도 아래에 보존했다.",
        '',
        f"상태: **{status}**. 2026-10-10 실행. 이 문서는 저장소 실험 보고서이며 논문·초록·LaTeX·PDF·PPT를 수정하지 않았다.",
        '',
        '## 1. 문제와 고정 방법',
        '',
        '초기 영상 키포인트의 작은 위치 오차가 PnP와 가로/깊이 가설 선택을 흔들 수 있다. N3는 기존 영상 특징과 canonical 치수 문맥으로 초기점을 이동하고, OpenCV cornerSubPix는 현재 원영상의 국소 밝기 기울기로 그 시작점을 정제한다. 둘을 순서대로 연결하는 표준 후처리의 검증이다. 설명용 이름은 **Feature-Initialized Image-Gradient Keypoint Refinement**이며 새로운 최적화 알고리즘의 독창성 주장이 아니다.',
        '',
        'N3와 cornerSubPix를 공동 학습하지 않았다. N3의 기존 합성 코너 이동 후보 확률 학습과 이번 영상 기울기 정제는 분리되어 있다. 단일 end-to-end loss, 미분 가능한 PnP, 전체 방법의 train-free 성격, 가린 코너의 별도 복구, 모든 기반 모델에서의 검증을 주장하지 않는다.',
        '',
        '![고정 통합 파이프라인](figures/01_method_overview.png)',
        '',
        '원영상 좌표의 Base 코너를 q₀, seed s의 기존 N3 결과를 qN, cornerSubPix 반환을 qS라고 둔다. N3는 221개 이동 후보와 0 이동 null 후보(총 222개)에 대해 `p_j = softmax(logit_j / T_s)`를 계산하고 `δ = λ Σ_j p_j d_j`를 적용한다. 기존 `λ=1`, seed별 합성 calibration에서 확정된 `T_s`와 checkpoint를 보존했다. N3 내부의 기존 float32 상한도 변경하지 않고 그 출력 캐시를 재사용했다.',
        '',
        '`qS = cornerSubPix(gray_uint8, qN, winSize=(5,5), zeroZone=(-1,-1), criteria=(EPS|COUNT,40,0.001))`이다. SUBPIX 단독은 q₀에서 시작한다. 각 코너 0–7에만 적용하고 원영상 밖 초기점·오류·비유한 출력에서는 시작점을 유지한다. 중심점 8과 미지원·결측 슬롯을 보존한다.',
        '',
        'cornerSubPix의 공식 설명은 국소점 pᵢ의 영상 기울기 gᵢ에 대해 `εᵢ=gᵢᵀ(q−pᵢ)`를 줄이는 정상방정식 `Gq=b`, `G=Σᵢgᵢgᵢᵀ`, `b=Σᵢgᵢgᵢᵀpᵢ`, `q=G⁻¹b`를 반복하는 방식이다. 새 q로 창을 옮겨 수렴 또는 최대 반복에서 멈춘다. winSize=(5,5)는 11×11 창이고 zeroZone=(−1,−1)는 중앙 제외 영역 없음이다. 실제 수치 처리는 고정 OpenCV 구현 그대로다. [OpenCV 4.9.0 공식 cornerSubPix 설명](https://docs.opencv.org/4.9.0/dd/d1a/group__imgproc__feature.html).',
        '',
        '최종 상한은 **원래 Base 기준으로 마지막에 한 번** 적용한다. `c = 0.01 sqrt(W_image²+H_image²)`, `Δ = qS−q₀`, `qFinal = q₀ + Δ min(1,c/||Δ||₂)`이며 0 이동은 그대로 둔다. N3 기준으로 추가 1%를 허용하지 않는다. 640×480 원영상이면 c=8 px다. 기존 캐시 float32 반올림에 따른 미세 상한 초과는 원행/입력 감사에 남기고 결합의 float64 최종 상한에서만 엄격히 제한한다.',
        '',
        '최종 좌표마다 기존 F를 실제 실행한다. prediction-only 가로/깊이 가설 선택, SQPnP와 `solvePnPRefineLM`을 보존했다. 객체 검출 후보·선택 index·점수·박스·지원 마스크·K·canonical 치수·proper 대칭 계약을 고정했다. 참조 코너, 사람이 표시한 가시성, 참조로 고른 코너 재배열은 추론 입력이 아니다.',
        '',
        '## 2. 자료와 실행 계약',
        '',
        'REAL_DEV **319장, 13세션**, 객체 매칭 311장, 유효 참조 코너 2,499개와 관측 코너 2,445개다. clean 153 / moderate 92 / severe 74는 기존 주석 난도 등급이며 물리적인 외부 가림률 측정이 아니다. severe를 포함한 전체 319장을 주평가로 유지했다. 이 DEV는 여러 설계에 반복 노출됐으므로 새 독립 실사 테스트라고 부르지 않는다.',
        '',
        '참조는 2D 및 등록 치수로 만든 기하 재구성값이고 물리 장치로 독립 실측한 GT가 아니다. 문맥 치수는 canonical `[W,D,H]`, 기존 PnP API의 치수 배열은 `[W,H,D]`다. 이 순서를 혼동하지 않고 `fixed_metadata.dimensions_pnp_WH_D_m` 및 lock의 두 배열을 그대로 보존했다. 허용 proper symmetry와 두 가설을 동등한 것으로 합치지 않았다.',
        '',
        f"실행 당시 최신 원격 main에서 분리한 worktree의 기준 SHA는 `{lock.get('remote_main_at_start','미기록')}`다. 사용자 원래 checkout과 기존 결과/입력은 별도 보존했다. 입력 해시 [감사](INPUT_AUDIT.json), [방법·319입력 lock](INPUT_AND_METHOD_LOCK.json), [실행/검증](EXECUTION_AND_VERIFICATION.json), [seed1 parity](SEED1_PARITY.json), [독립 검산](INDEPENDENT_VERIFICATION.json)을 공개했다.",
        '',
        f"정확도는 N3 seed별 기존 REAL_DEV 좌표 캐시를 재사용해 detector **{ex.get('detector_calls',0)}회**, N3 forward **{ex.get('N3_forwards',0)}회**, 실제 최종 F **{ex.get('actual_F_calls','미기록')}회**다. BASE/SUBPIX는 실제 F를 각각 319회 실행한 같은 결과를 seed 2/3에 공유했다. N3와 결합은 seed별 새 좌표로 F를 실행했다. 저장 원행은 4방법×3seed×319 = **{ex.get('rows','미기록')}행**이다. 새 학습/optimizer update **0**, 추가 합성 영상 **0**, 신규 시간측정 **0**이다.",
        '',
        'seed1은 **전체 319장×4방법**의 기존 좌표를 정확 일치 비교하고 자세 오차 및 가능한 기존 R/t 벡터를 절대 오차 `1e−7`로 비교한 뒤 seed2/3을 실행했다. 과거 BASE/SUBPIX의 R/t 벡터는 26장에만 저장돼 그 범위만 직접 벡터 비교가 가능했지만, 좌표·가설·오차는 319장 모두 비교했다. 이번 원행에는 모든 실제 R/t가 있다.',
        '',
        '## 3. 전체 정확도: seed별 결과와 seed-mean',
        '',
        '모든 SD는 **오차의 표본 표준편차(ddof=1)**이며 평균의 표준오차·신뢰구간·학습 seed 산포가 아니다. 코너 평균은 관측 코너 풀링(n=2,445), 자세 평균은 실제 성공 영상당 오차(n=319)다. 유한 큰 오류는 제거하지 않았다. 아래 `평균 ± SD; [95% CI]`의 CI는 13세션 단위 cluster bootstrap 10,000회다.',
        '',
    ]
    lines += ['N3 checkpoint와 고정 temperature (temperature를 다시 튜닝하지 않았다):','',
        table(['seed','temperature','checkpoint SHA256','REAL_DEV prediction SHA256'],
            [[v['seed'],v['temperature'],f"`{v['checkpoint']['sha256']}`",f"`{v['prediction']['sha256']}`"] for v in audit.get('models',[])]),'',
        '감사 기록 보존: 정확도 실행 시작의 INPUT_AUDIT 원본을 [INPUT_AUDIT_AT_EVALUATION_START.json](INPUT_AUDIT_AT_EVALUATION_START.json)에 보존했고 현재 INPUT_AUDIT도 그 원본 바이트로 유지하여 lock에 저장한 SHA와 일치한다. 이후 기존 runtime 독립 검산을 추가한 [INPUT_AUDIT_SUPPLEMENT.json](INPUT_AUDIT_SUPPLEMENT.json)은 별도 증거 부가본이다. 평가 입력이나 과거 lock을 결과에 맞춰 변경한 것이 아니다. [HISTORICAL_EVIDENCE_VERIFICATION.json](HISTORICAL_EVIDENCE_VERIFICATION.json)이 과거 runtime 원시간과 seed1 역순의 고정 commit 출처를 검산했다.','',
        '[EVALUATOR_CONTRACT_REVIEW.json](EVALUATOR_CONTRACT_REVIEW.json)은 공개 R/t에서 T·R·ADDsym 오차와 참조 코너 거리·최종 cap을 별도로 재계산해 최대 절대 차이 0을 확인했다. 기존 N3 pose 캐시와도 세 seed 각각 319장이 일치했다. seed1 캐시 해시는 과거 lock과 연결됐고 seed2/3 pose 캐시 해시는 이번 추가 parity 증거로 기록되어 과거에 인증된 해시라고 혼용하지 않는다.','']
    scopes=[(s,metrics['by_seed'][s]['ALL']) for s in ('1','2','3')]+[('seed-mean',sm)]
    lines += [table(['seed','방법','T cm: 평균±SD; 95% CI','R °: 평균±SD; 95% CI','ADDsym cm: 평균±SD; 95% CI','자세 성공/실패'],
        [[scope,m,*[statcell(summary[m]['metrics'][k]) for k in ('translation_cm','rotation_deg','ADDsym_cm')],
          f"{summary[m]['pose']['available']}/{summary[m]['pose']['failures']}"] for scope,summary in scopes for m in METHODS]),'',
        table(['seed','방법','코너 px: 평균±SD; 95% CI','관측/참조 코너','PCK@10','gross20 개수/비율'],
            [[scope,m,statcell(summary[m]['metrics']['corner_px']),
              f"{summary[m]['corner']['observed_corners']}/{summary[m]['corner']['reference_corners']}",
              f(summary[m]['corner']['PCK']['10'],6),
              f"{summary[m]['corner']['gross20_count']}/{f(summary[m]['corner']['gross20_rate'],6)}"] for scope,summary in scopes for m in METHODS]),'',
        'seed-mean의 PCK/gross20는 세 seed별 비율/개수의 산술 평균이고 개수가 소수가 될 수 있다. 평균 오차에 임계값을 적용한 값은 별도 `threshold_of_seed_mean_errors`에 보존했다. seed-mean의 오차 분포는 같은 ID의 세 seed **오차**를 먼저 산술 평균한 뒤 319영상/2,445코너의 분포를 계산한다. R/t를 평균해 새로운 자세를 만든 것이 아니다. 957장을 독립 영상으로 계산하지 않았다. BASE/SUBPIX는 N3 seed와 무관해 세 seed에서 같다. seed별 SD를 단순 평균한 값은 seed-mean SD와 다르며 보조 배열 `arithmetic_mean_of_seed_metrics`로만 제공한다.', '',
        '![seed별 자세 평균](figures/02_pose_accuracy_by_seed.png)', '',
        '<details>', '<summary>전체 표본분산·중앙값·P90·최댓값 (원행 재계산)</summary>', '']
    for metric in METRICS:
        lines += [f'**{KOREAN[metric]}**','',table(['seed','방법','n','표본분산','SD','중앙값','P90','최댓값'],
            [[scope,m,summary[m]['metrics'][metric]['n'],*[f(summary[m]['metrics'][metric][k]) for k in ('sample_variance','sample_std','median','P90','max')]]
             for scope,summary in scopes for m in METHODS]),'']
    lines += ['</details>','',
        'ADDsym는 기존 cuboid의 허용 proper 대칭 대응 오차다. m→cm 변환에서 평균/SD는 ×100, 분산은 ×10,000이다. 양자화·결측·대칭 참조 매칭 규칙을 모든 비교에 동일 적용했다. 모든 집합·seed의 원정밀 숫자는 [RESULTS.csv](RESULTS.csv)에 있다.','',
        '자세 오차는 기존 코드의 정의를 그대로 쓴다. `T=100||t_est−t_ref||₂` cm, `R=min_Q (180/π) acos(clip((tr((R_ref Q)ᵀR_est)−1)/2,−1,1))` degree, `ADDsym=100 min_Q mean_i ||R_est X_i+t_est−(R_ref Q X_i+t_ref)||₂` cm다. Q는 canonical 계약의 C1/C2/C4 proper 회전만, Xᵢ는 기존 cuboid 8코너다. 회전과 ADDsym가 각각 최소 proper 대응을 취하는 기존 정의를 변경하지 않았다.','',
        f'역순 `SUBPIX_THEN_N3`는 별도 [2026-10-09 seed1 탐색 결과]({REVERSE_URL})의 출처를 인용한다. 이번 3-seed 정방향 실험에는 역순을 추가하지 않았고, 서로 다른 seed 수의 평균으로 우열을 계산하지 않는다. 과거 동일 seed1 역순 T=39.4083 cm/R=18.5055°, 역순−정방향 T +0.5133 [−0.5968, 1.5617] cm/R +0.2532 [−0.6461, 1.0151]°다. 두 구간은 모두 0을 포함하는 탐색 결과다.','',
        '## 4. 단독 방법 대비 추가 효과와 불확실성','',
        '아래는 결합−비교방법이다. **음수=오차 감소**. 같은 영상의 차이를 계산하고, 13개 원세션을 균등 중복 추출하되 포함된 원영상·원코너 수로 풀링했다. 모든 seed/방법/지표/등급이 같은 fixed draws를 재사용한다. 평균 주변차이와 공통 유효 짝차이를 JSON에 별도로 보존했다. CI에 0이 있으면 차이 방향의 불확실성이 남으며 동등성의 증거도 아니다.','']
    pair_scopes=[(s,paired['by_seed'][s]) for s in ('1','2','3')]+[('seed-mean',pair)]
    lines += [table(['seed','결합−기준','코너 Δ px [CI]','T Δ cm [CI]','R Δ ° [CI]','ADDsym Δ cm [CI]','공통 자세/전체'],
        [[scope,m,*[f"{f(ps['N3_THEN_SUBPIX_minus_'+m]['statistics'][k]['mean_paired_difference'])} {interval(ps['N3_THEN_SUBPIX_minus_'+m]['statistics'][k]['CI95'])}" for k in METRICS],
          f"{ps['N3_THEN_SUBPIX_minus_'+m]['coverage']['common_success']}/319"] for scope,ps in pair_scopes for m in ('N3_DIM_SYM','SUBPIX','BASE')]),'',
        '![짝비교 95% CI](figures/03_paired_improvement_ci.png)','']
    for base in ('N3_DIM_SYM','SUBPIX'):
        tdirs=[];rdirs=[]
        for s in ('1','2','3'):
            ps=paired['by_seed'][s]['N3_THEN_SUBPIX_minus_'+base]['statistics']
            tdirs.append(f"seed{s} {ps['translation_cm']['mean_paired_difference']:+.3f} cm")
            rdirs.append(f"seed{s} {ps['rotation_deg']['mean_paired_difference']:+.3f}°")
        lines += [f"{base} 대비 위치 방향: {', '.join(tdirs)}. 회전 방향: {', '.join(rdirs)}. seed 하나가 좋다는 이유로 다른 seed를 제외하지 않았다.",'']
    lines += [f"단독 SUBPIX 대비 seed-mean의 T 변화는 {f(p_s['translation_cm']['mean_paired_difference'])} cm이며, 양수는 악화다. 구간 {interval(p_s['translation_cm']['CI95'])}를 함께 보아야 한다. 추가 R 효과는 {f(p_s['rotation_deg']['mean_paired_difference'])}° {interval(p_s['rotation_deg']['CI95'])}다. N3 대비 코너·ADDsym P90 변화는 각각 {f(sm['N3_THEN_SUBPIX']['metrics']['corner_px']['P90']-sm['N3_DIM_SYM']['metrics']['corner_px']['P90'])} px, {f(sm['N3_THEN_SUBPIX']['metrics']['ADDsym_cm']['P90']-sm['N3_DIM_SYM']['metrics']['ADDsym_cm']['P90'])} cm다. 평균 이득·tail·코너 손상을 합친 임의 가중 점수로 단일 승자를 만들지 않는다.",'',
        f"bootstrap draw SHA256: `{metrics['bootstrap']['draw_sha256']}`; seed `20260917`, 10,000회. seed-mean CI는 세 seed를 고정한 상태의 세션 불확실성이다. 학습 seed 모집단 전체의 불확실성을 세 seed로 확증한 것은 아니다. 다중 비교 보정도 적용하지 않았다.",'',
        '## 5. 난도 등급·손상·큰 실패','',
        '전체 319장 결과를 먼저 유지하고 기존 난도 주석으로만 분해한다. 가설 전환 및 참조 기반 변화는 사후 원인 분석이며 추론 게이트를 바꾸는 규칙이 아니다.','',
        table(['등급','방법','T cm 평균±SD','R ° 평균±SD','코너 px 평균±SD','ADDsym cm 평균±SD','성공/전체','PCK@10'],
            [[g,m,*[statcell(metrics['seed_mean'][g][m]['metrics'][k],False) for k in ('translation_cm','rotation_deg','corner_px','ADDsym_cm')],
              f"{metrics['seed_mean'][g][m]['pose']['available']}/{metrics['seed_mean'][g][m]['total_frames']}",f(metrics['seed_mean'][g][m]['corner']['PCK']['10'],6)]
             for g in ('clean','moderate','severe') for m in METHODS]),'',
        '![난도 등급별 결과](figures/04_grade_breakdown.png)','',
        '![큰 오류를 보존한 경험 분포](figures/05_error_distribution.png)','',
        '그림 05는 전체 유한 오류를 유지하며 1 이하 선형/그 이상 로그인 symlog 축을 명시했다. 중앙값·P90·최댓값과 평균을 구분한다. 자세 실패는 임의 0 또는 1e6 오차로 대치하지 않는다. 공통 자세 산출 영상의 비교와 전체 운영 실패율은 각각 위 표/원행에 남긴다.','']
    diag=metrics['diagnostics']['by_seed']
    lines += [table(['seed','Base→결합 양호 손상','N3→결합 양호 손상','N3→결합 큰오류 회복','N3→SubPix 추가 이동 px 평균±SD','총상한 코너/영상','가설 전환 vs N3/Base'],
        [[s,diag[s]['damage_BASE']['N3_THEN_SUBPIX']['good5_to_bad10'],diag[s]['damage_N3']['N3_THEN_SUBPIX']['good5_to_bad10'],
          diag[s]['damage_N3']['N3_THEN_SUBPIX']['bad20_to_good10'],statcell(diag[s]['motion']['SUBPIX_additional_move_from_N3_px'],False),
          f"{diag[s]['motion']['cap_active_corners']}/{diag[s]['motion']['cap_active_frames']}",
          f"{diag[s]['hypothesis']['N3_DIM_SYM']['switch_frames']}/{diag[s]['hypothesis']['BASE']['switch_frames']}"] for s in ('1','2','3')]),'',
        '양호 손상은 `before<5px → after>10px`, 큰오류 회복은 `before>20px → after≤10px`이며 같은 canonical 참조 코너 ID에서 계산했다. 임계점의 `<`/`≤` 차이를 원행 검산에도 유지했다. 이동상한은 모든 지원 코너에 적용되지만 손상 판정은 유효 참조 코너만 사용한다. 세 seed의 횟수 합은 같은 이미지의 반복 실행 횟수이며 신규 독립 표본 수가 아니다.','',
        '![손상·복구·상한·가설](figures/06_corner_damage_and_hypothesis.png)','']
    lines += [table(['seed','비교','T/R 모두 감소','T↓R↑','T↑R↓','모두 증가','불변','미산출'],
        [[s,m,*[diag[s]['hypothesis'][m]['pose_quadrants'].get(k,0) for k in ('BOTH_DECREASE','T_DECREASE_R_INCREASE','T_INCREASE_R_DECREASE','BOTH_INCREASE','ONE_OR_BOTH_EXACTLY_UNCHANGED','UNAVAILABLE')]]
         for s in ('1','2','3') for m in ('N3_DIM_SYM','SUBPIX')]),'',
        '원본 RGB의 공개 유통 권한은 확인되지 않았다. 원본 RGB·작업장·제3자 식별 가능한 사진은 공개하지 않고 **비식별 좌표 이동 그림**으로 대체했다. 참조는 평가 후 저장된 기하 재구성 코너만 사용한다. 추론으로 참조를 전달하지 않았다.','',
        '![성공·손상·무차이·큰 실패 좌표 사례](figures/07_qualitative_examples.png)','',
        table(['사후 선택 범주','ID','seed','ΔT cm','ΔR °','선택 규칙'],
            [[e['category'],e['id'],e['seed'],f(e['translation_delta_cm']),f(e['rotation_delta_deg']),e['selection_rule']] for e in index.get('qualitative_examples',[])]),'',
        '사례는 seed1에서 결합−N3 T 변화의 최솟값/최댓값, `|ΔT cm|+|ΔR degree|`가 가장 작은 프레임, 결합 T 절대오차 최댓값으로 선택하고 ID 사전순으로 tie를 깼다. 서로 다른 단위를 더한 무차이 사례 점수는 사례를 고르는 설명용 규칙이며 방법의 성능 순위가 아니다. 좌표는 x/width, y/height로 정규화하고 이동 확대부만 ×1000 했다. 가장 큰 추가 이동의 지원 코너를 확대하며 조작된 위치/합성 영상은 없다. 이 사후 사례로 전체 성능을 입증하지 않는다.','',
        '**가장 큰 실제 자세 오류를 삭제하지 않은 ID** (각 seed 결합 T 상위 3장):','',
        table(['seed','ID','주석 등급','T cm'],[[s,v['id'],v['grade'],f(v['error'])] for s in ('1','2','3') for v in metrics['by_seed'][s]['ALL']['N3_THEN_SUBPIX']['largest_errors']['translation_cm'][:3]]),'']
    failrows=[]
    for scope,summary in scopes:
        for m in METHODS:
            failrows.append([scope,m,summary[m]['pose']['failures'],', '.join(summary[m]['pose']['failure_ids']) or '없음'])
    lines += [table(['seed','방법','no_pose 수','실패 ID'],failrows),'',
        'cornerSubPix fallback은 자세 실패와 다르다. 실패한 코너는 qS에서 시작점을 유지한 뒤 모든 결합 코너에 원래 Base 기준 최종 cap을 적용한다. 기존 N3 float32 미세 초과 때문에 fallback의 qFinal이 qN과 미세하게 달라질 수 있다. seed별 실패 반환의 이유·빈도:','',
        table(['seed','방법','fallback 코너/영상','이유/수'],
            [[s,m,f"{diag[s]['correction_by_method'][m]['fallback_corners']}/{diag[s]['correction_by_method'][m]['fallback_frames']}",
              json.dumps(diag[s]['correction_by_method'][m]['fallback_counts'],ensure_ascii=False)]
             for s in ('1','2','3') for m in ('SUBPIX','N3_THEN_SUBPIX')]),'',
        table(['seed','OpenCV corner 호출','fallback 이유/수','최종 N3와 완전 동일 영상','최종 SUBPIX와 완전 동일 영상'],
            [[s,diag[s]['motion']['OpenCV_corner_calls'],json.dumps(diag[s]['motion']['fallback_counts'],ensure_ascii=False),
              diag[s]['motion']['final_exactly_N3_frames'],diag[s]['motion']['final_exactly_SUBPIX_frames']] for s in ('1','2','3')]),'',
        '모든 손상 ID/코너 index·회복 ID·cap ID·가설 전환 ID는 [METRICS.json](METRICS.json)의 `diagnostics.by_seed`에 있다. 원본 이미지 없이 같은 행을 추적할 수 있다.','',
        '## 6. 비용: 기존 같은 환경 전체 경로의 실제 측정','',
        '새 시간 벤치마크를 실행하지 않았다. **2026-10-08 seed1**의 같은 하드웨어·시점·패널에서 측정한 네 방법의 기존 실제 전체 경로 [RUNTIME.json](../pallet_n3_subpix_20261008_v1/RUNTIME.json)을 재사용한다. seed2/3의 새 측정값으로 부르지 않으며 다른 시점/모델 숫자와 섞지 않는다.','']
    runtime=read(ROOT/'_docs/experiments/pallet_n3_subpix_20261008_v1/RUNTIME.json',{})
    rt_names={'BASE':'BASE','N3_DIM_SYM':'N3','SUBPIX':'SUBPIX','N3_THEN_SUBPIX':'N3_SUBPIX'}
    lines += [table(['방법 (과거 seed1)','n','평균±SD ms','분산 ms²','중앙값 ms','P90 ms'],
        [[m,runtime['summaries'][alias]['full_pipeline']['n'],
          f"{f(runtime['summaries'][alias]['full_pipeline']['mean_ms'])} ± {f(runtime['summaries'][alias]['full_pipeline']['sample_std_ms'])}",
          *[f(runtime['summaries'][alias]['full_pipeline'][k]) for k in ('sample_variance_ms2','median_ms','p90_ms')]] for m,alias in rt_names.items()]),'',
        'RTX 3080, torch 2.1.1+cu118, OpenCV 4.9.0, batch 1, torch 4 threads/OpenCV 1 thread다. 기존 13세션×2장=26프레임, 방법별 warmup 20회와 5반복(각 130개 측정행), 고정 교차 실행 순서와 GPU 동기화를 사용했다. RAM 원영상→검출/객체선택→보정→실제 기존 F를 연속 측정했다. 파일 디코딩·모델 로딩·정답 채점·receipt 작성은 제외했다. 캐시 재생 시간이나 단계 시간의 합산이 아니다. 원시간 [RUNTIME_ROWS.jsonl.gz](../pallet_n3_subpix_20261008_v1/RUNTIME_ROWS.jsonl.gz)와 간섭 점검 기록도 보존했다.','',
        f"당시 평균 전체경로 결합 추가시간은 N3 대비 {f(runtime['latency_deltas_ms']['N3_SUBPIX_minus_N3']['mean_ms'])} ms, SUBPIX 대비 {f(runtime['latency_deltas_ms']['N3_SUBPIX_minus_SUBPIX']['mean_ms'])} ms다. 현재 정확도 재생 시간 {f(ex.get('elapsed_seconds'))}초는 캐시 기반 검증 실행 소요시간이며 배포 전체 경로 latency가 아니다. N3 자체는 과거 합성 학습이 필요했고 cornerSubPix 단계만 새 학습 없이 동작한다.",'',
        '## 7. 검산·한계·최종 판정','',
        f"입력 감사 `{audit.get('status','미완료')}`, seed1 전319 parity `{ex.get('seed1_parity','미완료')}`, 독립 원행 검산 `{verification.get('status','미완료')}`다. [verify.py]({CODE}/verify.py)는 집계 생성 코드와 별도 구현으로 n/평균/표본분산/SD/중앙값/P90/최대, cm 변환, 공통 분모, 13세션 CI, seed-mean과 그림 SHA256을 점검한다. PNG의 실제 시각 검토 상태는 [EXECUTION_AND_VERIFICATION.json](EXECUTION_AND_VERIFICATION.json)에 기록한다.",'',
        '원행/표/짝비교/그림은 같은 고정 입력을 따른다. 중심점8·미지원점·K·치수·원본 검출 metadata·선택 index와 1% 총이동 계약을 보존했다. GT는 평가에만 사용하고 추론 입력에 쓰지 않았다. 성공 표본 제외에 따른 개선 주장·실패를 임의 큰 숫자로 대치·나쁜 seed 선택 제외는 하지 않았다.','',
        '확인 가능한 주장은 이 동일 개발자료·고정 세 seed에서 관측된 평균 변화와 분포/손상/시간 trade-off다. N3 대비 추가 T/R 효과, SUBPIX 대비 추가 가치, CI의 0 포함 여부, PCK·gross20·P90 악화 여부를 서로 구분해 해석해야 한다. 코너 관측 정밀화와 극단 자세 실패 복구는 다른 문제이며 작은 이동 상한이 큰 실패를 해결했다는 주장은 이 결과로 뒷받침되지 않는다.','',
        '논문에 쓸 수 있는 범위: 고정 N3 초기화 뒤 원영상 기울기 정제를 연결한 재현 가능한 경로, 같은 REAL_DEV에서의 세 seed 수치, 불확실성과 부정적 결과. 아직 주장할 수 없는 범위: 독립 real GT로 검증한 보편 우위, 모든 가림 해결, 공동학습된 새 알고리즘, 새로운 독립 테스트 결과. 세 seed와 13세션의 제한, 반복 DEV 사용, 기하 참조의 편향, 다중 비교 미보정을 함께 밝혀야 한다.','',
        '## 8. 재현','',REPRODUCTION,'',
        '## 9. 산출물 연결','',
        '[원행](PREDICTIONS.jsonl.gz) · [집계 JSON](METRICS.json) · [결과 CSV](RESULTS.csv) · [짝비교 CI](PAIRED_COMPARISONS.json) · [그림 manifest](FIGURE_INDEX.json) · [행 추적 안내](REVIEW_GUIDE_KO.md) · [목록](README.md)','']
    return '\n'.join(lines),status


def guide():
    return '''# 프레임 하나의 보정과 자세를 추적하는 방법

공개 원행 [PREDICTIONS.jsonl.gz](PREDICTIONS.jsonl.gz)는 seed별 319×4행, 총 3,828행의 gzip JSONL이다. 원본 RGB·가중치·개인 절대경로는 포함하지 않는다. 모든 2D 좌표는 원영상 픽셀이고 `raw_hw=[height,width]`다.

| 필드 | 실제 의미 |
| --- | --- |
| `id`, `session`, `grade`, `seed`, `method` | 같은 원본 영상과 반복 seed/방법을 연결하는 키 |
| `q0` | 원래 Base의 9×2 좌표 |
| `qN` | N3/결합 행의 seed별 기존 N3 출력. BASE/SUBPIX 행에서는 q0 |
| `qS` | 결합은 N3에서 시작한 cornerSubPix의 native 반환, SUBPIX는 Base에서 시작한 native 반환. 최종 cap 전 |
| `qFinal` | 실제 최종 F에 전달한 좌표; 결합은 원래 Base 기준 마지막 총상한 후 |
| `prediction_support` | 기존 추론 지원 마스크; 사람이 표시한 가시성 아님 |
| `fixed_metadata.selected_index` | 실제 검출 객체 선택. 과거 scorer의 최상위 selected_index와 혼용 금지 |
| `fixed_metadata.K` | 원영상 intrinsics |
| `fixed_metadata.dimensions_pnp_WH_D_m` | 기존 PnP API 순서 `[W,H,D]`의 미터 치수 |
| `fixed_metadata.canonical_symmetry_order` | 기존 허용 proper 대칭 계약 |
| `correction.diagnostics.corner_records` | 코너별 attempted/status/fallback_reason/changed/movement_px |
| `correction.additional_subpix_px8` | 결합의 native 추가 이동 `norm(qS−qN)`; SUBPIX는 `norm(qS−q0)` |
| `correction.total_before_cap_px8` | `norm(qS−q0)` |
| `correction.total_final_px8` | `norm(qFinal−q0)` |
| `correction.cap_active8`, `cap_px` | 원래 Base 기준 1% 총상한의 코너별 발동/상한 |
| `F_attempt`, `F_complete`, `reused_from_seed1` | 실제 F 호출/완료 여부와 동일 BASE/SUBPIX 결과의 공유 여부 |
| `PnP_counts` | 이 행의 실제 최종 F 내부 solvePnP/RefineLM 호출 수 |
| `actual_pose.R_cf`, `R_physical`, `centroid` | 실제 F가 반환한 회전/미터 위치. centroid가 t이며 추정 자세를 사후 재구성한 값 아님 |
| `actual_pose.selected_hypothesis`, `final_hypothesis` | 실제 선택된 W/D 가설 |
| `pose.available` | 자세 오차의 실제 유효 산출 여부 |
| `pose.translation_cm`, `rotation_deg`, `ADDsym_m` | 참조 오차; 마지막은 미터이며 표/그림은 ×100하여 cm |
| `corner.observed_errors`, `canonical_errors`, `canonical_valid` | 관측 코너 px 오차와 기존 canonical ID/유효 참조 매칭 |
| `canonical_observed` | 관측 풀링 분모를 canonical ID로 보존한 마스크 |
| `evaluation_reference_points`, `evaluation_reference_valid` | 평가 후 부가한 기하 재구성 참조. 추론 미사용 |
| `evaluation_permutation` | `evaluation_reference_points[permutation]`으로 native q 순서의 참조 overlay를 만드는 평가 전용 대응 |
| `evaluation_reference_used_in_inference` | 항상 false. 참조는 평가/그림에만 사용 |

중심점은 index 8이며 정제 대상은 0–7이다. qS가 fallback으로 qN을 유지해도 최종 Base 기준 엄격 cap에서 과거 float32 반올림 초과를 미세하게 제거할 수 있다. 이 경우 qS=qN과 qFinal=qN은 다른 검사항목이다.

예: 같은 ID의 네 방법을 JSONL에서 읽기 (Python 표준 라이브러리만 필요):

```bash
export FRAME_ID="eval_cad:1778653003088339968"
"$PALLET_PYTHON" - <<'PY'
import gzip, json, os
path = "_docs/experiments/pallet_n3_subpix_final_20261010/PREDICTIONS.jsonl.gz"
with gzip.open(path, "rt", encoding="utf-8") as stream:
    for line in stream:
        row = json.loads(line)
        if row["id"] == os.environ["FRAME_ID"] and row["seed"] == 1:
            print(json.dumps({key: row.get(key) for key in
                ("id", "seed", "method", "q0", "qN", "qS", "qFinal",
                 "correction", "actual_pose", "pose", "final_hypothesis")}, indent=2))
PY
```

集계는 `METRICS.json.by_seed.{seed}.{ALL|clean|moderate|severe}.{method}`에서 읽는다. `metrics.{corner_px|translation_cm|rotation_deg|ADDsym_cm}` 안에 n/mean/sample_variance/sample_std/median/P90/max/CI95가 있다. `seed_mean`은 원영상별 세 seed의 **오차**를 평균한 결과이지 새로운 pose 행이 아니다.

`PAIRED_COMPARISONS.json.by_seed.{seed}.N3_THEN_SUBPIX_minus_{N3_DIM_SYM|SUBPIX|BASE}.statistics.{metric}`의 `mean_paired_difference`, `CI95`, `common_eligible_frames`, `excluded_frames`를 확인한다. `coverage`는 전체319와 공통 성공/한쪽만 성공/양쪽 실패를 구분한다. 음수는 오차 감소다.

손상은 `METRICS.json.diagnostics.by_seed.{seed}.damage_N3.N3_THEN_SUBPIX.harmed_corner_records`와 `damage_BASE`에서, 상한은 `motion.cap_corner_records`, 가설 전환은 `hypothesis.{comparison}.switch_ids`에서 찾는다. 전체 worst IDs는 각 summary의 `largest_errors`에 있다. 성공/실패 ID 목록은 `pose.successful_ids`/`failure_ids`다.

[검산 코드](../../../scripts/research/pallet_n3_subpix_final_20261010/verify.py)와 [INDEPENDENT_VERIFICATION.json](INDEPENDENT_VERIFICATION.json)으로 원행/표/CI/단위/그림 해시를 검사한다. [FIGURE_INDEX.json](FIGURE_INDEX.json)은 각 그림의 근거파일 해시와 07의 사후 선택 규칙을 보존한다. 원본 RGB의 배포 권한이 확인되지 않아 좌표 그림만 생성했으며 사진을 공개하지 않았다.
'''.replace('集계','집계')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--doc',type=Path,default=DOC);args=p.parse_args();doc=args.doc
    metrics=read(doc/'METRICS.json');paired=read(doc/'PAIRED_COMPARISONS.json');lock=read(doc/'INPUT_AND_METHOD_LOCK.json')
    ex=read(doc/'EXECUTION_AND_VERIFICATION.json',{});verification=read(doc/'INDEPENDENT_VERIFICATION.json',{})
    index=read(doc/'FIGURE_INDEX.json',{});audit=read(doc/'INPUT_AUDIT.json',{})
    report,status=build_report(doc,metrics,paired,lock,ex,verification,index,audit)
    (doc/'REPORT_KO.md').write_text(report,encoding='utf-8');(doc/'REVIEW_GUIDE_KO.md').write_text(guide(),encoding='utf-8')
    figures=index.get('figures',[])
    readme=f'''# N3→cornerSubPix 3-seed 최종 실사 검증 (2026-10-10)

상태: **{status}**. 동일 REAL_DEV 319장·13세션, N3 seed 1/2/3, 새 학습 0. BASE/N3/SUBPIX/N3→SUBPIX를 같은 최종 F로 비교했다.

[한국어 결과 보고서](REPORT_KO.md) · [프레임 추적 안내](REVIEW_GUIDE_KO.md) · [재현 명령](REPORT_KO.md#8-재현)

[전체 수치 CSV](RESULTS.csv) · [통계 JSON](METRICS.json) · [짝비교/95% CI](PAIRED_COMPARISONS.json) · [원행 gzip JSONL](PREDICTIONS.jsonl.gz)

[입력 해시 감사](INPUT_AUDIT.json) · [감사 부가본](INPUT_AUDIT_SUPPLEMENT.json) · [입력/방법 lock](INPUT_AND_METHOD_LOCK.json) · [seed1 전319 parity](SEED1_PARITY.json) · [실행/검증](EXECUTION_AND_VERIFICATION.json) · [독립 검산](INDEPENDENT_VERIFICATION.json) · [R/t 독립 재계산](EVALUATOR_CONTRACT_REVIEW.json) · [과거 증거 검산](HISTORICAL_EVIDENCE_VERIFICATION.json) · [그림/권리 manifest](FIGURE_INDEX.json)

원본 RGB의 공개 배포 권한이 확인되지 않아 좌표만으로 사례를 그렸다. 반복 개발자료와 기하 재구성 참조의 한계, PCK/큰오류/손상 등 부정적 결과를 보고서에 함께 보존했다.

'''+table(['PNG','내용'],[[f"[{v['path']}]({v['path']})",v['title']] for v in figures])+f'''

[재현 코드]({CODE}/) · [집계]({CODE}/summarize.py) · [독립 검산]({CODE}/verify.py) · [그림 생성]({CODE}/figures.py) · [보고서 생성]({CODE}/report.py)
'''
    (doc/'README.md').write_text(readme,encoding='utf-8')
    print(json.dumps(dict(status=status,markdown_files=['README.md','REPORT_KO.md','REVIEW_GUIDE_KO.md']),ensure_ascii=False))


if __name__=='__main__':main()
