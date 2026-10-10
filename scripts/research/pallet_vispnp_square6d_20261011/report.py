"""Korean report from complete public A0/A1/A2 and blocked-square evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import common as C
from . import verdict as V
from .figures import dataset

CODE = "../../../scripts/research/pallet_vispnp_square6d_20261011"
METRICS = (("T_cm", "위치 cm", 1), ("R_deg", "회전 °", 1),
           ("ADDsym_m", "ADDsym cm", 100), ("IoU3D", "IoU3D", 1))

SUCCESS_DEFINITIONS = (
    "[확인] 기존 [scripts/self_training/metrics.py:134-143](../../../scripts/self_training/metrics.py#L134)는 "
    "`T=100*||t_gt-t_pred||₂` cm, `R_single=degrees(acos(clip((trace(R_gt R_predᵀ)-1)/2,-1,1)))`를 계산해 "
    "strict `T<5 cm and R_single<5°`로 성공을 판정한다. 이번에는 동일한 T와, 고정 물리 축으로 변환한 회전에서 "
    "proper 대칭 그룹 G의 `R_sym=min_Q∈G degrees(acos(clip((trace((R_gt Q)ᵀ R_pred)-1)/2,-1,1)))`를 사용하여 "
    "strict `T<5 cm and R_sym<5°`로 판정한다. "
    "[현재 회전·위치 오차](../../../scripts/research/pallet_dim_conditioned_p_v1/pose.py#L39), "
    "[현재 성공 판정](../../../scripts/research/pallet_vispnp_square6d_20261011/verdict.py#L26). "
    "따라서 두 코드의 ‘5 cm·5° 성공률’은 회전 정의가 달라 같은 지표로 합치지 않는다. "
    "이번 실사 ALL/VIS는 둘 다 동일한 R_sym을 사용하므로 짝 비교의 정의는 일치한다."
)


def read(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def f(value, scale=1):
    return "미산출" if value is None else f"{value*scale:.4f}"


def ci(value, scale=1):
    return "미산출" if value is None else f"[{f(value[0],scale)}, {f(value[1],scale)}]"


def mean_sd(value,scale=1):
    return f"{f(value['mean'],scale)} ± {f(value['std'],scale)}"


def table(headers, rows):
    def escape(value):
        return str(value).replace("|",r"\|").replace("\n","<br>")
    return "\n".join(["| "+" | ".join(headers)+" |","| "+" | ".join(["---"]*len(headers))+" |"]+
                     ["| "+" | ".join(escape(v) for v in row)+" |" for row in rows])


METHOD = '''# VIS_RULE_V1과 square 참조 게이트

[확인] 새 기능은 코너를 옮기는 보정기가 아니라, 각 경로의 최종 2D 코너로 면의 방향을 계산하여 PnP fit에 쓰는 대응점만 줄이는 추론 후단이다. 원래 BASE, N3_DIM_SYM, SUBPIX, N3_THEN_SUBPIX의 좌표·중심 8·지원 정보·선택 검출·box·score·confidence·K·등록 치수는 유지한다. [visibility.py](../../../scripts/research/pallet_vispnp_square6d_20261011/visibility.py)와 [adapter.py](../../../scripts/research/pallet_vispnp_square6d_20261011/adapter.py).

![실제 적용 범위](figures/01_method_flow.png)

## 기하 마스크

[확인] x는 오른쪽, y는 아래인 원영상 pixel 좌표다. 기존 cuboid 인덱스는 0 좌상, 1 우상, 2 우하, 3 좌하, 4~7은 대응하는 뒤쪽 꼭짓점이며 8은 중심이다. 실제 F의 [기존 3D 정의](../../../scripts/paper/pose_metric_closure_v1/run_pose_evaluation.py#L38)는 X 오른쪽, Y 아래, Z 앞 방향을 사용한다.

```text
front (0,1,2,3)    back (4,7,6,5)
top (0,4,5,1)      bottom (3,2,6,7)
left (0,3,7,4)     right (1,5,6,2)
area = Σ(x_i*y_next - x_next*y_i)/2
facing = area > 0
visible(k) = any(incident face is facing or unknown)
retained = visible & original_usable
effective = original_usable if count(retained)<4 else retained
```

[확인] 규정된 면 순서는 기존 3D 축에서 inward winding이다. 첨부의 “외향 법선 기준” 설명과는 다르지만, 사전 지정된 2D 양수 규칙을 그대로 보존했으며 A0의 수치 parity를 검산했다. unknown 면은 하나라도 비지원·결측·비유한·[-1,-1] 코너를 포함한 경우다. incident unknown이 있으면 해당 코너는 보수적으로 보임으로 둔다. 넓이가 정확히 0이면 facing이 아니다. 새 threshold를 고르지 않았다. [A0.json](A0.json), [METHOD_LOCK.json](METHOD_LOCK.json).

[확인] 실제 지원 코너가 4개보다 적게 남으면 원래 전체 대응점으로 fallback한다. 이 상태는 후단 no_pose와 별개다. 마스크 생성에는 참조 자세, GT 코너, 사람 visibility를 사용하지 않는다. 사람 상태와의 교차표는 추론 입력이 아닌 A0 사후 진단이다.

## 동일 F에서 바뀌는 부분

[확인] `adapter.correspondence_mask(q9,K,mask8)`는 각 기존 F 호출 동안 `cv2.solvePnP` 및 `solvePnPRefineLM`에 전달되는 canonical 대응점만 같은 인덱스로 필터링한다. K, 원래 8점의 순서·값과 3D 좌표를 확인한다. `projectPoints`는 그대로 유지하므로 가설 점수는 **원래 9점 전체**로 계산한다. 따라서 제외한 코너도 기존 가설 선택 점수에는 남는다. fit과 selector의 적용 범위를 구분한다.

[확인] ALL 마스크는 OpenCV 인자를 그대로 전달한다. selector는 원래 q9를 받고 기존 W/D 후보, SQPnP, RefineLM, proper 대칭 오차를 유지한다. 미지원 코너나 잘못된 shape에서 adapter 계약이 성립하지 않으면 조용히 no_pose로 바꾸지 않고 중단한다. [adapter.py](../../../scripts/research/pallet_vispnp_square6d_20261011/adapter.py), [UNIT_CHECKS.json](UNIT_CHECKS.json).

[확인] 3D API의 치수 순서는 [W,H,D], 등록 문맥은 [W,D,H]다. 기존 cornerSubPix의 win=(5,5), zeroZone=(-1,-1), EPS|COUNT 40 및 0.001, 원래 Base 중심의 최종 대각선 1% cap을 보존한다. 이미 저장된 실사 네 경로는 좌표를 그대로 재사용한다. 새 VIS 코너 지표는 기존 ALL과 정확히 같아야 한다.

## 사전등록 주지표와 gate

[확인] 주비교는 N3_THEN_SUBPIX_VIS−N3_THEN_SUBPIX다. 혼동은 `rotation_deg>45 and abs(yaw_deg)>=60`, 성공은 `translation_cm<5 and rotation_deg<5`다. 등호 경계를 바꾸지 않는다. 각 seed에서 이진 판정을 먼저 하고 같은 ID의 세 판정을 평균한다. 평균 오차에 threshold를 적용하지 않는다.

[확인] SUPPORTED는 주지표 하나 이상의 개선 CI가 0을 제외하고, 다른 주지표에 유의한 악화가 없으며, 해당 개선 방향이 2개 이상의 seed에서 같을 때다. 혼동 CI 하한>0 또는 성공 CI 상한<0이면 WORSENED다. 나머지는 UNRESOLVED다. 주비교에 미산출 자세가 하나라도 있으면 NOT_ESTIMABLE로 결론을 유보한다. 전체 분모의 기술적 이진 비율에서는 미산출을 false로 기록하되, 그것을 개선 증거로 쓰지 않는다. [verdict.py](../../../scripts/research/pallet_vispnp_square6d_20261011/verdict.py).

[확인] A1의 고정 frame bootstrap에서 WORSENED/NOT_ESTIMABLE이면 A2 실사를 실행하지 않는다. A1의 scenario cluster CI는 별도 보조 진단이다. A2는 원래 13세션 paired cluster bootstrap 10,000회, seed 20260917을 사용한다. 다중비교 보정은 하지 않는다. 결과를 보고 지표, threshold, 주경로, seed, 사례 규칙을 바꾸지 않는다.

[확인] 원 실행의 A0는 실제 ALL F 1,276회와 정확한 parity를 확인했지만, 새 A0 코드의 실행 직전 SHA는 당시 기록하지 못했다. A1/A2의 핵심 9개 소스는 새 VIS 결과가 나오기 전에 SHA를 봉인하고 불변성을 검사했다. 이후 preflight에 추가한 pre-A0 SHA 기록은 향후 재현의 provenance 보완이며 원 A0의 누락을 소급해서 채우지 않는다. [SOURCE_LOCK.json](SOURCE_LOCK.json), [A0.json](A0.json), [독립 검산](VERIFICATION.json).

## B: 같은 참조 절차라는 전제의 불일치

[확인] B1의 화면 내 직접 수동 코너≥4 규칙은 구현했지만, 현재 직사각형 참조 생성기는 finite 코너≥6과 전체 `keypoint_annotations.xy`를 사용한다. source를 직접 수동점으로 필터링하지 않고 두 W/D 가설을 SQPnP→RefineLM으로 풀어 평균 재투영 잔차가 작은 것을 고른다. 기존 5 px 품질바는 검토용이며 자동 제외하지 않는다. 생성기 자체에는 LOO/robust-z가 없다. [SQUARE_INPUT_AUDIT.json](SQUARE_INPUT_AUDIT.json)의 source SHA와 행 번호를 근거로 한다.

[확인] 별도 legacy `qa_risk.py`는 non-sentinel projected_cuboid로 PnP를 풀며 4·5점에서는 ITERATIVE, 6점 이상에서는 SQPnP를 사용한다. LOO의 남은 점<4이면 NaN이다. 최대 LOO 및 중앙 재투영 잔차를 bbox diagonal로 정규화하고 1.4826×MAD(MAD=0이면 nanstd)로 robust-z를 계산한다. hard flag 또는 z>5는 RED, z>3은 AMBER다. 별도 `audit_gt_data.py`의 저장 pose 비교 QA와 섞지 않는다.

[확인] 따라서 직접 수동점만으로 B1 적격 118장 모두를 “같은 참조 생성+동일 LOO QA”로 완료할 수 없다. 새 절차·threshold·예외를 정하거나 PnP 파생점으로 부족분을 채우지 않았다. `SQUARE_REFERENCE_POSES.json`은 차단 상태와 빈 `frames`를 기록하며 실제 참조 자세 파일로 해석하면 안 된다. B4 검수 sheet나 사용자 축 승인 요청에 도달하지 않았고 승인 전 B5 평가도 실행하지 않았다.
'''
METHOD = METHOD.replace("## 사전등록 주지표와 gate\n", "## 사전등록 주지표와 gate\n\n"+SUCCESS_DEFINITIONS+"\n", 1)


def reproduction():
    return '''# 재현 및 공개 산출물 검산

[확인] 기존 학습 환경을 사용한다. 설치, 재학습, 새 RGB·합성 데이터·수동 주석 생성은 없다. 정확도 재현에는 원본 비공개 입력이 필요하며 공개 JSON의 통계·그림 검산은 원본 RGB나 checkpoint 없이 가능하다.

```bash
export PALLET_PYTHON="/path/to/existing/pallet-pose/bin/python"
export PALLET_SOURCE_ROOT="/path/to/original/pallet-pose"
export PALLET_BASELINE_ROOT="/path/to/existing/baseline-inputs"
export PALLET_PRIVATE_OUTPUT="/path/to/private/replay-records"
export PALLET_LEGACY_QA_SOURCE="/path/to/existing/qa_risk.py"
export PALLET_VIS_OUTPUT="/path/to/nonexistent/fresh-vis-results"
export PYTHONDONTWRITEBYTECODE=1
# 게시한 저장소 checkout의 루트에서 실행한다.
"$PALLET_PYTHON" -B -m scripts.research.pallet_vispnp_square6d_20261011.replay --output "$PALLET_VIS_OUTPUT"
"$PALLET_PYTHON" -B -m scripts.research.pallet_vispnp_square6d_20261011.square_audit --doc "$PALLET_VIS_OUTPUT" --legacy-qa "$PALLET_LEGACY_QA_SOURCE"
"$PALLET_PYTHON" -B -m scripts.research.pallet_vispnp_square6d_20261011.figures --doc "$PALLET_VIS_OUTPUT" --private-cases "$PALLET_PRIVATE_OUTPUT"
"$PALLET_PYTHON" -B -m scripts.research.pallet_vispnp_square6d_20261011.verify --doc "$PALLET_VIS_OUTPUT"
"$PALLET_PYTHON" -B -m scripts.research.pallet_vispnp_square6d_20261011.report --doc "$PALLET_VIS_OUTPUT"
```

[확인] `replay --output`은 존재하지 않는 새 출력 디렉터리에서 `preflight`(방법·마스크 봉인, A0 ALL parity)→소스 SHA lock→`synth`(A1 ALL/VIS·집계·gate)→gate 통과 시에만 `real`과 `summarize_real`을 호출한다. A1 WORSENED/NOT_ESTIMABLE이면 REAL F를 호출하지 않는다. 기존 게시 또는 중단 산출물을 덮어쓰지 않는다. 위 명령은 학습 모델을 새로 실행하지 않으며 고정된 예측과 기존 영상·기하 입력을 사용한다.

[확인] `square_audit`와 `verify`의 새 결과는 별도로 생성된다. `verify`에서 비공개 preservation 인자를 생략하면 공개 수치 검산은 수행하고 원 checkout 보존 검사는 NOT_RUN_PUBLIC_ONLY로 표시한다. 이번 실행의 보존 기록은 원 세션의 START/snapshot을 필요로 하므로 다른 사용자 재현에서 그 기록을 만들어 낸 것처럼 보고하지 않는다.

```bash
# 원본 RGB·checkpoint 없이 게시한 공개 산출물을 다시 검산한다.
export PALLET_PUBLIC_DOC="$PWD/_docs/experiments/pallet_vispnp_square6d_20261011"
export PALLET_RECHECK_OUTPUT="/path/to/nonexistent/public-recheck.json"
"$PALLET_PYTHON" -B -m scripts.research.pallet_vispnp_square6d_20261011.verify --doc "$PALLET_PUBLIC_DOC" --output "$PALLET_RECHECK_OUTPUT"
# 아래 두 명령은 저장된 수치의 파생 PNG와 Markdown만 재생성한다.
"$PALLET_PYTHON" -B -m scripts.research.pallet_vispnp_square6d_20261011.figures --doc "$PALLET_PUBLIC_DOC"
"$PALLET_PYTHON" -B -m scripts.research.pallet_vispnp_square6d_20261011.report --doc "$PALLET_PUBLIC_DOC"
```

[확인] square_audit는 기존 `qa_risk.py`의 텍스트와 AST를 읽고 원래 threshold를 기록하며 그 QA나 PnP를 실행하지 않는다. 기존 output이 있으면 보존하고 멈춘다. source·manual JSON·이미지 SHA를 읽기 전후 비교한다. B2/B3 계약 충돌을 bypass하거나 B4 승인 없이 6D 평가하는 명령은 없다.

[확인] 공개 PNG는 수치 그림이다. 실제 RGB 사례는 `--private-cases`를 명시할 때에만 저장하며 Git 저장소 내부 출력은 거부한다. A1은 기존 SOURCE_MANIFEST의 heldout RGB에서 reflection padding 100 px를 제거한 뒤 raw qFinal을 그대로 겹친다. A2는 고정 INPUT_AUDIT의 실사 원영상 binding을 사용한다. `cases_A1_SYNTH.png` 또는 `cases_A2_REAL.png`와 receipt는 비공개 local directory에 남긴다. 정사각형의 새 참조 overlay는 생성하지 않았다.
'''


def report(doc):
    real,prefix,m,p,failures,verdict,_,_=dataset(doc)
    a0=read(doc/"A0.json");square=read(doc/"SQUARE_INPUT_AUDIT.json");index=read(doc/"FIGURE_INDEX.json")
    if a0["status"]!="PASS" or len(index.get("figures",[]))!=7:
        raise RuntimeError("A0 PASS and all seven scientific figures are required before reporting")
    phase="A2 실사" if real else "A1 합성"
    label=verdict["verdict"];primary=p["primary"]
    lines=[f"[확인] A={label} ({phase}); B=BLOCKED_REFERENCE_PROCEDURE_CONFLICT. 사전등록 판정과 참조 절차 불일치를 기록했다.",""]
    if real:
        b=m["seed_mean"]["ALL"][V.PRIMARY_METHOD];v=m["seed_mean"]["ALL"][V.PRIMARY_METHOD+"_VIS"]
        sm=read(doc/"SYNTH_METRICS.json");sp=read(doc/"SYNTH_PAIRED.json")
        sb=sm["seed_mean"]["ALL"][V.PRIMARY_METHOD];sv=sm["seed_mean"]["ALL"][V.PRIMARY_METHOD+"_VIS"]
        lines += [f"[확인] **SUPPORTED는 실사 주경로의 5 cm·5° 성공률 증가에 한정한다.** {f(b['rates']['success_rate']['rate'],100)}%→{f(v['rates']['success_rate']['rate'],100)}%, Δ{f(primary['success_rate']['delta'],100)} %p {ci(primary['success_rate']['CI95'],100)}이며 세 seed 모두 같은 개선 방향이다. 혼동률 Δ{f(primary['confusion_rate']['delta'],100)} %p {ci(primary['confusion_rate']['CI95'],100)}는 미해결이다.",
            f"[확인] T 평균은 {f(b['metrics']['T_cm']['mean'])}→{f(v['metrics']['T_cm']['mean'])} cm로 감소했지만 R 평균 {f(b['metrics']['R_deg']['mean'])}→{f(v['metrics']['R_deg']['mean'])}°, T 중앙값 {f(b['metrics']['T_cm']['median'])}→{f(v['metrics']['T_cm']['median'])} cm, T P90 {f(b['metrics']['T_cm']['P90'])}→{f(v['metrics']['T_cm']['P90'])} cm는 악화했다. 큰 실패를 모두 포함한 결과이며 전반적인 6D 정확도 개선으로 확대하지 않는다.",
            f"[확인] 선행 합성에서는 직렬 성공률 {f(sb['rates']['success_rate']['rate'],100)}%→{f(sv['rates']['success_rate']['rate'],100)}%로 세 seed 모두 감소했다(Δ{f(sp['primary']['success_rate']['delta'],100)} %p, CI {ci(sp['primary']['success_rate']['CI95'],100)}). 합성 판정 UNRESOLVED는 안전성 또는 동등성을 확증한 결과가 아니다.",""]
    lines += ["# 기하 자기가림 코너를 제외한 PnP와 GREEN0918 참조 감사","",
        f"[확인] 주비교 직렬 VIS−ALL의 혼동률 차이는 {f(primary['confusion_rate']['delta'],100)} %p {ci(primary['confusion_rate']['CI95'],100)}, 5 cm·5° 성공률 차이는 {f(primary['success_rate']['delta'],100)} %p {ci(primary['success_rate']['CI95'],100)}다. 상세 판정은 [{prefix+'VERDICT.json'}]({prefix+'VERDICT.json'})에 있다.","",
        "[확인] 폴더 이름의 20261011은 요청된 실험 namespace다. 실제 실행 client date는 METHOD_LOCK.json에 기록된 2026-10-10이며 날짜를 소급해서 표시하지 않는다. 새 학습·새 합성 RGB·새 촬영·새 수동 주석은 모두 0이다. 기존 실험 폴더와 원고·LaTeX·PDF는 보존했다.","",
        "## A0: 전제와 실제 parity","",
        table(["검사","실제 결과","근거"],[
            ["319장×네 경로 seed1 ALL F 재실행",a0["actual_F_calls"],"A0.json"],
            ["자세 지표 최대 절대차",a0["max_pose_metric_absolute_delta"],"허용≤1e−7"],
            ["모든 R/t 및 코너 필드",a0["all_R_t_and_corner_fields_PASS"],"기존 원행과 대조"],
            ["N3 seed1 front facing",a0["visibility_parity"]["N3_DIM_SYM"]["front_facing"],"319/319"],
            ["N3 숨김 1개/2개",json.dumps(a0["visibility_parity"]["N3_DIM_SYM"]["hidden_counts"]),"첨부 사후 진단과 정확 일치"],
            ["BASE 숨김 1개/2개",json.dumps(a0["visibility_parity"]["BASE"]["hidden_counts"]),"첨부 사후 진단과 정확 일치"]]),"",
        "[확인] 사람 visibility와의 교차표는 마스크를 먼저 SHA 봉인한 뒤 읽었다. 원래 코드의 3D 면 winding은 첨부의 외향 설명과 달랐으므로 규정된 2D 넓이 양수 규칙을 그대로 적용하고 수치 parity로 확인했다. [A0.json](A0.json), [COORDINATES_SEAL.json](COORDINATES_SEAL.json), [METHOD_LOCK.json](METHOD_LOCK.json).","",
        SUCCESS_DEFINITIONS,"",
        table(["경로","상태","숨김","보임"],[[base,state,values["hidden"],values["visible"]]
            for base,entry in a0["visibility_parity"].items() for state,values in entry["direct_index_diagnostic"].items()]),"",
        "## 실행 gate와 분모","",
        ("[확인] 합성 gate를 통과한 뒤 실사 A2를 한 번 실행했다. ALL은 고정된 기존 3,828행, VIS는 같은 319장·네 경로·세 seed의 새 F 결과다. 원래 13세션, clean 153/moderate 92/severe 74 및 plastic 194/wood 125를 유지했다. 실사는 반복 개발 DEV이며 독립 확증 holdout이 아니다."
         if real else
         f"[확인] 합성 A1 판정은 {label}이며 사전등록 gate에 따라 실사 A2를 실행하지 않았다. 합성 결과를 REAL319 결과로 바꾸어 표시하지 않는다. A0의 실사 ALL parity는 입력 계약 검사이며 VIS 정확도 평가와 구분한다. 합성 held-out도 후속 개발 진단에 재사용된 자료로 독립 확증 자료가 아니다."),"",
        f"[확인] 이번 정확도 표의 모집단은 {m['population']}, 원래 영상 {m['unique_frames']}장, 고정 seed 1·2·3이다. 각 seed의 이진 결과를 ID별로 평균하므로 3배 영상 수를 독립 분모로 취급하지 않는다. 미산출 자세의 T/R/ADD/IoU 통계는 available-only로 n을 명시한다. 주지표의 전체 분모 기술 비율에서는 미산출 false를 기록하지만 주비교에 미산출이 있으면 NOT_ESTIMABLE로 중단한다.","",
        "[확인] SD는 표본 오차 산포(ddof=1)이며 평균의 95% CI, 표준오차, seed 간 산포와 구분한다. 큰 유한 오차나 나쁜 영상, seed, 그룹을 제외하거나 clipping/winsorization하지 않았다. 두 경로의 코너 지표는 그대로 같으며 변화는 pose fit에서 발생한다.","",
        ]
    if real:
        sm=read(doc/"SYNTH_METRICS.json");sp=read(doc/"SYNTH_PAIRED.json");sv=read(doc/"SYNTH_VERDICT.json")
        lines += ["### A1 합성 gate의 실제 선행 결과","",
            f"[확인] SYNTH_HELDOUT 1,985장·1,505 scenario에서 ALL/VIS 각각 23,820행을 실제 평가했다. G38 1,025장, P0 480장, TEX 480장과 C1/C2를 고정했다. A1 판정은 **{sv['verdict']}**이며 독립 확증이 아니라 harm gate의 미해결 결과로 A2 진입을 허용했다. 합성 frame primary와 scenario 보조 CI를 구분한다.","",
            table(["경로","ALL 혼동/성공률 %","VIS 혼동/성공률 %","Δ혼동 %p [CI]","Δ성공 %p [CI]"],
                [[base," / ".join(f(sm["seed_mean"]["ALL"][base]["rates"][k]["rate"],100) for k in ("confusion_rate","success_rate")),
                  " / ".join(f(sm["seed_mean"]["ALL"][base+"_VIS"]["rates"][k]["rate"],100) for k in ("confusion_rate","success_rate")),
                  *[f(entry[k]["delta"],100)+" "+ci(entry[k]["CI95"],100) for k in ("confusion_rate","success_rate")]]
                 for base in C.METHODS for entry in [sp["seed_mean"]["ALL"][base+"_VIS__minus__"+base]]]),"",
            table(["주비교 seed","Δ혼동 %p [frame CI]","Δ성공 %p [frame CI]"],
                [[str(s),*[f(entry[k]["delta"],100)+" "+ci(entry[k]["CI95"],100) for k in ("confusion_rate","success_rate")]]
                 for s in C.SEEDS for entry in [sp["by_seed"][str(s)]["ALL"][V.PRIMARY_CONTRAST]]]),"",
            table(["seed-mean 主지표","scenario cluster 보조 Δ%p [CI]"],
                [[k,f(entry["delta"],100)+" "+ci(entry["CI95"],100)]
                 for k in ("confusion_rate","success_rate") for entry in [sp["primary"][k]["scenario_cluster_secondary"]]]),"",
            "[확인] 합성의 모든 seed·경로·고정 하위집단 통계, 실패 ID와 gate는 [SYNTH_METRICS.json](SYNTH_METRICS.json), [SYNTH_PAIRED.json](SYNTH_PAIRED.json), [SYNTH_FAILURES.json](SYNTH_FAILURES.json), [SYNTH_VERDICT.json](SYNTH_VERDICT.json)에 그대로 보존한다. 실사 수치와 합산하지 않는다.",""]
    lines += ["## 모든 경로·seed의 정확도","","![모든 정확도](figures/02_all_vis_accuracy.png)",""]
    ss=[(str(s),m["by_seed"][str(s)]["ALL"]) for s in C.SEEDS]+[("seed-mean",m["seed_mean"]["ALL"])]
    names=[name for base in C.METHODS for name in (base,base+"_VIS")]
    lines += [table(["seed","경로","자세 산출/전체","미산출","T cm 평균±SD [CI]","R ° 평균±SD [CI]","ADDsym cm 평균±SD [CI]","IoU3D 평균±SD [CI]"],
        [[seed,name,f"{summary['available']}/{summary['frames']}",summary["unavailable"],
          *[mean_sd(summary["metrics"][key],scale)+" "+ci(summary["metrics"][key]["CI95"],scale) for key,_,scale in METRICS]]
         for seed,scope in ss for name in names for summary in [scope[name]]]),"",
        table(["seed","경로","혼동률 % [CI]","5 cm·5° 성공률 % [CI]","분모"],
            [[seed,name,*[f(summary["rates"][key]["rate"],100)+" "+ci(summary["rates"][key]["CI95"],100) for key in ("confusion_rate","success_rate")],summary["frames"]]
             for seed,scope in ss for name in names for summary in [scope[name]]]),"",
        "<details><summary>표본분산·SD·중앙값·P90·최대·유효 n: 모든 경로와 seed</summary>",""]
    for key,title,scale in METRICS:
        lines += [f"**{title}**","",table(["seed","경로","n","표본분산","SD","중앙값","P90","최대"],
            [[seed,name,stat["n"],f(stat["variance"],scale*scale),*[f(stat[k],scale) for k in ("std","median","P90","max")]]
             for seed,scope in ss for name in names for stat in [scope[name]["metrics"][key]]]),""]
    lines += ["</details>","","[확인] ADDsym m→cm의 평균·SD·중앙값·P90·최대·CI는 ×100, 분산은 ×10,000이다. seed-mean 숫자 오차는 같은 ID의 세 scalar 오차를 평균한 분포이며 R/t를 평균해 새 pose를 만들지 않는다. 코너 PCK는 세 seed의 기존 비율 평균이고 VIS에 따른 좌표 변화는 0이다.","",
        "[확인] BASE와 SUBPIX는 고정 좌표이므로 각 seed에서 같은 baseline을 반복 표기했다. seed 1·2·3은 N3 checkpoint 구분이며 새 N3 forward를 수행하지 않고 각 seed의 기존 저장 예측을 사용했다.","",
        "## 사전등록 주비교 및 모든 짝 비교","","![주비교 CI](figures/03_primary_paired_ci.png)",""]
    pair_scopes=[(str(s),p["by_seed"][str(s)]["ALL"]) for s in C.SEEDS]+[("seed-mean",p["seed_mean"]["ALL"])]
    lines += [table(["seed","VIS−ALL 경로","혼동률 Δ%p [CI]","성공률 Δ%p [CI]","주비교 개선 seed 수 혼동/성공"],
        [[seed,base,*[f(entry[key]["delta"],100)+" "+ci(entry[key]["CI95"],100) for key in ("confusion_rate","success_rate")],
          f"{entry['confusion_rate']['improved_seeds']}/{entry['success_rate']['improved_seeds']}"]
         for seed,scope in pair_scopes for base in C.METHODS for entry in [scope[base+"_VIS__minus__"+base]]]),"",
        table(["seed","VIS−ALL 경로","ΔT cm [CI]","ΔR ° [CI]","ΔADDsym cm [CI]","ΔIoU3D [CI]"],
            [[seed,base,*[f(entry["metrics"][key]["delta"],scale)+" "+ci(entry["metrics"][key]["CI95"],scale) for key,_,scale in METRICS]]
             for seed,scope in pair_scopes for base in C.METHODS for entry in [scope[base+"_VIS__minus__"+base]]]),"",
        "[확인] CI는 짝 bootstrap으로 계산한다. A1의 primary는 frame, scenario cluster는 보조이며 A2는 원래 13세션 cluster다. CI가 0을 포함하는 것을 동등성 증거로 해석하지 않고 다중비교 보정은 하지 않았다. 학습 seed의 모집단 변동까지 세 seed로 확증하지 않는다.","",
        table(["범위","bootstrap"],[[scope,json.dumps(value,ensure_ascii=False)] for scope,value in p["bootstrap"].items()]),"",
        "## 하위집단·손상·가설 전환·미산출","","![모든 하위집단](figures/04_subgroups.png)",""]
    for scope,group in m["seed_mean"].items():
        if scope=="ALL":continue
        lines += [f"**{scope}**","",table(["경로","전체/산출 자세","T cm 평균±SD [CI]","R ° 평균±SD [CI]","ADDsym cm 평균±SD [CI]","혼동률 %","성공률 %"],
            [[name,f"{group[name]['frames']}/{group[name]['available']}",*[mean_sd(group[name]["metrics"][key],scale)+" "+ci(group[name]["metrics"][key]["CI95"],scale) for key,_,scale in METRICS[:3]],
              f(group[name]["rates"]["confusion_rate"]["rate"],100),f(group[name]["rates"]["success_rate"]["rate"],100)] for name in names]),""]
    lines += ["<details><summary>각 seed·경로의 고정 숨김 코너 수별 진단</summary>","",
        "[확인] 숨김 수의 집단은 해당 seed·경로의 봉인한 마스크로 고정했다. 따라서 서로 다른 경로나 seed의 같은 숨김 수를 같은 ID 집단으로 간주하지 않는다. 각 셀의 ID, bootstrap 및 전체 통계는 METRICS의 diagnostics.hidden_count_by_seed에 보존한다.","",
        table(["seed","경로","숨김 수","전체/ALL·VIS 산출 자세","ALL T/R 평균","VIS T/R 평균","ALL 혼동/성공 %","VIS 혼동/성공 %","손상/회복"],
            [[str(s),base,count,f"{r['frames']}/{r['ALL']['available']}·{r['VIS']['available']}",
              " / ".join(f(r["ALL"]["metrics"][k]["mean"]) for k in ("T_cm","R_deg")),
              " / ".join(f(r["VIS"]["metrics"][k]["mean"]) for k in ("T_cm","R_deg")),
              " / ".join(f(r["ALL"]["rates"][k]["rate"],100) for k in ("confusion_rate","success_rate")),
              " / ".join(f(r["VIS"]["rates"][k]["rate"],100) for k in ("confusion_rate","success_rate")),
              f"{r['damage']['success_to_failure']}/{r['damage']['failure_to_success']}"]
             for s in C.SEEDS for base,counts in m["diagnostics"]["hidden_count_by_seed"][str(s)].items()
             for count,r in counts.items()]),"","</details>",""]
    lines += ["![손상과 전환](figures/05_hidden_damage_switch.png)","",
        table(["seed","경로","성공→실패","실패→성공","W/D 전환","<4 fallback","VIS no_pose","숨김 코너 수 분포"],
            [[str(s),base,r["success_to_failure"],r["failure_to_success"],r["hypothesis_changes"],r["fallback_lt4"],len(r["unavailable_VIS_ids"]),json.dumps(r["hidden_count_distribution"])]
             for s in C.SEEDS for base in C.METHODS for r in [failures["per_seed"][str(s)][base]]]),"",
        f"[확인] 손상·회복은 같은 영상의 사전 정의 5 cm·5° 성공 여부로 판단했다. 가설 전환을 자동으로 올바른 복구로 판정하지 않는다. 전체 실패 ID와 전환별 ΔT/ΔR은 [{prefix+'FAILURES.json'}]({prefix+'FAILURES.json'})에 남겼다. fallback과 pose 미산출을 합치지 않았다.","",
        "## 실제 사례: 나쁜 결과까지 보존","","![실제 코너와 마스크](figures/06_numeric_cases.png)","",
        table(["잠금 규칙","ID","seed","ALL T cm/R °","VIS T cm/R °","ΔT cm","ΔR °"],
            [[r["rule"],r["id"],r["seed"],f(r["ALL_T_cm"])+" / "+f(r["ALL_R_deg"]),
              f(r["VIS_T_cm"])+" / "+f(r["VIS_R_deg"]),f(r["delta_T_cm"]),f(r["delta_R_deg"])] for r in index["examples"]]),"",
        "[확인] seed 1의 성공 회복·손상 부분집합에서 최대 T 개선·악화, 전체 최대 T 개선·악화를 선택했고 동률은 ID 사전순이다. 없는 범주는 없다고 그림에 표시했다. ALL/VIS의 코너는 같으므로 그림은 fit 유지·제외 코너를 보여준다. 최대 악화 영상도 남겼다. 공개 PNG는 수치 좌표 그림이며 실제 RGB overlay는 비공개 `"+("cases_A2_REAL.png" if real else "cases_A1_SYNTH.png")+"`에 저장했다. 원본 RGB는 게시하지 않는다.","",
        "## B0/B1과 B2/B3의 실제 차단","","![수동 코너 분포](figures/07_square_manual_counts.png)",""]
    sq=square["counts"]
    lines += [table(["검사","실제 값"],[
        ["영상/세션",f"{sq['frames']}/{sq['sessions']}"],["선언/화면 내 직접 수동 코너",f"{sq['manual_declared_corners']}/{sq['manual_in_frame_corners']}"],
        ["화면 내 수동 코너 분포",json.dumps(sq["manual_in_frame_count_distribution"])],["B1 ≥4 적격",sq["B1_eligible_manual_in_frame_ge4"]],
        ["기존 참조 ≥6 만족",sq["existing_reference_min6_satisfied"]],["B1 통과·기존 ≥6 부족",sq["B1_eligible_but_existing_min6_incompatible"]],
        ["B1 적격의 6점까지 산술 부족 합",sq["B1_eligible_existing_min6_shortfall_corners"]],["수동 4점의 기존 LOO 불가",sq["B1_eligible_with_unsupported_manual_only_LOO4"]],
        ["기존 canonical_pose",sq["canonical_pose_present"]],["새 참조/수동 주석/축 검수/6D 평가","0/0/0/0"]]),"",
        "[확인] 112는 단순한 산술 부족분이며 추가 주석 요청이나 작업 목표가 아니다. 029844는 화면 내 3점으로 B1 부적격이다. 029710 코너 0 및 029844 코너 4는 직접 수동점이지만 화면 밖이다. 원본 238개 RGB/JSON binding과 기존 절차 source를 포함한 입력 전후 SHA 검사는 SQUARE_INPUT_AUDIT.json에 있다. 좌표 수 계산에 예측을 사용하지 않았다.","",
        "[확인] 실제 직사각형 생성기는 `build_geometry_resolved_pose_gt.py:102-108`의 finite≥6, `:55-60`의 SQPnP→RefineLM, `:122`의 재투영 최소를 사용한다. manifest는 `build_axis_review_manifest.py:126-127`에서 source를 거르지 않고 xy를 가져온다. 기존 319장 코너 source는 "+json.dumps(square["existing_rectangular_procedure"]["source_corner_counts_actual_319_manifest"],ensure_ascii=False)+"이다. [기존 생성기](../../../scripts/paper/pose_metric_closure_v1/build_geometry_resolved_pose_gt.py#L102), [manifest 생성기](../../../scripts/paper/pose_metric_closure_v1/build_axis_review_manifest.py#L126), [원 source SHA와 행 번호](SQUARE_INPUT_AUDIT.json).","",
        "[확인] legacy qa_risk.py:48/75/80은 직접 manual_click만이 아닌 non-sentinel projected_cuboid를 풀고, :91은 LOO 남은 점<4를 NaN으로 둔다. :124-133의 원래 scale과 RED>5/AMBER>3을 찾았으므로 threshold 파일 부재가 차단 원인은 아니다. 별도 audit_gt_data의 stored excess 1 px, gross 10 px, 회전 0.5°, 위치 0.01 m 검사와도 구분한다. 직접 수동점만 허용한 B와 기존 생성·QA 입력/최소점 계약이 일치하지 않는 것이 원인이다.","",
        "[확인] C4 proper 대칭은 기존 계약에 있으며 중심 8을 고정한다. 정사각형에서는 90° W/D 회전이 등가류이므로 직사각형의 90° 혼동 지표를 그대로 해석할 수 없다. 이번에는 참조를 새로 생성하지 않아 6D 수치를 보고하지 않는다. [SQUARE_REFERENCE_POSES.json](SQUARE_REFERENCE_POSES.json)은 빈 frames와 차단 이유를 기록한다. B4 승인 요청·참조 overlay를 만들지 않았고 B5 평가는 실행하지 않았다.","",
        "## 검산·실행량·한계·재현","",
        f"[확인] 모든 원행·집계·실패 자료는 [{prefix+'METRICS.json'}]({prefix+'METRICS.json'}), [{prefix+'PAIRED.json'}]({prefix+'PAIRED.json'}), [{prefix+'FAILURES.json'}]({prefix+'FAILURES.json'}) 및 [FIGURE_INDEX.json](FIGURE_INDEX.json)에 연결된다. 독립 scalar 모멘트·분위수·고정 bootstrap·봉인 마스크·실제 solver 전달점 검산은 [VERIFICATION.json](VERIFICATION.json), 실제 호출 횟수는 [A0.json](A0.json), [SYNTH_EXECUTION.json](SYNTH_EXECUTION.json)"+("과 [REAL_EXECUTION.json](REAL_EXECUTION.json)" if real else "")+"에 기록했다. [재현 명령](REPRODUCE.md).","",
        "[확인] 원 A0의 새 코드 실행 직전 SHA는 기록하지 못했다. A0는 실제 1,276회 ALL F parity가 정확히 통과했으며 A1/A2의 핵심 9개 파일은 새 결과 전에 [SOURCE_LOCK.json](SOURCE_LOCK.json)으로 봉인했다. 현재 preflight의 추가 pre-A0 SHA 기록은 향후 재현의 보완이다. 원 A0 결과를 소급 변경하거나 누락을 없는 것으로 표시하지 않는다.","",
        "[추정] 코너를 일부 제외하면 잘못된 대응점의 영향이 줄 수 있지만, 남은 기하가 덜 안정적이거나 기존 9점 가설 점수와 fit 점 집합이 달라 자세가 악화할 수도 있다. 이는 가능한 해석이며 효과가 있다고 확정하는 설명이 아니다. 실제 전체 결과, seed 방향, CI, 손상과 전환을 먼저 판단한다.","",
        "[확인] 개발 자료 재사용, geometric proxy 참조, 세 seed, 합성 frame의 상관성, 13개 실사 세션 또는 합성 scenario의 제한, 미보정 다중비교를 한계로 유지한다. 합성·실사·정사각형의 실행 상태와 분모를 서로 합치지 않는다. 부정 결과와 사전등록 중단도 완료된 결과로 기록하며 gate 이후의 미실행을 실행한 것처럼 표시하지 않는다.","",
        "[방법](METHOD_KO.md) · [재현](REPRODUCE.md) · [입력/마스크 봉인](COORDINATES_SEAL.json) · [A0](A0.json) · [square 감사](SQUARE_INPUT_AUDIT.json)"]
    verification=read(doc/"VERIFICATION.json")
    if verification:
        lines += ["",f"[확인] 게시한 독립 검산의 실제 상태는 **{verification['status']}**, scalar 수치 비교 {verification['independent_numeric_comparisons']:,}개다. 분산·SD·분위수·CI·짝 차이·이진 비율·마스크·solver 전달점과 원 입력 불변성을 점검했다. 검산 자체의 F·모델·학습 호출은 0이다. 구현 SHA `{verification['source_sha256']}`는 [verify.py]({CODE}/verify.py)와 [VERIFICATION.json](VERIFICATION.json)에 연결된다.",""]
    execution=read(doc/"SYNTH_EXECUTION.json")
    real_execution=read(doc/"REAL_EXECUTION.json")
    lines += [table(["단계","실제 F 호출","원 영상 분모","근거"],[
        ["A0 ALL parity",a0["actual_F_calls"],319,"A0.json"],
        ["A1 ALL+VIS",execution["actual_F_calls"],1985,"SYNTH_EXECUTION.json"],
        ["A2 VIS",real_execution["actual_F_calls"] if real_execution else 0,319 if real_execution else "gate에 따라 미실행","REAL_EXECUTION.json" if real_execution else "SYNTH_VERDICT.json"],
        ["B0/B1 입력 감사·참조 생성·B5 평가",0,119,"SQUARE_INPUT_AUDIT.json: 새 참조/평가 0"],
        ["그림·보고서·독립 검산",0,"저장된 실제 결과","추가 F/모델 호출 없음"]]),""]
    return "\n".join(lines)+"\n",label,phase,prefix


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--doc",type=Path,default=C.DOC);args=parser.parse_args();doc=args.doc
    text,label,phase,prefix=report(doc)
    (doc/"REPORT_KO.md").write_text(text.rstrip()+"\n");(doc/"METHOD_KO.md").write_text(METHOD);(doc/"REPRODUCE.md").write_text(reproduction())
    index=read(doc/"FIGURE_INDEX.json")
    limited_summary=("SUPPORTED는 실사 주경로의 5 cm·5° 성공률 증가에 한정한다. 회전 평균·위치 중앙값/P90은 악화했고 합성 성공률은 세 seed 모두 감소했다. 전체 6D 지표 개선으로 확대하지 않는다."
                     if phase=="A2 실사" else "A1 합성 gate에 따라 실사 A2를 실행하지 않았다. 합성 결과를 실사 정확도나 안전성 확증으로 확대하지 않는다.")
    README=f'''# VIS PnP / square 6D 감사 (namespace 20261011)

[확인] A **{label}** ({phase}), B **BLOCKED_REFERENCE_PROCEDURE_CONFLICT**. 실제 실행일은 2026-10-10이다. 사전등록 gate와 동일 참조 절차의 불일치를 기록하며 새 학습·새 주석 없이 종료했다.

{limited_summary}

[한국어 보고서](REPORT_KO.md) · [방법](METHOD_KO.md) · [재현](REPRODUCE.md) · [사전등록 lock](METHOD_LOCK.json) · [A0 parity](A0.json)

[정확도]({prefix}METRICS.json) · [짝 비교]({prefix}PAIRED.json) · [판정]({prefix}VERDICT.json) · [실패·손상·전환 ID]({prefix}FAILURES.json) · [square 입력 감사](SQUARE_INPUT_AUDIT.json) · [square 차단 기록](SQUARE_REFERENCE_POSES.json) · [그림 manifest](FIGURE_INDEX.json)

원본 RGB는 공개하지 않는다. 공개 사례는 실제 수치 코너·마스크이며 비공개 실제 RGB sheet와 구분한다. 좋은 seed나 영상으로 다시 선택하지 않고 나쁜 결과도 유지했다.

'''+table(["PNG","내용"],[[f"[{r['path']}]({r['path']})",r["title"]] for r in index["figures"]])+f"\n\n[새 코드]({CODE}/) · [기하 규칙]({CODE}/visibility.py) · [대응점 adapter]({CODE}/adapter.py) · [단위검사]({CODE}/test_adapter.py) · [B 감사]({CODE}/square_audit.py)\n"
    (doc/"README.md").write_text(README)
    print(json.dumps(dict(A=label,phase=phase,B="BLOCKED_REFERENCE_PROCEDURE_CONFLICT"),ensure_ascii=False))


if __name__=="__main__":main()
