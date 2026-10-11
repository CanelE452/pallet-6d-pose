"""Korean paper-table report from sealed saved results; no inference or fitting."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
DOC=Path(os.environ.get("PALLET_PAPER_OUTPUT",ROOT/"_docs/experiments/pallet_paper_tables_245_20261011"))
BACKBONES=("YOLO","DOPE","ResNet-18")


def read(name):return json.loads((DOC/name).read_text())
def fmt(value, digits=4):return "—" if value is None else f"{value:.{digits}f}"
def g(metrics,scope,backbone,method):return metrics["scopes"][scope]["groups"][backbone+"::"+method]
def stat(group,key):return group["seed_mean"]["metrics"][key]
def rate(group,key):return stat(group,key)["mean"]*100


def paired_summary(paired):
    out=["| 표 | 범위 | 기반 | 비교 | Δ성공 %p [95% CI] | t p / Holm p | bootstrap p / Holm p |",
         "| --- | --- | --- | --- | --- | --- | --- |"]
    for table,entries in paired["tables"].items():
        for entry in entries:
            if entry["metric"]!="success_rate":continue
            interval=entry.get("CI95")
            ci="—" if interval is None else "["+", ".join(fmt(v*100) for v in interval)+"]"
            out.append(f"| {table} | {entry.get('scope','')} | {entry.get('backbone','YOLO')} | {entry.get('contrast','')} | {fmt(entry.get('delta')*100 if entry.get('delta') is not None else None)} {ci} | {fmt(entry.get('p_t'),6)} / {fmt(entry.get('p_t_holm'),6)} | {fmt(entry.get('p_bootstrap'),6)} / {fmt(entry.get('p_bootstrap_holm'),6)} |")
    return "\n".join(out)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--private-dir",type=Path,required=True);args=parser.parse_args()
    assert not (DOC/"REPORT_KO.md").exists(), "Preserve completed report"
    required=("METHOD_LOCK.json","INPUT_AUDIT.json","TABLE_DISPLAY_PARITY.json","STAGE0_F_PARITY.json","POSEFIX_319_PARITY.json","EVALUATION_EXECUTION.json","METRICS.json","PAIRED.json","FAILURES.json","LATENCY.json","PAPER_NUMBERS.json","FIGURE_INDEX.json","CASE_SELECTION.json","PRIVATE_FIGURES_RECEIPT.json")
    assert all((DOC/name).is_file() for name in required), "Report only completed artifacts"
    metrics=read("METRICS.json");paired=read("PAIRED.json");numbers=read("PAPER_NUMBERS.json");cases=read("CASE_SELECTION.json");latency=read("LATENCY.json")
    summary=[]
    for b in BACKBONES:
        base=g(metrics,"RECT245",b,"BASE");seq=g(metrics,"RECT245",b,"N3_THEN_SUBPIX")
        summary.append(f"{b} {rate(base,'success_rate'):.2f}%→{rate(seq,'success_rate'):.2f}%")
    text=["[확인] 직사각형 245장과 정사각형 118장의 논문용 표·그림 재집계를 완료했다; 직사각형 Base→Two-stage 성공률은 "+", ".join(summary)+"다.","",
        "# 논문용 표·그림 재집계: 직사각형 245장 / 정사각형 118장","",
        "[확인] 직사각형 주범위는 REAL_DEV319 중 주석 등급 clean153+moderate92=245장, 13세션이다. severe74장을 제외하는 범위는 **개발 결과를 검토한 뒤 정한 범위**이므로 독립 시험자료가 아니다. grade는 주석 범주이며 물리적 가림률을 나타내지 않는다. 전체319장도 같은 정의로 보충 표에 남겼다. [ID 목록·SHA](RECT_245_IDS.json), [고정 계약](METHOD_LOCK.json), [입력 감사](INPUT_AUDIT.json).","",
        "[확인] 정사각형은 GREEN0918의 118장·1세션이며 직접 수동 코너≥4의 SQPnP→RefineLM 참조를 사용한다. 직사각형은 참조 생성 절차가 다르므로 두 모집단을 합산하거나 직접 우열을 비교하지 않는다. 정사각형 수치는 **FEASIBILITY_ONLY, 단일 세션의 기술 결과**다. 90° W/D 교환은 proper C4 대칭이어서 정사각형 뒤바뀜 열은 NA다. [기존 정사각형 참조 보고서](../pallet_square6d_manualpnp_20261010/REPORT_KO.md).","",
        "## 반올림 충돌 해결과 실제 재현","",
        "[확인] 이전 중단 사유는 문서의 반올림 표 값과 절대차≤1e−6 검산을 동시에 적용한 데 있었다. 사용자 ‘해결할수있게 해줘’에 따라 T/R은 소수3자리, 비율·분산은 소수1자리로 표시한 값과 대조했다. 원래24개 원시 비교 실패와 실제 전체 정밀도 값을 지우지 않았으며, 기존 F의 **≤0.01cm·≤0.01° 기준은 유지**했다. [TABLE_DISPLAY_PARITY.json](TABLE_DISPLAY_PARITY.json), [METHOD_LOCK.json:user_clarification](METHOD_LOCK.json).","",
        "[확인] 모든 고정 seed1·2·3을 사용했다. 새 학습·새 모델 선택·새 합성 생성·주석 수정은 하지 않았다. 새 좌표 생성기에는 예측·RGB·K·등록 치수의 추론용 whitelist만 전달했고 참조나 GT를 입력하지 않았다. 단계별 좌표/F 출력을 SHA로 봉인한 뒤 그 단계의 오차를 채점했다. 기존 좌표의 Stage0/PoseFix parity에서는 별도로 참조를 읽지만 이 값을 새 좌표 생성기나 F에 전달하지 않는다. cornerSubPix는 (5,5), zeroZone(−1,−1), EPS|COUNT40/0.001이며 원래 Base에서 원영상 대각선1% 상한을 마지막에 한 번 적용한다. 표의 팔레트 대각선과 이 **원영상 대각선**은 서로 다른 값이다. [평가 코드](../../../scripts/research/pallet_paper_tables_245_20261011/evaluate.py).","",
        "## 표 1: 기반별 주결과","",
        "[확인] 영상별 seed 평균을 만든 뒤 영상 통계를 계산했다. 성공은 seed별 strict T<5cm·R<5°를 먼저 판정하고 평균했다. T/R/ADD/IoU는 산출된 값만 사용하고 n을 적었으며, 미산출은 성공률과 PCK의 전체 분모에서 실패로 계산했다. PCK@10px는 기존 대칭 대응과 전체 참조 코너 분모를 유지한다. SD는 영상 간 표본산포(ddof=1)이며 seed를 독립 영상으로 취급하지 않는다.",""]
    for name in ("TABLE_1.md","TABLE_2.md","TABLE_3.md"):
        assert (DOC/name).exists(),name
        text.append((DOC/name).read_text());text.append("")
        if name=="TABLE_1.md":
            text += ["[확인] 평균이 낮아져도 분산·꼬리 오차가 낮아졌다는 뜻은 아니다. 다음 표는 모든 기반의 Base→Two-stage 분산과 P90을 그대로 나란히 놓는다.","",
                "| 범위 | 기반 | T 분산 cm² Base→Two-stage | R 분산 °² Base→Two-stage | T P90 cm Base→Two-stage | R P90 ° Base→Two-stage |","| --- | --- | --- | --- | --- | --- |"]
            for scope in ("RECT245","SQUARE118"):
                for backbone in BACKBONES:
                    a=g(metrics,scope,backbone,"BASE");b=g(metrics,scope,backbone,"N3_THEN_SUBPIX")
                    def arrow(metric,key):return fmt(stat(a,metric)[key])+"→"+fmt(stat(b,metric)[key])
                    text.append(f"| {scope} | {backbone} | {arrow('T_cm','variance')} | {arrow('R_deg','variance')} | {arrow('T_cm','P90')} | {arrow('R_deg','P90')} |")
            ybase=g(metrics,"RECT245","YOLO","BASE");yn3=g(metrics,"RECT245","YOLO","N3_DIM_SYM");yseq=g(metrics,"RECT245","YOLO","N3_THEN_SUBPIX")
            text += ["",f"[확인] YOLO245의 Two-stage 회전 평균은 {fmt(stat(yseq,'R_deg')['mean'])}°이며 Stage1 단독 {fmt(stat(yn3,'R_deg')['mean'])}°보다 높다. T 분산은 Base {fmt(stat(ybase,'T_cm')['variance'])}cm²→Two-stage {fmt(stat(yseq,'T_cm')['variance'])}cm²이다. 평균 위치 오차·성공률의 변화와 이 산포 악화를 함께 보고한다.",""]
            dbase=g(metrics,"RECT245","DOPE","BASE");dseq=g(metrics,"RECT245","DOPE","N3_THEN_SUBPIX")
            text += [f"[확인] DOPE245의 Two-stage R 평균은 Base {fmt(stat(dbase,'R_deg')['mean'])}°→{fmt(stat(dseq,'R_deg')['mean'])}°, W/D 뒤바뀜은 {fmt(rate(dbase,'confusion_rate'))}%→{fmt(rate(dseq,'confusion_rate'))}%로 악화됐다. 자세 산출 n={dbase['seed_mean']['available_frames_any_seed']}/{dbase['frames']}를 연속 오차 분모에 명시하고, 성공률·PCK는245장 전체 분모를 유지한다.",""]
            text += ["![그림 A](figures/FIGURE_A_errors.png)","","[확인] 그림 A는 두 참조 모집단을 별도 패널로 놓고 로그 y축을 사용한다. 모든 유한 이상치를 점으로 남겼으며 미산출 n을 표시했다. 0 오차는 epsilon으로 바꾸지 않고 발생 ID·수를 [FIGURE_INDEX.json](FIGURE_INDEX.json)에 별도로 기록한다.","","## 표 2: 외부 비교와 처리 시간",""]
        elif name=="TABLE_2.md":
            text += ["[확인] PoseFix-style은 기존 첫 RAW chain/pass1의 `points[:,0,1]`를 그대로 사용하며 checkpoint·pass를 고르지 않았다. 다른 시점의 원고 latency25.24ms를 이번 측정값과 섞지 않았다. Base/Stage1/Stage2/Two-stage/PoseFix/PoseFix+Stage2를 같은 GPU·환경에서 batch1, 준비20회/경로, 13세션26프레임×5회로 측정했다. 전처리부터 최종 PnP까지 포함하고 디코딩·모델 로딩은 제외했다. [LATENCY.json](LATENCY.json).",""]
            pf=g(metrics,"RECT245","YOLO","POSEFIX_SUBPIX");ours=g(metrics,"RECT245","YOLO","N3_THEN_SUBPIX")
            text += [f"[확인] PoseFix-style+cornerSubPix의 T/R/성공률은 {fmt(stat(pf,'T_cm')['mean'])}cm /{fmt(stat(pf,'R_deg')['mean'])}° /{fmt(rate(pf,'success_rate'))}%로, Two-stage의 {fmt(stat(ours,'T_cm')['mean'])}cm /{fmt(stat(ours,'R_deg')['mean'])}° /{fmt(rate(ours,'success_rate'))}%보다 정확도가 좋다. 처리 시간은 {fmt(latency['arms']['POSEFIX_SUBPIX']['mean_ms'])}ms 대 {fmt(latency['arms']['N3_THEN_SUBPIX']['mean_ms'])}ms로 Two-stage가 짧다. 이 결과는 모든 방법·지표에서 제안법이 우세하다는 주장을 뒷받침하지 않는다.","","## 표 3: 절제와 공동 보정 비교",""]
    text += ["[확인] N0/N1/N2 영상별 seed 예측의 존재 여부를 입력 감사에서 확인했고, 존재하는 고정 예측을 같은 F로 평가했다. 표에서 결과가 나쁜 경로·seed도 남겼다. [P/D/L·등급별 보충 표](SUPP.md), [모든 seed](SEED_TABLES.csv), [319장 표1](SUPPLEMENT_319_TABLE_1.md), [319장 표2](SUPPLEMENT_319_TABLE_2.md)에 같은 정의로 남겼다. severe74장도319장 보충에서 제외하지 않았다.","",
        "## 같은 실행에서 측정한 여섯 경로의 처리 시간","","| 경로 | 평균±SD ms | 측정 n |","| --- | --- | --- |"]
    latency_names={"BASE":"Base","N3_DIM_SYM":"Stage 1 only","SUBPIX":"Stage 2 only","N3_THEN_SUBPIX":"Two-stage","POSEFIX":"PoseFix-style","POSEFIX_SUBPIX":"PoseFix-style + Stage 2"}
    for arm in latency_names:
        v=latency["arms"][arm];text.append(f"| {latency_names[arm]} | {fmt(v['mean_ms'])} ± {fmt(v['sd_ms'])} | {v['n']} |")
    text += ["","[확인] 각 경로 준비20회는 위 n에서 제외했다. GPU 동기화 벽시계 시간을 사용했고6경로 모두 동일 고정26프레임을5회 측정했다. 캐시의 평가실행 시간이나 이전 보고서의 다른 시점 latency를 여기에 대입하지 않았다.",""]
    if (DOC/"LATENCY_PREFLIGHT_REPAIR.json").exists():
        text += ["[확인] 최초 latency 로딩은 집계용 `pallet-pose` 환경의 오래된 Ultralytics가 기존 YOLO checkpoint의 C3k2를 지원하지 않아 실패했다. 실패 시 실제 forward/F/측정 호출은0이었다. 기존 원 runtime 환경인 `pallet-yolo26`(Torch2.1.1+cu118, Ultralytics8.4.60, OpenCV4.9.0)으로 같은 checkpoint·프로토콜을 실행했다. 모델/설정을 바꾸거나 수치를 보고 환경을 고른 것은 아니다. 이 환경 선택 오류와0호출을 [LATENCY_PREFLIGHT_REPAIR.json](LATENCY_PREFLIGHT_REPAIR.json)에 남겼다.",""]
    if (DOC/"LATENCY_INSTRUMENTATION_REPAIR.json").exists():
        text += ["[확인] 이어진 첫 Base 준비 호출에서 Ultralytics의 내부 최초 초기화 forward를 계측 guard가 두 번째 영상 호출로 잘못 셌다. 이 시도는 F1회·detector hook2회 이후 중단됐고 measured 호출은0이었다. 실패한 source lock·journal·로그를 보존하고 명시적900개 파이프라인과 내부 초기화 forward1개를 구분하도록 계측만 고쳤다. 최종 측정과 별도로 실패 준비 호출1회를 남겼으며 예측·모델·수치 기준은 바꾸지 않았다. [LATENCY_INSTRUMENTATION_REPAIR.json](LATENCY_INSTRUMENTATION_REPAIR.json).",""]
    text += [
        "## 짝 비교, p값과 해석","",
        "[확인] 직사각형은 기존319장/13세션 paired cluster draw10,000회(seed20260917, SHA `63e288a51d7b0612616beefac28fcc625e76e8c5d7b5a0ecc5e8b85c73048fa5`)에서 고정245장 ID만 취했다. 새 draw나 좋은 seed를 고르지 않았다. t-검정은 동일13세션의 평균 차이를 같은 가중치로 사용(df12)하고, CI/bootstrap 양측 p는 영상 가중 효과를 대상으로 하므로 추정 대상이 다르다. 정사각형은 기존118장 frame draw와 영상 단위 paired t-검정을 기술용으로 사용한다.","",
        "[확인] 표마다 사전 정의한 비교·지표 전체를 한 family로 묶어 t와 bootstrap p에 Holm 보정을 각각 적용했다. 표1은 직사각형과 정사각형의 선언 검정을 함께 보정하며 정사각형 W/D NA는 제외한다. 표3의 중복/역방향 비교도 삭제하지 않았다. 개발 후 선정·다중 비교·단일 정사각형 세션 때문에 p값을 독립 확증으로 해석하지 않는다. [PAIRED.json](PAIRED.json).","",paired_summary(paired),"",
        "[확인] 원래 p와 Holm p, 모든 지표의 CI와 유효 짝 n은 [PAIRED_TABLE_1.md](PAIRED_TABLE_1.md), [PAIRED_TABLE_2.md](PAIRED_TABLE_2.md), [PAIRED_TABLE_3.md](PAIRED_TABLE_3.md)에 나란히 기재했다. 표본분산·SD·중앙값·P90·최댓값·95%CI와 raw seed별 값은 [METRICS.json](METRICS.json), 실패·손상·회복 ID 전체는 [FAILURES.json](FAILURES.json)에서 확인할 수 있다.","",
        "[확인] DOPE의 직사각형 연속 오차는 eval_night09에서 공통 산출 자세가0개이므로 해당 T/R/IoU 세션 t-검정3개를 NA로 남겼다. df12 조건을 유지하며13번째 세션을 임의로 빼거나 df11 p값으로 대신하지 않았다. Holm은 실제 계산 가능한 원래 family의 p에만 적용했다.","",
        "## 그림 B: 고도와 가로·깊이 뒤바뀜","","![그림 B](figures/FIGURE_B_wd_elevation.png)","",
        "[확인] YOLO Two-stage의 직사각형245장, 고도<5/5–10/10–20/20–30/≥30°에서 seed별 R>45° 및 |yaw|≥60°를 먼저 판정했다. 구간의 장수를 막대 아래 표시했다. GT-parity oracle은 참조 긴 축에 맞는 가설을 공급하는 **진단용 경로**이며 오차가 가장 작은 가설을 선택한 것이 아니다. 결과가 악화된 구간도 표시했다. [구간별 수치·출처](PAPER_NUMBERS.json)의 `altitude_confusion_bins`.","",
        "## 그림 C·D: 고정 사례와 공개 범위","","![그림 C 좌표 버전](figures/FIGURE_C_coordinates.png)","",
        "[확인] 그림 C는 고정seed1에서 각 기반의 ΔT<0 개선 사례 중 개선량 중앙값에 가장 가까운 영상과 ΔT>0 최대 악화 영상이다. 동률은 ID 사전순이다. 직사각형 사례 풀은245장 중 final-test4세션27장을 제외한218장이고, 정사각형 YOLO는118장이다. 자세가 둘 다 산출된 짝에서만 사례를 고르며 미산출 ID도 기록했다. **그림 규칙을 결과에 맞춰 바꾸지 않았다.** [CASE_SELECTION.json](CASE_SELECTION.json).","",
        "| 모집단 | 기반 | 선택 규칙 | ID | ΔT cm |","| --- | --- | --- | --- | --- |"]
    for case in cases["C"]:text.append(f"| {case['population']} | {case['backbone']} | {case['kind']} | {case['id']} | {fmt(case['delta_T_cm'])} |")
    text += ["","![그림 D 좌표 버전](figures/FIGURE_D_coordinates.png)","",
        "[확인] 그림 D는 plastic/wood의 적격218장 풀과 square118장 풀에서 형태별 팔레트 대각선 중앙값에 가장 가까운 ID다. 대각선은 native pixel에서 참조 코너0–7의 축 정렬 bounding box의 `hypot(ptpX,ptpY)`이며 화면 경계로 clipping하지 않는다. 기존 투영·외삽 코너를 이 영상 크기 진단에 포함했다. Square의 보조 투영 코너를 수동 PnP 참조 생성에 새로 넣지는 않았다. 원고 인용용 대각선 중앙값은 이 사례 풀과 별도로 **전체245장**에서 계산했다.","",
        "[확인] RGB 버전은 저장소 밖 지정 `--private-dir`에만8개 C와3개 D를 저장했다. 저장소에는 같은 ID의 코너·투영 상자 좌표 버전만 게시했다. 로컬 파일명·SHA는 [PRIVATE_FIGURES_RECEIPT.json](PRIVATE_FIGURES_RECEIPT.json)에 있고 실제 로컬 경로는 비공개 `FIGURE_INPUTS.json`에 기록된다. 공개 PNG에 RGB·배경 texture를 포함하지 않았다.","",
        "## 원고 인용 수치와 한계","","| 항목 | 값 | 출처 |","| --- | --- | --- |"]
    for backbone,v in numbers["N3_PCK10_increase_percentage_points"].items():text.append(f"| {backbone} Stage1 PCK@10px 증가 | {fmt(v['value'])}%p | PAPER_NUMBERS.json:N3_PCK10_increase_percentage_points |")
    for method,v in numbers["YOLO_success_RECT245_percent"].items():text.append(f"| YOLO {method} 5cm·5° | {fmt(v['value'])}% | TABLE_2.csv |")
    d=numbers["RECT245_pallet_diagonal_px"];text.append(f"| 245장 팔레트 대각선 중앙값 /10px 비율 | {fmt(d['median_px'])}px /{fmt(d['ten_px_percent_of_median'])}% | PAPER_NUMBERS.json:RECT245_pallet_diagonal_px |")
    for backbone,v in numbers["square_success_Base_to_Two_stage_percent"].items():text.append(f"| Square {backbone} Base→Two-stage | {fmt(v['BASE'])}%→{fmt(v['N3_THEN_SUBPIX'])}% | TABLE_1.csv |")
    for name in ("YOLO::BASE","YOLO::N3_THEN_SUBPIX"):
        v=numbers["confusion_share_among_R_gt45_percent"][name];text.append(f"| {name} R>45° 중 W/D 뒤바뀜 비율 | {fmt(v['value'])}% | PAPER_NUMBERS.json:confusion_share_among_R_gt45_percent |")
    text += ["","[확인] 모든 인용 값에 원본 표의 칸 또는 METRICS 키를 연결한 [PAPER_NUMBERS.json](PAPER_NUMBERS.json)을 만들었다. 유한 큰 오차와 악화·미산출을 제거하지 않았고 모든 seed를 보충 파일에 남겼다. 원고·LaTeX·PDF와 기존 실험 폴더는 수정하지 않았다.","",
        "[추정] seed 간 공유 학습 자료, 13세션의 제한된 환경, 개발 결과를 본 뒤 severe를 제외한 범위 선택이 외부 일반화에 영향을 줄 수 있다. 이 분석만으로 물리적 가림 강도에 따른 인과 효과나 독립 시험 성능을 주장할 수 없다. 정사각형 수동 PnP 참조 자체의 불확실성은 이전 보고서의 잔차·LOO·4점/≥5점 분석과 함께 해석한다.","",
        "## 검산·보존·재현","",
        "[확인] 기존 입력·checkpoint·결과 파일은 읽기 전용으로 사용했고 작업 전후 SHA를 대조했다. 실제 재현·독립 검산·원 작업트리 보존·원격 게시 증거는 아래 실행 receipt에 기록한다. [재현 방법](REPRODUCE.md)."]
    for name in ("STATISTICS_VERIFICATION.json","VERIFICATION.json"):
        if not (DOC/name).exists():continue
        receipt=read(name)
        assert receipt["status"]=="PASS", "Do not report successful verification after a failed gate"
        detail=""
        if receipt.get("numerical_comparisons") is not None:detail+=f", 수치 비교 {receipt['numerical_comparisons']:,}개"
        if receipt.get("max_absolute_difference") is not None:detail+=f", 최대 절대차 {receipt['max_absolute_difference']:.3g}"
        text.append(f"[확인] [{name}]({name})의 독립 검산은 {receipt['status']}{detail}다.")
    text.append("")
    if (DOC/"NUMERICAL_AMENDMENT.json").exists():
        text += ["[확인] 독립 검산에서 bootstrap p의 이산 동일값을 부동소수점 차이로 서로 다르게 센 경계 비교를 발견했다. 초기 PAIRED와 source를 보존하고 동률 처리의 수치 구현을 고쳤다. 원본 좌표·오차·평균·분산·CI·비교 정의는 바꾸지 않았고 p/Holm과 표시 표만 saved 결과에서 다시 계산했다. [NUMERICAL_AMENDMENT.json](NUMERICAL_AMENDMENT.json).",""]
    for name in ("STAGE0_F_PARITY.json","POSEFIX_319_PARITY.json","COORDINATE_EXECUTION.json","EVALUATION_EXECUTION.json","STATISTICS_EXECUTION.json","TABLES_EXECUTION.json","REPLAY_VERIFICATION.json","VERIFICATION.json","STATISTICS_VERIFICATION.json","WORKSPACE_PRESERVATION.json","PUBLIC_CONTENT_CHECK.json","PUBLICATION.json"):
        if (DOC/name).exists():text.append(f"- [{name}]({name})")
    (DOC/"REPORT_KO.md").write_text("\n".join(text)+"\n")
    (DOC/"REPRODUCE.md").write_text("""# 재현과 공개 수치 검산

[확인] 기존 Python/GPU 환경과 비공개 기존 입력을 사용한다. 새 학습·합성 생성·annotation·checkpoint 변경은 없으며 원고·기존 실험 결과를 덮어쓰지 않는다. 입력 SHA와 METHOD_LOCK, 245장 ID를 먼저 확인한다. 출력이 있으면 각 실행은 중단하며 새 출력 경로를 만들어 재현한다.

```bash
export PALLET_SOURCE_ROOT="/path/to/original/input-checkout"
export PALLET_BASELINE_ROOT="/path/to/existing/baseline-checkout"
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PALLET_PYTHON="/path/to/existing/pallet-pose/bin/python"
export PALLET_RUNTIME_PYTHON="/path/to/existing/pallet-yolo26/bin/python"
export PALLET_PRIVATE_PAPER="/path/outside/repository/private-paper-output"
export PALLET_PAPER_OUTPUT="/path/to/nonexistent/fresh-paper-output"
mkdir -p "$PALLET_PAPER_OUTPUT"
cp _docs/experiments/pallet_paper_tables_245_20261011/METHOD_LOCK.json "$PALLET_PAPER_OUTPUT/METHOD_LOCK.json"
```

[확인] 실행 순서는 입력/schema·표시 정밀도 parity → 기존 F seed1/4경로 및 PoseFix319 parity → 추론용 whitelist만 받는 전체319 새 좌표 생성·SHA봉인 → 기존 F 출력봉인·metric → 같은환경6경로 처리시간 → 표/짝 비교/독립 수치검산/그림/보고서다. 기존좌표 parity 채점의 참조는 새 생성기/F에 전달하지 않는다. 아래 코드의 진입점을 사용한다. 공개 수치 검산은 model·PnP를 새로 실행하지 않고 saved R/t/코너/개별 오차와 bootstrap draw를 대조한다.

- [입력 확인](../../../scripts/research/pallet_paper_tables_245_20261011/preflight.py)
- [좌표·F 평가](../../../scripts/research/pallet_paper_tables_245_20261011/evaluate.py)
- [통계·p/Holm](../../../scripts/research/pallet_paper_tables_245_20261011/stats.py)
- [표 생성](../../../scripts/research/pallet_paper_tables_245_20261011/tables.py)
- [그림·private RGB](../../../scripts/research/pallet_paper_tables_245_20261011/figures.py)
- [한국어 보고서](../../../scripts/research/pallet_paper_tables_245_20261011/report.py)

```bash
"$PALLET_PYTHON" -B -m scripts.research.pallet_paper_tables_245_20261011.preflight
"$PALLET_PYTHON" -B -m scripts.research.pallet_paper_tables_245_20261011.evaluate
"$PALLET_RUNTIME_PYTHON" -B -m scripts.research.pallet_paper_tables_245_20261011.latency
"$PALLET_PYTHON" -B -m scripts.research.pallet_paper_tables_245_20261011.stats
"$PALLET_PYTHON" -B -m scripts.research.pallet_paper_tables_245_20261011.correct_pvalues
"$PALLET_PYTHON" -B -m scripts.research.pallet_paper_tables_245_20261011.tables
"$PALLET_PYTHON" -B -m scripts.research.pallet_paper_tables_245_20261011.verify_stats
"$PALLET_PYTHON" -B -m scripts.research.pallet_paper_tables_245_20261011.figures --private-dir "$PALLET_PRIVATE_PAPER"
"$PALLET_PYTHON" -B -m scripts.research.pallet_paper_tables_245_20261011.report --private-dir "$PALLET_PRIVATE_PAPER"
```

[확인] 집계/평가는 기존 pallet-pose 환경을, 실제 latency는 기존 checkpoint와 호환되는 원래 pallet-yolo26 환경을 사용한다. 새 환경·package·checkpoint를 설치하거나 변경하지 않는다. 호환되지 않는 첫 로딩의0forward/F는 LATENCY_PREFLIGHT_REPAIR에 보존했다.

[확인] 수치 집계만 재현하려면 별도 fresh output에 고정 METHOD_LOCK과 공개 RECT_ROWS/SQUARE_ROWS를 복사한 뒤 stats부터 실행한다. 표2의 처리 시간 열을 동일하게 재현하려면 공개 LATENCY.json도 복사한다. 실행 시간을 새로 측정하는 경우 LATENCY를 복사하지 않고 위6경로 benchmark를 실행한다. 그림은 기존 audit RGB의 SHA가 일치할 때 private FIGURE_INPUTS.json을 자동 생성한다. 이 파일과 RGB그림8+3장은 저장소 밖에만 저장한다. 세션 master draw는 기존13세션 hash를 그대로 쓰며245장은 같은 draw의 고정 subset이다. Square는 기존118 frame draw다. 세부 실행횟수와 검산값은 실행 receipt와 METRICS/PAIRED에 남긴다.
""")
    (DOC/"README.md").write_text("[확인] 직사각형245장·정사각형118장 논문용 표·그림 재집계. 개발 후 범위 선택·단일Square세션의 탐색/기술 결과.\n\n[한국어 보고서](REPORT_KO.md) · [방법 봉인](METHOD_LOCK.json) · [원고 인용 수치](PAPER_NUMBERS.json) · [표1](TABLE_1.md) · [표2](TABLE_2.md) · [표3](TABLE_3.md) · [그림](FIGURE_INDEX.json) · [재현](REPRODUCE.md)\n")
    print("PAPER_REPORT_COMPLETE",flush=True)


if __name__=="__main__":main()
