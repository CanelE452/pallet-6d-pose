"""Write Korean results from sealed measurements; never select/retrain a model."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .common import read, rows, sha, write, csv_write
from .registry import BASIC_GROUPS, VISUAL_GROUPS, SEEDS
from .figures import NAMES, group_name, label, groups_and_representatives


def number(value, digits=3):
    return "NA" if value is None else f"{value:.{digits}f}"


def table(headers, values):
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(x) for x in row) + " |" for row in values]
    return "\n".join(lines)


def metric_table(result, population, selected=None):
    data = result['populations'][population]
    return table(["군 / seed", "성공/분모", "위치 평균 ± SD cm", "위치 분산 cm²", "위치 P90/P95/최대 cm",
        "회전 평균 ± SD °", "회전 분산 °²", "회전 P90/P95/최대 °", "T>50/100cm · R>45°", "no-pose"],
        ([a, f"{m['successes']}/{m['denominator']}",
          f"{number(m['translation_cm']['mean'])} ± {number(m['translation_cm']['sample_std'])}",
          number(m['translation_cm']['variance_ddof1']),
          '/'.join(number(m['translation_cm'][k]) for k in ('P90','P95','max')),
          f"{number(m['rotation_deg']['mean'])} ± {number(m['rotation_deg']['sample_std'])}",
          number(m['rotation_deg']['variance_ddof1']),
          '/'.join(number(m['rotation_deg'][k]) for k in ('P90','P95','max')),
          f"{m['T_gt50cm']}/{m['T_gt100cm']} · {m['R_gt45deg']}", m['no_pose']]
         for a, m in data.items() if selected is None or a in selected))


def contrast_rows(result, population, pairs):
    values=[]
    for before, after in pairs:
        p=result['paired'][population].get(before+'__TO__'+after)
        if p is None:
            continue
        d=p['both_failure_continuous']
        values.append([before+' → '+after, p['counts']['RECOVERY'],p['counts']['DAMAGE'],
            p['counts']['BOTH_SUCCESS'],p['counts']['BOTH_FAILURE'],
            number(p['deltas']['translation_cm']['mean']),number(p['deltas']['rotation_deg']['mean']),
            f"{d['denominator']}: {number(d['translation_cm']['mean'])} / {number(d['rotation_deg']['mean'])}"])
    return values


def write_reports(run_dir, public):
    run_dir, public=Path(run_dir),Path(public)
    evaluation=run_dir/'evaluation_v1'
    result=read(evaluation/'EVALUATION.json')
    manifest=read(run_dir/'PREDICTION_MANIFEST.json')
    seal=read(evaluation/'EVALUATION_SEAL.json')
    if seal['prediction_manifest_sha256']!=sha(run_dir/'PREDICTION_MANIFEST.json'):
        raise ValueError('Evaluation/prediction linkage changed')
    groups,reps=groups_and_representatives(result)
    new=[a for a in reps if group_name(a) in BASIC_GROUPS+VISUAL_GROUPS]
    fits=[read(p) for p in sorted((run_dir/'training').glob('*/RUN_RECEIPT.json'))]
    attempts=list(rows(run_dir/'GLOBAL_FIT_ATTEMPTS.jsonl'))
    p=result['populations']['PRIMARY192']
    lines=['# 고정 후보에서 위험 감독·양쪽 family 정보·국소 RGB의 효과', '',
        '운영 기본값과 논문은 변경하지 않았다. 이 결과는 반복 개발에 사용한 실사 자료에 대한 연구 비교이며, 새 독립 실사 test는 **0장**이다. 기존 H1의120/192와 full bank 전부실패41장이라는 과거 정보는 설계 배경으로 이미 알려져 있었다. 이번 출력 봉인은 새 실사 결과를 본 뒤 튜닝하지 않았다는 증거이며 미관측 test를 만든 것이 아니다.', '',
        '## 실제 실행과 비교 계약', '',
        f"새 optimizer fit 시도 **{len(attempts)}회**, 완료 **{len(fits)}회**. 동일한 원 D1 full bank만 선택했고 새 PnP·수치 자세 최적화·후보 생성·본체 forward/재학습은 0회다. F1 고정 자세 비용 재평가와 작은 선택기 forward는 별도 수행했다.", '',
        table(['항목','실제 상태'],[
            ['원 D1/F0/기준','실사238·합성1003 재사용; F0 188와 원 normalizer·alias·물리 pose 순서 보존'],
            ['R_CONT_F0 / R_ORDER_F0','각3seed; 회귀 minimax / 성공 우선·실패 Pareto 순위'],
            ['R_ORDER_F1_ZERO / R_ORDER_F1','각3seed; 동일476입력 구조·초기값·공통 epoch의 순서·쌍; 추가288블록만 ZERO'],
            ['F1','두 family 각각72scalar+72validity; 총144+144 추가; train768 valid-only 정규화'],
            ['B','합성1003 별도 GT 진단; 후보/모델 특징/훈련에 GT 주입0'],
            ['V','각3seed 실행; RGB3+validity1의20지역. NULL은 RGB3만0' if manifest['visual_executed'] else '미실행: V_GATE.json의 실제 조건/이유 참조'],
            ['모델 고정','각자 합성 validation loss 최저 checkpoint; 대표seed20261018 고정'],
            ['실사 평가','모든238행×등록군 출력 봉인 후 참조 decode; PRIMARY192=49+143, CAL46 별도'],
            ['자료 보존','기존 후보·모델·모든 실패 기록·사용자 작업·운영 설정·논문 보존']]), '',
        '기술 오류도 보존했다. R_CONT 첫 fit 완료 뒤 다음 seed 진입 전 interop thread 재설정으로 CLI가 abort했고, idempotent 설정만 고쳐 완료 checkpoint를 재학습하지 않고 나머지 seed를 실행했다. F1 smoke의 잘못된 interpreter/cv2 import 실패와 SciPy 임시파일 guard 충돌·segfault, B/독립 감독 export의 NumPy scalar JSON 오류도 원시 로그와 source version을 private에 남겼다. B의 엄격 roundtrip FAIL은 표현 차이 진단과 별도로 유지했다. CLI/포맷 오류를 나쁜 성능이나 실패 optimizer fit과 혼동하지 않는다.', '',
        '성공은 **한 개의 실제 반환 pose**의 위치≤5cm와 회전≤5°를 동시에 만족하는 경우다. 실사 대칭은 기존 I/Y(π), 합성은 원행 C1/C2를 유지했다. 정사각형 90° 대칭이나 GT 회전 직교화를 새로 넣지 않았다. 상태/CAL을 사용한 시스템과 합성 학습 PoseFix는 사용 정보가 같지 않으며, 같은 구조의 대조에서만 추가 정보 효과를 해석한다.', '',
        '## PRIMARY192 전체 결과', '',
        '각 군의 대표는 사전에 고정한 첫 seed다. 아래 전체표와 CSV에는 나머지 seed·실패·극단값을 모두 남겼다. 평균은 꼬리 영향을 포함하고, 표본 SD·분산은 ddof=1이다.', '',
        metric_table(result,'PRIMARY192'), '',
        '![성공과 큰 오류](figures/success_and_large_errors.png)', '',
        '![전체 분포](figures/whole_error_distributions.png)', '',
        '## 회복·손상과 목표/정보의 대조', '',
        '회복/손상은 동일 영상의 공동성공 여부 전이다. 연속 Δ는 이후−이전으로 양수면 오차 악화다. 서로 다른 목적함수의 loss 숫자를 직접 비교하지 않았다.', '',
        table(['비교','회복','손상','공동성공','공동실패','전체 ΔT cm','전체 ΔR °','공동실패 n: ΔT/ΔR'],
            contrast_rows(result,'PRIMARY192',[(b,a) for a in new for b in ('N0','E0','D1_H1_MLP_s20261018','D1_H1_PAIR_ROLE_AFFINE_s20261018')])), '',
        '같은 seed의 F1_ZERO→F1은 추가 정보의 효과, V_NULL→V_RGB는 실제 RGB 추가 효과를 비교한다. R_ORDER_F0→F1_ZERO는 용량 대조이며 정보 추가 효과로 세지 않는다.', '',
        table(['비교','회복','손상','공동성공','공동실패','전체 ΔT cm','전체 ΔR °','공동실패 n: ΔT/ΔR'],
            contrast_rows(result,'PRIMARY192',[(a+'_s'+str(s),b+'_s'+str(s)) for s in SEEDS
                for a,b in [('R_ORDER_F0','R_CONT_F0'),('R_ORDER_F0','R_ORDER_F1_ZERO'),
                            ('R_ORDER_F1_ZERO','R_ORDER_F1')]+([('R_ORDER_F1','V_ORDER_NULL'),('V_ORDER_NULL','V_ORDER_RGB')] if manifest['visual_executed'] else [])])), '',
        '![회복·손상](figures/recovery_damage_and_common_failure.png)', '',
        '![동일 정보 대조](figures/matched_information_objective_controls.png)', '',
        '## 좋은 후보가 없는 실패의 연속 손상', '',
        '원 full bank에서 성공 가능한 영상은 **151/192**, 전부 실패는 **41장**이다. pair는133/192와59장으로 서로 다른 고정 집합이다. 이번에는 새 후보를 만들지 않아 full41장을 새로 성공시킬 수 없다. 이 집합은 출력 봉인 후 참조로 확인했고 추론·훈련 입력으로 쓰지 않았다.', '']
    with (evaluation/'ALL_FAILURE_RISK.csv').open(newline='') as f:
        af=list(csv.DictReader(f))
    lines += [table(['고정 집합','군','n','N0 대비 ΔT 합/평균 cm','T 개선/악화/동일','N0 대비 ΔR 합/평균 °','R 개선/악화/동일'],
        ([r['fixed_set'],r['arm'],r['denominator'],
          r['translation_cm/sum']+'/'+r['translation_cm/mean'],
          '/'.join(r['translation_cm/'+k] for k in ('improved_n','worsened_n','equal_n')),
          r['rotation_deg/sum']+'/'+r['rotation_deg/mean'],
          '/'.join(r['rotation_deg/'+k] for k in ('improved_n','worsened_n','equal_n'))]
          for r in af if r['population']=='PRIMARY192' and r['before']=='N0' and r['arm'] in reps)), '',
        '![고정41장 연속 변화](figures/fixed_failure41_continuous_scatter.png)', '',
        '## 실제 선택과 회복 출처', '',
        'N0/E0 출처는 각각 원 fallback과 bank의 e0_regenerated 물리 pose와 수치적으로 같은지 비교하며, 둘과 같으면 기존 우선순위대로 N0로 센다. 비교 기준 E0는 별도로 보존한 historical 출력이고 E0_REGENERATED와의 동일성도 검사했다. OTHER는 기존 bank의 다른 후보이며 새로운 B 생성 성과가 아니다. PoseFix 기준의 OTHER는 full bank 소속을 의미하지 않는다. 신규 선택군은 모두 원 full bank 행동에 정확히 포함됨을 별도 검사했다. SELECTION_DIAGNOSIS의 FULL 표시는 baseline까지 같은 진단 열에 놓은 것이며151상한은 원 bank를 고르는 신규 선택군에 적용한다.', '',
        '전체 선택 출처는 [ACTION_SOURCE_COUNTS.csv](ACTION_SOURCE_COUNTS.csv), 회복/손상별 출처는 [GAIN_DAMAGE_ACTUAL_SOURCE.csv](GAIN_DAMAGE_ACTUAL_SOURCE.csv), 정확한8칸 교차는 [THREE_WAY_SUCCESS.csv](THREE_WAY_SUCCESS.csv)에 있다.', '',
        '![출처](figures/actual_sources_and_recovery_sources.png)', '',
        '## 감독과 B 기하 계약에서 확인한 공백', '',
        'F1 추가 정보의 전달 자체도 점검했다. 저장한 train25,613/validation7,973/PRIMARY6,518/CAL2,181 행동에서144scalar의 validity와 값 변동을 집계했다. train에서 지지가0인4개 partial pose-projection count/fraction 채널은 실사에서도 모두 invalid였다. 네 집단에서 집단 내 값이 변하는 scalar는114/114/114/112개, 같은 frame의 행동 간에 한 번이라도 값이 달랐던 scalar는74/74/74/72개다. 나머지66/66/66/68개의 유효 채널에는 상태·관측 수 같은 frame 공유 정보가 있어 무용하다는 뜻은 아니다. 이 가용성 집계는 실사 정답 없이 수행했고 성능 기여의 인과 증거로 사용하지 않는다. [채널별 원수치](F1_INFORMATION_AVAILABILITY.csv)를 남겼다.', '',
        '순위 감독은 성공→실패와 실패 간 두 축 Pareto 우열만 사용한다. 두 성공, 실패 tradeoff, 수치 우열 불일치는 쌍을 만들지 않는다. train768 중465장, validation235 중152장이 순위 loss에 기여했다. train 전부실패416장 중215장은 Pareto 쌍이 있으나201장은 구별할 우열이 없다. validation은171장 중104/67이다. 회귀는 두 축이 유효한 전체768/235를 사용했다. 실제 수치 우열 불일치는0쌍이며 원 라벨을 수정하지 않았다.', '',
        '6cm와100cm 실패도 다른 축이 반대로 좋아지는 tradeoff라면 R_ORDER는 어느 쪽이 우월한지 감독하지 않는다. R_CONT는 log1p(T/5), log1p(R/5)를 학습하고 두 예측축의 최대값이 작은 후보를 고르므로 연속 크기를 구분할 수 있지만, 성공률 최적화와 동일하지 않다.', '',
        '합성1003 중 B관측은673장, B없음은330장이다. ORIGINAL_WD에서 허용 GT phase를 모두 시도해도 유한 지지를 통과하지 못한 영상은296/673장이다. 같은 fixed G13 package에서 허용 GT와 실패 E0가 모두 feasible인325장 중203장에서는 실패 E0의 비용이 최선 feasible GT phase보다도 낮았다. 이는 원 관측·비용과 GT의 충돌 진단이며 실제 E0의 계층적 G10→G13 선택을 재현했다는 뜻이 아니다. G10/G13의 목적함수 값을 서로 비교하지 않았다.', '',
        '원 저장 float32 회전과 backend parameterized 회전의 엄격1e−10 roundtrip assertion FAIL660/3010은 지우지 않았다. 독립 기하 검사에서는 원 GT와 parameterized 표현을 나누어 추적했고 finite 판정 차이는0이었다. 원 solver feasibility tolerance1e−7은 그대로다. 원 annotation에 material variant와 scene visibility metadata는 있으나, 관측별 실제 mesh-face/pixel owner는 **미검증**이다. cuboid 선이 재료 경계라는 보장은 없다. [B 진단 상세](B_REFERENCE_RESULTS.md)를 함께 봐야 한다.', '',
        '## 불확실성과 비용', '',
        '짝지은 bootstrap은 원6 recording-cluster를 단위로10000회(seed20260917), 원8 named-session은 민감도로 별도 계산했다. 원 저장 draw 행렬은 없어서 같은 고정 helper/seed의 일정을 재생성했다. seed3개를 독립576장으로 늘리지 않았다. 0을 포함하는 CI는 동등성 증명이 아니며, 포함하지 않아도 반복 개발·작은 군집 수·여러 비교의 한계가 남는다. 원수치와 두 CI는 [EVALUATION.json](EVALUATION.json)에 있다.', '',
        '실제 비용은 [COST_AND_EXECUTION.json](COST_AND_EXECUTION.json)에 parent wall, CPU, 각 fit, F1/patch I/O, 작은 head inference를 나누었다. 원 cache 생산비와 계측이 없는 시간은 unknown이며0으로 적지 않았다. RTX3080의 cached 실험 시간을 Jetson FPS나 배포 지연으로 바꾸지 않았다.', '',
        '![모든 fit 곡선](figures/all_fit_learning_and_pose_curves.png)', '',
        '## 독립 검사와 실제 영상', '',
        '검사 수준은 입력 일치, 실행 완료, 수치 재현, 감독 정의, 기하 표현, 모델의 실제 pixel 검토, 독립 물리 정답, 원격 게시를 분리했다. Python audit hook은 OS sandbox가 아니며 native I/O·사전 로드 객체를 완전히 증명하지 못한다. 새 predictor는 입력 decode 전 whitelist를 설치했다. P0 감사는 명시한 허용 입력/체크포인트를 로드한 뒤 hook을 설치했으므로 그 범위를 구별한다.', '',
        '[VERIFICATION.json](VERIFICATION.json), [PIXEL_REVIEW.json](PIXEL_REVIEW.json), [실행 검토표](EXECUTION_REVIEW_CHECKLIST.md), [공개 파일 검사](PUBLIC_PACK_AUDIT.json)를 확인할 수 있다. 모델 pixel 검토는 사람 검토나 독립 물리 정답 인증이 아니다.', '',
        '[전체192장 갤러리](gallery/index.html)는 회복·손상·최대 위치/회전 악화·각 축 최대 개선을 사전 규칙대로 함께 보여준다. [사례 선택](gallery/GALLERY_MANIFEST.json)에는 결합/중복 규칙이 있다. 원관측·추정 pose·참조는 별도 표시하고, F1은 같은 pose의 양쪽 family, V는 같은 선택 pose의20지역 RGB/NULL을 보여준다. 표시 사본의 기존 privacy redaction은 측정 입력을 바꾸지 않았다. GitHub에서는 HTML이 실행되지 않으므로 저장소를 내려받아 로컬 서버로 열어야 한다.', '',
        '## 남는 판단', '',
        '최종 수치 해석과 다음 수정 한 가지는 [DECISION_KO.md](DECISION_KO.md)에 기록했다. 운영 기본값과 논문은 자동 교체하지 않았다. 후보가 없는41장, 구별되지 않는 실패 tradeoff, B 관측의 실제 재료 지지 미확정, 합성→실사 차이는 계속 남는다.', '']
    for population in ('TEST49','REMAINDER143','CAL46'):
        lines += ['## '+population+' — 모든 기준·모든 seed', '',metric_table(result,population),'']
    public.mkdir(parents=True,exist_ok=True)
    (public/'REPORT_KO.md').write_text('\n'.join(lines),encoding='utf-8')
    summary=[]
    for arm in reps:
        m=p[arm]
        variants=[p[arm]] if arm in result['arms'] and group_name(arm)==arm else [p[group_name(arm)+'_s'+str(s)] for s in SEEDS]
        summary.append([label(arm),'/'.join(str(v['successes']) for v in variants),
            number(m['translation_cm']['mean'],2),number(m['rotation_deg']['mean'],2),
            f"{m['T_gt50cm']}/{m['T_gt100cm']}/{m['R_gt45deg']}"])
    (public/'README.md').write_text('\n'.join(['# 위험 감독과 지지 정보 실험','',
        f"동일 D1 후보 bank에서 {len(attempts)}회 새 fit을 실행했다. 모든238행 출력을 고정한 뒤 PRIMARY192/TEST49/REMAINDER143/CAL46을 평가했다. 독립 실사 test0장; 운영 기본값·논문 변경 없음.",'',
        '대표는 seed20261018이다. 성공 수는 고정 세 seed 순서18/19/20이며 기준 단일 출력은 한 숫자다. 평균은 대표의 전체192장 값이다.','',
        table(['방법','성공 /192 (모든seed)','평균 위치cm','평균 회전°','T>50/100 · R>45'],summary),'',
        '[전체 보고서](REPORT_KO.md) · [판단과 다음 한 가지](DECISION_KO.md) · [전체 결과 CSV](COMPARISON_ALL_ARMS.csv) · [모든192장 갤러리](gallery/index.html) · [재현](REPRODUCE.md) · [검산](VERIFICATION.json)','',
        '![전체 성공과 큰 오류](figures/success_and_large_errors.png)','',
        '![회복과 손상](figures/recovery_damage_and_common_failure.png)','',
        'GitHub는 HTML을 실행하지 않는다. 재현 안내의 로컬 서버로 상대 자산을 함께 열 수 있다. 원 RGB·private bank·weights는 공개하지 않았다. 결과 패널의 RGB는 사용자 권리가 확인된 PRIMARY 자료의 표시 사본이다. CAL/합성 RGB는 게시하지 않았다.','']),encoding='utf-8')
    write(run_dir/'REPORT_BUILD_RECEIPT.json',dict(status='WRITTEN_PENDING_FINAL_DECISION_AND_PUBLIC_AUDIT',
        prediction_manifest_sha256=sha(run_dir/'PREDICTION_MANIFEST.json'),evaluation_seal_sha256=sha(evaluation/'EVALUATION_SEAL.json'),
        source_sha256=sha(__file__),artifacts={n:sha(public/n) for n in ('README.md','REPORT_KO.md')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--public',type=Path,required=True)
    args=parser.parse_args()
    write_reports(args.run_dir,args.public)
