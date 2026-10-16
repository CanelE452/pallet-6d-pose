"""Record completed decoder verification after the initial necessary-gate audit."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def main():
    before=ROOT/'output';decoder=ROOT/'decoder';final=ROOT/'final';final.mkdir()
    s=json.loads((decoder/'SUMMARY.json').read_text());assert s['complete']
    checks=json.loads((decoder/'CHECKS.json').read_text());assert checks['passed']
    necessary=json.loads((before/'SUMMARY.json').read_text())
    rows=[]
    for arm,c in s['counts'].items():
        bad=c.get('error_gt8px:selected_decoder_corner',0)
        rows.append(f"| {arm} | 128 | {c['conditional_rows']} | {c['conditional_selected_corner']} | {bad} | {c['accepted_corners']} | {c['frames_ge4']} |")
    four=[]
    for r in s['previously_unknown_four_rows']:
        c=r['final_decoder_candidate'];errs=[x['error_px'] for x in r['support_query_errors']]
        four.append(f"| {r['id']} | {r['corner']} | {r['saved_error_px']:.6f} | {max(errs):.6f} | {c['absolute_normal_determinant']:.6f} | {c['radius_px']:.6f} | {c['nominal_query_spacing_px']:.6f} | {c['max_extrapolation_ratio']:.6f} | selected |")
    querytable=[]
    for arm,r in necessary['saved_step3000_query_TEST_metrics'].items():
        querytable.append(f"| {arm} | {r['counts']['positive_accepted']}/4497 | {100*r['positive_adoption_rate']:.3f}% | {r['positive_candidate_error_px_mean_conditional_accepted']:.6f}px | {100*r['no_match_false_acceptance']:.3f}% |")
    report='''가림 판단이 일부 틀려도 다른 유효 대응점으로 자세를 구하고 자기 가림 코너를 재투영했을 때, 기존 단순 대조보다 실제 위치·회전이 좋아졌는가?

**아직 방법 전체의 개선을 확인하지 못했다.** 고정 H의 LOO 제외와 최종 재투영 계약은 지켰지만, 최신 쉬움153+중간92=245 실사 결과는 단순 대조보다 위치·회전 평균이 나쁘다. 이번에는 저장된 합성 경계 출력의 마지막 승인 단계까지 실제 검산해, 코너 정의 문제·관측 오차·PnP 해 선택 문제를 구분했다. 새 학습·영상·모델 추론·PnP·자세 성능 재실험은 시작하지 않았다.

**지금 확인된 문제는 두 단계다.** 첫째, query의 경계 위치가 정확해도 두 선의 교점으로 만든 코너는 충분히 정확하지 않을 수 있고, 기존 uncertainty/extrapolation 승인 조건이 일부 큰 오차를 통과시킨다. 둘째, 실사에서는 보정점을 전혀 쓰지 않아도 잘못된 대응을 포함한 더 큰 consensus가 최종 PnP 순위에서 이겨 자세가 악화된다. self-occ를 초기 좌표로 되돌리거나 모든 가림을 완벽히 분류하는 것이 해결책이라는 근거는 없다.

**계획으로 끝내지 않고 마지막 decoder까지 확인했다.** 먼저 기존 세 모델의 CAL128 geometry573행에서 영상 범위와 `radius/initial_query_spacing<=1`만 산술 검산했다. IMAGE_ROLE >8px 후보33개 중29개는 그 선쌍의 필수조건 위반으로 확실히 거절되고4개는 원행만으로 최종 승인이 미확정이었다. 이어 이미 저장된84×66 logits를 기존 decoder·동일 CAL threshold·동일 원 좌표·동일 NumPy1.26.4로 검산했다. **그4개는 모두 source decoder에서 코너 관측으로 실제 선택된다.** line/TLS와 intersection은 이전 출력의 고정 계산을 재현했으며 calibration이나 threshold를 바꾸지 않았다. 실사 LOO/최종 자세에 이4개가 들어갔다고 주장하지 않는다.

| 모델 | 기존 CAL 프레임 | 모두 POS support인 조건부 참조 후보 | 그중 decoder 선택 | 그중 오차 >8px 선택 | 전체 decoder 코너 선택 | 코너4개 이상 프레임 |
|---|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(rows)+'''

세 arm의 source confidence cutoff와 조건부 후보 모집단은 서로 다르다. 이 표를 실사 위치·회전의 모델 우열로 치환하지 않았다. >8px는 기존 진단 집계 구간일 뿐이며 실제 admission을 8px로 바꾸지 않았다. 선택231 중 certified reference가 있는214만 위치 오차를 확인했다. 나머지17을 BAD·NONE으로 처리하거나 전체 source 코너 정확도를 추정하지 않는다. 전체 source 코너4개 이상22프레임도 새 pose 산출을 뜻하지 않는다.

![조건부 source 후보 전573행의 원래 필수조건](ALL573_SAVED_CANDIDATES.png)

이 그림은 첫 단계의 저장행 감사다. 가로축은 반경/초기 query 간격, 세로축은 actual-wire support의 참조 교점과 후보의 오차다. x=1은 기존 gate, y=8px는 설명용이다. 파란색은 두 필수조건을 통과했음을 표시하며 그림의 `final unknown`은 첫 단계의 한계를 보존한 표시다. 후속 decoder 검산은 그 상태를 아래 원행으로 확정했다. 모든573행을 넣었고 새 실사 영상이나 미실행 결과는 넣지 않았다.

**통과한4개 원행:** 각 후보는 geometry enable·원영상 범위·예측 교점의 외삽 cap·불확실성/간격 조건을 통과했고 같은 corner의 최종 decoder 후보로 선택됐다. support 최대오차가0.885px와1.114px인 두 사례에서도 코너 오차가8.661px와11.449px다. 다른 두 사례는 query 오차도4px 이상이다. 이것은 모든 잔여 오류가 교차각만의 문제라는 증거가 아니다.

| source ID | corner | 참조 교점 오차 px | support query 최대오차 px | 두 법선의 절대det | 반경 px | 초기 간격 px | 예측 외삽률 | 최종 decoder |
|---|---:|---:|---:|---:|---:|---:|---:|---|
'''+ '\n'.join(four)+'''

가장 큰258.055898px 중간 후보는 영상 밖이어서 실제 decoder가 거절한다. 그 후보의 반경1342.919435px도 초기 간격을 크게 넘는다. `참조오차<=recorded_radius`라는 사후 비교는 admission 자체가 아니다. 원행의 `true_corner_support_extrapolation_ratio`는 참조 교점의 외삽률이며, 이번 실제 decoder의 예측 교점 외삽률과 구분했다. 기존 uncertainty gate가 큰 증폭을 상당수 걸러낸 효과와 위4개를 통과시키는 한계가 동시에 있다.

**코너의 특성 정의 자체에 새 확정 오류를 찾지는 못했다.** 모델은8코너나 가림 종류를 직접 출력하지 않고12edge×7query 각각에 rawpixel ±32의65 후보 또는 NONE을 출력한다. 선3개를 요구하는 것이 아니라 서로 다른 query3개 이상으로 하나의 semantic edge를 만들고, 같은 native corner에 incident한 두 선의 교점을 만든다. 기존 actual-wire 가상 교점과 native projected-cuboid의 CAL348 차이는 평균0.0000306744px/최대0.0002457794px였다. 이 조건부 지원 선쌍에서는 edge ordering·bin·axis·코너 ID 정의 불일치가 확인되지 않았다. physical mesh vertex 존재 또는 실사 경계 ownership까지 입증한 것은 아니다. unsupported는 source 모델 지원 부족이고 physical edge 부재를 뜻하지 않는다.

**오류 수정과 학습의 시점을 분리한다.** 수정 전 별도 브랜치 `training.py.targets()`의 numerical ray miss를 no_match로 감독한 것은 실제 오류다. 대응이 있는2164 query를 잘못 NONE 처리했고, 실제 mesh·depth·wire 증거로 POSITIVE/NONE/IGNORE를 수정했다. 당시 감사 시점에는 준비·CPU gradient만 완료됐지만, 이후 승인된 고정3×3000 업데이트는 실제 완료됐다. 이번 추가 학습은0회다. 기존 IMAGE_ROLE0/319는 수정 전 모델의 실패 기록으로 보존하며 수정 후에도0/319라고 재표기하지 않는다. main N3→cornerSubPix의 학습 오류로 부르지도 않는다.

다음은 이미 저장된 마지막step3000의 source TEST128 query probe다. 공통 POS4497/NONE1966/IGNORE4289 중 raw66-way argmax를 평가했다. POS 조건부 평균 bin 오차에 TLS·교점·LOO·PnP가 포함되지 않아0.806732px를 코너 또는 자세 정확도로 읽을 수 없다. TEST는 P0 내부 분할이고 이전 감독 오류 진단에서 이미 열람했으므로 독립 실사 일반화 시험도 아니다.

| 모델 | 채택 POS | POS 채택률 | 채택POS 평균 query 오차 | NONE 오채택률 |
|---|---:|---:|---:|---:|
'''+ '\n'.join(querytable)+'''

GEOMETRY_ONLY는 기존 구현상 image0:19를 없애며 geometry와ROLE25:28을 유지한다. IMAGE_NO_ROLE는ROLE만 없애고 IMAGE_ROLE은28채널 모두 사용한다. ROLE은 hull의 BOUNDARY/INTERNAL/UNAVAILABLE 입력이며 별도 가림 정답 출력이 아니다. 이 입력 정의를 바꾸거나 세 모델을 더 학습하지 않았다. P0 외부 가림·비가림 방해물·저대비의 네 통제 변형은 여전히 별도 검증되지 않았다.

**실사 관측 선택의 실패:** 기존245 원행에서는85프레임/133코너가 실제 보정점으로 채택됐다. reference 오차가 N3보다 줄어든80점과 커진53점이 함께 있으며, 기존8px point proxy의 GOOD120/BAD13이다. 고정 H∪{k}를 뺀 LOO pose에서 q 잔차가 native보다 작아졌다고 q 위치가 실제로 더 정확하다는 보장은 없다. remaining 대응이 일부 잘못됐거나 LOO pose가 모호하면 그 비교 자체가 잘못된 관측을 선택할 수 있다. 이번 source4개를 이 실사53점과 동일 사례로 연결하지 않았다.

**실사 강건 PnP 선택의 실패:** 기존 전체-U 기본 해법이 모두 거절된59프레임 중 known-GOOD4 이상24프레임은 모두 그4점 조합과 유효 root를 가졌고22프레임은 그 root에서 승인된 refit도 있었다. 위치·회전이 모두 나빠진8프레임 중7프레임은 경계 보정점 채택0개였다. `eval_pallet07:1778652138515809024`의 선택 해는 BAD2,4까지 포함한6 inlier로 GOOD4 출발 가설의5 inlier보다 개수/SSE 순위가 앞섰다. N3 2.188809cm/0.454581°에서 새 출력10.716346cm/86.942401°로 악화됐다. GOOD4는 기존 annotation의8px proxy이고 정확한 무오차 대응이 아니다. 다른 root의 실제 T/R가 더 낫다고 새로 채점하지 않았으며, 단순히 GOOD4가 있었다는 것만으로 선택 해가 좋아야 한다고 단정하지 않는다.

**LOO는 self-occ 초기 좌표를 무시하는 것이 맞다.** corner LOO는고정 predicted-H와 검사점k를 함께 제외한 H∪{k}, endpoint LOO는 H∪{a,b}다. 새 최종pose의 fit/score에는 H 초기좌표와center8을 넣지 않는다. 남은 실제 관측으로 강건 PnP를 구하고 새 R,t가 있으면 H 코너를 한 번 재투영해 교체한다. 재투영좌표를 독립 관측으로 다시 fit하지 않는다. 예측 마스크와 사람 마스크가 다르다는 이유로 프레임을 실패시키지 않고, 충분한 유효 대응·배치·수치 안정성과 다중해 여부를 별도 기록한다. 새pose가 없을 때 반환하는 기존N3는 별도fallback이다. 이 계약 준수는 위치·회전 개선의 증거와 구분한다.

기존 전체245 운용 결과는최신 경로240 NEW/5 fallback/0 완전 실패다. mean(T cm/R deg/ADDsym cm)은 N3 9.754548/10.914842/17.028647, 단순HYBRID ordinary 9.207154/10.613464/16.439686, 최신강건 경로9.745696/12.622492/18.580216이다. ordinary 대조는최종 H 제외 계약이 같지 않으므로 그대로요구 방법의 성공으로대체할 수없다. 모든 실사자세 수치는기존 annotation/K/cuboid로 만든 DEV GEOMETRIC_PROXY이며독립 physical R,t가아니다. 전체분포·평균/분산/표준편차/중앙값/P90·공통산출집합·운용결과·실제RGB 사례는 [전체245 보고서](../pallet_whole_u_closure_20261011_v21/RESULT_KO.md), [가설/LOO 원행 감사](../pallet_consensus_cause_audit_20261011_v23/RESULT_KO.md)에 보존했다. Severe74는이번245에서제외했으며 역사적319를삭제하지않았다. 이번 감사 시간을 기존 실제전체경로 timing으로대신하지않았다.

**남은 최소 검증:** source4개의 최종decoder 여부를확인하는 작업은 이번에완료했다. 남은 핵심은같은관측·K·치수·H·finite4 roots를고정하고, LOO 관측선택과 PnP 해선택의기준을 사전에명시해 별도허용된 자세평가에서확인하는것이다. 테스트오차를보고 threshold를바꿔반복하거나 GOOD proxy를배포판선택기에넣지않는다. 독립 physical pose/ownership 진실값 부재도남아있다. source4개 오차만 줄여놓고 실사목표가 해결됐다고 보고할 수없다. **정확도 목표는아직완료아님**이다.

코드 위치는 [query/line/intersection/admission](../../../scripts/research/pallet_boundary_corner_refiner_20261010_v2/observations.py#L117), [조건부source 참조](../../../scripts/research/pallet_boundary_corner_refiner_20261010_v2/calibration.py#L140), [native코너 계약](../../../scripts/research/pallet_corner_mechanism_audit_20261010_v1/cal_corner_contract.py#L138), [query TEST probe](../../../scripts/research/pallet_kp_supervision_repair_20261010_v1/retrain.py#L288), [고정 H의LOO](../../../scripts/research/pallet_cornerwise_independent_20261010_v4/selection.py#L135)다. 이번 추가 코드는 [저장행 감사](../../../scripts/research/pallet_source_corner_admission_20261011_v24/saved_admission_audit.py), [동일decoder 검산](../../../scripts/research/pallet_source_corner_admission_20261011_v24/replay_saved_decoder.py)에서확인할수있다.

원자료는 [첫단계573행](ROWS.jsonl.gz), [첫단계통계](SUMMARY.json), [decoder573결정행](decoder/DECISIONS.jsonl.gz), [decoder384프레임](decoder/DECODED_FRAMES.jsonl.gz), [최종4행과실행량](decoder/SUMMARY.json), [원logits/좌표/코드SHA 결속](decoder/PROTOCOL.json), [실제검산결과](decoder/CHECKS.json)다. decoder의 모든573후보 좌표·σ는기존저장값과일치했고, IMAGE_ROLE의 selected_queries3757/lines606/accepted_corners231/frames_ge4 22/line-pair hypotheses9656/unknown_IGNORE49의여섯집계는이전검산과일치했다. UNKNOWN을 NONE으로변경하지않았다.

실제추가실행량은첫단계저장573행과, source decoder384프레임·TLS dispatch2620회·교점 dispatch707회다. 이는 이미평가된고정logits에대한검산이다. 새 detector/head/posefit/pose-score/학습/ray/RGB는모두0이다. 독립stdlib검산은저장선1302개의공분산과교점707개의교점·공분산·영상범위·외삽·uncertainty gate·코너별최소선택을확인했다. NumPy decoder를단순다시호출한검사로대체하지않았다.

게시된 파생행과 decoder 산술의 검산 명령은아래와같다. 상류supervision 전체·새모델·실사ownership·물리정답을재실행했다고주장하지않는다.

```bash
python3 -I -S -B scripts/research/pallet_source_corner_admission_20261011_v24/check_saved_public.py _docs/experiments/pallet_source_corner_admission_20261011_v24
python3 -I -S -B scripts/research/pallet_source_corner_admission_20261011_v24/check_decoder_public.py _docs/experiments/pallet_source_corner_admission_20261011_v24/decoder
```

상류파일이복원된checkout에서는 `saved_admission_audit.py --source CHECKOUT --output NEW_OUTPUT`과 `replay_saved_decoder.py --source CHECKOUT --output NEW_DECODER_OUTPUT --necessary-rows NEW_OUTPUT/ROWS.jsonl`로같은join과decoder검산을재실행할수있다. 후자는기존 NumPy1.26.4를필요로하며 torch/OpenCV/SciPy/영상모듈import를막는다. 모든원본inputs와코드SHA는실행전에봉인했고실행후입력hash가동일했다. branch전용정상commit/push와main보존은 PUBLICATION_PROTOCOL/REMOTE_CONFIRMATION에기록한다.
'''
    with (final/'RESULT_KO.md').open('x') as f:f.write(report)
    with (final/'GOAL_STATUS.json').open('x') as f:json.dump(dict(status='active',accuracy_goal_achieved=False,
        source_final_decoder_admission_verified=True,four_SOURCE_candidates_admitted=True,
        method_accuracy_not_claimed=True,new_training=0,new_RGB=0,new_pose_accuracy=0,
        remaining=['Real observation selection and finite-root selection accuracy.','Independent physical truth.','P0 controlled variations.']),f,indent=2);f.write('\n')
    print(json.dumps(dict(prepared=True,report_lines=len(report.splitlines()),four_source_admissions_closed=True)))


if __name__=='__main__':
    main()
