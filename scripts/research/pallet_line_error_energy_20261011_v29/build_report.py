"""Build a Korean report from sealed scientific rows and verified checks."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'output'
ARMS = ('GEOMETRY_ONLY', 'IMAGE_NO_ROLE', 'IMAGE_ROLE')
summary = json.loads((OUT/'SUMMARY.json').read_text())
assert json.loads((ROOT/'PRIVATE_READER_RESULT.json').read_text())['checks']['passed']
assert json.loads((ROOT/'figures/CHECKS.json').read_text())['all1194_rows_plotted']
lines = [
'가림 판단이 일부 틀려도 다른 유효 대응점으로 자세를 구하고 자기 가림 코너를 재투영했을 때, 기존 단순 대조보다 실제 위치·회전이 좋아졌는가?',
'',
'**현재 쉬움·중간245장에서는 개선을 확인하지 못했다. 기존 V21 mean 위치9.745696cm/회전12.622492deg는 HYBRID 일반 PnP9.207154cm/10.613464deg보다 나쁘다. 이번에는 새 성능 실험 없이, 경계 관측이 자체 잔차로는 곧게 보여도 잘못된 위치에 있을 수 있다는 문제를 저장 원행에서 확인했다. 해결 완료로 보고하지 않는다.**',
'',
'현재 범위는 Clean153+Moderate92=245이며 Severe74와 역사적 전체319는 보존한다. 실제 자세 참조는 기존 DEV GEOMETRIC_PROXY다. 독립 실측 physical truth나 실사 경계 ownership 정답이라고 바꾸어 부르지 않는다.',
'',
'**self-occ는 최종 fit에서 제외하고, 새 R,t가 나오면 재투영으로 바꾼다.** LOO는 초기 예측 self-occ H와 검사 대상k 모두 제외한 H∪{k}로 계산한다. 임시 heldout k를 self-occ로 분류하지 않는다. 최종 fit/score에는 H의 초기 좌표와center8을 넣지 않고, NEW 뒤 H를 한 번 재투영해 대체하며 재투영점으로 다시 fit하지 않는다. 남은 유효 대응이 충분하면 가림 분류 일부 오판만으로 프레임을 실패시키지 않는다.',
'',
'**문제가 있는 부분은 세 곳으로 구분된다.** 먼저 학습 감독의 numerical ray miss→NONE 오류는 이미 고쳤다. 다음으로 현재 관측 선택기는 경계 support의 내부 일치도와 교점 반경을 계산하지만 실제 위치의 편향까지 검증하지 못할 수 있다. 마지막으로 남은 N3로 구한 LOO projection도 틀릴 수 있고, 최종 PnP는 실제로 틀린 대응이 더 큰 합의를 얻을 때 그 해를 고를 수 있다. 자기 가림을 fit에서 제외하는 계약을 없애거나 초기 좌표를 유지하는 방식으로 바꾸어 해결하지 않는다.',
'',
'**새 실행은 고정 source CAL 선의 normal 오차 에너지 감사다.** 기존128 CAL 장면×3 arm=384 frame에서 선택된1302 선을 모두 읽었으며, 모든 support가 실제mesh shared-wire POSITIVE인1194 선에서만 좌표 오차를 계산했다. NONE/IGNORE가 섞인108 선은 좌표 오차를 정의할 수 없는 원행으로 남겼다. source anchor는 Base이며 source native N3/H/LOO는 저장되지 않았다. 이 자료를 실사 N3/LOO 검사로 대체하지 않는다.',
'',
'저장된 predicted normal n, support pᵢ, 실제 POS 대응 gᵢ와 기존 query radius rᵢ를 그대로 사용했다. wᵢ=1/max(0.5,rᵢ)², eᵢ=n·(pᵢ−gᵢ), μ=Σwᵢeᵢ/Σwᵢ로 두면 가중 MSE=μ²+Σwᵢ(eᵢ−μ)²/Σwᵢ다. μ는 V25 center bias의 부호 반전이므로 새 발견으로 세지 않는다. 새 정보는 평균 성분의 에너지 비율과 내부 편차, 동일 가중치의 actual RMS/TLS self RMS 차이다. within variance는 선택된 한 선의 가중 중심2차 모멘트이며 통계적 query 상관·공통 random noise·실사 covariance 보정계수 추정이 아니다.',
'',
'| 조건 | 전체 선 | all-POS 선 | 오차 미정의 선 | actual normal MSE > TLS self MSE | 평균 성분 비율 중앙값 | 총에너지 중 평균 성분 |',
'|---|---:|---:|---:|---:|---:|---:|',
]
for arm in ARMS:
    g = summary['groups'][arm]
    lines.append(f"| {arm} | {g['all_lines']} | {g['certified_POS_lines']} | {g['undefined_POS_error_lines']} | {g['actual_error_exceeds_self_error_POS_lines']} | {g['mean_component_fraction']['median']:.6f} | {g['aggregate_equal_line_mean_energy_fraction']:.6%} |")
lines += ['', '총에너지 비율은 선별 가중MSE를 선마다 같은 비중으로 더한 뒤 Σμ²/ΣMSE로 계산했다. 선별 비율의 평균·중앙값, 전체 query를 한 번에 가중한 비율과 다르다. 오차 에너지0인 선은0개였다. 아래 통계는 각 arm의 all-POS 선 원행에서 계산한 모집단 분산(ddof=0)·선형 보간 중앙값/P90이며 독립 표본 추론이 아니다.', '',
'| 조건 | 지표 | n | 평균 | 분산 | 표준편차 | 중앙값 | P90 |',
'|---|---|---:|---:|---:|---:|---:|---:|']
for arm in ARMS:
    for field, label in [('weighted_actual_normal_RMS_px','actual 가중normal RMS(px)'), ('TLS_self_RMS_weighted_px','TLS 가중self RMS(px)'), ('actual_RMS_unweighted_px','actual 비가중normal RMS(px)'), ('TLS_self_RMS_unweighted_px','TLS 비가중self RMS(px)'), ('mean_component_fraction','평균 성분 에너지 비율'), ('weighted_within_support_variance_px2','support 내부2차 모멘트(px²)'), ('abs_mean_over_local_offset_sigma','abs(μ)/저장 offset sigma')]:
        s = summary['groups'][arm][field]
        lines.append(f"| {arm} | {label} | {s['n']} | {s['mean']:.9g} | {s['variance']:.9g} | {s['std']:.9g} | {s['median']:.9g} | {s['P90']:.9g} |")
lines += ['', '![전체1194 선의 동일 가중치 actual/self RMS와 오차 에너지 분포](figures/ALL1194_LINE_ERROR_ENERGY.png)', '',
'위 산점도는1194개 전부이며 두 RMS의 가중치를 맞췄다. x축 symlog는0과 매우 작은 self RMS도 보존한다. 실제 값1194개 모두 축 범위 안에 있는 것을 ROOT가 확인했다. 아래CDF 점선은 선별 비율의 중앙값이고 제목의 aggregate energy는 총에너지 비율이다. [내보내기 SVG](figures/ALL1194_LINE_ERROR_ENERGY.svg)도 보존한다.', '',
'**공분산 수식이 그대로 계산됐다는 것과 정확한 위치를 인증한다는 것은 다르다.** [observations.py fit_line](../../../scripts/research/pallet_boundary_corner_refiner_20261010_v2/observations.py#L50)의 local scale=max(1, 비가중 standardized TLS residual MSE)는1302 선 모두1이었다. 최종 support가 |TLS residual|≤query radius 조건을 만족하면 그 normalized residual도1 이하라 이 floor에 머무를 수 있다. 저장2×2 parameter covariance 전부를 같은 DᵀWD로 재현했다. 따라서 단순 계산 버그라고 단정하거나 scale1 자체를 실패 조건으로 추가하지 않는다. 선 전체가 함께 이동하면 self RMS는 작아도 actual 위치 오차가 클 수 있다는 진단이다.', '',
'query radius=mode sigma×global CAL query_scale, line covariance=local scale×(DᵀWD)⁻¹, corner radius=intersection sigma×global CAL corner_scale은 서로 다르다. [교점 공분산 전파](../../../scripts/research/pallet_boundary_corner_refiner_20261010_v2/observations.py#L112)는 두 line variance를 diag로 넣고 cross-line covariance를 포함하지 않는다. parameter covariance2×2 자체가 항상 diagonal이라는 뜻은 아니다. [source CAL 보정](../../../scripts/research/pallet_three_head_observation_20261010_v7/calibration.py#L128)은 같은CAL에서 confidence/scale을 계산하므로 그CAL coverage를 독립 검증이나 실사 보증으로 쓰지 않는다.', '',
'**기존 IMAGE_ROLE 큰 코너4개를 임의로 뽑아 바꾸지 않았다.** 아래는 V25에서 이미 decoder-selected이며 코너 오차>8px인 네 경우의 두 support line 에너지다. source point-admission의 기존radius≤8 gate만으로3개는 제외됐고1개는 남았다. source native N3/H/LOO가 없어 이것을 최종 실사 채택·자세 결과라고 부를 수 없다.', '',
'| source frame / corner | 기존 코너오차px | point radius gate | edge | actual 가중normal RMSpx | TLS 가중self RMSpx | μ px | 내부2차 모멘트px² | 평균 성분 비율 |',
'|---|---:|---|---:|---:|---:|---:|---:|---:|']
for c in summary['existing_four_large_corner_cases']:
    for r in c['line_energy_rows']:
        lines.append(f"| {c['frame_id']} / k{c['corner']} | {c['existing_corner_error_px']:.6f} | {'통과' if c['existing_point_admission_accepted'] else '제외'} | {r['edge']} | {r['weighted_actual_normal_RMS_px']:.6f} | {r['TLS_self_RMS_weighted_px']:.6g} | {r['weighted_mean_error_px']:.6f} | {r['weighted_within_support_variance_px2']:.6f} | {r['mean_component_fraction']:.6f} |")
lines += ['', '![기존4개 source RGB와 물리적 경계 대응·교점 사례](../pallet_point_admission_mechanism_20261011_v25/source_plot/ALL4_SOURCE_BOUNDARY_AND_CORNER.png)', '',
'이미지는 기존 보존자료를 재사용했다. P0__shard_07_f0942/k1은 corner radius2.660px/실제 코너오차8.144px이며 교점 condition≈1.012로 수치상 잘 배치된 경우다. edge0/9의 실제normal RMS1.656/3.207px와 작은self RMS0.292/0.226px가 공존한다. V25의 edge9 bias2.984px와 orientation·distance 성분4.932px를 합한 corner normal residual7.915px와 연결되지만, 이번 energy만으로 잘못된 RGB 경계를 선택한 원인을 확정하지 않는다. [기존 기하 분해](../pallet_point_admission_mechanism_20261011_v25/RESULT_KO.md)와 [source ID/reference 검산](../pallet_source_corner_semantics_20261011_v26/RESULT_KO.md)을 함께 볼 수 있다.', '',
'**실사 LOO와 최종 자세 실패는 별도로 검증했다.** [V27](../pallet_observation_uncertainty_20261011_v27/RESULT_KO.md)의 실제 채택133점 중53점은 코너 좌표가 악화됐어도 원 LOO gain이 양수였다. G_LOO=G_REF−2(native−candidate)·(LOO−reference)의 exact 분해로 검증 기준의 오차가 선택을 뒤집을 수 있음을 확인했다. [V28](../pallet_loo_root_envelope_20261011_v28/RESULT_KO.md)에서 최대inlier 수의 저장 해 전부가 후보를 선호한64점 중22점도 좌표가 악화됐다. LOO 다중해 동의만으로 보정 정확도를 인증할 수 없다. 실사 참조가독립 physical truth라는 주장은 하지 않는다.', '',
'[V4 selection.py](../../../scripts/research/pallet_cornerwise_independent_20261010_v4/selection.py#L135)는 H∪{k} 제외 뒤 선택 projection과의 raw8px/squared gain으로 후보를 채택한다. 후보 반경은 LOO 자세의 predictive uncertainty가 아니다. [V21 solver policy](../../../scripts/research/pallet_whole_u_closure_20261011_v21/solver.py#L19)는 point-only full-U inlier count와 truncated SSE 및 수치 gate로 최종 자세를 고른다. source 선 에너지와 실사 LOO·최종T/R 실패 사이의 인과관계나 covariance correction 효과는 이번 감사로 입증하지 않았다. [V23](../pallet_consensus_cause_audit_20261011_v23/RESULT_KO.md)에는 boundary채택0인 실패에서도 잘못된 N3 대응을 포함한 해가 더 큰 합의를 얻은 사례가 있다. 그러므로 새 경계 head만 고치면 최종PnP가 해결된다고 할 수 없다.', '',
'**이미 고친 오류와 남은 문제.** [원 training.py targets](../../../scripts/research/pallet_observation_refiner_20261009_v1/training.py#L59)의 수치 광선miss를no_match로 감독한 오류로 source-test 실제 대응2164개가NONE이 됐었다. 실제mesh/depth/wire 기반 POSITIVE/NONE/IGNORE 감독 및 CPU gradient 준비를 완료했고, 이후 별도 승인된고정3×3000 재학습도 실제 완료했다([retrain adapter](../../../scripts/research/pallet_kp_supervision_repair_20261010_v1/retrain.py)). 감사 요청 당시에는 준비만 완료였다는 역사와 구분한다. 수정 전IMAGE_ROLE0/319는 수정 전 모델 결과이며 수정 후 결과로 재사용하지 않는다. main의 N3→cornerSubPix 오류가 아니다. 이번 추가학습0이다. P0 외부가림·비가림방해물·저대비의 네통제 변형은 미검증으로 남는다.', '',
'가림 분류를 완벽하게 만드는 것은 현재 해결의 필요조건이 아니다. 남은 대응의 수·배치·정확도를 확인하고, 잘못 남은 대응을 이상치로 버릴 수 있어야 한다. ≥4 대응/rank6/작은 내부잔차/다중해 동의는 각각 실제 위치·회전 정확도를 보장하지 않는다. 재투영은 구해진 자세의 결과를 숨은 코너에 배정하는 단계이며, 독립 관측을 추가하거나 잘못된자세를 정확하게 만드는 근거가 아니다.', '',
'**남은 최소 성능 검증.** 새 학습·RGB·seed 추가를 먼저 할 필요는 없다. 기존 고정 관측/가중치와 같은 finite4-subset solver로 마스크없는 강건PnP 대조를 유지하고, 하나의 사전고정 관측채택/가설선택 가설을 한 번 평가해야 한다. source에서 nativeN3/H/LOO를 갖춘 독립 감독으로 후보오차와heldout-prediction 오차를 분리해 검증하기 전에는 radius/root spread를 참조오차 상한으로 사용하면 안 된다. 현재DEV GT로 root/threshold를 사후 선택하면 배포 방법 검증이 아니다. 전체245의T/R/ADDsym 분포·공통산출집합·전체운용·NEW/fallback/fail·직접가시 손상·H재투영·실제전체경로 시간을 함께 비교해야 한다. 현재 읽기전용 요청이 새 정책 성능재실험을 금지하므로 이번에는 실행하지 않았고 효과는 미검증이다. 해당 범위를 열기 전에는 audit를 반복한 횟수로 정확도 개선을 대신하지 않는다.', '',
'**실행·재검산·게시.** 새 head inference/선fit/PnP/projection/학습/RGB/자세채점/전체경로시간 측정은 모두0이다. primary1회와 stdlib 독립reader1회가 각384/1302/1194를 검산했고, 게시 복사본reader도 같은Git blob을 읽어 검산한다. 독립reader는 within variance를 pairwise 가중차 공식으로 별도 재계산하고147분포cell·4사례·모든covariance·NONE/IGNORE108보존을 검사했다. 입력은 parent0c210572의Git blob SHA/bytes와 맞춰 sparse checkout의 누락 파일에도 private경로없이 재현된다. [실행영수증](PRIVATE_READER_RESULT.json), [봉인protocol](PROTOCOL.json), [원행](LINE_ROWS.jsonl.gz), [384 frame](FRAME_ROWS.jsonl.gz), [전체통계](SUMMARY.json), [검산](CHECKS.json), [실행량](EXECUTION_LEDGER.json)을 보존한다.', '',
'```bash',
'python3 -I -S -B scripts/research/pallet_line_error_energy_20261011_v29/check_public.py _docs/experiments/pallet_line_error_energy_20261011_v29 --checkout .',
'```', '',
'reader는기존Git parent의READY/decoded/V25 line·corner 및3개CAL/코드blob을 읽는다. 새clone나4.7GBarchive 복원은 이번검산에 필요없다. 원본 영상·가중치·사용자 변경·SOURCE의기존staged자료를 보존하며 research/observation-refiner-robust-pnp-20261009에 명시된파일만 정상commit/push한다. main/forcepush/자동merge는 하지 않는다. remote확인은 실제 [게시영수증](REMOTE_CONFIRMATION.json)에 기록한다. [기존전체실사결과와runtime](../pallet_whole_u_closure_20261011_v21/RESULT_KO.md)은 이번시간과 합산하지 않는다.',
]
dest = OUT/'RESULT_KO.md'
with dest.open('x') as f: f.write('\n'.join(lines)+'\n')
checks = dict(passed=True, report_lines=len(lines), bytes=dest.stat().st_size,
    sha256=hashlib.sha256(dest.read_bytes()).hexdigest(), complete_source_statistics_table=True,
    all_four_cases_eight_lines_included=True, new_pose_accuracy_claimed=False, accuracy_goal_achieved=False)
with (ROOT/'REPORT_BUILD_CHECKS.json').open('x') as f: json.dump(checks,f,indent=2);f.write('\n')
print(json.dumps(checks))
