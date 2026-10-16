"""Explain completed saved-row audits without changing the pose method."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def main():
    src=json.loads((ROOT/'output/SUMMARY.json').read_text());real=json.loads((ROOT/'real_loo/SUMMARY.json').read_text())
    bounds=json.loads((ROOT/'bounds/SUMMARY.json').read_text());assert src['complete'] and real['complete'] and bounds['passed']
    bs={(r['frame_id'],r['corner']):r for r in bounds['all_BAD_cases']}
    sourcetable=[]
    for a,c in src['counts'].items():
        sourcetable.append(f"| {a} | {c['decoder_selected_corners']} | {c['point_admission:accepted']} | {c['point_admission:UNCERTAINTY_EXCEEDS_PNP_OBSERVATION_RADIUS']} | {c['frames_ge4_point_admitted_before_H']} |")
    fourtable=[]
    for r in src['four_decoder_admitted_large_error_cases']:
        fourtable.append(f"| {r['id']} | {r['corner']} | {r['saved_corner_error_px']:.6f} | {r['recorded_radius_px']:.6f} | {r['normal_matrix_condition']:.6f} | {r['error_amplification_ratio']:.6f} | {'point admission通過・N3/H/LOO未確認' if r['existing_point_admission_accepted'] else '既存radius gateで除外'} |")
    fourtable=[s.replace('point admission通過・N3/H/LOO未確認','point admission 통과·N3/H/LOO 미검증').replace('既存radius gateで除外','기존 radius gate에서 제외') for s in fourtable]
    realtable=[]
    allgood=[]
    for r in real['all_BAD_cases']:
        b=r['selected_point'];lower=bs[(r['frame_id'],r['corner'])]['LOO_projection_error_lower_bound_px']
        badids=r['LOO_fit_quality_ids']['BAD'];delta=r['saved_delta_vs_N3']
        realtable.append(f"| {r['frame_id']} | {r['corner']} | {b['candidate_error_px']:.3f} | {b['candidate_LOO_residual_px']:.3f} | {len(r['native_remaining_quality_ids']['GOOD'])} | {badids} | {lower:.3f} | {delta['translation_cm']:+.3f} | {delta['rotation_deg']:+.3f} |")
        if not badids and not r['LOO_fit_quality_ids']['UNKNOWN']:allgood.append(r)
    assert len(allgood)==3
    report='''가림 판단이 일부 틀려도 다른 유효 대응점으로 자세를 구하고 자기 가림 코너를 재투영했을 때, 기존 단순 대조보다 실제 위치·회전이 좋아졌는가?

**현재 전체 245장에서는 개선을 확인하지 못했다.** 이번에는 코너 후보가 실제 최종 관측으로 공급되는 단계까지 추적하고, 실사에서 채택된 133점 전부의 LOO 근거를 확인했다. 새 학습·합성 생성·영상 추론·LOO/PnP·자세 성능 실험과 설정 변경은 모두 0이다. 현재 방법의 정확도 문제를 해결했다고 보고하지 않는다.

**저장 자료에서 실패가 확인된 단계는 LOO의 관측 선택과 최종 PnP의 가설 선택이다.** LOO에서 자기 가림 H와 검사점 k를 제외하는 처리는 맞다. 하지만 held-out 투영도 틀릴 수 있다. 남은 N3 점의 오차·배치·다중해가 이 오류에 각각 얼마나 기여했는지는 아직 분리하지 못했다. 후보가 틀린 투영에 더 가까우면 reference 좌표가 나빠져도 채택된다. 이후 같은 픽셀 잔차를 보는 강건 PnP에서도 잘못된 점이 inlier로 남을 수 있다. 이것은 완벽한 가림 분류를 요구해야 한다는 결론이나 새 산술 버그의 확인과 다르다.

```mermaid
flowchart LR
 A[Base 기반 query 위치 또는 NONE] --> B[query 합의와 TLS]
 B --> C[같은 코너의 두 선 교점과 불확실성]
 C --> D[decoder 영상·외삽·반경/간격 검사]
 D --> E[별도 최종선 일관성·반경 8px 검사]
 E --> F[N3 거리와 H 제외]
 F --> G[H와 k를 뺀 LOO 잔차 비교]
 G --> H[실제 관측 finite4 합의 PnP]
 H --> I[새 R,t로 자기 가림 코너 재투영]
```

**코너 query 좌표와 native N3를 섞지 않았다.** source의 `frozen_selected_points`는 Base 예측이다. native N3나 실제 hidden-H가 저장된 값이 아니므로, 이를 N3 대신 넣어 source native-basin/LOO 통과를 계산하지 않았다. radius 제한은 N3/H와 독립이어서 저장 자료만으로 검산할 수 있다. [Base source 생성](../../../scripts/research/pallet_observation_refiner_20261009_v1/training.py#L117), [실사 Base/N3 분리](../../../scripts/research/pallet_three_head_observation_20261010_v7/pipeline.py#L215), [source cache 결속](../../../scripts/research/pallet_three_head_observation_20261010_v7/calibration.py#L171)에서 provenance를 확인했다.

**기존 point-admission은 실제로 효과가 있다.** 지난 [decoder 검산](../pallet_source_corner_admission_20261011_v24/RESULT_KO.md)은 8.14~19.30px 오차 후보 4개가 source decoder에서 선택됨을 확인했다. 이번에는 그 뒤 [별도 최종 관측 admission](../../../scripts/research/pallet_boundary_corner_refiner_20261010_v2/pipeline.py#L65)의 calibrated radius≤8px 조건을 검산했다. **4개 중 3개는 기존 코드가 이미 제외한다.** 남은 1개만 이 admission을 통과하며, 실제 N3/H/LOO나 자세 사용 여부는 미검증이다. 4개 모두 최종 PnP에 들어간 오류라고 읽으면 안 된다.

| 모델 | source decoder 선택 코너 | 별도 point admission 통과 | 기존 반경8px로 제외 | H 적용 전 admission 코너4개 이상 프레임 |
|---|---:|---:|---:|---:|
'''+ '\n'.join(sourcetable)+'''

같은 source CAL128/arm의 저장 프레임 384개·선1302개를 검사했다. 최종 선 일관성은 전1302개에서 만족됐다. 이 수는 boundary-only 입력 후보이며 H·native-basin·LOO를 통과한 수나 실제 새 자세 산출률이 아니다. arm별 조건부 모집단과 cutoff가 다르므로 이 표를 실사 자세 모델 우열로 치환하지 않는다. NONE/IGNORE 지원 선에는 POS 위치 오차를 정의하지 않았다.

| source ID | corner | 참조 교점 오차 px | calibrated 반경 px | 법선행렬 condition | 두 선 잔차→교점오차 증폭 | 별도 admission |
|---|---:|---:|---:|---:|---:|---|
'''+ '\n'.join(fourtable)+'''

**교점 오차의 원인도 저장 값으로 분해했다.** 두 예측 선의 법선을 행으로 가진 A와 offset b에 대해 예측 교점 p는 Ap=b다. 기존 actual-wire 참조 교점 g의 두 선 잔차 r=Ag−b라면 **g−p=A⁻¹r**다. 새 선이나 자세를 fit하지 않고 573행 모두에서 이 항등식을 확인했다. 법선이 거의 평행하면 작은 선 오차가 큰 교점 오차로 증폭된다. 예측 선에 대한 참조 support의 공통 이동과, support 중심에서 코너까지의 방향/거리 성분도 따로 기록했다.

그러나 남은 source `P0__shard_07_f0942:k1`은 condition=1.011507이고 증폭=0.997336으로, 거의 평행한 선의 문제가 아니다. edge9의 예측 support RMS는0.223425px인데 실제 POS support 중심의 예측선 bias는2.983511px다. 코너까지의 방향/거리 성분4.931870px가 더해져 참조 코너에서 선 잔차7.915380px가 된다. 최종 코너 오차8.144162px에 비해 반경은2.660038px다. 선 내부에 잘 맞는 coherent한 query 집합이 실제 경계 위치에 정확히 놓였다는 보장은 없다. 다른 원인 세 사례를 일괄 같은 유형으로 정의하지 않았다.


![source 4개 전체의 저장 경계 support와 코너 확대](source_plot/ALL4_SOURCE_BOUNDARY_AND_CORNER.png)

위 행은 각 두 선의 retained support이고 아래 행은 같은 교점 주변70px 확대다. 점은 예측 query, +는 certified source POS target, 빨간×는 예측 교점, 초록별은 저장 wire 참조, 보라원은 calibrated 반경이다. 앞3개는 기존 cap에서 이미 제외되고 마지막은 admission만 통과한다. 거의 평행한 앞3개와 거의 직교한 마지막 사례를 구분할 수 있다. 물리 자세 GT나 source N3/H/LOO 성공을 나타내는 그림이 아니다. 새 line fit은0이다.

**uncertainty는 어디서 쓰이고 어디서 빠지는가:** query 반경은 TLS weight에 들어가고 local covariance surrogate를 만든다. 두 선의 covariance에서 corner 반경을 계산해 decoder와 point admission에서 사용한다. 이 surrogate는 query 간 상관과 두 선의 cross-covariance를 생략한다. source q95 보정은 해당 CAL의 조건부 지원집합에 대한 경험적 보정이며 실사 전체의95% 보장이 아니다. 이후 [LOO 비교](../../../scripts/research/pallet_cornerwise_independent_20261010_v4/selection.py#L159)와 [point consensus](../../../scripts/research/pallet_finite_edge_consensus_repair_20261011_v13/solver.py#L123)는 raw Euclidean 픽셀 잔차를 쓰고 covariance로 whitening하지 않는다. 현재 명시된 고정 정책과 일치하므로 covariance 미사용을 새 구현 버그라고 재수정하지 않았다.

**실사에서 실제로 통과한 잘못된 관측은13개다.** 쉬움153+중간92=245장의 기존 원행을 전부 join했다. 실제로 채택한85프레임/133점 중 기존 annotation8px proxy의 GOOD120/BAD13이고, N3 대비 좌표 오차가 좋아진80점·나빠진53점이다. BAD13은 서로 다른13프레임이다. **13점 모두 최종 NEW pose의 fit과 최종 inlier에 남았다.** 13점의10개는 LOO fit에 이미 BAD인 N3 점이 있었고,3개는 LOO fit이 전부 known-GOOD여도 잘못된 후보를 선택했다. GOOD은 무오차 물리 대응이 아니라 기존8px proxy라는 한계를 유지한다.

![실제 채택된 전133점의 LOO 잔차와 reference 오차](plot/ALL133_SAVED_ADOPTED_POINTS.png)

왼쪽에서 모든 후보는 LOO 잔차≤8px지만13점의 reference 오차는8px를 넘는다. 보라색 테두리는 LOO fit이 모두 known-GOOD였던3점이다. 오른쪽은 같은133점의 N3 오차와 채택 후보 오차를 비교하며 대각선 위53점은 좌표가 나빠졌다. 좋은/나쁜 결과를 빼지 않았다. 이 그림의 reference는 DEV pixel proxy이고 physical ownership은 UNKNOWN이다.

아래는13점 전체다. remaining-GOOD는 H∪{k}를 뺀 finite/sentinel-valid native pool에서 **native N3 오차**로 계산했다. 이 pool 집계에는 영상 범위 검사가 없으므로 실제 LOO eligibility의 보증으로 읽지 않는다. 최종 hybrid의 input 오차를 native-only LOO에 잘못 적용하지 않았다. 모두 H∪{k} 제외를 지켰다. `LOO lower`는 후보 reference 오차 e와 후보-LOO 잔차 v로부터 삼각부등식 |e−v|≤LOO projection reference 오차를 계산한 하한이다. 새 투영이나 R,t 채점을 하지 않았다. ΔT/ΔR는 이전에 저장된 최종 자세−N3의 오차 차이다.

| frame ID | k | 후보 오차 px | LOO 잔차 px | remaining finite GOOD 수 | LOO fit의 BAD ID | LOO lower px | 저장 ΔT cm | 저장 ΔR deg |
|---|---:|---:|---:|---:|---|---:|---:|---:|
'''+ '\n'.join(realtable)+'''

BAD13 중6점은 이 finite native pool의 known-GOOD가4개 이상이었다. 실제 in-image eligible 점 수나 정확도·다중해 해소를 그 수만으로 보장하지 않았다. BAD13 중8점은 LOO projection의 reference 오차 하한 자체가8px를 넘는다. LOO fit이 전부 known-GOOD였던3점 중에서도2점은 이 하한이8px를 넘는다. 따라서 잘못된 가림만 완벽히 고치면 이 문제가 반드시 없어지는 것이 아니다.

관측 오류와 최종 자세 결과도 구분했다. BAD13 중3점은 N3보다 좌표 오차가 줄었지만 여전히8px를 넘고,10점은 좌표 오차가 커졌다. 이13프레임의 최종 자세는 위치·회전 모두 개선1/모두 악화8/혼합 또는 동일4다. **잘못된 점이 남았는데 자세는 좋아진 경우도 있다.** 단일점이 자세 변화를 일으켰다는 인과관계로 이 집계를 해석하지 않았다. 예측 마스크와 사람 마스크가 다르다는 이유로 프레임을 실패시키지도 않았다.

**PnP에서 남은 문제:** 기존59장 특별 분기 감사에서 known-GOOD4 조합과 유효 root가 존재해도 BAD 대응을 포함한 더 큰 consensus가 count/SSE 순위에서 앞서는 사례가 있었다. 그 both-worse8프레임 중7프레임은 경계 보정점 채택0개였다. 즉 head와 LOO만 고치면 최종 자세가 반드시 좋아진다는 근거도 없다. 현재 point 순위는 count 우선·truncated SSE 다음이고, current V21은 non-equivalent 해가 같은 count와 수치적으로 같은 SSE인 경우에 모호성을 거부한다. 이는 국소 rank6이나 실제 noise 불확실성 아래의 유일성을 증명하는 조건이 아니다. [기존 전체 가설 추적](../pallet_consensus_cause_audit_20261011_v23/RESULT_KO.md)과 [current solver 변환](../../../scripts/research/pallet_whole_u_closure_20261011_v21/solver.py#L76)에 근거를 보존했다.

**self-occ 처리 조건은 유지한다.** LOO는 H∪{k}, 선 endpoint LOO는 H∪{a,b}를 제외한다. 실제 최종 fit에는 H의 초기좌표와center8을 넣지 않는다. 다른 실제 관측으로 새R,t가 나오면 H 좌표를 한 번 재투영으로 대체하며 다시 PnP에 넣지 않는다. 새pose 없음의 기존N3 반환은 별도fallback이다. visible점을 일부 잘못 제외하거나 wrong point 일부가 남는 것만으로 전체 프레임을 실패시키지 않는다. 원 관측 수·배치·정확도·다중해·수치 상태를 구분한다.

최신 전체245 NEW240/fallback5/완전실패0의 mean(T cm/R deg/ADDsym cm)은9.745696/12.622492/18.580216이다. 고정N3는9.754548/10.914842/17.028647, 단순HYBRID ordinary는9.207154/10.613464/16.439686이다. ordinary 대조의 H fit 계약은 다르므로 이를 요구 방법의 성공으로 대체하지 않는다. 이번에 새 성능을 계산하지 않았고, 기존 평균·분산·표준편차·중앙값·P90·공통 산출집합·전체 운용·실제RGB·전체 경로 시간은 [전체245 보고서](../pallet_whole_u_closure_20261011_v21/RESULT_KO.md)에 유지했다. Severe74는 현재245에서 제외하고 과거319 결과는 보존했다. pose 값은 기존 annotation/K/cuboid의 DEV GEOMETRIC_PROXY이며 독립 실측 R,t가 아니다.

**이미 고친 것:** 별도 `training.py.targets()`에서 실제 대응2164 query를 numerical ray miss 때문에 no_match로 잘못 감독한 오류는 실제mesh/depth/wire POS/NONE/IGNORE로 고쳤다. 당시 준비/CPU gradient만 완료였던 감사 시점과 이후 승인된 고정3×3000 실제 재학습을 구분한다. 현재 추가 학습0이다. 수정 전 IMAGE_ROLE0/319를 수정 후 결과로 재사용하지 않는다. main의N3→cornerSubPix 오류로 부르지도 않는다. 이번에는 기존 radius gate를 새로 추가하거나 고친 척하지 않았다. P0 네 통제 변형과 독립 physical truth는 여전히 미검증이다.

**다음에 검증해야 할 최소 범위는 명확해졌다.** 충분한 관측 수와 작은 raw LOO 잔차에 더해, retained 대응의 오차가 held-out prediction에 어떻게 전파되는지와 서로 다른 PnP 해가 noise 범위에서 구분되는지를 평가해야 한다. N3 point noise와 source covariance의 실사 유효성을 검증하지 않고 weight 또는 ambiguity threshold를 바꿔 좋은 결과가 나올 때까지 반복하지 않는다. 성능 재실험을 금지한 현재 감사 범위에서 새 정책의 자세 개선을 검증했다고 주장할 수 없다. 현재 방법은 실행·검산 가능하지만, 요구한 정확도 목표는 **아직 완료 아님**이다.

실행 자료는 [source573 교점 분해](source/CORNER_ROWS.jsonl.gz), [source1302 선](source/LINE_ROWS.jsonl.gz), [source admission384프레임](source/FRAME_ROWS.jsonl.gz), [실사133 LOO 원행](real_loo/ROWS.jsonl.gz), [실사13 전체근거](real_loo/SUMMARY.json), [거리 하한133행](bounds/ROWS.jsonl.gz)다. SHA/bytes/코드는 실행 전 protocol에 봉인했고 실행 후 원본은 같았다. 모든 수치 실행은ROOT가 수행했으며 subagent는 소스만 읽었다. 실제 추가 모델/line-fit/LOO/PnP/투영/자세평가/학습/RGB는0이다. 573 교점의 저장 값 항등식과133 관측의 저장 결과 join을 검사했다.

공개 파생행의 독립 검산:

이 reader는 파생행의 항등식·저장 잔차·집계·SHA를 검산하며 원 head/LOO/PnP나 상류 join을 재실행하지 않는다. 별도 source replay는 아래 명령으로 이전 Git blob 10개와 현재 코드 2개의 명시적 부분집합에서 실제 실행했고, source 파생행 gzip 3개·SUMMARY·CHECKS가 이번 원 실행과 바이트 단위로 같았다. fresh clone이나 physical truth 재인증으로 보고하지 않는다. [실제 replay 검산](PUBLIC_SOURCE_REPLAY_CHECKS.json)에 범위를 기록했다.

```bash
python3 -I -S -B scripts/research/pallet_point_admission_mechanism_20261011_v25/check_public.py _docs/experiments/pallet_point_admission_mechanism_20261011_v25
```

상류 source READY와 이전 V24 공개자료가 준비된checkout에서는 source admission·오차분해 join도 재실행할 수 있다. 가중치나 영상 없이 동작하며 새 출력 디렉터리를 지정해야 한다.

```bash
python3 -I -S -B scripts/research/pallet_point_admission_mechanism_20261011_v25/replay_public_source.py --checkout CHECKOUT --output NEW_OUTPUT
```

실사133 join의 상류는 이미 공개한 V21 geometry/Posthoc와 V23 selected rows다. 원 archive 복원은 기존 [public reader](../pallet_whole_u_closure_20261011_v21/PUBLIC_READER_KO.md)에 따른다. 경로를 맞춘 read-only 입력을 사용해야 하며 검산이 원 supervision/physical ownership/새 자세 성능까지 증명한다고 주장하지 않는다. 전용 브랜치 정상commit/push·원격SHA 일치·main/사용자변경 보존은 게시 receipt에서 확인한다.

원 real saved join도 정책·함수 본문을 바꾸지 않고 ROOT/G/P/SELECTED 경로만 바꿔 실제 실행했다. frozen geometry 복사와 이전 Git blob을 사용한 명시적 fixture이며 fresh clone은 아니다. 전체245 join 결과의133행 gzip·SUMMARY·CHECKS는 원 실행과 바이트 단위로 같았다. [실제 real replay 검산](PUBLIC_REAL_REPLAY_CHECKS.json)과 [fixture provenance](PUBLIC_REAL_FIXTURE_PROTOCOL.json)에 남겼다. 위 V21 archive를 복원한 checkout에서는 다음 명령이 같은 입력SHA/bytes와 출력 동일성을 검사한다.

```bash
python3 -I -S -B scripts/research/pallet_point_admission_mechanism_20261011_v25/replay_public_real.py --checkout CHECKOUT --output NEW_OUTPUT
```

'''
    with (ROOT/'RESULT_KO.md').open('x') as f:f.write(report)
    plot=ROOT/'plot/ALL133_SAVED_ADOPTED_POINTS.png'
    with (ROOT/'ROOT_VISUAL_REVIEW.json').open('x') as f:json.dump(dict(reviewed=True,
        method='ROOT view_image actual PNG display',image_sha256=hashlib.sha256(plot.read_bytes()).hexdigest(),
        all133_points_included=True,all3_all_GOOD_LOO_BAD_points_marked=True,
        axes_units_and_proxy_scope_readable=True,new_RGB=False),f,indent=2);f.write('\n')
    with (ROOT/'SOURCE_REVIEW.json').open('x') as f:json.dump(dict(source_only_reviewers=['v7_report','v7_scalar_review'],
        code_bug_confirmed=False,findings=[
            'Source frozen_selected_points are Base anchor, not native N3.',
            'Decoder normalized uncertainty and later8px point admission are distinct.',
            'Source covariance is a local diagonal-error surrogate with no independent real coverage guarantee.',
            'Corner radius participates upstream; point LOO/fit/rank use raw pixel residuals without covariance whitening.',
            'Current V21 replaces any-cost same-support ambiguity veto with tied-only, not the literal immutableS15 branch.'
        ],subagent_artifact_numeric_import_write_calls=0),f,indent=2);f.write('\n')
    with (ROOT/'GOAL_STATUS.json').open('x') as f:json.dump(dict(status='active',accuracy_goal_achieved=False,
        source_point_admission_and_all133_real_LOO_audits_completed=True,
        source_3_of_4_rejected_by_existing_gate=True,
        all13_real_BAD_points_in_final_fit_and_inliers=True,
        original_error_already_repaired=True,new_method_bug_confirmed=False,
        new_learning=0,new_pose_accuracy_runs=0,remaining=['Real observation uncertainty/selection accuracy.',
        'Finite-root ranking under observation noise.','Independent physical truth and P0 variants.']),f,indent=2);f.write('\n')
    print(json.dumps(dict(prepared=True,report_lines=len(report.splitlines()),all13_real_cases_included=True)))


if __name__=='__main__':
    main()
