"""Write the source-only semantic result and clarify projection count units."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def save(p,r):
    with p.open('x') as f:json.dump(r,f,ensure_ascii=False,indent=2);f.write('\n')


def main():
    out=ROOT/'output';summary=json.loads((out/'SUMMARY.json').read_text());table=[]
    for arm,g in summary['groups'].items():
        s=g['reference_to_canonical_px']
        table.append(f"| {arm} | {s['n']} | {s['mean']:.9g} | {s['variance']:.9g} | {s['std']:.9g} | {s['median']:.9g} | {s['P90']:.9g} | {s['max']:.9g} | {g['old_reference_gt8_count']}/{g['canonical_reference_gt8_count']} |")
    four=[]
    for r in summary['four_known_cases']:
        four.append(f"| {r['id']} | {r['corner']} | {r['reference_to_canonical_px']:.9g} | {r['saved_error_to_wire_reference_px']:.6f} | {r['error_to_existing_source_canonical_px']:.6f} |")
    report='''가림 판단이 일부 틀려도 다른 유효 대응점으로 자세를 구하고 자기 가림 코너를 재투영했을 때, 기존 단순 대조보다 실제 위치·회전이 좋아졌는가?

**아직 전체245 실사에서 개선은 확인하지 못했다. 이번 검산은 실패 원인 하나를 분리했다.** 실제 경계 support의 연장 교차가 잘못된 canonical corner ID를 가리켜 큰 오차가 생겼다는 가설은 검사한 source CAL573 pair에서 지지되지 않았다. 저장 wire 참조와 같은 ID의 기존 합성 모델 코너 투영 차이는 최대0.000032551px였다. 8.14~19.30px 사례는 참조를 canonical 투영으로 바꿔 비교해도 유지된다. 이것이 실사 경계 ownership이나 실제 자세 개선의 증명은 아니다.

**코너의 정의 자체와 예측 위치의 정확도는 구분해야 한다.** [수정 POS wire 규칙](../../../scripts/research/pallet_kp_supervision_gate_20261010_v1/actual_wire_proposal.py#L98)은 actual wire의 축·고정 좌표·실제 point가 intended source edge와 기존1e−5×diagonal 허용오차 내에서 일치하도록 검사한다. 실제 triangle들이 공유하는 비공면 wire·depth·mask도 검사한다. bbox는 depth bound에 사용하며 nonexistent bbox segment를 positive target으로 만들어내지 않는다. [관측 decoder](../../../scripts/research/pallet_boundary_corner_refiner_20261010_v2/observations.py#L154)는 동일 corner ID에 incident한 두 관측선을 교차한다. 실제 partial edge의 연장으로 virtual cuboid corner를 얻는 것은 그 자체로 오류가 아니다. 그 virtual point가 실제 메쉬 정점이어야 한다는 조건도 없다.

이번에는 source CAL128개 라벨을 기존 SHA/bytes와 대조하고, READY의 기존K,R,t·dimensions와 같은 값인지 확인했다. [기존 source geometry](../../../scripts/research/pallet_observation_refiner_20261009_v1/source_audit.py#L85)의 dimensions 순서, annotation별 nearest8/unique8 phase 배정, 최대오차<0.05px 조건을 그대로 사용했다. 저장된573개 virtual reference를 같은 native corner ID의 기존 합성 자세 투영과 비교했다. 새 line fit·PnP·학습·RGB·실사 자세 성능 평가·설정 변경은0이다.

아래는 **source 참조 간 픽셀 차이**의 전체 분포다. 실사 위치·회전 성능 표가 아니다. 분산은 모집단 분산, P90은 선형 보간이다.

| arm | 모든 pair 수 | mean px | variance px² | std px | median px | P90 px | max px | 기존/같은ID canonical 참조에서 >8px 후보 수 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
TABLE

source annotation과 canonical projection 사이 phase 오차의 최대는0.000263134px로 기존0.05px 검사를 통과했다. certified support가 해당 canonical edge의 투영 직선에서 벗어난 수직거리도 검사했다. arm별 최대는 GEOMETRY_ONLY0.000018147px, IMAGE_NO_ROLE0.000018506px, IMAGE_ROLE0.000028065px였다. 따라서 검사한 source에서 두 true support line의 정의가 head 교점의8~19px 오류를 설명한다는 근거는 없다. 전체 source, unseen source나 실사까지 같은 결론으로 확대하지 않는다.

| 기존 source case | corner | wire 참조→canonical 차이 px | 예측→wire 오차 px | 예측→canonical 오차 px |
|---|---:|---:|---:|---:|
FOUR

![같은4개 source 사례의 기존 실제 support·예측 경계·교점](../pallet_point_admission_mechanism_20261011_v25/source_plot/ALL4_SOURCE_BOUNDARY_AND_CORNER.png)

이 그림은 V25의 동일한 저장4사례이며 새 fit으로 바꾸지 않았다. 앞3개는 기존 최종 radius8px admission에서 제외된다. 마지막의 거의 직교한 두 경계는 admission을 통과하지만 source N3/H/LOO는 보존되지 않아 UNKNOWN이다. 이번 참조 검산은 초록 wire reference가 같은 corner ID의 기존 모델 투영에 거의 같음을 추가 확인한 것이다. 보라 반경이 reference를 충분히 포함하는지와 실제 자세 산출 여부는 별도 조건이다.

**실제로 남은 문제는 예측 관측의 오차와 검증/가설 선택이다.** [V25 전체133점 LOO 추적](../pallet_point_admission_mechanism_20261011_v25/RESULT_KO.md)에서는 잘못 채택된 실사13점 모두 최종 fit/inlier에 남았다. 3점은 LOO fit이 전부 기존8px proxy GOOD여도 잘못 채택됐다. 또 [V23 PnP 가설 감사](../pallet_consensus_cause_audit_20261011_v23/RESULT_KO.md)에서는 경계 채택0인 경우에도 BAD 대응을 포함한 더 큰 합의가 선택되는 실패가 있었다. source 정의 정합성을 확인한 것과 이 문제를 해결한 것은 다르다. LOO retained noise·배치·다중해의 기여와 실사 covariance의 유효성은 아직 분리하지 못했다. 가림을 완벽히 분류해야 한다는 결론으로 바꾸지 않는다.

**실행량 단위를 명확히 한다.** 원 protocol/receipt의 `new_synthetic_known_pose_projection_calls=128`은 frame 단위 투영 작업 수였다. 실제 scalar `project()`는8점×128프레임=**1024회** 호출됐다. 원 코드·receipt를 덮어쓰지 않고 [단위 설명](EXECUTION_UNIT_CLARIFICATION.json)을 추가했다. 기존 합성R,t로 순방향 투영한 것이며 새R,t 추정이나 실사 재평가가 아니다. 공개 reader가 독립 검산하면서 하는1024회는 원 실행과 별도로 기록한다.

**self-occ 계약과 현재 실사 결과는 유지한다.** LOO는 H∪{k}, endpoint LOO는 H∪{a,b}를 제외한다. 최종 fit에는 H 초기좌표와center8을 넣지 않으며 NEW R,t가 나오면 H를 한 번 재투영하고 다시 fit하지 않는다. 마스크 일부 오류만으로 프레임을 실패시키지 않고 관측 부족·수치 실패·다중해·N3 fallback을 구분한다. [전체245 원 보고서](../pallet_whole_u_closure_20261011_v21/RESULT_KO.md)의 NEW240/fallback5/완전실패0, mean T/R/ADDsym=9.745696cm/12.622492deg/18.580216cm와 N3·ordinary 대조/분포/공통 산출집합/전체 운용/시간은 이번에 재계산하거나 바꾸지 않았다. Clean153/Moderate92이며 Severe74는 제외했다. 실사 수치는 DEV GEOMETRIC_PROXY이고 독립 실측 R,t가 아니다.

2164 query numerical-ray miss→NONE 오류는 이미 수정됐다. 감사 당시 준비/CPU gradient와 이후 승인된 고정9000업데이트의 실제 학습을 구분한다. 이번 추가학습0이며 수정 전 IMAGE_ROLE0/319를 수정 후 결과로 주장하지 않는다. main N3의 오류도 아니다. P0 네 통제변형, 실사 실제 경계 ownership, source covariance의 실사 transfer는 미검증이다.

원행은 [128 frame phase/기존 source 좌표](FRAME_ROWS.jsonl.gz), [573 pair와 canonical-line support 잔차](CORNER_ROWS.jsonl.gz), [모든 분포·같은4사례](SUMMARY.json), [기존128라벨 SHA/bytes와 실행 전 봉인](PROTOCOL.json), [실행 후 원본 불변](CHECKS.json)에 있다. 공개 compact reader는 upstream READY/CEILING/V25 SHA와 projection·phase·거리·집계를 검산한다. 원 source 라벨128개의 바이트를 다시 읽거나 physical ownership을 재인증하는 reader는 아니므로 원 producer의 실제 라벨 검산 범위와 구분한다.

```bash
python3 -I -S -B scripts/research/pallet_source_corner_semantics_20261011_v26/check_public.py _docs/experiments/pallet_source_corner_semantics_20261011_v26 --checkout CHECKOUT
```

요구한 위치·회전 개선 목표는 아직 미달이다. 참조/ID 정의 차이를 현재 source 오류 원인으로 임의 수정하지 않으며, retained noise와 다중해를 구분하는 검증 없이 threshold·weight를 바꿔 성능이 좋아질 때까지 반복하지 않는다. 새 정책을 시험하지 않은 상태에서 개선됐다고 보고하지 않는다.
'''.replace('TABLE','\n'.join(table)).replace('FOUR','\n'.join(four))
    (out/'RESULT_KO.md').write_text(report.rstrip()+'\n')
    save(out/'EXECUTION_UNIT_CLARIFICATION.json',dict(original_field='new_synthetic_known_pose_projection_calls=128',
        original_code_and_receipts_preserved=True,source_frames_projected=128,
        scalar_project_function_calls_original=1024,model_points_projected=1024,
        original_field_unit='frame operations, not scalar Python project calls',
        new_PnP=0,new_line_fit=0,new_training=0,new_RGB=0,new_real_pose_accuracy=0,
        public_checker_projection_calls_are_separate=True))
    save(out/'SOURCE_REVIEW.json',dict(source_only_review=True,reviewer='corner_semantics_source_v25',
        canonical_phase_source_rule_agrees=True,virtual_reference_not_physical_vertex_claim=True,
        projection_count_unit_clarified=True,agent_artifact_import_execute_write_Git_calls=0,
        scientific_values_computed_by_ROOT=True))
    save(out/'GOAL_STATUS.json',dict(status='active',accuracy_goal_achieved=False,
        source_reference_semantic_gap_audit_completed=True,source_current_large_errors_not_explained_by_reference_gap=True,
        real_pose_improvement_not_verified=True,new_training=0,new_real_pose_accuracy=0))
    print(json.dumps(dict(passed=True,report_lines=len(report.splitlines()),source_scalar_projections=1024)))


if __name__=='__main__':main()
