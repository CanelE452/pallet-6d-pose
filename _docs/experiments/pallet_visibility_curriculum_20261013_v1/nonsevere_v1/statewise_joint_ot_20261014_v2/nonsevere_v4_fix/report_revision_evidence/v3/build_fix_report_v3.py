"""Write the final report from sealed actual-RGB evaluations, without tuning."""
from collections import Counter
from pathlib import Path

from . import common as C
from . import evaluation_interval_v4_2b as V
from . import statewise_evaluation_v4 as E


def main():
    run, fix = V.OUTPUT, E.FIX
    _, protocol, _ = E._verify_prediction_seal(run)
    score = C.read(run / 'POSTSEAL_SCORING.json')
    assert score['prediction_seal']['sha256'] == C.sha(run / 'PREDICTION_SEAL.json')
    assert score['target_achieved']
    cal = C.read(fix / 'truncation_pose_validation_v4_2/CAL_POSE_VALIDATION_SCORE.json')
    model = C.read(V.MODEL)
    previous = C.read(fix / 'evaluation_bias_v4_1/POSTSEAL_SCORING.json')
    test = score['summary']['TEST']
    rows = [r for r in score['rows'] if r['cohort'] == 'PRIMARY_NONSEVERE']
    assert len(rows) == 95 and test['frames'] == 49
    old_by_id = {r['id']: r for r in previous['rows']}
    counts = {part: dict(Counter(s for r in rows if r['partition'] == part for s in r['states']))
              for part in ('CAL', 'TEST')}
    changes = [{'id': r['id'], 'partition': r['partition'],
                'native_ids': [k for k, (a, b) in enumerate(zip(old_by_id[r['id']]['states'], r['states'])) if a != b]}
               for r in rows if old_by_id[r['id']]['states'] != r['states']]
    table = []
    for name, label in [('ORIGINAL_N3_PNP', '초기 backbone + PnP'),
                        ('FROZEN_STATEWISE_V3', '수정 전 상태별 방법 v3'),
                        ('OLD_RGB_SINGLE_FAMILY', '기존 RGB 보정'),
                        ('BOUNDARY_PROGRESSIVE', '기존 경계 보정'),
                        (E.ARM_ALL, '최종 상태별 방법 v4.2')]:
        m = test['pose'][name]
        table.append({'method': name, 'label': label, 'translation_cm': m['metrics']['translation_cm']['mean'],
                      'rotation_deg': m['metrics']['rotation_deg']['mean'], 'success': m['success_count'], 'n': 49})
    table_text = '\n'.join(f'| {r["label"]} | {r["translation_cm"]:.3f} | {r["rotation_deg"]:.3f} | {r["success"]}/49 |' for r in table)
    main_m = test['pose'][E.ARM_ALL]
    no_ot = test['pose'][E.ARM_NO_OT]
    rgb = test['pose']['OLD_RGB_SINGLE_FAMILY']
    delta = main_m['metrics']['translation_cm']['mean'] - rgb['metrics']['translation_cm']['mean']
    changed = test['OT_information']['BI_ge3']['pose_changed_vs_NO_OT']
    ot_frames = test['OT_information']['BI_ge3']['frames_with_OT']
    p07 = 'eval_pallet07:1778652168786111744'
    new_case = next(r for r in rows if r['id'] == p07)
    old_case = old_by_id[p07]
    human_self_as_BI = {part: sum(x['n'] for x in score['summary'][part]['state_confusion']
                                  if x['actual'] == 'SELF_OCCLUDED' and x['predicted'] in ('B', 'I'))
                       for part in ('CAL', 'TEST')}
    test_confusion = test['state_confusion']
    true_O = sum(x['n'] for x in test_confusion if x['actual'] == 'EXTERNAL_OCCLUDED')
    true_O_found = sum(x['n'] for x in test_confusion
                       if x['actual'] == 'EXTERNAL_OCCLUDED' and x['predicted'] == 'O')
    visible_as_O = sum(x['n'] for x in test_confusion
                       if x['actual'] == 'DIRECT_VISIBLE' and x['predicted'] == 'O')
    report = f'''**최종 상태별 키포인트 방법 v4.2 — 실제 영상 비교와 수정 결과**

이번 수정은 보이는 점의 오차를 과신한 자세 선택과, 초기 추정만으로 화면 밖 점을 확정한 문제를 고쳤다. 같은 clean/moderate TEST49에서 최종 방법은 평균 위치 오차 **{main_m['metrics']['translation_cm']['mean']:.3f}cm**, 회전 오차 **{main_m['metrics']['rotation_deg']['mean']:.3f}°**, 성공 **{main_m['success_count']}/49**다. 성공은 위치 오차 5cm 미만과 회전 오차 5° 미만을 모두 만족한 경우다.

| 방법 | 평균 위치 오차(cm) ↓ | 평균 회전 오차(°) ↓ | 성공 수 ↑ |
|---|---:|---:|---:|
{table_text}

기존 RGB보다 평균 위치 오차 차이는 {delta:.6f}cm다. 이는 반복 사용한 49장의 탐색 평가 결과다. 일반적인 우월성, 통계적 유의성, 외부 SOTA보다 우수함을 입증한 결과로 표현하지 않는다. 경계 보정과 성공 수는 같고 평균 오차가 낮다.

**키포인트 정의와 실행 순서**

| 기호 | 이름 | 최종 처리 |
|---|---|---|
| S | Self-occluded keypoint | 초기 기하가 S로 정한 ID를 점 관측·영상 프롬프트·PnP 대응점에서 제외하고, 마지막 자세로만 투영한다. |
| B | Visible boundary keypoint | 보이는 파렛트–배경 경계점. 실제 경계선 교점으로 보정을 시도하고, 근거가 없으면 원예측을 fallback 관측으로 보존한다. |
| I | Visible interior keypoint | 파렛트 내부에 있는 보이는 점. 내부 RGB 선 교점으로 보정을 시도하고, 근거가 없으면 원예측을 fallback 관측으로 보존한다. |
| O | Externally occluded keypoint | 외부 물체에 가려졌다고 판정한 점. 점마다 가능한 위치 영역을 공동 제약으로 사용한다. |
| T | Truncated keypoint | 화면 밖 근거를 확인한 점. 가능한 화면 밖 공간을 공동 제약으로 사용한다. |

U는 별도 물리 상태가 아니라 판정 보류다. 초기 backbone 좌표·등록 PnP·카메라 K·알려진 파렛트 치수를 입력받아 S와 B/I 기하 역할을 정하고, non-S의 실제 보임을 판정한다. B/I를 서로 다른 영상 방식으로 보정한 후 그 좌표와 원 공분산을 고정한다. 그 관측, CAL 오차 모형, 초기 자세 사전항, 모든 개별 O/T 영역을 한 강체 자세에 적용한다. S는 그 자세의 투영으로 완성한다. 아홉 번째 중심 metadata와 native/Q 대응 관계는 유지한다.

실제 영상 근거로 보정된 B/I는 TEST 266개 중 185개이며, 81개는 원예측 fallback이다. CAL은 267개 중 152개 보정·115개 fallback이다. 이후 '고정한 B/I 관측'에는 이 fallback도 포함하므로, 모든 B/I를 영상 교점으로 보정했다는 뜻이 아니다.

**왜 이전 결과가 좋지 않았는가**

1. 보정 B/I의 좁은 공분산과 강제 한계가 실제 오차를 충분히 포함하지 못했다. 특히 보이는 점이 적으면, 2D 오차가 줄어도 잘못된 PnP 후보가 선택될 수 있었다. CAL에서 총 관측 오차를 추정하고 부드러운 강건 손실과 유한한 초기 자세 사전항을 적용했다. CAL 세션 제외 검증으로 선택한 강도는 λ=1, 편향 반영 비율은 β=1이다.
2. 초기 backbone 점과 초기 자세의 투영이 모두 화면 밖이라는 조건으로 T를 강제했지만, 두 위치의 오차 범위를 고려하지 않았다. 실제 CAL에는 주석상 화면 안에 있는 T 4개가 있었다. 이 잘못된 제약은 올바른 자세를 제외할 수 있었다.
3. O 영역이 넓으면 B/I와 사전항이 이미 선택한 자세를 포함하므로 추가 정보를 주지 못한다. 상태를 정의하고 제약을 추가했다는 사실만으로 자세 오차 감소가 보장되지는 않는다.

T 수정은 CAL의 유효 non-S 관측 264개에서 구한 정규화 위치 오차의 **최댓값 {model['radius_ratio']:.12f}**을 사용한다. 큰 오차를 제거하거나 native 대응 관계를 사후 변경하지 않았다. 초기 non-S backbone 점을 중심으로 한 오차 원 전체가 화면과 떨어져 있을 때만 조건부 T를 유지한다. 원이 화면과 겹치거나 접하면 U로 보류한다. 독립적으로 확인한 T와 모든 O 영역은 유지하며, 남은 T의 crop 제약도 그대로 둔다.

CAL의 기존 T 13개는 모두 U가 됐다. 최종 전체 주 평가에서도 기존 조건부 T 38개(CAL13/TEST25)가 모두 U로 보류돼 **실제로 남은 T 제약은 0개**다. 이는 T를 잘 검출했다는 성과가 아니라, 확정할 수 없는 T를 강제하지 않은 결과다. CAL 성공은 {cal['current_interval']['success_count']}/46으로 유지됐지만 평균 위치 오차는 {cal['frozen_beta1_baseline']['metrics']['translation_cm']['mean']:.6f}→{cal['current_interval']['metrics']['translation_cm']['mean']:.6f}cm로 조금 늘었다. 이를 숨기지 않고, 사전에 정한 CAL 수락 기준을 통과한 뒤 전체 평가했다. 이 최댓값은 미래 오차 포함을 보장하지 않는다. wood 세션을 제외한 모형은 해당 세션 114점 중 9점을 포함하지 못했다.

**O/T가 실제로 얼마나 기여했는가**

최종 TEST에서 BI≥3이면서 O/T 영역이 있는 프레임은 {ot_frames}개이며, 같은 관측·오차 모형·사전항에서 O/T를 제거했을 때 자세가 달라진 프레임은 **{changed}개**다. NO_OT 대조군은 위치 오차 {no_ot['metrics']['translation_cm']['mean']:.6f}cm, 성공 {no_ot['success_count']}/49다. 따라서 이번 TEST 개선을 O 영역이 자세 후보를 더 잘 줄인 성과로 설명할 수 없다. 개선은 관측 오차 처리·자세 후보 선택을 고치고 잘못된 T 제약을 피한 결과다. CAL의 BI 두 점 + O 영역 사례는 모호한 자세 집합으로 따로 기록하며, 유일한 자세를 증명하지 않는다.

실제 T 오판 사례 `{p07}`에서는 위치 오차가 {old_case['metrics'][E.ARM_ALL]['translation_cm']:.3f}→{new_case['metrics'][E.ARM_ALL]['translation_cm']:.3f}cm로 바뀌었다. [이 실제 사진의 수정 전후](figures_interval_effect_v4_2/{p07.replace(':', '__')}.png), [기존 O/T가 자세를 바꾼 모든 실제 사례](figures_interval_effect_v4_2/index.html), [최종 B/I·O 영역·S 투영 예시](figures_interval_v4_2/index.html)를 함께 제공한다. O 타원은 CAL 오차로 확대한 backbone 위치 범위이며 가림 물체의 실제 윤곽이 아니다.

**검증 범위와 남은 한계**

이번 CAL 학습·주 평가에서 severe는 제외했다. CAL46과 TEST49의 모든 실패·fallback을 포함했으며, 실제 crop 3개는 주 지표에서 제외했다. 98장 전부 현재 RGB 상태 판정·마스크·B/I 보정·전체 API를 실행했다. 초기 N3_SUBPIX backbone 예측은 동결한 입력이다. DOPE·ResNet을 새로 학습하거나 모든 backbone에서 같은 개선을 확인한 실험은 아니다.

초기 S 제외는 **초기 기하가 S로 정한 ID**에 대한 보장이다. 초기 자세나 등록 형상이 틀리면 실제 자기 가림점이 B/I로 잘못 분류될 수 있다. 실제 사람 SELF가 B/I로 남은 수는 CAL {human_self_as_BI['CAL']}개, TEST {human_self_as_BI['TEST']}개다. native 모델과 평가 reference의 불일치 9장(CAL2/TEST7)도 전체 점수에서 제외하지 않았다. 단순한 색 비교는 예측 위치 자체의 오차와 실제 모서리의 보임 여부를 구분하지 못하므로 이 문제를 모두 해결하지 않는다.

TEST의 실제 외부 가림 {true_O}점 중 {true_O_found}점을 O로 찾았지만, 실제 보이는 {visible_as_O}점도 O로 잘못 분류했다. 따라서 가림 판별 자체가 완전히 해결됐다고 말할 수 없다. 현재 T 규칙은 보수적이며 많은 조건부 T를 보류한다. 자연 영상에서의 T 검출 능력이나 O 영역의 추가 자세 개선은 이 결과만으로 입증되지 않았다. 상태 판정기·B/I 보정·O 영역 모형은 기존 CAL 학습을 동결했으므로 새 보정 부분의 세션 제외 검증을 전체 파이프라인 OOF라고 부르지 않는다. 평가 자세는 기존 주석에서 파생한 C2 대칭 reference이며 독립적인 물리 계측 GT가 아니다.

수식과 fallback 규칙은 [방법 설명](METHOD_DETAILS_V4_1_KO.md), 고정한 비교 목표는 [실험 계획](FIX_PROTOCOL_V1.json), 수치와 프레임별 결과는 [최종 점수](evaluation_interval_v4_2b/POSTSEAL_SCORING.json), 실제 RGB 실행 계약은 [검사 기록](evaluation_interval_v4_2b/CONTRACT_CHECKS.json)에 있다. 이전 v3/v4/v4.1 결과·소스·실패 기록을 보존했다.

**실제 이미지 한 장 실행**

저장소 루트에서 필요한 기존 모델 파일이 있는 `pallet-yolo26` 환경으로 실행한다. 입력 JSON에는 raw keypoints, 초기 pose, K, native 치수/모델이 필요하며 GT 좌표는 사용하지 않는다. 이미 있는 output은 덮어쓰지 않는다.

```bash
python -m scripts.research.pallet_visibility_curriculum_20261013_v1.statewise_joint_cli_v4_2 \\
  --image data/evaluation/pallet_eval_v1/dev_existing/sessions/eval_outside/rgb/1778651519235162880.png \\
  --input _docs/experiments/pallet_visibility_curriculum_20261013_v1/nonsevere_v1/statewise_joint_ot_20261014_v2/EXAMPLE_INPUT.json \\
  --output /tmp/pallet-statewise-interval-example.json
```

실제 실행 결과와 전체 평가의 동일 프레임이 일치하는지는 [CLI 검증 기록](RUNTIME_CLI_V4_2_RECEIPT.json)에서 확인할 수 있다. 실제 영상에서 초기 S 좌표만 크게 교란한 쌍별 검사도 [별도 기록](actual_s_smoke_v4_2/COMPLETION_RECEIPT.json)에 남겼다. 이 수정은 연구용 API와 명령으로 제공하며 기존 기본 추정기를 자동 교체하지 않는다.
'''
    for name in ('README_KO.md', 'FINAL_RESULTS_V4_2.json'):
        if (fix / name).exists():
            raise FileExistsError('Preserve final report; use a versioned amendment')
    (fix / 'README_KO.md').write_text(report, encoding='utf-8')
    C.write(fix / 'FINAL_RESULTS_V4_2.json', {
        'schema': 'STATEWISE_INTERVAL_FINAL_RESULTS_V4_2', 'table': table,
        'target_checks': score['target_checks'], 'all_numeric_targets_met': score['target_achieved'],
        'primary_state_counts': counts, 'changed_T_frames': changes,
        'TEST_OT_information': test['OT_information'], 'human_SELF_predicted_BI': human_self_as_BI,
        'TEST_O_counts': {'actual_external': true_O, 'correct_O': true_O_found, 'visible_as_O': visible_as_O},
        'statistical_or_general_superiority_proved': False,
        'report_sha256': C.sha(fix / 'README_KO.md'), 'report_builder_sha256': C.sha(__file__),
        'prediction_seal_sha256': C.sha(run / 'PREDICTION_SEAL.json'),
        'score_sha256': C.sha(run / 'POSTSEAL_SCORING.json'),
        'frozen_solver_configuration': protocol['solver_configuration']})
    print(table_text)


if __name__ == '__main__':
    main()
