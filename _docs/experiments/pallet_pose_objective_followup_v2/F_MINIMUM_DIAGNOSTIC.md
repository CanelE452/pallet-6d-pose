# F 최소 진단 — 기존 실제 가림 입력 계약의 재사용 감사

결론은 `PASS_EXISTING_ASSET_FEASIBILITY_ONLY`다. **현재 F의 교사 품질이나 학생 전달 성공을 측정하지 않았다.** 현재217/361에 새로 fit하거나 추론하지 않았고 과거 TRAIN 저장물5쌍을 CPU로 읽었다. 상세 SHA/검사 결과는 [JSON](F_MINIMUM_DIAGNOSTIC.json)에 있다. process wall1.08초, 내부 검사0.12초, GPU0/fit0/update0/신규추론0이다.

## 실제로 확인한 것

기존 [occlusion refiner v2 입력 lock](../pallet_occlusion_refiner_transfer_v2/E2_INPUT_LOCK.json)에는 DAY264에서 필터 통과한253쌍, 실제 가림185쌍이 있다. 이 중 가림이 있는 ID를 사전식 정렬한 첫5개만 선택했다. 성능·잔차·평가 결과로 고르지 않았다.

5개 artifact 및 원본 RGB의 SHA를 확인했다. clean/occluded RGB digest와 crop RGB tensor가 달랐고, 저장된 OCC 초기점은 실제 `occluded_R0_prediction`의 점을 OCC crop으로 변환한 값이었다. OCC bbox도 같은 가림 R0의 예측 bbox와 정확히 같았으며 crop matrix를 그 bbox에서 다시 계산해 일치시켰다. 원래 잘 맞던 hint나 GT bbox를 OCC 입력으로 대신 쓴 계약이 아니었다. 새 R0 추론으로 해당 저장 추론을 재현한 것은 아니며 기존 marker와 저장값/코드를 검증했다.

두 crop에서 target을 역변환하면 같은 원영상 target으로 돌아왔다. 5쌍의 공통 support는 각각8점, center 제외, 최대 성분 오차는1.18e−5px로 기존 허용1e−4px 이하였다. RGB/초기점은 finite였다. 공개 산출물에는 실제 좌표·bbox·K 배열을 복사하지 않았다.

## 현재 후보와 다른 점

| 항목 | 기존 저장 실험 | 현재 F가 요구하는 대조 |
| --- | --- | --- |
| TRAIN | DAY264 중253, 현재 Plastic217과 RGB SHA 교집합0 | 현재 적격 TRAIN에서 membership/support 통제 |
| target | 원본 Replay + self-occlusion PnP 보완 | 선언한 기존 허용 원본 감독; PnP 보완을 S1로 승격하지 않음 |
| teacher 초기값 | synthetic-only PoseFix | 같은 현재 frozen Replay에서 T_KEEP/T_OCC 모두 시작 |
| 입력 가림 | 기존 CAD8 75% recipe, frame당 고정1variant | 사전 고정 recipe와 실제 가림 R0의 q/box |
| 감독 crop | clean/OCC crop의 공통 support 교집합 | support 변경량을 공개하고 양 teacher·학생 비교에서 통제 |
| fit 범위 | teacher300step 2×2 source on/off | 최소 T_KEEP/T_OCC + 같은 학생 전달2fit |
| 주요 해석 | 과거 refiner 출력 품질; source on/off는 compute-matched 아님 | 학생 RGB+D9 T/R 개선; teacher-only 개선은 부족 |

[기존 결과](../pallet_occlusion_refiner_transfer_v2/RESULTS_KO.md)는 회복/보존 및 source forgetting의 절충을 보고했다. 이 저장물은 실제 `I_occ → R0(q_occ, box_occ) → refiner input` 경로의 **구현 가능성**을 재사용할 근거다. 다른 모집단·target 보완·초기값의 결과를 현재학생의 향상이나 실패로 복사할 수 없다. DAY264의 approximate-clean 구간 선정도 per-frame 사람 난도 인증이 아니다. 기존 teacher와 DAY recording이 겹친다는 이전 제한도 유지한다.

새 F를 온전히 실행하면 같은 teacher300step 두 적응과 student320step 두 전달이 최소4fit/1,240update이며 추가 seed까지 별도다. 이번 후속의 마지막 후보 cycle에서는 A의 낮은 실제 covered 노출을 조절하는 더 작은 학생 C 대조를 우선하도록 부모가 선택했다. 이는 F 최소 진단을 생략한 것이 아니며 F 방법을 실험적으로 반증한 것도 아니다. 최종 fit disposition과 잔여 예산은 중앙 보고서 및 candidate matrix에서 확정한다.
