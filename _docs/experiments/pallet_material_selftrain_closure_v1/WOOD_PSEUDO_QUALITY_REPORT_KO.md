# Wood pseudo-label 품질의 참조 적격성 감사

**Q1_WOOD = UNRESOLVED.** 교사의 수동 감독 recording과 분리된 Wood 평가 45장에는 직접 클릭 출처가 확인되는 가시점이 0개다. 따라서 raw와 corrected의 검증된 가시점 PCK5/10/20·중앙값·P90·20px 초과·coverage·paired 진입/이탈은 모두 **미평가(`null`)**다. 0% 또는 보정 실패라는 뜻이 아니다. 이후 legacy reference 기반 학생 평가와 이 결론을 혼동하지 않는다.

이 감사는 새 추론·점수 계산 없이 출처·ID·SHA·point provenance만 확인했다. GT 좌표는 이 보고서나 동반 JSON에 싣지 않는다. 기계 판독 값과 입력 해시는 [WOOD_PSEUDO_QUALITY.json](WOOD_PSEUDO_QUALITY.json)에 있다.

## 고정 teacher와 recording 중복

Plastic main과 같은 frozen `pallet_posefix_replay_v1/last300.pt`를 유지한다. checkpoint SHA256은 `fc9b3d7b7f2e7c38b608a12461cf16a58b76669f12ef9be5c8dc35d81104a41d`다. 수동 감독 예산 9장/38코너는 Wood 주간 6장/23코너와 Plastic 야간 3장/15코너로 구성된다. 결과를 보고 종류별 teacher로 바꾸지 않는다.

| Teacher 감독 | Recording | 같은 recording의 Wood 평가 |
|---|---|---:|
| wood_day_01 6장/23코너 | REC_001 | wood_day_01 20장 |
| plastic_night_01 3장/15코너 | REC_002 | wood_night_01 51장 |

`plastic_night_01`과 `wood_night_01`은 재료·session 이름은 다르지만 동일 원천 `real_unlabeled_night_20260830`의 **REC_002**다. session 문자열만 비교해 wood_day_01만 제외하면 96장이 남지만, 이는 teacher-recording-disjoint 평가가 아니다. 출처에 따라 REC_001과 REC_002의 71장을 모두 제외한다. 모델 오차나 성능은 제외 기준에 사용하지 않았다.

Historical Wood116과 teacher 학습 영상의 exact ID/SHA 중복은 각각 5장이다: `wood_day_01:002141`, `004032`, `004481`, `005286`, `026085`. 교사 이미지 9장 및 Wood116의 실제 RGB 파일 SHA 125건을 재계산해 기존 바인딩과 일치함을 확인했다.

## Teacher-recording-disjoint 참조 45장

| Session | Recording | 이미지 | Clean | Moderate | Severe |
|---|---|---:|---:|---:|---|
| wood_183705 | REC_039 | 25 | 18 | 7 | 없음 |
| wood_184309 | REC_042 | 20 | 20 | 0 | 없음 |
| 합계 | 2개 | 45 | 38 | 7 | 평가 자료 없음 |

이 집합의 teacher exact ID/SHA/recording 중복은 모두 0이다. Student pseudo-train과의 중복 및 최종 학습 집합 적격성은 별도의 `WOOD_PROVENANCE_AUDIT.json`에서 확인한다. 이 문서는 teacher-reference 출처 감사를 담당한다.

45장×8코너의 **360개 point annotation은 전부 `source=unknown`**이다. `objects[0].gt_source=manual`, 정수 좌표, `visible` 표시, `manual_gt`라는 파일 이름만으로 직접 클릭 여부를 보증할 수 없다. 좌표를 다시 해석하거나 자동 수정하지 않는다.

직접 가시점 적격 규칙은 0..7번 코너 중 `source=manual_click`, `visibility=2`, `reason=visible`, `in_frame=true`, native 영상 내부의 유한 좌표이며, teacher 수동 감독 recording과도 분리되어야 한다. 중심 8번은 제외한다. 이 규칙을 충족하는 점은 0개다. 기존 FINAL_V2 verified 참조 18장에도 Wood 영상은 없다.

## 나머지 수동점이 주 Q1을 대신하지 못하는 이유

Historical Wood116의 928개 코너 출처는 unknown 360, manual_click 292, pnp_projected 268, extrapolated 8이다. 유효·영상 내부·직접 가시 클릭으로 남는 것은 **291점**(wood_day_01 87점, wood_night_01 204점)이다. 이 291점은 모두 teacher가 수동 감독을 받은 REC_001/002에 속한다. 모두 교사가 직접 학습한 동일 점이라는 뜻은 아니지만, recording 분리는 성립하지 않는다.

291점에 나중에 점수를 계산하더라도 teacher-recording overlap을 명시한 보조 자료일 뿐, 주 45장의 검증된 pseudo-quality 또는 독립 확인으로 대체할 수 없다. 이번 문서는 그 점수를 계산하거나 legacy/mixed reference를 검증된 정답으로 승격하지 않는다. 116개 annotation 파일의 실제 SHA도 frozen split과 모두 일치했다.

## 논문에 반영할 범위

- 쓸 수 있는 표현: “교사 감독 recording과 분리된 Wood 평가 자료에는 직접 클릭 출처가 보증된 가시점이 없어, Wood pseudo-label 품질의 엄격한 Q1 검증은 미해결이다. 학생 비교는 기존 legacy reference에 한정해 별도로 평가한다.”
- 쓰면 안 되는 표현: “Wood에서도 검증된 가시점 pseudo-label 품질이 개선됐다”, “Wood teacher 정확성이 확인됐다”, “독립적인 Wood 확인을 완료했다.”
- Q1 미해결은 공정한 Wood RAW/REF 학생 비교의 결과를 미리 정하지 않는다. 추가 수동 레이블·teacher 변경·새 fit을 이 공백의 자동 해결책으로 요청하지 않는다.

## 근거와 재현

- Teacher checkpoint: [기존 Replay FIT](../pallet_posefix_replay_v1/FIT.json)의 `checkpoint`.
- Teacher ID: [Replay INPUT_LOCK](../pallet_posefix_replay_v1/INPUT_LOCK.json)의 `real_ids`.
- Teacher image SHA·수동 코너 수: [원래 SPLIT](../pallet_large_error_refiner_v1/SPLIT.json)의 `train`과 [TRAIN_SUPPORT](../pallet_posefix_large_error_v1/TRAIN_SUPPORT.json).
- Wood116 ID·annotation 경로·기대 SHA·severity: [Wood SPLIT](../pallet_replay_by_type_v1/wood/SPLIT.json)의 `evaluation`.
- Recording alias: 로컬 `data/pallet/results/site_environment_audit_v1/SOURCE_RECORDING_GROUPS.json`의 `groups`.
- 검수 참조의 Wood 포함 여부: 로컬 `data/pallet/results/pallet_verified_anchor_v1/metadata_conflict_qa/VERIFIED_LABELS_FINAL_PRIVATE.json`의 `frames`.

동반 JSON의 `source_bindings`로 파일 SHA를 검증한 뒤, Wood split의 annotation들을 읽어 코너 0..7의 출처를 위 규칙대로 집계하면 같은 결과를 얻는다. annotation 116건의 실제 `{id,path,sha256}` 목록을 id로 정렬한 canonical JSON SHA256은 `1c5ffc83332c15cff0cca5d47e3b3effcc3c82c45cca4477cb69ed02e86ef136`이다. private 좌표나 원본 RGB 파일은 이 문서에 포함하지 않는다.
