# Pool·난도·감독 출처 감사

이 감사는 기존 manifest·주석 metadata·학습 목록을 읽은 결과다. 새 좌표/난도 주석, 이미지 수동 검수, fit, GPU 작업은 없다. 공개 문서에는 좌표·K·bbox 배열을 복사하지 않았다. 모델의 필터 통과와 사람이 확인한 정확도/가림 난도를 분리한다.

## 후보 → 통과 → 실제 학습

| 집합 | 후보 | 필터 통과 | 실제 사용 고유 실사 | 사람 난도 확인 가능 수 | 난도 상태 |
| --- | ---: | ---: | ---: | ---: | --- |
| Plastic | 1,000 | 249 | 217 | 0 | 각 단계 전체 unknown |
| Wood | 1,000 | 676 | 361 | 0 | 각 단계 전체 unknown |
| 기존 teacher TRAIN | 9장/38점 | 해당 없음 | Plastic3장/15점 + Wood6장/23점 | 0 | 9장 모두 저장 `occlusion_level=unknown` |

여기서 0은 Clean/Moderate/Severe가 0장이라는 뜻이 아니라 **그 구분으로 확정할 수 있는 수가 0**이라는 뜻이다. 현재 학생 풀을 clean-only로, teacher9/38을 clean-only로 부르지 않는다. 교사 TRAIN과 학생 풀의 난도 unknown은 평가128/45의 기존 난도 표를 재사용하여 채울 수 없다.

확인한 자료:

- [Plastic pool](../pallet_type_selftrain_v1/POOL.json)의 `records`에는 ID·RGB binding·recording·session·type·K만 있고 severity/difficulty/occlusion 필드가 없다. GREEN1,000은 별도이며 이번 Plastic1,000과 합치지 않았다. [pool 생성 코드](../../../scripts/research/pallet_type_selftrain_v1/pool.py)는 기존 metadata/recording 제외·샘플링만 하며 사람 난도를 만들지 않는다.
- [Wood pool](../pallet_material_selftrain_closure_v1/POOL.json)도 severity/difficulty/occlusion 필드가 없다. annotated/teacher319의 RGB SHA를 제외한 뒤 day/night 비례 및 frame-ID 간격으로 후보1,000을 만들었다. 후보 day675/night325, 통과 day496/night180은 **조명/수집 계층**이지 가림 난도가 아니다.
- 기존 private `PSEUDO_ACCEPTED.json`과 `PSEUDO_DECISIONS.json` 두 재질 모두 확인했다. Plastic 탈락은 confidence728, raw flip/LOO13, corrected all8 LOO10; Wood는 각각266/30/28이다. 통과 항목에도 사람 난도 필드는 없다. confidence/flip/LOO 통과를 `verified`나 `CLEAN`으로 변환하지 않는다.
- manifest에 필드가 없다는 이유만으로 조사를 끝내지 않았다. 기존 `data/evaluation/pallet_eval_v1/manifests/frames.csv`를 RGB SHA로 join하면 Plastic 후보/통과/사용에서 각각68/22/17장이 연결되지만 `occlusion`은 모두 `unknown`이다. Wood는 세 단계 모두 연결0이다. 기존 `pallet_easy_rgb_review_v1/SCOUT.json`과 후보의 교집합은 Plastic12장이나 모두 `RGB_NOT_YET_REVIEWED`; EXPANDED와의 교집합 및 Wood 교집합은0이다. 따라서 이 기존 검토 기록에서도 확정 난도 수는 늘지 않는다.
- [Plastic pose-only protocol](../pallet_type_selftrain_v1/selftrain_recovery_v1/pose_only/PROTOCOL.json)과 [Wood TRAIN protocol](../pallet_material_selftrain_closure_v1/WOOD_TRAIN_PROTOCOL.json)의 실제 사용 수는 각각217/361이다. 두 재질 모두 고정 source512 + real512 replacement slots, 5epoch/320update이며 후보 전체를 균등하게 한 번씩 본다는 뜻이 아니다.
- [teacher SPLIT](../pallet_large_error_refiner_v1/SPLIT.json)에 연결된 기존9개 annotation의 `objects[].occlusion_level`을 직접 읽었다. 모두 `unknown`이며, `visibility`/manual mask를 사람이 붙인 자연 가림 등급으로 대체하지 않았다. 새 자동 난도 추정도 수행하지 않았다.

기존 annotation/metadata로 이 세 단계의 난도 분포를 복원할 근거를 찾지 못했다. 이는 known-clean subset을 새로 만들기 위한 사용자 주석 요청 사유가 아니며 현재 pooled A 대조를 막는 gate도 아니다. 실험명은 `FILTERED_INPUT_TO_OCCLUDED`다.

## 감독 계층과 허용 범위

| 계층 | 현재 확인된 정보 | 허용하는 해석·주의 |
| --- | --- | --- |
| S1 | 이미 teacher가 본 TRAIN9장/직접 클릭38점, Plastic15 + Wood23 | 기존 stored-index 클릭 감독이다. teacher in-sample이며 독립 calibration이 아니다. 이전 감사의 `camera_dynamic_0123`/`UNCONFIRMED_SIGNED_AXIS` 제한을 유지한다. 물리적 signed-axis GT pose로 승격하지 않는다. |
| S2 | 기존 source512의 원래 synthetic RGB/label을 그대로 재사용 | [기존 verification](../pallet_type_selftrain_v1/DATA_VERIFICATION.json)은 `source_labels_original=true`; Wood 준비는 Plastic source512의 RGB·label SHA까지 일치시킨다. 이 감사는 renderer/camera/번호 계약을 새로 재렌더링 검증하지 않았다. 기존 합성 감독의 재사용이지 새 물리정확성 인증 또는 실사 calibration이 아니다. |
| P | 원래 RGB에 대한 고정 RAW/REF 의사좌표와 기존 공통 stored support | 필터 통과는 확인됐지만 실제 정확도는 미검증이다. RAW도 teacher 기반 accepted membership/common support를 공유하고, A에서는 추가로 REF-conditioned mask 계획을 공유한다. |
| U | 현재 허용 감독 밖인 v1 true-ignore 점·미확인 번호·PnP 보완 등을 통한 비허용 좌표 | 가림을 더했다고 U→S1/S2/P로 승격하지 않는다. v1은 위치/RLE/kobj에서 제외한다. v0는 기존 존재항 음성이며 가림 직사각형 안에 있다고 v0/v1로 다시 쓰지 않는다. |

A는 원래 label·visibility tensor를 덮어쓰지 않는다. 따라서 자연히 보이는 점을 가린 뒤에도 **원래 허용 감독만** 유지한다. center는 가림 위치 선정의 covered/remaining 계산에서만 제외되며 기존 center 손실의 감독 규칙은 바꾸지 않는다. S1/S2/P를 합쳐 모두 실사 GT라고 하지 않는다.

## 재사용 데이터의 독립성 제한

[Wood provenance audit](../pallet_material_selftrain_closure_v1/WOOD_PROVENANCE_AUDIT.json)은 teacher와 학생 후보가 REC_001/REC_002를 공유함을 명시한다. 추가 클릭 정보는 아니지만 교사와 학생이 독립 수집 집합이라는 뜻도 아니다. 현재 Wood main45와 teacher/후보의 ID·SHA·recording 겹침은 없다는 기존 전수감사를 재사용한다. 과거116에는 teacher 정확 중복5장이 있어 그116 결과를 현재45와 혼합하지 않는다. Plastic 현재128은 teacher recording을 제외한 기존 main이며, 본 후속은 [INPUT_BINDINGS](INPUT_BINDINGS.json)로 그 기존 자산을 고정한다.

현재128/45 및 verified66은 반복 DEV다. 평가 reference를 fit·sampler·loss weight·타깃 보정 규칙에 사용하지 않는다. 과거 Clean19/87을 가져오면 teacher 감독 예산 및 recording 조건이 달라지므로 자동 재사용하지 않았다. 신규 annotation0과 기존 수동정보 사용범위 확대0은 구분하며, 이번 A의 추가 수동정보 사용은 없다.

## A에서 실제 관측된 가림 노출

[독립 A 감사](AUDIT_A_INPUT_OCCLUSION.md)는 Plastic seed42 전수320batch를 읽었다. real2,560 occurrence 중 scheduled1,302, 실제 적용542(21.17%), 32회 내 위치 미발견760이다. REF 위치 기준 계획은 RAW/REF 동일하지만 실제 가려진 감독점은 RAW781/REF824다. 원래 affine의 좌표별 경계 처리 때문에 실제 support는3batch에서 달랐다(v2 합계 RAW21,819/REF21,823). 이 차이를 사후 제거하거나 가림에 의해 신규 생성된 감독으로 취급하지 않았다. 따라서 공통 RGB/계획 대조이며 masked-target 노출량까지 동일한 대조라는 주장은 하지 않는다.
