# V33: 가장 큰 연결 영역을 사용한 전경 포함 비용 대조 — 공개 초안

이 문서는 소스·봉인·공개 범위를 먼저 정리한 초안이다. **V33 전체 238장 예측·독립 수치 감사·고정 참조 점수·점수 재집계의 최종 결과는 아직 이 초안에 넣지 않았다.** 현재 기본값은 기존 e이며, 이 초안은 새 방법의 채택이나 192장 목표 달성을 선언하지 않는다.

이번 대조는 기존 mask의 전경을 모두 합친 관측 대신, 면적이 가장 큰 8-연결 영역 하나를 고른다. 동률이면 OpenCV의 가장 낮은 양의 label을 쓰고, 구멍 채우기·팽창·침식·임계값 탐색은 하지 않는다. 이후의 전경 외곽 관측기는 기존 V13, 전경 포함 비용(C)의 계산과 계수는 기존 C25를 그대로 사용한다. 선택 규칙은 CAL·TEST·추가143에 동일하다.

원 native 점/선·CAL32 분산과 접선 계수·fresh BI PnP·시작점 선택·O/S 비용·hardguard는 바꾸지 않는다. 실제 입력에서 바뀌는 항목은 `regularizers.outline_observation` 하나다. e는 수치 worker에 들어가지 않는 비교 예측이고, worker의 네 결과가 나온 뒤 부모 프로세스가 별도 봉인한 e 행을 합친다.

## 무엇을 확인하고 무엇을 증명하지 않는가

관측 감사는 연결 영역의 면적·동률 선택·bbox·픽셀 해시와 기존 외곽 표본 생성을 다시 계산한다. 수치 감사는 선택한 자세의 native+O+S+C 비용과 제약, 입력에서 C 관측 하나만 바뀌었는지, 원 native 두 방법과 e의 전체 출력이 같은지를 확인한다. **이 검사가 통과해도 큰 연결 영역이 실제 팔레트라는 뜻은 아니며, 물리적 자세 정확도나 전역 최적을 증명하지 않는다.**

분할 모델의 quality 점수는 물체 정체의 정답 확률이 아니다. 외곽 관측은 전경의 convex hull에서 얻는 표본이며 검출된 실제 물리 모서리와 구분한다. 가장 큰 영역을 고르면 잘못 붙은 작은 조각을 뺄 수 있지만, 끊겨 보이는 실제 팔레트 조각도 버릴 수 있다. target mask 밖이라는 사실만으로 다른 물체의 가림을 확정하지 않는다. 별도 generic instance 후보는 확정 occluder mask로 연결하지 않았다.

S는 영상 점·국소 질의·2D 비용 없이 모델/PnP 투영과 기존 coarse 유도 bounded S 항만 유지한다. U는 대략적인 범위만 사용한다. CAL46은 설계·검증용이며 원 compatible44와 구분하고, TEST49+추가143=주 평가192장에 합치지 않는다. severe는 이 정량 집합에 0장이다. 빈 mask·unsupported·fallback·자세 없음도 원 분모에 남겨야 한다.

## 같은 192장의 기존 비교값

다음은 V33 결과가 아니라 이미 봉인된 과거 예측의 점수다. [동일 ID·참조와 metric 기록 재집계](HISTORICAL_BASELINE_SCORE_RECOUNT_V1.json)에서 현재 V32와 TEST49·추가143의 순서가 모두 같음을 확인했다. N3는 raw YOLO 단독이 아닌 저장된 N3+SUBPIX 초기 자세이고, PoseFix RAW는 statewise 보정을 적용하지 않은 저장 예측이다.

| 고정 과거 방법 | 위치≤5cm /192 | 위치≤5cm·회전≤5° /192 | 평균 위치 cm | 최대 위치 cm |
|---|---:|---:|---:|---:|
| N3 + SUBPIX | 112 | 108 | 11.3058 | 174.7580 |
| PoseFix RAW | 85 | 80 | 13.8357 | 185.7835 |

과거 공동 기준은 `<5cm 및 <5°`였지만 두 baseline에는 정확히 5인 사례가 없어 현재의 `≤`로 센 수와 같다. 두 방법 모두 192개 metric은 유한했으나, 과거 accepted/fallback flag가 없어 fallback 0이라고 해석하지 않는다. TEST는 반복 사용한 탐색 평가이며 추가143 참조는 annotation/PnP 유도값이다. signed axis와 intrinsics의 불확실성도 남는다.

V33 전체192 결과 표에는 원 V32 union 대조와 V33 LCC 대조를 같은 정책·방법끼리 나란히 표시한다. 위치·회전·공동 성공 수, 평균·최대, accepted/fallback와 원 TEST49의 퇴행을 모두 보존한다. **이 표의 V33 수치는 ROOT의 전체 점수·재집계 인계 뒤 별도 최종 문서에서 채운다.** 원 native 두 방법과 e는 238장 전체 출력의 exact 불변 검사를 요구한다.

평균은 성공한 장만의 평균과 다르다. 이미 봉인한 **V32 SOURCE+C**에서 117장은 위치·회전 동시 통과 수이고, 위치만≤5cm인 수는 120장이다. 그 120장의 평균 위치 오차는 1.89018cm, 나머지 72장은 37.11509cm여서 전체192장의 평균은 15.09952cm가 된다. `120×1.89018 + 72×37.11509`를 192로 나눈 값이다. 많은 장이 잘 맞아도 일부 큰 오차 때문에 전체 평균이 커질 수 있다.

V32의 최대 위치오차 사례 `wood_night_01:031426`은 191.3134cm였다. 이 값은 영상 위 선 간격이 아니라 3D 모델 중심의 거리 오차다. ROOT가 보관 자료에서 확인한 깊이 차이만 약1.849m이므로, 2D 외곽이 일부 겹친다고 3D 깊이까지 맞는 것은 아니다. 이 설명은 기존 V32의 닫힌 결과이며 아직 V33에서 개선됐다는 뜻이 아니다.

## 보관할 실제 과정과 사후 진단

[V32의 저장 C 표본·최종 hull 진단](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/clean_fresh_BI_containment_pixel_diagnosis_v4_32_r2/REPORT_KO.md)과 [수정·출처 보충](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/clean_fresh_BI_containment_pixel_diagnosis_v4_32_r2/PROVENANCE_ADDENDUM_KO_V2.md)은 V9에 포함되지 않았던 닫힌 사후 진단이다. 새 영상 탐색·PnP·중간 optimizer 자세를 만들어낸 자료가 아니다. 초기 schema 실패와 정정된 출처도 구분해 보존한다.

V33 과정 그림은 ACTUAL의 별도 소스·계획·봉인과 함께, 전체 점수 뒤에 보관된 동일 사례를 설명한다. 현재 그 실제 그림 결과는 이 초안에 포함하지 않았다. 선 관측과 전경 hull 표본을 구분하고, 서로 다른 방법의 최종 자세를 optimizer의 연속 중간 과정처럼 표현하지 않는다.

## 공개 범위와 재현 경로

- [중앙 선언](../nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1/largest_component_global_outline_method_plan_v4_33/PREDECLARATION.json): 원 238행·정책·기존 수학과 바뀐 C 관측 규칙.
- [LCC8 소스](../../../../scripts/research/pallet_visibility_curriculum_20261013_v1/largest_component_global_outline_v4_33.py), 관측기·controller·독립 감사·scorer·재집계의 소스와 테스트, 설계·owner·peer·ROOT 기록.
- 닫힌 CAL46·REST192 관측의 작은 protocol·index·seal, 독립 픽셀 감사 및 전체 선택 자세 수치 감사·점수·재집계.
- 전체 238×5개의 저장 최종 자세를 담은 compact archive와 실제 과정 이미지: 원 `actual_pose`, top-level WD, accepted/가용 여부, source flag, 저장된 선택 비용 요약만 복사한다. seed/start/전체 costplan/trajectory의 대체물은 아니다.

원 INPUTS/E_COMPARISON/PREDICTIONS/OBSERVATIONS 저널, mask NPZ, per-row spool과 중복 전체 code snapshot은 이 main 공개 묶음에서 제외한다. 공개 Git HEAD에 이미 같은 바이트가 있는 의존 소스는 재사용 기록으로 남기고, 빠진 현재 의존 코드는 추가한다. 달라진 과거 소스는 SHA별 한 개만 historical로 구분해 보존한다. 보호 대상 사용자 파일 6개는 바꾸지 않는다.

[최초 원시 증거 Release](https://github.com/CanelE452/pallet-6d-pose/releases/tag/visibility-curriculum-snapshot-20261015-v1)가 이후 생성한 V32·V33 raw archive를 포함한다고 주장하지 않는다. V10의 최종 파일 목록·해시·로컬 링크·봉인 검증은 모든 결과가 닫힌 뒤 별도 manifest에 기록하고, 복사·commit·push는 ROOT가 수행한다.
