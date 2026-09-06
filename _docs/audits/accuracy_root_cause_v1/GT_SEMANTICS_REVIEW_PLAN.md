# GT semantics 사람 리뷰 계획

작성 2026-09-06 · HEAD `2e5ec0e`

```
STATUS = HUMAN_REVIEW_REQUIRED
```

CLI 가 사람 어노테이션을 대신했다고 주장하지 않는다. 여기까지가 자동 산출물이고,
사람이 채운 뒤에야 후속 분석이 가능하다.

---

## 이 계획은 지시문 §3 에서 **범위가 좁혀졌다**

§3 은 대표 30장 + 실패 농축 30장으로 GT semantics 전반을 사람에게 묻자고 했다.
그런데 `FAILURE_DECOMPOSITION.md` 가 그 사이에 다음을 확정했다 [확인]:

- 2D keypoint 오차는 GT 출처로 설명되지 않는다 (외삽 7.38 px vs 클릭 6.38 px,
  가림 코너 안에서는 클릭이 **더** 나쁘다).
- 프레임 최악 오차의 75% 가 **보이는** 코너에서 난다.
- 계약 대칭 {0,180}이 구제하는 프레임은 0장이고, 8! 자유 배정도 4% 만 개선한다.

즉 "GT 가 2D 정확도의 floor 를 만든다" 는 가설은 이미 기각됐다. 60장 전면 재검수는
답이 정해진 질문에 사람 시간을 쓰는 것이 된다. 그래서 **실제로 미해결인 두 질문**으로
좁혔다. 좁힌 근거를 남기는 것이 이 절의 목적이다.

## 남은 두 질문

**(가) W/D 축 배정(yaw 90도 분기)이 사람 관측으로 갈리는가**

[확인] `dimensions_m` 은 프레임마다 `{w1.1,d1.3}` 81장 / `{w1.3,d1.1}` 59장 두 값을 갖는다.
클릭 점만으로 두 가설을 비교하면 margin p50 1.48 px 이고 **130장 중 51장이 1 px 미만 차이**,
5장은 반대 가설이 더 잘 맞는다. 외삽 점을 넣으면 9.14 px 로 부풀지만 그 점은 채택 pose 의
투영이라 순환이다(`annotate_io.py:518-521`).
클릭이 6개 이상인 44장에서는 스왑 우세 0 → GT 가 틀린 게 아니라 **증거량 부족**이다.

**(나) `occluded` 로 표시된 코너의 좌표가 무엇을 뜻하는가**

[확인] `gt_v2` 는 코너별 `visibility`/`in_frame`/`reason` 을 갖지만 `source` 는 전 파일
`"unknown"` 이라 **클릭인지 PnP 투영인지 이관되지 않았다**. 그리고 가림 코너 520개 중
44개는 사람이 직접 찍었는데(정수 좌표) 그쪽 오차가 더 크다(18.91 vs 14.73 px).
사람이 "안 보이는 코너를 추측해서 찍은 것" 이라면 그건 모델을 채점할 기준이 아니다.

## 리뷰 대상 — 증거로 뽑았다 (임의 표집 아님)

`data/pallet/results/accuracy_root_cause_v1/GT_REVIEW_FRAME_LIST.csv` (42행) [확인]

```
AXIS_AMBIGUOUS  30장   클릭만으로 W/D margin |m| < 1 px 인 프레임 (margin 작은 순)
FEW_CLICKS      12장   클릭 코너가 4개 이하라 증거량 자체가 부족한 프레임
폴더 분포        eval_outside 15 · eval_pallet09 11 · eval_pallet07 7 · eval_noapril 6
                eval_cad 2 · eval_night08 1
```

## 리뷰 패킷 (미생성 — 사람이 착수할 때 만든다)

`_docs/audits/accuracy_root_cause_v1/human_review/` 아래 프레임마다
`01_raw/` `02_gt_only/` `03_geometry_only/` `04_review_form/`.

★ **anchoring 방지가 이 설계의 핵심이다.** 리뷰어에게 현재 모델 예측·multi-teacher 예측·
Hough 예측·기존 실패 라벨을 보이지 않는다. `02_gt_only` 는 저장된 keypoint 만,
`03_geometry_only` 는 두 W/D 가설의 cuboid 를 **어느 쪽이 채택본인지 표시하지 않고**
좌우 무작위로 배치해 제시한다.

## 리뷰 시트 스키마 (새 CSV — 기존 annotation schema 를 바꾸지 않는다)

코너 단위:
`frame_id, corner_id, directly_visible, occluded_but_geometrically_inferable, outside_image,
physical_surface_corner, virtual_cuboid_corner, ambiguous, reviewer_xy_u, reviewer_xy_v,
semantic_role_confident, note`

프레임 단위(가 질문 전용):
`frame_id, hypothesis_A_better, hypothesis_B_better, cannot_tell, confidence_1to5, note`

## repeatability (§4)

42장 중 **20장을 2회** 독립 리뷰한다(같은 리뷰어, 최소 3일 간격, 순서 무작위).
이 숫자는 통계적 최적값이 아니라 **작업량 제한용 [추정][미검증] pilot 설정**이다.

측정: visible point 반복 Euclidean 거리 · semantic role 불일치율 ·
physical/virtual 불일치율 · visible/occluded 불일치율.

★ 측정된 GT 산포를 모델 오차에서 **빼지 않는다.** 별도 표로 나란히 낸다.
method 간 차이가 GT repeatability 이하이면 `SMALL_GAIN_BELOW_GT_RESOLUTION` 으로 표기한다.

## 이 리뷰가 바꿀 수 있는 것 / 없는 것

바꿀 수 있는 것: 6D rotation 의 90도 분기 신뢰도, 가림 코너를 감독에서 뺄지 여부.
바꿀 수 없는 것: 2D keypoint 정확도 판정. 위에서 이미 기각됐다.
→ 그래서 이 리뷰는 **다음 실험의 선행 조건이 아니다.** 6D 를 논문에 쓰려 할 때의 선행 조건이다.
