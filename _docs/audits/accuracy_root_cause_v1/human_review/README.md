# GT semantics 사람 리뷰 — 리뷰어 안내 (1페이지)

생성 2026-09-06 · 계획 문서: `../GT_SEMANTICS_REVIEW_PLAN.md`

## 무엇을 판정하나 — 두 가지뿐이다

**(가) W/D 축 배정** — 팔레트의 두 변 중 어느 쪽이 폭(width)이고 어느 쪽이 깊이(depth)인가.
저장된 GT 는 둘 중 하나를 골라 놓았는데, 클릭 점만으로는 두 가설의 재투영 차이가
1 px 미만이라 **점만 봐서는 갈리지 않는다.** 갈리는 것은 사진 자체다.
→ `REVIEW_FORM_frame.csv`

**(나) 코너 좌표의 의미** — `02_gt_only.png` 에 찍힌 각 코너가 사진에서 실제로 보이는
물리적 모서리인가, 아니면 보이지 않는데 추측/계산으로 놓인 점인가.
→ `REVIEW_FORM_corner.csv`

그 외(2D keypoint 정확도가 GT 탓인지 등)는 이 리뷰의 질문이 아니다. 이미 별도로 기각됐다.

## 폴더마다 있는 4개 파일

```
01_raw.png          원본 사진. 아무것도 그려져 있지 않다. 먼저 이것부터 본다.
02_gt_only.png      저장된 keypoint 9개 (0~7 = 코너, 8 = 중심).
                      ● 채운 원   = 사람이 마우스로 찍은 점 (정수 좌표)
                      □ 빈 사각형 = 찍지 않고 계산으로 채운 점 (실수 좌표)
                    cuboid 선은 일부러 그리지 않았다 — 그리면 (가)의 답을 암시한다.
                    번호가 이미지 가장자리에 붙어 있고 그 자리에 마커가 없으면,
                    그 코너는 화면 밖에 있다는 뜻이다.
03_geometry_A.png   W/D 가설 하나의 cuboid 를 투영한 것
03_geometry_B.png   나머지 가설
```

`03_*` 의 cuboid pose 는 **사람이 찍은 점만으로** 다시 계산했다. 계산으로 채운 점을 쓰면
저장된 가설이 자기 자신을 채점하게 되어 항상 이긴다.

## 일부러 보여주지 않는 것

- **모델 예측을 어디에도 그리지 않았다.** 현재 모델·multi-teacher·Hough 예측, 기존 실패
  라벨, 재투영 오차 수치 전부 뺐다. 숫자가 보이면 사람이 사진 대신 그 숫자를 보고 고른다.
- **A/B 중 어느 쪽이 저장된 가설인지 표시하지 않았다.** A/B 배정은 프레임마다 무작위이고,
  정답 매핑은 이 폴더 밖에 따로 있다.
- 파일 이름·이미지 안 글자 어디에도 채택 여부를 알 수 있는 단서가 없다.

이 설계의 목적 하나가 anchoring 방지다. 아는 정보로 보정하려 하지 말고, 사진만 보고 답하라.

## 채우는 법

1. `01_raw.png` 를 먼저 충분히 본다.
2. `03_geometry_A.png` / `03_geometry_B.png` 를 번갈아 보고
   `REVIEW_FORM_frame.csv` 의 해당 `frame_id` 행을 채운다.
   `hypothesis_A_better` / `hypothesis_B_better` / `cannot_tell` 중 **하나만** `1`,
   `confidence_1to5` 는 1(전혀 확신 없음) ~ 5(확실).
3. `02_gt_only.png` 를 보고 `REVIEW_FORM_corner.csv` 를 코너 0~8 별로 채운다.
   - `directly_visible` / `occluded_but_geometrically_inferable` / `outside_image`:
     그 코너가 사진에서 어떤 상태인가 (하나만 `1`).
   - `physical_surface_corner` / `virtual_cuboid_corner` / `ambiguous`:
     찍힌 좌표가 팔레트의 실제 모서리인가, 직육면체를 가정해야만 생기는 가상의 점인가
     (하나만 `1`).
   - `reviewer_xy_u`, `reviewer_xy_v`: 본인이 다시 찍는다면 어디인가 (픽셀 좌표, 선택).
     저장값과 다르다고 판단할 때만 적으면 된다.
   - `semantic_role_confident`: 그 번호(예: 4 = far-top-left)가 이 코너에 맞다고
     확신하면 `1`.
4. 판단이 안 서면 **`cannot_tell` / `ambiguous` 를 고르는 것이 정답을 찍는 것보다 낫다.**
   억지 판정 한 건이 리뷰 전체의 신뢰도를 깎는다. 모르겠다는 답도 데이터다.

## phase 2 로 덧붙인 16장 (실패농축 표본)

처음 42장은 W/D 축 배정이 애매한 프레임을 골라 담은 것이다. 뒤이어 16장을 더했는데,
이쪽은 **모델이 실제로 크게 틀린 프레임만 모은 실패농축 표본**이다. 무작위 표본이 아니므로
여기서 나온 비율을 전체 평가셋 비율로 읽으면 안 된다. 판정 방법과 양식은 앞의 42장과
완전히 같고, `REVIEW_FORM_*.csv` 아래쪽에 이어서 행이 들어 있다.

각 프레임이 왜 뽑혔는지는 `HUMAN_REVIEW_MANIFEST_PHASE2.json` 의 `role` 에 적혀 있다
(리뷰 중에는 열지 않는 편이 낫다 — anchoring 이 걸린다).

```
GROSS_KEYPOINT_ERROR       예측 keypoint 가 아주 크게 어긋난 프레임 (수백 px 단위)
AXIS_PERMUTATION_SUSPECT   코너 위치는 맞는데 번호만 90도 돌아간 것으로 보이는 프레임
CORRECT_BOX_BAD_KEYPOINT   검출 박스는 물체에 잘 맞았는데 keypoint 만 틀린 프레임
LINE_BETTER_THAN_POINT     두 방식 중 line 쪽 회전 오차가 더 작았던 프레임
LINE_WORSE_THAN_POINT      두 방식 중 line 쪽 회전 오차가 더 컸던 프레임
```

`LINE_*` 두 role 은 따로 읽어야 한다. 이 프레임들에서 **두 방식의 corner 예측은 완전히
같다** — 겹치는 93개 프레임 전부에서 코너 좌표가 동일했다(실측). 달라진 것은 회전 오차뿐이다.
그래서 "line 이 코너를 더 잘 찍었다/못 찍었다" 로 읽으면 틀린다. 리뷰에서 물어보는 것도
여전히 GT 두 가지(축 배정 · 코너 좌표의 의미)뿐이고, 두 방식 중 무엇이 나은지는 묻지 않는다.

`03_*` 가 없는 프레임이 4개 있다. 사람이 찍은 점이 3개뿐이라 cuboid pose 를 풀 수 없었다
(4개 이상 필요). 그 프레임은 `01_raw` 와 `02_gt_only` 만 보고 코너 문항만 채운다.
프레임 문항은 `cannot_tell` 로 둔다.

## `_rep2` 행은 무엇인가

처음 42장 중 20장은 **두 번** 본다(같은 사람, 최소 3일 간격). 두 번째 리뷰 결과를 `_rep2`
접미가 붙은 행에 적는다. 목적은 GT 자체의 재현 산포를 재는 것이다.
두 번째로 볼 때 첫 번째 답을 **보지 않는다.**

## 리뷰가 끝난 뒤

정답 매핑(어느 쪽이 저장된 가설인지)과 클릭 전용 재투영 오차는
`data/pallet/results/accuracy_root_cause_v1/_ANSWER_KEY.csv` 에 있다.
리뷰 시트를 다 채우기 전에는 열지 않는다.

생성 내역·프레임별 클릭 수·무작위 seed 는
`data/pallet/results/accuracy_root_cause_v1/HUMAN_REVIEW_MANIFEST.json` (처음 42장) 과
`.../HUMAN_REVIEW_MANIFEST_PHASE2.json` (덧붙인 16장).

기존 GT JSON 은 이 리뷰로 수정되지 않는다. 리뷰 결과는 위 CSV 두 개에만 쌓인다.
