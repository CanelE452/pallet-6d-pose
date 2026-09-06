# 오검출 육안 검수 — R0 는 무엇을 팔레트로 보는가

작성 2026-09-06 · 새 추론 0 회 · 모집단 `DEV_NEG2689` (real negative, 팔레트 없음)
시트 `_docs/audits/accuracy_root_cause_v1/false_positive_review/sheet_01..04.png` (79셀 전수 육안)
라벨 `data/pallet/results/accuracy_root_cause_v1/R0_FALSE_POSITIVES_LABELLED.csv`

지시문 §19 는 **카테고리를 미리 정하지 말고 이미지를 먼저 보라**고 했다. 그대로 했다 —
아래 이름은 79셀을 다 본 **뒤에** 붙였다. "traffic cone / rack / forklift / box / floor pattern"
같은 사전 후보는 하나도 쓰지 않았고, 실제로 그 중 어느 것도 나오지 않았다.

---

## 무엇에 붙었나 (배포 임계 conf 0.4 이상, 79건 전수)

```
category                          n    비중    conf p50
SLATTED_WOODEN_BENCH             49   62.0%    0.665     나무 슬랫 벤치
RIBBED_TRANSLUCENT_LID           23   29.1%    0.720     투명 플라스틱 컵뚜껑(동심 리브)
LOUVRE_VENT_GRILLE                3    3.8%    0.596     흰 가전의 환기 슬롯 배열
WALL_SIGN_PLAQUE                  1    1.3%    0.760     벽면 표지판
BRIGHT_PLANAR_SCREEN              1    1.3%    0.568     밝은 화면
KEYBOARD_KEY_ROWS                 1    1.3%    0.434     키보드 키 배열
DIFFUSE_GROUND_NO_STRUCTURE       1    1.3%    0.430     구조 없는 바닥/풀
```

최고 신뢰도 오검출은 **컵뚜껑 conf 0.905** 다. 배포 임계(0.4)의 두 배가 넘는다.

## 한 문장으로

**91%(72/79)가 "평면 위에 나란한 요소가 반복되는 것"이다.**
벤치 슬랫, 컵뚜껑 리브, 환기 그릴 슬롯, 키보드 키 열 — 재질도 크기도 실내외도 다른데
공통점은 하나다. 그리고 그것이 팔레트 상판의 시각적 서명이다.

★ 이 목록에 **팔레트를 닮은 물건은 하나도 없다.** 랙도, 상자도, 지게차도 없다.
크기·맥락·3D 구조가 전부 다르고 **반복 슬랫 텍스처만 같다.**

## 왜 이게 병목 판정에 중요한가

이 감사가 확정한 병목은 저앙각(edge-on) 레짐의 코너 위치추정이다
(`FAILURE_DECOMPOSITION.md`). 위 관찰은 그 원인에 대한 가설을 하나 세운다.

[추정] **R0 는 팔레트를 3D 구조가 아니라 상판의 반복 슬랫 텍스처로 찾는 것으로 보인다.**
그렇다면 앙각이 낮아질수록 상판이 얇은 띠로 압축돼(bbox 단축 중앙값 34.9 px,
`FAILURE_DECOMPOSITION.md` 참조) 그 주 단서가 무너진다.
저앙각 실패와 이 오검출 패턴이 **같은 표현을 두 방향에서 보여주는** 셈이다.

이건 가설이지 확정이 아니다. 확정하려면 슬랫 텍스처를 지우거나 가린 팔레트에서
검출률이 떨어지는지 봐야 하는데, 이번 감사 범위 밖이다.

## 기존 판정은 바뀌지 않는다

`SOURCE_REAL_GAP_AUDIT.md` 의 `NEGATIVE_INTERVENTION_TOUCHES_LOCALISATION = NO` 는
그대로다. 이 79건은 **오검출**이고 ranking·검출 축이다. 코너 위치추정 축이 아니다.
memory `negative-supervision-suppresses-detection-too` 도 유효하다 —
negative 감독은 검출을 같이 죽였다.

즉 **"이 벤치들을 negative 로 더 넣자" 는 결론이 아니다.** 그건 이미 해봤고 해로웠다.
여기서 나온 것은 **어떤 표현을 학습했는가에 대한 단서**다.

## 표본의 한계 (봉합하지 않는다)

- `DEV_NEG2689` 는 편향 표본이다 [확인, memory `yolo-conf-threshold-is-not-the-lever`].
  실제로 79건이 사실상 **6개 장면**에서 나온다 — 벤치 클립, 컵 클립, 실내 가전/키보드,
  공원. 프레임 번호가 연속이다(예: 000046~000060 이 전부 같은 컵).
  따라서 **비율(62%/29%)을 "실제 환경의 오검출 분포" 로 읽으면 안 된다.**
  읽을 수 있는 것은 **종류**뿐이다.
- 이 셋에 없는 실제 산업 환경 물체(팔레트 랙, 적재된 상자, 다른 지게차)에서
  R0 가 어떻게 반응하는지는 **모른다.**
- conf 0.4 미만은 보지 않았다. 배포 임계 기준으로 잘랐다.
