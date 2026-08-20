# REAL NEGATIVE CAPTURE PROTOCOL

실제 팔레트가 **없는** real 프레임. pose annotation 은 필요 없고,
`pallet_present = false` 만 확실히 검수한다.

## 1. 구성 비율 (권장)

```
REAL_MATCHED_EMPTY      25~30%   팔레트만 치운 같은 장소·같은 카메라
REAL_STRUCTURAL_HARD    45~50%   랙 / 평행 레일 / 프레임 / 선반 / 파이프
REAL_PALLET_LIKE        20~25%   나무 상자, 적층 목재, 유사 비율 구조물
```

★ 비중을 STRUCTURAL_HARD 에 크게 준 이유가 있다. **합성 negative 에서 잔여 FP 가
거기 몰렸다** — `N0_MATCHED_EMPTY` 는 FP/img 0.005~0.0075 로 사실상 해결이었고,
남은 건 `N1_STRUCTURAL_HARD` 0.41~0.43 / `N2_PALLET_LIKE_HARD` 0.19~0.20 이었다.
real 에서도 같은 곳이 어려울 것으로 보고 표본을 그쪽에 준다. [추정 — 합성 결과의
외삽이며 real 에서 재확인해야 한다]

## 2. MATCHED_EMPTY 를 제대로 찍는 법

같은 세션, 같은 카메라 위치, 같은 조명에서 **팔레트만 치운** 프레임.
장소를 바꾸면 그건 matched 가 아니라 그냥 다른 장면이다.

positive 프레임과 **쌍**으로 기록해 두면 나중에 "무엇이 반응을 만들었는가" 를
직접 비교할 수 있다. `paired_positive_frame_id` 필드에 남긴다.

## 3. 검수

```
사람이 전수 확인:  프레임 안에 팔레트가 정말 없는가
경계 사례          화면 구석에 팔레트 일부가 걸린 프레임
                   -> negative 아님. positive 로 보내거나 폐기
```

부분적으로 보이는 팔레트를 negative 로 넣으면 모델에게 "이건 팔레트가 아니다"
를 잘못 가르치게 된다. 이건 학습이 아니라 평가용이지만, threshold 를 그 위에서
고르면 같은 왜곡이 threshold 로 들어간다.

## 4. 세션 분리

positive 와 **같은 규칙**으로 세션 단위 DEV/TEST 분리. 한 세션의 negative 가
DEV/TEST 양쪽에 들어가면 안 된다.

## 5. 표본 목표

```
REAL_NEG_DEV    300~400
REAL_NEG_TEST   700~1000
```

## 6. 역할

```
REAL_NEG_DEV    score_4kp threshold 를 여기서 정한다  <- FINAL THRESHOLD SOURCE
REAL_NEG_TEST   마지막에 한 번. 여기서 threshold 를 다시 만지지 않는다
```

합성 negative 에서 얻은 threshold 는 **초기 range/reference** 일 뿐이고
"FINAL threshold" 라고 부르지 않는다.
