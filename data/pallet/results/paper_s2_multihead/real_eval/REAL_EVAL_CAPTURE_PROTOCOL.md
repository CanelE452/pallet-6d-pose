# REAL EVAL CAPTURE PROTOCOL

최종 성능 주장은 여기서 난다. synthetic 은 이제 training diversity 담당이고,
**REAL IN-HOUSE DEV/TEST 가 유일한 평가셋**이다.

## 1. 분할 단위는 프레임이 아니라 세션

```
금지   한 세션에서 찍은 프레임을 섞어서 DEV/TEST 로 random split
이유   같은 조명·같은 배경·같은 팔레트 배치가 양쪽에 들어간다.
       그건 held-out 이 아니라 같은 장면의 다른 프레임이다.
```

```
DEV    session A / B / C
TEST   session D / E / F
```

가능하면 **날짜·장소·조명·배경**까지 갈라라. 최소한 셋 중 둘은 갈라야 한다.
같은 날 같은 창고에서 조명만 바꾼 두 세션을 DEV/TEST 로 나누면 세션 분리의
효과가 거의 없다.

## 2. 표본 목표

```
                권장        하한
REAL_POS_DEV    250~300     200
REAL_POS_TEST   500~700     400
REAL_NEG_DEV    300~400     250
REAL_NEG_TEST   700~1000    600
```

세션 수는 **5~10개 이상**. 한 세션이 한 split 의 절반을 넘지 않게 한다.

## 3. 연속 프레임을 세지 않는다

```
금지   동영상에서 인접 프레임 수백 장을 표본으로 세기
이유   n=500 이라고 적히지만 독립 표본은 20~30개다. CI 가 거짓으로 좁아진다.
```

규칙: **같은 세션 안에서 최소 1초 간격**, 그리고 카메라나 팔레트가 실제로
움직인 뒤에 뽑는다. 정지 상태에서 연사한 프레임은 한 장만 채택한다.

## 4. 커버해야 할 축 (합성에서 이미 아는 실패 지점)

```
elevation      저앙각(<8도)을 반드시 포함 — real 배포의 지배적 구간
distance       근거리(<2m)와 원거리(>4.5m) 양쪽
truncation     화면 가장자리에서 잘린 팔레트. 좌/우 측면 잘림 위주
occlusion      적재물·지게차·사람에 의한 부분 가림
illumination   주간 / 야간 / 역광 / 형광등
background     창고 랙, 주차장, 실외 아스팔트
pallet         가능하면 2종 이상. 목재 색·마모 상태 다양하게
```

★ 합성 대비 **appearance gap 이 이미 측정돼 있다** — synthetic luma p50 ≈ 55.8,
real ≈ 123. real 에는 synthetic 에 없는 1280x720 / 2560x720 / 592x1280 이 있다.
이건 limitation 으로 기록만 하고, **지금 새 합성 데이터를 만들지 않는다.**
REAL_DEV 에서 appearance 별로 실제 성능이 깨질 때만 후속 개입 후보다.

## 5. 두 벌을 따로 만든다

```
MAIN        balanced / challenge split   — 주 벤치마크
SECONDARY   NATURAL_REAL_SEQUENCE        — 실제 운용 흐름 그대로, quota 맞추지 않음
```

`NATURAL_REAL_SEQUENCE` 는 positive/negative 비율을 인위로 맞추지 않고 규칙적
간격으로 뽑는다. 목적은 **실제 prevalence 에서의 Precision / FP 빈도** 확인이다.

**둘을 섞어 하나의 AP 로 만들지 않는다.** 섞으면 어느 쪽 숫자도 해석 불가능해진다.

## 6. 세션 메타 (촬영 시 반드시 기록)

```
session_id        고유. 예: 20260821_warehouseA_day_01
date              YYYY-MM-DD
location          장소 식별자
lighting          day | night | backlit | fluorescent | mixed
background        warehouse_rack | parking | outdoor_asphalt | indoor_plain
camera            기종 + 해상도 + 초점거리(알면)
intrinsics        캘리브레이션 파일 경로 (없으면 null — 추정치 채우지 말 것)
pallet_ids        등장한 팔레트 식별자
notes             자유 기술
```

## 7. lock 순서 (결과를 보기 전에)

```
1  촬영
2  session 단위로 DEV / TEST 배정
3  manifest 작성 + sha256 lock      <- 여기까지 모델 돌리기 전에
4  annotation
5  DEV 로 threshold 결정 / 오류 진단
6  TEST 는 마지막에 한 번만
```

**TEST 결과를 보고 generator·threshold·모델을 바꾸면 그 TEST 는 소진된다.**
바꿔야 한다면 새 TEST 세션이 필요하다.
