# REAL_TEST — 촬영 프로토콜

현재 `REAL_DEV_OPEN_56` 과 `REAL_CHALLENGE_DEV_105` 는 **둘 다 development** 다.
이미 여러 분석에 썼으므로 최종 수치에 쓸 수 없다. 최종 논문 수치는 여기서 나온다.

## 0. 언제 찍나

모델 · 데이터 · threshold 가 **전부 freeze 된 뒤**. 그 전에 촬영해도 좋지만
**결과를 열지 않는다.**

## 1. 반드시 다른 팔레트

```
학습 asset 과 다른 pallet instance
가능하면 다른 geometry (두께비·종횡비가 다른 것)
```
지금 확인된 coverage 구멍이 **두께비**다 — BROAD 에 target(0.0923)보다 얇은
프레임이 0.29% 뿐이다. TEST 에 두께가 다양한 팔레트를 넣으면 그 축을 실제로 잰다.

## 2. session strata (각각 별도 세션)

```
indoor day          indoor dim/night
outdoor day         outdoor night
partial occlusion   truncation (좌/우 측면 잘림 위주)
far/small           near/large
```
한 세션이 한 stratum 의 절반을 넘지 않게 한다.

## 3. 연속 프레임 금지

같은 세션 안에서 **최소 1초 간격**, 그리고 카메라나 팔레트가 실제로 움직인 뒤에
뽑는다. 정지 상태 연사는 한 장만 채택한다. 안 그러면 n 은 크고 독립 표본은 작다.

## 4. 반드시 기록

```
camera K          캘리브레이션. 모르면 null — 추정치로 채우지 않는다
pallet dimensions 실측 (줄자). 스펙값을 실측이라 적지 않는다
session metadata  date / location / lighting / background / camera / pallet_id
```

## 5. Positive GT

```
보이는 keypoint 를 가능한 만큼 클릭. 최소 4
안 보이는 코너를 추정해서 찍지 않는다
평면이면 IPPE, 아니면 EPnP+RANSAC -> refineLM
8 코너 재투영 overlay 로 사람이 검수 — 클릭하지 않은 코너까지 물체 위에 앉는지
10~20% 는 서로 다른 두 사람이 독립 어노테이션해 GT 노이즈 바닥을 잰다
```

## 6. Negative 도 별도 세션

`REAL_MATCHED_EMPTY` / `REAL_STRUCTURAL_HARD` / `REAL_PALLET_LIKE`.
positive 와 같은 세션 분리 규칙을 적용한다.

## 7. lock 순서

```
촬영 -> session 단위 배정 -> manifest + sha256 lock -> annotation
-> (여기까지 모델 안 돌림) -> 최종 평가 1회
```
**TEST 결과를 보고 model / data / threshold 를 고치면 그 TEST 는 소진된다.**
