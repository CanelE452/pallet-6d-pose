# BROAD_FAMILY_GEOMETRY_V2 — 후보

## 왜 geometry 인가 (근거)

```
unique mesh instance  4 개뿐 (40,000 프레임은 그 4 개의 W/D/H 스케일 변형)
  eur_pallet_bk_cc0.glb              10,182 프레임  25.5%
  woodpallet_block_jtoastie_ccby.glb 10,099        25.2%
  scene_1.usd                        10,095        25.2%
  scene.usd                           9,624        24.1%
effective asset count 3.999 — 균등하지만 **4 개를 넘지 않는다**
```
frame 다양성(스케일 랜덤화)은 mesh 다양성이 아니다. 지금 모델은 **팔레트 4 종**만
본 셈이고, 평가 대상은 그 넷 중 어느 것과도 mesh 가 다르다
(exact hash 불일치, 회전불변 형상 히스토그램 L1 0.256 / 0.395).

## 금지

```
target OBJ / decimation / remesh / material 만 바꾼 복사
target 치수비 0.0923 을 그대로 겨냥한 단일 asset
기존 4 mesh 의 축 스케일만 늘리는 것 (이미 하고 있고, 그게 부족했다)
```

## 축 (target 값을 겨냥하지 않고 **구간**으로)

```
footprint aspect     1.0 ~ 1.6      thin/medium/thick 이 아니라 넓게
thickness / min(W,D) 0.06 ~ 0.20    ← target 0.0923 은 이 구간의 한 점일 뿐
deck board 배치      solid / slatted / perimeter
runner / block       3-runner / 9-block / 4-way / 2-way
fork opening 비율    0.15 ~ 0.45
상/하부 구조         double-face / single-face / reversible
재질 구조            plastic 일체형 / wood 조립형
```
**stratum 은 thin / medium / thick 세 개를 모두 둔다.** target 근처만 채우면
그 물체에만 맞춘 데이터가 된다.

## 후보

수치는 asset inventory 이후 산출한다. 아래 `unique mesh` 는 **확보해야 할 서로
다른 topology instance 수**이고, frame 수보다 이쪽이 우선이다.

```
후보              unique mesh   frame    geometry coverage gain          render cost   위험
G_CONSERVATIVE      +8          [보류]   thin stratum 만 채움            낮음         thin 편중으로 target 맞춤화 우려
G_BALANCED          +20         [보류]   3 stratum 균등 + 구조 다양화    중간         가장 무난
G_BROAD             +40         [보류]   구조 축까지 전면 확대           높음         asset 확보 자체가 병목
```

★ frame 수를 지금 정하지 않는다. **"몇 장 더" 가 아니라 "서로 다른 mesh 를 몇 개"**
가 이 후보들의 축이다. mesh 를 확보한 뒤 stratum 당 frame 을 배분한다.

## asset 확보 경로 (렌더 전에 먼저 풀어야)

현재 4 mesh 중 2 개가 CC 라이선스 인터넷 모델이다. +20 mesh 를 모으려면
라이선스가 확인된 공개 팔레트 모델 소스가 필요하고, **그게 이 계획의 실질 병목**이다.
procedural 생성(deck/runner 파라미터화)이 대안이나 사실성이 떨어질 수 있다.
