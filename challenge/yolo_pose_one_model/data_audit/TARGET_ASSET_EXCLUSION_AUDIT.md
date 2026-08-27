# TARGET ASSET EXCLUSION AUDIT

**PASS**

파일명으로 판단하지 않았다. 라벨이 기록한 asset 식별자 전수와, 프레임마다
기록된 `dimensions_m` 에서 뽑은 회전·평행이동·스케일 불변 기하 signature 로
대조했다.

## 한계 — 먼저 적는다

```
BROAD 가 쓴 asset 4 종 중 파일 단위 mesh hash 대조가 가능한 것: 0 개
  scene.usd / scene_1.usd        pxr 없이 파싱 불가
  *.glb 2 종                      이 머신에 부재 (렌더는 Windows)
대신 라벨의 dimensions_m 기하 signature 로 대조했다 — 실제 렌더에 쓰인
기하이므로 파일명보다 강한 증거이나, mesh hash 는 아니다.
```

## 평가 대상

```
pallet_full.obj  vertices 186,036  faces 180,040
extents [1.3, 1.1, 0.12] m
footprint aspect 1.1818   thickness ratio 0.0923
canonical vertex hash 7cd470d798427395...
```

## BROAD 가 쓴 asset (라벨 전수)

```
eur_pallet_bk_cc0.glb                           10182
woodpallet_block_jtoastie_ccby.glb              10099
scene_1.usd                                     10095
scene.usd                                        9624
```

target(pallet_full / palletobj / scan) 계열 문자열 **0 건**.

## 기하 signature 대조

```
type           n          footprint aspect           thickness ratio
                           min / med / max           min / med / max
Pallet_0    9624       1.0 / 1.198 / 1.609  0.0956 / 0.1252 / 0.1666
Pallet_1   10095       1.0 / 1.202 / 1.604  0.0864 / 0.1144 / 0.1514
Pallet_2   10099       1.0 / 1.175 / 1.564  0.1173 / 0.1562 / 0.2065
Pallet_3   10182     1.125 / 1.496 / 1.982   0.0897 / 0.1206 / 0.159
TARGET                              1.1818                    0.0923
```

## ★ 누수는 없지만 coverage 구멍이 있다

```
target 종횡비 1.1818 — BROAD 는 1.0~1.982 를 덮는다  OK
target 두께비 0.0923 — BROAD 에서 그보다 얇거나 같은 프레임은 0.29% 뿐
```

누수 문제가 아니라 **pallet-family coverage** 문제다. 학습 데이터가
평가 대상보다 두꺼운 팔레트로 거의 채워져 있다.

