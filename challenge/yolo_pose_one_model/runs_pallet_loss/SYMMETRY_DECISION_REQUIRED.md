# SYMMETRY_DECISION_REQUIRED — SR arm 이 막혔습니다

`role_symmetry_class` 를 4 asset 중 **2개만** 근거로 정할 수 있었습니다.
브리프의 "UNRESOLVED asset 을 임의 분류하지 않는다" 규칙에 따라
**SR arm(A1, A3) 을 BLOCK** 하고, **PC-only A2 는 실행 가능**으로 둡니다.
UNRESOLVED asset 을 train 에서 삭제하지 않았습니다.

## 판정

```
asset       source                                class                근거
─────────────────────────────────────────────────────────────────────────────────
Pallet_0    scene.usd                             SYM180_EQUIVALENT    로컬 mesh 실측
Pallet_1    scene_1.usd                           SYM180_EQUIVALENT    로컬 mesh 실측
Pallet_2    woodpallet_block_jtoastie_ccby.glb    UNRESOLVED           파일 없음
Pallet_3    eur_pallet_bk_cc0.glb                 UNRESOLVED           파일 없음
```

## SYM180 두 개의 근거 (실측)

**기하** — usd-core 로 정점을 파싱해 up 축 둘레 Rz(pi) 를 적용하고,
양방향 최근접 이웃 거리를 2° 회전 대조군과 비교했습니다.

```
                Rz180 p95    2도회전(대조) p95    비율     정점 수
scene.usd        0.002734       0.025796         0.11    413,451
scene_1.usd      0.000861       0.038552         0.02      4,539
```
(정규화 좌표 기준. 비율이 1 에 가까우면 대칭 아님, 0 에 가까우면 자기 일치.)

**외형** — 텍스처가 기하 대칭을 깨는지 확인했습니다.
```
scene.usd    mesh 7 · GeomSubset 0 · UV 보유 mesh 0 · bound material 1종
scene_1.usd  mesh 1 · GeomSubset 0 · bound material 1종 (blinn1)
렌더 재질     flat color 6종 (black / ind_red / graphite / ind_blue / ind_green / mid_gray)
```
면별 재질 분할이 없고 UV 도 없어 **텍스처가 방향성을 만들지 않습니다.**

**task** — `n_openings = 2` (양쪽 면). 180° 회전 시 포크 삽입 기하가 같습니다 `[추정]`.
이건 기하로부터의 추론이지 별도 검증은 아닙니다.

## UNRESOLVED 두 개 — 왜 못 정했나

```
파일 부재      두 GLB 모두 이 머신에 없습니다 (렌더는 Windows).
              기존 감사 TARGET_ASSET_EXCLUSION_AUDIT_V2 도 "4 중 2 만 파일 대조 가능" 으로
              같은 상태를 기록하고 있습니다.
재질 위험      두 GLB 는 **wood 텍스처 11종**을 씁니다
              (pine_warm / dark_knot / weathered_brown / aged_reddish / ...).
              목재 결·옹이·각인은 방향성을 만들 수 있고, 기하가 대칭이어도
              외형이 비대칭이면 ASYM180 입니다.
```

라벨에는 `role_symmetry_class` 필드가 **없습니다**. `front_face_axis` / `bottom_open` 은
프레임마다 선택된 앞면의 속성이라(같은 asset 안에서 True/False 가 섞임) 대칭 판정에
쓸 수 없습니다.

## 결정해 주셔야 할 것

```
(가) 두 GLB 를 Windows 렌더 PC 에서 가져온다   -> 같은 기하·재질 검정을 돌려 확정.
                                                 가장 확실하고, 파일만 오면 수 분.
(나) 렌더 이미지로 외형 대칭을 경험적으로 판정   -> 같은 asset 의 yaw 차 ~180도 프레임 쌍을
                                                 대조. lighting/재질 랜덤화가 교락이라
                                                 근거 강도가 약합니다.
(다) SR arm 을 두 SYM180 asset 으로 제한         -> Pallet_0/1 프레임에만 symmetry-aware
                                                 min 을 적용하고 나머지는 identity-only.
                                                 실행 가능하지만 "topology-conditioned"
                                                 주장의 범위가 절반으로 줄어듭니다.
(라) SR 포기, PC 만 진행                        -> A2 만. 지금 바로 가능.
```

권고는 **(가)** 입니다. 파일 두 개만 오면 근거가 확정되고, A1/A3 의 설계가 그대로
살아납니다. 그 사이 **A2(PC-only)는 막히지 않으므로 먼저 돌릴 수 있습니다.**

## 참고 — 이번에 확정된 것 (SR 과 무관하게 유효)

```
P180        (5,4,7,6,1,0,3,2)   800/800 프레임, 4 asset 동일, involution·bijection ✓
            ★ camera-facing perm_v4 를 재사용하지 않고 fixed 3D corner 에 실제
              Rz(pi) 를 적용해 유도했습니다. 수치가 perm_v4 의 한 값과 같은 것은
              그 경우가 실제로 180도 회전이기 때문이며 재사용이 아닙니다.
대각쌍       (0,6) (1,7) (2,4) (3,5)   800/800, 정확히 4쌍, 8코너 1회씩
            ★ 인덱스를 하드코딩하지 않고 Xi+Xj≈2C 로 유도했습니다.
```
