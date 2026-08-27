# BROAD_FAMILY_V2 — RENDER PLAN

## RENDER = **HARD BLOCK**

렌더를 막는 것은 예산도 시간도 아니라 **mesh 가 없다**는 사실이다.

```
독립 topology (로컬 실측)        6 개
  BROAD 가 쓰는 것               4  (scene, scene_1, GLB x2)
  BROAD 밖 신규 후보             4  (scene_2, scene_3, SM_PaletteA_01, _02)
                                    ※ 위 4 중 2 는 BROAD 미사용 project USD
G_CONSERVATIVE 요구              +8   ← 도달 불가
```

**THIN 층이 로컬에 하나도 없다.** 그리고 평가 대상 cell 이 바로 `MID/THIN` 이다.

```
aspect x thickness 9 cell 중
  BROAD 4 가 덮는 cell     MID/MID                     1 개
  로컬 전체가 덮는 cell     LOW/THICK, MID/MID,
                          MID/THICK, HIGH/THICK        4 개
  평가 대상 cell           MID/THIN                    ← 아무도 안 덮음
```

## 1. EXCLUSION — 2/4 (변화 없음)

```
scene.usd    exact 불일치  회전불변 L1 0.2556  치수비 L1 0.0455
scene_1.usd  exact 불일치  회전불변 L1 0.3952  치수비 L1 0.0370
GLB 2 종     파일 부재 -> UNRESOLVED
MESH_EXCLUSION_EXACT = PARTIAL
```
렌더 머신에서 GLB 2 종 또는 그 서명(vertex/face count, extents, 정렬 치수비,
canonical 정점 서명, 회전·평행이동·균일스케일 불변 hash)을 받아야 닫힌다.

## 2. MESH BANK

```
raw mesh 파일 검사        6
mesh 읽기 성공            6
독립 topology 클러스터    6      (스케일 변형은 같은 클러스터로 묶임)
BROAD 밖 신규 topology    4
license 확인 완료         0      ← Isaac 은 NVIDIA Omniverse EULA,
                                   project USD 는 출처 미상
```
★ **license 확인 0 건**은 그 자체로 논문 배포의 블로커다. 렌더보다 먼저 푼다.

## 3. GEOMETRY COVERAGE

`GEOMETRY_COVERAGE_BEFORE_AFTER.csv` 참조. before 1 cell → after(로컬) 4 cell.
그러나 THIN 세 cell(LOW/THIN, MID/THIN, HIGH/THIN)은 여전히 비어 있다.

★ **target 치수비 0.0923 을 center 로 삼지 않는다.** THIN 을 `0.06~0.10` 구간으로
잡고 그 안에서 여러 mesh 를 확보한다. 한 점을 겨냥하면 그 물체 전용 데이터가 된다.

## 4. APPEARANCE

`APPEARANCE_STRATA.json` — 6 strata (DAY_BRIGHT / INDOOR_BRIGHT / INDOOR_DIM /
NIGHT / MIXED_TEMPERATURE / HIGH_CONTRAST) + 확률적 요인
(exposure, gamma, WB, noise, blur, defocus, local shadow/highlight, low contrast).

real DEV 이미지를 배경·텍스처로 복사하지 않는다. real 통계(p50 ~123, night ~48,
현재 합성 55.8)는 구간 참조로만 쓴다.

## 5. FACTOR DESIGN

```
G0A0  기존 mesh + 기존 appearance   기존 BROAD 재사용 (렌더 불필요)
G1A0  신규 mesh + 기존 appearance   렌더 필요 — geometry effect
G0A1  기존 mesh + 신규 appearance   렌더 필요 — appearance effect
G1A1  신규 mesh + 신규 appearance   렌더 필요 — interaction
```
target object 는 네 cell 어디에도 없다.
viewpoint / distance / screen size / truncation 분포를 네 cell 에서 동일하게 유지한다
— G1 이 더 멀리 생성되거나 A1 이 더 잘리면 그 축을 재게 된다.

## 6. MIXTURE — 추천 M1_BALANCED

```
G0A0 0.50 (20,000, 기존 재사용)
G1A0 0.17 ( 6,800)
G0A1 0.17 ( 6,800)
G1A1 0.16 ( 6,400)
                총 40,000, 신규 렌더 20,000
```
V1 과 총량이 같아 "수가 늘어 좋아졌다" 는 반론을 설계에서 차단한다.

## 7. GENERIC SAFETY — 결과 전 freeze 완료

```
generic 5cm5 >= 0.574          (baseline 0.624, absolute -5pp)
generic R median 악화 <= 10%
초과 시 GENERIC_SAFETY_FAIL
★ generic = 56 프레임. 1~2 프레임이 ~2pp 를 움직인다. point estimate 만으로
  결론 내지 말고 프레임 수 변화도 함께 보고한다.
```
`GENERIC_SAFETY_LOCK.json` 에 기록. V2 결과를 본 뒤 바꾸지 않는다.

## 8. 렌더 전에 풀어야 할 것 — 순서대로

```
1. GLB 2 종 회수 또는 서명 export      -> MESH_EXCLUSION_EXACT = PASS
2. THIN 층 mesh 확보                   -> 현재 0 개. 이게 최대 병목
3. 독립 topology 를 최소 8 개까지       -> 현재 신규 4 개
4. 확보한 mesh 전부의 license 확인      -> 현재 0 건
```

**1~4 가 풀리기 전에는 렌더 명령을 내지 않는다.** 지금 렌더하면 THIN 이 없는
데이터를 20,000 장 더 만드는 것이고, 그건 현재 실패를 재생산한다.

## 9. mesh 확보 경로 (선택지, 사용자 결정 필요)

```
(가) 공개 3D 모델 사이트에서 라이선스 확인된 팔레트 mesh 수집
     장점 사실적  단점 THIN 팔레트가 흔하지 않아 8 개 채우기 어려울 수 있음
(나) procedural 생성 — deck board 수/두께/간격, runner/block, fork opening 을
     파라미터로 두고 mesh 를 합성
     장점 THIN 을 포함해 cell 을 원하는 대로 채움. license 문제 없음
     단점 사실성이 떨어져 sim2real 갭이 커질 수 있음 [추정]
(다) (가)+(나) 혼합 — 사실적 mesh 로 앵커를 잡고 procedural 로 빈 cell 을 채움
```
제 권고는 **(다)** 다. THIN 은 (나)로만 채울 수 있고, 사실성은 (가)가 준다.
다만 이건 렌더 스펙이 아니라 **asset 파이프라인을 새로 만드는 일**이라,
착수 전에 별도 합의가 필요하다.

## 9. `LOW_ANGLE_ROLE_DISAMBIGUATION_COVERAGE` — 추가 요구 (2026-08-21)

출처: `analysis_pre_v2/KEYPOINT_PERMUTATION_AUDIT.json` (D3) — 재계산하지 않았다.

```
gross R>10 프레임                    24
best permutation = identity          11   (45.8%)
best permutation = near_far_swap     11   -> near/far swap 회수 비율 0.4583
best permutation = top_bottom         2
role confusion rate                  0.5417
```

gross rotation 실패의 **절반 이상이 좌표 정밀도가 아니라 역할 혼동**이다.
그 중 near/far swap 하나만으로 **45.8%** 가 회수된다. 저앙각에서 near face 와
far face 가 화면에서 겹쳐 보이는 것이 유력한 기전이다 `[추정]`.

따라서 V2 는 네 cell 모두에서 다음을 만족해야 한다:

```
요구                                    측정 방법
─────────────────────────────────────────────────────────────────
저앙각(<8 deg) 프레임 비율이 real DEV     프레임별 elevation 각을 계산해
분포를 덮을 것                            히스토그램으로 대조
near/far face 를 구분 가능하게 하는       비대칭 구조(블록 위치·엣지 마감·
비대칭 단서가 mesh 에 존재할 것            텍스처 방향)를 mesh bank 에 기록
near face 와 far face 의 화면 겹침        렌더 시 두 face 의 2D bbox IoU 를
정도를 분포로 통제할 것                    기록하고 cell 간 동일하게 유지
```

★ **near/far swap oracle 을 main inference 에 사용하지 않는다.** 이건 진단
결과이지 추론 경로가 아니다. V2 는 모델이 이 구분을 스스로 배우게 만드는 것이
목표다.

★ 이 축은 §5 의 confound control 대상에 포함된다 — G1 cell 만 저앙각이 많으면
geometry effect 를 재는 게 아니라 viewpoint effect 를 재게 된다.
