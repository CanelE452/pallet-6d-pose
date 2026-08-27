# BROAD_FAMILY_V2 — 계획

**RENDER = DO_NOT_RUN. 사용자 승인 대기.**

## 1. 현재 모델이 WEAK_PASS 인 이유

```
CHALLENGE_105  availability 0.629  R 5.63deg  5cm5 0.152
게이트          >=0.75            <=5.0      >=0.20      → 셋 다 미달
OPEN_56 5cm5 0.571 >= 0.45 만 통과
```
실패의 절반 이상이 **pose 이전**에서 난다 — `NO_BOX 37.1%`, `KP_BAD 24.8%`.
generic 셋은 0.624 로 잘 되고 target/night 만 무너진다.

## 2. EXACT EXCLUSION — 0/4 → 2/4

```
scene.usd    exact hash 불일치  회전불변 형상 L1 0.2556  치수비 L1 0.0455
scene_1.usd  exact hash 불일치  회전불변 형상 L1 0.3952  치수비 L1 0.0370
*.glb 2 종   파일 부재로 대조 불가
MESH_EXCLUSION_EXACT = PARTIAL
```
`usd-core` 를 설치해 USD 2 개를 실제 파싱했다. 두 GLB 는 렌더 머신에서 회수해야
닫힌다. 이름이 CC 인터넷 모델처럼 보인다는 것은 근거로 쓰지 않는다.

## 3. GEOMETRY GAP

```
[확인]  unique mesh instance 가 4 개뿐. 40,000 프레임은 그 4 개의 스케일 변형이다.
        effective asset count 3.999, 최대 share 0.2545 (균등하지만 4 를 못 넘는다).
[확인]  평가 대상 mesh 는 그 4 개 중 어느 것과도 다르다 (exact/회전불변 모두 불일치).
[확인]  elev>=8 & bright 에서 NO_BOX 0.00, GOOD 0.70 — 조건이 맞으면 잘 된다.
[추정]  저앙각 NO_BOX(elev<8 & bright 에서 0.46)의 원인이 mesh 다양성 부족이라는 것.
        저앙각은 실루엣이 얇아져 4 종만 본 모델이 일반화하기 어렵다는 해석이나,
        이 데이터로 증명되지 않았다.
[미확인] target 세션과 night 세션이 **같은 물체**라 geometry 와 appearance 의
        인과를 분리할 수 없다. 이 한계는 새 데이터로만 풀린다.
```

## 4. APPEARANCE GAP

```
[확인]  luma 가 낮을수록 단조롭게 나빠진다 (dark<60 GOOD 0.23 / mid100-140 GOOD 0.67).
[확인]  night 세션 luma median 48~52 로 DEV 최저.
[확인]  어둠은 검출보다 **키포인트 정밀도**를 깎는다 —
        elev>=8 & dark 에서 NO_BOX 0.09 인데 KP_BAD 0.73.
[확인]  합성 luma p50 55.8 vs real 123 (2026-08-20 측정).
[추정]  야간 합성을 넣으면 night 셋 5cm5 0.108 이 오른다는 것.
[미확인] geometry 와 분리된 순수 appearance 효과.
```

## 5. 후보

```
GEOMETRY     G_CONSERVATIVE(+8 mesh) / G_BALANCED(+20) / G_BROAD(+40)
APPEARANCE   A_CONSERVATIVE(+3 strata) / A_BALANCED(+6) / A_BROAD(+10)
MIXTURE      MIX_CONSERVATIVE 75/15/10 · MIX_BALANCED 55/25/20 · MIX_AGGRESSIVE 40/35/25
```
frame 수는 **지금 정하지 않는다.** geometry 후보의 축은 "몇 장" 이 아니라
**"서로 다른 mesh 를 몇 개 확보하느냐"** 다.

## 6. 권고 — 하나

```
G_BALANCED (+20 unique mesh) + A_BALANCED (+6 photometric strata)
                 를  MIX_BALANCED (old 55 / geo 25 / appear 20) 로
                 총 40,000 프레임 고정 조립
```

**예상 결과** — NO_BOX 37.1% 가 주 병목이고 그 절반이 저앙각·야간에 몰려 있으므로,
mesh 4→24 와 야간 strata 추가로 availability 0.629 가 먼저 오른다. 5cm5 는 그
뒤를 따른다.

**목적 지지** — target mesh 를 쓰지 않고 family 를 넓히는 것이므로 unseen-object
주장이 유지된다. 총 프레임을 40,000 으로 고정해 "수가 늘어 좋아졌다" 는 반론을
설계 단계에서 차단한다.

**최상위 도달** — STRONG gate 4 개 중 availability 와 5cm5 를 직접 겨냥한다.
R median 5.63→5.0 은 그 둘이 오르면 따라온다 [추정].

**독자의 첫 질문** — "target 을 안 썼다는 걸 어떻게 믿나." 그래서
`TARGET_ASSET_EXCLUSION_AUDIT_V2` 를 mesh hash 수준으로 올렸고, 아직 2/4 라는
것도 숨기지 않았다. **렌더 전에 GLB 2 개를 회수해 4/4 로 닫는 것이 이 계획의
첫 작업이다.**

## 7. 승인 전 선결 (렌더보다 먼저)

```
1. 렌더 머신에서 GLB 2 종 회수 -> MESH_EXCLUSION_EXACT = RESOLVED
2. +20 unique mesh 의 확보 경로 확정 (라이선스 확인된 공개 모델 or procedural)
   ★ 이게 실질 병목이다. 없으면 G_BALANCED 는 실행 불가
3. generic 5cm5 0.624 의 허용 하락폭을 사용자가 freeze
```

## 8. 승인 후 비교 (사전등록)

```
BASE  YOLO26N_PAPER_GENERIC_V1   BROAD40K   60ep seed42
V2    YOLO26N_PAPER_GENERIC_V2   controlled 40K  60ep seed42
차이 0: COCO init / SGD / batch32 / epochs60 / augmentation / padding / PnP

STRONG gate 그대로:
  CHALLENGE availability >= 0.75, R median <= 5.0, 5cm5 >= 0.20
  OPEN 5cm5 >= 0.45
safety: generic 5cm5 가 0.624 대비 승인된 하락폭 안
STRONG_PASS 후에만 seed43/44
```

## 9. FINAL TEST

OPEN56 + CHALLENGE105 는 이 설계에 사용됐으므로 **최종 논문 수치에 쓸 수 없다.**
V2 와 threshold 가 freeze 된 뒤 새 session-disjoint REAL_TEST 를 수집한다
(프로토콜: `paper_generic_pipeline/real_test/`).
