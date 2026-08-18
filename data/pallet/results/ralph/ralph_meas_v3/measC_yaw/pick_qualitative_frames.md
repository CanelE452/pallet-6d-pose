# 정성 비교 그림용 프레임 (조건 강제 선정)

조건: R0 det<6(PnP 불가) AND Ours det=8/8 AND Ours ADD <= 도메인 median

```
[Day] outside_ft  n=60  R0미검출=25  Ours ADD median=0.334  후보=3
   capturepallet09_manual_gt 1778653884903302656  ADD=0.115 yaw=5.74 dist=2.81m
                          challenge/data/01_real/manual_gt/capturepallet09_manual_gt/1778653884903302656.png
   capturepallet09_manual_gt 1778653804674198784  ADD=0.227 yaw=5.25 dist=3.66m
                          challenge/data/01_real/manual_gt/capturepallet09_manual_gt/1778653804674198784.png
   capturepallet09_manual_gt 1778653798391620864  ADD=0.236 yaw=7.93 dist=3.50m
                          challenge/data/01_real/manual_gt/capturepallet09_manual_gt/1778653798391620864.png

[Night] night_ft  n=39  R0미검출=15  Ours ADD median=0.297  후보=0

```

---

## 확정 (2026-08-08)

### Day — 그림 생성함
- 프레임 `challenge/data/01_real/manual_gt/capturepallet09_manual_gt/1778653884903302656.png`
  (split=eval, gt_source=manual, GT reproj 0.96px)
- 좌 R0: det **4/8** → PnP 실패 / 우 Ours(h8 loo+flip): det **8/8**, ADD **0.115 m**
  (outside_ft Ours ADD median 0.334 의 1/3)
- 그림 = `_docs/figures/qual_day_R0_vs_ours.png` (2400x1200, 2.00:1), 스크립트 `fig_qual_day.py`
- ★캡션 주의: **20px 초과 코너가 양 패널 모두 2개**다(R0 는 검출 4개 중 2개, Ours 는 8개 중 2개).
  "주황 = 20px 초과 코너" 규칙을 두 패널에 동일 적용했기 때문이며, 캡션에서 개수를 언급할 때
  분모(4 vs 8)를 함께 적지 않으면 오독된다.

### Night — 그림 생성하지 않음 (조건 통과 0개)
정본 final-test night_ft(n08 17 + n09 25 → GT 유효 39) 에서 세 조건을 모두 만족하는 프레임은
**한 장도 없다**. 조건을 완화해 억지로 만들지 않기로 결정(2026-08-08 사용자 (b) 채택).
```
깔때기                  Day(outside_ft)   Night(night_ft)
① R0 det<6                   25               15
② + Ours det==8               7                0   ← 전멸 (Ours 최대 det=6)
③ + ADD<=median               3                0
```
① 통과 15장의 Ours 상태: det=6 이 3장(ADD 0.311 / 0.874 / 1.080, 전부 median 0.297 초과),
det≤5 가 12장(PnP 불가). = **night 에서는 "R0 미검출 → Ours 완전검출" 사례가 존재하지 않는다.**
night_ft 검출률이 R0 24/39 → Ours 26/39 (+2장) 로 미미했던 정량 결과와 정합한다.
포스터/논문 본문에는 이 사실을 서술한다(그림 대신).
