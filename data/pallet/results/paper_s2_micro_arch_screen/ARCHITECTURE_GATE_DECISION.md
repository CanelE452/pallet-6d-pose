# ARCHITECTURE GATE DECISION — PAPER_S2 micro screen

[관찰]
- ep57: F2 far2D 43.344px / F1 GT-in-frame 검출률 0.219
- M0_B: F2 far2D 43.344px / F1 GT-in-frame 검출률 0.219
- B1: F2 far2D 43.342px / F1 GT-in-frame 검출률 0.219
- M0_A: F2 far2D 44.020px / F1 GT-in-frame 검출률 0.219
- A1: F2 far2D 43.763px / F1 GT-in-frame 검출률 0.219

[M0 fine-tuning effect]
- M0_B: far_2d_median_px 43.344 → 43.344
- M0_A: gt_inframe_detection_rate 0.219 → 0.219

[B1 결과]
- verdict **FAIL**
  - far_2d_median_reduction_pct = 0.005
  - gross_rate_reduction_pp = 0.000
  - yaw_median_reduction_pct = -2.560
  - reproj_median_reduction_pct = 0.307
  - matched_2d_reduction_pct = 0.065

[A1 결과]
- verdict **FAIL**
  - gt_inframe_recovery_pp = 0.000
  - median_detected_corner_gain = 0.000
  - pose_success_gain_pp = 0.000
  - far_detection_gain_pp = 0.000

[B2 결과 또는 미실행 사유]
- **미실행**: B1 이 FAIL 이므로 조건부 규칙에 따라 structural loss 를 시험하지 않았다.

[지지 증거]
- [확인] 모든 비교가 같은 frame 위 paired 이고, control 은 후보와 동일한 manifest / seed / sampler order / epoch 예산을 쓴다.
- [확인] B1 은 zero-init identity(max|H_final-H_base|=0)에서 출발하므로, 측정된 변화는 재파라미터화가 아니라 학습 결과다.

[반증 증거]
- [확인] B1 FAIL — primary 기준 미달.
- [확인] A1 FAIL — primary 기준 미달.

[현재 판정]
- [확인] B1 은 **시험 불가(NOT TESTABLE)** 다.  target failure mode(F2)가 학습 소스에 없다 — pool far error 중앙값 2.12px / 최대 16.02px vs real F2 중앙값 43.3px, >20px 프레임 0개.
- [확인] 학습 도메인(synthetic hard) 자체에서도 far 오차가 거의 안 줄었다 → residual 이 못 고친 게 아니라 **고칠 오차가 없었다**.
- [판정] 'B1 FAIL → final-stage residual 이 너무 늦다 → backbone 으로 이동' 이라는 Phase 16 규칙은 **이 결과에 적용하지 않는다**.  적용하려면 real F2 분포를 담은 학습 소스에서 다시 시험해야 한다.
- [확인] A1 FAIL → target semantics 결함은 존재하고 방향도 일관되지만(F1 frame peak 가 20/24 프레임에서 상승), 크기가 +0.002 수준이라 검출 임계 0.3 을 넘기지 못한다.  현재 F1 의 주된 response failure 를 해결하는 **충분조건이 아니다**.
- [주의] A1 의 primary 지표(GT-in-frame 검출률)는 이 표본에서 비트 단위로 상수였다.  따라서 이 FAIL 은 '효과 0' 의 증거가 아니라 **판별력 부족**의 증거이기도 하다.  코드 결함은 수정하되 architecture contribution 으로 주장하지 않는다.

[승격 후보]
- 없음

[폐기/보류 후보]
- B1: **보류(시험 불가)** — 폐기 아님.  학습 소스에 target failure mode 가 없어 판정 자체가 성립하지 않았다.
- A1: **보류(판별력 부족)** — 방향은 일관되나 크기가 검출 임계에 미달하고, primary 지표가 상수였다.  단 target semantics 결함 자체는 실재하므로 코드 수정은 별도로 진행한다 (architecture contribution 주장 금지).
- B2: 미실행 (B1 gate 미통과, 조건부 규칙 준수)

[다음 admissible experiment]
1. **B 계열 재시험의 전제 확보**: real F2 분포(far error 20~160px)를 담은 학습 소스를 먼저 만든다.  현재 synthetic pool 은 far 중앙값 2.1px 로 F2 를 담고 있지 않다.  이것이 없으면 어떤 final-stage/backbone 후보도 F2 에 대해 판정할 수 없다.
2. 그 전까지 F2 는 **학습 후보 문제가 아니라 데이터/도메인 문제**로 다룬다 (sim2real 전이갭).  기존 STAGE16 결론과 같은 방향이다.
3. A1 재시험 시에는 검출률 대신 **연속 지표(frame median peak, GT 위치에서의 belief 값)**를 primary 로 쓴다.  현재 지표는 F1 peak(0.10)와 임계(0.30) 격차 때문에 상수다.
4. 이번 결과만으로 논문 main claim 을 쓰지 않는다.  승격 후보가 생기면 새 공개 데이터셋에서 clean 3-seed 로 재검증한다.  final-test 는 열지 않는다.
