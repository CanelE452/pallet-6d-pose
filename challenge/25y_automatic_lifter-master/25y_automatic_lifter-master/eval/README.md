# 리프터 모션 캘리브레이션 평가 (eval/)

FSM과 분리된 구동 프리미티브 정확도 검증 도구.
설계: `docs/superpowers/specs/2026-06-04-lifter-motion-eval-design.md`

## 워크플로 (거리는 순서 의존 — ★중요)
1. `python eval/eval_motion.py --calib <T> --dir fwd` 여러 T(가속~정속) 반복 → `results/calib_fwd.csv`
2. `python eval/fit_fwd_model.py results/calib_fwd.csv --dir fwd` → 새 FWD_T0/T1/A + R²/RMSE
3. 출력값을 `depth_cam/calib/config.py` 의 FWD_T0/FWD_T1/FWD_A 에 반영
4. `python eval/eval_motion.py --drive <D> --dir fwd` (held-out) — ★ 3 완료 후에만
> refit 전에 --drive 실행 금지 (옛 파라미터로 검증하는 사고).

## 회전 평가
`python eval/eval_motion.py --rotate 90 --dir ccw` (RealSense 필요)
- IMU closed-loop로 멈춤. stop 후 1~2초 raw 연장 기록 → over-rotate 정착 캡처.
- imu@stop / imu@settle / AprilTag 실측 3값 → 오차 부분 분해.

## 실험 프로토콜 (통계 최소선)
- 거리 calib: 짧은 T(가속 d≲0.65m) 여러 + 긴 T(정속) 여러, 각 ≥10회, fwd/back
- 거리 eval: 각 D ≥10회 (calib과 disjoint run), fwd/back
- 회전: 90° CW·CCW 각 ≥10–15회, 고정 power(byte118) 주 결과

## Scope (이 도구로 대체 금지)
- 프리미티브(회전·거리) 정확도 검증 **전용**.
- closed-loop 정렬(직교 시퀀스 전후 d_lateral, 삽입 성공률)은 **별도 실험**.
- 단일 고정 구동 강도(전진 델타60/회전 byte118)에서만 t(d) 적합·검증.

## 논문 서술 메모
- 회전 error = 자이로 적분오차 + ECU 관성 합산. imu@settle로 부분 분해 가능.
- AprilTag 측정은 Olson(2011) 인용 — overhead + 상단 태그(in-plane yaw) + 카메라 캘리브 확인.
- refit한 FWD_T0/T1/A 가 논문 t(d) 를 대체. calib fit품질(R²/RMSE) + eval held-out 오차 둘 다 보고.

## Scope-out (별도 챙김)
- AprilTag yaw 산출 도구(실제 GT)의 캘리브레이션 신뢰성.
