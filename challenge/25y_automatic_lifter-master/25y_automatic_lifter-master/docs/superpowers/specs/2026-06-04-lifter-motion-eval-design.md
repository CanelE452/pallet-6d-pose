# 리프터 모션 캘리브레이션 평가 — 설계 (2026-06-04)

## 목적 / 판단지표

- **목적**: 리프터가 *믿는* 값 vs *실제* 값의 오차를 측정한다 (구동 프리미티브 캘리브레이션 검증).
  - 회전: "IMU가 믿는 90°" vs "AprilTag 실측 각"
  - 거리: "t(d) 명령 N m" vs "자(ruler) 실측 거리"
- **판단지표**:
  - 회전: AprilTag 실측각 − 명령각 (평균±std, RMSE)
  - 거리: 자 실측거리 − 명령거리 (held-out 오차) + refit 품질(R²/RMSE)
- **독립 GT 원칙**: 검증 대상(IMU·t(d))과 *독립적인* GT(AprilTag·자)를 쓴다.

## Scope (중요 — 이 eval로 대체 금지)

- 본 eval은 **프리미티브(회전·거리) 정확도 검증 전용**이다.
- **closed-loop 정렬 실험(직교 시퀀스 전후 d_lateral, 삽입 성공률, 반복 횟수)은 별도**로 해야
  기여 1·2 통합이 완성된다. 이 스크립트는 그것을 대체하지 않는다.
- 본 시스템은 **단일 고정 구동 강도**로 동작하며(`control.py` 하드코딩: 전진 byte67/델타60, 회전 byte118),
  t(d)는 해당 강도에서 적합·검증된다 (정직한 scope 한정).

## 추적으로 확인된 사실 (`[확인]`)

```
[확인] 회전 종료 = RealSense IMU closed-loop:
       main_rec.py RelYawEstimator(gyro.Y 적분) → fsm.step(rel_yaw)
       → align.py: rel_yaw_ref 스냅 → delta=wrap180(rel_yaw-ref) >= 목표각 → STOP
       매 기동마다 ref 새로 latch → 단기 DR이라 drift 짧게 끊김
[확인] 전진 = open-loop 시간모델 (motion_models.time_from_distance_piecewise)
[확인] control.py: heartbeat만 백그라운드. movement는 호출 시 1회 전송(중복억제 없음)
       → 회전/전진 유지하려면 평가 루프가 movement 프레임을 주기(~20ms) 재전송
[확인] control.py 자동화 전진 강도 = AN_FORWARD=67 (=127-JOYSTICK_FORWARD60), 단일 고정
[확인] config FWD_A=0.07156, T1=4.278 (vmax≈0.306) → 2026-05-27 도입,
       fit에 쓴 power·raw 데이터 출처는 repo에 기록/존재하지 않음 (다른 머신/세션)
[확인] lifter t(d) piecewise fit 스크립트 repo에 없음 → 새로 만든다
```

→ fit power를 repo에서 확정 불가 + 옛 캘리브 데이터가 mixed-power(0.27 vs 0.14)였음
   → **refit이 정답**: 현재 자동화 명령(단일 고정 power)으로 새 run 모아 t(d) 재적합 + held-out.
   → refit은 오히려 옛 fit보다 엄밀: 자(독립 측정) + 단일 확정 power → 측정독립성·power일치 동시 확보.

## 아키텍처

```
eval/
  eval_motion.py     # 진입점: CLI(1회) + 대화형 메뉴, 회전/거리(calib·eval) 오케스트레이션,
                     #          estop/타임아웃/시작확인, CSV 로깅
  imu_yaw.py         # RealSense IMU 백그라운드 폴링 스레드 → rel_yaw 실시간 제공
                     #   (main_rec.py RelYawEstimator 로직 재사용, reset()으로 ref 스냅)
  fit_fwd_model.py   # calib raw CSV → piecewise 적합 → FWD_T0/T1/A + R²/RMSE 출력
  results/
    motion_eval.csv  # 실험 요약 1줄/회
    calib_fwd.csv    # (T, d, direction) refit용 raw
    raw/{ts}_rotate.csv  # 회전 시계열 (t, rel_yaw) — stop 후 1~2초 연장 포함 (★보정2)
                         # 거리 raw는 미기록 — 개루프라 실시간 측정값 없음 (★보정3)
```

**모듈 경계:**
- `imu_yaw.py` — "지금 rel_yaw 몇 도?"만 책임. RealSense 폴링/자이로 적분을 백그라운드 스레드로,
  `reader.rel_yaw` 프로퍼티 + `reader.reset()`. RealSense 없으면 import 단계에서 감지 → 거리 전용 모드.
- `eval_motion.py` — CAN(control.py)·메뉴·시퀀스·로깅. IMU 내부는 인터페이스로만 사용.
- 기존 코드는 **import 재사용**(복사 아님): control.py(`can_init/issue_command_*/can_close`),
  motion_models.py(`time_from_distance_piecewise`), main_rec.py(`RelYawEstimator`).

## 워크플로 순서 (★보정1 — 거리는 순서 의존)

```
(a) drive_calib 으로 calib run 수집 (calib_fwd.csv)
(b) fit_fwd_model.py 로 refit → 새 FWD_T0/T1/A + R²/RMSE
(c) config.py / motion_models 파라미터 갱신
(d) drive_eval (held-out) — ★ refit·config 갱신 후에만 실행
```
> ⚠️ refit 전에 eval 을 돌리면 **옛 파라미터**로 검증하는 사고. eval 은 (c) 완료 후에만.
> (회전은 순서 의존 없음 — 언제든 독립 실행.)

## 동작 시퀀스

### 회전 `rotate_eval(target_deg, direction, power=고정)`
```
1. imu.reset()  → ref = 현재 rel_yaw 스냅
2. 루프(~20ms): issue_command_rotate_in_place(dir) 재전송
                delta = wrap180(imu.rel_yaw - ref); 실시간 |delta|/target 표시
                raw/{ts}_rotate.csv 에 (t, rel_yaw) append
                if |delta| >= target_deg: reached=True; imu_stop=delta; break  # IMU closed-loop(C안)
                if t > TIMEOUT: reached=False; break             # 안전
                if estop: break
3. issue_command_stop()
3b. ★보정2: stop 후 1~2초 raw (t, rel_yaw) 계속 기록 → ECU 관성 over-rotate 정착 캡처
            imu_settled = 정착 후 |delta| (관성까지 반영된 gyro 값)
4. 결과: target / imu_stop(@stop명령) / imu_settled(@정착) / 소요 t / reached
5. 수동 입력: "AprilTag 실측각?" → motion_eval.csv append
```
- 고정 power(byte118) = **주 결과(배치 조건)**. `--power` sweep = 선택적 robustness ablation(생략 가능).
- error = AprilTag − target = **자이로 적분오차 + ECU 관성 합산**.
- ★보정2 부분 분해 (3값 비교):
  - `imu_settled − imu_stop` = gyro가 본 **ECU 관성 over-rotate**
  - `AprilTag − imu_settled` = **gyro 스케일/드리프트** 잔차
  - 논문에서 "오차가 어디서 오는가"를 합산이 아닌 분해로 한 단계 더 설명 가능.

### 거리 calib `drive_calib(T_sec, direction)`  — refit 데이터 수집
```
1. 확인 프롬프트 → issue_command_forward()/backward(), T초 재전송 루프(raw 미기록·★보정3) → stop
2. 수동 입력: "자 실측거리 d(m)?" → calib_fwd.csv (T, d, direction) append
★보정1: T를 가속(d≲d_acc≈0.65m)~정속 양쪽에 고루 분포 → a·T1 구속 (옛 fit 약점 회피)
        특히 측면보정 이동(~0.5m)이 가속 구간 → 여기 안 맞으면 직교 정렬 거리 틀어짐
```

### 거리 eval `drive_eval(D_m, direction)`  — held-out 검증
```
1. t_cmd = time_from_distance_piecewise(D)  (refit된 파라미터 사용)
2. 확인 프롬프트("{D}m → {t_cmd:.2f}s") → 전진 t_cmd초 재전송 루프 → stop
3. 수동 입력: "자 실측거리?" → motion_eval.csv (error = measured − D)
★보정2: calib과 disjoint run (fit범위 내 거리 + 선택적 범위 밖 1점=외삽 확인)
```

## fit_fwd_model.py (★보정3·5)

**★보정3 — 논문 식(13–14) 형태 고정** (모델 형태 유지, 파라미터만 현재 장비에 재적합):
```
vmax = a·t1,  d_acc = ½·a·t1²
t(d) = t0 + √(2d/a)              (d ≤ d_acc, 가속)
       t0 + t1 + (d − d_acc)/vmax (d > d_acc, 정속)
```

**★보정5 — 적합 방향: d = f(T)로 적합 후 t(d)로 해석적 역산**
- 이유: T가 정확한 입력(명령), d가 노이즈 측정(자). OLS는 x축 무오차 가정 →
  d를 노이즈 변수로 두고 d=f(T) 적합(잔차를 d에서 최소화)해야 errors-in-variables 편향 회피.
- 적합 형태(식 13–14의 역함수, 파라미터 a/t1/vmax/t0 동일):
```
T0 < T ≤ T0+T1: d = ½·a·(T−T0)²              (가속)
T > T0+T1:      d = d_acc + vmax·(T−T0−T1)    (정속)
```
- scipy `curve_fit`으로 (T → d) 적합, vmax=a·t1 제약. 그 뒤 t(d)로 역산해 config 갱신값 출력.
- 출력: 새 FWD_T0/T1/A + 적합 품질 R²/RMSE.
- 방향별(fwd/back) 따로 적합.

## CSV 스키마 (★보정2: power 컬럼)

`results/motion_eval.csv`:
```
timestamp, kind, target, direction, power, imu_stop_deg, imu_settled_deg, elapsed_s, cmd_time_s, measured, reached, error, note
```
```
kind            : rotate | drive_eval
target          : 명령값 (rotate=90.0°, drive=2.0m)
direction       : cw/ccw | fwd/back
power           : 구동 강도 (전진 델타60 / 회전 byte118) — 분석 필수
imu_stop_deg    : 회전 stop명령 시점 |delta| (≈target, drive면 공란)
imu_settled_deg : 회전 stop 후 정착 |delta| (관성 반영, drive면 공란·★보정2)
elapsed_s   : 실제 소요시간
cmd_time_s  : drive_eval의 계산 명령시간 (rotate면 공란)
measured    : 수동 실측 (AprilTag 각 / 자 거리)
reached     : True/False (타임아웃·estop이면 False)
error       : measured − target (자동 계산)
note        : 자유 메모
```
`results/calib_fwd.csv`: `timestamp, T_sec, d_measured, direction, power`
`results/raw/{ts}_rotate.csv`: 회전 `(t, rel_yaw)` — stop 후 1~2초 연장 포함. 전진 raw는 미기록(★보정3)

## 안전장치 (A안)

- estop [space]: 별도 스레드 키 감시 → 즉시 `issue_command_stop()` + 루프 탈출. heartbeat는 유지(워치독).
- 타임아웃: 회전 = target/예상각속도×2, 거리 = t_cmd×1.5 (config 상수). 초과 시 강제 STOP.
- 시작 확인 프롬프트: 동작 전 목표값 표시 + 엔터 대기.

## 인터페이스 (C안)

**대화형(인자 없음):**
```
[r] 회전  [f] 전진eval  [c] 전진calib  [t] 목표값변경  [d] 방향변경  [space] estop  [q] 종료
```
**CLI(인자 있으면 1회):**
```
python eval/eval_motion.py --rotate 90 --dir ccw
python eval/eval_motion.py --drive 2.0 --dir fwd
python eval/eval_motion.py --calib 5.0 --dir fwd     # 고정시간 calib run
python eval/eval_motion.py --rotate 90 --no-measure  # 실측입력 생략(자동값만)
```
공통: `--channel`, `--timeout`, `--csv`, `--power`(회전 sweep용).

## 실험 프로토콜 (★보정6 — 반복수 명시)

```
거리 calib : 짧은 T(가속) 여러 + 긴 T(정속) 여러, 각 ≥10회, fwd/back 각각
거리 eval  : 각 D ≥10회 (calib과 disjoint), fwd/back
회전       : 90° CW·CCW 각 ≥10–15회 (+가능하면 45°/180°), 고정 power 주 결과
```
통계(평균±std, RMSE, R²)가 의미 있으려면 최소선 준수 (Ren et al. 1000회 대비 최소 기준).

## 논문 서술 메모

- 회전 error = 자이로 적분오차 + ECU 관성 합산 (★보정4 — reviewer 오해 방지 위해 명시)
- AprilTag 측정 = **Olson(2011) 인용** — "확립된 도구로 측정"으로 정당화
- ★보정4: refit한 FWD_T0/T1/A가 논문 t(d)를 **대체**. 옛 값 보고 + 다른 값 검증 = 불일치이므로,
  검증에 쓴 새 파라미터를 보고. calib fit품질(R²/RMSE) + eval held-out 오차 **둘 다** 보고.
- 단일 고정 구동 강도 scope 한정 명시.

## Scope-out 체크 (이 스크립트 밖, 따로 챙김)

- **AprilTag 측정 도구**: measured 수동입력의 실제 GT는 AprilTag yaw 산출 도구.
  overhead 카메라 + 상단 태그(in-plane yaw) + 카메라 캘리브레이션이 제대로 됐는지 별도 확인.
- **closed-loop 정렬 실험**: 위 Scope 참조 — 별도 실험 세트 필요.
