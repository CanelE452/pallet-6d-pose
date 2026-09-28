# 현재 frozen pool의 좌표·whole-output oracle

[확인] R0 / 같은 Replay9장38점 teacher / 현재320-update RAW / REF만 사용했다. 신규 fit·GPU inference·평가 기반 target 생성은0이다. 아래 모든 oracle는 GT_DEPENDENT / DIAGNOSTIC_ONLY이며 배포 성능이 아니다. raw 선택/오류는 private data namespace에만 저장했다.

## Legacy reference의 strict native identity

같은 corner0..7 ID끼리 비교하고 original selected-detection match gate와 전체 분모를 유지했다. teacher의 좋은 점을 다른 번호로 붙이거나 점별 symmetry를 고르지 않았다. 아래 median/P90은 실패 penalty 포함 전체점이며 기존 main의 matched-only median/P90과 구분한다. 기존 whole-object symmetry 지표는 JSON의 별도 supplement다.

| Material | output / oracle | PCK10 | % | full median px | full P90 px |
| --- | --- | --- | --- | --- | --- |
| PLASTIC | R0 | 484/985 | 49.137 | 10.178 | 70.626 |
| PLASTIC | TEACHER | 562/985 | 57.056 | 8.414 | 71.161 |
| PLASTIC | RAW_LR5 | 468/985 | 47.513 | 10.458 | 70.136 |
| PLASTIC | REF_LR5 | 507/985 | 51.472 | 9.642 | 70.359 |
| PLASTIC | whole-output PCK10 oracle | 589/985 | 59.797 | 8.372 | 70.626 |
| PLASTIC | per-point fixed-ID oracle | 613/985 | 62.234 | 7.035 | 65.957 |
| WOOD | R0 | 167/346 | 48.266 | 10.453 | 66.777 |
| WOOD | TEACHER | 198/346 | 57.225 | 7.888 | 80.506 |
| WOOD | WOOD_RAW_LR5 | 163/346 | 47.110 | 10.639 | 67.839 |
| WOOD | WOOD_REF_LR5 | 165/346 | 47.688 | 10.623 | 68.985 |
| WOOD | whole-output PCK10 oracle | 210/346 | 60.694 | 7.808 | 77.084 |
| WOOD | per-point fixed-ID oracle | 228/346 | 65.896 | 6.186 | 64.535 |

## 검수66점

| output / oracle | PCK10 | median px | P90 px |
| --- | --- | --- | --- |
| R0 | 44/66 | 7.144 | 18.390 |
| TEACHER | 50/66 | 5.489 | 13.759 |
| RAW_LR5 | 43/66 | 7.046 | 19.574 |
| REF_LR5 | 43/66 | 7.097 | 17.343 |
| whole-output PCK10 oracle | 53/66 | 5.740 | 17.685 |
| per-point fixed-ID oracle | 53/66 | 4.746 | 13.116 |

16 reused DEV frames의 직접 검수점이며 reference selection/PnP-assisted first pass 이력이 있다. 학생TRAIN217의 정답으로 부르지 않는다. Wood strict verified는 NA_REFERENCE_NOT_VERIFIED다.

| student/reference output | px | 둘다정확 | 교사만 | 학생만 | 둘다오류 |
| --- | --- | --- | --- | --- | --- |
| R0 | 10 | 41 | 9 | 3 | 13 |
| R0 | 20 | 59 | 4 | 1 | 2 |
| RAW_LR5 | 10 | 40 | 10 | 3 | 13 |
| RAW_LR5 | 20 | 59 | 4 | 1 | 2 |
| REF_LR5 | 10 | 40 | 10 | 3 | 13 |
| REF_LR5 | 20 | 62 | 1 | 1 | 2 |

이 교사만정확한 DEV점들이 학생학습에 들어갔는데도 못배웠다는 의미는 아니다. recording별 같은 교차표와 raw 오류를 함께 보존했다.

## Whole production-pose expert oracle

| material | REF production AUC | whole-pose oracle AUC | gap | coverage |
| --- | --- | --- | --- | --- |
| PLASTIC | 0.35901563 | 0.44182422 | 0.08280859 | 128/128 |
| WOOD | 0.66503333 | 0.71068889 | 0.04565556 | 45/45 |

각 모델이 원래 선택한 D9 pose 하나씩, frame마다 하나만 선택했다. W/D 대안 후보의 union oracle와 다르다. ADD 목적의 GT선택이며 PCK/R/yaw/axis 동시최적이 아니다. 기존 동일 normalized ADDsym 적분(0–0.1,1001 trapezoidal,실패분모포함)을 재사용했다.

## 조건부 검출 후보 선택

| arm | GT objective | 8실패 중 복수후보 | box회수가능 | 전체985점PCK10% | 매칭회수 |
| --- | --- | --- | --- | --- | --- |
| R0 | best_box_IoU | 6 | 3 | 49.340 | 3 |
| R0 | best_PCK10 | 6 | 3 | 49.442 | 2 |
| TEACHER | best_box_IoU | 6 | 3 | 57.259 | 3 |
| TEACHER | best_PCK10 | 6 | 3 | 57.360 | 2 |
| RAW_LR5 | best_box_IoU | 6 | 3 | 47.817 | 3 |
| RAW_LR5 | best_PCK10 | 6 | 3 | 47.919 | 2 |
| REF_LR5 | best_box_IoU | 6 | 3 | 51.675 | 3 |
| REF_LR5 | best_PCK10 | 6 | 3 | 51.777 | 2 |

Plastic 원래120/128 매칭의 실패8장만 저장된 검출후보를 검사했다. best-box IoU와 best-PCK10 선택을 따로 계산했고 PCK에는 원래IoU≥.5 gate·whole-object symmetry·전체분모를 유지했다. 정상매칭120장을 새로선택하지 않았고 정답crop 재추론도 없다. Wood45는 모두매칭되어 이진단NA/불필요다. 추가GT선택정보에 따른 이득을배포성능으로쓰지않는다.

## 반경 내 이상적 이동

| REF material | radius native px | optimistic PCK10 | full P90 px |
| --- | --- | --- | --- |
| PLASTIC | 0 | 507/985 | 70.359 |
| PLASTIC | 2 | 582/985 | 68.359 |
| PLASTIC | 4 | 623/985 | 66.359 |
| PLASTIC | 8 | 700/985 | 62.359 |
| PLASTIC | 12 | 752/985 | 58.359 |
| WOOD | 0 | 165/346 | 68.985 |
| WOOD | 2 | 192/346 | 66.985 |
| WOOD | 4 | 209/346 | 64.985 |
| WOOD | 8 | 252/346 | 60.985 |
| WOOD | 12 | 269/346 | 56.985 |

사전에고정한 descriptive r=0/2/4/8/12px, max(e−r,0)이다. 결측/box mismatch에는 적용하지 않았다. 모든 점을GT방향으로 움직일 수 있다고 가정하므로 edge/color cue로회수가능하거나 rigid pose가된다는 증거가아니다. teacher학생선택·후보선택·국소이동 gap을더하지않는다.

검증: 기존 nativefixed-ID와whole-symmetry 수치재현, verified66 same teacher/input parity, whole-output≤per-point 및 기존arm≤oracle, arm순서/tie 불변, missing분모·radius불변을 검사했다. 자세한 난도/recording/임계5·10·20·손익은 [ORACLE_COORDINATE_RESULTS.json](ORACLE_COORDINATE_RESULTS.json), 입력과 고정 규칙은 [ORACLE_COORDINATE_PROTOCOL.json](ORACLE_COORDINATE_PROTOCOL.json)에 있다.
