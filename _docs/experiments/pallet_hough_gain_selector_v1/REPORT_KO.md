# Frozen Hough gain selector v1 — 최종 보고

## 관찰

판정: `GAIN_SELECTOR_SYNTHETIC_FAIL`. 고정 corrected Hough seed1의 Q와 Point P를 그대로 사용했다. selector seed1/2/3은 각각2000 optimizer updates를 실제 수행했다. upstream updates는0이고 frozen checkpoint/기존 Q와 원본 산출물 11484개의 보존을 확인했다.

P가 고른 공통 whole-object symmetry로 target과 지표를 평가했다. oracle P-or-Q는 primary 2.179% 개선, Q 선택률 41.016%, 전체 평균 gain 0.179800px로 사전 headroom gate를 통과했다. GT-assisted 진단이며 배포 성능이 아니다.

| 방법 | primary mean/diag | median px | P90 px | coverage | good damage % | catastrophic |
|---|---:|---:|---:|---:|---:|---:|
| Point | 0.009474144 | 2.026449 | 7.333869 | 4027 | 0.0000 | 0 |
| Always-Q | 0.009654724 | 2.089689 | 7.987683 | 4027 | 0.5383 | 0 |
| Oracle P-or-Q | 0.009267744 | 1.930641 | 7.034890 | 4027 | 0.0000 | 0 |
| S_gain seed1 | 0.009456190 | 2.026152 | 7.233891 | 4027 | 0.0897 | 0 |
| S_gain seed2 | 0.009474144 | 2.026449 | 7.333869 | 4027 | 0.0000 | 0 |
| S_gain seed3 | 0.009498658 | 2.031241 | 7.480183 | 4027 | 0.2093 | 0 |

Selector parameter count: 52417. 입력: FEATURE_SCHEMA.json. frozen ROI spatial/local evidence + P/Q coordinates, delta, lines, endpoint relations. normalization은 train만 사용했다.

| seed | tau px | Q 선택 % | G_hat MAE px | Spearman | selected-Q mean gain px | false-harm % | 전체 mean gain px |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.0 | 2.148 | 0.540283 | 0.164209 | 0.757778 | 36.363636 | 0.016280 |
| 2 | None | 0.000 | 0.523783 | 0.206586 | N/A | N/A | 0.000000 |
| 3 | 0.25 | 16.016 | 0.557459 | 0.161916 | -0.124010 | 54.878049 | -0.019861 |

tau는 calibration256의 사전 고정 grid와 제약에서만 선택했다. tau가 null이면 가능한 후보가 없어 사전 고정 always-P fallback을 적용한 것이다. 선택되지 않은 frame gain은0이며 selected subset의 conditional gain과 전체 평균 gain을 구별한다.

## 해석

현재 frozen P/Q 후보에는 GT oracle에서 제한적인 선택 여지가 있었지만, 이 단일 gain-regression selector/고정 예산으로 3 seed 모두 안정적인1% 이득을 회수하지 못했다. 현재 논문은 검증된 기존 결과와 failure analysis를 바탕으로 마무리한다. 새 loss/architecture/threshold/seed 탐색을 시작하지 않는다.

## 미확정

train1792는 upstream Hough의 학습 frame이기도 하므로 selector train target의 낙관 편향 가능성이 있다. frame/이미지/hash는 train/cal/test 사이에 분리됐지만 G38/P0/TEX session/source 범주가 겹친다. session-held-out 일반화, 예측선과 selector 각각의 독립 인과 기여도, Hough 일반의 우열/불가능성은 주장할 수 없다.

real DEV 실행: False; FINAL 열람: False. real DEV가 실행되지 않았다면 실사2D/6D 개선은 미평가다.

13개 회귀검사는 REGRESSION_TESTS.json, 공통 assignment target과 exact inference 및 전체 좌표 지표 독립 검사는 FINAL_INDEPENDENT_AUDIT.json, source/session/difficulty별 target 및 상관은 GAIN_TARGET_AUDIT.json에 기록했다. GT line error는 진단에서만 쓰며 selector 입력에는 없다.

기존 v2 평가기는 각 방법의 최적 symmetry를 따로 고르므로 Always-Q primary가 과거 보고값과 조금 다를 수 있다. 본 실험에서는 action/target 일치를 위해 baseline assignment를 공통 고정했다. 기존 결과를 수정하지 않았다.

대형 observation cache, selector checkpoint와 inference tensor는 data/pallet/results/pallet_hough_gain_selector_v1에 보관하고 기존 Git ignore 정책을 따른다. 코드와 작은 결과/감사 파일만 main에 commit/push한다.
