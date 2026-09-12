# Frozen P/Q gain selector — one bounded experiment

시작 main: `4a6c0abfd185fef9d06d9db92a22cc0c63249b9f`, origin/main 일치,
clean worktree, corrected WLS 기준 commit ancestor 확인. 새 branch를 만들지 않는다.

## 성능 확인 전 고정한 선택

- Q는 corrected Hough **seed 1**의 final checkpoint. seed 번호 순서로 선택하며,
  이번 실험에서 upstream seed를 비교하거나 바꾸지 않는다. selector seed만 1/2/3.
- P가 선택한 C1/C2 whole-object symmetry를 P/Q 양쪽에 동일 적용한다.
  target과 모든 본 실험 point metric은 이 공통 assignment를 사용한다.
  기존 v2 evaluator의 method별 독립 최적 assignment와 구별한다.
- target은 frame mean gain / raw image diagonal. 수치 조건을 위해 network는
  그 값의 1000배를 출력한다. SmoothL1 beta=1 (normalized gain에서 .001).
- MLP hidden 64–64, SiLU, dropout .1 한 곳. AdamW lr=.001, wd=.0001,
  batch64, seed별2000 updates, grad clip10, final checkpoint만 사용.
- 입력은 P/Q 좌표·delta·line·관계 정보와 frozen 192-channel ROI를 고정 12그룹으로
  압축한 global/2×2 spatial/P와 Q corner 및 P edge midpoint local summaries.
  local feature는 3×3 pooling 후 bilinear sample. 새 backbone은 없다.
  feature normalization은 train mean/std만 사용하며 z는 [-10,10] clip.
- inference는 predicted normalized gain × 해당 frame diagonal > tau_px일 때만
  exact Q, 아니면 exact P. NaN/비유한 예측은 P. blend 금지.
- tau_px=[0,.05,.1,.25,.5,1]만 calibration에서 비교한다. median/P90 nonworse와
  good damage≤.005를 만족하는 후보 중 primary 최소, 1e-12 이내 tie는 높은 tau.
  가능한 후보가 없으면 추가 threshold 없이 always-P fallback (tau는 null로 기록).
- test oracle gate는 primary≥1% 개선 AND median/P90 nonworse. 실패하면 selector
  optimizer update=0으로 중단한다. test GT oracle은 이 사전 stop gate와 진단에만 쓴다.
- 성공은 각 seed에서 요청된 6개 gate와 전체 seed 평균 실제 gain>0,
  각 selected-Q subset 평균 gain>0. 실패 후 추가 학습/튜닝은 없다.

## 데이터·해석 한계

기존 train1792/calibration256/synth_val512를 frame별로 분리하고 ID, 이미지,
observation hash 중복을 검사한다. G38/P0/TEX session/source가 분할 간 겹치므로
session-held-out 일반화 주장은 하지 않는다. Hough가 train1792를 이미 학습하여
selector train target이 낙관적일 수 있다. 기존 export 밖 holdout을 새로 만들거나
새 rendering을 하지 않는다. source/session, GT, symmetry 정보는 selector 입력에서 제외.

원본 v1/v2/corrected-v2/point_line_v4/line_pose_v1 산출물과 checkpoint/GT는
보존 해시로 검사한다. upstream checkpoint를 eval/requires_grad=False로 읽고
torch.inference_mode에서 새 train/cal proposal만 만든다. test는 저장된 Q를 그대로 쓴다.
selector 학습은 detached cache만 읽는다. CUDA는 process-local userspace library로만
실행하고 시스템 변경/재부팅을 하지 않는다. real DEV는 3 seed 통과 시에만,
FINAL은 어떤 경우에도 열지 않는다. optional S_prob는 실행하지 않는다.
