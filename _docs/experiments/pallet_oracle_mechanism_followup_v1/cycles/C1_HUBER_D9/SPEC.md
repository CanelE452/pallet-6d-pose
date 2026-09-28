# C1 — 같은 후보의 강건 재투영 선택 대조

[확인] 현재 Plastic REF의 고정 W/D 후보 AUC여지는0.091078125, Wood REF는0.002366667다. 후보마다 center포함9점 RMSE와 cheirality/invariant/upright/degeneracy penalty를 합치는 기존 D9다. 선택기는 corner0..7으로 풀고 center를 선택 residual에만 사용한다.

[추정·미검증] 일부 큰 재투영 residual이 선택을 지배한다면 Huber가 유용할 수 있다. 동일 후보·pose·9개 residual·기존 penalty를 유지하고 RMSE항만 sqrt(mean(Huber제곱형))으로 바꾼다. Huber는 r≤δ이면r², 그밖은2δr−δ². δ=12px를 고정한다. 이는 저장소의 과거 solver-swap Huber12 표준값을 가져온 설명적 대조이며 현재DEV로 scale을 fitting하지 않는다. EPro-PnP나 MAGSAC의 구현이라고 부르지 않는다.

공식 원리: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
로컬 scale 출처: scripts/paper/pose_metric_closure_v1/evaluate_pose_solver_swap.py 의 Huber12. 과거 학습된 GEO모델과 다른 무학습 대조다.

입력: 기존 frozen R0/RAW/REF/teacher의 RGB예측과 K/등록치수로 만든 동일 후보. GT는 선택에없음. 새로운 후보·좌표·학습·수동정보0. 동점은 후보이름 사전순으로 고정한다. 모든 material/arm을 보고하며 가장좋은 것만 선택하지 않는다.

주 지표: 같은 ADDsym AUC와 fixed-set gap회수율. 보조 R/yaw/t/axis/IoU/coverage 및 난도/recording. native2D는 바뀌지않음. 후보선택을 먼저 저장/hashlock한 뒤 oracle측정원자료로 scoring한다. DEV를 이미 본 사후개발이며 독립확인아님.

반론: W/D가 모두 reprojection을 잘맞추면 강건화로 의미역할을 구별할 정보가 생기지 않는다. 정확한후보의 존재는 그후보를고를cue의존재와다르다. 음의 결과이면 residual강건화 가설의 한 설정만 반박하고 전체selector계열을 기각하지 않는다.

예산: 신규fits0/update0/GPU0, CPU 수분이내. 결과가 나빠도 scale sweep·추가후보생성 없음. BASELINE/RAW/REF와 oracle를모두보존한다.
