# 기존 self-training과의 경계

`_docs/paper/final/EXPERIMENT_STOP_LOCK.json`과 V1–V5의 결과·checkpoint·teacher
cache·label·method/exposure lock은 역사적 기록으로 보존한다. 새 명시 승인 실험은
Point pseudo-label selection V6가 아닌 training-only DHT 선 supervision의 가능성 감사다.
실험이 좋아져도 paper claim/table/abstract를 자동 수정하지 않는다.

정본 Point teacher는 기존 R0이며 요구 SHA와 실제 파일이 일치한다. 저장된
synthetic P3/P4 feature와 Point 출력의 EXPORT_PROVENANCE도 같은 R0 SHA를 가리킨다.
학생은 같은 `ultralytics.nn.tasks.PoseModel / Pose26`, 9×3 keypoint schema이고
stock inference graph에 DHT/Hough module이 없음을 실제 forward 계약으로 확인한다.

Point pseudo-label 계약은 V3A_TRUE_IGNORE의 기존 파일273개를 그대로 사용한다.
V3B의 추가 ambiguity 처리는 택하지 않는다. 이는 새 filter/threshold 탐색이 아니라
이미 존재하는 true-ignore source의 재사용이다. full pool1000 중273개이며,
그 차이는 기존box acceptance 계약에서 온다. corner가 전부ignore여도 box frame은 유지된다.
C1/C2는 동일273개 label 파일을 가리킨다. 이 실험에서 label 좌표를 repair하지 않는다.

기존 paper exposure는900 updates이며 V3는 synthetic21600 / real7200 exposures다.
본 계획은 모든 arm에 동일synthetic24/update를 적용하고 C0의real8 slot을
zero-gradient padding으로 둔다. 따라서 기존 R0-CONT처럼 real slot을 합성으로
채우지 않으며, synth 노출량이 동일한 optimizer control이다. 실제 학생 실행 여부는
TRAINING_AUDIT에서 별도로 확인해야 하며 계획을 실행 실적으로 계산하지 않는다.

기존 V3 augmentation은 hsv(.015,.5,.35), translate.1, scale.25, mosaic.15,
close_mosaic3, erasing.4, flip0, mixup/copy-paste0이다. gate 통과 후 학생 구현 시
line annotation도 instance별 exact transform을 함께 받아야 한다. 현재 static
homogeneous transform 검사는 full mosaic trainer가 실행됐다는 증명이 아니다.

POOL∩PAPER_EVAL의 image hash/filename 교집합을 원본RGB로 재확인한다.
실사 annotation은 열지 않는다. membership metadata의GT 경로를 읽더라도 그
경로를dereference하지 않는다. provenance로 never-consulted임을 입증한 target
평가군은 확보하지 못했으므로 증거 등급은 POSTHOC_DEVELOPMENT_ONLY다.
Stage A에서 종료하면 PAPER_EVAL/real DEV도 평가하지 않는다. FINAL은 금지다.
