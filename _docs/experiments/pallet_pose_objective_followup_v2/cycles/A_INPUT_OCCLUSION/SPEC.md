# A — FILTERED_INPUT_TO_OCCLUDED

## 질문 / 근거 / 반론

고정 pseudo 좌표를 원래 기본증강으로 변환하고 RGB에만 가림을 더하면, 자연 Moderate+Severe 99장의 학생 T/R이 개선되는가? ICCV2021 easy-hard와 CVPR2023 Cut-Occlude의 입력난도 원리를 참고한다. 원문은 사람 heatmap/서로 다른 학생·teacher 또는 다른 이미지 limb patch를 사용하므로 여기의 fixed-pseudo/noise rectangle은 재현이 아니다. 가장 강한 반론은 filtered pseudo 자체의 오류를 가린 상태에서 더 강요할 수 있고 사각형 overlay가 자연가림과 다르다는 것이다.

새 줄: pooled217+pose/flow-only+lr1e-5의 기존 계약에서 학생 입력만 random occlusion.
재사용 줄: 동일 저장 RAW/REF 타깃·source512·R0·기본기하/색상증강·criterion·D9·last320.
남은 공백: 기존 Clean19 전층/다른교사/다른평가 결과는 이 계약의 대조가 아니다.

## 변경축과 공정 대조

NEW_RAW_OCC / NEW_REF_OCC는 같은 RGB/순서/박스/원래 support와 같은 frozen REF-conditioned mask plan을 사용한다. 좌표 타깃만 raw/corrected로 다르다. 가린 RGB에서 교사를 다시 실행하지 않는다. 원래 support 값을 추가하거나 hidden점을 GT로 채우지 않는다. planning은 center를 세지 않지만 center 손실은 원래대로다.

기본 변환은 원래 그대로 실행한다. 각 원본 label의 복사본에 저장 REF 좌표를 넣어 같은 RNG로 변환한 뒤 입력 RGB/박스 동등성을 assert한다. 원래 RNG 상태를 복원하므로 새 mask의 난수는 기존 학습 경로를 소비하지 않는다. canonical REF는 양 팔의 mask feasibility에만 쓰므로 RAW도 teacher 선정정보를 공유하는 대조다.

old random/structured augmentation의 area fraction {.1,.2,.3}, aspect {.5,1,2}, schedule .5, 8×8 random RGB→bilinear fill을 재사용한다. S2와의 pairing/edge 요구를 제거한 첫 유효 random proposal(최대32)을 사용한다. covered≥1, remaining≥2인 조건 때문에 최소3개 planning support가 필요하다. 기존 구조대조용≥4 gate를 적용하지 않는다. 최종640 input canvas를 쓰며 rect area는 bbox의 약10–30%; 원래 affine 이후이므로 native canvas와 같다고 쓰지 않는다. 무효 occurrence는 입력을 유지하고 사유를 기록한다.

## TRAIN 사전 확인

64 real+64 source에서 128개 RGB/박스 paired exact, 원래 타깃·RNG exact. 64 source bit-exact. 64 real 중14가림, REF 감독22점 가림. 1건은 원래 affine의 좌표별 out-of-frame 처리 때문에 RAW/REF support가 달랐다. 이를 고치거나 posthoc 공통화하지 않는다. 저장 공통 support는 같지만 모든 실제 증강 텐서가 같다는 주장은 하지 않는다. 실제320개 batch에서 RGB/박스/순서/plan과 support 차이 전수기록.

## 평가 / 예산 / 다음 결정

Plastic 먼저 2fit×320updates, seed42, 같은R0, last-only. 학습 전 T/R 기준은 METRIC_AND_SELECTION_LOCK. eval128/66 reference는 학습에 쓰지 않는다. 예측과 D9를 먼저 고정한다. 주99 medianT/R, 전체/난도/recording·P90·coverage·source·보조2D/AUC를 함께 보고한다. 기존 main checkpoint는 역사적 대조이며 numerical prefix 동등성은 이번에 새baseline fit을 하지 않아 완전 증명하지 못한다. 재현 단계에서 같은R0+추가seed의 paired baseline을 확보한다.

3cycle 전체 중 첫 cycle. A만 무차이/악화여도 독립적으로 근거 있는 후보는 계속 검토한다. 유효 signal이면 제2원리와 결합 또는 마지막4fit 예약 재현. lr/epoch/mask강도 sweep과 평가기반 subset선정 금지. NaN/보호텐서변경/pairedRGB불일치는 기술오류로 보존하고 영향구간만 처리한다.
