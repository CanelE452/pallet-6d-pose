# 목적 트리

## [소비처]
논문 §method / §experiments 의 loss 기여 절, 그리고 "다음 GPU 예산을 어디에
쓸 것인가" 결정. Stage 0 만으로도 후자에는 답이 나온다.

## [문장]
"같은 크기의 2D keypoint 오차라도 PnP depth 에 주는 영향은 시점 기하에 따라
26 배까지 다르고, 현재 학습 목적함수는 그 차이를 전혀 반영하지 않는다."

## 최상위 목표

```
monocular RGB -> YOLO26 pose keypoints -> known dimensions + PnP -> 6D pose
```
에서 실제 translation/depth error 를 줄인다.

## 왜 symmetry 가 주항이 아닌가 [확인]

GT 물리축을 oracle 로 공급해도 translation 7.8969 -> 7.7425 cm (-2.0%),
depth 7.5186 -> 7.2537 cm (-3.5%) 에 그친다
(`POSE_EVALUATION_R0.json`, MAIN 대 ORACLE). 축 ambiguity 를 완벽히 풀어도
7.25 cm 가 남는다. symmetry 는 필수 기반 계약이지 7.9 cm 의 원인이 아니다.

★그리고 Stage 0A 가 더 강한 사실을 냈다 — **PAPER_EVAL 319 에는 정사각 물체가
한 장도 없다.** 두 평가 물체는 1.10x1.30(비 1.182)과 0.80x0.59(비 1.356)이고
둘 다 C2({0,180})다. C4 는 이 모집단에서 정의상 발동하지 않는다.

## 왜 translation-risk 가 필요한가 [확인]

같은 픽셀 크기의 오차가 depth 에 주는 영향은 프레임마다 다르다. 319 프레임에서
translation Jacobian 의 기하 인자만 떼어 보면 p05 대 p95 가 **26.0 배** 벌어진다.
그리고 순수 2D 잔차 크기(RMS)는 실제 depth 오차를 **거의 예측하지 못한다**
(Spearman +0.163, 저오차 구간에서는 -0.160 / -0.285). 반면 기하 인자만으로도
+0.537 이다. 즉 "어느 점이 얼마나 틀렸나"보다 **"그 틀림이 어느 방향인가"** 가
depth 를 정한다.

## 판정 지표

Stage 0: surrogate 와 실제 depth 오차의 Spearman, **그리고 순수 잔차 크기 대조군**.
Stage A: 319 프레임의 translation/depth median. 사전등록 게이트는 METHOD_LOCK_DRAFT.md.
