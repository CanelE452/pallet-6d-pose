# 후속 실험 단 하나 — 설계만, 실행하지 않음

질문: 같은 real TRAIN/pseudo target을 유지하면서 hard 입력의 노출량과 동반 코너 오류 구조를 바꾸면, 기존 natural TRAIN 적합을 보존하면서 recording-separated DEV로 복구가 전이되는가?

Frozen target coupled-hard-dose FULL125 1개 vs 저장된 FULL125. 동일 PRIOR1/seed1/300step/source order+corruption/real253 order/손실/BN동결/crop1.25 유지. real batch8 중 사전 고정4개 슬롯만 점 입력을 변형하고 나머지4개 natural OCC 유지. 변형4개 중2개는 한 코너,2개는 수직 인접 코너쌍 (0,3),(1,2),(4,7),(5,6)에 같은 방향 이동. 목표 주변 원영상 반경20/40/60px·네 방향을 deterministic cycle로 배정. 입력 validity·bbox·RGB·center8·pseudo target 고정. TRUE GT/eval outcome으로 샘플/반경/쌍 선택 금지.

이 실험은 hard-dose+구조를 묶은 하나의 개입이며 각 효과를 따로 증명하지 못한다. 기존 capacity는 simple perturbation에서 이미 높으므로 같은 단일점 jitter만 추가하는 것보다 미관측 큰 동반오류에 초점을 둔다. 단일 recording memorization/physical pseudo correctness는 여전히 별도 한계다.

last300만 기존 분모와 paired recovery/damage/source/green 안전지표로 비교. 최종 모델 자동 승격·실패 후 재학습·sweep 없음. 별도 실행 승인 필요. 이번 작업에서는 optimizer step0으로 종료.
