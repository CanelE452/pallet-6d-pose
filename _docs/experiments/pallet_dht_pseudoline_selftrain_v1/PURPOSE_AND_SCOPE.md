# DHT pseudo-line supervision: one gated development experiment

시작 main `6f5db05e2b4de249f21a6e3aad5262c4f094b585`와 origin/main은 일치하며 clean이었다.
이번은 사용자가 명시 승인한 독립 training-time teacher 실험이며 기존V1–V5와
paper stop lock을 갱신하지 않는다. 최종 학생은stock YOLO26n pose 그대로, DHT는teacher만 담당한다.

Point teacher는R0 `970a0913b38ed4c9e3662837abccbf9d91b8b0858deafae854c1055e477644f7`이다.
DHT는corrected v2 Hough seed1, 최소numeric seed로 고정하며 seed를 비교하지 않는다.
기존GT-free calibration256/test512 예측 캐시를source SHA와 연결해 사용한다.
WLS의Q를pseudo-point로 쓰지 않고 raw line과ambiguity만 선 교사로 사용한다.

Stage A 실패면C0/C1/C2의9-fit을 실행하지 않고 실제update0을 명시해 종료한다.
Stage A의output-space gradient 진단은student parameter-space 효과의 증명이 아니다.
통과한 경우에만 기존true-ignore point recipe와900-update budget의 학생 실험으로 진행한다.
실사GT는pseudo-label 생성·학습에서 격리한다. FINAL은 열지 않는다.
