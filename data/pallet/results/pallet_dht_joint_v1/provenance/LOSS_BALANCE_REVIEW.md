# 점·선 손실 연결 검토

기존 소스와 완료 결과만 읽은 검토다. 새 forward·gradient 측정·학습·평가는 수행하지 않았다.

DHT 피드백은 stock Pose26 head 전에 들어간다. One2one은 이 특징을 detach해서 받으므로, 공유 backbone/Hough로 전파되는 stock 손실은 one2many 경로다. Line loss는 전체 stock E2ELoss 뒤에 한 번 더해진다.

| 실제 학습 epoch | one2many 계수 | one2one 계수 | line 계수 | line/공유 stock 계수비 |
|---|---:|---:|---:|---:|
| 1 | 0.8 | 0.2 | 0.1 | 0.125 |
| 2 | 0.1 | 0.9 | 0.1 | 1.0 |

계수비는 8배 커진다. **이는 실제 gradient 크기가 8배 커졌다는 뜻이 아니다.** 내부 pose/RLE 가중치, 손실 정규화, 미분값 및 optimizer가 실제 기여를 결정한다. 또한 기록된 stock loss_items는 one2one 항목이며, 역전파하는 손실은 두 경로의 가중합이다. 기록된 pose_loss와 line_loss 수치만으로 공유 gradient의 우열을 판단할 수 없다.

근거: [integration.py](/home/minjae/Documents/github/pallet-pose/scripts/research/pallet_dht_joint_v1/integration.py:45)의 피드백, 62–96행의 손실 결합; [head.py](/home/minjae/anaconda3/envs/pallet-yolo26/lib/python3.10/site-packages/ultralytics/nn/modules/head.py:161)의 detach; [loss.py](/home/minjae/anaconda3/envs/pallet-yolo26/lib/python3.10/site-packages/ultralytics/utils/loss.py:1179)의 계수·반환값·decay; [trainer.py](/home/minjae/anaconda3/envs/pallet-yolo26/lib/python3.10/site-packages/ultralytics/engine/trainer.py:526)의 epoch 종료 update. 원문 파일 SHA와 줄 번호는 동명 JSON에 저장했다.

기존 CPU 생성 텐서 검사는 aux=0에서도 출력 residual이 열린 뒤 point-location+RLE gradient가 backbone/Hough reduce/line head에 도달함을 보였다. 이는 연결성 검사다. 학습 완료 모델의 점·선 gradient norm/cosine 검사는 고정 16장 batch와 14 GiB guard를 준비했으나 RAM 12.59 GiB에서 중단됐다. **해당 모델 forward와 gradient 측정은 모두 0회다.** 따라서 gradient 충돌이나 PCGrad 효과는 아직 확인하지 않았다. 준비된 검사도 raw final weights/train BN의 한 고정 batch를 보는 국소 진단이며, EMA 실사 성능 악화의 인과 증명이 될 수 없다.

3seed 평균 최종 합성 point mAP50–95는 point-only 94.698%, feature-only 95.025%, joint 94.554%였다. Joint의 weighted line loss는 epoch 평균 0.03389→0.03024로 줄었다. 보조 목적 개선이 점 성능 개선을 보장하지 않는 관찰이며, 음의 gradient cosine을 입증하지 않는다. 실험은 R0에서 2개 전체 epoch·6998 update를 추가한 조건이고 수렴을 증명한 비교는 아니다.

이전 rho 셀 사이의 zero-backprojection은 본 학습 전에 Kρ 평활화와 물리적 footprint mask로 수정·검사됐다. 손실 균형/gradient 방향 문제와 구별해야 한다. 선 GT는 기존 corner GT에서 계산하므로 독립적인 정답 정보가 늘지는 않지만, 전역 선 특징 집계와 기하 제약은 점 목적과 다른 표현 편향을 제공할 수 있다. 현재 결과로 그 효과의 원인을 확정할 수 없다.

기존 소스·프로토콜·결과·history SHA 불변을 저장 직전 확인했다. 이 문서는 원인 후보에 대한 코드 근거이며 성능 개선 또는 gradient surgery의 필요성을 판정하지 않는다.
