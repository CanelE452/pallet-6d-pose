# 재투영 잔차 방향: 학습 전 TRAIN 표현 감사

안정적인 T·R 공동 개선이 목표다. [직전 부호 감독 실험](../pallet_pose_signed_axes_sign_20261001_v1/REPORT_KO.md)은 실제 네 fit의 수렴을 인증했으나 source43/45로 실패했고, 안전 개선과 한 축 이상 악화된 선택이 함께 늘었다. 같은 손실의 가중치나 seed를 다시 탐색하기 전에 현재 입력에 무엇이 보존되는지 확인한다. 이번 감사의 신규 selector 학습·가중치 시험·이미지 forward·PnP·실사 평가는 모두 예산0이다.

## 코드에서 확인한 정보와 아직 확인하지 않은 것

기존 raw94는 각 점의 `norm(projected−observed)`와 요약, pose·치수·confidence·box 비율 등을 저장한다. 점별 x/y 잔차 방향을 직접 저장하지 않는다. context189와 RBF253은 그 요약의 변환이다. 다만 전체94에는 pose·투영 관련 다른 정보가 있으므로, 이 사실만으로 실제 PnP 결과의 전체 특징이 서로 충돌하거나 물리 T/R 예측이 불가능하다고 결론 내릴 수 없다.

253차분의 앞94에서 frame 공통 항이 상쇄돼도, 그 값은 RBF 거리의 context에 남을 수 있다. 따라서 confidence나 anchor context를 전혀 사용하지 않는다는 주장도 하지 않는다. 기존 RGB MLP, signed utility, structured DHT의 음성 결과를 유지하며, 방향이나 RGB가 모든 문제의 해결책이라는 전제는 두지 않는다.

## 고정한 추가 정보 한 묶음

고정 camera-facing pose의 cuboid8과 중심 P8을 기존 K로 투영한다. 원래 prediction의 관측 q9를 빼고 **기존 bbox 대각선으로 나눈 signed residual 18개**를 점0x,0y,…,8x,8y 순으로 만든다. 18개 벡터의 점별 norm은 기존 residual_bboxnorm9와 같은 크기를 갖도록 정의한다. 카메라 ray로 추가 정규화를 바꾸지 않고, 방향을 드러내는 한 정보 묶음으로 한정한다.

원래 source K와 좌표에는 padding이 적용돼 있다. 추가 padding·물리/C2 축 변환·새 PnP 없이 저장된 `R_cf`, `centroid`, `cf_extents`를 사용한다. 캐시에 projected_keypoints가 직접 있다고 가정하지 않는다. 원래 pose 경로와 raw94 생성 경로는 별도 PnP 결과의 allclose 정합을 사용했으므로 bitexact parity도 가정하지 않는다.

모든 유효 점에서 기존 pixel residual9와 corner8 cache를 `atol=1e−4 px`, `rtol=1e−6`으로 대조한다. bbox 정규화 비교에도 같은 pixel 허용값을 대각선으로 나눈 값을 사용한다. 실패하면 중단하며 사후에 tolerance를 높이거나 후보를 제거하지 않는다. 새18은 float32로 저장하고, R0의 같은 TRAIN 유효 두 후보만으로 FP32 mean/std·floor1e−6을 계산한다. 기존94 정규화와 RBF64는 재계산하거나 바꾸지 않는다.

## 두 단계와 기각 조건

첫 단계는 입력만 읽는 특징 검산이다. 기존 적격 TRAIN2,598행과 원래 valid mask를 유지하고 모든 후보가 실패한1행도 zero placeholder로 남긴다. 역사적 특징·pose·metadata 컨테이너는 전체5,120행을 포함하지만, 투영·예측 파일 접근은 적격 TRAIN행에만 한정한다. VAL 품질·실사·TRAIN 타깃 파일은 읽지 않는다.

특징 검산을 통과하면 별도 봉인한 두 번째 단계에서 기존 TRAIN 타깃만 읽는다. 이전253차분과 signed T/R 타깃의 모델별6개 해시를 재현한 뒤 다음을 확인한다.

- 정확히 같은 입력 그룹에서 타깃 부호가 충돌하는가. 반올림하지 않고 ±0만 같은 수로 처리한다. anchor의 강제0 그룹, nonanchor, 역할 간 충돌을 나눈다. 새18이 기존 충돌을 구분하는지와 여전히 남는 충돌도 보고한다. 충돌을 못 찾는 것은 식별 가능성의 증명이 아니다.
- 새18차분이 기존253 입력 열의 선형 결합으로 표현되는가. TRAIN의 유효 nonanchor만 사용해 입력 행렬의 SVD를 계산한다. rank 허용값은 `max(shape)·float64_eps·최대 singular value`다. 타깃은 이 함수에 전달하지 않는다. 모든 네 pool의 잔차 Frobenius 비율이1e−4 이하면 선형 비중복을 추가 근거로 삼지 않는다. 이 값은 입력 중복 검사의 고정 기준이며 성능 gate나 추론 threshold가 아니다.
- 가상 배열에서 기존253 가중치에 새18의0 가중치를 붙이는 분리 내적이 기존 출력을 보존하는지 확인한다. 실제 checkpoint나 새 선택 규칙을 적용하지 않는다. anchor0·invalid0과 전체 행을 보존한다.

같은 expert의 W/D 후보 차이에서는 관측 q가 상쇄되어 새 정보가 두 투영의 차이로 환원된다. expert 간 비교에는 관측 이동 방향도 남는다. 8코너 잔차는 이미 SQPnP+LM의 제약을 받으며, 새18에도 독립 RGB·가시성·정답 정보는 없다. 따라서 비중복이 관측돼도 안전 선택이나 실사 전이가 보장되지는 않는다.

향후 학습은 별도 프로토콜을 요구한다. 원래 세 seed·공유 R0_ONLY, source45, original/matched 실사5 조건을 유지해야 하며 이번 감사가 학습·실사 진행 승인이나 성능 판정을 대신하지 않는다.

[입력 감사 계약](AUDIT_PROTOCOL.json) · [후속 표현 감사 계약](REPRESENTATION_PROTOCOL.json) · [상세 결과](REPORT_KO.md)
