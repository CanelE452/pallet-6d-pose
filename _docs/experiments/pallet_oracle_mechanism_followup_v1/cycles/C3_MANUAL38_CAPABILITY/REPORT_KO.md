# C3 기존 9장/38점 직접감독 capability 대조

[확인] 판정은 `PARTIAL_TRAIN_FOLLOWING / NO_MAIN_PROMOTION`이다. RAW9→MANUAL9에서 TRAIN PCK10은 32/38→35/38로 증가했지만 Wood의 큰 오차 3점은 남았다. DEV Plastic은 +5/985, Wood는 −22/346, verified66은 −3/66이었다. 따라서 완전한 TRAIN 적합 후 순수 일반화 실패라고도, pose/flow-only 표현이 불가능하다고도 결론 내리지 않는다. 이 두 fit으로 C3를 닫으며 추가 최적화·재현·모델 승격을 하지 않는다.

[확인] 새로운 수동 라벨 없이 기존 교사 TRAIN 정보를 직접 학생에 사용했다. RAW9/MANUAL9는 같은 9장·38점 support·R0·source512·real512·증강·320update이고 감독 좌표만 다르다. 실사 affine OFF, HSV/합성 증강은 유지했다. MAIN217/361 전체 GT 학습 upper bound나 새 self-training recipe의 단독 효과가 아니다.

## TRAIN 저장-index 적합성

| 모델 | RAW38 평균px | MANUAL38 평균px | MANUAL38 PCK10 |
| --- | --- | --- | --- |
| R0 | 0.0000 | 18.6609 | 84.21% |
| RAW9 | 1.1526 | 18.8796 | 84.21% |
| MANUAL9 | 4.0643 | 15.5550 | 92.11% |


교사에 이미 노출된 TRAIN 재평가다. MANUAL9는 수동 타깃 쪽으로 일부 움직였지만 완전히 적합하지 않았다. 전체 manual residual median은 RAW9 4.974→MANUAL9 2.810px, P90은 14.509→7.572px였다. Plastic15점은 13→15/15 PCK10, mean 6.629→3.654px이고 Wood23점은 19→20/23, mean 26.869→23.317px이다. Wood의 같은 한 TRAIN frame에 남은 3점은 모두 여전히 >20px이다. 이 관측은 물리적 축 정답 또는 독립 일반화를 뜻하지 않으며, 한 LR/320update/고정부 조건의 부분 적합을 표현 불가능의 증명으로 해석하지 않는다.

## 현재 반복 DEV — 전부 보고

| 재료 | 모델 | PCK10 | full-penalty median px | full-penalty P90 px | ADDsym AUC |
| --- | --- | --- | --- | --- | --- |
| PLASTIC | R0 | 49.137% | 10.178 | 70.626 | 0.337965 |
| PLASTIC | OLD_RAW | 47.513% | 10.458 | 70.136 | 0.334723 |
| PLASTIC | OLD_REF | 51.472% | 9.642 | 70.359 | 0.359016 |
| PLASTIC | SYN | 48.629% | 10.299 | 70.287 | 0.338945 |
| PLASTIC | RAW9 | 48.934% | 10.208 | 68.946 | 0.334086 |
| PLASTIC | MANUAL9 | 49.442% | 10.051 | 70.335 | 0.346543 |
| WOOD | R0 | 48.266% | 10.453 | 66.777 | 0.670500 |
| WOOD | OLD_RAW | 47.110% | 10.639 | 67.839 | 0.656433 |
| WOOD | OLD_REF | 47.688% | 10.623 | 68.985 | 0.665033 |
| WOOD | SYN | 47.399% | 10.570 | 68.191 | 0.658122 |
| WOOD | RAW9 | 46.243% | 11.133 | 68.208 | 0.661300 |
| WOOD | MANUAL9 | 39.884% | 11.747 | 68.674 | 0.618033 |


통제된 비교는 MANUAL9−RAW9다. R0/OLD_REF와의 비교는 다른 학습 모집단을 가진 참고선이다. recording별 손익·leave-one-recording-out·verified66는 RESULTS.json에 모두 남겼다. Wood45의 직접 visible 출처 검증은 없어 그 별도 점수는 NA다.

camera_dynamic_0123_v4 / UNCONFIRMED_SIGNED_AXIS / MANUAL_REVIEW_REQUIRED 메타데이터를 그대로 보존했다. 중심·unknown·PnP projected 점은 감독하지 않았다. 기존 모델 자동 교체 없음.

## 통제된 paired 손익과 꼬리

다음은 MANUAL9−RAW9다. Plastic의 작은 PCK10 증가는 큰 오차 회복을 의미하지 않는다. 두 재료 모두 >20→≤10px 회복은 0점이며 Plastic의 >20px 수는 오히려 263→274, Wood는 84→91이었다. Plastic은 OLD_REF보다 PCK10 20점, AUC 0.012473 낮다. Wood는 OLD_REF보다 27점, AUC 0.047000 낮다. 재료 분모는 합치지 않는다.

| 재료 | PCK10 RAW→MANUAL | Δpp | ΔADDsym AUC | ≤10 손실 / 회복 | >20→≤10 | <5→>10 | match 실패 / 회복 | branch 변화 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PLASTIC | 482→487 / 985 | +0.508 | +0.012457 | 25 / 30 | 0 | 0 | 0 / 0 | 0 |
| WOOD | 160→138 / 346 | −6.358 | −0.043267 | 35 / 13 | 0 | 6 | 0 / 0 | 1 |

| 재료 | 모델 | PCK5 / PCK10 / PCK20 정답 수 | >20 / >50 / >100px 수 |
| --- | --- | --- | --- |
| PLASTIC | R0 | 205 / 484 / 716 | 269 / 127 / 81 |
| PLASTIC | OLD_RAW | 200 / 468 / 715 | 270 / 123 / 82 |
| PLASTIC | OLD_REF | 243 / 507 / 728 | 257 / 122 / 82 |
| PLASTIC | SYN | 203 / 479 / 717 | 268 / 126 / 82 |
| PLASTIC | RAW9 | 209 / 482 / 722 | 263 / 126 / 82 |
| PLASTIC | MANUAL9 | 212 / 487 / 711 | 274 / 125 / 81 |
| WOOD | R0 | 60 / 167 / 263 | 83 / 43 / 28 |
| WOOD | OLD_RAW | 57 / 163 / 261 | 85 / 43 / 28 |
| WOOD | OLD_REF | 60 / 165 / 261 | 85 / 43 / 28 |
| WOOD | SYN | 60 / 164 / 259 | 87 / 43 / 28 |
| WOOD | RAW9 | 57 / 160 / 262 | 84 / 43 / 28 |
| WOOD | MANUAL9 | 46 / 138 / 255 | 91 / 43 / 28 |

Plastic은 모든 arm에서 검출128/128, matching120/128(93.75%), observed931/전체985점이다. Wood는 검출·matching45/45이고 전체346점이다. PCK/tail은 전체 분모와 실패 penalty를 포함한다. fixed-ID와 현재 symmetry-aware 집계의 PCK5/10/20은 동일하며, 이를 E_sym/E_fixed 전체의 동일성으로 해석하지 않는다.

## Pose 보조 지표 — 동일한 legacy pose reference

R/yaw/t/IoU는 median / P90이다. IoU P90은 높은 쪽 분위수이지 나쁜 tail을 뜻하지 않는다. 모든 arm pose coverage는 Plastic128/128, Wood45/45다. 이 legacy geometry-derived reference는 독립적인 물리적 6D GT가 아니다.

| 재료 | 모델 | axis 정답/frames | R deg | yaw deg | t cm | IoU3D |
| --- | --- | --- | --- | --- | --- | --- |
| PLASTIC | R0 | 83/128 | 3.530 / 88.733 | 2.756 / 88.533 | 9.353 / 79.122 | 0.5865 / 0.7998 |
| PLASTIC | OLD_RAW | 81/128 | 3.942 / 89.052 | 2.626 / 88.600 | 9.226 / 77.884 | 0.5808 / 0.7999 |
| PLASTIC | OLD_REF | 82/128 | 3.766 / 88.809 | 2.550 / 88.656 | 9.186 / 78.938 | 0.5928 / 0.8267 |
| PLASTIC | SYN | 83/128 | 3.507 / 88.681 | 2.590 / 88.341 | 9.203 / 78.605 | 0.5948 / 0.7959 |
| PLASTIC | RAW9 | 82/128 | 3.764 / 88.837 | 2.585 / 88.678 | 9.486 / 78.559 | 0.5722 / 0.8047 |
| PLASTIC | MANUAL9 | 85/128 | 4.250 / 88.981 | 2.682 / 88.554 | 9.044 / 78.275 | 0.5784 / 0.8144 |
| WOOD | R0 | 40/45 | 1.545 / 9.190 | 0.471 / 3.583 | 2.095 / 8.011 | 0.7898 / 0.8943 |
| WOOD | OLD_RAW | 40/45 | 1.598 / 9.084 | 0.536 / 3.469 | 2.349 / 8.259 | 0.7704 / 0.8856 |
| WOOD | OLD_REF | 40/45 | 1.605 / 9.337 | 0.499 / 3.588 | 2.071 / 7.985 | 0.7740 / 0.8672 |
| WOOD | SYN | 40/45 | 1.612 / 9.277 | 0.613 / 3.271 | 2.195 / 7.963 | 0.7713 / 0.8960 |
| WOOD | RAW9 | 40/45 | 1.602 / 9.486 | 0.559 / 3.846 | 2.069 / 8.013 | 0.7714 / 0.8960 |
| WOOD | MANUAL9 | 40/45 | 2.328 / 10.041 | 0.614 / 3.350 | 2.273 / 7.930 | 0.7793 / 0.8821 |

## Recording·가림별 통제 손익

모든 행은 MANUAL9−RAW9이며 사후 반복 DEV의 기술 통계다. 가림·recording을 보고 학습/선택을 바꾸지 않았다.

| 재료 | 그룹 | 점 수 | PCK10 Δ정답 | Δpp | ΔADDsym AUC | ≤10 손실 / 회복 |
| --- | --- | --- | --- | --- | --- | --- |
| PLASTIC | REC_007 | 262 | −8 | −3.053 | +0.021985 | 8 / 0 |
| PLASTIC | REC_021 | 136 | +11 | +8.088 | +0.001000 | 6 / 17 |
| PLASTIC | REC_022 | 122 | +1 | +0.820 | −0.002062 | 2 / 3 |
| PLASTIC | REC_025 | 197 | +3 | +1.523 | −0.013667 | 5 / 8 |
| PLASTIC | REC_027 | 92 | −2 | −2.174 | −0.012167 | 2 / 0 |
| PLASTIC | REC_041 | 80 | +1 | +1.250 | +0.053950 | 0 / 1 |
| PLASTIC | REC_044 | 96 | −1 | −1.042 | +0.071625 | 2 / 1 |
| PLASTIC | CLEAN | 229 | +11 | +4.803 | +0.030224 | 6 / 17 |
| PLASTIC | MODERATE_OCCLUSION | 154 | −4 | −2.597 | +0.006738 | 8 / 4 |
| PLASTIC | SEVERE_OCCLUSION | 602 | −2 | −0.332 | +0.007391 | 11 / 9 |
| WOOD | REC_039 | 187 | −17 | −9.091 | −0.067840 | 22 / 5 |
| WOOD | REC_042 | 159 | −5 | −3.145 | −0.012550 | 13 / 8 |
| WOOD | CLEAN | 301 | −16 | −5.316 | −0.030026 | 28 / 12 |
| WOOD | MODERATE_OCCLUSION | 45 | −6 | −13.333 | −0.115143 | 7 / 1 |

Wood에는 SEVERE_OCCLUSION frame이 없어 NA다. Plastic leave-one-recording-out PCK10 변화는 −0.707~+1.798pp로 부호가 바뀌며, REC_021을 제외하면 −6점이다. AUC 변화는 모든 제외 조건에서 +0.006336~+0.019441이다. Wood는 어느 recording을 제외해도 PCK10/AUC가 감소한다. 이것은 새 fit을 한 교차검증이 아니라 같은 frozen 예측의 부분집합 요약이다.

## Verified66 보조 참조

16frame/66점의 fixed identity·direct visible manual 참조이며 TRAIN에 사용하지 않았다. 모든 arm coverage66/66이다. RAW9→MANUAL9 PCK10은 44→41점, PCK20은 60→59점이고 >20px는 6→7점이다. median 7.057→7.918px, P90 18.368→20.230px로 악화했다. 비교 참고선은 R0 44/66, OLD_RAW 43/66, OLD_REF 43/66, SYN 44/66 PCK10이다. Wood 직접 visible 참조 점수는 검증 자료가 없어 NA다.

## Source32 보존과 실행 무결성

기존 g38 synthetic val32에서 실제 framework `results.csv`의 마지막 epoch5 행이다. 32장은 작은 source 보존 진단이지 실사 정확도나 완전한 합성 분포 보존의 증명이 아니다. checkpoint는 마지막320update만 사용했으며 이 CSV로 선택하지 않았다.

| arm | box mAP50 | box mAP50-95 | pose mAP50 | pose mAP50-95 | val pose loss | train pose loss |
| --- | --- | --- | --- | --- | --- | --- |
| RAW9 | 0.99500 | 0.94734 | 0.99500 | 0.98863 | 0.09477 | 0.13177 |
| MANUAL9 | 0.99500 | 0.94734 | 0.99500 | 0.99154 | 0.10839 | 0.76471 |

CSV는 private `data/pallet/results/pallet_oracle_mechanism_followup_v1/cycles/C3_MANUAL38_CAPABILITY/runs/{arm}/results.csv`에 있다. SHA256은 RAW9 `3da098ea9b16c5f9e615251015623c5db97e3cc8da7f8261c300e8add776a7d8`, MANUAL9 `e042c28d5cc4479efa940b3877695853cc5d69ab65e713de9ece9791708e6544`이다.

[TRAINING_PARITY.json](TRAINING_PARITY.json)은 실제320batch의 RGB·bbox·mask·order가 두 arm 사이에 정확히 같고 좌표만 다름을 확인한다. arm당 source/real 각2,560회, real supervised10,880·ignore12,160·invisible0 노출이다. 독립 checkpoint 감사에서 보호해야 할747 state tensor 전부 R0와 exact이며, 바뀐 파라미터는 RAW9 116개/MANUAL9 126개 모두 허용된 pose/flow 부위다. 매 epoch 보호 state747 감사도 통과했다.

## 비용과 기술 이력

총2fits·640optimizer updates. fit GPU 예약시간137.865346초, 추론8.352705초, 합계146.218051초다. CPU preparation3.261573초, preflight17.629672초, scoring2.380962초는 별도다. unit test·수동 분석·독립 재검산의 wall time은 별도 계측하지 않았으므로 위 합계를 전체 연구 wall time으로 부르지 않는다. 부모 누적 장부에 이미 이 항목을 반영하며 중복 합산하지 않는다.

첫 일반 sandbox CUDA 접근은 모델/optimizer 생성 전 실패해0fit·0update·0GPU초였다. 접근 실패 wall time은 계측하지 않았고 [SANDBOX_ACCESS_ATTEMPT.json](SANDBOX_ACCESS_ATTEMPT.json)과 private log를 보존했다. host 권한으로 동일 명세 두 fit만 실행했다. fit 전 공개 실제 좌표 제거·subprocess module 수정·평가 필드 및 binding 보강은 [PREFIT_TECHNICAL_CORRECTION.json](PREFIT_TECHNICAL_CORRECTION.json)에 남겼고 원 [PROTOCOL.json](PROTOCOL.json)을 보존한 [PROTOCOL_V2.json](PROTOCOL_V2.json) overlay를 사용했다. 실행 후 코드/프로토콜/예측/RESULTS를 바꾸지 않았으며 이 최종 설명만 보완했다.
